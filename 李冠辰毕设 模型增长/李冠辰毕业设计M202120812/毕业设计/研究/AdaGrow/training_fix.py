import logging
import os
import argparse
import numpy as np
import utils
import timm
import yaml
from datetime import datetime

import torch
import torch.nn as nn

from optimizer import get_optimizer, cosine_scheduler
from dataset import get_dataset, DATASETS
from train_and_val import train, test
import utils
import models


def get_args_parser():
    parser = argparse.ArgumentParser(description='Fixed Training')
    
    # Optimizer and Learning Rate
    parser.add_argument('--optimizer', '--opt', default='sgd', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--min-lr', default=1e-5, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--weight-decay-end', default=5e-6, type=float)
    parser.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model', default='', type=str)
    parser.add_argument('--init-meta', default=0.2, type=float, help='a meta parameter for initializer')
    
    # Distributed
    parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='how many training processes to use (default: 1)')
    parser.add_argument('--dist_eval', type=bool, default=True, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true', default=True, help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=bool, default=False)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--worker-seeding', type=str, default='all', help='worker seed mode (default: all)')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--checkpoint-dir', type=str, default='')

    # Dataset
    parser.add_argument('--dataset-name', default='CIFAR10', type=str, help='dataset name (CIFAR10, CIFAR100, SVHN, MNIST)')
    parser.add_argument('--num-classes', default=10, type=int, help='dataset classes number (default 10.)')
    parser.add_argument('--image-channels', default=3, type=int)
    parser.add_argument('--input-size', default=(3, 32, 32), type=tuple, help='dataset input image size (default 32.)')
    parser.add_argument('--batch-size', '--bz', default=128, type=int, help='batch size')
    parser.add_argument('--crop-pct', default=None, type=float, metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--aa', type=str, default="rand-m9-mstd0.5-inc1", metavar='NAME', help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    parser.add_argument('--train-interpolation', type=str, default='bicubic', help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    return parser.parse_args()


def get_module(name, args, **kwargs):
    try:
        net = timm.create_model(name, num_classes=DATASETS[args.dataset_name]['num_classes'],
                                in_chans=DATASETS[args.dataset_name]['image_channels'])
        if "resnet" in name:
            net.conv1 = nn.Conv2d(3, 64, (3, 3), 1, padding=(1,1), bias=False)
    except:
        net = getattr(models, name)(num_classes=DATASETS[args.dataset_name]['num_classes'], 
                                    image_channels=DATASETS[args.dataset_name]['image_channels'])
    net = net.to('cuda')
    print(net)
    return net


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    save_path = os.path.join('./results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if utils.is_main_process(): 
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            args_dict = vars(args)
            with open(os.path.join(save_path, "config.ymal"), 'w') as file:
                yaml.dump(args_dict, file)
            logging.basicConfig(filename=os.path.join(save_path, 'log.txt'), level=logging.INFO)
        else:
            raise OSError('Directory {%s} exists. Use a new one.' % save_path)
    logger = logging.getLogger('main')
    logger.addHandler(logging.StreamHandler())
    logger.info("Saving to %s", save_path)

    dataset_train, dataset_val = get_dataset(args)
    logger.info(f"Training datasets length is: {len(dataset_train)}, Validation datasets length is: {len(dataset_val)}")
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    logger.info("Sampler_train = {}".format(str(sampler_train)))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    trainloader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    testloader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    num_training_steps_per_epoch = len(trainloader)
    lr_schedule_values = cosine_scheduler(args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch)
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)

    logger.info('==> Building model..')

    net = get_module(args.model, args=args, num_classes=args.num_classes)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(net, args)

    visualizer = list()
    for epoch in range(args.epochs):
        train_loss, train_acc, lr, wd = train(args, logger, trainloader, device, epoch, net, optimizer, criterion, lr_schedule_values, wd_schedule_values, num_training_steps_per_epoch)
        test_loss, test_acc = test(args, logger, testloader, device, epoch, net, criterion, save_path, save=True)
        visualizer.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc, "lr": lr, "wd": wd})
    
    with open(os.path.join(save_path, "visualizer.txt"), "w") as f:
        for item in visualizer:
            f.write(str(item))
            f.write("\n")


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
