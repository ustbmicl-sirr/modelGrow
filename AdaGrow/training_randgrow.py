import logging
import os
import argparse
import numpy as np
import models
import utils
import yaml
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn

from optimizer import get_optimizer, cosine_scheduler
from dataset import get_dataset
from train_and_val import train, test
import utils
from utils import list_to_str, params_id_to_name, save_all


def get_args_parser():
    parser = argparse.ArgumentParser(description='RandGrowing Training')
    
    # Optimizer and Learning Rate
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer (adam, adamw, sgd, lion, ...)')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--min-lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay rate')
    parser.add_argument('--weight-decay-end', default=5e-6, type=float, help='weight decay end rate')
    parser.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER', help='LR scheduler (default: "step"')
    parser.add_argument('--epochs', default=300, type=int, help='the number of epochs')
    parser.add_argument('--use-checkpoint', default=None, type=str, help='like: /data/liguanchen/rep-grow/results/2024-01-11_13-47-12')

    # Growing configurations
    parser.add_argument('--grow-threshold', default=0.001, type=float, help='the accuracy threshold to grow or stop')
    parser.add_argument('--grow-threshold-tolerate', default=3, type=int, help='Tolerate the number of times growth does not reach the threshold')
    parser.add_argument('--grow-interval', default=3, type=int, help='an interval (in epochs) to grow new structures')
    parser.add_argument('--stop-interval', default=100, type=int, help='an interval (in epochs) to grow new structures')
    parser.add_argument('--growing-metric', default='max', type=str, help='the metric for growing (max or avg)')
    parser.add_argument('--net', default='0-0-0', type=str, help='starting net')
    parser.add_argument('--max-net', default='2-5-1', type=str, help='The maximum net')
    parser.add_argument('--model', default='represnet_bottleneck_tiny', type=str, help='the type of block')
    parser.add_argument('--initializer', '--init', default='gaussian', type=str, help='initializers of new structures (zero, uniform, gaussian)')
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


def get_module(name, arch, args):
    if "vit" in name or "bert" in name:
        net = getattr(models, name)(depth=arch[0], num_classes=args.num_classes)
    else:
        net = getattr(models, name)(depths=arch, num_classes=args.num_classes, image_channels=args.image_channels)
    net = net.to('cuda')
    return net


def load_growed(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch


def load_all(logger, model, optimizer, path):
    checkpoint = torch.load(path)
    old_name_id_map = checkpoint['name_id_map']
    new_id_name_map = params_id_to_name(model)
    # load existing params, and initializing missing ones
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    new_params = []
    for n, p in model.named_parameters():
        if n not in old_name_id_map:
            logger.info('======> Reinitializing param {} '.format(n))
            new_params.append(p)
            if args.initializer == 'zero':
                logger.info(' as zeros...')
                p.data.zero_()
            elif args.initializer == 'uniform':
                logger.info(' by uniform noises...')
                p.data.uniform_(0.0, to=args.init_meta)
            elif args.initializer == 'gaussian':
                logger.info(' by gaussian noises')
                p.data.normal_(0.0, std=args.init_meta)
            else:
                logger.fatal('Unknown --initializer.')
                exit()

    new_checkpoint = deepcopy(optimizer.state_dict())
    old_checkpoint = checkpoint['optimizer_state_dict']
    for new_id in new_checkpoint['param_groups'][0]['params']:
        name = new_id_name_map[new_id]
        if name in old_name_id_map:
            old_id = old_name_id_map[name]
            if old_id in old_checkpoint['state']:
                new_checkpoint['state'][new_id] = old_checkpoint['state'][old_id]
            else:
                new_checkpoint['state'][new_id] = {}
        else:
            if new_id not in new_checkpoint['state']:
                new_checkpoint['state'][new_id] = {}
            else:
                logger.info('skipping param {} state (initial state exists)...'.format(name))
    optimizer.load_state_dict(new_checkpoint)
    epoch = checkpoint['epoch']
    return epoch


def can_grow(maxlim, arch):
    for maxv, a in zip(maxlim, arch):
        if maxv > a:
            return True
    return False


def next_group(g, maxlim, arch, logger):
    if g < 0 or g >= len(maxlim):
        logger.info('group index %d is out of range.' % g)
        return -1
    for i in range(len(maxlim)):
        idx = (g+i+1)%len(maxlim)
        if maxlim[idx] > arch[idx]:
            return idx
    return -1


def next_arch(maxlim, arch, group=0):
    tmp_arch = [v for v in arch]
    if group >= 0 and group < len(arch):
        tmp_arch[group] += 1
    res = []
    for r, m in zip(tmp_arch, maxlim):
        if r > m:
            res.append(m)
        else:
            res.append(r)
    return res


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.use_checkpoint is not None:
        args.net = args.max_net

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
    current_arch = list(map(int, args.net.split('-')))
    max_arch = list(map(int, args.max_net.split('-')))
    if len(current_arch) != len(max_arch):
        logger.fatal('max_arch has different size.')
        exit()
    growing_group = -1
    grown_group = None
    for cnt, v in enumerate(current_arch):
        if v < max_arch[cnt]:
            growing_group = cnt
            break

    net = get_module(args.model, arch=current_arch, args=args)
    print(net)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(net, args)

    if args.growing_metric == 'max':
        ema = utils.MovingMaximum()
    elif args.growing_metric == 'avg':
        ema = utils.ExponentialMovingAverage(decay=0.95)
    else:
        logger.fatal('Unknown --growing-metric')
        exit()

    last_epoch = -1
    growing_epochs = []
    intervals = (args.epochs - 1) // args.grow_interval + 1
    curves = np.zeros((args.epochs, 5))  # train_loss, train_accu, test_loss, test_accu, lr, wd, 
    visualizer = list()
    
    if args.use_checkpoint is not None:
        done_epoch = sum(max_arch) * args.grow_interval
        load_epoch = load_growed(net, optimizer, args.use_checkpoint)
        assert done_epoch > load_epoch, "Only breakpoints after grow are supported to continue training!"
        for epoch in range(load_epoch + 1, args.epochs):
            trainloader.sampler.set_epoch(epoch)
            train_loss, train_acc, lr, wd = train(args, logger, trainloader, device, epoch, net, optimizer, criterion, lr_schedule_values, wd_schedule_values, num_training_steps_per_epoch)
            test_loss, test_acc = test(args, logger, testloader, device, epoch, net, criterion, save_path, save=True)
            ema.push(test_acc)
            visualizer.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc, "lr": lr, "wd": wd})

        with open(os.path.join(save_path, "visualizer.txt"), "w") as f:
            for item in visualizer:
                f.write(str(item))
                f.write("\n")
        exit(0)
    
    for interval in range(0, intervals):
        for epoch in range(interval * args.grow_interval, (interval + 1) * args.grow_interval):
            trainloader.sampler.set_epoch(epoch)
            train_loss, train_acc, lr, wd = train(args, logger, trainloader, device, epoch, net, optimizer, criterion, lr_schedule_values, wd_schedule_values, num_training_steps_per_epoch)
            test_loss, test_acc = test(args, logger, testloader, device, epoch, net, criterion, save_path, save=True)
            ema.push(test_acc)
            visualizer.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc, "lr": lr, "wd": wd})

        # limit max arch
        grow_delta_accu = ema.delta(-1 - args.grow_interval, -1)
        delta_accu_fail_time = 0
        logger.info('======> improved %.5f (ExponentialMovingAverage) in the last %d epochs' % (grow_delta_accu, args.grow_interval))
        if grow_delta_accu < args.grow_threshold and delta_accu_fail_time >= args.grow_threshold_tolerate:
            delta_accu_fail_time += 1
            if grown_group is not None:
                max_arch[grown_group] = current_arch[grown_group]
                logger.info('======> stop growing group %d permanently. Limited as %s .' % (grown_group, list_to_str(max_arch)))
            break

        if can_grow(max_arch, current_arch):
            save_ckpt = os.path.join(save_path, 'current_model.pth')  # save current model
            save_all((interval + 1) * args.grow_interval - 1, curves[(interval + 1) * args.grow_interval - 1, 2], net, optimizer, save_ckpt)
            # create a new net and optimizer
            current_arch = next_arch(max_arch, current_arch, group=growing_group)
            logger.info('======> growing to arch-%s before epoch %d' % (list_to_str(current_arch), (interval + 1) * args.grow_interval))
            net = get_module(args.model, arch=current_arch, args=args)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=True)
            optimizer = get_optimizer(net, args)
            loaded_epoch = load_all(logger, net, optimizer, save_ckpt)
            logger.info('testing new model ...')
            test(args, logger, testloader, device, loaded_epoch, net, criterion, save_path)
            growing_epochs.append((interval + 1) * args.grow_interval)
            grown_group = growing_group
            growing_group = next_group(growing_group, max_arch, current_arch, logger)
        else:
            logger.info('======> reach stop growing limitation. Finished in advance @ epoch %d' % last_epoch)
            break
        last_epoch = (interval + 1) * args.grow_interval - 1

    for epoch in range(last_epoch + 1, args.epochs):
        trainloader.sampler.set_epoch(epoch)
        train_loss, train_acc, lr, wd = train(args, logger, trainloader, device, epoch, net, optimizer, criterion, lr_schedule_values, wd_schedule_values, num_training_steps_per_epoch)
        test_loss, test_acc = test(args, logger, testloader, device, epoch, net, criterion, save_path, save=True)
        ema.push(test_acc)
        visualizer.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc, "lr": lr, "wd": wd})
        save_ckpt = os.path.join(save_path, 'current_model.pth')
        save_all(epoch, curves[epoch, 2], net, optimizer, save_ckpt)
        
    with open(os.path.join(save_path, "visualizer.txt"), "w") as f:
        for item in visualizer:
            f.write(str(item))
            f.write("\n")



if __name__ == "__main__":
    args = get_args_parser()
    main(args)
