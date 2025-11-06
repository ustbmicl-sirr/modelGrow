import logging
import os
import re
import argparse
import numpy as np
import models
import utils
import yaml
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from optimizer import get_optimizer, cosine_scheduler
from dataset import get_dataset
from train_and_val import train, test
from utils import params_id_to_name, save_all
from models import model_utils


def get_args_parser():
    parser = argparse.ArgumentParser(description='AdaGrowing Training')
    
    # Optimizer and Learning Rate
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--min-lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay rate')
    parser.add_argument('--weight-decay-end', default=5e-6, type=float, help='weight decay end rate')
    parser.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER', help='LR scheduler (default: "step"')
    parser.add_argument('--epochs', default=300, type=int)

    # Growing configurations
    parser.add_argument('--grow-mode', type=str, default="width-width-depth")
    parser.add_argument('--grow-threshold', default=0.001, type=float, help='the accuracy threshold to grow or stop')
    parser.add_argument('--grow-threshold-tolerate', default=3, type=int, help='Tolerate the number of times growth does not reach the threshold')
    parser.add_argument('--grow-interval', default=3, type=int, help='an interval (in epochs) to grow new structures')
    parser.add_argument('--stop-interval', default=100, type=int, help='an interval (in epochs) to grow new structures')
    parser.add_argument('--growing-metric', default='max', type=str, help='the metric for growing (max or avg)')
    parser.add_argument('--net', default='d2h6', type=str, help='starting net')
    parser.add_argument('--max-net', default='d1000h1000', type=str, help='The maximum net')
    parser.add_argument('--max-params', default='70B', type=str, help='The maximum net params')
    parser.add_argument('--model', default='get_ada_growing_vit_patch2_32', type=str)
    parser.add_argument('--initializer', '--init', default='gaussian', type=str)
    parser.add_argument('--opt-initializer', '--opt-init', default='gaussian', type=str, help='valid when doing optimizer re-parameterization')
    parser.add_argument('--init-meta', default=0.2, type=float, help='a meta parameter for initializer')
    parser.add_argument('--optim-reparam', action='store_true', help='do optimizer re-parameterization')
    parser.add_argument('--opt-init-meta', default=0.2, type=float, help='a meta parameter for optimizer')
    
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
    parser.add_argument('--dataset-name', default='CIFAR10', type=str)
    parser.add_argument('--num-classes', default=10, type=int, help='dataset classes number (default 10.)')
    parser.add_argument('--image-channels', default=3, type=int)
    parser.add_argument('--input-size', default=(3, 32, 32), type=tuple, help='dataset input image size (default 32.)')
    parser.add_argument('--batch-size', '--bz', default=128, type=int, help='batch size')
    parser.add_argument('--crop-pct', default=None, type=float, metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--aa', type=str, default="rand-m9-mstd0.5-inc1", metavar='NAME', help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    parser.add_argument('--train-interpolation', type=str, default='bicubic', help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    return parser.parse_args()


def get_module(name, depth, heads, args):
    net = getattr(models, name)(depth=depth, heads=heads, num_classes=args.num_classes)
    net = net.to('cuda')
    return net


def load_all(args, logger, model, optimizer, trainloader, criterion, path, add_layer_idx):
    checkpoint = torch.load(path)
    old_name_id_map = checkpoint['name_id_map']
    new_id_name_map = params_id_to_name(model)

    if "bert" in args.model:
        backbone_name = "module.transformer.layers."
        layer_num_idx = 2
    else:
        backbone_name = "module.layers."
        layer_num_idx = 3

    inserted_state_dict = {}
    inserted_old_name_id_map = {}
    for key in checkpoint['model_state_dict']:
        if key.startswith(backbone_name):
            layer_num = int(key.split('.')[layer_num_idx])
            if layer_num >= add_layer_idx and add_layer_idx != -1:
                new_key = backbone_name + str(layer_num + 1) + '.' + '.'.join(key.split('.')[layer_num_idx+1:])
            else:
                new_key = key
        else:
            new_key = key
        inserted_state_dict[new_key] = checkpoint['model_state_dict'][key]
        inserted_old_name_id_map[new_key] = old_name_id_map[key]

    new_state_dict = model.state_dict()
    for key in inserted_state_dict:
        if key in new_state_dict:
            original_param = inserted_state_dict[key]
            new_param_shape = new_state_dict[key].shape
            if args.initializer == 'zero':
                new_param = torch.zeros(new_param_shape)
            elif args.initializer == 'uniform':
                new_param = torch.rand(new_param_shape) * args.init_meta
            elif args.initializer == 'original':
                new_param = new_state_dict[key]
            elif args.initializer in ('local_fitting_left', 'local_fitting_right', "local_fitting_between", "global_fitting"):
                mean = torch.mean(original_param).item()
                std = torch.std(original_param).item()
                new_param = torch.randn(new_param_shape) * std + mean
            else:  # args.initializer == 'gaussian'
                new_param = torch.randn(new_param_shape) * args.init_meta
            slices = tuple(slice(0, min(original, new)) for original, new in zip(original_param.shape, new_param_shape))
            new_param[slices] = original_param
            new_state_dict[key] = new_param
    model.load_state_dict(new_state_dict, strict=False)
    
    new_params = []
    for n, p in model.named_parameters():
        if n not in inserted_old_name_id_map:
            new_params.append(p)
            if args.initializer == 'zero':
                logger.info('======> Initializing param {} as zeros...'.format(n))
                p.data.zero_()
            if args.initializer == 'original':
                logger.info('======> Initializing param {} as original...'.format(n))
            elif args.initializer == 'uniform':
                logger.info('======> Initializing param {} as uniform(min={}, max={}) ...'.format(n, -args.init_meta, args.init_meta))
                p.data.uniform_(-args.init_meta, to=args.init_meta)
            elif args.initializer == 'gaussian':
                logger.info('======> Initializing param {} as gaussian(mean=0, std={})...'.format(n, args.init_meta))
                p.data.normal_(0.0, std=args.init_meta)
            elif args.initializer == 'adam':
                logger.info('======> Initializing param {} by adam optimizer...'.format(n))
            elif args.initializer == 'global_fitting':
                params = torch.cat([p.flatten() for p in model.parameters() if p.requires_grad])
                mean = torch.mean(params).item()
                std = torch.std(params).item()
                logger.info('======> Initializing param {} by global fitting with gaussian(mean={}, std={})...'.format(n, mean, std))
                p.data.normal_(mean, std=std)
            elif args.initializer == 'local_fitting_left':
                try:
                    local_module_path = f"{backbone_name}{add_layer_idx - 1}"
                    local_module = model_utils.get_module_by_path(model, local_module_path)
                except:
                    local_module_path = f"{backbone_name}{add_layer_idx + 1}"
                    local_module = model_utils.get_module_by_path(model, local_module_path)
                params = torch.cat([p.flatten() for p in local_module.parameters() if p.requires_grad]) 
                mean = torch.mean(params).item()
                std = torch.std(params).item()
                logger.info('======> Initializing param {} by global fitting with gaussian(mean={}, std={})...'.format(n, mean, std))
                p.data.normal_(mean, std=std)
            elif args.initializer == 'local_fitting_right':
                try:
                    local_module_path = f"{backbone_name}{add_layer_idx + 1}"
                    local_module = model_utils.get_module_by_path(model, local_module_path)
                except:
                    local_module_path = f"{backbone_name}{add_layer_idx - 1}"
                    local_module = model_utils.get_module_by_path(model, local_module_path)
                params = torch.cat([p.flatten() for p in local_module.parameters() if p.requires_grad])
                mean = torch.mean(params).item()
                std = torch.std(params).item()
                logger.info('======> Initializing param {} by global fitting with gaussian(mean={}, std={})...'.format(n, mean, std))
                p.data.normal_(mean, std=std)
            elif args.initializer == 'local_fitting_between':
                local_module_path_left = f"{backbone_name}{add_layer_idx - 1}"
                try:
                    local_module_left = model_utils.get_module_by_path(model, local_module_path_left)
                except:
                    local_module_left = None
                local_module_path_right = f"{backbone_name}{add_layer_idx + 1}"
                try:
                    local_module_right = model_utils.get_module_by_path(model, local_module_path_right)
                except:
                    local_module_right = None
                params_list = []
                if local_module_left is not None:
                    params_list.extend([p.flatten() for p in local_module_left.parameters() if p.requires_grad])
                if local_module_right is not None:
                    params_list.extend([p.flatten() for p in local_module_right.parameters() if p.requires_grad])
                params = torch.cat(params_list)
                mean = torch.mean(params).item()
                std = torch.std(params).item()
                logger.info('======> Initializing param {} by global fitting with gaussian(mean={}, std={})...'.format(n, mean, std))
                p.data.normal_(mean, std=std)
            else:
                logger.fatal('Unknown --initializer.')
                exit()

    if len(new_params) and args.initializer == 'adam':
        logger.info('======> Using adam to find a good initialization')
        old_train_accu = checkpoint['train_accu']
        local_optimizer = optim.Adam(new_params, lr=0.001, weight_decay=5e-4)
        max_epoch = 10
        founded = False
        for e in range(max_epoch):
            _, accu = train(args, logger, trainloader, args.device, e, model, local_optimizer, criterion, None, None, None)
            if accu > old_train_accu - 0.5:
                logger.info('======> Found a good initial position with training accuracy %.2f (vs. old %.2f) at epoch %d' % (accu, old_train_accu, e))
                founded = True
                break
        if not founded:
            logger.info('======> failed to find a good initial position in %d epochs. Continue...' % max_epoch)
    
    if args.optim_reparam:
        old_optimizer_state_dict = checkpoint['optimizer_state_dict']
        new_optimizer_state_dict = deepcopy(optimizer.state_dict())
        for new_id, name in new_id_name_map.items():
            if name in inserted_old_name_id_map:
                old_id = inserted_old_name_id_map[name]
                if old_id in old_optimizer_state_dict['state']:
                    for state_key, old_state_value in old_optimizer_state_dict['state'][old_id].items():
                        param_shape = model.state_dict()[name].shape
                        if isinstance(old_state_value, torch.Tensor):
                            if args.opt_initializer == 'zero':
                                logger.info('======> re-parameterize optimizer {} with zeros ... '.format(state_key))
                                new_state_value = torch.zeros(param_shape)
                            elif args.opt_initializer == 'uniform':
                                logger.info('======> re-parameterize optimizer {} with uniform noises ... '.format(state_key))
                                new_state_value = torch.rand(param_shape) * args.opt_init_meta
                            else:
                                logger.info('======> re-parameterize optimizer {} with gaussian noises ... '.format(state_key))
                                new_state_value = torch.randn(param_shape) * args.opt_init_meta
                            slices = tuple(slice(0, min(old, new)) for old, new in zip(old_state_value.shape, param_shape))
                            new_state_value[slices] = old_state_value[slices]
                            if new_id not in new_optimizer_state_dict['state']:
                                new_optimizer_state_dict['state'][new_id] = {}
                            new_optimizer_state_dict['state'][new_id][state_key] = new_state_value
                        else:
                            if new_id not in new_optimizer_state_dict['state']:
                                new_optimizer_state_dict['state'][new_id] = {}
                            new_optimizer_state_dict['state'][new_id][state_key] = old_state_value
                else:
                    new_optimizer_state_dict['state'][new_id] = {}
            else:
                if new_id not in new_optimizer_state_dict['state']:
                    new_optimizer_state_dict['state'][new_id] = {}
        optimizer.load_state_dict(new_optimizer_state_dict)
    
    epoch = checkpoint['epoch']
    return epoch


def can_grow(depth_max, heads_max, depth_curr, heads_curr):
    if depth_curr == depth_max and heads_curr == heads_max:
        return False
    return depth_curr <= depth_max and heads_curr <= heads_max


def can_grow_params(max_params, model, logger):
    def parse_parameter_string(param_str):
        param_str = param_str.upper().replace(" ", "")
        conversion_factors = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
        if param_str[-1] in conversion_factors:
            number = float(param_str[:-1])
            factor = conversion_factors[param_str[-1]]
            return number * factor
        else:
            return float(param_str)
    
    def format_number_with_unit(number):
        thresholds = [(1e12, 'T'), (1e9, 'B'), (1e6, 'M'), (1e3, 'K')]
        for threshold, unit in thresholds:
            if number >= threshold:
                return f"{number / threshold:.2f}{unit}"
        return str(number)
    
    this_params = sum(p.numel() for p in model.parameters())
    logger.info("Current Params: {}".format(format_number_with_unit(this_params)))
    return this_params < parse_parameter_string(max_params)


def next_arch(net, depth_curr, heads_curr, depth_max, heads_max, mode="head"):
    if depth_curr == depth_max:
        return depth_curr, heads_curr + 1, -1
    if heads_curr == heads_max or mode != "head":
        saliency = model_utils.get_saliency(net)
        max_saliency_index = max(range(len(saliency)), key=lambda i: saliency[i]['saliency'])
        return depth_curr + 1, heads_curr, max_saliency_index
    return depth_curr, heads_curr + 1, -1


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
    depth_init, heads_init = [int(match) for match in re.findall(r'\d+', args.net)]
    depth_max,  heads_max  = [int(match) for match in re.findall(r'\d+', args.max_net)]
    depth_curr, heads_curr = depth_init, heads_init

    net = get_module(args.model, depth=depth_init, heads=heads_init, args=args)
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
    intervals = (args.epochs - 1) // args.grow_interval + 1
    curves = np.zeros((args.epochs, 5))  # train_loss, train_accu, test_loss, test_accu, lr, wd, 
    visualizer = list()
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
            break
        if can_grow(depth_max, heads_max, depth_curr, heads_curr) and can_grow_params(args.max_params, net, logger):
            save_ckpt = os.path.join(save_path, 'ckpt.pth')  # save current model
            save_all((interval + 1) * args.grow_interval - 1, curves[(interval + 1) * args.grow_interval - 1, 2], net, optimizer, save_ckpt)
            if args.grow_mode == "width-depth":
                mode = "depth" if (interval+1) % 2 == 0 else "head"
            if args.grow_mode == "width-width-depth":
                mode = "depth" if (interval+1) % 3 == 0 else "head"
            if args.grow_mode == "width-width-width-depth":
                mode = "depth" if (interval+1) % 4 == 0 else "head"
            if args.grow_mode == "width-depth-depth":
                mode = "head" if (interval+1) % 3 == 0 else "depth"
            if args.grow_mode == "width-depth-depth-depth":
                mode = "head" if (interval+1) % 4 == 0 else "depth"
            depth_curr, heads_curr, add_layer_idx = next_arch(net, depth_curr, heads_curr, depth_max, heads_max, mode=mode)
            logger.info('======> growing to Depth=%d; Heads=%d before epoch %d' % (depth_curr, heads_curr, (interval + 1) * args.grow_interval))
            net = get_module(args.model, depth=depth_curr, heads=heads_curr, args=args)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=True)
            optimizer = get_optimizer(net, args)
            if mode == "depth":
                pass
            loaded_epoch = load_all(args, logger, net, optimizer, trainloader, criterion, save_ckpt, add_layer_idx)
            logger.info('testing new model ...')
            test(args, logger, testloader, device, loaded_epoch, net, criterion, save_path)
        else:
            logger.info('======> reach stop growing limitation. Finished in advance @ epoch %d' % last_epoch)
            last_epoch = (interval + 1) * args.grow_interval - 1
            break
        last_epoch = (interval + 1) * args.grow_interval - 1

    for epoch in range(last_epoch + 1, args.epochs):
        trainloader.sampler.set_epoch(epoch)
        train_loss, train_acc, lr, wd = train(args, logger, trainloader, device, epoch, net, optimizer, criterion, lr_schedule_values, wd_schedule_values, num_training_steps_per_epoch)
        test_loss, test_acc = test(args, logger, testloader, device, epoch, net, criterion, save_path, save=True)
        ema.push(test_acc)
        visualizer.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc, "lr": lr, "wd": wd})

    with open(os.path.join(save_path, "visualizer.txt"), "w") as f:
        for item in visualizer:
            f.write(str(item))
            f.write("\n")



if __name__ == "__main__":
    args = get_args_parser()
    main(args)
