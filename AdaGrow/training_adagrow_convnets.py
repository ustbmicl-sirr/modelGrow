import logging
import os
import argparse
import numpy as np
import models
import random
import utils
import re
import yaml
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from optimizer import get_optimizer, cosine_scheduler
from dataset import get_dataset
from train_and_val import train, test
import utils
from utils import params_id_to_name, params_name_to_id
from models import model_utils
from models.adagrow import reparameterizer


seed = 42
random.seed(seed)


def get_args_parser():
    parser = argparse.ArgumentParser(description='RepGrowing Training')
    
    # Optimizer and Learning Rate
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer (adam, adamw, sgd, lion, ...)')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--min-lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay rate')
    parser.add_argument('--weight-decay-end', default=5e-6, type=float, help='weight decay end rate')
    parser.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER', help='LR scheduler (default: "step"')
    parser.add_argument('--epochs', default=300, type=int, help='the number of epochs')

    # Growing configurations
    parser.add_argument('--growing-depth-ratio', type=float, default=0.3)
    parser.add_argument('--grow-mode', type=str, default="width-depth")
    parser.add_argument('--grow-threshold', default=0.001, type=float, help='the accuracy threshold to grow or stop')
    parser.add_argument('--grow-threshold-tolerate', default=3, type=int, help='Tolerate the number of times growth does not reach the threshold')
    parser.add_argument('--grow-interval', default=3, type=int, help='an interval (in epochs) to grow new structures')
    parser.add_argument('--stop-interval', default=100, type=int, help='an interval (in epochs) to grow new structures')
    parser.add_argument('--growing-metric', default='max', type=str, help='the metric for growing (max or avg)')
    parser.add_argument('--net', default='0-0-0', type=str, help='starting net')
    parser.add_argument('--max-net', default='2-5-1', type=str, help='The maximum net')
    parser.add_argument('--max-params', default='70B', type=str, help='The maximum net params')
    parser.add_argument('--model', default='represnet_bottleneck_tiny', type=str, help='the type of residual block (CifarSwitchResNetBasic, CifarPlainNoBNNet, CifarPlainNet, PlainNet, PlainNoBNNet, ResNetBasic or ResNetBottleneck or CifarResNetBasic)')
    parser.add_argument('--initializer', '--init', default='gaussian', type=str, help='initializers of new structures (zero, uniform, gaussian, adam)')
    parser.add_argument('--init-meta', default=0.2, type=float, help='a meta parameter for initializer')
    parser.add_argument('--optim-reparam', action='store_true', help='do optimizer re-parameterization')

    # Distributed
    parser.add_argument('--num-workers', type=int, default=12, metavar='N', help='how many training processes to use (default: 1)')
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
    net = getattr(models, name)(depths=arch, num_classes=args.num_classes, image_channels=args.image_channels)
    net = net.to('cuda')
    return net


def can_grow(maxlim, arch):
    for maxv, a in zip(maxlim, arch):
        if maxv > a:
            return True
    return False


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
    
    tmp_model = deepcopy(model.module)
    tmp_model.eval()
    tmp_model.switch_to_deploy()
    this_params = sum(p.numel() for p in tmp_model.parameters())
    logger.info("Current Params: {}".format(format_number_with_unit(this_params)))
    return this_params < parse_parameter_string(max_params)


def grow(net, optimizer, current_arch, max_arch, mode, last_grown_layer, trainloader, criterion, args, logger):
    print("last_grown_layer : ", last_grown_layer)
    original_net = net.module if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net
    
    if mode == "depth":
        can_grow_index_include = [i for i, (x, y) in enumerate(zip(current_arch, max_arch)) if x < y]
        can_grow_index = [i for i, (x, y) in enumerate(zip(current_arch, max_arch)) if x < y and i != last_grown_layer]
        group_saliency = model_utils.get_saliency(original_net)
        if len(can_grow_index_include) == 1 and len(can_grow_index) == 0:
            max_saliency_index = can_grow_index_include[0]
        else:
            max_saliency_index = max(can_grow_index, key=lambda i: group_saliency[i]['saliency'])
        current_arch[max_saliency_index] += 1
        grown_net = get_module(args.model, current_arch, args)
        logger.info("====> Create new depth grown net done.")
        
        # 把original-net中的较宽的层给新net复制一份 -- 参数移植
        new_params_in_grown_net = {}
        for name, grown_module in grown_net.named_modules():
            original_module = dict(original_net.named_modules()).get(name, None)
            if original_module:
                if isinstance(grown_module, nn.ModuleDict) and isinstance(original_module, nn.ModuleDict):
                    for key, layer in original_module.items():
                        if key not in grown_module:
                            grown_module.add_module(key, layer)
        original_params = {name: param for name, param in original_net.named_parameters()}
        for name, param in grown_net.named_parameters():
            if name in original_params:
                with torch.no_grad():
                    param.data.copy_(original_params[name].data)
            else:
                if name not in new_params_in_grown_net:
                    new_params_in_grown_net[name] = param
        logger.info("====> Modify grown net modules and params done.")
        
        # 初始化新参数
        to_train_params = []
        for n, p in grown_net.named_parameters():
            if n in new_params_in_grown_net:
                to_train_params.append(p)
                if args.initializer == 'zero':
                    logger.info('========> Initializing param {} as zeros...'.format(n))
                    p.data.zero_()
                elif args.initializer == 'original':
                    logger.info('========> Initializing param {} as original...'.format(n))
                elif args.initializer == 'uniform':
                    logger.info('========> Initializing param {} as uniform(min={}, max={}) ...'.format(n, -args.init_meta, args.init_meta))
                    p.data.uniform_(-args.init_meta, to=args.init_meta)
                elif args.initializer == 'gaussian':
                    logger.info('========> Initializing param {} as gaussian(mean=0, std={})...'.format(n, args.init_meta))
                    p.data.normal_(0.0, std=args.init_meta)
                elif args.initializer == 'adam':
                    logger.info('========> Initializing param {} by adam optimizer...'.format(n))
                elif args.initializer == 'global_fitting':
                    params = torch.cat([p.flatten() for p in original_net.parameters() if p.requires_grad])
                    mean = torch.mean(params).item()
                    std = torch.std(params).item()
                    logger.info(f'========> Initializing param {n} by global fitting with gaussian(mean={mean:.5f}, std={std:.5f})...')
                    p.data.normal_(mean, std=std)
                elif args.initializer == 'local_fitting':
                    try:
                        sequential_module_path = f"layer{max_saliency_index+1}"
                        sequential_module = model_utils.get_module_by_path(original_net, sequential_module_path)
                        sequential_module_len = len(sequential_module)
                        local_module_path = f"layer{max_saliency_index+1}.{sequential_module_len - 2}"
                        local_module = model_utils.get_module_by_path(original_net, local_module_path)
                        params = torch.cat([p.flatten() for p in local_module.parameters() if p.requires_grad]) 
                        mean = torch.mean(params).item()
                        std = torch.std(params).item()
                    except:
                        mean, std = 0.0, args.init_meta
                    logger.info(f'========> Initializing param {n} by global fitting with gaussian(mean={mean:.5f}, std={std:.5f})...')
                    p.data.normal_(mean, std=std)
                else:
                    logger.fatal('Unknown --initializer.')
                    exit()

        if args.initializer == 'adam':
            logger.info('======> Using adam to find a good initialization')
            local_optimizer = optim.Adam(to_train_params, lr=0.001, weight_decay=5e-4)
            max_epoch = 2
            for e in range(max_epoch):
                train(args, logger, trainloader, args.device, e, grown_net, local_optimizer, criterion, None, None, None)
        
        # optimizer创建 & optimizer移植
        new_optimizer = get_optimizer(grown_net, args)
        if args.optim_reparam:
            old_name_id_map = params_name_to_id(original_net)
            new_id_name_map = params_id_to_name(grown_net)
            new_checkpoint = deepcopy(new_optimizer.state_dict())
            old_checkpoint = deepcopy(optimizer.state_dict())
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
            new_optimizer.load_state_dict(new_checkpoint)
            logger.info("====> Inherit Optimizer done.")

        return grown_net, new_optimizer, current_arch, max_saliency_index
    else:  # mode == "width"
        def generate_module_pool(repunit):
            len_module = len(repunit.torep_extractor)
            in_dim, out_dim, stride, groups = repunit.in_dim, repunit.out_dim, repunit.stride, repunit.groups
            base_kernel_size = repunit.base_kernel_size
            pool = [[i, j] for i in range(1, base_kernel_size[0] + 1, 2) for j in range(1, base_kernel_size[0] + 1, 2)]
            return pool, len_module, in_dim, out_dim, stride, groups
        
        # 给original-net添加宽度: 哪些 RepUnit 重要; 一定比例的重要 RepUnit 生成结构采样空间
        grown_net = deepcopy(original_net)
        inner_layer_saliency = model_utils.get_inner_layer_saliency(grown_net)
        num_to_save = int(len(inner_layer_saliency) * args.growing_depth_ratio)
        top_saliency = dict(sorted(inner_layer_saliency.items(), key=lambda item: item[1], reverse=True)[:num_to_save])
        new_params_in_grown_net = {}
        for module_name in top_saliency.keys():
            to_width_grow_repunit = model_utils.get_module_by_path(grown_net, module_name)
            pool, len_module, in_dim, out_dim, stride, groups = generate_module_pool(to_width_grow_repunit)
            sampled_size = random.choice(pool)
            new_module_name = f"to_rep_conv_{len_module}"
            new_module = reparameterizer.RepScaledConv(
                in_dim=in_dim, out_dim=out_dim,
                kernel_size=sampled_size, stride=stride, groups=groups).to(next(to_width_grow_repunit.torep_extractor["to_rep_conv_0"].conv.parameters()).device)
            to_width_grow_repunit.torep_extractor.add_module(new_module_name, new_module)
            for param_name, param in new_module.named_parameters():
                full_param_name = f"{module_name}.torep_extractor.{new_module_name}.{param_name}"
                new_params_in_grown_net[full_param_name] = param
        logger.info("====> Insert new branches done.")

        # 初始化新参数
        to_train_params = []
        for n, p in grown_net.named_parameters():
            if n in new_params_in_grown_net:
                to_train_params.append(p)
                if args.initializer == 'zero':
                    logger.info('========> Initializing param {} as zeros...'.format(n))
                    p.data.zero_()
                elif args.initializer == 'original':
                    logger.info('========> Initializing param {} as original...'.format(n))
                elif args.initializer == 'uniform':
                    logger.info('========> Initializing param {} as uniform(min={}, max={}) ...'.format(n, -args.init_meta, args.init_meta))
                    p.data.uniform_(-args.init_meta, to=args.init_meta)
                elif args.initializer == 'gaussian':
                    logger.info('========> Initializing param {} as gaussian(mean=0, std={})...'.format(n, args.init_meta))
                    p.data.normal_(0.0, std=args.init_meta)
                elif args.initializer == 'adam':
                    logger.info('========> Initializing param {} by adam optimizer...'.format(n))
                elif args.initializer == 'global_fitting':
                    params = torch.cat([p.flatten() for p in original_net.parameters() if p.requires_grad])
                    mean = torch.mean(params).item()
                    std = torch.std(params).item()
                    logger.info(f'========> Initializing param {n} by global fitting with gaussian(mean={mean:.5f}, std={std:.5f})...')
                    p.data.normal_(mean, std=std)
                elif args.initializer == 'local_fitting':
                    try:
                        local_module = model_utils.get_module_by_path(original_net, re.search(r"(.+\.torep_extractor)", n).group(1))
                        params = torch.cat([p.flatten() for p in local_module.parameters() if p.requires_grad]) 
                        mean = torch.mean(params).item()
                        std = torch.std(params).item()
                    except:
                        mean, std = 0.0, args.init_meta
                    logger.info(f'========> Initializing param {n} by global fitting with gaussian(mean={mean:.5f}, std={std:.5f})...')
                    p.data.normal_(mean, std=std)
                else:
                    logger.fatal('Unknown --initializer.')
                    exit()
        
        if args.initializer == 'adam':
            logger.info('======> Using adam to find a good initialization')
            local_optimizer = optim.Adam(to_train_params, lr=0.001, weight_decay=5e-4)
            max_epoch = 2
            for e in range(max_epoch):
                train(args, logger, trainloader, args.device, e, grown_net, local_optimizer, criterion, None, None, None)

        new_optimizer = get_optimizer(grown_net, args)
        if args.optim_reparam:
            old_name_id_map = params_name_to_id(original_net)
            new_id_name_map = params_id_to_name(grown_net)
            new_checkpoint = deepcopy(new_optimizer.state_dict())
            old_checkpoint = deepcopy(optimizer.state_dict())
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
            new_optimizer.load_state_dict(new_checkpoint)
            logger.info("====> Inherit Optimizer done.")

        return grown_net, new_optimizer, current_arch
    


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

    # Model
    logger.info('==> Building model..')
    current_arch = list(map(int, args.net.split('-')))
    max_arch = list(map(int, args.max_net.split('-')))

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
    last_grown_layer = -1
    intervals = (args.epochs - 1) // args.grow_interval + 1
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

        if can_grow(max_arch, current_arch) and can_grow_params(args.max_params, net, logger):
            if args.grow_mode == "width-depth":
                mode = "depth" if interval % 2 == 0 else "width"
            if args.grow_mode == "width-width-depth":
                mode = "depth" if interval % 3 == 0 else "width"
            if args.grow_mode == "width-width-width-depth":
                mode = "depth" if interval % 4 == 0 else "width"
            if args.grow_mode == "width-depth-depth":
                mode = "width" if interval % 3 == 0 else "depth"
            if args.grow_mode == "width-depth-depth-depth":
                mode = "width" if interval % 4 == 0 else "depth"
            grown_package = grow(net, optimizer, current_arch, max_arch, mode, last_grown_layer, trainloader, criterion, args, logger)
            if len(grown_package) == 4:
                net, optimizer, current_arch, last_grown_layer = grown_package
            else:
                net, optimizer, current_arch = grown_package
            logger.info(" ■ Current grown arch is : " + str(current_arch))
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=True)
            logger.info(' ■ Testing new model ...')
            test(args, logger, testloader, device, (interval + 1) * args.grow_interval - 1, net, criterion, save_path)
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
