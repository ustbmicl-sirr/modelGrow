import argparse
import torch
import numpy as np
from thop import profile, clever_format
from tqdm import trange
import pandas as pd
import timm
import models


def tuple_of_ints(string):
    try:
        return tuple(map(int, string.strip('()').split('-')))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value for tuple_of_ints: {string}")


@torch.inference_mode()
def throughput(model, shape, optimal_batch_size=64, length=1000, warmup=20, args=None):
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    if args.bert_mode:
        dummy_input = torch.randint(1, 30000, shape, device=device)
    else:
        dummy_input = torch.randn(*shape, dtype=torch.float).to(device)
    total_time = 0
    for _ in range(warmup):
        _ = model(dummy_input)
    with torch.no_grad():
        for _ in trange(length):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (length*optimal_batch_size) / total_time
    return Throughput


@torch.inference_mode()
def speed(model, shape, length=1000, warmup=20, args=None):
    device = torch.device("cuda")
    model.to(device)
    if args.bert_mode:
        dummy_input = torch.randint(1, 30000, shape, device=device)
    else:
        dummy_input = torch.randn(*shape, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((length,1))
    for _ in range(warmup):
        _ = model(dummy_input)
    with torch.no_grad():
        for rep in trange(length):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / length
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    return mean_syn, std_syn, mean_fps


@torch.inference_mode()
def get_flops_and_params(model, shape, do_clever_format=False, args=None):
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    if args.bert_mode:
        dummy_input = torch.randint(1, 30000, shape, device=device)
    else:
        dummy_input = torch.randn(*shape, dtype=torch.float).to(device)
    flops, params = profile(model, (dummy_input ,), verbose=False)
    if do_clever_format:
        return clever_format([flops, params])
    return flops, params


def get_module(name, arch, args):
    if hasattr(args, "depth"):
        net = getattr(models, name)(depth=args.depth, heads=args.head, num_classes=args.num_classes)
        net = net.to('cuda')
        return net
    try:
        if args.bert_mode:
            net = getattr(models, name)(depth=arch, num_classes=args.num_classes)
        elif isinstance(arch, tuple):
            net = getattr(models, name)(depths=arch, num_classes=args.num_classes, image_channels=args.image_channels)
        elif isinstance(arch, int):
            net = getattr(models, name)(depth=arch, num_classes=args.num_classes, image_channels=args.image_channels)
        else:
            net = getattr(models, name)(num_classes=args.num_classes, image_channels=args.image_channels)
    except:
        net = timm.create_model(name, num_classes=args.num_classes, in_chans=args.image_channels)
    net = net.to('cuda')
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Options')
    parser.add_argument('--model', default='get_growing_mobilenetv3', type=str)
    parser.add_argument('--model-arch', default='-', type=str)
    parser.add_argument('--num-classes', default=10, type=int)
    parser.add_argument('--image-channels', default=3, type=int)
    parser.add_argument('--image-size', default=224, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--warmup', default=20, type=int)
    parser.add_argument('--length', default=1000, type=int)
    parser.add_argument('--params-and-flops', action="store_true")
    parser.add_argument('--latency', action="store_true")
    parser.add_argument('--throughput', action="store_true")
    parser.add_argument('--do-clever-format', action="store_true")
    parser.add_argument('--bert-mode', action="store_true")
    parser.add_argument('--seqlen', default=256, type=int)
    args = parser.parse_args()
    
    target_size_latency = [1, args.image_channels, args.image_size, args.image_size]
    target_size_throughput = [args.batch_size, args.image_channels, args.image_size, args.image_size]
    if args.bert_mode:
        target_size_latency = [1, args.seqlen]
        target_size_throughput = [args.batch_size, args.seqlen]
    if args.model_arch[0] == "d":
        import re
        depth, head = [int(match) for match in re.findall(r'\d+', args.model_arch)]
        args.depth = depth
        args.head = head
        model_arch = ""
    elif len(args.model_arch) > 1:
        model_arch = tuple_of_ints(args.model_arch)
    elif len(args.model_arch) == 1 and args.model_arch != '-':
        model_arch = int(args.model_arch)
    else:
        model_arch = None
    model = get_module(name=args.model, arch=model_arch, args=args)

    results = {}
    results["shape"] = str(target_size_latency)
    results["model"] = args.model
    results["model-arch"] = args.model_arch

    if args.params_and_flops:
        flops, params = get_flops_and_params(model, target_size_latency, args.do_clever_format, args=args)
        results["flops"] = flops
        results["params"] = params
    if args.latency:
        mean_latency, std_latency, mean_fps = speed(model, target_size_latency, args.length, args.warmup, args=args)
        results["mean_latency"] = mean_latency
        results["std_latency"] = std_latency
        results["mean_fps"] = mean_fps
    if args.throughput:
        throughput_total = throughput(model, target_size_throughput, args.batch_size, args.length, args.warmup, args=args)
        results["throughput"] = throughput_total
    
    data_items = list(results.items())
    df = pd.DataFrame(data_items, columns=['Attribute', 'Value'])
    markdown_table = df.to_markdown(index=False)
    print(markdown_table)
    print()
    