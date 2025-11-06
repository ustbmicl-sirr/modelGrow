import argparse
import copy
import os
import torch
import torch.nn as nn
from datautils import *
from modelutils import *
from quant import *
from trueobs import *
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--arch', nargs='+', type=int, default=[], help='Architecture list for the model, if applicable')
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--compress', type=str, default="quant", choices=['quant', 'nmprune', 'unstr', 'struct', 'blocked'])
parser.add_argument('--load', type=str, default='')
parser.add_argument('--datapath', type=str, default='')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save', type=str, default='')
parser.add_argument('--nsamples', type=int, default=128)
parser.add_argument('--batchsize', type=int, default=-1)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--nrounds', type=int, default=-1)
parser.add_argument('--noaug', action='store_true')
parser.add_argument('--wbits', type=int, default=32)
parser.add_argument('--abits', type=int, default=32)
parser.add_argument('--wperweight', action='store_true')
parser.add_argument('--wasym', action='store_true')
parser.add_argument('--wminmax', action='store_true')
parser.add_argument('--asym', action='store_true')
parser.add_argument('--aminmax', action='store_true')
parser.add_argument('--rel-damp', type=float, default=0)
parser.add_argument('--prunen', type=int, default=2)
parser.add_argument('--prunem', type=int, default=4)
parser.add_argument('--min-sparsity', type=float, default=0)
parser.add_argument('--max-sparsity', type=float, default=0)
parser.add_argument('--delta-sparse', type=float, default=0)
parser.add_argument('--sparse-dir', type=str, default='')
args = parser.parse_args()

trainloader, dataloader, testloader = get_loaders(
    args.dataset, path=args.datapath, 
    batchsize=args.batchsize, workers=args.workers,
    nsamples=args.nsamples, seed=args.seed,
    noaug=args.noaug
)
if args.nrounds == -1:
    args.nrounds = 10 
    if args.noaug:
        args.nrounds = 1

aquant = args.compress == 'quant' and args.abits < 32
wquant = args.compress == 'quant' and args.wbits < 32

# modelp = get_model(args.model, args.arch)
if args.compress == 'quant' and args.load:
    modelp = torch.load(args.load)["deploy"]
    # new_state_dict = {k.replace("module.", ""): v for k, v in original_state_dict.items()}
    # modelp.load_state_dict(new_state_dict)
    # train(modelp, trainloader)
    test(modelp, testloader)
if aquant:
    add_actquant(modelp)
# modeld = get_model(args.model, args.arch)
modeld = copy.deepcopy(modelp)
ori_layersp = find_layers(modelp)
ori_layersd = find_layers(modeld)

if "resnet" in args.model:
    layersp = find_layers(modelp)
    layersd = find_layers(modeld)
    # layersd.pop("fc", None)
    # layersd.pop("conv1", None)
    # layersp.pop("fc", None)
    # layersp.pop("conv1", None)
    # ori_layersd.pop("fc", None)
    # ori_layersd.pop("conv1", None)
    # ori_layersp.pop("fc", None)
    # ori_layersp.pop("conv1", None)
    # layersd = {k: v for k, v in ori_layersd.items() if "downsample" not in k}
    # layersp = {k: v for k, v in ori_layersp.items() if "downsample" not in k}
    # layersd = {k: v for k, v in layersd.items() if "layer4" not in k}
    # layersp = {k: v for k, v in layersp.items() if "layer4" not in k}
else:  # vits
    layersd = {k: v for k, v in ori_layersd.items() if k.startswith("transformer") and "to" in k}
    layersp = {k: v for k, v in ori_layersp.items() if k.startswith("transformer") and "to" in k}

trueobs = {}
for name in layersp:
    layer = layersp[name]
    if isinstance(layer, ActQuantWrapper):
        layer = layer.module
    trueobs[name] = TrueOBS(layer, rel_damp=args.rel_damp)
    if aquant:
        layersp[name].quantizer.configure(
            args.abits, sym=args.asym, mse=not args.aminmax
        )
    if wquant:
        trueobs[name].quantizer = Quantizer()
        trueobs[name].quantizer.configure(
            args.wbits, perchannel=not args.wperweight, sym=not args.wasym, mse=not args.wminmax
        )

if not (args.compress == 'quant' and not wquant):
    cache = {}
    def add_batch(name):
        def tmp(layer, inp, out):
            trueobs[name].add_batch(inp[0].data, out.data)
        return tmp
    handles = []
    for name in trueobs:
        handles.append(layersd[name].register_forward_hook(add_batch(name)))
    for i in range(args.nrounds):
        for j, batch in enumerate(dataloader):
            print(i, j)
            with torch.no_grad():
                run(modeld, batch)
    for h in handles:
        h.remove()
    for name in trueobs:
        print(name)
        if args.compress == 'quant':
            print('Quantizing ...')
            trueobs[name].quantize()
        if args.compress == 'nmprune':
            if trueobs[name].columns % args.prunem == 0:
                print('N:M pruning ...')
                trueobs[name].nmprune(args.prunen, args.prunem)
        trueobs[name].free()

if aquant:
    print('Quantizing activations ...')
    def init_actquant(name):
        def tmp(layer, inp, out):
            layersp[name].quantizer.find_params(inp[0].data)
        return tmp
    handles = []
    for name in layersd:
        handles.append(layersd[name].register_forward_hook(init_actquant(name)))
    with torch.no_grad():
        run(modeld, next(iter(dataloader)))
    for h in handles:
        h.remove()

if args.save:
    torch.save(modelp.state_dict(), args.save)

test(modelp, testloader)
