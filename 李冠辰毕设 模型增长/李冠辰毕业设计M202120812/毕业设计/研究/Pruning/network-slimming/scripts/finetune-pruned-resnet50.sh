#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
ROOT_DIR=$(realpath "$SCRIPT_DIR/..")
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/mask-impl:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

python main.py \
    --refine /root/Pruning/network-slimming/logs/resnet-50-pruned-0.5.tar/pruned.pth.tar \
    --dataset cifar10 --arch resnet --depth 56 --epochs 20 \
    --save /root/Pruning/network-slimming/logs/finetune-0.5-pruned-resnet50/

python main.py \
    --refine /root/Pruning/network-slimming/logs/resnet-50-pruned-0.7.tar/pruned.pth.tar \
    --dataset cifar10 --arch resnet --depth 56 --epochs 20 \
    --save /root/Pruning/network-slimming/logs/finetune-0.7-pruned-resnet50/