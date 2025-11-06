#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
ROOT_DIR=$(realpath "$SCRIPT_DIR/..")
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/mask-impl:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# python resprune.py --dataset cifar10 --depth 56 \
#     --percent 0.5 \
#     --model /root/Pruning/network-slimming/logs/resnet-50-pretrained.tar \
#     --save /root/Pruning/network-slimming/logs/resnet-50-pruned-0.5.tar

# SECONDS=0
time python resprune.py --dataset cifar10 --depth 56 \
    --percent 0.7 \
    --model /root/Pruning/network-slimming/logs/resnet-50-pretrained.tar \
    --save /root/Pruning/network-slimming/logs/resnet-50-pruned-0.7.tar
# echo "The command took $SECONDS seconds."