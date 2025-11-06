#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
ROOT_DIR=$(realpath "$SCRIPT_DIR/..")
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/mask-impl:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

python main.py --dataset cifar10 --arch resnet --depth 56