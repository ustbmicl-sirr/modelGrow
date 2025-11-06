#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
ROOT_DIR=$(realpath "$SCRIPT_DIR/..")
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

python cifar_pretrain.py -l 56 \
    --save_dir "./cifarmodel" --epochs 100 \
    --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4