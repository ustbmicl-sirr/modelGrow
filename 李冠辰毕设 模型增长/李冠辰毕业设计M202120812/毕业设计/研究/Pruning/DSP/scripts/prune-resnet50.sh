#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
ROOT_DIR=$(realpath "$SCRIPT_DIR/..")
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

python cifar_dsp.py -l 56 -g 4 -r 5e-4

python cifar_finetune.py -l 56 -g 4 -p 0.5

python cifar_finetune.py -l 56 -g 4 -p 0.7