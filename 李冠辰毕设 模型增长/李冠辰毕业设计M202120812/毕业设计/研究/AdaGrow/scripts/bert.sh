#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
ROOT_DIR=$(realpath "$SCRIPT_DIR/../../..")
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models/baselines:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models/adagrow:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models/randgrow:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models/runtime:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0


python3 benchmark.py \
    --model get_rand_growing_bert --bert-mode --model-arch d4h6 \
    --num-classes 2 --batch-size 128 --length 300 \
    --params-and-flops --latency --throughput --do-clever-format


python3 benchmark.py \
    --model get_rand_growing_bert --bert-mode --model-arch d3h6 \
    --num-classes 2 --batch-size 128 --length 300 \
    --params-and-flops --latency --throughput --do-clever-format