#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
ROOT_DIR=$(realpath "$SCRIPT_DIR/../../..")
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models/adagrow:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models/randgrow:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models/runtime:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models/baselines:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0


python3 benchmark.py \
    --model get_ada_growing_vit_patch2_32 --model-arch d4h6 \
    --num-classes 10 --image-channels 3 --image-size 32 \
    --batch-size 128 --length 300 \
    --params-and-flops --latency --throughput --do-clever-format


python3 benchmark.py \
    --model get_ada_growing_vit_patch4_64 --model-arch d6h6 \
    --num-classes 200 --image-channels 3 --image-size 64 \
    --batch-size 128 --length 300 \
    --params-and-flops --latency --throughput --do-clever-format


python3 benchmark.py \
    --model get_ada_growing_vit_patch16_224 --model-arch d5h7 \
    --num-classes 10 --image-channels 3 --image-size 224 \
    --batch-size 128 --length 300 \
    --params-and-flops --latency --throughput --do-clever-format


python3 benchmark.py \
    --model get_ada_growing_vit_patch16_224 --model-arch d6h7 \
    --num-classes 100 --image-channels 3 --image-size 224 \
    --batch-size 128 --length 300 \
    --params-and-flops --latency --throughput --do-clever-format