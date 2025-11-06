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
    --model get_ada_growing_basic_resnet --model-arch 1-2-2 \
    --num-classes 10 --image-channels 3 --image-size 32 \
    --batch-size 128 --length 300 \
    --params-and-flops --latency --throughput --do-clever-format

python3 benchmark.py \
    --model get_ada_growing_basic_resnet --model-arch 1-2-2-1 \
    --num-classes 10 --image-channels 3 --image-size 32 \
    --batch-size 128 --length 300 \
    --params-and-flops --latency --throughput --do-clever-format

python3 benchmark.py \
    --model get_ada_growing_bottleneck_resnet --model-arch 2-2-2 \
    --num-classes 10 --image-channels 3 --image-size 32 \
    --batch-size 128 --length 300 \
    --params-and-flops --latency --throughput --do-clever-format

python3 benchmark.py \
    --model get_ada_growing_bottleneck_resnet --model-arch 2-2-2-2 \
    --num-classes 10 --image-channels 3 --image-size 32 \
    --batch-size 128 --length 300 \
    --params-and-flops --latency --throughput --do-clever-format

python3 benchmark.py \
    --model get_ada_growing_vgg --model-arch 1-2-2 \
    --num-classes 10 --image-channels 3 --image-size 32 \
    --batch-size 128 --length 300 \
    --params-and-flops --latency --throughput --do-clever-format

python3 benchmark.py \
    --model get_ada_growing_vgg --model-arch 1-1-2-2 \
    --num-classes 10 --image-channels 3 --image-size 32 \
    --batch-size 128 --length 300 \
    --params-and-flops --latency --throughput --do-clever-format

python3 benchmark.py \
    --model get_ada_growing_mobilenetv3 --model-arch 2-3-3 \
    --num-classes 10 --image-channels 3 --image-size 32 \
    --batch-size 128 --length 300 \
    --params-and-flops --latency --throughput --do-clever-format

python3 benchmark.py \
    --model get_ada_growing_mobilenetv3 --model-arch 2-2-3-2 \
    --num-classes 10 --image-channels 3 --image-size 32 \
    --batch-size 128 --length 300 \
    --params-and-flops --latency --throughput --do-clever-format