#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
ROOT_DIR=$(realpath "$SCRIPT_DIR/..")
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models/baselines:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models/adagrow:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models/randgrow:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/models/runtime:$PYTHONPATH"

python3 main_trueobs.py \
    --model get_ada_growing_basic_resnet \
    --arch 1 2 2 1 \
    --load /root/pruning/OBC/checkpoints/grown_resnet_4b/best_ckpt.pth \
    --save xxx \
    --wbits 8 --abits 8 \
    --wperweight --wasym --asym \
    --nsamples 128


# parser.add_argument('--wperweight', action='store_true')
# parser.add_argument('--wasym', action='store_true')
# parser.add_argument('--wminmax', action='store_true')
# parser.add_argument('--asym', action='store_true')
# parser.add_argument('--aminmax', action='store_true')
