#!/bin/bash
# Usage: bash run_pretrain.sh <data_path> [epochs]
# Example: bash run_pretrain.sh ./hoi4d_npz 200
#
# Paper: --batch-size 128 on 8× A100; --mask-ratio 0.60 (DiMP); --num-points 1024.
# Reduce --batch-size if OOM.

DATA_PATH=${1:?"Please provide data path: bash run_pretrain.sh /path/to/DiMP/hoi4d_npz"}
EPOCHS=${2:-200}

echo "Data path: $DATA_PATH"
echo "Epochs: $EPOCHS"
echo ""

python 0-pretraining-dp.py \
    --dataset hoi4d \
    --data-path "$DATA_PATH" \
    --data-meta ./datasets/hoi4d_pretrain.list \
    --log-dir ./log_ssl_hoi4d \
    --model Full_pretrain \
    --num-points 1024 \
    --epochs $EPOCHS \
    --batch-size 128 \
    --workers 8 \
    --mask-ratio 0.60
