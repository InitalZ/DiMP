#!/bin/bash
# Usage: bash run_finetune.sh <data_path> <pretrained_checkpoint>
# Example: bash run_finetune.sh ./hoi4d_npz ./log_ssl_hoi4d/pretrain/checkpoint_199.pth

DATA_PATH=${1:?"Please provide data path"}
CHECKPOINT=${2:?"Please provide pretrained checkpoint path: bash run_finetune.sh <data_path> <checkpoint_path>"}

echo "Data path: $DATA_PATH"
echo "Checkpoint: $CHECKPOINT"
echo ""

python 1-finetune-dp.py \
    --dataset hoi4d \
    --data-path "$DATA_PATH" \
    --data-meta ./datasets/hoi4d_finetune.list \
    --finetune "$CHECKPOINT" \
    --output-dir ./log_finetune_hoi4d \
    --log-dir ./log_finetune_hoi4d \
    --num-points 2048 \
    --epochs 20 \
    --batch-size 48 \
    --lr 0.0005
