#!/bin/bash
# Usage: bash run_pretrain.sh <data_path> <motion_path> [epochs]
# Example: bash run_pretrain.sh ./hoi4d_npz ./hoi4d_motion 200
#
# <motion_path>: offline CorrNet3D inter-frame displacements, one {video_stem}.npz per video
# (key "motion", same layout as the point cloud NPZ, computed at --frame-stride 2).
# Paper: --batch-size 128 on 8× A100; --mask-ratio 0.60 (DiMP); --num-points 1024.
# Reduce --batch-size if OOM.

DATA_PATH=${1:?"Please provide data path: bash run_pretrain.sh /path/to/DiMP/hoi4d_npz <motion_path>"}
MOTION_PATH=${2:?"Please provide CorrNet3D motion path: bash run_pretrain.sh <data_path> /path/to/hoi4d_motion"}
EPOCHS=${3:-200}

echo "Data path: $DATA_PATH"
echo "Motion path: $MOTION_PATH"
echo "Epochs: $EPOCHS"
echo ""

python 0-pretraining-dp.py \
    --dataset hoi4d \
    --data-path "$DATA_PATH" \
    --motion-path "$MOTION_PATH" \
    --data-meta ./datasets/hoi4d_pretrain.list \
    --log-dir ./log_ssl_hoi4d \
    --model Full_pretrain \
    --num-points 1024 \
    --clip-len 24 \
    --frame-stride 2 \
    --epochs $EPOCHS \
    --batch-size 128 \
    --workers 8 \
    --mask-ratio 0.60
