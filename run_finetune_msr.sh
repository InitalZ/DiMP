#!/bin/bash
# MSRAction3D classification finetune launcher
# Paper: --num-points 2048; batch size may differ from 8× A100 pretrain (128).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
NPZ_ROOT="${1:-$PROJECT_ROOT/MSRAction3D/msr_npz}"
CHECKPOINT="${2:-$PROJECT_ROOT/log_ssl_MSR/MSR_pretrain/checkpoint_last.pth}"
EPOCHS="${3:-30}"

echo "NPZ root:    $NPZ_ROOT"
echo "Checkpoint:  $CHECKPOINT"
echo "Epochs:      $EPOCHS"
echo ""

python "$PROJECT_ROOT/1-finetune-dp.py" \
    --dataset msr \
    --data-path "$NPZ_ROOT" \
    --finetune "$CHECKPOINT" \
    --output-dir "$PROJECT_ROOT/log_finetune_MSR" \
    --log-dir "$PROJECT_ROOT/log_finetune_MSR" \
    --model MSR_cls \
    --radius 0.1 \
    --clip-len 24 \
    --clip-stride 1 \
    --num-points 2048 \
    --batch-size 32 \
    --epochs "$EPOCHS" \
    --lr 0.0005 \
    --lr-warmup-epochs 2 \
    --lr-milestones 20 25
