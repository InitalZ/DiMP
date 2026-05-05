#!/bin/bash
# MSRAction3D preprocessing + DiMP pretrain launcher
#
# Requires scripts/msr_depth_to_npz.py (not shipped in this repo) to convert Depth/*.bin
# to msr_npz/*.npz. If you already have NPZ under MSRAction3D/msr_npz/, run only the
# 0-pretraining-dp.py block, or set SKIP_DEPTH_PREP=1 and prepare NPZ offline.
#
# Pretrain (paper): --num-points 1024 --mask-ratio 0.60; --batch-size 128 on 8× A100 (32 here if smaller GPU).

set -e

# Repository root (directory containing this script); override with env if needed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
DEPTH_ROOT="${1:-$PROJECT_ROOT/MSRAction3D/Depth}"
NPZ_ROOT="${2:-$PROJECT_ROOT/MSRAction3D/msr_npz}"
EPOCHS="${3:-200}"

echo "Depth root: $DEPTH_ROOT"
echo "NPZ root:   $NPZ_ROOT"
echo "Epochs:     $EPOCHS"
echo ""

PREP_SCRIPT="$PROJECT_ROOT/scripts/msr_depth_to_npz.py"
# Depth→NPZ: use same num-points as pretraining (1024) so stored clouds match the paper setup.
if [ "${SKIP_DEPTH_PREP:-0}" != "1" ] && [ -f "$PREP_SCRIPT" ]; then
    python "$PREP_SCRIPT" \
        --depth-root "$DEPTH_ROOT" \
        --output "$NPZ_ROOT" \
        --num-points 1024
else
  echo "Skipping depth->npz (SKIP_DEPTH_PREP=1 or missing $PREP_SCRIPT). Using existing NPZ in $NPZ_ROOT."
fi

python "$PROJECT_ROOT/0-pretraining-dp.py" \
    --dataset msr \
    --data-path "$NPZ_ROOT" \
    --log-dir "$PROJECT_ROOT/log_ssl_MSR" \
    --model Full_MSR_pretrain \
    --radius 0.1 \
    --clip-len 25 \
    --clip-stride 1 \
    --sub-clips 5 \
    --num-points 1024 \
    --batch-size 32 \
    --epochs "$EPOCHS" \
    --workers 8 \
    --mask-ratio 0.60
