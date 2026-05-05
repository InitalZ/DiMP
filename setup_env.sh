#!/bin/bash
# One-shot DiMP environment setup (Linux server)
# Usage: bash setup_env.sh [cuda_version]
# cuda_version: 102 | 118 | 121 (optional; auto-detected if omitted)

set -e

echo "=========================================="
echo "  DiMP environment setup"
echo "=========================================="

# Detect CUDA version
if [ -z "$1" ]; then
    if command -v nvcc &> /dev/null; then
        CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/' | tr -d '.')
        echo "Detected CUDA version: $CUDA_VER"
    else
        CUDA_VER="118"
        echo "nvcc not found; using default PyTorch (CUDA 11.8)"
    fi
else
    CUDA_VER=$1
fi

# Create conda environment
echo ""
echo "[1/5] Creating Conda environment..."
conda create -n dimp python=3.8 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dimp

# Install PyTorch
echo ""
echo "[2/5] Installing PyTorch..."
case $CUDA_VER in
    102)
        pip install torch==1.7.1+cu102 torchvision==0.8.2+cu102 -f https://download.pytorch.org/whl/torch_stable.html
        ;;
    118|121)
        pip install torch torchvision
        ;;
    *)
        pip install torch torchvision
        ;;
esac

# Install Python dependencies
echo ""
echo "[3/5] Installing Python dependencies..."
pip install numpy timm einops tensorboardX pyyaml termcolor

# Build PointNet++ extension
echo ""
echo "[4/5] Building PointNet++ CUDA extension..."
cd modules
python setup.py install
cd ..

# Build Chamfer Distance extension
echo ""
echo "[5/5] Building Chamfer Distance extension..."
cd extensions/chamfer_dist
python setup.py install
cd ../..

echo ""
echo "=========================================="
echo "  Setup finished."
echo "=========================================="
echo ""
echo "Activate: conda activate dimp"
echo ""
echo "Pretrain: python 0-pretraining-dp.py --dataset hoi4d --data-path ./hoi4d_npz --data-meta ./datasets/your_meta.list"
echo "Finetune: python 1-finetune-dp.py --finetune ./log_ssl_hoi4d/pretrain/checkpoint_199.pth"
echo ""
echo "See DEPLOYMENT.md for details"
