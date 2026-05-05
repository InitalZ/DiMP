# DiMP server deployment guide

End-to-end steps to run DiMP pretraining and finetuning on a server.

---

## 1. Requirements

| Component | Minimum |
| --------- | ------- |
| OS | Linux (Ubuntu 18.04+ recommended) |
| GPU | NVIDIA GPU with ≥ 8 GB VRAM |
| CUDA | 10.2 or 11.x / 12.x |
| Python | 3.7 – 3.10 |
| Disk | ~150 GB free (including datasets) |

---

## 2. Environment setup

Install **Python 3.8–3.10** (Conda optional). Install **PyTorch** exactly as directed on the official page: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) (choose your platform and CUDA version; use the generated `pip` or `conda` command—**do not** point `pip` at unofficial package indexes). From the repository root, install the rest with `pip install -r requirements.txt`.

Build the **CUDA extensions** (required):

```bash
cd /path/to/DiMP
cd modules && python setup.py install && cd ..
cd extensions/chamfer_dist && python setup.py install && cd ../..
```

If compilation fails, align the NVIDIA driver / CUDA toolkit with the PyTorch build you installed and see the [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

---

## 3. Data preparation

### 3.1 Point cloud video format (NPZ)

- **HOI4D** — one flattened `.npz` per clip/sequence. Loaders read `np.load(path, allow_pickle=True)['data']` with shape `(num_frames, num_points, 3)`. Optional scaling: meters → centimeters (`* 100`) in dataset code; keep preprocessing consistent.
- **MSRAction3D** — pretraining / classification (`--dataset msr`) read `['point_clouds']` with the same frame/point layout. Filenames must match `a##_s##_e##.npz` (see Section 3.4).

### 3.2 Supported datasets

- **HOI4D** — layouts in Section 3.3; you need a **meta list** (Section 3.5) and `--data-path` → `DiMP/hoi4d_npz/` (flat NPZ at the repo root).
- **MSRAction3D** — layout in Section 3.4; `--data-path` should point at the folder that **directly contains** the `a##_s##_e##.npz` files (usually `msr_npz/` or `smoke_npz/`). Optional helpers: `run_pretrain_msr.sh`, `run_finetune_msr.sh`.

### 3.3 HOI4D — flat preprocessed NPZ under DiMP (`hoi4d_npz/`)

Store the HOI4D NPZ export **under your DiMP repository root** (same level as `modules/`, `datasets/`, etc.—often gitignored). Samples live in a **single flat directory** (no subfolders):

```
DiMP/hoi4d_npz/
├── ZY20210800001_H1_C11_N07_S185_s02_T2.npz
├── ZY20210800001_H1_C11_N08_S185_s02_T2.npz
└── …
```

- **Naming:** stems align with annotation paths. Pattern is  
  `{ZY}_{H#}_{C##}_{N##}_{S###}_{s##}_{T#}.npz`  
  (exact digit widths follow your actual exports; example stem above is illustrative).

Set `--data-path` / `root` to `DiMP/hoi4d_npz`, e.g. `/path/to/DiMP/hoi4d_npz` or `./hoi4d_npz` from the repo root. Meta list lines use the same stem without `.npz` (Section 3.5).

### 3.4 MSRAction3D — layout under the DiMP project (`MSRAction3D/`)

Place the MSR tree **under your DiMP repository root** (same level as `modules/`, `datasets/`, etc.—often listed in `.gitignore`). It uses **flat NPZ exports** under `msr_npz/` (or `smoke_npz/`), plus raw depth, skeleton text, and archives:

```
DiMP/MSRAction3D/
├── Depth/                                   # depth sequences (binary)
│   └── a##_s##_e##_sdepth.bin               # e.g. a01_s01_e01_sdepth.bin
├── MSRAction3DSkeleton(20joints)/           # 20-joint 2D skeleton (text)
│   └── a##_s##_e##_skeleton.txt
├── msr_npz/                                 # preprocessed NPZ (training path for this repo)
│   ├── _meta.txt                            # optional: ids, frame counts, etc. (one line per sample)
│   └── a##_s##_e##.npz
├── smoke_npz/                               # small/smoke set, same idea (may include _meta.txt)
├── a##_s##_e##_skeleton3D.txt               # 3D skeleton sequence at MSRAction3D root (text)
├── Depth.rar
├── MSRAction3DSkeleton(20joints).rar
└── MSRAction3DSkeletonReal3D.rar
```

**Naming (official MSRAction3D convention):**

- `a##` — action id  
- `s##` — subject (actor)  
- `e##` — execution / repetition of the same action  

Example: `a01_s03_e02_skeleton3D.txt` — action 1, subject 3, 2nd repetition, 3D skeleton file.

**For DiMP scripts:** pass `--data-path` to `DiMP/MSRAction3D/msr_npz` (or `smoke_npz`)—e.g. `/path/to/DiMP/MSRAction3D/msr_npz` or `./MSRAction3D/msr_npz` from the repo root—not `MSRAction3D/` itself. This repo’s loaders scan that directory for `a##_s##_e##.npz` and do not require `_meta.txt`.

### 3.5 HOI4D meta list format

Text file: one line per sequence (whitespace-separated):

```
video_stem num_frames [class_id]
```

- **video_stem** — basename without `.npz`; file must exist as `{DiMP/hoi4d_npz}/{video_stem}.npz` (same directory as `--data-path`).
- **num_frames** — frame count (must be enough for clip length and frame stride).
- **class_id** — optional; required for HOI4D classification (`1-finetune-dp.py`).

Example:

```
ZY20210800001_H1_C11_N07_S185_s02_T2 120 0
ZY20210800001_H1_C11_N08_S185_s02_T2 95 3
```

Use separate lists for train / val as needed (`--data-meta`, `--meta-train`, `--meta-val`, etc.).

---

## 4. Pretraining

**Paper (appendix) alignment:** **Masking ratio 0.60.** **HOI4D and MSR pretraining** use **1,024 points per frame**. **Batch size 128** is reported on **8× A100**; smaller GPUs should reduce batch size (examples below use 32 with an explicit comment).

> **DiMP vs. baseline:** `--model` must start with **`Full_`** to use **`DiMPModelFull`** (paper DiMP). Otherwise the script uses **`DiMPModel`** (SHOT-style baseline). The examples below use **`Full_pretrain`** / **`Full_MSR_pretrain`**.

### 4.1 HOI4D

`--data-meta` is **required** for `hoi4d`.

```bash
conda activate dimp
cd /path/to/DiMP

# Paper: --batch-size 128 on 8× A100; reduce if OOM.
python 0-pretraining-dp.py \
    --dataset hoi4d \
    --data-path /path/to/DiMP/hoi4d_npz \
    --data-meta ./datasets/hoi4d_pretrain.list \
    --log-dir ./logs \
    --model Full_pretrain \
    --num-points 1024 \
    --epochs 200 \
    --batch-size 32 \
    --workers 8 \
    --mask-ratio 0.60
```

### 4.2 MSRAction3D

```bash
# Paper: --batch-size 128 on 8× A100; --mask-ratio 0.60.
python 0-pretraining-dp.py \
    --dataset msr \
    --data-path /path/to/DiMP/MSRAction3D/msr_npz \
    --log-dir ./logs \
    --model Full_MSR_pretrain \
    --radius 0.1 \
    --num-points 1024 \
    --epochs 200 \
    --batch-size 32 \
    --workers 8 \
    --mask-ratio 0.60
```

More flags (center / motion diffusion): `python 0-pretraining-dp.py --help`.

### 4.3 Common arguments

| Argument | Role |
| -------- | ---- |
| `--data-path` | Flat NPZ folder: `/path/to/DiMP/hoi4d_npz` (HOI4D) or `.../DiMP/MSRAction3D/msr_npz` (MSR) |
| `--data-meta` | Meta list (**required** for `hoi4d`) |
| `--dataset` | `hoi4d` or `msr` |
| `--num-points` | **1024** for pretraining (appendix); script default matches paper |
| `--log-dir` | Logs and checkpoints |
| `--epochs` | Training epochs |
| `--batch-size` | **128** in paper (8× A100); lower if OOM |
| `--workers` | DataLoader workers |
| `--mask-ratio` | **0.60** (pretraining, appendix) |
| `--resume` | Resume checkpoint path |

### 4.4 Outputs

Checkpoints are written under `{log_dir}/{model}/`, e.g. `logs/Full_pretrain/checkpoint_last.pth` and optionally `checkpoint_{epoch}.pth` when `--save-freq` is set.

---

## 5. Finetuning (classification)

After pretraining, load weights with `--finetune`.

### MSRAction3D

Point `--data-path` at the directory that contains `a##_s##_e##.npz` (usually `msr_npz/`).

```bash
# MSR classification: 2048 pts/frame (appendix).
python 1-finetune-dp.py \
    --dataset msr \
    --data-path /path/to/DiMP/MSRAction3D/msr_npz \
    --finetune ./logs/Full_pretrain/checkpoint_last.pth \
    --log-dir ./log_finetune_msr \
    --output-dir ./log_finetune_msr \
    --num-points 2048 \
    --epochs 20 \
    --batch-size 48 \
    --lr 0.0005
```

### HOI4D

```bash
# HOI4D subject classification: 2048 pts/frame (appendix).
python 1-finetune-dp.py \
    --dataset hoi4d \
    --data-path /path/to/DiMP/hoi4d_npz \
    --data-meta ./datasets/hoi4d_finetune_train.list \
    --val-meta ./datasets/hoi4d_finetune_val.list \
    --finetune ./logs/Full_pretrain/checkpoint_last.pth \
    --log-dir ./log_finetune_hoi4d \
    --output-dir ./log_finetune_hoi4d \
    --num-points 2048 \
    --epochs 20 \
    --batch-size 48 \
    --lr 0.0005
```

**Notes**

- `--finetune`: path to pretraining checkpoint.
- **`--num-points`:** **1024** during **pretraining** (appendix); **2048** for **HOI4D action segmentation** and **MSR / HOI4D classification** finetuning (appendix). **HOI4D semantic segmentation** uses **4096** (`2-finetune-sem-seg.py`).
- If you changed `--num-points` during pretraining, align finetuning for a strict reproduction or follow the appendix table exactly.

Semantic segmentation (**4096** pts/frame) and temporal action segmentation (**2048** pts/frame):

```bash
python 2-finetune-sem-seg.py \
    --data-path /path/to/DiMP/hoi4d_npz \
    --meta-train ./datasets/semseg_train.list \
    --meta-val ./datasets/semseg_val.list \
    --pretrained /path/to/pretrain/checkpoint_last.pth \
    --num-points 4096 \
    --log-dir ./logs \
    --model sem_seg

python 2-finetune-action-seg.py \
    --data-path /path/to/DiMP/hoi4d_npz \
    --meta-train ./datasets/action_train.list \
    --meta-val ./datasets/action_val.list \
    --pretrained /path/to/pretrain/checkpoint_last.pth \
    --num-points 2048 \
    --log-dir ./logs \
    --model action_seg
```

See [README.md](README.md) for more; script defaults match these appendix values where noted above.

---

## 6. Troubleshooting

### Q1: `ModuleNotFoundError: No module named 'pointnet2'`

```bash
cd modules && python setup.py install && cd ..
```

### Q2: `ModuleNotFoundError: No module named 'chamfer'`

```bash
cd extensions/chamfer_dist && python setup.py install && cd ../..
```

### Q3: `RuntimeError: symeig` or `AttributeError: 'module' has no attribute 'symeig'`

Fixed in this repo via `torch.linalg.eigh` on PyTorch 2.x. If it persists, use the latest `modules/SHOT.py`.

### Q4: CUDA out of memory

- Lower `--batch-size` (e.g. 64, 32)
- Lower `--num-points` (e.g. 1024)

### Q5: Slow data loading

- Raise `--workers` (e.g. 16, 32)
- Put data on SSD or local disk

### Q6: Single-GPU `'DiMPModel' object has no attribute 'module'`

On one GPU `model` may not be wrapped in `DataParallel`, but `add_weight_decay` uses `model.module`. You can always wrap when CUDA is available:

In `0-pretraining-dp.py` and `1-finetune-dp.py`, change:

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

to:

```python
if torch.cuda.is_available():
    model = nn.DataParallel(model)
```

### Q7: Smoke test without a full dataset

1. Create a few tiny `.npz` files with the expected `data` array shape.
2. Write a short meta list pointing to them.
3. Run pretraining with `--epochs 2` to verify the environment.

---

## 7. Quick sanity-check script

The repo may include `run_pretrain.sh` (HOI4D). Example inline:

```bash
#!/bin/bash
# Smoke test only; paper settings: --num-points 1024 --mask-ratio 0.60 --batch-size 128 (8× A100).
python 0-pretraining-dp.py \
    --dataset hoi4d \
    --data-path "$1" \
    --data-meta ./datasets/hoi4d_pretrain.list \
    --log-dir ./log_test \
    --model Full_pretrain \
    --num-points 1024 \
    --epochs 2 \
    --batch-size 8 \
    --workers 4 \
    --mask-ratio 0.60
```

Run: `bash run_pretrain.sh /path/to/DiMP/hoi4d_npz`

---

## 8. Checklist

```
1. conda create -n dimp python=3.8
2. conda activate dimp
3. pip install torch torchvision
4. pip install -r requirements.txt
5. cd modules && python setup.py install && cd ..
6. cd extensions/chamfer_dist && python setup.py install && cd ../..
7. Prepare `DiMP/hoi4d_npz/` (flat) + HOI4D meta lists, and/or `DiMP/MSRAction3D/msr_npz/`
8. Pretrain: `--num-points 1024 --mask-ratio 0.60` (appendix); batch 128 on 8× A100 in paper; for `hoi4d` include `--data-meta`.
9. Finetune: classification/action seg `--num-points 2048` (HOI4D/MSR); semantic seg `--num-points 4096`.
10. python 1-finetune-dp.py / 2-finetune-*.py with `--finetune` or `--pretrained` as appropriate.
```
