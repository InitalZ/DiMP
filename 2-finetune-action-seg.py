import os
import sys
import argparse
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'modules'))

from models.ActionSegModel import ActionSegModel
from models.ActionSegModel_Full import ActionSegModelFull
from datasets.hoi4d_action_seg import HOI4DActionSeg
from logger import setup_logger
from utils import AverageMeter, WarmupCosineLR

def edit_score(pred_seq, gt_seq, norm=True):

    p, q = [str(x) for x in pred_seq], [str(x) for x in gt_seq]

    def collapse(seq):
        out = []
        for s in seq:
            if not out or out[-1] != s:
                out.append(s)
        return out
    p = collapse(p)
    q = collapse(q)
    m, n = len(p), len(q)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[i - 1] == q[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    dist = dp[m][n]
    if norm:
        score = (1.0 - dist / max(m, n)) * 100.0 if max(m, n) > 0 else 100.0
    else:
        score = dist
    return score

def f1_at_overlap(pred_seq, gt_seq, overlap):

    def to_segments(seq):
        segs = []
        start = 0
        for i in range(1, len(seq) + 1):
            if i == len(seq) or seq[i] != seq[i - 1]:
                segs.append((seq[start], start, i - 1))
                start = i
        return segs

    pred_segs = to_segments(pred_seq)
    gt_segs = to_segments(gt_seq)

    tp, fp = 0, 0
    used = [False] * len(gt_segs)
    for p_label, p_start, p_end in pred_segs:
        best_iou = 0.0
        best_j = -1
        for j, (g_label, g_start, g_end) in enumerate(gt_segs):
            if used[j] or g_label != p_label:
                continue
            inter = max(0, min(p_end, g_end) - max(p_start, g_start) + 1)
            union = max(p_end, g_end) - min(p_start, g_start) + 1
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= overlap:
            tp += 1
            used[best_j] = True
        else:
            fp += 1
    fn = sum(1 for u in used if not u)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1 * 100.0

def get_args():
    parser = argparse.ArgumentParser('HOI4D temporal action segmentation finetuning')

    parser.add_argument('--data-path', required=True, type=str)
    parser.add_argument('--meta-train', required=True, type=str)
    parser.add_argument('--meta-val', required=True, type=str)
    parser.add_argument('--num-points', default=2048, type=int)
    parser.add_argument('--clip-len', default=32, type=int)
    parser.add_argument('--clip-step', default=8, type=int)
    parser.add_argument('--frame-stride', default=2, type=int)

    parser.add_argument('--radius', default=0.05, type=float)
    parser.add_argument('--nsamples', default=32, type=int)
    parser.add_argument('--spatial-stride', default=32, type=int)
    parser.add_argument('--temporal-kernel-size', default=3, type=int)
    parser.add_argument('--temporal-stride', default=1, type=int)
    parser.add_argument('--en-dim', default=384, type=int)
    parser.add_argument('--en-depth', default=6, type=int)
    parser.add_argument('--en-heads', default=6, type=int)
    parser.add_argument('--en-head-dim', default=64, type=int)
    parser.add_argument('--en-mlp-dim', default=768, type=int)
    parser.add_argument('--dropout1', default=0.05, type=float)
    parser.add_argument('--dropout-seg', default=0.3, type=float)
    parser.add_argument('--num-classes', default=19, type=int,
                        help='HOI4D action classes for segmentation (paper appendix: 19)')

    parser.add_argument('--pretrained', default='', type=str, help='Pretrained checkpoint')
    parser.add_argument('--resume', default='', type=str, help='Resume checkpoint (includes optimizer/epoch)')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--wd', default=0.05, type=float)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--eval-freq', default=5, type=int)

    parser.add_argument('--model', default='action_seg', type=str)
    parser.add_argument('--log-dir', default='logs/', type=str)
    parser.add_argument('--seed', default=0, type=int)

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    set_seed(args.seed)
    cudnn.benchmark = True

    log_dir = os.path.join(args.log_dir, args.model)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(output=log_dir, name=args.model)
    logger.info(args)

    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    train_dataset = HOI4DActionSeg(
        root=args.data_path, meta=args.meta_train,
        clip_len=args.clip_len, clip_step=args.clip_step,
        frame_stride=args.frame_stride, num_points=args.num_points, train=True
    )
    val_dataset = HOI4DActionSeg(
        root=args.data_path, meta=args.meta_val,
        clip_len=args.clip_len, clip_step=args.clip_step,
        frame_stride=args.frame_stride, num_points=args.num_points, train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    logger.info(f'train clips={len(train_dataset)}, val clips={len(val_dataset)}')

    num_classes = args.num_classes

    if args.model.startswith('Full_'):
        ModelCls = ActionSegModelFull
    else:
        ModelCls = ActionSegModel

    model = ModelCls(
        radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
        temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
        en_emb_dim=args.en_dim, en_depth=args.en_depth, en_heads=args.en_heads,
        en_head_dim=args.en_head_dim, en_mlp_dim=args.en_mlp_dim,
        clip_len=args.clip_len,
        num_classes=num_classes,
        dropout1=args.dropout1,
        dropout_seg=args.dropout_seg
    ).cuda()

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=True)
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        logger.info(f"=> resumed from {args.resume} (start_epoch={start_epoch})")
    elif args.pretrained:
        model.load_pretrained(args.pretrained)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.resume:
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            logger.info("=> optimizer state restored")
        else:
            logger.info("=> resume checkpoint has no optimizer; training will continue with fresh optimizer")
    scheduler = WarmupCosineLR(
        optimizer, T_max=args.epochs,
        warmup_iters=5, warmup_factor=0.0, warmup_method='linear'
        , last_epoch=start_epoch - 1
    )

    best_acc = 0.0
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, criterion, train_loader, optimizer, epoch, args, logger)
        scheduler.step()

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            frame_acc, edit, f1_10, f1_25, f1_50 = evaluate(model, val_loader, args, logger)
            logger.info(
                f'Eval: FrameAcc:{frame_acc:.2f}  Edit:{edit:.2f}  '
                f'F1_10:{f1_10:.2f}  F1_25:{f1_25:.2f}  F1_50:{f1_50:.2f}'
            )
            if frame_acc > best_acc:
                best_acc = frame_acc
                torch.save({'epoch': epoch, 'model': model.state_dict()},
                           os.path.join(log_dir, 'best.pth'))

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, os.path.join(log_dir, f'checkpoint_{epoch}.pth'))

def train_one_epoch(model, criterion, loader, optimizer, epoch, args, logger):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for i, (clips, labels) in enumerate(loader):
        clips = clips.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        logits = model(clips)
        B, T, C = logits.shape
        loss = criterion(logits.reshape(B * T, C), labels.reshape(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=-1)
        acc = (pred == labels).float().mean().item() * 100.0
        loss_meter.update(loss.item(), B)
        acc_meter.update(acc, B)

        if i % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch:[{epoch}][{i}/{len(loader)}]  '
                f'lr:{lr:.5f}  '
                f'Loss:{loss_meter.val:.4f} ({loss_meter.avg:.4f})  '
                f'FrameAcc:{acc_meter.val:.1f} ({acc_meter.avg:.1f})'
            )

@torch.no_grad()
def evaluate(model, loader, args, logger):
    model.eval()

    all_preds = []
    all_gts = []

    for clips, labels in loader:
        clips = clips.cuda(non_blocking=True)
        logits = model(clips)
        preds = logits.argmax(dim=-1).cpu().numpy()
        gts = labels.numpy()
        for b in range(preds.shape[0]):
            all_preds.append(preds[b])
            all_gts.append(gts[b])

    total_correct = sum(np.sum(p == g) for p, g in zip(all_preds, all_gts))
    total_frames = sum(len(g) for g in all_gts)
    frame_acc = total_correct / total_frames * 100.0 if total_frames > 0 else 0.0

    edit_scores = [edit_score(p.tolist(), g.tolist()) for p, g in zip(all_preds, all_gts)]
    edit = np.mean(edit_scores)

    f1_10 = np.mean([f1_at_overlap(p.tolist(), g.tolist(), 0.10) for p, g in zip(all_preds, all_gts)])
    f1_25 = np.mean([f1_at_overlap(p.tolist(), g.tolist(), 0.25) for p, g in zip(all_preds, all_gts)])
    f1_50 = np.mean([f1_at_overlap(p.tolist(), g.tolist(), 0.50) for p, g in zip(all_preds, all_gts)])

    return frame_acc, edit, f1_10, f1_25, f1_50

if __name__ == '__main__':
    args = get_args()
    main(args)
