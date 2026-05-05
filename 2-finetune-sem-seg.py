import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'modules'))

from models.SemSegModel import SemSegModel
from models.SemSegModel_Full import SemSegModelFull
from datasets.hoi4d_sem_seg import HOI4DSemSeg
from logger import setup_logger
from utils import AverageMeter, WarmupCosineLR

NUM_CLASSES = 39
# HOI4D appendix: 39 semantic classes (point labels 0..38). Add ids to this set if you
# must exclude void / unlabeled from mIoU or loss (mapped to ignore_index=-1).
MIOU_SKIP_CLASSES = frozenset()

def get_args():
    parser = argparse.ArgumentParser('HOI4D semantic segmentation finetuning')

    parser.add_argument('--data-path', required=True, type=str)
    parser.add_argument('--meta-train', required=True, type=str)
    parser.add_argument('--meta-val', required=True, type=str)
    parser.add_argument('--num-points', default=4096, type=int,
                        help='Points per frame (paper appendix: 4096 for HOI4D semantic segmentation)')
    parser.add_argument('--clip-len', default=3, type=int,
                        help='Frames per clip; default 3 aligns with C2P / HOI4D_SemSeg')
    parser.add_argument('--clip-step', default=3, type=int,
                        help='Stride between clip start frames (default 3 when clip_len=3)')
    parser.add_argument('--frame-stride', default=1, type=int)

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
    parser.add_argument('--num-classes', default=NUM_CLASSES, type=int,
                        help='HOI4D semantic classes (paper appendix: 39)')

    parser.add_argument('--pretrained', default='', type=str)
    parser.add_argument('--resume', default='', type=str,
                        help='Resume from finetune checkpoint (model/optimizer/best_miou); mutually exclusive with --pretrained')
    parser.add_argument('--best-miou', default=-1.0, type=float,
                        help='If resumed ckpt lacks best_mIoU, use this as historical best (-1: run val once to estimate)')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--lr-encoder', default=-1.0, type=float,
                        help='Encoder LR (<0: same as --lr; 0: freeze encoder)')
    parser.add_argument('--wd', default=0.05, type=float)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--eval-freq', default=5, type=int)

    parser.add_argument('--model', default='sem_seg', type=str)
    parser.add_argument('--log-dir', default='logs/', type=str)
    parser.add_argument('--seed', default=0, type=int)

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_miou(pred, target, num_classes, skip_classes=MIOU_SKIP_CLASSES):

    iou_list = []
    for cls in range(num_classes):
        if cls in skip_classes:
            continue
        gt_c = (target == cls)
        if gt_c.sum() == 0:
            continue
        pred_c = (pred == cls)
        inter = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        if union > 0:
            iou_list.append(inter / union)
    return float(np.mean(iou_list)) * 100.0 if iou_list else 0.0

def prepare_labels_for_loss(labels, skip_classes=MIOU_SKIP_CLASSES):

    labels_for_loss = labels.clone()
    for cls in skip_classes:
        labels_for_loss[labels_for_loss == cls] = -1
    return labels_for_loss

def build_optimizer(model, args, logger):
    if args.lr_encoder < 0:
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    encoder_modules = [model.tube_embedding, model.encoder_transformer]
    if hasattr(model, 'encoder_pos_linear'):
        encoder_modules.append(model.encoder_pos_linear)
    if hasattr(model, 'encoder_pos_embed'):
        encoder_modules.append(model.encoder_pos_embed)
    if hasattr(model, 'encoder_norm'):
        encoder_modules.append(model.encoder_norm)
    head_modules = [model.seg_head]

    encoder_params = []
    for module in encoder_modules:
        encoder_params.extend(list(module.parameters()))
    head_params = []
    for module in head_modules:
        head_params.extend(list(module.parameters()))

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.lr_encoder},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=args.wd)

    if args.lr_encoder == 0:
        logger.info(f'Encoder frozen (lr=0), seg head lr={args.lr}')
    else:
        logger.info(f'Encoder lr={args.lr_encoder}, seg head lr={args.lr}')
    return optimizer

def main(args):
    set_seed(args.seed)
    cudnn.benchmark = True

    num_classes = args.num_classes

    log_dir = os.path.join(args.log_dir, args.model)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(output=log_dir, name=args.model)
    logger.info(args)

    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    train_dataset = HOI4DSemSeg(
        root=args.data_path, meta=args.meta_train,
        clip_len=args.clip_len, clip_step=args.clip_step,
        frame_stride=args.frame_stride, num_points=args.num_points, train=True
    )
    val_dataset = HOI4DSemSeg(
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

    if args.model.startswith('Full_'):
        ModelCls = SemSegModelFull
    else:
        ModelCls = SemSegModel

    model = ModelCls(
        radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
        temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
        en_emb_dim=args.en_dim, en_depth=args.en_depth, en_heads=args.en_heads,
        en_head_dim=args.en_head_dim, en_mlp_dim=args.en_mlp_dim,
        num_classes=num_classes,
        dropout1=args.dropout1,
        dropout_seg=args.dropout_seg,
    ).cuda()

    if args.resume:
        if args.pretrained:
            logger.info('--resume set; ignoring --pretrained (weights from checkpoint)')
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        start_epoch = int(ckpt['epoch']) + 1
        if 'best_miou' in ckpt:
            best_miou = float(ckpt['best_miou'])
        elif args.best_miou >= 0.0:
            best_miou = args.best_miou
            logger.info(f'===> Using --best-miou as historical best: {best_miou:.2f}')
        else:
            best_miou = evaluate(model, val_loader, args, logger, num_classes)
            logger.info(
                f'===> Old checkpoint has no best_mIoU; using current val mIoU={best_miou:.2f} as best baseline; '
                f'if this disagrees with your logs, resume with --best-miou'
            )
        logger.info(f'===> Resume from {args.resume}  start_epoch={start_epoch}  best_mIoU={best_miou:.2f}')
    elif args.pretrained:
        model.load_pretrained(args.pretrained)
        start_epoch = 0
        best_miou = 0.0
    else:
        start_epoch = 0
        best_miou = 0.0

    criterion = nn.CrossEntropyLoss(ignore_index=-1).cuda()

    optimizer = build_optimizer(model, args, logger)
    if args.resume:
        optimizer.load_state_dict(ckpt['optimizer'])
        for g in optimizer.param_groups:
            if 'initial_lr' not in g:
                g['initial_lr'] = g['lr']
    sched_last = start_epoch - 1 if start_epoch > 0 else -1
    scheduler = WarmupCosineLR(
        optimizer, T_max=args.epochs,
        warmup_iters=5, warmup_factor=0.0, warmup_method='linear',
        last_epoch=sched_last
    )

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, criterion, train_loader, optimizer, epoch, args, logger, num_classes)
        scheduler.step()

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            miou = evaluate(model, val_loader, args, logger, num_classes)
            logger.info(f'Eval: mIoU:{miou:.2f}')
            if miou > best_miou:
                best_miou = miou
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_miou': best_miou,
                }, os.path.join(log_dir, 'best.pth'))

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_miou': best_miou,
        }, os.path.join(log_dir, f'checkpoint_{epoch}.pth'))

def train_one_epoch(model, criterion, loader, optimizer, epoch, args, logger, num_classes):
    model.train()
    loss_meter = AverageMeter()
    miou_meter = AverageMeter()

    for i, (clips, labels) in enumerate(loader):
        clips = clips.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        logits = model(clips)
        B, T, N, C = logits.shape
        labels_for_loss = prepare_labels_for_loss(labels)
        loss = criterion(
            logits.reshape(B * T * N, C),
            labels_for_loss.reshape(B * T * N)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_np = logits.argmax(dim=-1).detach().cpu().numpy().reshape(-1)
        gt_np = labels.cpu().numpy().reshape(-1)
        miou = compute_miou(pred_np, gt_np, num_classes)

        loss_meter.update(loss.item(), B)
        miou_meter.update(miou, B)

        if i % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch:[{epoch}][{i}/{len(loader)}]  '
                f'lr:{lr:.5f}  '
                f'Loss:{loss_meter.val:.4f} ({loss_meter.avg:.4f})  '
                f'mIoU:{miou_meter.val:.1f} ({miou_meter.avg:.1f})'
            )

@torch.no_grad()
def evaluate(model, loader, args, logger, num_classes):
    model.eval()

    all_pred = []
    all_gt = []

    for clips, labels in loader:
        clips = clips.cuda(non_blocking=True)
        logits = model(clips)
        preds = logits.argmax(dim=-1).cpu().numpy().reshape(-1)
        gts = labels.numpy().reshape(-1)
        all_pred.append(preds)
        all_gt.append(gts)

    all_pred = np.concatenate(all_pred)
    all_gt = np.concatenate(all_gt)
    miou = compute_miou(all_pred, all_gt, num_classes)
    return miou

if __name__ == '__main__':
    args = get_args()
    main(args)
