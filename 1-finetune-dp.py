import os
import sys
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'modules'))

from models.DiMP_Model_Full import DiMPModelFull
from datasets.hoi4d import HOI4DSubject
from datasets.msr import MSRAction3D
from logger import setup_logger
from utils import AverageMeter, WarmupMultiStepLR, accuracy

def get_args():
    parser = argparse.ArgumentParser('DiMP point cloud video classification finetuning')

    parser.add_argument('--dataset', default='hoi4d', type=str, help='hoi4d or msr')
    parser.add_argument('--data-path', required=True, type=str)
    parser.add_argument('--data-meta', default='', type=str, help='HOI4D train meta list')
    parser.add_argument('--val-meta', default='', type=str, help='HOI4D val meta (empty: infer from --data-meta)')
    parser.add_argument('--num-points', default=2048, type=int)
    parser.add_argument('--clip-len', default=24, type=int)
    parser.add_argument('--clip-stride', default=50, type=int)
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
    parser.add_argument('--dropout-cls', default=0.5, type=float)

    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--weight-decay', '--wd', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr-warmup-epochs', default=10, type=int)
    parser.add_argument('--lr-milestones', default=[10, 15], nargs='+', type=int)
    parser.add_argument('--lr-gamma', default=0.1, type=float)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--print-freq', default=20, type=int)

    parser.add_argument('--finetune', default='', type=str, help='Pretrained checkpoint path')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--output-dir', default='log_finetune_hoi4d/', type=str)

    parser.add_argument('--model', default='E0_cls', type=str, help='Experiment / run name')
    parser.add_argument('--log-dir', default='logs/', type=str)
    parser.add_argument('--seed', default=0, type=int)

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_pretrained_encoder(model, pretrained_path, logger):

    ckpt = torch.load(pretrained_path, map_location='cpu')
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))

    state = {k.replace('module.', ''): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    epoch = ckpt.get('epoch', -1)
    logger.info(f'===> Loading checkpoint for finetune \'{pretrained_path}\'')
    logger.info(f'===> Finetune load_state_dict: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}')
    logger.info(f'===> Loaded checkpoint with epoch {epoch}')

def main(args):
    set_seed(args.seed)
    cudnn.benchmark = True
    dataset_name = args.dataset.lower()

    log_dir = os.path.join(args.log_dir, args.model)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(output=log_dir, name=args.model)
    logger.info(args)

    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    if dataset_name == 'hoi4d':
        val_meta = args.val_meta
        if not args.data_meta:
            raise ValueError('--data-meta is required when --dataset=hoi4d')
        if not val_meta:
            val_meta = args.data_meta.replace('train', 'val')
            if not os.path.exists(val_meta):
                val_meta = args.data_meta

        train_dataset = HOI4DSubject(
            root=args.data_path,
            meta=args.data_meta,
            frames_per_clip=args.clip_len,
            step_between_clips=args.clip_stride,
            step_between_frames=args.frame_stride,
            num_points=args.num_points,
            train=True
        )
        val_dataset = HOI4DSubject(
            root=args.data_path,
            meta=val_meta,
            frames_per_clip=args.clip_len,
            step_between_clips=args.clip_stride,
            step_between_frames=args.frame_stride,
            num_points=args.num_points,
            train=False
        )
    elif dataset_name == 'msr':
        train_dataset = MSRAction3D(
            root=args.data_path,
            frames_per_clip=args.clip_len,
            step_between_clips=args.clip_stride,
            num_points=args.num_points,
            train=True
        )
        val_dataset = MSRAction3D(
            root=args.data_path,
            frames_per_clip=args.clip_len,
            step_between_clips=args.clip_stride,
            num_points=args.num_points,
            train=False
        )
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    num_classes = train_dataset.num_classes
    logger.info(f'Dataset: {dataset_name}')
    logger.info(f'num_classes={num_classes}, train={len(train_dataset)}, val={len(val_dataset)}')

    model = DiMPModelFull(
        radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
        temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
        en_emb_dim=args.en_dim, en_depth=args.en_depth, en_heads=args.en_heads,
        en_head_dim=args.en_head_dim, en_mlp_dim=args.en_mlp_dim,
        num_classes=num_classes,
        dropout1=args.dropout1,
        dropout_cls=args.dropout_cls,
        pretraining=False,
    ).cuda()

    if args.finetune:
        load_pretrained_encoder(model, args.finetune, logger)

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=args.lr_milestones,
        gamma=args.lr_gamma,
        warmup_factor=0.0,
        warmup_iters=args.lr_warmup_epochs,
        warmup_method='linear'
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        args.start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info(f'Resumed from epoch {ckpt["epoch"]}')

    best_acc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, train_loader, optimizer, epoch, args, logger)
        scheduler.step()

        test_loss, test_top1, test_top5, class_accs = evaluate(
            model, criterion, val_loader, num_classes, logger
        )
        if test_top1 > best_acc:
            best_acc = test_top1
            torch.save({'epoch': epoch, 'model': model.state_dict()},
                       os.path.join(log_dir, 'best.pth'))
        logger.info(
            f'Epoch {epoch}: Top1={test_top1:.3f}%  Top5={test_top5:.3f}%  '
            f'Best={best_acc:.3f}%'
        )

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, os.path.join(log_dir, 'checkpoint_last.pth'))

def train_one_epoch(model, criterion, loader, optimizer, epoch, args, logger):
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for i, (clips, labels, _) in enumerate(loader):
        clips = clips.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        output = model(clips)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        loss_meter.update(loss.item(), clips.size(0))
        top1_meter.update(acc1.item(), clips.size(0))
        top5_meter.update(acc5.item(), clips.size(0))

        if i % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch: [{epoch}][{i}/{len(loader)}]\t'
                f'lr: {lr:.5f}\t'
                f'Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Top1: {top1_meter.val:.3f} ({top1_meter.avg:.3f})\t'
                f'Top5: {top5_meter.val:.3f} ({top5_meter.avg:.3f})'
            )

@torch.no_grad()
def evaluate(model, criterion, loader, num_classes, logger):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    vid_scores = {}
    vid_labels = {}

    for i, (clips, labels, vid_idxs) in enumerate(loader):
        clips = clips.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        vid_idxs = vid_idxs.cuda(non_blocking=True)

        output = model(clips)
        loss = criterion(output, labels)

        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        loss_meter.update(loss.item(), clips.size(0))
        top1_meter.update(acc1.item(), clips.size(0))
        top5_meter.update(acc5.item(), clips.size(0))

        scores = torch.softmax(output, dim=1).cpu().numpy()
        for j in range(clips.size(0)):
            vid = vid_idxs[j].item()
            lbl = labels[j].item()
            if vid not in vid_scores:
                vid_scores[vid] = scores[j]
            else:
                vid_scores[vid] += scores[j]
            vid_labels[vid] = lbl

        if i % 20 == 0:
            logger.info(
                f'Test: [{i}/{len(loader)}]\t'
                f'Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Top1: {top1_meter.val:.3f} ({top1_meter.avg:.3f})\t'
                f'Top5: {top5_meter.val:.3f} ({top5_meter.avg:.3f})'
            )

    correct = sum(
        1 for vid in vid_scores
        if np.argmax(vid_scores[vid]) == vid_labels[vid]
    )
    total_acc = correct / len(vid_scores) if vid_scores else 0.0
    logger.info(f'Video-level Total-acc: {total_acc:.5f}')

    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    for vid in vid_scores:
        lbl = vid_labels[vid]
        if lbl < num_classes:
            class_total[lbl] += 1
            if np.argmax(vid_scores[vid]) == lbl:
                class_correct[lbl] += 1

    class_acc = np.array([
        class_correct[c] / float(class_total[c]) if class_total[c] > 0 else float('nan')
        for c in range(num_classes)
    ])
    mean_class_acc = np.nanmean(class_acc) * 100
    logger.info(f'Video-level Class-acc: {np.round(class_acc, 3)}')
    logger.info(f'Mean-class-acc (classes with samples): {mean_class_acc:.2f}%')

    return loss_meter.avg, top1_meter.avg, top5_meter.avg, class_acc

if __name__ == '__main__':
    args = get_args()
    main(args)
