import os
import sys
import argparse
import time
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

from models.DiMP_Model_Full import DiMPModelFull
from data_aug.DiMP_HOI4D import DiMPHOI4DSubject
from data_aug.DiMP_MSR import DiMPPretrainDataset as DiMPMSRAction3D
from logger import setup_logger
from utils import AverageMeter, WarmupCosineLR

def get_args():
    parser = argparse.ArgumentParser('DiMP self-supervised pretraining for point cloud video')

    parser.add_argument('--dataset', default='hoi4d', type=str, help='hoi4d or msr')
    parser.add_argument('--data-path', required=True, type=str, help='Root directory of NPZ files')
    parser.add_argument('--data-meta', default='', type=str, help='HOI4D meta list file')
    parser.add_argument('--num-points', default=1024, type=int,
                        help='Points per frame (paper: 1024 for HOI4D/MSR pretraining)')
    parser.add_argument('--clip-len', default=32, type=int, help='Frames per clip')
    parser.add_argument('--clip-stride', default=8, type=int, help='Clip stride')
    parser.add_argument('--frame-stride', default=2, type=int, help='Frame stride')
    parser.add_argument('--sub-clips', default=5, type=int, help='Number of sub-clips for MSR pretraining')

    parser.add_argument('--radius', default=0.05, type=float,
                        help='P4D spatial radius; use 0.1 for MSRAction3D pretraining if matching paper/scripts')
    parser.add_argument('--nsamples', default=32, type=int)
    parser.add_argument('--spatial-stride', default=32, type=int)
    parser.add_argument('--temporal-kernel-size', default=3, type=int)
    parser.add_argument('--temporal-stride', default=1, type=int)
    parser.add_argument('--en-dim', default=384, type=int)
    parser.add_argument('--en-depth', default=6, type=int)
    parser.add_argument('--en-heads', default=6, type=int)
    parser.add_argument('--en-head-dim', default=64, type=int)
    parser.add_argument('--en-mlp-dim', default=768, type=int)

    parser.add_argument('--de-dim', default=256, type=int)
    parser.add_argument('--de-depth', default=3, type=int)
    parser.add_argument('--de-heads', default=4, type=int)
    parser.add_argument('--de-head-dim', default=64, type=int)
    parser.add_argument('--de-mlp-dim', default=512, type=int)
    parser.add_argument('--mask-ratio', default=0.60, type=float,
                        help='Mask ratio (paper appendix: 0.60)')
    parser.add_argument('--dropout1', default=0.05, type=float)
    parser.add_argument('--dropout-cls', default=0.5, type=float)

    parser.add_argument('--epochs', default=200, type=int,
                        help='Training epochs (paper: 200)')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight-decay', '--wd', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr-warmup-epochs', default=10, type=int)
    parser.add_argument('--workers', default=32, type=int)
    parser.add_argument('--print-freq', default=800, type=int)

    parser.add_argument(
        '--model', default='pretrain', type=str,
        help='Run name; checkpoints are saved under log_dir/model/.'
    )
    parser.add_argument('--center-diffusion-mode', default='full', type=str,
                       choices=['base', 'vis_only', 'mask_only', 'full'],
                       help='E3/Full center diffusion mode')
    parser.add_argument('--gamma-center', default=0.1, type=float, help='Center loss weight gamma')
    parser.add_argument('--motion-weight', default=1.0, type=float, help='Motion diffusion loss weight')
    parser.add_argument('--use-patch-diffusion', action='store_true', help='Enable patch diffusion decoder')
    parser.add_argument('--use-motion-diffusion', dest='use_motion_diffusion', action='store_true',
                       help='Enable motion diffusion branch')
    parser.add_argument('--no-motion-diffusion', dest='use_motion_diffusion', action='store_false',
                       help='Disable motion diffusion branch')
    parser.set_defaults(use_motion_diffusion=True)
    parser.add_argument('--motion-h-intervals', default=4, type=int,
                       help='Stratified timestep intervals h for motion diffusion (paper Sec. 3.2)')
    parser.add_argument('--diffusion-T', default=2000, type=int, help='Number of diffusion steps T')
    parser.add_argument('--log-dir', default='logs/', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument(
        '--save-freq',
        default=0,
        type=int,
        help='Also save checkpoint_{epoch}.pth every N epochs (0 = only checkpoint_last and final epoch)'
    )

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
        if not args.data_meta:
            raise ValueError('--data-meta is required when --dataset=hoi4d')
        train_dataset = DiMPHOI4DSubject(
            root=args.data_path,
            meta=args.data_meta,
            frames_per_clip=args.clip_len,
            step_between_clips=args.clip_stride,
            step_between_frames=args.frame_stride,
            num_points=args.num_points,
            train=True
        )
    elif dataset_name == 'msr':
        train_dataset = DiMPMSRAction3D(
            root=args.data_path,
            frames_per_clip=args.clip_len,
            step_between_clips=args.clip_stride,
            num_points=args.num_points,
            sub_clips=args.sub_clips,
            train=True
        )
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f'Dataset: {dataset_name}')
    logger.info(f'Train clips: {len(train_dataset)}')

    model_kwargs = dict(
        radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
        temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
        en_emb_dim=args.en_dim, en_depth=args.en_depth, en_heads=args.en_heads,
        en_head_dim=args.en_head_dim, en_mlp_dim=args.en_mlp_dim,
        de_emb_dim=args.de_dim, de_depth=args.de_depth, de_heads=args.de_heads,
        de_head_dim=args.de_head_dim, de_mlp_dim=args.de_mlp_dim,
        mask_ratio=args.mask_ratio,
        dropout1=args.dropout1,
        dropout_cls=args.dropout_cls,
        pretraining=True,
        center_diffusion_mode=args.center_diffusion_mode,
        gamma_center=args.gamma_center,
        motion_weight=args.motion_weight,
        use_patch_diffusion=args.use_patch_diffusion,
        use_motion_diffusion=args.use_motion_diffusion,
        motion_h_intervals=args.motion_h_intervals,
        diffusion_T=args.diffusion_T,
    )
    model = DiMPModelFull(**model_kwargs).cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = WarmupCosineLR(
        optimizer,
        T_max=args.epochs,
        warmup_iters=args.lr_warmup_epochs,
        warmup_factor=0.0,
        warmup_method='linear'
    )

    if args.resume:
        if os.path.isfile(args.resume):
            ckpt = torch.load(args.resume, map_location='cpu')
            args.start_epoch = ckpt['epoch'] + 1
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            logger.info(f'Resumed from epoch {ckpt["epoch"]}')
        else:
            logger.warning(f'Resume file not found: {args.resume}')

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, train_loader, optimizer, epoch, args, logger)
        scheduler.step()

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'args': args
        }

        torch.save(ckpt, os.path.join(log_dir, 'checkpoint_last.pth'))

        if args.save_freq and (epoch % args.save_freq == 0):
            torch.save(ckpt, os.path.join(log_dir, f'checkpoint_{epoch}.pth'))

        if epoch == args.epochs - 1:
            torch.save(ckpt, os.path.join(log_dir, f'checkpoint_{epoch}.pth'))
        logger.info('====================================')

def train_one_epoch(model, loader, optimizer, epoch, args, logger):
    model.train()
    loss_meter = AverageMeter()

    for i, (clips, _) in enumerate(loader):
        clips = clips.cuda(non_blocking=True)
        if clips.dim() == 5:

            bsz, sub_clips, sub_len, num_points, channels = clips.shape
            clips = clips.view(bsz, sub_clips * sub_len, num_points, channels)

        loss = model(clips)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), clips.size(0))

        if i % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch:[{epoch}][{i}/{len(loader)}]\t'
                f'lr:{lr:.5f}\t'
                f'Loss:{loss_meter.val:.6f} ({loss_meter.avg:.6f})\t'
            )

if __name__ == '__main__':
    args = get_args()
    main(args)
