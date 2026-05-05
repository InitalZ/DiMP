import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from point_4d_convolution import P4DConv
from transformer import Transformer
from timm.models.layers import trunc_normal_

class ActionSegModel(nn.Module):
    def __init__(self,
                 radius=0.05, nsamples=32, spatial_stride=32,
                 temporal_kernel_size=3, temporal_stride=1,
                 en_emb_dim=384, en_depth=6, en_heads=6,
                 en_head_dim=64, en_mlp_dim=768,
                 clip_len=32,
                 num_classes=19,
                 dropout1=0.05,
                 dropout_seg=0.3):
        super().__init__()

        self.clip_len = clip_len

        self.tube_embedding = P4DConv(
            in_planes=0, mlp_planes=[en_emb_dim],
            mlp_batch_norm=[False], mlp_activation=[False],
            spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
            temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride,
            temporal_padding=[1, 0],
            operator='+', spatial_pooling='max', temporal_pooling='max'
        )

        self.encoder_pos_embed = nn.Conv1d(
            in_channels=4, out_channels=en_emb_dim, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.encoder_transformer = Transformer(
            en_emb_dim, en_depth, en_heads, en_head_dim, en_mlp_dim, dropout=dropout1
        )
        self.encoder_norm = nn.LayerNorm(en_emb_dim)

        self.seg_head = nn.Sequential(
            nn.Linear(en_emb_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_seg),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_seg),
            nn.Linear(128, num_classes)
        )

    def load_pretrained(self, pretrained_path):
        ckpt = torch.load(pretrained_path, map_location='cpu')
        state = ckpt.get('model', ckpt.get('state_dict', ckpt))

        state = {k.replace('module.', ''): v for k, v in state.items()}
        msg = self.load_state_dict(state, strict=False)
        print(f'[ActionSegModel] missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}')

    def forward(self, clips):

        xyzs, features, _, _ = self.tube_embedding(clips)

        features = features.permute(0, 1, 3, 2)
        B, L, N_s, C = features.shape

        xyzts = []
        xyzs_list = torch.split(xyzs, 1, dim=1)
        for t_idx, xyz in enumerate(xyzs_list):
            xyz = xyz.squeeze(1)
            t_val = torch.full((B, N_s, 1), t_idx + 1,
                               dtype=torch.float32, device=clips.device)
            xyzts.append(torch.cat([xyz, t_val], dim=-1))
        xyzts = torch.stack(xyzts, dim=1)

        xyzts_flat = xyzts.reshape(B, L * N_s, 4)
        feats_flat = features.reshape(B, L * N_s, C)

        pos_emb = self.encoder_pos_embed(
            xyzts_flat.permute(0, 2, 1)
        ).permute(0, 2, 1)

        x = feats_flat + pos_emb
        x = self.encoder_transformer(x)
        x = self.encoder_norm(x)

        x = x.reshape(B, L, N_s, C).mean(dim=2)

        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=self.clip_len, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)

        logits = self.seg_head(x)
        return logits
