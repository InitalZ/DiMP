import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from point_4d_convolution import P4DConv
from transformer import Transformer, VisMaskTransformer
from timm.models.layers import trunc_normal_
from extensions.chamfer_dist import ChamferDistanceL2

def compute_relative_centers(xyzs):

    frame_mean = xyzs.mean(dim=2, keepdim=True)
    return xyzs - frame_mean

class LinearNoiseSchedule:

    def __init__(self, T=2000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def alpha_bar(self, t):
        return self.alphas_cumprod[t]

    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0, device=x_0.device)
        alpha_bar = self.alpha_bar(t).to(x_0.device)
        if alpha_bar.dim() == 0:
            alpha_bar = alpha_bar.unsqueeze(0)
        for _ in range(x_0.dim() - 1):
            alpha_bar = alpha_bar.unsqueeze(-1)
        return torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1.0 - alpha_bar) * noise

class TimeEmbedding(nn.Module):

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.freq = nn.Parameter(torch.linspace(0, math.log(max_period), dim // 2))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, ts):
        freq = torch.exp(self.freq).to(ts.device)
        emb = ts.float()[:, None] * freq[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.mlp(emb)

class MotionDiffusionHead(nn.Module):

    def __init__(self, motion_dim, cond_dim, hidden_dim, time_dim):
        super().__init__()
        self.noise_proj = nn.Linear(motion_dim, hidden_dim)
        self.cond_proj  = nn.Linear(cond_dim,   hidden_dim)
        self.time_proj  = nn.Linear(time_dim,   hidden_dim)
        self.reset_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Linear(hidden_dim, motion_dim)

    def forward(self, m_t, t_emb, cond):
        h_n = self.noise_proj(m_t)
        h_c = self.cond_proj(cond)
        h_t = self.time_proj(t_emb).unsqueeze(1).expand(-1, h_n.size(1), -1)
        gate = self.reset_gate(torch.cat([h_n, h_c, h_t], dim=-1))
        h = self.fuse(gate * h_c + (1 - gate) * h_n + h_t)
        return self.out(h)


class ContrastiveLearningModelFull(nn.Module):
    def __init__(self,
                 radius=0.1, nsamples=32, spatial_stride=32,
                 temporal_kernel_size=3, temporal_stride=1,
                 en_emb_dim=384, en_depth=6, en_heads=6,
                 en_head_dim=64, en_mlp_dim=768,
                 de_emb_dim=512, de_depth=4, de_heads=8,
                 de_head_dim=256, de_mlp_dim=1024,
                 mask_ratio=0.6, num_classes=60,
                 dropout1=0.05, dropout_cls=0.5,
                 pretraining=True, vis=False,

                 center_diffusion_mode='mask_only',
                 diffusion_T=2000,
                 gamma_center=1.0,

                 use_patch_diffusion=True,

                 use_motion_diffusion=True,
                 motion_weight=1.0,
                 motion_h_intervals=4):
        super().__init__()

        self.pretraining = pretraining
        self.vis = vis
        self.nsamples = nsamples
        self.tk = temporal_kernel_size
        self.center_diffusion_mode = center_diffusion_mode
        self.gamma_center = gamma_center
        self.use_patch_diffusion = use_patch_diffusion
        self.use_motion_diffusion = use_motion_diffusion
        self.motion_weight = motion_weight
        self.motion_h_intervals = motion_h_intervals

        self.tube_embedding = P4DConv(
            in_planes=0, mlp_planes=[en_emb_dim],
            mlp_batch_norm=[False], mlp_activation=[False],
            spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
            temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride,
            temporal_padding=[1, 0],
            operator='+', spatial_pooling='max', temporal_pooling='max',
        )

        self.encoder_transformer = VisMaskTransformer(
            en_emb_dim, en_depth, en_heads, en_head_dim, en_mlp_dim, dropout=dropout1,
        )

        if self.pretraining:

            self.noise_schedule = LinearNoiseSchedule(T=diffusion_T)

            self.center_head_vis = nn.Sequential(
                nn.Linear(en_emb_dim, en_emb_dim), nn.GELU(), nn.Linear(en_emb_dim, 3),
            )
            self.center_head_mask = nn.Sequential(
                nn.Linear(en_emb_dim, en_emb_dim), nn.GELU(), nn.Linear(en_emb_dim, 3),
            )
            trunc_normal_(self.center_head_vis[0].weight, std=0.02)
            trunc_normal_(self.center_head_vis[2].weight, std=0.02)
            trunc_normal_(self.center_head_mask[0].weight, std=0.02)
            trunc_normal_(self.center_head_mask[2].weight, std=0.02)

            self.decoder_embed = nn.Linear(en_emb_dim, de_emb_dim, bias=True)
            self.decoder_pos_linear = nn.Linear(4, de_emb_dim)
            self.decoder_transformer = Transformer(
                de_emb_dim, de_depth, de_heads, de_head_dim, de_mlp_dim, dropout=dropout1,
            )
            self.decoder_norm = nn.LayerNorm(de_emb_dim)

            if self.use_patch_diffusion:
                self.patch_noise_embed = nn.Conv1d(
                    3 * nsamples * temporal_kernel_size, de_emb_dim, 1,
                )
                self.time_embed = TimeEmbedding(de_emb_dim)
            else:
                self.mask_token = nn.Parameter(torch.zeros(1, 1, de_emb_dim))
                trunc_normal_(self.mask_token, std=0.02)

            self.points_predictor = nn.Conv1d(de_emb_dim, 3 * nsamples * temporal_kernel_size, 1)
            self.motion_time_embed = TimeEmbedding(de_emb_dim)
            motion_dim = 3 * nsamples * (temporal_kernel_size - 1)
            self.motion_head = MotionDiffusionHead(
                motion_dim=motion_dim,
                cond_dim=de_emb_dim,
                hidden_dim=de_emb_dim,
                time_dim=de_emb_dim,
            )
            self.criterion_dist = ChamferDistanceL2().cuda()
            self.mask_ratio = mask_ratio
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, en_emb_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, en_emb_dim))
            trunc_normal_(self.cls_token, std=0.02)
            trunc_normal_(self.cls_pos, std=0.02)
            self.mlp_head = nn.Sequential(
                nn.Linear(en_emb_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_cls),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_cls),
                nn.Linear(256, num_classes),
            )

    def _norm_t(self, xyzts_flat):
        t = xyzts_flat[..., 3:].clone()
        if t.numel() > 0:
            t_max = torch.max(t)
            if t_max > 0:
                t = t / t_max
        return torch.cat([xyzts_flat[..., :3], t], dim=-1)

    def _get_pe(self, C_xyz, t_f, t_c=None, noisy=False):

        if noisy and t_c is not None:
            alpha_bar = self.noise_schedule.alpha_bar(t_c).to(C_xyz.device)
            if alpha_bar.dim() == 0:
                alpha_bar = alpha_bar.unsqueeze(0)
            for _ in range(C_xyz.dim() - 1):
                alpha_bar = alpha_bar.unsqueeze(-1)
            C_norm = C_xyz / torch.sqrt(1.0 - alpha_bar).clamp(min=1e-8)
        else:
            C_norm = C_xyz

        t_float = t_f.float().unsqueeze(-1)
        t_max = t_float.max()
        t_norm = t_float / t_max if t_max > 0 else t_float
        return self.encoder_pos_linear(torch.cat([C_norm, t_norm], dim=-1))

    def _get_decoder_pe(self, C_xyz, t_f):

        t_float = t_f.float().unsqueeze(-1)
        t_max = t_float.max()
        t_norm = t_float / t_max if t_max > 0 else t_float
        return self.decoder_pos_linear(torch.cat([C_xyz, t_norm], dim=-1))

    def _should_noise_vis(self):
        return self.center_diffusion_mode in ('vis_only', 'full')

    def _should_noise_mask(self):
        return self.center_diffusion_mode in ('mask_only', 'full')

    def random_masking(self, x):
        import numpy as np
        B, G, _ = x.shape
        if self.mask_ratio == 0:
            return torch.zeros(x.shape[:2]).bool()
        num_mask = int(self.mask_ratio * G)
        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([np.zeros(G - num_mask), np.ones(num_mask)])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        return torch.from_numpy(overall_mask).to(torch.bool).to(x.device)

    def forward_encoder(self, x):
        xyzs, features, xyzs_neighbors, _ = self.tube_embedding(x)
        features = features.permute(0, 1, 3, 2)
        batch_size, L, N, C = features.shape

        C0 = compute_relative_centers(xyzs)

        xyzts = []
        xyzs_split = torch.split(xyzs, 1, dim=1)
        xyzs_split = [torch.squeeze(xx, dim=1).contiguous() for xx in xyzs_split]
        for t_idx, xyz in enumerate(xyzs_split):
            t = torch.ones((batch_size, N, 1), dtype=torch.float32, device=x.device) * (t_idx + 1)
            xyzts.append(torch.cat((xyz, t), dim=2))
        xyzts = torch.stack(xyzts, dim=1)

        xyzts = xyzts.reshape(batch_size, L * N, 4)
        C0_flat = C0.reshape(batch_size, L * N, 3)
        features = features.reshape(batch_size, L * N, C)

        if self.pretraining:
            xyzs_neighbors = xyzs_neighbors.reshape(batch_size, L * N, self.tk, self.nsamples, 3)

            bool_masked_pos = self.random_masking(xyzts)

            fea_vis = features[~bool_masked_pos].reshape(batch_size, -1, C)
            fea_mask = features[bool_masked_pos].reshape(batch_size, -1, C)
            C0_vis = C0_flat[~bool_masked_pos].reshape(batch_size, -1, 3)
            C0_mask = C0_flat[bool_masked_pos].reshape(batch_size, -1, 3)
            xyzts_vis = xyzts[~bool_masked_pos].reshape(batch_size, -1, 4)
            xyzts_mask = xyzts[bool_masked_pos].reshape(batch_size, -1, 4)
            t_f_vis = xyzts_vis[..., 3]
            t_f_mask = xyzts_mask[..., 3]

            t_c = torch.randint(1, self.noise_schedule.T, (batch_size,), device=x.device).long()

            C_vis_input = self.noise_schedule.add_noise(C0_vis, t_c) if self._should_noise_vis() else C0_vis
            C_mask_input = self.noise_schedule.add_noise(C0_mask, t_c) if self._should_noise_mask() else C0_mask

            pe_vis = self._get_pe(C_vis_input, t_f_vis, t_c, noisy=self._should_noise_vis())
            pe_mask = self._get_pe(C_mask_input, t_f_mask, t_c, noisy=self._should_noise_mask())

            x_vis = fea_vis + pe_vis
            x_mask = fea_mask + pe_mask

            T_v, T_m = self.encoder_transformer(x_vis, x_mask)

            pred_C0_vis = self.center_head_vis(T_v)
            pred_C0_mask = self.center_head_mask(T_m)
            center_loss = 0.5 * (F.mse_loss(pred_C0_vis, C0_vis) + F.mse_loss(pred_C0_mask, C0_mask))

            return (T_v, T_m, bool_masked_pos, xyzts, xyzs_neighbors,
                    C0_vis, C0_mask, pred_C0_mask, t_f_vis, t_f_mask, t_c, center_loss)
        else:
            pos_input = self._norm_t(xyzts)
            pos_emb = self.encoder_pos_linear(pos_input)

            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            cls_pos = self.cls_pos.expand(batch_size, -1, -1)
            features = torch.cat((cls_tokens, features), dim=1)
            pos_emb = torch.cat((cls_pos, pos_emb), dim=1)

            x_all = features + pos_emb
            x_empty = torch.zeros(batch_size, 0, C, device=x.device)
            T_v, _ = self.encoder_transformer(x_all, x_empty)

            concat_f = torch.cat([T_v[:, 0], T_v[:, 1:].max(1)[0]], dim=-1)
            return self.mlp_head(concat_f)

    def forward_decoder(self, T_v, mask, xyzts, xyzs_neighbors,
                        C0_vis, C0_mask, pred_C0_mask, t_f_vis, t_f_mask, t_c):
        emb_vis = self.decoder_embed(T_v)
        batch_size, N_vis, C_dec = emb_vis.shape

        gt_points = xyzs_neighbors[mask].reshape(batch_size, -1, self.tk, self.nsamples, 3)
        N_masked = gt_points.shape[1]

        if self.use_patch_diffusion:
            gt_flat = gt_points.reshape(batch_size, N_masked, -1)
            noisy_patches = self.noise_schedule.add_noise(gt_flat, t_c)
            mask_tokens = self.patch_noise_embed(
                noisy_patches.transpose(1, 2)
            ).transpose(1, 2)
        else:
            mask_tokens = self.mask_token.expand(batch_size, N_masked, -1)

        pe_vis_dec = self._get_decoder_pe(C0_vis, t_f_vis)
        pe_mask_dec = self._get_decoder_pe(pred_C0_mask.detach(), t_f_mask)

        emb_all = torch.cat([emb_vis, mask_tokens], dim=1)
        pos_all = torch.cat([pe_vis_dec, pe_mask_dec], dim=1)

        if self.use_patch_diffusion:
            t_emb = self.time_embed(t_c).unsqueeze(1).expand(-1, N_vis + N_masked, -1)
            emb_all = emb_all + pos_all + t_emb
        else:
            emb_all = emb_all + pos_all

        emb_all = self.decoder_transformer(emb_all)
        emb_all = self.decoder_norm(emb_all)

        masked_emb = emb_all[:, -N_masked:, :].transpose(1, 2)

        pre_points = self.points_predictor(masked_emb).transpose(1, 2)
        pre_points = pre_points.reshape(batch_size * N_masked, self.tk, self.nsamples, 3)
        pred_list = [torch.squeeze(x, dim=1).contiguous()
                     for x in torch.split(pre_points, 1, dim=1)]

        gt_points_r = gt_points.reshape(batch_size * N_masked, self.tk, self.nsamples, 3)
        gt_list = [torch.squeeze(x, dim=1).contiguous()
                   for x in torch.split(gt_points_r, 1, dim=1)]

        point_loss = sum(self.criterion_dist(pred_list[i], gt_list[i]) for i in range(self.tk)) / self.tk

        delta_gt = gt_points[:, :, 1:, :, :] - gt_points[:, :, :-1, :, :]
        delta_gt_flat = delta_gt.reshape(batch_size, N_masked, -1)

        T_sched = self.noise_schedule.T
        h = self.motion_h_intervals
        d = T_sched // h
        cond = emb_all[:, -N_masked:, :]

        motion_loss = torch.zeros((), device=delta_gt_flat.device)
        if self.use_motion_diffusion:
            for i in range(h):
                lo = d * i + 1
                hi = T_sched - 1 if i == h - 1 else d * (i + 1)
                t_i = torch.randint(lo, hi + 1, (batch_size,), device=cond.device).long()
                eps = torch.randn_like(delta_gt_flat)
                M_t_i = self.noise_schedule.add_noise(delta_gt_flat, t_i, noise=eps)
                t_emb = self.motion_time_embed(t_i)
                eps_hat = self.motion_head(M_t_i, t_emb, cond)
                motion_loss = motion_loss + F.mse_loss(eps_hat, eps)
            motion_loss = motion_loss / h

        if self.vis:
            vis_points = xyzs_neighbors[~mask].reshape(batch_size, -1, self.tk, self.nsamples, 3)
            return (pre_points.reshape(batch_size, N_masked, self.tk, self.nsamples, 3),
                    gt_points, vis_points, mask)

        return point_loss, motion_loss

    def forward(self, clips):
        if self.pretraining:
            enc_out = self.forward_encoder(clips)
            if not isinstance(enc_out, tuple):
                return enc_out

            (T_v, T_m, mask, xyzts, xyzs_neighbors,
             C0_vis, C0_mask, pred_C0_mask, t_f_vis, t_f_mask, t_c, center_loss) = enc_out

            if self.vis:
                return self.forward_decoder(
                    T_v, mask, xyzts, xyzs_neighbors,
                    C0_vis, C0_mask, pred_C0_mask, t_f_vis, t_f_mask, t_c)

            point_loss, motion_loss = self.forward_decoder(
                T_v, mask, xyzts, xyzs_neighbors,
                C0_vis, C0_mask, pred_C0_mask, t_f_vis, t_f_mask, t_c)

            loss = self.gamma_center * center_loss + point_loss + self.motion_weight * motion_loss
            return loss
        else:
            return self.forward_encoder(clips)

DiMPModelFull = ContrastiveLearningModelFull
