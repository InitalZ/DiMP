import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

import pointnet2_utils
from point_4d_convolution import P4DConv
from transformer import Transformer, VisMaskTransformer
from timm.models.layers import trunc_normal_
from extensions.chamfer_dist import ChamferDistanceL2

def compute_relative_centers(xyzs):
    # Global mean over all K tubes of the clip: removes rigid translation while keeping
    # inter-frame motion (a per-frame mean would cancel it).
    global_mean = xyzs.mean(dim=(1, 2), keepdim=True)
    return xyzs - global_mean

class CosineNoiseSchedule:
    # DDPM cosine schedule (Nichol & Dhariwal, 2021); timesteps are 1-indexed, t in {1, ..., T}.

    def __init__(self, T=2000, s=0.008, max_beta=0.999):
        self.T = T
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        self.betas = (1.0 - f[1:] / f[:-1]).clamp(max=max_beta).float()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def alpha_bar(self, t):
        return self.alphas_cumprod.to(t.device)[t - 1]

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
                 gamma_center=0.1,

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

        # phi_enc: joint (xyz, t_f) positional embedding; same module/shape as the
        # downstream models so it transfers at finetuning time.
        self.encoder_pos_linear = nn.Linear(4, en_emb_dim)
        self.encoder_transformer = VisMaskTransformer(
            en_emb_dim, en_depth, en_heads, en_head_dim, en_mlp_dim, dropout=dropout1,
        )

        if self.pretraining:

            self.noise_schedule = CosineNoiseSchedule(T=diffusion_T)

            # g_m: center predictor on masked tokens only (L_cen supervises the masked branch).
            self.center_head_mask = nn.Sequential(
                nn.Linear(en_emb_dim, en_emb_dim), nn.GELU(), nn.Linear(en_emb_dim, 3),
            )
            trunc_normal_(self.center_head_mask[0].weight, std=0.02)
            trunc_normal_(self.center_head_mask[2].weight, std=0.02)

            self.decoder_embed = nn.Linear(en_emb_dim, de_emb_dim, bias=True)
            self.decoder_pos_linear = nn.Linear(4, de_emb_dim)
            self.decoder_transformer = Transformer(
                de_emb_dim, de_depth, de_heads, de_head_dim, de_mlp_dim, dropout=dropout1,
            )
            self.decoder_norm = nn.LayerNorm(de_emb_dim)

            if self.use_patch_diffusion:
                # LinearProj(P^m_{t_c}) -> e^m_{t_c}: masked content embedding shared by encoder and decoder.
                self.patch_noise_embed = nn.Linear(3 * nsamples * temporal_kernel_size, en_emb_dim)
                self.decoder_patch_embed = nn.Linear(en_emb_dim, de_emb_dim)
                self.time_embed = TimeEmbedding(de_emb_dim)
            else:
                self.enc_mask_token = nn.Parameter(torch.zeros(1, 1, en_emb_dim))
                trunc_normal_(self.enc_mask_token, std=0.02)
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

    def _get_pe(self, C_xyz, t_f):

        t_float = t_f.float().unsqueeze(-1)
        t_max = t_float.max()
        t_norm = t_float / t_max if t_max > 0 else t_float
        return self.encoder_pos_linear(torch.cat([C_xyz, t_norm], dim=-1))

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

    @torch.no_grad()
    def _group_tube_motion(self, clips, motion, anchors):
        # Gathers the offline (CorrNet3D) per-point displacements of each tube's neighbours.
        # Ball query is repeated with P4DConv's radius / anchors / padding, so neighbour order
        # matches xyzs_neighbors. Returns (B, L, N, tk - 1, nsamples, 3): frames 1..tk-1 of each
        # tube, whose displacement to the next frame is always inside the clip.
        conv = self.tube_embedding
        pad_l, pad_r = conv.temporal_padding
        xyz_frames = [f.contiguous() for f in torch.unbind(clips, dim=1)]
        mot_frames = [f.contiguous() for f in torch.unbind(motion, dim=1)]
        zeros = torch.zeros_like(mot_frames[0])
        # Replicate-padded frames are copies of their neighbour, i.e. zero displacement.
        xyz_frames = [xyz_frames[0]] * pad_l + xyz_frames + [xyz_frames[-1]] * pad_r
        mot_frames = [zeros] * pad_l + mot_frames + [zeros] * pad_r

        half = conv.temporal_kernel_size // 2
        tubes = []
        for l, t in enumerate(range(half, len(xyz_frames) - half, conv.temporal_stride)):
            anchor = anchors[:, l].contiguous()
            per_frame = []
            for i in range(t - half, t + half):
                idx = pointnet2_utils.ball_query(conv.r, conv.k, xyz_frames[i], anchor)
                grouped = pointnet2_utils.grouping_operation(
                    mot_frames[i].transpose(1, 2).contiguous(), idx)
                per_frame.append(grouped.permute(0, 2, 3, 1))
            tubes.append(torch.stack(per_frame, dim=2))
        return torch.stack(tubes, dim=1)

    def forward_encoder(self, x, motion=None):
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
            motion_tubes = None
            if motion is not None:
                motion_tubes = self._group_tube_motion(x, motion, xyzs).reshape(batch_size, L * N, -1)

            bool_masked_pos = self.random_masking(xyzts)

            fea_vis = features[~bool_masked_pos].reshape(batch_size, -1, C)
            fea_mask = features[bool_masked_pos].reshape(batch_size, -1, C)
            C0_vis = C0_flat[~bool_masked_pos].reshape(batch_size, -1, 3)
            C0_mask = C0_flat[bool_masked_pos].reshape(batch_size, -1, 3)
            xyzts_vis = xyzts[~bool_masked_pos].reshape(batch_size, -1, 4)
            xyzts_mask = xyzts[bool_masked_pos].reshape(batch_size, -1, 4)
            t_f_vis = xyzts_vis[..., 3]
            t_f_mask = xyzts_mask[..., 3]

            # Shared timestep t_c ~ U{1, T} for masked centers and masked point coordinates.
            t_c = torch.randint(1, self.noise_schedule.T + 1, (batch_size,), device=x.device).long()

            C_vis_input = self.noise_schedule.add_noise(C0_vis, t_c) if self._should_noise_vis() else C0_vis
            C_mask_input = self.noise_schedule.add_noise(C0_mask, t_c) if self._should_noise_mask() else C0_mask

            pe_vis = self._get_pe(C_vis_input, t_f_vis)
            pe_mask = self._get_pe(C_mask_input, t_f_mask)

            # Masked tokens never see clean tube features: their content is e^m_{t_c} = LinearProj(P^m_{t_c}).
            if self.use_patch_diffusion:
                P0_mask = xyzs_neighbors[bool_masked_pos].reshape(batch_size, -1, self.tk * self.nsamples * 3)
                P_mask_tc = self.noise_schedule.add_noise(P0_mask, t_c)
                e_mask = self.patch_noise_embed(P_mask_tc)
            else:
                e_mask = self.enc_mask_token.expand(batch_size, fea_mask.shape[1], -1)

            x_vis = fea_vis + pe_vis
            x_mask = e_mask + pe_mask

            T_v, T_m = self.encoder_transformer(x_vis, x_mask)

            pred_C0_mask = self.center_head_mask(T_m)
            # L_cen = 1/K_m * sum_j ||C_hat_j - C_j||_2^2 (masked tubes only)
            center_loss = ((pred_C0_mask - C0_mask) ** 2).sum(dim=-1).mean()

            return (T_v, T_m, bool_masked_pos, xyzts, xyzs_neighbors, e_mask, motion_tubes,
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

    def forward_decoder(self, T_v, mask, xyzts, xyzs_neighbors, e_mask, motion_tubes,
                        C0_vis, C0_mask, pred_C0_mask, t_f_vis, t_f_mask, t_c):
        emb_vis = self.decoder_embed(T_v)
        batch_size, N_vis, C_dec = emb_vis.shape

        gt_points = xyzs_neighbors[mask].reshape(batch_size, -1, self.tk, self.nsamples, 3)
        N_masked = gt_points.shape[1]

        if self.use_patch_diffusion:
            # Same e^m_{t_c} (same noise sample) as the encoder's masked tokens.
            mask_tokens = self.decoder_patch_embed(e_mask)
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

        T_sched = self.noise_schedule.T
        h = self.motion_h_intervals
        d = T_sched // h
        # Motion head is conditioned on the full Z_dec (visible + masked tokens).
        cond = emb_all

        motion_loss = torch.zeros((), device=emb_all.device)
        if self.use_motion_diffusion and motion_tubes is not None:
            # M_0 in the same [visible, masked] token order as Z_dec.
            D = motion_tubes.shape[-1]
            M0 = torch.cat([motion_tubes[~mask].reshape(batch_size, -1, D),
                            motion_tubes[mask].reshape(batch_size, -1, D)], dim=1)
            for i in range(h):
                lo = d * i + 1
                hi = T_sched if i == h - 1 else d * (i + 1)
                t_i = torch.randint(lo, hi + 1, (batch_size,), device=cond.device).long()
                eps = torch.randn_like(M0)
                M_t_i = self.noise_schedule.add_noise(M0, t_i, noise=eps)
                t_emb = self.motion_time_embed(t_i)
                eps_hat = self.motion_head(M_t_i, t_emb, cond)
                motion_loss = motion_loss + F.mse_loss(eps_hat, eps)
            motion_loss = motion_loss / h

        if self.vis:
            vis_points = xyzs_neighbors[~mask].reshape(batch_size, -1, self.tk, self.nsamples, 3)
            return (pre_points.reshape(batch_size, N_masked, self.tk, self.nsamples, 3),
                    gt_points, vis_points, mask)

        return point_loss, motion_loss

    def forward(self, clips, motion=None):
        if self.pretraining:
            if self.use_motion_diffusion and motion is None and not self.vis:
                raise ValueError('Motion diffusion needs the offline CorrNet3D displacements '
                                 '(pass `motion` with the same shape as `clips`).')
            enc_out = self.forward_encoder(clips, motion)
            if not isinstance(enc_out, tuple):
                return enc_out

            (T_v, T_m, mask, xyzts, xyzs_neighbors, e_mask, motion_tubes,
             C0_vis, C0_mask, pred_C0_mask, t_f_vis, t_f_mask, t_c, center_loss) = enc_out

            dec_args = (T_v, mask, xyzts, xyzs_neighbors, e_mask, motion_tubes,
                        C0_vis, C0_mask, pred_C0_mask, t_f_vis, t_f_mask, t_c)
            if self.vis:
                return self.forward_decoder(*dec_args)

            point_loss, motion_loss = self.forward_decoder(*dec_args)

            loss = self.gamma_center * center_loss + point_loss + self.motion_weight * motion_loss
            return loss
        else:
            return self.forward_encoder(clips)

DiMPModelFull = ContrastiveLearningModelFull
