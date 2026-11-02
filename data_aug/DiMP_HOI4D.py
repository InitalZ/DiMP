import os
import numpy as np
from torch.utils.data import Dataset

def clip_normalize(clip: np.ndarray):

    pc = clip.reshape(-1, 3)
    centroid = pc.mean(axis=0)
    pc = pc - centroid
    m = np.max(np.linalg.norm(pc, axis=1))
    clip = (clip - centroid) / (m + 1e-8)
    return clip, m + 1e-8

def random_scale(clip: np.ndarray, lo: float = 0.9, hi: float = 1.1):

    scales = np.random.uniform(lo, hi, size=(1, 1, 3)).astype(np.float32)
    return clip * scales, scales

def random_jitter(clip: np.ndarray, sigma: float = 0.01) -> np.ndarray:

    return clip + np.random.normal(0, sigma, clip.shape).astype(np.float32)

class DiMPHOI4DSubject(Dataset):
    """HOI4D pretraining clips.

    If ``motion_root`` is given, ``{motion_root}/{name}.npz`` must hold key ``motion`` with the
    same (frames, points, 3) layout as ``data``: motion[f, i] is the offline CorrNet3D
    displacement of point i of frame f to its correspondence in frame f + step_between_frames
    (zeros where that frame does not exist). Items are then (clip, motion, vid_idx).
    """

    def __init__(self, root, meta, frames_per_clip=24, step_between_clips=2,
                 num_points=2048, step_between_frames=2, train=True,
                 scale_m_to_cm=True, motion_root=None):
        super().__init__()

        self.root = root
        self.motion_root = motion_root
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.step_between_frames = step_between_frames
        self.num_points = num_points
        self.train = train
        self.scale = 100.0 if scale_m_to_cm else 1.0

        min_frames = step_between_frames * (frames_per_clip - 1) + 1

        self.videos: list[str] = []
        self.motions: list[str] = []
        self.index_map: list[tuple[int, int]] = []
        vid_idx = 0
        missing_motion = 0

        with open(meta) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                name, nframes = parts[0], int(parts[1])
                if nframes < min_frames:
                    continue
                npz_path = os.path.join(root, name + '.npz')
                if not os.path.exists(npz_path):
                    continue
                if motion_root is not None:
                    motion_path = os.path.join(motion_root, name + '.npz')
                    if not os.path.exists(motion_path):
                        missing_motion += 1
                        continue
                    self.motions.append(motion_path)
                for t in range(0, nframes - min_frames + 1, step_between_clips):
                    self.index_map.append((vid_idx, t))
                self.videos.append(npz_path)
                vid_idx += 1

        if missing_motion:
            print(f'[DiMPHOI4DSubject] skipped {missing_motion} videos without motion file in {motion_root}')

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        vid_idx, t = self.index_map[idx]
        video = np.load(self.videos[vid_idx], allow_pickle=True)['data']

        frame_indices = [t + i * self.step_between_frames
                         for i in range(self.frames_per_clip)]
        clip = video[frame_indices]

        point_idx = self._sample_indices(clip)
        clip = np.stack([clip[i, point_idx[i]] for i in range(len(frame_indices))])

        clip = clip * self.scale
        clip, norm = clip_normalize(clip)

        if self.train:
            clip, scales = random_scale(clip)
            clip = random_jitter(clip)

        if self.motion_root is None:
            return clip.astype(np.float32), vid_idx

        # Displacements follow the same scaling as the coordinates (no translation / jitter).
        motion = np.load(self.motions[vid_idx], allow_pickle=True)['motion'][frame_indices]
        motion = np.stack([motion[i, point_idx[i]] for i in range(len(frame_indices))])
        motion = motion * self.scale / norm
        if self.train:
            motion = motion * scales
        return clip.astype(np.float32), motion.astype(np.float32), vid_idx

    def _sample_indices(self, clip: np.ndarray) -> list:

        F, N, _ = clip.shape
        result = []
        for _ in range(F):
            if N == self.num_points:
                idx = np.arange(N)
            elif N > self.num_points:
                idx = np.random.choice(N, self.num_points, replace=False)
            else:
                idx = np.concatenate([
                    np.arange(N),
                    np.random.choice(N, self.num_points - N, replace=True)
                ])
            result.append(idx)
        return result
