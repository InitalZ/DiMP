import os
import numpy as np
from torch.utils.data import Dataset

def clip_normalize(clip: np.ndarray) -> np.ndarray:

    pc = clip.reshape(-1, 3)
    centroid = pc.mean(axis=0)
    m = np.max(np.linalg.norm(pc - centroid, axis=1))
    clip = (clip - centroid) / (m + 1e-8)
    return clip

def random_scale(clip: np.ndarray, lo: float = 0.9, hi: float = 1.1) -> np.ndarray:
    scales = np.random.uniform(lo, hi, size=(1, 1, 3)).astype(np.float32)
    return clip * scales

def random_jitter(clip: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    return clip + np.random.normal(0, sigma, clip.shape).astype(np.float32)

class HOI4DSubject(Dataset):

    def __init__(self, root, meta, frames_per_clip=23, step_between_clips=2,
                 num_points=2048, step_between_frames=2, train=True,
                 scale_m_to_cm=True):
        super().__init__()

        self.root = root
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.step_between_frames = step_between_frames
        self.num_points = num_points
        self.train = train
        self.scale = 100.0 if scale_m_to_cm else 1.0

        min_frames = step_between_frames * (frames_per_clip - 1) + 1

        self.videos: list[str] = []
        self.labels: list[int] = []
        self.index_map: list[tuple[int, int]] = []
        vid_idx = 0

        missing, short = 0, 0
        with open(meta) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                name = parts[0]
                nframes = int(parts[1])
                label = int(parts[2]) if len(parts) >= 3 else 0
                if nframes < min_frames:
                    short += 1
                    continue
                npz_path = os.path.join(root, name + '.npz')
                if not os.path.exists(npz_path):
                    missing += 1
                    continue
                for t in range(0, nframes - min_frames + 1, step_between_clips):
                    self.index_map.append((vid_idx, t))
                self.videos.append(npz_path)
                self.labels.append(label)
                vid_idx += 1

        if missing:
            print(f'[HOI4DSubject] Warning: {missing} npz files missing.')
        if short:
            print(f'[HOI4DSubject] Warning: {short} sequences too short (< {min_frames} frames).')

        if not self.labels:
            raise RuntimeError(f'HOI4DSubject: No valid sequences loaded from {meta}')

        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        vid_idx, t = self.index_map[idx]
        video = np.load(self.videos[vid_idx], allow_pickle=True)['data']
        label = self.labels[vid_idx]

        frame_indices = [t + i * self.step_between_frames
                         for i in range(self.frames_per_clip)]
        clip = video[frame_indices]

        clip = self._sample_points(clip)

        clip = clip * self.scale
        clip = clip_normalize(clip)

        if self.train:
            clip = random_scale(clip)
            clip = random_jitter(clip)

        return clip.astype(np.float32), label, vid_idx

    def _sample_points(self, clip: np.ndarray) -> np.ndarray:
        F, N, _ = clip.shape
        if N == self.num_points:
            return clip
        result = np.empty((F, self.num_points, 3), dtype=clip.dtype)
        for i in range(F):
            if N >= self.num_points:
                idx = np.random.choice(N, self.num_points, replace=False)
            else:
                idx = np.concatenate([
                    np.arange(N),
                    np.random.choice(N, self.num_points - N, replace=True)
                ])
            result[i] = clip[i, idx]
        return result
