import os
import re

import numpy as np
from torch.utils.data import Dataset

VIDEO_RE = re.compile(r"^a(?P<action>\d+)_s(?P<subject>\d+)_e(?P<trial>\d+)\.npz$")

def parse_video_name(video_name):
    match = VIDEO_RE.match(video_name)
    if not match:
        return None
    action = int(match.group("action"))
    subject = int(match.group("subject"))
    trial = int(match.group("trial"))
    return action, subject, trial

def clip_normalize(clip: np.ndarray) -> np.ndarray:
    pc = clip.reshape(-1, 3)
    centroid = pc.mean(axis=0)
    m = np.max(np.linalg.norm(pc - centroid, axis=1))
    return (clip - centroid) / (m + 1e-8)

def random_rotate_y(clip: np.ndarray, max_deg: float = 15.0) -> np.ndarray:
    theta = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    return clip @ rotation.T

def random_jitter(clip: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    return clip + np.random.normal(0.0, sigma, clip.shape).astype(np.float32)

class MSRAction3D(Dataset):
    def __init__(self, root, frames_per_clip=16, step_between_clips=1, num_points=2048,
                 train=True, split='cross_subject', train_subjects=(1, 3, 5, 7, 9)):
        super(MSRAction3D, self).__init__()

        if split != 'cross_subject':
            raise ValueError(f'Unsupported MSRAction3D split: {split}')

        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.num_points = num_points
        self.train = train
        self.split = split
        self.train_subjects = set(train_subjects)
        self.videos = []
        self.labels = []
        self.index_map = []

        index = 0
        for video_name in sorted(os.listdir(root)):
            parsed = parse_video_name(video_name)
            if parsed is None:
                continue

            action, subject, _ = parsed
            is_train_subject = subject in self.train_subjects
            if train != is_train_subject:
                continue

            video_path = os.path.join(root, video_name)
            video = np.load(video_path, allow_pickle=True)['point_clouds']
            nframes = video.shape[0]
            if nframes < step_between_clips * (frames_per_clip - 1) + 1:
                continue

            self.videos.append(video_path)
            self.labels.append(action - 1)
            for t in range(0, nframes - step_between_clips * (frames_per_clip - 1), step_between_clips):
                self.index_map.append((index, t))
            index += 1

        if not self.labels:
            raise RuntimeError(f'No valid MSRAction3D sequences found in {root} for train={train}')

        self.num_classes = 20

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = np.load(self.videos[index], allow_pickle=True)['point_clouds']
        label = self.labels[index]

        clip = [video[t + i * self.step_between_clips] for i in range(self.frames_per_clip)]
        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip, dtype=np.float32)

        if self.train:
            clip = random_rotate_y(clip, max_deg=15.0)
            scales = np.random.uniform(0.9, 1.1, size=(1, 1, 3)).astype(np.float32)
            clip = clip * scales
            clip = random_jitter(clip, sigma=0.005)

        clip = clip_normalize(clip)

        return clip.astype(np.float32), label, index

if __name__ == '__main__':
    np.random.seed(0)
    dataset = MSRAction3D(root='/ssd2/szq/MSRAction/processed_data', frames_per_clip=16)
    clip, label, video_idx = dataset[0]
    print(clip)
    print('clip.shape:', clip.shape)
    print(len(dataset))
    print(label)
    print(dataset.num_classes)
