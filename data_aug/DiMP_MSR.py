import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.msr import clip_normalize

VIDEO_RE = re.compile(r"^a(?P<action>\d+)_s(?P<subject>\d+)_e(?P<trial>\d+)\.npz$")

def parse_video_name(video_name):
    match = VIDEO_RE.match(video_name)
    if not match:
        return None
    action = int(match.group("action"))
    subject = int(match.group("subject"))
    trial = int(match.group("trial"))
    return action, subject, trial

class DiMPPretrainDataset(Dataset):
    def __init__(self, root, frames_per_clip=16, step_between_clips=1, num_points=2048,
                 sub_clips=5, train=True, split='cross_subject', train_subjects=(1, 3, 5, 7, 9)):
        super(DiMPPretrainDataset, self).__init__()

        if split != 'cross_subject':
            raise ValueError(f'Unsupported MSRAction3D split: {split}')
        if frames_per_clip % sub_clips != 0:
            raise ValueError(
                f'frames_per_clip ({frames_per_clip}) must be divisible by sub_clips ({sub_clips})'
            )

        self.sub_clips = sub_clips
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.num_points = num_points
        self.train = train
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
            scales = np.random.uniform(0.9, 1.1, size=(1, 1, 3)).astype(np.float32)
            clip = clip * scales
            clip = clip + np.random.normal(0.0, 0.005, clip.shape).astype(np.float32)

        clip = clip_normalize(clip).astype(np.float32)

        clips = np.split(clip, indices_or_sections=self.sub_clips, axis=0)
        clips = torch.tensor(np.array(clips))

        return clips, index

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python data_aug/DiMP_MSR.py <msr_npz_root>')
        sys.exit(0)
    np.random.seed(0)
    dataset = DiMPPretrainDataset(root=sys.argv[1], train=True)
    clips, video_index = dataset[0]
    print('clips.shape:', tuple(clips.shape), clips.dtype)
    print('len(dataset):', len(dataset))
