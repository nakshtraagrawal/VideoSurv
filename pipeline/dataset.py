"""
dataset.py
Shared dataset for all three models.
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class DroneDataset(Dataset):
    """
    Args:
        scene_dir: path to scene (e.g. data/drone/bike)
        split: 'training' or 'testing'
        clip_length: frames per sample. Set 1 for AE/ASTNet, 5 for HSTforU
        img_size: resize to this square size
        stride: step between clips (1 = dense, clip_length = non-overlapping)
    """

    def __init__(
        self,
        scene_dir,
        split="training",
        clip_length=5,
        img_size=256,
        stride=1,
    ):
        self.clip_length = clip_length
        self.stride = stride
        self.transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.Grayscale(1),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )
        self.clips = []
        frames_root = os.path.join(scene_dir, split, "frames")
        if not os.path.isdir(frames_root):
            return
        for vid in sorted(os.listdir(frames_root)):
            vid_path = os.path.join(frames_root, vid)
            if not os.path.isdir(vid_path):
                continue
            frames = sorted(
                [f for f in os.listdir(vid_path) if f.endswith(".jpg")]
            )
            if len(frames) < clip_length:
                continue
            for start in range(0, len(frames) - clip_length + 1, stride):
                self.clips.append((vid_path, frames, start))

        self.annotations = {}
        if split == "testing":
            ann_dir = os.path.join(scene_dir, "annotation")
            if os.path.exists(ann_dir):
                for f in os.listdir(ann_dir):
                    if f.endswith(".npy"):
                        self.annotations[f[:-4]] = np.load(
                            os.path.join(ann_dir, f)
                        )

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        vid_path, frames, start = self.clips[idx]
        clip = []
        for i in range(self.clip_length):
            img = Image.open(
                os.path.join(vid_path, frames[start + i])
            ).convert("RGB")
            clip.append(self.transform(img))
        clip = torch.stack(clip, dim=0)
        return clip


class DroneTestDataset(DroneDataset):
    """Same as DroneDataset but also returns video name, target index, and label."""

    def __getitem__(self, idx):
        vid_path, frames, start = self.clips[idx]
        clip = []
        for i in range(self.clip_length):
            img = Image.open(
                os.path.join(vid_path, frames[start + i])
            ).convert("RGB")
            clip.append(self.transform(img))
        clip = torch.stack(clip, dim=0)
        vid_name = os.path.basename(vid_path)
        target_idx = start + self.clip_length - 1
        label = 0
        if vid_name in self.annotations:
            ann = self.annotations[vid_name]
            if target_idx < len(ann):
                label = int(ann[target_idx])
        return clip, vid_name, target_idx, label
