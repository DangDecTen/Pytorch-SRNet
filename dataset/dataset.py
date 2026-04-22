"""This module provide the data sample for training."""

import os
import torch
from torch.utils.data import Dataset
import imageio as io

class DatasetLoad(Dataset):
    def __init__(self, cover_path, stego_path, transform=None):
        self.cover_path = cover_path
        self.stego_path = stego_path
        self.transform = transform

        # Get file list safely
        self.files = sorted(os.listdir(cover_path))

        # Optional: ensure matching files
        stego_files = set(os.listdir(stego_path))
        self.files = [f for f in self.files if f in stego_files]

    def __len__(self):
        return len(self.files) * 2   # Include both cover and stego images

    def __getitem__(self, index):
        filename = self.files[index]

        cover_img = io.imread(os.path.join(self.cover_path, filename))
        stego_img = io.imread(os.path.join(self.stego_path, filename))

        if self.transform:
            cover_img = self.transform(cover_img)
            stego_img = self.transform(stego_img)

        # Labels (stay on CPU)
        label_cover = torch.tensor(0, dtype=torch.long)
        label_stego = torch.tensor(1, dtype=torch.long)

        return {
            "cover": cover_img,
            "stego": stego_img,
            "label": [label_cover, label_stego],
        }