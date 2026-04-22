"""This module provide the data sample for training."""

import os
import random
import numpy as np
import imageio as io

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

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
            # The SAME random transform is applied to both images in a pair
            seed = np.random.randint(0, 10000)

            random.seed(seed)
            cover_img = self.transform(cover_img)

            random.seed(seed)
            stego_img = self.transform(stego_img)
            
        # Labels (stay on CPU)
        label_cover = torch.tensor(0, dtype=torch.long)
        label_stego = torch.tensor(1, dtype=torch.long)

        return {
            "cover": cover_img,
            "stego": stego_img,
            "label": [label_cover, label_stego],
        }

class RandomRotate90:
    def __call__(self, x):
        k = random.randint(0, 3)
        return TF.rotate(x, 90 * k)

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),  # random mirroring
    RandomRotate90(),
    transforms.ToTensor(),
])