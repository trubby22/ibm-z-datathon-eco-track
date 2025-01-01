import os

import imageio
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import glob
import pathlib
from tqdm import tqdm

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        path = pathlib.Path(img_dir)
        self.paths = list(path.glob("**/*.jpg"))
        print(img_dir, len(self.paths))
        self.transform = transform
        self.target_transform = target_transform
        # self.images = []
        # for i in tqdm(range(len(self.paths))):
        #     img_path = str(self.paths[i])
        #     image = read_image(img_path)
        #     self.images.append(transform(image))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = str(self.paths[idx])
        image = read_image(img_path)
        # image = imageio.imread(img_path)
        # image = self.images[idx]
        label = 1 if "fire" in img_path.split("/") else 0
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label