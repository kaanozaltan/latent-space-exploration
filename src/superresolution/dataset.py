import os

from torch.utils.data import Dataset
from PIL import Image


class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform

        self.lr_filenames = [filename for filename in os.listdir(lr_dir)]
        self.hr_filenames = [filename for filename in os.listdir(hr_dir)]

    def __len__(self):
        return len(self.lr_filenames)

    def __getitem__(self, idx):
        lr_img = Image.open(os.path.join(self.lr_dir, self.lr_filenames[idx]))
        hr_img = Image.open(os.path.join(self.hr_dir, self.hr_filenames[idx]))

        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)

        return lr_img, hr_img
