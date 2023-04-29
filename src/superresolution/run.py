# Modified from https://github.com/adamian98/pulse

from math import log10, ceil

from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from pathlib import Path
from PIL import Image
import torchvision

from models import PULSE


class Images(Dataset):
    def __init__(self, root_dir, duplicates):
        self.root_path = Path(root_dir)
        self.image_list = list(self.root_path.glob("*.jpg"))
        
        # Number of times to duplicate the image in the dataset to produce multiple HR images
        self.duplicates = duplicates

    def __len__(self):
        return self.duplicates*len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx//self.duplicates]
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        if self.duplicates == 1:
            return image,img_path.stem
        else:
            return image,img_path.stem+f"_{(idx % self.duplicates)+1}"


duplicates = 1
batch_size = 1
save_intermediate = True

dataset = Images('../inputs', duplicates=duplicates)
out_path = Path('../outputs')
out_path.mkdir(parents=True, exist_ok=True)

dataloader = DataLoader(dataset, batch_size=batch_size)

model = PULSE()
model = DataParallel(model)

toPIL = torchvision.transforms.ToPILImage()

for ref_im, ref_im_name in dataloader:
    if save_intermediate:
        padding = ceil(log10(100))
        for i in range(batch_size):
            int_path_HR = Path(out_path / ref_im_name[i] / "HR")
            int_path_LR = Path(out_path / ref_im_name[i] / "LR")
            int_path_HR.mkdir(parents=True, exist_ok=True)
            int_path_LR.mkdir(parents=True, exist_ok=True)
        for j,(HR,LR) in enumerate(model(ref_im)):
            for i in range(batch_size):
                toPIL(HR[i].cpu().detach().clamp(0, 1)).save(
                    int_path_HR / f"{ref_im_name[i]}_{j:0{padding}}.png")
                toPIL(LR[i].cpu().detach().clamp(0, 1)).save(
                    int_path_LR / f"{ref_im_name[i]}_{j:0{padding}}.png")
    else:
        for j,(HR,LR) in enumerate(model(ref_im)):
            for i in range(batch_size):
                toPIL(HR[i].cpu().detach().clamp(0, 1)).save(
                    out_path / f"{ref_im_name[i]}.png")
