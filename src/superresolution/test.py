import numpy as np
import torch
from PIL import Image
from torchvision import transforms

transform = transforms.ToTensor()
path = '../../inputs_original/272.jpg'
input_img = transform(Image.open(path))
print("img range:", torch.min(input_img), torch.max(input_img))
print("img shape:", input_img.shape)
