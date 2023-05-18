import os 

import torch
from torch import nn
import numpy as np

from stylegan import G_synthesis


synthesis = G_synthesis().cuda()

with open('../models/synthesis.pt', 'rb') as f:
    synthesis.load_state_dict(torch.load(f))

for param in synthesis.parameters():
    param.requires_grad = False

noise = []
for i in range(18):
    res = (32, 1, 2**(i//2+2), 2**(i//2+2))
    new_noise = torch.randn(res, dtype=torch.float, device='cuda')
    new_noise.requires_grad = False
    noise.append(new_noise)
