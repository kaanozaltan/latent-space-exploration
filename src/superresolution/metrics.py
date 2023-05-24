import math

import torch
import torchvision
import lpips


def psnr(output, target):
    mse = torch.mean((output - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / torch.sqrt(mse))


def ssim(output, target):
    ssim_value = torchvision.transforms.functional.ssim(output, target)
    return ssim_value.item()


def lpips(output, target):
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    return loss_fn_vgg(output, target)
