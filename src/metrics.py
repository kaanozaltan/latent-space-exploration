import math
import os

import torch
import lpips
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from PIL import Image


def get_psnr(output, target):
    max_pixel = 1.0
    mse = torch.mean((output-target)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_pixel / torch.sqrt(mse))


def get_ssim(output, target):
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    return ssim(output, target, channel_axis=0, data_range=1)


def get_lpips(output, target):
    output = output * 2 - 1
    target = target * 2 - 1
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    return loss_fn_vgg(output, target).item()


def calculate_metrics(output_dir, target_dir):
    transform = transforms.ToTensor()
    psnr_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    num_images = 0

    for filename in os.listdir(output_dir):
        output_path = os.path.join(output_dir, filename)
        target_path = os.path.join(target_dir, filename)

        output = transform(Image.open(output_path))
        target = transform(Image.open(target_path))

        psnr_total += get_psnr(output, target)
        ssim_total += get_ssim(output, target)
        lpips_total += get_lpips(output, target)
        num_images += 1

    avg_psnr = psnr_total / num_images
    avg_ssim = ssim_total / num_images
    avg_lpips = lpips_total / num_images
    return avg_psnr, avg_ssim, avg_lpips


avg_psnr, avg_ssim, avg_lpips = calculate_metrics('../outputs', '../targets')
print("Average PSNR:", avg_psnr)
print("Average SSIM:", avg_ssim)
print("Average LPIPS:", avg_lpips)
