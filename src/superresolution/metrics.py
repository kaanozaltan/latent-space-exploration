import math
import os

import torch
import lpips
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from PIL import Image


def get_psnr(output, target):
    max_pixel = 1.0
    mse = torch.mean((output - target) ** 2)
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


transform = transforms.ToTensor()
# input_files = os.listdir('../../inputs')
# output_files = os.listdir('../../outputs')
# psnr_sum, ssim_sum, lpips_sum = 0, 0, 0

# for input_file, output_file in zip(input_files, output_files):
#     input_img = transform(Image.open(input_file))
#     output_img = transform(Image.open(output_file))

#     psnr_value = get_psnr(output_img, input_img)
#     ssim_value = get_ssim(output_img, input_img)
#     lpips_value = get_lpips(output_img, input_img)

#     psnr_sum += psnr_value
#     ssim_sum += ssim_value
#     lpips_sum += lpips_value

#     print(f"Metrics for {output_file}:")
#     print("PSNR:", psnr_value)
#     print("SSIM:", ssim_value)
#     print("LPIPS:", lpips_value)
#     print()

# print("Average PSNR:", psnr_sum / len(input_files))
# print("Average SSIM:", ssim_sum / len(input_files))
# print("Average LPIPS:", lpips_sum / len(input_files))

path1 = '../../inputs/272.jpg'
path2 = '../../outputs/272.jpg'
img1 = transform(Image.open(path1))
img2 = transform(Image.open(path2))
print("img range:", torch.min(img1), torch.max(img1))
print("img shape:", img1.shape)
print("PSNR:", get_psnr(img1, img2))
print("SSIM:", get_ssim(img1, img2))
print("LPIPS:", get_lpips(img1, img2))
