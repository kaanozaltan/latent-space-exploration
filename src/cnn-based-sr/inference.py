import os

import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from models import SRCNN, VDSR, EDSR
import metrics


def test(model):
    model_name = model.get_name()
    print(f"Performing inference for {model_name.upper()} model")
    model.load_state_dict(torch.load(f'../models/{model_name}.pt'))
    model.eval()

    if not os.path.exists('../../dataset/inference/outputs'):
        os.makedirs('../../dataset/inference/outputs')

    psnr_total = 0.0
    num_images = 0

    for filename in os.listdir('../../dataset/inference/lr'):
        input_img = Image.open(os.path.join('../../dataset/inference/lr', filename).replace('\\', '/'))
        target_img = Image.open(os.path.join('../../dataset/inference/hr', filename).replace('\\', '/'))
        transform = transforms.ToTensor()

        input = transform(input_img).unsqueeze(0)
        target = np.array(target_img)

        with torch.no_grad():
            output = model(input)

        output = output.squeeze(0).detach().numpy()
        output *= 255.0
        output = output.clip(0, 255)
        output = output.transpose(1, 2, 0).astype('uint8')

        output_img = Image.fromarray(output, mode='RGB')
        output_img.save(os.path.join('../../dataset/inference/outputs', filename).replace('\\', '/'))

        psnr_total += metrics.psnr(output / 255, target / 255)
        num_images += 1

    print(f"Average PSNR: {psnr_total / num_images:.2f}")


# test(model=SRCNN())
# test(model=VDSR())
test(model=EDSR())
