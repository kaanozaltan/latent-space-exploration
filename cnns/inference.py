import os

import metrics
import numpy as np
import torch
from models import EDSR, SRCNN, VDSR
from PIL import Image
from torchvision import transforms


def test(model):
    model_name = model.get_name()
    print(f"Performing inference for {model_name.upper()} model")
    model.load_state_dict(torch.load(f"models/{model_name}.pt"))
    model.eval()

    if not os.path.exists("dataset/inference/outputs"):
        os.makedirs("dataset/inference/outputs")

    psnr_total = 0.0
    num_images = 0

    for filename in os.listdir("dataset/inference/lr"):
        input_img = Image.open(os.path.join("dataset/inference/lr", filename))
        target_img = Image.open(os.path.join("dataset/inference/hr", filename))
        transform = transforms.ToTensor()

        input = transform(input_img).unsqueeze(0)
        target = np.array(target_img)

        with torch.no_grad():
            output = model(input)

        output = output.squeeze(0).detach().numpy()
        output *= 255.0
        output = output.clip(0, 255)
        output = output.transpose(1, 2, 0).astype("uint8")

        output_img = Image.fromarray(output, mode="RGB")
        output_img.save(os.path.join("dataset/inference/outputs", filename))

        psnr_total += metrics.psnr(output / 255, target / 255)
        num_images += 1

    print(f"Average PSNR: {psnr_total / num_images:.2f}")


if __name__ == "__main__":
    # test(model=SRCNN())
    # test(model=VDSR())
    test(model=EDSR())
