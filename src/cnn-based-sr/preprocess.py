import os

from PIL import Image


def resize_all(src_path, dst_path, new_width, new_height, quality=75):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for filename in os.listdir(src_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img_path = os.path.join(src_path, filename).replace('\\', '/')
            img = Image.open(img_path)
            resized_img = img.resize((new_width, new_height))
            resized_img.save(os.path.join(dst_path, filename).replace('\\', '/'), quality=quality)


w, h = 256, 256

resize_all('../../dataset/train/original', '../../dataset/train/hr', w, h)
resize_all('../../dataset/inference/original', '../../dataset/inference/hr', w, h)

resize_all('../../dataset/train/original', '../../dataset/train/lr', 32, 32)
resize_all('../../dataset/inference/original', '../../dataset/inference/lr', 32, 32)

resize_all('../../dataset/train/lr', '../../dataset/train/lr', w, h)
resize_all('../../dataset/inference/lr', '../../dataset/inference/lr', w, h)
