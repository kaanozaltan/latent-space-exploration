import os

from PIL import Image


def resize_all(src_path, dst_path, new_width, new_height, quality=75):
    for filename in os.listdir(src_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img_path = os.path.join(src_path, filename).replace('\\', '/')
            img = Image.open(img_path)
            resized_img = img.resize((new_width, new_height))
            resized_img.save(os.path.join(dst_path, filename).replace('\\', '/'), quality=quality)


resize_all('../samples_in_hr', '../samples_in', 256, 256)
