import os

from PIL import Image


def resize_all(src_dir, dst_dir, new_width, new_height):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for filename in os.listdir(src_dir):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img_path = os.path.join(src_dir, filename)
            img = Image.open(img_path)
            resized_img = img.resize((new_width, new_height))
            resized_img.save(os.path.join(dst_dir, filename))


resize_all('../inputs_original', '../inputs', 32, 32)
resize_all('../inputs_original', '../targets', 256, 256)
