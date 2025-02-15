import os
import torch
import imghdr
import numpy as np
from glob import glob
from PIL import Image

from nodes import node_helpers, ImageSequence, ImageOps
from .utils import mie_log


# From ComfyUI Core
def load_image_core(image_path):
    img = node_helpers.pillow(Image.open, image_path)

    output_images = []
    output_masks = []
    w, h = None, None

    excluded_formats = ['MPO']

    for i in ImageSequence.Iterator(img):
        i = node_helpers.pillow(ImageOps.exif_transpose, i)

        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")

        if len(output_images) == 0:
            w = image.size[0]
            h = image.size[1]

        if image.size[0] != w or image.size[1] != h:
            continue

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1 and img.format not in excluded_formats:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return (output_image, output_mask)


def get_image_files(directory):
    return [f for f in glob(os.path.join(directory, "*")) if os.path.isfile(f) and imghdr.what(f)]


def save_description(image_file, description, directory_to_save=None):
    mie_log(f"Saving description for {image_file} to {directory_to_save}")
    if directory_to_save:
        os.makedirs(directory_to_save, exist_ok=True)
        txt_file = os.path.join(directory_to_save, os.path.basename(os.path.splitext(image_file)[0]) + ".txt")
    else:
        txt_file = os.path.splitext(image_file)[0] + ".txt"

    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(description)


def describe_images_core(directory, save_to_new_directory, new_directory, describe_function, *argv):
    if not save_to_new_directory:
        new_directory = None

    image_files = get_image_files(directory)
    if not image_files:
        return f"No image files found in the {directory}.",

    for image_file in image_files:
        image = load_image_core(image_file)[0]
        answer = describe_function(image, *argv)
        save_description(image_file, answer, new_directory)

    the_log_message = f"Described {len(image_files)} images in {directory}."
    mie_log(the_log_message)
    return the_log_message,
