import os
import torch
import hashlib
import numpy as np
from glob import glob
from PIL import Image

from nodes import node_helpers, ImageSequence, ImageOps
from .utils import mie_log


# Image extensions we consider valid inputs for batch captioning. Filtering by
# extension first (instead of opening every file with PIL) is dramatically faster
# and far more robust on network mounts (NFS/SMB) and inside containers, where
# directories routinely contain `.DS_Store`, `Thumbs.db`, partial transfers, and
# other non-image entries that would otherwise each trigger a slow `Image.open`.
# This is the root cause of issue #13 ("Cannot figure out path format (linux)"):
# the directory was reachable but the per-file probe failed/slowed on mixed content.
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff",
}


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


def is_image_file(file_path):
    """
    Check if a file is a valid image using Pillow.

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - bool: True if the file is a valid image, False otherwise.
    """
    try:
        with Image.open(file_path) as img:
            return img.format is not None  # Returns True if the image format is valid
    except (IOError, FileNotFoundError, Image.UnidentifiedImageError, ValueError):
        return False


def normalize_directory_path(directory):
    """Normalize a user-supplied directory path.

    Handles the common copy-paste / cross-platform pitfalls that caused
    issue #13: leading/trailing whitespace, a leading ``~`` (POSIX home),
    mixed separators, and a trailing separator. Returns the cleaned path
    (without resolving symlinks, so NFS mounts are left as-is).
    """
    if directory is None:
        return directory
    directory = directory.strip()
    directory = os.path.expanduser(directory)
    directory = os.path.normpath(directory)
    return directory


def assert_model_complete(model_path, repo_id=None, required_files=("config.json",)):
    """Verify a downloaded model directory actually contains the files needed
    to load, not just the weights.

    A known failure mode (seen on a LAN V9 deploy): ``snapshot_download`` wrote
    ``model.safetensors`` (1GB) but the small text files (``config.json``,
    tokenizer, modeling code) were never written -- e.g. interrupted download
    or partial network failure. Loading then fails with a cryptic
    ``'NoneType' object has no attribute 'model_type'`` because transformers
    falls back to an empty default config. This raises a clear, actionable
    error instead.

    Parameters:
    - model_path: the local directory the model was downloaded into.
    - repo_id: original HF repo id, included in the message to help re-download.
    - required_files: files that MUST be present. ``config.json`` is the
      minimum; callers can add more (e.g. tokenizer files).
    """
    missing = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    if not missing:
        return
    where = f" from '{repo_id}'" if repo_id else ""
    raise FileNotFoundError(
        f"Model directory is incomplete{where}: {model_path!r}\n"
        f"Missing required file(s): {missing}.\n"
        f"This usually means a previous download was interrupted. Please delete "
        f"the directory (and its .cache) and let the node re-download, or copy "
        f"the full model files from a known-good source."
    )


def get_image_files(directory):
    """Return the list of image files in ``directory`` (non-recursive).

    Filters by extension first (fast, robust on network mounts), then drops
    anything that is not a regular file. We intentionally do NOT call
    ``Image.open`` per file here -- on NFS/SMB a directory may contain
    non-image entries (``.DS_Store``, partial transfers, etc.) and probing
    each one is both slow and a source of cryptic failures. Extension
    filtering matches how other ComfyUI image nodes behave.
    """
    entries = glob(os.path.join(directory, "*"))
    image_files = []
    for f in entries:
        if not os.path.isfile(f):
            continue
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
            image_files.append(f)
    return image_files


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

    # Distinguish the three real failure modes so the user gets an actionable
    # message instead of a generic "no images". Issue #13 was unresolvable
    # largely because the error gave no clue which of these it was.
    if not os.path.exists(directory):
        return (
            f"Directory does not exist: {directory!r}. Check the path spelling, "
            f"that the mount is active, and (if it is relative to ComfyUI) enable "
            f"the 'is_relative_path' option or pass an absolute path."
        ),
    if not os.path.isdir(directory):
        return f"Path is not a directory: {directory!r} (it is a file).",

    try:
        image_files = get_image_files(directory)
    except PermissionError:
        return (
            f"Permission denied reading directory: {directory!r}. Check that the "
            f"ComfyUI process has read access to it (common inside Docker / on NFS)."
        ),

    if not image_files:
        # Help the user see WHY nothing was found: list what is actually there.
        try:
            all_entries = [os.path.basename(p) for p in glob(os.path.join(directory, "*"))]
        except Exception:
            all_entries = []
        exts_seen = sorted({os.path.splitext(p)[1].lower() for p in all_entries if os.path.splitext(p)[1]})
        sample = ", ".join(all_entries[:5])
        hint_exts = ", ".join(sorted(IMAGE_EXTENSIONS))
        return (
            f"No image files found in {directory!r}. Found {len(all_entries)} entries "
            f"with extensions: [{exts_seen}]. Only these extensions are recognized: "
            f"[{hint_exts}]. Sample entries: {sample}"
        ),

    for image_file in image_files:
        image = load_image_core(image_file)[0]
        answer = describe_function(image, *argv)
        save_description(image_file, answer, new_directory)

    the_log_message = f"Described {len(image_files)} images in {directory}."
    mie_log(the_log_message)
    return the_log_message,


def hash_seed(seed):
    # Convert the seed to a string and then to bytes
    seed_bytes = str(seed).encode('utf-8')
    # Create a SHA-256 hash of the seed bytes
    hash_object = hashlib.sha256(seed_bytes)
    # Convert the hash to an integer
    hashed_seed = int(hash_object.hexdigest(), 16)
    # Ensure the hashed seed is within the acceptable range for set_seed
    return hashed_seed % (2 ** 32)


def image_to_pil_image(image):
    """
    Convert a BCHW/CHW/HWC PyTorch Tensor or NumPy array to a PIL Image in RGB format.

    Parameters:
        image (torch.Tensor or np.ndarray): Input image.

    Returns:
        PIL.Image.Image: Converted PIL Image (RGB).
    """
    # Step 1: Handle BCHW (Batch, Channel, Height, Width)
    if len(image.shape) == 4:  # Batch input
        image = image[0]

    # Step 2: Ensure image values are in [0, 255] and type is uint8
    if isinstance(image, torch.Tensor):
        image = torch.clamp(image, 0, 1)  # Clamp values to [0, 1]
        image = (image * 255).byte().cpu().numpy()  # Convert to uint8 NumPy array
    elif isinstance(image, np.ndarray):
        image = np.clip(image, 0, 1)  # Clamp values to [0, 1]
        image = (image * 255).astype(np.uint8)  # Convert to uint8
    else:
        raise TypeError(f"Unsupported input type {type(image)}. Expected torch.Tensor or numpy.ndarray.")

    # Step 3: Handle [C, H, W] -> [H, W, C] conversion
    if len(image.shape) == 3:
        if image.shape[0] in [3, 4]:  # [C, H, W] format
            image = np.transpose(image, (1, 2, 0))  # Convert to [H, W, C]
        elif image.shape[2] not in [3, 4]:  # Invalid channels
            raise ValueError(f"Unsupported channel size: {image.shape[2]}. Must be 3 (RGB) or 4 (RGBA).")

    # Step 4: Convert to PIL Image
    mode = {3: 'RGB', 4: 'RGBA'}.get(image.shape[2])
    if mode is None:
        raise ValueError(f"Unsupported channel size: {image.shape[2]}. Must be 3 (RGB) or 4 (RGBA).")
    pil_image = Image.fromarray(image, mode=mode).convert('RGB')  # Always convert to RGB

    return pil_image
