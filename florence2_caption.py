# learn from https://github.com/kijai/ComfyUI-Florence2
from collections.abc import Callable
import torch
import os
import numpy as np

from PIL import Image
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor, set_seed

# workaround for unnecessary flash_attn requirement
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

import folder_paths
import comfy.model_management as mm
from .common import hash_seed, mie_log, describe_images_core

script_directory = os.path.dirname(os.path.abspath(__file__))
model_directory = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(model_directory, exist_ok=True)

# Ensure ComfyUI knows about the LLM model path
folder_paths.add_model_folder_path("LLM", model_directory)

MY_CATEGORY = "🐑 Florence2Caption"


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    try:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
    except:
        print(f"No flash_attn import to remove")
        pass
    return imports


def create_path_dict(paths: list[str], predicate: Callable[[Path], bool] = lambda _: True) -> dict[str, str]:
    """
    Creates a flat dictionary of the contents of all given paths: ``{name: absolute_path}``.

    Non-recursive.  Optionally takes a predicate to filter items.  Duplicate names overwrite (the last one wins).

    Args:
        paths (list[str]):
            The paths to search for items.
        predicate (Callable[[Path], bool]):
            (Optional) If provided, each path is tested against this filter.
            Returns ``True`` to include a path.

            Default: Include everything
    """

    flattened_paths = [item for path in paths for item in Path(path).iterdir() if predicate(item)]

    return {item.name: str(item.absolute()) for item in flattened_paths}


prompts_map = {
    # Official & MiaoshouAI Florence2 prompts
    'caption': '<CAPTION>',
    'detailed_caption': '<DETAILED_CAPTION>',
    'more_detailed_caption': '<MORE_DETAILED_CAPTION>',

    # MiaoshouAI prompts
    'tags': '<GENERATE_TAGS>',
    'mixed': '<MIX_CAPTION>',
    'extra_mixed': '<MIX_CAPTION_PLUS>',
    'analyze': '<ANALYZE>',
}


def describe_single_image(image, model, processor, prompt, device, dtype, num_beams=3, max_new_tokens=1024,
                          do_sample=True):
    # ComfyUI中的图像格式是 BCHW (Batch, Channel, Height, Width)
    if len(image.shape) == 4:  # BCHW format
        if image.shape[0] == 1:
            image = image.squeeze(0)  # 移除batch维度，现在是 [H, W, C]

    # 确保值范围在[0,1]之间并转换为uint8
    image = (torch.clamp(image, 0, 1) * 255).cpu().numpy().astype(np.uint8)

    # 转换为PIL图像
    pil_image = Image.fromarray(image, mode='RGB')

    inputs = processor(text=prompt, images=pil_image, return_tensors="pt", do_rescale=False).to(dtype).to(
        device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
    )

    results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    print(results)
    # cleanup the special tokens from the final list

    clean_results = str(results)
    clean_results = clean_results.replace('</s>', '')
    clean_results = clean_results.replace('<s>', '')

    return clean_results


class Florence2ModelLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (
                [
                    'microsoft/Florence-2-base',
                    'microsoft/Florence-2-base-ft',
                    'microsoft/Florence-2-large',
                    'microsoft/Florence-2-large-ft',
                    'MiaoshouAI/Florence-2-base-PromptGen-v1.5',
                    'MiaoshouAI/Florence-2-large-PromptGen-v1.5',
                    'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
                    'MiaoshouAI/Florence-2-large-PromptGen-v2.0'
                ],
                {
                    "default": 'MiaoshouAI/Florence-2-base-PromptGen-v2.0'
                }),
            "precision": (['fp16', 'bf16', 'fp32'],
                          {
                              "default": 'fp16'
                          }),
            "attention": (
                ['flash_attention_2', 'sdpa', 'eager'],
                {
                    "default": 'sdpa'
                }),
        }
        }

    RETURN_TYPES = ("MIE_FLORENCE2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = MY_CATEGORY

    def load_model(self, model_name, precision, attention):
        device = mm.get_torch_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_path = os.path.join(model_directory, model_name.rsplit('/', 1)[-1])

        if not os.path.exists(model_path):
            mie_log(f"Downloading Florence2 model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_name,
                              local_dir=model_path,
                              local_dir_use_symlinks=False)

        mie_log(f"Florence2 using {attention} for attention")
        with patch("transformers.dynamic_module_utils.get_imports",
                   fixed_get_imports):  # workaround for unnecessary flash_attn requirement
            model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation=attention, device_map=device,
                                                         torch_dtype=dtype, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        florence2_model = {
            'model': model,
            'processor': processor,
            'dtype': dtype
        }

        return (florence2_model,)


class Florence2DescribeImage:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MIE_FLORENCE2_MODEL",),
                "image": ("IMAGE",),
                "task": (list(prompts_map.keys()), {"default": "more_detailed_caption"}),
                "seed": ("INT", {"default": 42, "min": 1, "max": 0xffffffffffffffff}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 64}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "describe_image"
    CATEGORY = MY_CATEGORY

    def describe_image(self, image, model, task, num_beams, max_new_tokens,
                       do_sample, seed, keep_model_loaded):
        device = mm.get_torch_device()
        processor = model['processor']
        dtype = model['dtype']
        model = model['model']
        model.to(device)
        set_seed(hash_seed(seed))

        out_result = describe_single_image(image, model, processor,
                                           prompts_map.get(task, '<CAPTION>'),
                                           device, dtype, num_beams, max_new_tokens, do_sample)

        if not keep_model_loaded:
            mie_log("Offloading model...")
            model.to(mm.unet_offload_device())
            mm.soft_empty_cache()

        mie_log(f"Described single image: {out_result}")
        return out_result,


class Florence2CaptionImageUnderDirectory:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MIE_FLORENCE2_MODEL",),
                "directory": ("STRING", {"default": "X://path/to/files"}),
                "task": (list(prompts_map.keys()), {"default": "more_detailed_caption"}),
                "seed": ("INT", {"default": 42, "min": 1, "max": 0xffffffffffffffff}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 64}),
                "do_sample": ("BOOLEAN", {"default": True, }),
                "keep_model_loaded": ("BOOLEAN", {"default": True, }),
                "save_to_new_directory": ("BOOLEAN", {"default": False, }),
            },
            "optional": {
                "save_directory": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "describe_images"
    CATEGORY = MY_CATEGORY

    def describe_images(self, model, directory, task, num_beams, max_new_tokens,
                        do_sample, seed, save_to_new_directory, save_directory, keep_model_loaded):
        device = mm.get_torch_device()
        processor = model['processor']
        dtype = model['dtype']
        model = model['model']
        model.to(device)
        set_seed(hash_seed(seed))

        task_prompt = prompts_map.get(task, '<CAPTION>')

        mie_log(
            f"Describing images in {directory} and save to {save_directory if save_to_new_directory else directory}")
        result = describe_images_core(directory, save_to_new_directory, save_directory, describe_single_image,
                                      model, processor, task_prompt, device, dtype, num_beams, max_new_tokens,
                                      do_sample)

        if not keep_model_loaded:
            mie_log("Offloading model...")
            model.to(mm.unet_offload_device())
            mm.soft_empty_cache()

        return result
