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
from .common import hash_seed, mie_log, describe_images_core, image_to_pil_image, normalize_directory_path, assert_model_complete

import transformers

from safetensors.torch import save_file

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
    # 转换为PIL图像
    pil_image = image_to_pil_image(image)

    # transformers >= 5.0 no longer pulls `do_resize` / `size` / `resample`
    # from the preprocessor config defaults. The Florence2Processor wrapper
    # does NOT forward a `size=` kwarg to its image_processor, so passing
    # `size` here raises TypeError. The robust fix is to resize the PIL
    # image ourselves before handing it to the processor (with
    # `do_resize=False` to prevent the image_processor from re-applying its
    # own broken resize path). The DaViT vision tower requires square
    # feature maps; non-square inputs are rejected with
    # "only support square feature maps for now" in `_encode_image`.
    img_proc = getattr(processor, "image_processor", None)
    if img_proc is not None and getattr(img_proc, "do_resize", True):
        proc_size = dict(img_proc.size) if hasattr(img_proc.size, "__getitem__") else img_proc.size
        proc_resample = getattr(img_proc, "resample", 3)
        target_h = int(proc_size.get("height", 768))
        target_w = int(proc_size.get("width", 768))
        if pil_image.size != (target_w, target_h):
            pil_image = pil_image.resize((target_w, target_h), resample=proc_resample)
        inputs = processor(text=prompt, images=pil_image, return_tensors="pt", do_resize=False, do_rescale=False).to(dtype).to(device)
    else:
        inputs = processor(text=prompt, images=pil_image, return_tensors="pt", do_resize=False, do_rescale=False).to(dtype).to(device)

    # NOTE(v9-compat): the Florence-2 / BART-style decoder in this plugin does not
    # correctly interact with the transformers >= 5.0 `EncoderDecoderCache` API.
    # The decoder code path was originally written for the 4.x tuple-of-tuples
    # contract and has not been ported to honour `cache_position` / 4-D attention
    # masks. Enabling `use_cache=True` on 5.x produces degenerate output (e.g.
    # one token repeated hundreds of times). The robust workaround is to disable
    # the cache entirely -- the model produces correct captions at the cost of
    # O(max_new_tokens) instead of O(1) extra decoder steps. (Round 7+ attempts
    # at porting to 5.x cache.update() + is_updated dict also produced gibberish;
    # the cache object mutation interacts poorly with the BART decoder
    # cache_position expectations.)
    try:
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            use_cache=False,
        )
    except AttributeError as e:
        if "NoneType object has no attribute 'shape'" in str(e) or "NoneType' object has no attribute 'shape'" in str(e):
            # 遇到 transformers past_key_values bug，自动重试
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
                use_cache=False,
            )
        else:
            raise

    results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

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

        # Guard against a partially-downloaded model (e.g. only model.safetensors
        # present, config.json missing). Without this, loading fails with a
        # cryptic "'NoneType' object has no attribute 'model_type'".
        assert_model_complete(model_path, repo_id=model_name,
                              required_files=("config.json", "configuration_florence2.py",
                                              "modeling_florence2.py", "tokenizer.json"))

        mie_log(f"Florence2 using {attention} for attention")

        if transformers.__version__ < '4.51.0':
            with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
                 model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation=attention, torch_dtype=dtype,trust_remote_code=True)
        else:
            from .modeling_florence2 import Florence2ForConditionalGeneration
            model = Florence2ForConditionalGeneration.from_pretrained(model_path, attn_implementation=attention, torch_dtype=dtype)
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
                "is_relative_path": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "describe_images"
    CATEGORY = MY_CATEGORY

    def describe_images(self, model, directory, task, num_beams, max_new_tokens,
                        do_sample, seed, save_to_new_directory, save_directory, keep_model_loaded,
                        is_relative_path=False):
        device = mm.get_torch_device()
        processor = model['processor']
        dtype = model['dtype']
        model = model['model']
        model.to(device)
        set_seed(hash_seed(seed))

        task_prompt = prompts_map.get(task, '<CAPTION>')

        if is_relative_path:
            directory = os.path.join(folder_paths.base_path, directory)
            save_directory = os.path.join(folder_paths.base_path, save_directory) if save_directory else None
        # Normalize whatever the user typed (strip whitespace, expand ~, drop
        # trailing separator) so copy-pasted / POSIX paths resolve cleanly.
        directory = normalize_directory_path(directory)
        if save_directory:
            save_directory = normalize_directory_path(save_directory)

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
