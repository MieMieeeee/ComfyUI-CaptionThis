import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM

import folder_paths
import comfy.model_management as mm

from .janus.models import VLChatProcessor
from .common import describe_images_core, image_to_pil_image
from .utils import mie_log

MY_CATEGORY = "🐑 JanusProCaption"
MODELS_DIR = os.path.join(folder_paths.models_dir, "Janus-Pro")


class JanusProModelLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    ["deepseek-ai/Janus-Pro-7B", "deepseek-ai/Janus-Pro-1B"],
                    {
                        "default": "deepseek-ai/Janus-Pro-7B"
                    }
                ),
            },
        }

    RETURN_TYPES = ("MIE_JANUS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = MY_CATEGORY

    def load_model(self, model_name):
        the_model_path = os.path.join(MODELS_DIR, os.path.basename(model_name))
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = mm.get_torch_device()

        if not os.path.exists(the_model_path):
            mie_log(f"Local model {model_name} not found at {the_model_path}, download from huggingface")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_name, local_dir=the_model_path, local_dir_use_symlinks=False)
        model = AutoModelForCausalLM.from_pretrained(
            the_model_path,
            trust_remote_code=True
        )

        try:
            dtype = torch.bfloat16
            torch.zeros(1, dtype=dtype, device=device)
        except RuntimeError:
            dtype = torch.float16

        model = model.to(dtype).to(device).eval()

        processor = VLChatProcessor.from_pretrained(the_model_path)

        return {"model": model, "processor": processor},


# Learn from https://github.com/CY-CHENYUE/ComfyUI-Janus-Pro
def describe_single_image(image, model, question, seed, temperature, top_p, max_new_tokens):
    processor = model['processor']
    model = model['model']

    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # 转换为PIL图像
    pil_image = image_to_pil_image(image)

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [pil_image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    prepare_inputs = processor(
        conversations=conversation,
        images=[pil_image],
        force_batchify=True
    ).to(model.device)

    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=processor.tokenizer.eos_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        use_cache=True,
    )

    answer = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

    return answer


class JanusProDescribeImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MIE_JANUS_MODEL",),
                "image": ("IMAGE",),
                "question": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail."
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "temperature": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0
                }),
                "max_new_tokens": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 2048
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "describe_image"
    CATEGORY = MY_CATEGORY

    def describe_image(self, model, image, question, seed, temperature, top_p, max_new_tokens, keep_model_loaded):
        answer = describe_single_image(image, model, question, seed, temperature, top_p, max_new_tokens)

        if not keep_model_loaded:
            mie_log("Offloading model...")
            model["model"].to(mm.unet_offload_device())
            mm.soft_empty_cache()

        return (answer,)

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed


class JanusProCaptionImageUnderDirectory:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MIE_JANUS_MODEL",),
                "directory": ("STRING", {"default": "X://path/to/files"}),
                "question": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail."
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "temperature": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0
                }),
                "max_new_tokens": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 2048
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                }),
                "save_to_new_directory": ("BOOLEAN", {
                    "default": False,
                }),
            },
            "optional": {
                "save_directory": ("STRING",),
                "is_relative_path": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "describe_images"
    CATEGORY = MY_CATEGORY

    def describe_images(self, model, directory, question, seed, temperature, top_p, max_new_tokens,
                        save_to_new_directory, save_directory, keep_model_loaded, is_relative_path=False):

        if is_relative_path:
            directory = os.path.join(folder_paths.base_path, directory)
            save_directory = os.path.join(folder_paths.base_path, save_directory) if save_directory else None

        mie_log(
            f"Describing images in {directory} and save to {save_directory if save_to_new_directory else directory}")
        result = describe_images_core(directory, save_to_new_directory, save_directory, describe_single_image,
                                      model, question, seed, temperature, top_p, max_new_tokens)
        if not keep_model_loaded:
            print("Offloading model...")
            model["model"].to(mm.unet_offload_device())
            mm.soft_empty_cache()

        return result

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed
