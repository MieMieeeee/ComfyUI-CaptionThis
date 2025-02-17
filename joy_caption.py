import os
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

import folder_paths
import comfy.model_management as mm

from .common import bf16_or_fp16, mie_log, describe_images_core, image_to_pil_image

MY_CATEGORY = "🐑 JoyCaption"
MODELS_DIR = os.path.join(folder_paths.models_dir, "Joy_caption_alpha")


class JoyCaptionLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (
                [
                    'fancyfeast/llama-joycaption-alpha-two-hf-llava'
                ],
                {
                    "default": 'fancyfeast/llama-joycaption-alpha-two-hf-llava'
                }),
        }
        }

    RETURN_TYPES = ("MIE_JOYCAPTION_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = MY_CATEGORY

    def load_model(self, model_name):
        the_model_path = os.path.join(MODELS_DIR, os.path.basename(model_name))
        device = mm.get_torch_device()

        if not os.path.exists(the_model_path):
            mie_log(f"Downloading {model_name} to: {the_model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_name,
                              local_dir=the_model_path,
                              local_dir_use_symlinks=False)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        dtype = bf16_or_fp16()
        model = model.to(dtype).to(device).eval()

        processor = AutoProcessor.from_pretrained(model_name)

        return {
            'model': model,
            'processor': processor,
        },


def describe_single_image(image, model, question, seed, temperature, top_p, max_new_tokens, do_sample):
    processor = model['processor']
    model = model['model']
    device = mm.get_torch_device()
    dtype = bf16_or_fp16()

    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    with torch.no_grad():
        # Load image
        pil_image = image_to_pil_image(image)

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        # Format the conversation
        # WARNING: HF's handling of chat's on Llava models is very fragile.  This specific combination of processor.apply_chat_template(), and processor() works
        # but if using other combinations always inspect the final input_ids to ensure they are correct.  Often times you will end up with multiple <bos> tokens
        # if not careful, which can make the model perform poorly.
        convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

        # Process the inputs
        inputs = processor(text=[convo_string], images=[pil_image], return_tensors="pt").to(device)
        inputs['pixel_values'] = inputs['pixel_values'].to(dtype)

        # Generate the captions
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            suppress_tokens=None,
            use_cache=True,
            temperature=temperature,
            top_k=None,
            top_p=top_p,
        )[0]

        # Trim off the prompt
        generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

        # Decode the caption
        caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        caption = caption.strip()

        return caption


class JoyCaptionDescribeImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MIE_JOYCAPTION_MODEL",),
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
                "do_sample": ("BOOLEAN", {
                    "default": True
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "describe_image"
    CATEGORY = MY_CATEGORY

    def describe_image(self, model, image, question, seed, temperature, top_p, max_new_tokens, keep_model_loaded,
                       do_sample):
        answer = describe_single_image(image, model, question, seed, temperature, top_p, max_new_tokens, do_sample)

        if not keep_model_loaded:
            print("Offloading model...")
            model.to(mm.unet_offload_device())
            mm.soft_empty_cache()

        return (answer,)

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed


class JoyCaptionImageUnderDirectory:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MIE_JOYCAPTION_MODEL",),
                "directory": ("STRING", {"default": "X://path/to/files"}),
                "question": ("STRING", {
                    "multiline": True,
                    "default": "Write a descriptive caption for this image in a formal tone."
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
                "do_sample": ("BOOLEAN", {
                    "default": True
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True
                }),
                "save_to_new_directory": ("BOOLEAN", {
                    "default": False,
                }),
            },
            "optional": {
                "save_directory": ("STRING",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "describe_images"
    CATEGORY = MY_CATEGORY

    def describe_images(self, model, directory, question, seed, temperature, top_p, max_new_tokens, do_sample,
                        save_to_new_directory, save_directory, keep_model_loaded):
        mie_log(
            f"Describing images in {directory} and save to {save_directory if save_to_new_directory else directory}")
        result = describe_images_core(directory, save_to_new_directory, save_directory, describe_single_image,
                                      model, question, seed, temperature, top_p, max_new_tokens, do_sample)
        if not keep_model_loaded:
            print("Offloading model...")
            model.to(mm.unet_offload_device())
            mm.soft_empty_cache()

        return result

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed
