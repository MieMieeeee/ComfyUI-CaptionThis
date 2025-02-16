# ComfyUI-CaptionThis

English | [简体中文](README_CN)

**ComfyUI-CaptionThis** is a flexible tool for generating image descriptions, supporting several powerful captioning models such as **Janus Pro** and **Florence2**, with plans to integrate more models like **JoyCaption** and other future developments. This tool aims to simplify workflows for **image-to-image tasks** and **LoRA dataset preparation** or similar fine-tuning processes, providing an intuitive way to describe individual images or batch process entire directories.

Special thanks to:
- [DeepSeek-AI](https://github.com/deepseek-ai/Janus) for providing the robust **Janus Pro** model;
- [CY-CHENYUE](https://github.com/CY-CHENYUE/ComfyUI-Janus-Pro) and [kijai](https://github.com/kijai/ComfyUI-Florence2) for their practical implementations of **Janus Pro** and **Florence2** in their respective plugins, which served as key references and inspirations for the model integrations and functionality design in this project.

Building on these contributions and practices, this project introduces a refined **multi-model architecture**, allowing users to choose the most suitable model based on their specific needs.

---

## Features

1. **Single Image Description**
   Generate detailed captions for a single image using your chosen model. Users can input an image and optionally provide specific prompts or guiding questions to enrich the description.

2. **Batch Caption Generation**
   Automatically generate captions for multiple images in a specified directory. Each image will have its corresponding description saved as a `.txt` file, enabling efficient dataset preparation.

3. **Multi-Model Support**
   The system is designed to integrate multiple captioning models, offering users the flexibility to select based on their tasks. Currently, **Janus Pro** and **Florence2** are supported, with future updates bringing additional models to expand functionality and coverage.

---

## Coming Soon

- Integration of new models (e.g., **JoyCaption**) to further expand capabilities and adapt to more use cases.
- Advanced configuration options for fine-tuning caption outputs tailored to user requirements.

---

## Contact

- **Bilibili**: [@黎黎原上咩](https://space.bilibili.com/449342345)
- **YouTube**: [@SweetValberry](https://www.youtube.com/@SweetValberry)