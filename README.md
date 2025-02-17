# ComfyUI-CaptionThis  

[English](README.md) | [简体中文](README_CN.md)  

**ComfyUI-CaptionThis** is a flexible tool for generating image descriptions, supporting several powerful captioning models such as **Janus Pro** and **Florence2**, with plans to integrate more models like **JoyCaption** and other future developments. This tool aims to simplify workflows for **image-to-image tasks** and **LoRA dataset preparation** or similar fine-tuning processes, providing an intuitive way to describe individual images or batch process entire directories.  

---

## Workflow  

A **ShowText** or **DisplayText** node is needed to display the results of command execution. However, ComfyUI currently does not provide a native node for this purpose. In my example, I used the self-implemented **MieNodes** ([GitHub Repo](https://github.com/MieMieeeee/ComfyUI-MieNodes)), which I highly recommend. It’s easy to install, has no dependencies, and includes many caption-related file operation nodes.  

### Describe Image Using Janus Pro  
![JanusPro_Describe_Image](Images/JanusPro_Describe_Image.png)  

### Describe Image Using Florence2  
![Florence2_Describe_Image](Images/Florence2_Describe_Image.png)  

### Caption All Images in a Directory Using Janus Pro  
![JanusPro_Caption_Whole_Directory_1](Images/JanusPro_Caption_Whole_Directory_1.png)  

**Before:**  
![JanusPro_Caption_Whole_Directory_2](Images/JanusPro_Caption_Whole_Directory_2.png)  

**After:**  
![JanusPro_Caption_Whole_Directory_3](Images/JanusPro_Caption_Whole_Directory_3.png)  

### Caption All Images in a Directory Using Florence2  
![Florence2_Caption_Whole_Directory_1](Images/Florence2_Caption_Whole_Directory_1.png)  

**Works the same as Janus Pro**  

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

- Integration of new models (e.g., **JoyCaption**) to further enhance capabilities and adapt to more use cases.  
- Advanced configuration options to fine-tune caption outputs tailored to user requirements.  

---

## Credits  

Special thanks to:  
- [DeepSeek-AI](https://github.com/deepseek-ai/Janus) for providing the robust **Janus Pro** model.  
- [CY-CHENYUE](https://github.com/CY-CHENYUE/ComfyUI-Janus-Pro) and [kijai](https://github.com/kijai/ComfyUI-Florence2) for their practical implementations of **Janus Pro** and **Florence2** in their respective plugins, which served as key references and inspirations for the model integrations and functionality design in this project.  

Building on these contributions and practices, this project introduces a refined **multi-model architecture**, allowing users to choose the most suitable model based on their specific needs.  

---

## Contact  

- **Bilibili**: [@黎黎原上咩](https://space.bilibili.com/449342345)  
- **YouTube**: [@SweetValberry](https://www.youtube.com/@SweetValberry)  
