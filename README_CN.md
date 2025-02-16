# ComfyUI-CaptionThis

[English](README.md) | 简体中文

**ComfyUI-CaptionThis** 是一个灵活的图像描述生成工具，支持多个强大的图像字幕生成模型，例如 **Janus Pro** 和 **Florence2**，并计划未来集成更多模型，例如 **JoyCaption** 以及其他可能的模型。该工具旨在简化 **图像到图像任务** 和 **LoRA 数据集准备** 或类似的微调流程，为用户提供描述单张图片或批量处理整个文件夹的直观方式。

特别感谢：
- [DeepSeek-AI](https://github.com/deepseek-ai/Janus) 提供了功能强大的 **Janus Pro** 模型；
- [CY-CHENYUE](https://github.com/CY-CHENYUE/ComfyUI-Janus-Pro) 和 [kijai](https://github.com/kijai/ComfyUI-Florence2) 在各自插件中对模型 **Janus Pro** 和 **Florence2** 的实践提供了重要参考，启发了本项目的模型整合与功能设计。

本项目借鉴了这些贡献和实践经验，并进一步完善了 **多模型架构**，使用户可以根据具体需求选择最适合的模型。

---

## 功能亮点

1. **单张图片描述**
   使用您选择的模型为单张图片生成详细的描述。用户可以输入图片，并根据需要添加具体提示或问题，丰富描述内容。

2. **批量描述生成**
   自动生成指定目录中多张图片的描述。每张图片都会生成相应的描述并保存为 `.txt` 文件，为数据集的高效准备提供便利。

3. **多模型支持**
   系统设计支持集成多个图像字幕生成模型，用户可根据任务需求灵活选择。目前已支持 **Janus Pro** 和 **Florence2**，未来更新中还将添加更多模型以增强功能覆盖范围。

---

## 即将推出

- 集成新模型（如 **JoyCaption**），进一步拓展功能，满足更多场景需求。
- 提供高级配置选项，以便根据用户需求微调生成的描述内容。

---

## 联系方式

- **哔哩哔哩**: [@黎黎原上咩](https://space.bilibili.com/449342345)
- **YouTube**: [@SweetValberry](https://www.youtube.com/@SweetValberry)