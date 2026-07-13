# ComfyUI-CaptionThis 兼容性修复 TODO

> 触发场景：在 ComfyUI 0.27.0 + Python 3.13.12 + transformers 5.9.0 环境下，启动 ComfyUI 时插件整体加载失败：
>
> ```
> File ".../ComfyUI-CaptionThis/janus/models/modeling_vlm.py", line 73, in <module>
>     class VisionConfig(PretrainedConfig):
>         ...
>         self.params = AttrDict(kwargs.get("params", {}))
> File ".../transformers/configuration_utils.py", line 316, in __init_subclass__
>     cls = dataclass(cls, repr=False, kw_only=True)
> File "dataclasses.py", line 1008, in _process_class
> ValueError: mutable default <class 'dict'> for field params is not allowed: use default_factory
> ```
>
> 根因：transformers 5.x 的 `PretrainedConfig.__init_subclass__` 在子类定义时把类包成 dataclass，原来的 `params: AttrDict = {}` 这种可变默认值在 dataclass 里直接禁止。

---

## 任务清单

### [ ] 1. 修复 `janus/models/modeling_vlm.py` 里 5 个 Config 子类的 `params` 默认值

**涉及类**（5 个）：
- `VisionConfig`（model_type = "vision"）—— 第 76 行
- `AlignerConfig`（model_type = "aligner"）—— 第 91 行
- `GenVisionConfig`（model_type = "gen_vision"）—— 第 106 行
- `GenAlignerConfig`（model_type = "gen_aligner"）—— 第 121 行
- `GenHeadConfig`（model_type = "gen_head"）—— 第 136 行

**当前代码（5 处都一样）**：
```python
class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: AttrDict = {}          # <-- 这里
```

**修法 A（推荐，最小改动）**：去掉类级默认值，只在 `__init__` 里赋值即可：
```python
class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    # params 不再类级声明；__init__ 里 self.params = AttrDict(kwargs.get("params", {})) 已经处理
```

这样 dataclass 看不到 `params` 字段就不会报错。`__init__` 仍会按原来逻辑把 params 写到实例上。

**修法 B（备选）**：保留类级声明但用 dataclass 字段工厂：
```python
from dataclasses import field

class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: AttrDict = field(default_factory=AttrDict)
```
注意 `field(...)` 必须在所有有默认值的字段最后面（Python 类字段顺序规则），并且 dataclass 装饰后的类不能继承非 dataclass 的父类，所以这个方案需要确认 `PretrainedConfig` 是否仍是合法父类。如果 transformers 5.x 用 `dataclass(cls, kw_only=True)` 包装后仍是 dataclass 化的，B 方案才稳。

**建议先试 A，不行再试 B**。

### [ ] 2. 验证 imports 还正确

确认 `janus/models/modeling_vlm.py` 顶部 imports 仍包含：
```python
from transformers.configuration_utils import PretrainedConfig
from attrdict import AttrDict
```
（已有，不用改）

如果选修法 B，需要加：
```python
from dataclasses import field
```

### [ ] 3. 检查 `MultiModalityConfig` 类是否也需要改

`MultiModalityConfig`（model_type = "multi_modality"）目前的写法是：
```python
class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig
    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig
```
全是类型注解，没默认值。dataclass 没问题，但要看 `__init__` 里有没有用 `kwargs.get("vision_config", ...)` 这种默认 dict 写法再确认。

### [ ] 4. （可选）跟 `transformers` 版本范围

`requirements.txt` 当前是 `transformers>=4.39.0,!=4.50.*`，没卡 5.x。建议改成：
```
transformers>=4.39.0,!=4.50.*,<6.0.0
```
防止未来 transformers 6.x 又一次破坏兼容性。

---

## 验证步骤

1. 改完后本地手动 import 一次：
   ```python
   from ComfyUI_CaptionThis.janus.models.modeling_vlm import VisionConfig, AlignerConfig, GenVisionConfig, GenAlignerConfig, GenHeadConfig
   ```
   不应再报 ValueError。

2. 拷贝改完的文件到 `E:\HH\Package\ComfyUI_Mie_2026_V9.0\ComfyUI\custom_nodes\comfyui_caption_this\janus\models\modeling_vlm.py`，重启 ComfyUI，看 `Cannot import ... module for custom nodes` 是否消失。

3. cu126 Package 那边也要同步拷一份：`E:\HH\Package\ComfyUI_Mie_2026_V9.0_cu126\ComfyUI\custom_nodes\comfyui_caption_this\janus\models\modeling_vlm.py`（文件名一样，路径前缀换）。

---

