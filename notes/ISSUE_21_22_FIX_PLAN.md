# Issue #21 / #22 修复方案：transformers 4.x / 5.x 双版本兼容

> 日期：2026-08-02
> 状态：已实现。生产代码（`florence2_processor.py`、`florence2_caption.py`、
> `janus/janusflow/`）与测试（`tests/test_florence2_processor_compat.py`、
> `tox.ini`）均已落地，三版本矩阵（4.49 / 4.57 / 5.9）全绿。
> 测试矩阵：`transformers==4.49.0`、`transformers==4.57.0`、`transformers==5.9.0`

## 1. 目标

1. 修复 Florence2 在 transformers 5.x 下加载 processor 时触发的 `forced_bos_token_id` AttributeError。
2. 同一份插件代码兼容 transformers 4.x 和 5.x，不要求用户降级或切换分支。
3. 不修改用户下载的模型文件，不依赖手工修补过的 Hugging Face 动态模块缓存。
4. 提供可重复的本地双版本单元测试，一条命令验证 4.57.0 与 5.9.0。
5. 测试不下载模型权重，也不依赖 ComfyUI 或本机既有模型缓存。

## 2. 已确认根因

v1.0.9 在 transformers 5.x 下先通过 branch B 成功加载插件本地模型类。随后，无条件执行的 `AutoProcessor.from_pretrained(model_path, trust_remote_code=True)` 加载模型目录中的 `processing_florence2.py`。

该 processor 在创建 tokenizer 时继续调用 `AutoTokenizer -> AutoConfig`，最终执行模型目录中的 remote `configuration_florence2.py`。MiaoshouAI 和 Microsoft 当前模型仓库里的文件仍裸读 `self.forced_bos_token_id`。

精确的 4.x/5.x 差异（已在本地 4.57.0 核对 `configuration_utils.py:318-321`）：

- **transformers 4.x**：`PretrainedConfig.__init__` 通过 setattr 循环把 `forced_bos_token_id` 等"生成默认参数"绑定为实例属性（值为 `None`）。于是 remote config 里 `self.forced_bos_token_id is None` 静默返回 `True`，**不崩溃**。
- **transformers 5.x**：删除了上述 setattr 循环，改为 `__post_init__` 中 pop-and-discard，该属性**从不绑定到实例**。remote config 裸读 `self.forced_bos_token_id` 直接抛 `AttributeError`。

插件本地 `configuration_florence2.py` 虽已改成 `getattr`（`d20e160`），但 processor 路径不会使用它——processor 路径加载的是 HF Hub 模型仓库自带的 remote config。

## 3. 方案比较

### 3.1 显式组装 processor，采用

动态加载模型目录中的 `Florence2Processor` 类，但显式创建 `BartTokenizerFast` 和 `CLIPImageProcessor`，再把两个组件传给 processor 构造函数。这样保留模型仓库自己的 processor 行为，同时绕过 `AutoProcessor -> AutoConfig`。
运行时数据流：

```text
model_path
  -> 读取 preprocessor_config.json 中的 auto_map.AutoProcessor
  -> get_class_from_dynamic_module(...) 取得 Florence2Processor
  -> BartTokenizerFast.from_pretrained(model_path)
  -> CLIPImageProcessor.from_pretrained(model_path)
  -> Florence2Processor(image_processor=..., tokenizer=...)
```

优点：不改模型文件；不依赖 remote config；继续使用与模型快照配套的 processor；4.x/5.x 共用一条路径。

### 3.2 覆盖模型目录中的 config，不采用

加载前把插件修复版 `configuration_florence2.py` 复制到模型目录，再保留 AutoProcessor。改动较小，但会修改用户文件，可能遇到只读目录、缓存失效和完整性校验问题。已有模型目录与新下载目录也必须同时处理。

### 3.3 将完整 processing_florence2.py 纳入插件，不采用

可以完全控制 processor，但会引入数百行上游代码。Microsoft/MiaoshouAI 更新处理逻辑时还需手工同步，当前问题不需要承担这项维护成本。

### 3.4 限制 transformers 小于 5.0，不采用

只能绕过问题，不能满足 4.x/5.x 同时兼容的目标。

## 4. 生产代码设计

### 4.1 新增独立 loader 模块

新增一个不依赖 ComfyUI 的小模块，例如 `florence2_processor.py`，暴露：

```python
def load_florence2_processor(model_path):
    ...
```

职责：

1. 使用标准 JSON 解析 `preprocessor_config.json`。
2. 读取并校验 `auto_map.AutoProcessor` 类引用。
3. 通过 transformers 的动态模块 API 从本地模型快照加载 processor 类。
4. 通过明确类型加载 Bart tokenizer 和 CLIP image processor。
5. 返回完成组装的 processor。

该模块不能导入 `folder_paths`、`comfy.model_management` 或其他 ComfyUI 组件，确保单元测试可在普通隔离 venv 中运行。

### 4.2 接入现有节点

将 `florence2_caption.py` 中唯一的 `AutoProcessor.from_pretrained(...)` 替换为新 helper。model 的 4.51 分支本轮不改，减少本次修复的行为范围。

### 4.3 错误处理

- 缺少 `preprocessor_config.json`：报告模型快照不完整，并给出缺失路径。
- 缺少或非法 `auto_map.AutoProcessor`：报告该模型不包含受支持的 Florence2 processor。
- 缺少 tokenizer/image processor 文件：保留 transformers 原始异常，并增加模型路径上下文。
- 不静默回退到 AutoProcessor，因为回退会重新引入本次 bug。

## 5. 测试设计

### 5.1 双版本矩阵

新增 `tox.ini`，创建三个完全隔离的环境，验证 processor loader / config helper /
Janus AST 检查在多个 transformers 版本下的兼容性：

| tox 环境 | 依赖 | 覆盖 |
|---|---|---|
| `transformers4_old` | `transformers==4.49.0`、Pillow | processor/helper/config 在 legacy 4.x（`< 4.51.0`）下的兼容性 |
| `transformers4` | `transformers==4.57.0`、Pillow | processor/helper/config 在 4.x 最新版（走 branch B） |
| `transformers5` | `transformers==5.9.0`、Pillow | processor/helper/config 在 5.x（Issue #21 目标，走 branch B） |

> **重要覆盖边界声明：** 这三个 tox 环境都不安装 torch，也不导入
> `florence2_caption.py` / `Florence2ModelLoader`，因此**不覆盖 branch A**
> （`< 4.51.0` 的 model `trust_remote_code` 分支）。它们验证的是 processor 组装逻辑、
> config 兼容性和 Janus AST 检查。branch A 的 model loading 需要完整的
> torch + ComfyUI 环境，本轮未纳入自动矩阵；现有三个 torch-bearing ComfyUI 环境
> （V8.0 / V9.0 / V9.0_cu126）当前实际都是 transformers 4.56.2，走 branch B，
> 因此也不能作为 branch A 已验证的证据。branch A model loading 本轮未自动验证。

三个环境执行相同测试：

```text
python tests/test_florence2_processor_compat.py
python tests/test_configuration_florence2_v9_compat.py
```

tox 环境不安装 torch，也不导入整个 ComfyUI 插件。测试夹具中的 processor 只依赖 `ProcessorMixin`，生产 helper 本身也不应要求 torch。真实 ComfyUI 环境仍由项目现有 E2E 覆盖。

### 5.2 本地最小模型夹具

测试在临时目录动态创建：

- `preprocessor_config.json`：声明动态 `Florence2Processor` 和最小 CLIP 配置。
- `processing_florence2.py`：行为与真实类构造接口一致的最小 processor。
- `vocab.json`、`merges.txt`、tokenizer 配置：供真实 `BartTokenizerFast` 加载。
- `config.json`：把 AutoConfig 指向危险 remote config。
- `configuration_florence2.py`：导入即抛出唯一 sentinel 异常。

该夹具有两个作用：先证明旧 AutoProcessor 路径确实会执行危险 config；再证明新 helper 能创建 processor 且 sentinel 从未触发。测试过程不访问网络。

### 5.2.1 锁定 processor 构造签名

Florence2Processor 的 `__init__(image_processor, tokenizer)` 签名来自 HF 上游，
插件无法控制。一旦上游改签名（例如未来加必填参数），方案 3.1 的显式组装会静默失败。
测试夹具需显式断言该构造接口（参数名包含 `image_processor` 和 `tokenizer`），让上游
签名变化时测试**立即失败**而不是运行时静默出错。

### 5.3 TDD 顺序

1. RED：添加新测试，目标 API 尚不存在，确认测试失败。
2. RED 证据：用同一夹具调用旧 AutoProcessor，确认命中 sentinel，证明夹具复现真实调用链。
3. GREEN：实现最小 loader，使新测试在当前开发环境通过。
4. MATRIX：在 transformers 4.57.0 和 5.9.0 两个 tox 环境运行。
5. REGRESSION：运行现有 Florence2 config/modeling/caption 相关测试。
6. REFACTOR：仅在全绿后整理错误消息和重复代码。

## 6. 验收标准

必须同时满足：

1. `tox -e transformers4_old,transformers4,transformers5` 返回 0。
2. 测试输出明确显示实际 transformers 版本分别为 4.49.0、4.57.0 和 5.9.0。
3. 三个版本均成功构造真实 BartTokenizerFast、CLIPImageProcessor 和动态 processor。
4. 危险 `configuration_florence2.py` 在新路径中从未被导入。
5. 现有 `test_configuration_florence2_v9_compat.py` 在两个版本均通过。
6. 现有 Florence2 相关本地回归测试无新增失败。
7. 生产代码不写入模型目录，也不删除 Hugging Face 缓存。
8. Issue #21 截图中的 processor 调用链不再可达。

## 7. 风险与控制

| 风险 | 控制 |
|---|---|
| transformers 动态模块 API 在 4.x/5.x 有差异 | 双版本 tox 直接执行相同 helper |
| 未来 Florence2 模型不再使用 Bart tokenizer | 对当前支持列表明确校验；遇到新模型时给出清晰错误，不自动走危险回退 |
| 模型 processor 类构造接口变化 | 测试锁定当前接口；真实模型 E2E 继续作为发布前检查 |
| tox 初次安装依赖需要网络 | 环境创建后使用本地 tox 缓存；测试运行本身离线 |
| 当前外部 V8/V9 测试路径和实际版本会漂移 | 新 tox 矩阵不依赖这些硬编码环境 |

## 8. 预计修改文件

| 文件 | 变更 |
|---|---|
| `florence2_processor.py` | 新增独立 processor loader |
| `florence2_caption.py` | 用新 helper 替换 AutoProcessor 调用 |
| `janus/janusflow/models/modeling_vlm.py` | 删除 3 处 `params: AttrDict = {}` 类级可变默认（Issue #22 的 janusflow 补全，见下方说明） |
| `tests/test_florence2_processor_compat.py` | 新增真实调用链回归测试和最小夹具 |
| `tox.ini` | 新增 4.49.0 / 4.57.0 / 5.9.0 隔离测试矩阵 |
| `.gitignore` | 忽略 `.tox/`（若当前未覆盖） |

### 关于 janusflow 修改的说明

`janus/janusflow/models/modeling_vlm.py` 不在原方案文件列表中。本次纳入的原因：
Issue #22 报告的 `ValueError: mutable default <class 'dict'> for field params is not allowed`
在 `janus/models/` 已由 v1.0.9 的 `a21c305` 修复，但 `janus/janusflow/` 的 3 个 Config 类
（`VisionUnderstandEncoderConfig` / `VisionGenerationEncoderConfig` /
`VisionGenerationDecoderConfig`）残留同样的 `params: AttrDict = {}` 类级可变默认。
janusflow 在本插件中是死代码（无任何生产代码 import），但它在 `janus/` 包内，一旦
transformers 5.x 的 `@dataclass` 装饰触发就会崩溃。为了彻底消除 Issue #22 而非只修一半，
本次用与 `a21c305` 完全相同的模式（去掉类级注解，保留 `__init__` 内
`self.params = AttrDict(kwargs.get("params", {}))`）一并修复。该变更由新增的
`test_janus_janusflow_modeling_vlm_no_mutable_default` AST 测试守护。

## 9. 本地验证命令

```powershell
.\.venv\Scripts\python.exe -m tox -e transformers4_old,transformers4,transformers5
```

如当前开发 venv 尚未安装 tox，只需安装测试工具；tox 创建的两个环境会自行安装各自锁定的 transformers 版本。

## 10. 非目标

- 本轮不重构 Florence2 model 的 4.51 版本分支（branch B 的 model 加载走插件本地 config，
  已安全，无需改动）。
- **本轮不自动验证 branch A（`< 4.51.0` 的 model `trust_remote_code` 分支）。** 该分支
  需要 torch + 完整 ComfyUI 环境才能执行 `Florence2ModelLoader`，不在 tox 矩阵覆盖内；
  现有三个 torch-bearing ComfyUI 环境当前都是 transformers 4.56.2（走 branch B），也不能
  作为 branch A 已验证的证据。branch A 在纯 4.x 下不触发本次 forced_bos bug（4.x setattr
  让属性存在），但本轮不对其 model loading 路径做自动回归。
- 本轮不修改 Hugging Face 模型仓库或用户缓存。
- 本轮不宣称覆盖 requirements 范围内每一个 4.x minor，只锁定选定的 4.49.0、4.57.0 与
  5.9.0 三个代表版本验证 processor/helper/config 兼容性（不覆盖 model branch A）。
- 本轮不处理 Issue #22 的 Manager 侧 requirements 解析问题：Issue #22 的核心矛盾
  （Janus dataclass `mutable default` ValueError）已由 v1.0.9 的 `a21c305` 修复；
  `requirements.txt` 被 ComfyUI Manager/Stability Matrix 解析失败属第三方包管理器
  bug，与 `pillow>= 10.2.0` 的空格无关，不在本轮范围。

## 11. 回滚

若显式组装 processor 在真实模型 E2E 出现不可接受的行为差异，可单独回滚 `florence2_caption.py` 的调用点和 helper，不影响已有本地 config/modeling 的 4.x/5.x 修复。
 