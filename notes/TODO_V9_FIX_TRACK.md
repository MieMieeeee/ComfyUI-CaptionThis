# ComfyUI-CaptionThis V9 Compatibility Fix — Tracking

> Status legend: `[ ]` todo · `[~]` in-progress · `[x]` done · `[-]` skip / not applicable

## 0. Diagnosis Snapshot

| Env | Python | transformers | attrdict | Plugin path | Pre-fix load status |
|---|---|---|---|---|---|
| `E:\FF\ComfyUI_Mie_2026_V8.0` | 3.13.11 | 4.56.2 | installed (broken `Mapping`) | `ComfyUI\custom_nodes\ComfyUI-CaptionThis\` | ✅ load OK (transformers 4.x does not dataclass-wrap; `janus/__init__.py` monkey-patches `collections.Mapping` for `attrdict`) |
| `E:\HH\Package\ComfyUI_Mie_2026_V9.0` | 3.13.12 | **5.9.0** | installed (broken `Mapping`) | `ComfyUI\custom_nodes\comfyui_caption_this\` | ❌ dataclass mutable-default crash at `VisionConfig` definition |
| `E:\HH\Package\ComfyUI_Mie_2026_V9.0_cu126` | 3.12.10 | **5.9.0** | **not installed** | `ComfyUI\custom_nodes\comfyui_caption_this\` | ❌ `ModuleNotFoundError: attrdict` first, then dataclass crash |

## 1. Tasks

- [x] 1.1 Write failing TDD test that imports `janus.models.modeling_vlm.{VisionConfig,...}` under transformers 5.x and asserts no `ValueError` — `tests/test_modeling_vlm_v9_compat.py`.
- [x] 1.2 Apply minimal fix in `janus/models/modeling_vlm.py`: drop the 5× class-level `params: AttrDict = {}` (Solution A from TODO_V9_COMPAT.md). Per-instance assignment in `__init__` keeps existing semantics.
- [x] 1.3 Re-run failing test on V8.0 (transformers 4.56.2) — **regression-clean** (4/4 PASS).
- [x] 1.4 Re-run failing test on V9.0 (transformers 5.9.0) — **4/4 PASS** (was 0/4 before fix).
- [x] 1.5 `pip install attrdict` into V9.0_cu126; re-run failing test — **4/4 PASS** (was 0/4 before).
- [x] 1.6 Copy fixed `janus/models/modeling_vlm.py` into V9.0 and V9.0_cu126 `custom_nodes/comfyui_caption_this/`. SHA-256 verified equal to source.
- [x] 1.7 End-to-end probe (`tests/probe_all_envs.py`): `from <plugin>.janus.models import VLChatProcessor, MultiModalityCausalLM` on all three envs — **3/3 PASS**.
- [x] 1.8 Update `requirements.txt`: added `transformers<6.0.0` upper bound + a comment annotating the `attrdict` requirement.
- [ ] 1.9 (Optional) Commit & push — pending user sign-off.

## 2. Files Touched

| File | Before | After |
|---|---|---|
| `janus/models/modeling_vlm.py` | 5× `params: AttrDict = {}` at class level | class-level annotation removed; per-instance `__init__` assignment preserved; comment block explaining the why |
| `requirements.txt` | `transformers>=4.39.0,!=4.50.*` | `transformers>=4.39.0,!=4.50.*,<6.0.0` + explanatory comment for `attrdict` |
| `tests/test_modeling_vlm_v9_compat.py` | n/a | new regression test (4 sub-tests) for unit-level Config class load + AutoConfig registration |
| `tests/probe_all_envs.py` | n/a | new cross-environment probe (subprocess per env) for plugin-level import smoke |
| V9.0 `custom_nodes/comfyui_caption_this/janus/models/modeling_vlm.py` | unfixed | synced from main (SHA-256 match) |
| V9.0_cu126 `custom_nodes/comfyui_caption_this/janus/models/modeling_vlm.py` | unfixed | synced from main (SHA-256 match) |

## 3. Risk / Out-of-Scope

- We do not touch `modeling_florence2.py` (already patched by Kijai commit 0877928 for transformers > 4.52).
- We do not modify `janus/janusflow/` — it is dead code in this plugin (no node references it) and has the same issue; will be addressed separately if/when anyone activates a Janus-Flow path.
- We did **not** attempt Solution B (`@dataclass(field(default_factory=AttrDict))` rewrite) — Solution A is the minimal-blast-radius fix and is sufficient.
- We did not delete the `attrdict` line from `requirements.txt` because `attrdict` is genuinely required at runtime by `projector.py` (`AttrDict(...)` call) and `janus/janusflow/models/modeling_vlm.py` (kept for forward-compat with potential future Janus-Flow loading).

## 4. Verification Commands Replay

```powershell
# One-off unit test on V9.0 (red→green demonstrated)
E:\HH\Package\ComfyUI_Mie_2026_V9.0\python_embeded\python.exe `
  C:\Users\administered\PycharmProjects\ComfyUI-CaptionThis\tests\test_modeling_vlm_v9_compat.py

# Cross-env probe (all three at once)
C:\Users\administered\PycharmProjects\ComfyUI-CaptionThis\.venv\Scripts\python.exe `
  C:\Users\administered\PycharmProjects\ComfyUI-CaptionThis\tests\probe_all_envs.py
```
