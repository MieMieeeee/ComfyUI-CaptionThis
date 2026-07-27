# ComfyUI-CaptionThis V9 Compatibility Fix — Tracking

> Status legend: `[ ]` todo · `[~]` in-progress · `[x]` done · `[-]` skip / not applicable

## 0. Diagnosis Snapshot

| Env | Python | transformers | attrdict | Plugin path | Pre-fix status |
|---|---|---|---|---|---|
| `E:\FF\ComfyUI_Mie_2026_V8.0` | 3.13.11 | 4.56.2 | installed (broken `Mapping`) | `ComfyUI\custom_nodes\ComfyUI-CaptionThis\` | Janus: ✅ (transformers 4.x does not dataclass-wrap) |
| `E:\HH\Package\ComfyUI_Mie_2026_V9.0` | 3.13.12 | **5.9.0** | installed (broken `Mapping`) | `ComfyUI\custom_nodes\comfyui_caption_this\` | Janus: ❌ dataclass mutable-default crash |
| `E:\HH\Package\ComfyUI_Mie_2026_V9.0_cu126` | 3.12.10 | **5.9.0** | **not installed** | `ComfyUI\custom_nodes\comfyui_caption_this\` | Janus: ❌ `ModuleNotFoundError: attrdict` first, then dataclass crash |

After the v9 round, both V9.0 environments also hit a **second** bug under
transformers 5.x: `Florence2ForConditionalGeneration.from_pretrained` fails
inside `configuration_florence2.py:265` with
`AttributeError: ''Florence2LanguageConfig'' object has no attribute ''forced_bos_token_id''`.
Root cause is the same family of 5.x behavioural drift — generation defaults
(including `forced_bos_token_id`) are popped-and-discarded in
`PretrainedConfig.__post_init__` instead of being set on the instance.

## 1. Tasks

### Round 1 — Janus dataclass mutable-default (commit a21c305)

- [x] 1.1 Write failing TDD test that imports `janus.models.modeling_vlm.{VisionConfig,...}` under transformers 5.x and asserts no `ValueError` — `tests/test_modeling_vlm_v9_compat.py`.
- [x] 1.2 Apply minimal fix in `janus/models/modeling_vlm.py`: drop the 5× class-level `params: AttrDict = {}` (Solution A from TODO_V9_COMPAT.md). Per-instance assignment in `__init__` keeps existing semantics.
- [x] 1.3 Re-run failing test on V8.0 (transformers 4.56.2) — **regression-clean** (4/4 PASS).
- [x] 1.4 Re-run failing test on V9.0 (transformers 5.9.0) — **4/4 PASS** (was 0/4 before fix).
- [x] 1.5 `pip install attrdict` into V9.0_cu126; re-run failing test — **4/4 PASS** (was 0/4 before).
- [x] 1.6 Copy fixed `janus/models/modeling_vlm.py` into V9.0 and V9.0_cu126 `custom_nodes/comfyui_caption_this/`. SHA-256 verified equal to source.
- [x] 1.7 End-to-end probe (`tests/probe_all_envs.py`): `from <plugin>.janus.models import VLChatProcessor, MultiModalityCausalLM` on all three envs — **3/3 PASS**.
- [x] 1.8 Update `requirements.txt`: added `transformers<6.0.0` upper bound + a comment annotating the `attrdict` requirement.
- [x] 1.9 Commit & push (`a21c305`).

### Round 2 — Florence2 forced_bos_token_id AttributeError

- [x] 2.1 Write failing TDD test reproducing the user-reported `AttributeError: ''Florence2LanguageConfig'' object has no attribute ''forced_bos_token_id''` on transformers 5.x — `tests/test_configuration_florence2_v9_compat.py`. 4 sub-tests including the exact `Florence2Config(**config_dict)` call from the user traceback.
- [x] 2.2 Apply minimal fix in `configuration_florence2.py:265`: replace `self.forced_bos_token_id is None` with `getattr(self, "forced_bos_token_id", None) is None`. 1-token change, BC-safe (on transformers 4.x `forced_bos_token_id` is already populated by the legacy `setattr` loop, so the branch is still a no-op there; the warning + backfill path stays intact when the kwarg is supplied).
- [x] 2.3 Re-run failing test on V8.0 (transformers 4.56.2) — **regression-clean** (4/4 PASS).
- [x] 2.4 Re-run failing test on V9.0 (transformers 5.9.0) — **4/4 PASS** (was 1/4 before fix).
- [x] 2.5 Re-run failing test on V9.0_cu126 (transformers 5.9.0 + attrdict) — **4/4 PASS** (was 1/4 before).
- [x] 2.6 Re-run round-1 test (`test_modeling_vlm_v9_compat.py`) on all three envs — still 4/4 (no regression from the Florence2 patch).
- [x] 2.7 Copy fixed `configuration_florence2.py` into all three `custom_nodes/comfyui_caption_this/` (and V8.0''s `ComfyUI-CaptionThis/`). SHA-256 verified equal to source (`02AC4D38F502E772AD3078A45FBB0687421457AEB4E923194ED44DE839FB09BE`).
- [x] 2.8 Extend `tests/probe_all_envs.py` to also exercise the Florence2 path: instantiate `Florence2LanguageConfig()` + `Florence2Config(**with text_config)` per env. **3/3 PASS** with both Janus and Florence2 paths green.
- [x] 2.9 Commit & push (this commit).

## 2. Files Touched

### Round 1 (commit a21c305)

| File | Before | After |
|---|---|---|
| `janus/models/modeling_vlm.py` | 5× `params: AttrDict = {}` at class level | class-level annotation removed; per-instance `__init__` assignment preserved; comment block explaining the why |
| `requirements.txt` | `transformers>=4.39.0,!=4.50.*` | `transformers>=4.39.0,!=4.50.*,<6.0.0` + explanatory comment for `attrdict` |
| `tests/test_modeling_vlm_v9_compat.py` | n/a | new regression test (4 sub-tests) for unit-level Config class load + AutoConfig registration |
| `tests/probe_all_envs.py` | n/a | new cross-environment probe (subprocess per env) for plugin-level import smoke |
| `TODO_V9_COMPAT.md` / `notes/TODO_V9_FIX_TRACK.md` | n/a | planning & tracking |
| V9.0 `custom_nodes/comfyui_caption_this/janus/models/modeling_vlm.py` | unfixed | synced from main |
| V9.0_cu126 `custom_nodes/comfyui_caption_this/janus/models/modeling_vlm.py` | unfixed | synced from main |
| V8.0 `custom_nodes/ComfyUI-CaptionThis/janus/models/modeling_vlm.py` | unfixed | synced from main |

### Round 2 (this commit)

| File | Before | After |
|---|---|---|
| `configuration_florence2.py` | `if self.forced_bos_token_id is None and kwargs.get(...)` (raises AttributeError on transformers 5.x) | `if getattr(self, "forced_bos_token_id", None) is None and kwargs.get(...)` (1-token change; comment block explaining the why) |
| `tests/test_configuration_florence2_v9_compat.py` | n/a | new regression test (4 sub-tests) covering `Florence2LanguageConfig` no-kwarg, BART-CNN backfill path, vision config, and `Florence2Config(**config_dict)` from the user traceback |
| `tests/probe_all_envs.py` | Janus-only probe | Janus + Florence2 paths per env |
| `notes/TODO_V9_FIX_TRACK.md` | round-1 status only | adds round-2 entries |
| V9.0 / V9.0_cu126 `custom_nodes/comfyui_caption_this/configuration_florence2.py` | unfixed | synced from main (SHA-256 `02AC4D38...`) |
| V8.0 `custom_nodes/ComfyUI-CaptionThis/configuration_florence2.py` | unfixed | synced from main |

## 3. Risk / Out-of-Scope

- We do not touch `modeling_florence2.py` further (Kijai commit 0877928 already addressed transformers > 4.52 there).
- We do not modify `janus/janusflow/` — it is dead code in this plugin (no node references it) and has the same issues; will be addressed separately if/when anyone activates a Janus-Flow path.
- We did **not** attempt Solution B (`@dataclass(field(default_factory=AttrDict))` rewrite) on the Janus side — Solution A was the minimal-blast-radius fix and is sufficient.
- Florence2 fix is purely defensive (`getattr` with default) — no behavioural change on transformers 4.x; no behavioural change on 5.x for the *default* path (the warning is suppressed because no caller actually passes `force_bos_token_to_be_generated=True` in modern configs).
