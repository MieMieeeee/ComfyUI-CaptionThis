# ComfyUI-CaptionThis V9 Compatibility Fix — Tracking

> Status legend: `[ ]` todo · `[~]` in-progress · `[x]` done · `[-]` skip / not applicable

## 0. Diagnosis Snapshot

| Env | Python | transformers | attrdict | Plugin path | Pre-fix status |
|---|---|---|---|---|---|
| `E:\FF\ComfyUI_Mie_2026_V8.0` | 3.13.11 | 4.56.2 | installed (broken `Mapping`) | `ComfyUI\custom_nodes\ComfyUI-CaptionThis\` | All paths ✅ |
| `E:\HH\Package\ComfyUI_Mie_2026_V9.0` | 3.13.12 | **5.9.0** | installed (broken `Mapping`) | `ComfyUI\custom_nodes\comfyui_caption_this\` | ❌ All three: Janus dataclass, Florence2 config `forced_bos_token_id`, Florence2 modeling `_tied_weights_keys` |
| `E:\HH\Package\ComfyUI_Mie_2026_V9.0_cu126` | 3.12.10 | **5.9.0** | **not installed** (now installed) | `ComfyUI\custom_nodes\comfyui_caption_this\` | ❌ All three: Janus dataclass + attrdict missing, Florence2 config, Florence2 modeling |

## 1. Tasks

### Round 1 — Janus dataclass mutable-default (commit a21c305)

- [x] 1.1-1.9 (committed in `a21c305`; full details in git log).

### Round 2 — Florence2 `forced_bos_token_id` AttributeError (commit d20e160)

- [x] 2.1-2.9 (committed in `d20e160`; full details in git log).

### Round 3 — Florence2 `_tied_weights_keys` list→dict (this commit)

- [x] 3.1 Write failing TDD test reproducing
      `AttributeError: ''list'' object has no attribute ''keys''` raised in
      `PreTrainedModel.get_expanded_tied_weights_keys` (transformers 5.x)
      when `modeling_florence2.py` declared the three `_tied_weights_keys` as
      flat lists — `tests/test_modeling_florence2_v9_compat.py` (4 sub-tests:
      structural dict check, regex fast-path, BART-style tying semantics,
      direct call into the failing code path). 0/4 PASS pre-fix on V9.0.
- [x] 3.2 Apply minimal fix: convert the 3 list-form `_tied_weights_keys` in
      `modeling_florence2.py` to dict form `{"target.weight": "source.weight"}`,
      matching the BART convention (decoder.embed_tokens + lm_head both tied
      to encoder.embed_tokens). 4/4 PASS post-fix on V9.0.
- [x] 3.3 Re-run on V8.0 (transformers 4.56.2). 3 PASS + 1 SKIP
      (`get_expanded_tied_weights_keys` does not exist in 4.x so the
      behavioural test is a no-op there; structural tests cover the BC fix).
- [x] 3.4 Re-run on V9.0_cu126 (transformers 5.9.0 + attrdict). 4/4 PASS.
- [x] 3.5 Re-run round-1 test (`test_modeling_vlm_v9_compat.py`) on all three
      envs: still 4/4 each.
- [x] 3.6 Re-run round-2 test (`test_configuration_florence2_v9_compat.py`)
      on all three envs: still 4/4 each.
- [x] 3.7 Copy fixed `modeling_florence2.py` into all three `custom_nodes/`
      plugin installs (V9.0, V9.0_cu126 `comfyui_caption_this/`, V8.0
      `ComfyUI-CaptionThis/`). SHA-256 verified equal to source
      (`25ED26DC0F1A4CD02F60EF8742955E815A8ACE508A03FA3FB95CFA95F646E14D`).
- [x] 3.8 Extend `tests/probe_all_envs.py` to also assert that the 3
      `_tied_weights_keys` are dicts (covers both the structural and
      behavioural checks in one go). 3/3 PASS on all envs.
- [x] 3.9 Commit & push (this commit).

## 2. Files Touched

### Round 1 (commit a21c305) — Janus dataclass
`janus/models/modeling_vlm.py`, `requirements.txt`, `tests/test_modeling_vlm_v9_compat.py`,
`tests/probe_all_envs.py`, `TODO_V9_COMPAT.md`, `notes/TODO_V9_FIX_TRACK.md`.

### Round 2 (commit d20e160) — Florence2 config
`configuration_florence2.py`, `tests/test_configuration_florence2_v9_compat.py`,
`tests/probe_all_envs.py`, `notes/TODO_V9_FIX_TRACK.md`.

### Round 3 (this commit) — Florence2 modeling

| File | Before | After |
|---|---|---|
| `modeling_florence2.py` | 3× `_tied_weights_keys = ["..."]` (flat lists) | 3× `_tied_weights_keys = {"target": "source"}` (BART-style dict tying); comment block explaining the why |
| `tests/test_modeling_florence2_v9_compat.py` | n/a | new regression test (4 sub-tests) for the dict contract + direct call into the failing code path |
| `tests/probe_all_envs.py` | Janus + Florence2 config probe | + structural check that all 3 `_tied_weights_keys` are dicts |
| `notes/TODO_V9_FIX_TRACK.md` | round-2 status only | adds round-3 entries |
| V9.0 / V9.0_cu126 / V8.0 plugin installs | unfixed | synced from main (SHA-256 `25ED26DC...E14D`) |

## 3. Risk / Out-of-Scope

- The 3 dict mappings preserve the BART convention: `decoder.embed_tokens` and
  `lm_head` both tied to `encoder.embed_tokens`. This matches the original
  intent of the list (every item in the list is a tied key whose canonical
  source is the encoder embeddings). No semantic change in tying behaviour
  for end-users; only the data structure passed to transformers changed.
- We did NOT refactor to use kijai''s `tie_weights()` method (custom override
  in `kijai/ComfyUI-Florence2/model/model.py:288-290`); the dict approach is
  the lower-blast-radius fix and stays compatible with both 4.x and 5.x.
- `modeling_florence2.py` Kijai''s `_supports_flash_attn_2` /
  `_supports_sdpa` property guards (commit 0877928) are unchanged.
