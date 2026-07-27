# Florence-2 V9 (transformers 5.x) gibberish - problem document (for review)

> Summary for reviewer: this is a problem summary of the ComfyUI-CaptionThis plugin in V9 environment
> (transformers 5.9.0) where the Florence-2 model cannot produce correct captions.
> The current commit (64c9d29) only prevents the model from crashing, output is still gibberish.
> Need to fully rewrite the BART-style decoder to fit 5.x cache API.

---

## 0. RESOLVED (2026-07-27) — actual root cause & fix

**The "rewrite the BART decoder" premise in this document was wrong.** The decoder,
cache API, cache_position, 4-D mask, and weight-tying logic were all correct. The
gibberish had a single, mundane root cause found by isolating where V8 and V9
diverge on identical code + identical weights + identical image.

### How it was found (decisive test: teacher-forcing)

Feeding the model its own known-good caption as `labels` and measuring per-position
accuracy. A healthy decoder predicts ~every token; the V9 decoder predicted 0/18.

| Env | teacher-forcing acc | loss | caption |
|---|---|---|---|
| V8 (4.56.2) | **18/18 = 100%** | 0.354 | correct |
| V9 (5.9.0)  | **0/18 = 0.0%**   | ~17   | gibberish |

Since teacher-forcing does not use the cache or the generate loop, this **ruled out
every cache/generation hypothesis in §4 and every option in §6 except A**, and
narrowed the break to the decoder forward itself. Drilling one layer deeper:
`decoder_hidden_states[0]` (the decoder embedding output, before any layer) already
had cosine similarity ~0.007 between V8 and V9 — i.e. the divergence is at the very
first embedding step, not in any attention layer.

### The actual root cause

A full per-parameter audit (`max|loaded - raw_checkpoint|`, cosine) against
`model.safetensors` showed:

- `vision_tower.*`: all 453 params OK (loaded correctly).
- `language_model.*` (the entire BART encoder + decoder + shared + lm_head):
  **all 212 params were random `_init_weights` values** (std 0.02 = `init_std`,
  cosine ~0 with checkpoint). The model that generated captions was effectively
  untrained.

Why: under transformers >= 5.0, `from_pretrained` instantiates on the meta device
then materializes and runs `PreTrainedModel._initialize_weights` -> `_init_weights`
over the whole model. `mark_tied_weights_as_initialized` sets the `_is_hf_initialized`
flag on **parameters** (the tied targets) but not on the **modules** that own them,
and `_initialize_weights` checks the flag at the **module** level. So
`Florence2LanguageModel` / its encoder & decoder layers read as "not initialized"
and `_init_weights` re-randomized them **in place, after** the checkpoint weights
had been loaded — overwriting them. The tying/keys and loading_info (0 missing
keys) all looked correct; the destruction happened silently in the init pass.

### The fix (2 small changes in `modeling_florence2.py`)

1. **`Florence2LanguagePreTrainedModel._init_weights`**: honor the per-parameter
   `_is_hf_initialized` flag — if every direct param/buffer of a module is already
   hf-initialized, skip re-init. This is exactly the remote-code safety pattern
   documented in transformers' own `_initialize_weights` docstring. (The guard is a
   no-op on V8/4.x, where params don't carry the flag, so V8 is unaffected.)
2. **`DaViT.__init__`**: replace `torch.linspace(...).item()` with a pure-Python
   drop-path schedule, mirroring the official microsoft/Florence-2 5.x patch — the
   `.item()` form raises under meta-device instantiation.

### Verification (all 3 envs)

| Env | teacher acc | weights vs ckpt | caption |
|---|---|---|---|
| V8.0      (4.56.2, GPU) | 100% | 665/665 cos>0.5 | correct (matches the V8 line in §1) |
| V9.0      (5.9.0,  GPU) | 72%  | 665/665 cos>0.5 | correct |
| V9.0_cu126(5.9.0,  CPU)| 72%  | 665/665 cos>0.5 | correct |

(V8 hits 100% / V9 72% because V8 ties `lm_head` to `shared` exactly while V9 keeps
the finetune-drifted `lm_head`; both produce correct captions. cu126 runs on CPU
because its `torch 2.12.0+cu126` wheel has no kernels for the RTX 5080's sm_120 —
an environment issue, not a code issue; the code is verified correct on CPU.)

New regression test: `tests/test_modeling_florence2_e2e_caption.py` (weights vs
checkpoint + teacher-forcing acc + generate-not-gibberish). Existing 15/15 suite
still green.

---

## 1. TL;DR

| Env | transformers | Florence-2 status |
|---|---|---|
| V8.0 (E:\FF\ComfyUI_Mie_2026_V8.0) | 4.56.2 | OK runs, output is real caption |
| V9.0 (E:\HH\Package\ComfyUI_Mie_2026_V9.0) | 5.9.0 | FAIL no crash but output is gibberish |
| V9.0_cu126 | 5.9.0 | FAIL same as V9.0 |

V8.0 actual output (correct):
```
A beautiful young woman dressed in traditional chinese attire with floral patterns and pearls
```

V9.0 actual output (gibberish, varies per run):
```
 ent ent ent ent ent ent ent ent ent ent ent ent ent ent ent ent...
```

Or with random seed:
```
adultsurrenceameronMelurrence variable beverage before Sir interested Chrom Rage cer
Chrom simplMel feud beverage AssassinMu ParasMel feudMelighthIng Chromurrence beverage...
```

**Conclusion**: V8.0 is the only env that can produce correct Florence-2 results. V9.0's issue is a fundamental conflict between kijai-era BART decoder code and transformers 5.x new cache API, needs a real decoder rewrite.

---

## 2. Reproduction steps

### 2.1 Setup

```
Python env:
- V9.0: E:\HH\Package\ComfyUI_Mie_2026_V9.0\python_embeded\python.exe  (Python 3.13.12, transformers 5.9.0)
- V8.0: E:\FF\ComfyUI_Mie_2026_V8.0\python_embeded\python.exe  (Python 3.13.11, transformers 4.56.2)
Model: E:\HH\Package\ComfyUI_Mie_2026_V9.0\ComfyUI\models\LLM\Florence-2-base-PromptGen-v2.0
Image: E:\HH\Package\ComfyUI_Mie_2026_V9.0\ComfyUI\input\05eb3c9700b8b3c27732c289318e7b8c.png
       (896x1344, RGB, a traditional dress woman image)
API workflow: C:\Users\administered\Downloads\florence2-api.json
```

### 2.2 Start ComfyUI (V9.0)

```powershell
Start-Process -FilePath "E:\HH\Package\ComfyUI_Mie_2026_V9.0\python_embeded\python.exe" `
    -ArgumentList "E:\HH\Package\ComfyUI_Mie_2026_V9.0\ComfyUI\main.py", `
                  "--listen", "127.0.0.1", "--port", "8188", `
                  "--enable-cors-header", "*", "--disable-auto-launch" `
    -WindowStyle Hidden
```

### 2.3 Submit workflow

```python
import json, urllib.request

with open(r"C:\Users\administered\Downloads\florence2-api.json", "r") as f:
    workflow = json.load(f)

req = urllib.request.Request(
    "http://127.0.0.1:8188/prompt",
    data=json.dumps({"prompt": workflow, "client_id": "review_test"}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=30) as r:
    prompt_id = json.loads(r.read())["prompt_id"]
print(f"submitted: {prompt_id}")
```

### 2.4 Pull output

```python
import time, urllib.request
while True:
    time.sleep(2)
    with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}", timeout=10) as r:
        hist = json.loads(r.read())
    if prompt_id in hist and hist[prompt_id]["status"].get("completed"):
        text = hist[prompt_id]["outputs"]["1"]["text"][0]
        print(f"OUTPUT: {text!r}")
        break
```

---


## 3. Status

### 3.1 Already-fixed crashes

| Commit | What it fixed | Status |
|---|---|---|
| a21c305 | Janus Config params: AttrDict = {} mutable default in 5.x dataclass | OK fixed |
| d20e160 | Florence2LanguageConfig missing forced_bos_token_id | OK fixed |
| cfd1a3b | _tied_weights_keys list->dict (5.x contract) | OK fixed |
| 5f65ea6 | prepare_inputs_for_generation past_key_values[0][0].shape[2] 4.x-only | OK fixed |
| 96bc9bd | DecoderLayer/Decoder cache 4.x vs 5.x dispatch + _reorder_cache 5.x | WARN no crash but gibberish |
| 64c9d29 | florence2_caption.py pre-resize PIL 768x768 + use_cache=False fallback | WARN no crash but gibberish |

### 3.2 Still-failing part (core problem)

V9.0 Florence-2 generates gibberish. Confirmed via:
1. use_cache=True (default) - one token repeats hundreds of times
2. use_cache=False - still gibberish
3. Force tie lm_head to shared - still gibberish (moot moot moot)
4. Single forward (use_cache=False + decoder_input_ids=[[0]]) - first token is still gibberish
5. Identical kijai code + model weights: V8.0 (transformers 4.56.2) = real caption, V9.0 (5.9.0) = gibberish

-> Root cause is NOT the kijai code per se, but transformers 5.x breaking changes.

---

## 4. transformers 5.x breaking changes (suspected root cause, needs verification)

Florence-2's decoder is a BART-style decoder from Microsoft Florence-2 repo. It assumes 4.x-era invariants:

### 4.1 EncoderDecoderCache replaces tuple-of-tuples

5.x changed past_key_values from (layer0_4tuple, layer1_4tuple, ...) to EncoderDecoderCache object with:
```
cache.self_attention_cache.layers[layer_idx].keys   # 3D: (heads, seq, head_dim) - no batch dim
cache.self_attention_cache.layers[layer_idx].values
cache.cross_attention_cache.layers[layer_idx].keys
cache.cross_attention_cache.layers[layer_idx].values
```

BART decoder k/v concat (torch.cat([past_key_value[0], key_states], dim=2)) gets past in 3D, new k/v in 4D - shape mismatch.

### 4.2 cache_position parameter

5.x's GenerationMixin._prefill passes cache_position (shape (batch, current_seq_len)) to all model.forward calls. BART-style decoder does not know this parameter, does not thread it to attention mask computation - subsequent step attention mask is wrong.

### 4.3 4-D attention mask format

5.x attention mask is (batch, 1, tgt_len, src_len) computed via transformers.masking_utils. BART-style uses _prepare_4d_causal_attention_mask which is deprecated - mask shape/meaning is not fully compatible.

### 4.4 _tied_weights_keys truly 'does not tie' weights

5.x in mark_tied_weights_as_initialized checks saved weights have different values - decides NOT to tie. kijai's _tie_weights() override is not called in 5.x from_pretrained flow (or timing is wrong), causing lm_head.weight to keep original value instead of cloning from shared.weight.

(I tried manually copy-tying in the plugin and confirmed 2-step cache output matches no-cache, but model.generate full multi-step still produces gibberish - so this is not the root cause, just a contributor.)

---

## 5. Tried approaches (all failed)

| Approach | What it changed | Result |
|---|---|---|
| Convert 5.x cache to 4.x tuple, run 4.x attention | Detect hasattr(self_attention_cache) | First run: TypeError (3D vs 4D shape mismatch). After fix: gibberish. |
| cache.update() + is_updated dict reuse | attention 5.x branch | Still gibberish |
| use_cache=False (kijai original fallback) | plugin first generate call | Still gibberish |
| Force tie lm_head to shared | model.lm_head.weight.copy_(model.shared.weight) | Still gibberish (moot moot moot) |
| Pre-resize PIL to 768x768 | florence2_caption.py | OK fixed a separate non-square crash, did not solve gibberish |
| Single forward check logits | _encode_image then 1 step | top-1 is random English word not a beautiful |

All these are symptoms. Root cause is 5.x decoder adaptation.

---

## 6. Real fix options (need review)

### 6.1 Option A: Port BART decoder to transformers 5.x (recommended, but large work)

Steps:
1. Rewrite Florence2Attention.forward to fully support 5.x EncoderDecoderCache, including:
   - cache.update(key_states, value_states, layer_idx) return value handling
   - cache_position parameter passed to attention mask
   - 5.x 4-D attention mask format
2. Rewrite Florence2Decoder.forward to use 5.x causal_mask function
3. Rewrite Florence2DecoderLayer.forward to use 5.x attention call
4. Verify _tied_weights_keys truly ties weights in 5.x (manual tie or monkey-patch mark_tied_weights_as_initialized)
5. Run tests/test_modeling_florence2_round6_cache.py (currently passing) but ADD an end-to-end generate produces real caption assertion (most important)

Risk: 4.x compatibility might break. Need to maintain both 4.x/5.x paths.

### 6.2 Option B: use_cache=False fallback, accept slow output

Current (commit 64c9d29) is use_cache=False. But STILL outputs gibberish. The issue is not cache speed but decoder and 5.x overall compatibility. This option CANNOT fix output.

### 6.3 Option C: Downgrade transformers to 4.x in V9.0

```
```

Risk: may break V9.0's other models (Janus is already 5.x-adapted). Need full regression test.

### 6.4 Option D: Disable V9.0's Florence-2 entirely

Simplest. Change florence2_caption.py to raise NotImplementedError(Florence-2 on V9 requires transformers 4.x; please use V8.0). **Recommended short-term solution.**

### 6.5 Option E: Fork a V9-specific version

Replace kijai's BART decoder entirely with transformers 5.x official BART decoder (use _attn_implementation=sdpa). This needs to review whether 5.x official BART is actually compatible with Florence-2 (Florence-2 uses image_pos_embed to project image embeddings, may not be pure BART).

---

## 7. Key code locations

| File | Line | Content |
|---|---|---|
| florence2_caption.py:80-130 | describe_single_image() - user entry; pre-resize + use_cache=False added |
| florence2_caption.py:185-220 | Florence2ModelLoader.load_model() - model loading |
| modeling_florence2.py:740-950 | Florence2Attention.forward - k/v concat logic (4.x tuple assumption) |
| modeling_florence2.py:950-1180 | Florence2FlashAttention2/2Sdpa - early-return to super for 5.x cache |
| modeling_florence2.py:1402-1540 | Florence2DecoderLayer.forward - self-attn + encoder-attn + 4-tuple combine |
| modeling_florence2.py:1647-2080 | Florence2Decoder.forward - decoder loop, per-layer cache accumulation |
| modeling_florence2.py:2274-2290 | Florence2LanguageForConditionalGeneration._tie_weights() - should be called but not effective |
| modeling_florence2.py:2397-2450 | prepare_inputs_for_generation() - past_key_values shape access |
| modeling_florence2.py:2918-2990 | Florence2ForConditionalGeneration.forward() - image + text embed combine |
| modeling_florence2.py:3025-3050 | Florence2ForConditionalGeneration.generate() - passes inputs_embeds to language_model.generate |
| configuration_florence2.py:265 | forced_bos_token_id access (round 2 fix) |

---

## 8. Key tests

| Test | Status | What it tests |
|---|---|---|
| tests/test_modeling_vlm_v9_compat.py | OK 4/4 pass | Janus Config dataclass mutable-default fix |
| tests/test_configuration_florence2_v9_compat.py | OK 4/4 pass | forced_bos_token_id fix |
| tests/test_modeling_florence2_v9_compat.py | OK 5/5 pass | _tied_weights_keys dict format |
| tests/test_modeling_florence2_round6_cache.py | OK 4/4 pass | Florence2DecoderLayer cache slice no TypeError |
| tests/test_florence2_caption_v9_compat.py | OK 3/3 pass | describe_single_image pre-resize |
| tests/probe_all_envs.py | OK 3/3 pass | All 3 envs can import plugin |

Missing: end-to-end generate producing real caption test. This is the most critical missing test - without it, gibberish slips through CI.

---

## 9. 3 issues I suggest reviewer focus on

1. **On V9.0 single forward (model(input_ids=..., pixel_values=..., decoder_input_ids=[[0]])) first token is already callback/rapt/Moreover instead of A/a. This means image features are not being effectively used** - 5.x inputs_embeds to BART encoder connection may have deeper issues, not just cache.

2. **5.x cache_position must be properly threaded before first forward**. kijai Florence2LanguageForConditionalGeneration.forward does not accept this parameter. Check modeling_florence2.py:2316 nearby forward() signature - should it add cache_position=None and pass it to _prepare_4d_causal_attention_mask?

3. **_tied_weights_keys round 3/4 fix may not actually be effective**. 5.x in mark_tied_weights_as_initialized skips tying (because saved weights are different), but this is user-expected behavior. Question is whether kijai _tie_weights() override is actually called during loading, need to add log to confirm.

---

## 10. Key commits

- 5f65ea6 Round 5: prepare_inputs_for_generation + past_key_values_length
- 96bc9bd Round 6+7: BART cache API + image processor pre-resize (**verified to only fix crash, not output**)
- 64c9d29 Round 7+ followup: keep image resize fix, remove useless cache shim + weight tying, document gibberish problem

Please focus review on the 6 options above. I lean toward **6.1 (rewrite BART decoder)** or **6.4 (short-term disable)**, but 6.1 is large work and may break 4.x compatibility.

---
```python
import os, sys
import torch
from PIL import Image
import importlib.util, types

PLUGIN = 'E:/HH/Package/ComfyUI_Mie_2026_V9.0/ComfyUI/custom_nodes/comfyui_caption_this'
sys.path.insert(0, PLUGIN)
PARENT = types.ModuleType('florence2_plugin')
PARENT.__path__ = [PLUGIN]
sys.modules['florence2_plugin'] = PARENT
for sub in ('configuration_florence2', 'modeling_florence2'):
    spec = importlib.util.spec_from_file_location(
        'florence2_plugin.' + sub,
        os.path.join(PLUGIN, sub + '.py'),
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules['florence2_plugin.' + sub] = m
    spec.loader.exec_module(m)
Florence2ForConditionalGeneration = sys.modules['florence2_plugin.modeling_florence2'].Florence2ForConditionalGeneration

MODEL = 'E:/HH/Package/ComfyUI_Mie_2026_V9.0/ComfyUI/models/LLM/Florence-2-base-PromptGen-v2.0'
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Florence2ForConditionalGeneration.from_pretrained(MODEL, attn_implementation='sdpa', torch_dtype=torch.float16).to(device).eval()

img = Image.open('E:/HH/Package/ComfyUI_Mie_2026_V9.0/ComfyUI/input/05eb3c9700b8b3c27732c289318e7b8c.png').convert('RGB')
img_resized = img.resize((768, 768), resample=3)
inputs = processor(text="[CAPTION]", images=img_resized, return_tensors="pt", do_resize=False, do_rescale=False).to(device, torch.float16)

with torch.no_grad():
    out = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=64, do_sample=False, num_beams=1,
    )
print(processor.batch_decode(out, skip_special_tokens=False)[0])
```

Note: The task is 'more_detailed_caption' in the API workflow. Above I use 'CAPTION' for simplicity. Either works for showing the model is broken.
