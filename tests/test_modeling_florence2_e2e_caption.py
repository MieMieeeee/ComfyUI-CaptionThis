"""
End-to-end regression test: Florence-2 must load real weights AND produce a
real caption (not gibberish) under transformers 5.x.

WHY THIS TEST EXISTS
--------------------
Before this test, the v9-compat suite only exercised the cache / config / tying
*mechanisms* with stub modules. It never loaded the real Florence-2 checkpoint,
so a silent regression where the model loads but emits gibberish slipped
through CI for 7 rounds (see notes/TODO_V9_FLORENCE2_5X_REVIEW.md).

The actual root cause turned out to be: under transformers >= 5.0,
`from_pretrained` instantiates on the meta device, then
`PreTrainedModel._initialize_weights` re-runs `_init_weights` on modules whose
`_is_hf_initialized` flag was set on *parameters* (by
`mark_tied_weights_as_initialized`) but not on the owning *module*. The BART
encoder/decoder/embeddings were therefore re-randomized in-place AFTER being
loaded, producing gibberish (teacher-forcing accuracy 0%, loss ~17).

The fix: `Florence2LanguagePreTrainedModel._init_weights` now honors the
per-parameter `_is_hf_initialized` flag (the remote-code safety pattern
documented in transformers' own `_initialize_weights` docstring) and skips
re-init when the module's params were already loaded.

This test guards against that whole class of regression by checking the three
things that collectively prove the model is correct:
  1. The loaded `language_model` weights match the checkpoint (not re-randomized).
  2. Teacher-forcing accuracy on a known caption is high (>50%), loss low.
  3. Free-running generation produces a caption containing expected keywords,
     NOT the gibberish signature (repeated single token / random words).

GATING
------
Loading the real Florence-2 checkpoint is heavy (~1GB, needs the model dir).
This test is skipped unless the caller points at a model via either:
  - env var  FLORENCE2_MODEL_PATH=/path/to/Florence-2-...
  - or a default location relative to a ComfyUI deploy, resolved below.

Run manually:
    set FLORENCE2_MODEL_PATH=E:\\HH\\Package\\ComfyUI_Mie_2026_V9.0\\ComfyUI\\models\\LLM\\Florence-2-base-PromptGen-v2.0
    python tests/test_modeling_florence2_e2e_caption.py

Must pass on:
  * transformers 4.56.2 (V8.0)
  * transformers 5.9.0  (V9.0, V9.0_cu126)  <-- the env the bug was seen on
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import torch


# --------------------------------------------------------------------------- #
# Model path resolution + skip gate
# --------------------------------------------------------------------------- #
def _resolve_model_path():
    p = os.environ.get("FLORENCE2_MODEL_PATH", "").strip()
    if p and os.path.isdir(p) and os.path.exists(os.path.join(p, "model.safetensors")):
        return p
    # default deploy locations (best-effort)
    for cand in (
        r"E:/HH/Package/ComfyUI_Mie_2026_V9.0/ComfyUI/models/LLM/Florence-2-base-PromptGen-v2.0",
        r"E:/HH/Package/ComfyUI_Mie_2026_V9.0_cu126/ComfyUI/models/LLM/Florence-2-base-PromptGen-v2.0",
    ):
        if os.path.exists(os.path.join(cand, "model.safetensors")):
            return cand
    return None


MODEL_PATH = _resolve_model_path()
HAS_REAL_MODEL = MODEL_PATH is not None

# A small constant image that ships with the deploys, used if present.
def _resolve_image_path():
    for cand in (
        os.environ.get("FLORENCE2_TEST_IMAGE", ""),
        r"E:/HH/Package/ComfyUI_Mie_2026_V9.0/ComfyUI/input/05eb3c9700b8b3c27732c289318e7b8c.png",
        r"E:/HH/Package/ComfyUI_Mie_2026_V9.0_cu126/ComfyUI/input/05eb3c9700b8b3c27732c289318e7b8c.png",
    ):
        if cand and os.path.exists(cand):
            return cand
    return None


IMAGE_PATH = _resolve_image_path()

if not HAS_REAL_MODEL:
    print("SKIP test_modeling_florence2_e2e_caption: no real Florence-2 model found.")
    print("     Set FLORENCE2_MODEL_PATH=... to enable. This is the most important")
    print("     regression test for the V9 gibberish bug; run it before releasing.")
    sys.exit(0)


# --------------------------------------------------------------------------- #
# Load the plugin's bundled modeling_florence2 (same path the plugin uses
# on transformers >= 4.51.0 in florence2_caption.py).
# --------------------------------------------------------------------------- #
def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(d)


_REPO_ROOT = _repo_root()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Provide a synthetic parent package so the relative imports inside
# modeling_florence2 (`from .configuration_florence2 import ...`) resolve.
_PARENT = types.ModuleType("comfyui_caption_this_test_pkg")
_PARENT.__path__ = [_REPO_ROOT]
sys.modules["comfyui_caption_this_test_pkg"] = _PARENT
for _sub in ("configuration_florence2", "modeling_florence2"):
    _spec = importlib.util.spec_from_file_location(
        "comfyui_caption_this_test_pkg." + _sub, os.path.join(_REPO_ROOT, _sub + ".py")
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["comfyui_caption_this_test_pkg." + _sub] = _mod
    _spec.loader.exec_module(_mod)
M = sys.modules["comfyui_caption_this_test_pkg.modeling_florence2"]

import transformers  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# On an env whose torch wheel lacks kernels for the local GPU (e.g. cu126 wheel
# on an RTX 5080 / sm_120), CUDA ops raise "no kernel image". Fall back to CPU
# so the CODE correctness is still asserted; the wheel/GPU mismatch is an
# environment issue, not a plugin regression.
def _safe_cuda_available():
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        print("NOTE: CUDA present but no kernel image for this GPU; running on CPU.")
        return False

if not _safe_cuda_available():
    DEVICE = "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


print(f"transformers: {transformers.__version__}  torch: {torch.__version__}")
print(f"model:  {MODEL_PATH}")
print(f"device: {DEVICE}  dtype: {DTYPE}")

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = M.Florence2ForConditionalGeneration.from_pretrained(
    MODEL_PATH, attn_implementation=("sdpa" if DEVICE == "cuda" else "eager"),
    torch_dtype=DTYPE,
).to(DEVICE).eval()

failures = []


# --------------------------------------------------------------------------- #
# TEST 1: loaded language_model weights match the checkpoint
# --------------------------------------------------------------------------- #
def test_weights_match_checkpoint():
    """The BART encoder/decoder/embeddings must NOT be re-randomized by _init_weights.

    "Re-randomized" means the loaded value looks like fresh `_init_weights` output
    (cosine similarity ~0 with the checkpoint value). A small max-diff is fine and
    expected: fp16 rounding (~1e-3), AND the legitimate tied-embedding case where
    `lm_head.weight` is deliberately set to the (finetune-drifted) `shared.weight`
    rather than the separately-stored `lm_head` checkpoint value — for a finetuned
    tied model these legitimately differ by the finetune drift (~0.03 here).
    We catch only the actual regression: params that became uncorrelated noise.
    """
    with safe_open(os.path.join(MODEL_PATH, "model.safetensors"), framework="pt") as f:
        raw = {k: f.get_tensor(k) for k in f.keys()}

    bad = []
    for name, p in model.named_parameters(remove_duplicate=False):
        if name not in raw:
            continue
        rp = raw[name].to(p.device).to(p.dtype)
        if rp.shape != p.shape:
            continue
        lp = p.float().flatten(); rp_f = rp.float().flatten()
        cos = torch.dot(lp, rp_f).item() / (lp.norm().item() * rp_f.norm().item() + 1e-9)
        # Before the fix: cos ~ 0 (uncorrelated random). After: cos ~ 1.0.
        # Threshold 0.5 is very lenient (real values are >0.99 or ==1.0).
        if cos < 0.5:
            bad.append((cos, name))
    n_checked = sum(1 for n, p in model.named_parameters(remove_duplicate=False) if n in raw)
    msg = f"params uncorrelated with checkpoint (cos<0.5): {len(bad)}/{n_checked}"
    if bad:
        worst = sorted(bad)[:3]
        msg += "; worst: " + ", ".join(f"{n}(cos={c:.3f})" for c, n in worst)
        failures.append(f"test_weights_match_checkpoint: {msg}")
    print(f"[{'PASS' if not bad else 'FAIL'}] test_weights_match_checkpoint: {msg}")


# --------------------------------------------------------------------------- #
# TEST 2: teacher-forcing accuracy on a known caption is high
# --------------------------------------------------------------------------- #
# The known-good V8 caption for the bundled test image (review doc). Even under
# fp16 we expect the great majority of positions to be predicted correctly.
GOLD_CAPTION = "A beautiful young woman dressed in traditional chinese attire with floral patterns and pearls"


def test_teacher_forcing_accuracy():
    if IMAGE_PATH is None:
        msg = "no test image; skipped"
        print(f"[SKIP] test_teacher_forcing_accuracy: {msg}")
        return
    from PIL import Image
    img = Image.open(IMAGE_PATH).convert("RGB").resize((768, 768), resample=3)
    enc = processor(text="<CAPTION>", images=img, return_tensors="pt",
                    do_resize=False, do_rescale=False).to(DTYPE).to(DEVICE)
    gold = processor.tokenizer(GOLD_CAPTION, return_tensors="pt",
                               add_special_tokens=True)["input_ids"].to(DEVICE)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    pixel_values=enc["pixel_values"], labels=gold, use_cache=False)
        acc = (out.logits.argmax(-1) == gold).float().mean().item()
        loss = out.loss.item()
    # Before the fix: acc ~0%, loss ~17. After: acc >50%, loss <2.5.
    ok = acc > 0.5 and loss < 2.5
    msg = f"acc={acc*100:.1f}% loss={loss:.3f} (require acc>50%, loss<2.5)"
    if not ok:
        failures.append(f"test_teacher_forcing_accuracy: {msg} -> looks like gibberish/re-init regression")
    print(f"[{'PASS' if ok else 'FAIL'}] test_teacher_forcing_accuracy: {msg}")


# --------------------------------------------------------------------------- #
# TEST 3: free-running generation produces a sensible caption (not gibberish)
# --------------------------------------------------------------------------- #
# Gibberish signatures observed during the bug:
#   " ent ent ent ent ..."            (single token repeated)
#   "adultsurrenceameronMelurrence..." (random tokens)
# A correct caption mentions woman / attire / floral / traditional etc.
KEYWORDS = ("woman", "attire", "floral", "traditional", "dress", "chinese", "asian",
            "pearl", "young", "portrait")


def test_generation_is_not_gibberish():
    if IMAGE_PATH is None:
        msg = "no test image; skipped"
        print(f"[SKIP] test_generation_is_not_gibberish: {msg}")
        return
    from PIL import Image
    img = Image.open(IMAGE_PATH).convert("RGB").resize((768, 768), resample=3)
    enc = processor(text="<CAPTION>", images=img, return_tensors="pt",
                    do_resize=False, do_rescale=False).to(DTYPE).to(DEVICE)
    with torch.no_grad():
        gen = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                             pixel_values=enc["pixel_values"], max_new_tokens=48,
                             do_sample=False, num_beams=1, use_cache=False)
    text = processor.batch_decode(gen, skip_special_tokens=True)[0].lower()
    # gibberish heuristics
    tokens = text.split()
    repeats = max((tokens.count(t) for t in set(tokens)), default=0)
    keyword_hits = sum(1 for k in KEYWORDS if k in text)
    ok = keyword_hits >= 2 and repeats < 8 and len(tokens) >= 4
    msg = f"keyword_hits={keyword_hits} max_repeat={repeats} text={text!r}"
    if not ok:
        failures.append(f"test_generation_is_not_gibberish: {msg} -> gibberish signature")
    print(f"[{'PASS' if ok else 'FAIL'}] test_generation_is_not_gibberish: {msg}")


if __name__ == "__main__":
    test_weights_match_checkpoint()
    test_teacher_forcing_accuracy()
    test_generation_is_not_gibberish()
    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S)")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("RESULT: ALL PASS")
