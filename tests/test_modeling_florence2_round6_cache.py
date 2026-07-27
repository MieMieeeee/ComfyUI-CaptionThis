"""
Round-6 TDD: Florence-2 BART 5.x Cache API round-trip.

Symptom (post-round-5, all 3 envs still failing):
    File ".../modeling_florence2.py", line 1262, in forward
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
    TypeError: expected Tensor as element 0 in argument 0, but got tuple

Root cause (see notes/TODO_V9_ROUND6_FIX.md):
  * Florence2Decoder.forward dropped the per-layer `[idx]` slice for 4.x.
  * Florence2DecoderLayer.forward had a reversed ternary on the
    past_key_value argument to self_attn / encoder_attn.

This test exercises the smallest possible end-to-end generation that hits
both branches, without needing a real Florence-2 vision tower.

The test MUST pass on:
  * transformers 4.56.2 (V8.0)  -> 4.x path: tuple-of-tuples
  * transformers 5.9.0 (V9.0, V9.0_cu126) -> 5.x path: EncoderDecoderCache
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
import traceback

import torch.nn as nn


class _ModuleForwardSpy(nn.Module):
    """Wraps a child nn.Module and records what past_key_value is passed
    to its forward().
    """

    def __init__(self, wrapped):
        super().__init__()
        self._wrapped = wrapped
        self.captured = {}

    def forward(self, *args, **kwargs):
        self.captured["past_key_value"] = kwargs.get("past_key_value", None)
        return self._wrapped(*args, **kwargs)


def _repo_root():
    d = __file__
    for _ in range(2):
        d = d.rsplit(chr(92), 1)[0] if chr(92) in d else d.rsplit("/", 1)[0]
    return d


_REPO_ROOT = _repo_root()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_florence2_modules():
    PKG = "_florence2_round6"
    for n in list(sys.modules):
        if n == PKG or n.startswith(PKG + "."):
            sys.modules.pop(n, None)
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [_REPO_ROOT]
    sys.modules[PKG] = pkg
    spec_cf = importlib.util.spec_from_file_location(
        f"{PKG}.configuration_florence2",
        os.path.join(_REPO_ROOT, "configuration_florence2.py"),
    )
    cf = importlib.util.module_from_spec(spec_cf)
    sys.modules[f"{PKG}.configuration_florence2"] = cf
    spec_cf.loader.exec_module(cf)
    spec_mf = importlib.util.spec_from_file_location(
        f"{PKG}.modeling_florence2",
        os.path.join(_REPO_ROOT, "modeling_florence2.py"),
    )
    mf = importlib.util.module_from_spec(spec_mf)
    sys.modules[f"{PKG}.modeling_florence2"] = mf
    spec_mf.loader.exec_module(mf)
    return cf, mf


def _build_tiny_language_model(mf, cf):
    """Build the smallest viable language model for round-tripping the cache."""
    import torch
    cfg = cf.Florence2LanguageConfig(
        vocab_size=128,
        d_model=32,
        encoder_layers=1,
        encoder_ffn_dim=32,
        encoder_attention_heads=2,
        decoder_layers=1,
        decoder_ffn_dim=32,
        decoder_attention_heads=2,
        num_hidden_layers=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        activation_dropout=0.0,
        dropout=0.0,
        activation_function="gelu",
        tie_word_embeddings=True,
        _attn_implementation="eager",
    )
    with torch.no_grad():
        model = mf.Florence2LanguageForConditionalGeneration(cfg)
    model.eval()
    return model, cfg


def test_generate_second_step_does_not_raise_typeerror():
    """Round-6 bug: the 2nd decode step in `model.generate` raises
    `TypeError: expected Tensor as element 0 in argument 0, but got tuple`
    because `past_key_value[0]` is a 4-tuple instead of a tensor.

    We assert that model.generate succeeds for max_new_tokens=4 and returns
    a non-empty Tensor.
    """
    import torch
    cf, mf = _load_florence2_modules()
    model, cfg = _build_tiny_language_model(mf, cf)
    input_ids = torch.tensor([[1, 5, 7, 9, 11]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=4,
                num_beams=1,
                do_sample=False,
            )
    except TypeError as exc:
        if "got tuple" in str(exc) or "Tensor as element 0" in str(exc):
            raise AssertionError(
                f"round-6 Cache bug regressed: {type(exc).__name__}: {exc}"
            ) from exc
        raise
    assert out is not None, "model.generate returned None"
    assert isinstance(out, torch.Tensor), f"expected Tensor, got {type(out).__name__}"
    assert out.dim() == 2, f"expected (batch, seq), got {out.shape}"
    assert out.shape[0] == 1, f"expected batch=1, got {out.shape[0]}"
    assert out.numel() > 0, "model.generate produced empty output"
    print(f"[PASS] test_generate_second_step_does_not_raise_typeerror out shape={tuple(out.shape)}")


def test_decoder_layer_self_attn_past_key_value_is_not_none_for_5x_cache():
    """Layer-level sanity check: when the layer is given a 5.x Cache object,
    the layer MUST pass the cache (not None) into the self-attention call.

    The original bug: `past_key_value=self_attn_past_key_value if is_cache_object else past_key_value`
    was reversed, so on 5.x attention saw `past_key_value=None`.
    """
    import torch
    cf, mf = _load_florence2_modules()
    model, cfg = _build_tiny_language_model(mf, cf)
    decoder_layer = model.model.decoder.layers[0]

    real_self_attn = decoder_layer.self_attn
    spy = _ModuleForwardSpy(real_self_attn)
    decoder_layer.self_attn = spy

    # Build a tiny 5.x-like Cache object with the attributes the layer probes.
    class _FakeCache:
        def __init__(self):
            self._updates = []
        @property
        def self_attention_cache(self):
            return self
        @property
        def cross_attention_cache(self):
            return self
        def get_seq_length(self):
            return 0
        def update(self, k, v, layer_idx):
            self._updates.append((layer_idx, k, v))
            return k, v
        is_updated = {}

    cache = _FakeCache()
    hidden = torch.randn(1, 1, cfg.d_model)
    try:
        with torch.no_grad():
            decoder_layer(
                hidden_states=hidden,
                attention_mask=None,
                encoder_hidden_states=hidden,
                encoder_attention_mask=None,
                past_key_value=cache,
                use_cache=True,
            )
    finally:
        decoder_layer.self_attn = real_self_attn

    saw = spy.captured.get("past_key_value")
    assert saw is cache, (
        f"decoder layer dropped the 5.x Cache: passed to self_attn as {type(saw).__name__}, "
        f"expected the cache object. This is the reversed-ternary bug."
    )
    print("[PASS] test_decoder_layer_self_attn_past_key_value_is_not_none_for_5x_cache")


def test_decoder_layer_self_attn_past_key_value_is_slice_for_4x_tuple():
    """The 4.x path: the layer is given a 4-tuple per-layer cache
    (self_attn_k, self_attn_v, cross_attn_k, cross_attn_v). The layer MUST
    pass the (self_attn_k, self_attn_v) slice, not the whole 4-tuple, to
    the self-attention call.
    """
    import torch
    cf, mf = _load_florence2_modules()
    model, cfg = _build_tiny_language_model(mf, cf)
    decoder_layer = model.model.decoder.layers[0]

    real_self_attn = decoder_layer.self_attn
    spy = _ModuleForwardSpy(real_self_attn)
    decoder_layer.self_attn = spy

    # Build a per-layer 4-tuple (bsz, heads, seq_len, d_head) shaped tensors.
    bsz, heads, seq, hd = 1, 2, 1, cfg.d_model // cfg.decoder_attention_heads
    self_k = torch.randn(bsz, heads, seq, hd)
    self_v = torch.randn(bsz, heads, seq, hd)
    cross_k = torch.randn(bsz, heads, seq, hd)
    cross_v = torch.randn(bsz, heads, seq, hd)
    present_4tuple = (self_k, self_v, cross_k, cross_v)

    hidden = torch.randn(1, 1, cfg.d_model)
    try:
        with torch.no_grad():
            decoder_layer(
                hidden_states=hidden,
                attention_mask=None,
                encoder_hidden_states=hidden,
                encoder_attention_mask=None,
                past_key_value=present_4tuple,
                use_cache=True,
            )
    finally:
        decoder_layer.self_attn = real_self_attn

    saw = spy.captured.get("past_key_value")
    if saw is not None:
        assert isinstance(saw, tuple), f"expected tuple, got {type(saw).__name__}"
        assert len(saw) == 2, f"expected 2-tuple (self_k, self_v), got len={len(saw)}"
        # The first 2 elements of the 4-tuple should be passed through.
        assert torch.equal(saw[0], self_k), "slice[0] should be self_attn_k"
        assert torch.equal(saw[1], self_v), "slice[1] should be self_attn_v"
    print("[PASS] test_decoder_layer_self_attn_past_key_value_is_slice_for_4x_tuple")


def test_decoder_loop_passes_per_layer_tuple_for_4x():
    """The decoder loop MUST slice `past_key_values[idx]` for the 4.x path
    (per-layer 4-tuple). The original kijai/BART contract; round 6 dropped
    the `[idx]` and the model regressed to passing the whole tuple-of-tuples.
    """
    import torch
    cf, mf = _load_florence2_modules()
    model, cfg = _build_tiny_language_model(mf, cf)

    num_layers = cfg.decoder_layers
    bsz, heads, seq, hd = 1, 2, 1, cfg.d_model // cfg.decoder_attention_heads
    layer_caches = []
    for _ in range(num_layers):
        layer_caches.append(
            (
                torch.randn(bsz, heads, seq, hd),
                torch.randn(bsz, heads, seq, hd),
                torch.randn(bsz, heads, seq, hd),
                torch.randn(bsz, heads, seq, hd),
            )
        )
    past_key_values_4x = tuple(layer_caches)

    decoder_input_ids = torch.tensor([[2]], dtype=torch.long)
    encoder_outputs = model.model.encoder(
        input_ids=torch.tensor([[1, 5, 7]], dtype=torch.long),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        return_dict=True,
    )

    out = model.model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_outputs.last_hidden_state,
        encoder_attention_mask=torch.ones(1, 3, dtype=torch.long),
        past_key_values=past_key_values_4x,
        use_cache=True,
        return_dict=True,
    )
    pkv = out.past_key_values
    assert pkv is not None, "decoder returned None past_key_values"
    assert isinstance(pkv, tuple), f"4.x expects tuple-of-tuples, got {type(pkv).__name__}"
    assert len(pkv) == num_layers, f"expected {num_layers} per-layer caches, got {len(pkv)}"
    for i, layer_past in enumerate(pkv):
        assert isinstance(layer_past, tuple), f"layer {i} past is {type(layer_past).__name__}, expected tuple"
        assert len(layer_past) == 4, f"layer {i} past has len={len(layer_past)}, expected 4 (self_k, self_v, cross_k, cross_v)"
    print(f"[PASS] test_decoder_loop_passes_per_layer_tuple_for_4x (num_layers={num_layers})")


def main():
    failures = []
    test_funcs = [
        test_generate_second_step_does_not_raise_typeerror,
        test_decoder_layer_self_attn_past_key_value_is_not_none_for_5x_cache,
        test_decoder_layer_self_attn_past_key_value_is_slice_for_4x_tuple,
        test_decoder_loop_passes_per_layer_tuple_for_4x,
    ]
    for fn in test_funcs:
        try:
            fn()
        except AssertionError as e:
            failures.append((fn.__name__, str(e)))
            print(f"[FAIL] {fn.__name__}: {e}")
        except Exception as e:
            failures.append((fn.__name__, f"unexpected {type(e).__name__}: {e}"))
            traceback.print_exc()
            print(f"[FAIL] {fn.__name__}: unexpected {type(e).__name__}: {e}")
    print()
    print(f"Summary: {len(test_funcs) - len(failures)}/{len(test_funcs)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
