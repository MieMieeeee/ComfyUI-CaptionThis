"""
Regression test for `modeling_florence2.py` v9 + transformers 5.x compatibility.

Symptom under transformers >= 5.0 (reported after the V9 dataclass fix):

    File ".../modeling_florence2.py", line 2557, in __init__
        self.post_init()
    File ".../transformers/modeling_utils.py", line 1394, in post_init
        self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(...)
    File ".../transformers/modeling_utils.py", line 2596, in get_expanded_tied_weights_keys
        if all(common_case_regex.match(k) for k in tied_mapping.keys() | tied_mapping.values()):
                                                           ^^^^^^^^^^^^^^^^^
    AttributeError: ''list'' object has no attribute ''keys''

Round 4 catch (reported in API workflow load):
    Loading weights: 100%|██████████| 667/667 [00:00<00:00, 2237.00it/s]
    Key                                              | Status  |
    language_model.model.encoder.embed_tokens.weight | MISSING |
    language_model.model.decoder.embed_tokens.weight | MISSING |
    AttributeError: Florence2LanguageForConditionalGeneration has no attribute `decoder`

Root cause
----------
Round 3 turned `_tied_weights_keys` from a flat list to a `dict[target -> source]`
mapping (transformers 5.x contract). But the dict targets were the SHORT
form (e.g. `decoder.embed_tokens.weight`) which is correct for the inner
`Florence2LanguageModel` but wrong for `Florence2LanguageForConditionalGeneration`
(where decoder is at `self.model.decoder` not `self.decoder`) and for the
top-level `Florence2ForConditionalGeneration` (where the language model
is at `self.language_model` and decoder inside that is at
`self.language_model.model.decoder`).

`PreTrainedModel.mark_tied_weights_as_initialized` (called during
`_finalize_model_loading`) iterates `all_tied_weights_keys.keys()` and calls
`self.get_parameter(tgt)` for each target. A target that is not navigable
from `self` raises `AttributeError`. The same code path also looks up the
source for actual weight tying, so the source has to be navigable too.

The fix: in all 3 PreTrainedModel subclasses, the dict targets and sources
must be navigable from `self` of the declaring class. Round 4 corrects the
round-3 dict to include the proper `model.` / `language_model.model.`
prefixes and switches the canonical source to `shared.weight` (matching
the codebase''s `_tie_weights` method, which ties to `self.shared` /
`self.model.shared`).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
import traceback
import types

def _repo_root():
    d = __file__
    for _ in range(2):
        d = d.rsplit(chr(92), 1)[0] if chr(92) in d else d.rsplit("/", 1)[0]
    return d

_REPO_ROOT = _repo_root()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_florence2_modules():
    PKG = "_florence2_synthetic"
    for n in list(sys.modules):
        if n == PKG or n.startswith(PKG + "."):
            sys.modules.pop(n, None)
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [_REPO_ROOT]
    sys.modules[PKG] = pkg
    spec_cf = importlib.util.spec_from_file_location(f"{PKG}.configuration_florence2", os.path.join(_REPO_ROOT, "configuration_florence2.py"))
    cf = importlib.util.module_from_spec(spec_cf)
    sys.modules[f"{PKG}.configuration_florence2"] = cf
    spec_cf.loader.exec_module(cf)
    spec_mf = importlib.util.spec_from_file_location(f"{PKG}.modeling_florence2", os.path.join(_REPO_ROOT, "modeling_florence2.py"))
    mf = importlib.util.module_from_spec(spec_mf)
    sys.modules[f"{PKG}.modeling_florence2"] = mf
    spec_mf.loader.exec_module(mf)
    return cf, mf


CLASS_ATTRS = [
    ("Florence2LanguageModel", "_tied_weights_keys"),
    ("Florence2LanguageForConditionalGeneration", "_tied_weights_keys"),
    ("Florence2ForConditionalGeneration", "_tied_weights_keys"),
]

# Per-class expected tying map (the round-4 BC contract). Each entry is
# (target -> source) in the BART convention used by this codebase: targets
# are the language-side embedding / lm_head layers; the canonical source
# is `shared.weight`. The actual prefixes are dictated by where the
# language sub-model sits in the class hierarchy:
#   - Florence2LanguageModel                       : self.shared / self.encoder / self.decoder
#   - Florence2LanguageForConditionalGeneration   : self.model.shared / self.model.encoder / self.model.decoder + self.lm_head
#   - Florence2ForConditionalGeneration (top)     : self.language_model.model.shared / ... + self.language_model.lm_head
EXPECTED_TYING = {
    "Florence2LanguageModel": {
        "encoder.embed_tokens.weight": "shared.weight",
        "decoder.embed_tokens.weight": "shared.weight",
    },
    "Florence2LanguageForConditionalGeneration": {
        "model.encoder.embed_tokens.weight": "model.shared.weight",
        "model.decoder.embed_tokens.weight": "model.shared.weight",
        "lm_head.weight": "model.shared.weight",
    },
    "Florence2ForConditionalGeneration": {
        "language_model.model.encoder.embed_tokens.weight": "language_model.model.shared.weight",
        "language_model.model.decoder.embed_tokens.weight": "language_model.model.shared.weight",
        "language_model.lm_head.weight": "language_model.model.shared.weight",
    },
}


def test_tied_weights_keys_are_dicts():
    _, mf = _load_florence2_modules()
    for class_name, attr_name in CLASS_ATTRS:
        cls = getattr(mf, class_name, None)
        assert cls is not None, f"{class_name} not present on modeling_florence2"
        value = getattr(cls, attr_name, None)
        assert value is not None, f"{class_name}.{attr_name} missing"
        assert isinstance(value, dict), f"{class_name}.{attr_name} must be a dict (transformers 5.x), got {type(value).__name__}: {value!r}"
        for k, v in value.items():
            assert isinstance(k, str) and k, f"{class_name}.{attr_name}: bad target key {k!r}"
            assert isinstance(v, str) and v, f"{class_name}.{attr_name}: bad source value {v!r}"
        print(f"[PASS] test_tied_weights_keys_are_dicts  {class_name}: {value}")


def test_tied_weights_keys_regex_short_circuits():
    _, mf = _load_florence2_modules()
    common_case_regex = re.compile(r"^[A-Za-z0-9_\.]+(weight)|(bias)$")
    for class_name, _ in CLASS_ATTRS:
        cls = getattr(mf, class_name)
        value = cls._tied_weights_keys
        if not value: continue
        all_strings = list(value.keys()) + list(value.values())
        bad = [s for s in all_strings if not common_case_regex.match(s)]
        assert not bad, f"{class_name}._tied_weights_keys contains non-trivial names: {bad}"
    print("[PASS] test_tied_weights_keys_regex_short_circuits")


def test_tied_weights_preserve_bart_style_tying():
    """Round-4 contract: targets are the language-side embeddings / lm_head,
    source is the language-side `shared.weight` (matches `_tie_weights` in
    this codebase). The exact prefixes depend on the class hierarchy.
    """
    _, mf = _load_florence2_modules()
    for class_name, expected in EXPECTED_TYING.items():
        val = getattr(mf, class_name)._tied_weights_keys
        assert val == expected, f"{class_name}._tied_weights_keys mismatch:\n  got:      {val!r}\n  expected: {expected!r}"
    print("[PASS] test_tied_weights_preserve_bart_style_tying")


def test_tied_weights_targets_navigable_on_instantiated_model():
    """The smoking gun for the round-3 bug: the dict targets must be
    navigable from a real instance of the class via `get_parameter`. We
    build a minimal config and instantiate the language-side models (the
    Florence2ForConditionalGeneration top-level is heavy because of DaViT;
    we cover the two language classes here and rely on the structural
    assertion in test_tied_weights_preserve_bart_style_tying for the third).
    """
    import torch
    _, mf = _load_florence2_modules()
    cf = sys.modules["_florence2_synthetic.configuration_florence2"]
    with torch.no_grad():
        cfg = cf.Florence2LanguageConfig(
            vocab_size=1024, d_model=64, encoder_layers=1, encoder_ffn_dim=64,
            encoder_attention_heads=2, decoder_layers=1, decoder_ffn_dim=64,
            decoder_attention_heads=2, num_hidden_layers=1, pad_token_id=0,
            bos_token_id=1, eos_token_id=2, decoder_start_token_id=2,
        )
        m = mf.Florence2LanguageForConditionalGeneration(cfg)
    for tgt, src in m._tied_weights_keys.items():
        try:
            p = m.get_parameter(tgt)
        except AttributeError as exc:
            raise AssertionError(
                f"Florence2LanguageForConditionalGeneration cannot navigate {tgt!r}: {exc}"
            ) from exc
        try:
            s = m.get_parameter(src)
        except AttributeError as exc:
            raise AssertionError(
                f"Florence2LanguageForConditionalGeneration cannot navigate source {src!r}: {exc}"
            ) from exc
        assert p.data_ptr() == s.data_ptr() or p.shape == s.shape, (
            f"target/source {tgt!r} -> {src!r} should point to compatible tensors"
        )
    print("[PASS] test_tied_weights_targets_navigable_on_instantiated_model")


def test_get_expanded_tied_weights_keys_does_not_raise():
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "get_expanded_tied_weights_keys"):
        print("[SKIP] test_get_expanded_tied_weights_keys_does_not_raise (transformers < 5.0)")
        return
    _, mf = _load_florence2_modules()
    cls = mf.Florence2LanguageForConditionalGeneration
    class _StubConfig:
        tie_word_embeddings = True
    stub = cls.__new__(cls)
    stub._tied_weights_keys = cls._tied_weights_keys
    stub.config = _StubConfig()
    try:
        result = cls.get_expanded_tied_weights_keys(stub, all_submodels=False)
    except AttributeError as exc:
        msg = str(exc)
        if chr(39) + "keys" + chr(39) in msg or ("list" in msg.lower() and "attribute" in msg.lower()):
            raise AssertionError(f"get_expanded_tied_weights_keys still raises the regression: {exc}") from exc
        raise
    assert isinstance(result, dict)
    print(f"[PASS] test_get_expanded_tied_weights_keys_does_not_raise result keys: {list(result.keys())}")


def main():
    failures = []
    test_funcs = [
        test_tied_weights_keys_are_dicts,
        test_tied_weights_keys_regex_short_circuits,
        test_tied_weights_preserve_bart_style_tying,
        test_tied_weights_targets_navigable_on_instantiated_model,
        test_get_expanded_tied_weights_keys_does_not_raise,
    ]
    for fn in test_funcs:
        try:
            fn()
        except AssertionError as e:
            failures.append((fn.__name__, str(e)))
            print(f"[FAIL] {fn.__name__}: {e}")
        except Exception as e:
            failures.append((fn.__name__, f"unexpected: {e}"))
            print(f"[FAIL] {fn.__name__}: unexpected {type(e).__name__}: {e}")
    print()
    print(f"Summary: {len(test_funcs) - len(failures)}/{len(test_funcs)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
