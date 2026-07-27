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
    AttributeError: 'list' object has no attribute 'keys'

Root cause: transformers 5.x changed `_tied_weights_keys` from a flat list
(every item in the list is a tied key) to a `dict[target -> source]` mapping.
The downstream code in `PreTrainedModel.get_expanded_tied_weights_keys`
calls `tied_mapping.keys() | tied_mapping.values()`, which raises
`AttributeError` when `tied_mapping` is a list.

The fix: convert each of the three list-form `_tied_weights_keys` in
`modeling_florence2.py` to a dict, where the canonical BART-style tying
(decoder.embed_tokens + lm_head both point to encoder.embed_tokens) is
preserved.
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
    # __file__ = .../tests/test_modeling_florence2_v9_compat.py
    # step up twice: .../tests/ -> .../ComfyUI-CaptionThis/
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
    common_case_regex = re.compile(r"^[A-Za-z0-9_.]+(weight)|(bias)$")
    for class_name, _ in CLASS_ATTRS:
        cls = getattr(mf, class_name)
        value = cls._tied_weights_keys
        if not value: continue
        all_strings = list(value.keys()) + list(value.values())
        bad = [s for s in all_strings if not common_case_regex.match(s)]
        assert not bad, f"{class_name}._tied_weights_keys contains non-trivial names: {bad}"
    print("[PASS] test_tied_weights_keys_regex_short_circuits")

def test_tied_weights_preserve_bart_style_tying():
    _, mf = _load_florence2_modules()
    val = mf.Florence2LanguageModel._tied_weights_keys
    assert val.get("decoder.embed_tokens.weight") == "encoder.embed_tokens.weight", f"Florence2LanguageModel: {val!r}"
    val = mf.Florence2LanguageForConditionalGeneration._tied_weights_keys
    assert val.get("lm_head.weight") == "encoder.embed_tokens.weight", f"Florence2LanguageForConditionalGeneration lm_head: {val!r}"
    assert val.get("decoder.embed_tokens.weight") == "encoder.embed_tokens.weight", f"Florence2LanguageForConditionalGeneration decoder: {val!r}"
    val = mf.Florence2ForConditionalGeneration._tied_weights_keys
    assert val.get("language_model.lm_head.weight") == "language_model.encoder.embed_tokens.weight", f"Florence2ForConditionalGeneration lm_head: {val!r}"
    assert val.get("language_model.decoder.embed_tokens.weight") == "language_model.encoder.embed_tokens.weight", f"Florence2ForConditionalGeneration decoder: {val!r}"
    print("[PASS] test_tied_weights_preserve_bart_style_tying")

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
            raise AssertionError(f"get_expanded_tied_weights_keys still raises: {exc}") from exc
        raise
    assert isinstance(result, dict)
    print(f"[PASS] test_get_expanded_tied_weights_keys_does_not_raise result keys: {list(result.keys())}")

def main():
    failures = []
    test_funcs = [
        test_tied_weights_keys_are_dicts,
        test_tied_weights_keys_regex_short_circuits,
        test_tied_weights_preserve_bart_style_tying,
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
