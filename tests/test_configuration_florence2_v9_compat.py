"""
Regression test for `configuration_florence2.py` v9 + transformers 5.x compatibility.

Symptom under transformers >= 5.0 (reported in V9.0 + Python 3.13):

    File ".../configuration_florence2.py", line 265, in __init__
        if self.forced_bos_token_id is None and kwargs.get(...):
    AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'

Root cause: transformers 5.x no longer calls
    for parameter_name, default_value in _get_global_generation_defaults().items():
        setattr(self, parameter_name, kwargs.pop(parameter_name, default_value))
inside `PretrainedConfig.__init__`. Generation defaults (including
`forced_bos_token_id`) are popped-and-discarded in `__post_init__` instead,
so they are never bound as instance attributes. Reading `self.forced_bos_token_id`
on a freshly-built config raises `AttributeError`.

Trigger path (from user traceback):
    Florence2ForConditionalGeneration.from_pretrained
        -> cls.from_pretrained
        -> cls.from_dict(config_dict)
        -> cls(**config_dict)              # Florence2Config(**config_dict)
        -> Florence2LanguageConfig(**text_config)   # line 336
        -> reads self.forced_bos_token_id  # line 265, AttributeError

This test reproduces the failure on the exact construction paths and verifies
they instantiate cleanly after the fix.
"""

from __future__ import annotations

import importlib
import sys
import traceback

# Standalone-runnable: prepend the project root so we can import the plugin.
_THIS_DIR = __file__.rsplit("\\", 1)[0] if "\\" in __file__ else __file__.rsplit("/", 1)[0]
_REPO_ROOT = _THIS_DIR.rsplit("\\", 1)[0] if "\\" in _THIS_DIR else _THIS_DIR.rsplit("/", 1)[0]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _import_isolated():
    """Re-import configuration_florence2 fresh so prior runs do not poison the cache."""
    for n in list(sys.modules):
        if n == "configuration_florence2":
            sys.modules.pop(n, None)
    return importlib.import_module("configuration_florence2")


def test_florence2_language_config_no_kwargs():
    """The minimal reproduction of the AttributeError reported by the user."""
    mod = _import_isolated()
    try:
        cfg = mod.Florence2LanguageConfig()
    except Exception as exc:
        raise AssertionError(
            f"Florence2LanguageConfig() raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    print("[PASS] test_florence2_language_config_no_kwargs")


def test_florence2_language_config_with_force_bos_token_flag():
    """The legacy BART-CNN branch (line 265-270) should still work.

    When force_bos_token_to_be_generated=True AND forced_bos_token_id is
    None/missing, the config should backfill forced_bos_token_id from
    bos_token_id. The fix must preserve this behavior.
    """
    mod = _import_isolated()
    try:
        cfg = mod.Florence2LanguageConfig(force_bos_token_to_be_generated=True)
    except Exception as exc:
        raise AssertionError(
            f"Florence2LanguageConfig(force_bos_token_to_be_generated=True) raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    fb = getattr(cfg, "forced_bos_token_id", None)
    if fb is None:
        raise AssertionError("forced_bos_token_id should be backfilled from bos_token_id when flag set")
    if fb != cfg.bos_token_id:
        raise AssertionError(
            f"forced_bos_token_id ({fb!r}) does not match bos_token_id ({cfg.bos_token_id!r})"
        )
    print("[PASS] test_florence2_language_config_with_force_bos_token_flag", "fbos=", fb)


def test_florence2_vision_config_no_kwargs():
    """Same construction but on the vision side; should be regression-clean."""
    mod = _import_isolated()
    try:
        cfg = mod.Florence2VisionConfig()
    except Exception as exc:
        raise AssertionError(
            f"Florence2VisionConfig() raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    print("[PASS] test_florence2_vision_config_no_kwargs")


def test_florence2_top_config_with_realistic_text_config():
    """Exact path from the user's traceback:
    Florence2Config(**config_dict) where config_dict includes a non-empty
    text_config. The constructor must build Florence2LanguageConfig(**text_config)
    without raising AttributeError on self.forced_bos_token_id.
    """
    mod = _import_isolated()
    config_dict = {
        "model_type": "florence2",
        "projection_dim": 1024,
        "text_config": {"model_type": "florence2_language", "vocab_size": 51289},
        "vision_config": {"model_type": "florence2_vision", "dim_embed": [128, 256, 512]},
    }
    try:
        cfg = mod.Florence2Config(**config_dict)
    except Exception as exc:
        raise AssertionError(
            f"Florence2Config(**config_dict) raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    assert cfg.text_config is not None, "text_config should be populated after construction"
    assert cfg.vision_config is not None, "vision_config should be populated after construction"
    print("[PASS] test_florence2_top_config_with_realistic_text_config")


def main():
    failures = []
    test_funcs = [
        test_florence2_language_config_no_kwargs,
        test_florence2_language_config_with_force_bos_token_flag,
        test_florence2_vision_config_no_kwargs,
        test_florence2_top_config_with_realistic_text_config,
    ]
    for fn in test_funcs:
        try:
            fn()
        except AssertionError as e:
            failures.append((fn.__name__, str(e), traceback.format_exc()))
            print(f"[FAIL] {fn.__name__}: {e}")
        except Exception as e:
            failures.append((fn.__name__, f"unexpected: {e}", traceback.format_exc()))
            print(f"[FAIL] {fn.__name__}: unexpected {type(e).__name__}: {e}")
    print()
    print(f"Summary: {len(test_funcs) - len(failures)}/{len(test_funcs)} passed")
    if failures:
        for name, msg, tb in failures:
            print(f"--- {name} ---\n{msg}\n")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
