"""
Tests for the model-completeness pre-check (assert_model_complete) added after
a LAN deploy failure: a model directory that contained only model.safetensors
(config.json / tokenizer / modeling code missing because a download was
interrupted) loaded with a cryptic
"'NoneType' object has no attribute 'model_type'".
The pre-check now raises a clear FileNotFoundError listing what's missing.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import tempfile
import types

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_common():
    PKG = "_captionthis_common_completeness_test"
    for n in list(sys.modules):
        if n == PKG or n.startswith(PKG + ".") or n == "nodes":
            sys.modules.pop(n, None)
    nodes_stub = types.ModuleType("nodes")
    nodes_stub.node_helpers = types.SimpleNamespace(pillow=lambda fn, *a, **k: fn(*a, **k))
    nodes_stub.ImageSequence = type("ImageSequence", (), {"Iterator": object})
    nodes_stub.ImageOps = types.SimpleNamespace(exif_transpose=lambda img: img)
    sys.modules["nodes"] = nodes_stub
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [_REPO_ROOT]
    sys.modules[PKG] = pkg
    utils_stub = types.ModuleType(f"{PKG}.utils")
    utils_stub.mie_log = lambda *a, **k: None
    sys.modules[f"{PKG}.utils"] = utils_stub
    spec = importlib.util.spec_from_file_location(f"{PKG}.common", os.path.join(_REPO_ROOT, "common.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"{PKG}.common"] = m
    spec.loader.exec_module(m)
    return m


_common = _load_common()
assert_model_complete = _common.assert_model_complete
failures = []


def test_complete_dir_no_raise():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "config.json"), "w").write("{}")
    assert_model_complete(d)  # default required_files=("config.json",)
    print("[PASS] test_complete_dir_no_raise")


def test_incomplete_dir_raises_clear_error():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "model.safetensors"), "w").write("x")  # only weights
    try:
        assert_model_complete(d, repo_id="org/model", required_files=("config.json", "tokenizer.json"))
        failures.append("test_incomplete_dir_raises_clear_error: did not raise")
        print("[FAIL] test_incomplete_dir_raises_clear_error: did not raise")
    except FileNotFoundError as e:
        msg = str(e)
        ok = "config.json" in msg and "tokenizer.json" in msg and "incomplete" in msg
        if ok:
            print("[PASS] test_incomplete_dir_raises_clear_error")
        else:
            failures.append(f"test_incomplete_dir_raises_clear_error: bad message: {msg!r}")
            print(f"[FAIL] test_incomplete_dir_raises_clear_error: bad message: {msg!r}")


def test_reports_only_actually_missing():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "config.json"), "w").write("{}")  # present
    try:
        assert_model_complete(d, required_files=("config.json", "tokenizer.json"))
        failures.append("test_reports_only_actually_missing: did not raise")
        print("[FAIL] test_reports_only_actually_missing: did not raise")
    except FileNotFoundError as e:
        msg = str(e)
        # the message's "Missing ..." part must list tokenizer.json, not config.json
        missing_part = msg.split("Missing required file(s):")[1] if "Missing required file(s):" in msg else ""
        if "tokenizer.json" in missing_part and "config.json" not in missing_part:
            print("[PASS] test_reports_only_actually_missing")
        else:
            failures.append(f"test_reports_only_actually_missing: {msg!r}")
            print(f"[FAIL] test_reports_only_actually_missing: {msg!r}")


def test_message_mentions_repo_id_when_given():
    d = tempfile.mkdtemp()
    try:
        assert_model_complete(d, repo_id="MiaoshouAI/Florence-2-base-PromptGen-v2.0")
        failures.append("test_message_mentions_repo_id_when_given: did not raise")
        print("[FAIL] test_message_mentions_repo_id_when_given")
    except FileNotFoundError as e:
        if "MiaoshouAI/Florence-2-base-PromptGen-v2.0" in str(e):
            print("[PASS] test_message_mentions_repo_id_when_given")
        else:
            failures.append("test_message_mentions_repo_id_when_given: repo_id not in msg")
            print("[FAIL] test_message_mentions_repo_id_when_given: repo_id not in msg")


def main():
    tests = [test_complete_dir_no_raise, test_incomplete_dir_raises_clear_error,
             test_reports_only_actually_missing, test_message_mentions_repo_id_when_given]
    for fn in tests:
        try:
            fn()
        except Exception as e:
            failures.append(f"{fn.__name__}: unexpected {type(e).__name__}: {e}")
            print(f"[FAIL] {fn.__name__}: unexpected {type(e).__name__}: {e}")
    print()
    print(f"Summary: {len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        for f in failures:
            print("  -", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
