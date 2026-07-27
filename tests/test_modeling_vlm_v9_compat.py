"""
Regression tests for `janus/models/modeling_vlm.py` v9 + transformers 5.x compatibility.

Symptom under transformers >= 5.x:

    File ".../configuration_utils.py", line 316, in __init_subclass__
        cls = dataclass(cls, repr=False, kw_only=True)
    ValueError: mutable default <class 'dict'> for field params is not allowed:
                use default_factory

Root cause: each `*Config(PretrainedConfig)` class declares
`params: AttrDict = {}` at class-level. transformers 5.x wraps PretrainedConfig
subclasses in @dataclass, which forbids mutable default values.

This test imports modeling_vlm.py fresh (in a per-test isolated module cache)
and instantiates each Config class. It is parameterized to be runnable on any
of the three target Python environments via:

    python tests/test_modeling_vlm_v9_compat.py

Each environment produces its own pass/fail summary so the same script can be
used as a smoke check after sync.
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback

# Standalone-runnable: prepend the project root so we can `import janus`.
_THIS_DIR = __file__.rsplit("\\", 1)[0] if "\\" in __file__ else __file__.rsplit("/", 1)[0]
_REPO_ROOT = _THIS_DIR.rsplit("\\", 1)[0] if "\\" in _THIS_DIR else _THIS_DIR.rsplit("/", 1)[0]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


CONFIG_CLASSES = (
    "VisionConfig",
    "AlignerConfig",
    "GenVisionConfig",
    "GenAlignerConfig",
    "GenHeadConfig",
)


def _import_isolated():
    """Re-import modeling_vlm fresh so prior test runs do not poison the cache."""
    for name in list(sys.modules):
        if name == "janus.models.modeling_vlm" or name.startswith("janus.models.modeling_vlm."):
            sys.modules.pop(name, None)
    # Trigger janus/__init__.py monkey-patch for the `attrdict` package.
    import janus  # noqa: F401
    mod = importlib.import_module("janus.models.modeling_vlm")
    return mod


def test_each_config_importable():
    mod = _import_isolated()
    classes = {name: getattr(mod, name) for name in CONFIG_CLASSES}
    missing = [name for name in CONFIG_CLASSES if name not in classes]
    assert not missing, f"missing Config classes on modeling_vlm: {missing}"
    for name, cls in classes.items():
        assert hasattr(cls, "model_type"), f"{name} missing model_type"
    print("[PASS] test_each_config_importable", {n: c.model_type for n, c in classes.items()})


def test_instantiate_each_config_with_no_kwargs():
    """Each Config class must be instantiable with no kwargs (default behaviour).

    Pre-fix this fails on transformers >=5.x with:
        ValueError: mutable default <class 'dict'> for field params is not allowed
    at class-definition time, so we never even get here.
    Post-fix the class body itself loads, and instantiate-with-no-args returns
    an object with `params` set to an empty AttrDict.
    """
    mod = _import_isolated()
    classes = [getattr(mod, name) for name in CONFIG_CLASSES]
    instantiated = []
    for cls in classes:
        try:
            instance = cls()
        except Exception as exc:
            raise AssertionError(
                f"failed to instantiate {cls.__name__}: {type(exc).__name__}: {exc}"
            ) from exc
        assert hasattr(instance, "params"), f"{cls.__name__} missing params attr"
        assert hasattr(instance, "cls"), f"{cls.__name__} missing cls attr"
        instantiated.append((cls.__name__, instance.cls, type(instance.params).__name__))
    print("[PASS] test_instantiate_each_config_with_no_kwargs",
          json.dumps(instantiated, ensure_ascii=False))


def test_instantiate_each_config_with_params_kwarg():
    """Passing `params={'k': 'v'}` to each Config must propagate to instance.params.

    We don't strictly require AttrDict; any object whose `.k` resolves to 'v' is fine.
    A plain dict will NOT satisfy downstream code (modeling_vlm / projector use dot
    notation), so we verify the existing AttrDict contract is preserved.
    """
    mod = _import_isolated()
    classes = [getattr(mod, name) for name in CONFIG_CLASSES]
    for cls in classes:
        try:
            instance = cls(params={"image_token_size": 1024, "n_embed": 2048})
        except Exception as exc:
            raise AssertionError(
                f"failed to instantiate {cls.__name__} with params kwarg: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        assert getattr(instance.params, "image_token_size", None) == 1024, (
            f"{cls.__name__}.params.image_token_size != 1024; "
            f"got type={type(instance.params).__name__}"
        )
        assert getattr(instance.params, "n_embed", None) == 2048
    print("[PASS] test_instantiate_each_config_with_params_kwarg")


def test_registration_block_executes():
    """After class definitions, the bottom AutoConfig/AutoModelForCausalLM.register
    calls must run without error and the registry must accept each model_type.
    """
    mod = _import_isolated()
    from transformers import AutoConfig
    for name in CONFIG_CLASSES:
        cls = getattr(mod, name)
        # AutoConfig.for_model returns a registered config class; we only need to
        # ensure `model_type` was registered. AutoConfig.for_model raises if not.
        try:
            AutoConfig.for_model(cls.model_type)
        except Exception as exc:
            raise AssertionError(
                f"AutoConfig.for_model({cls.model_type!r}) failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    print("[PASS] test_registration_block_executes")


def test_siglip_vit_meta_device_init():
    """The siglip VisionTransformer must instantiate under the meta-device context
    that transformers >= 5.0 `from_pretrained` uses.

    Regression for issue #19: the drop-path schedule was computed as
    `[x.item() for x in torch.linspace(0, drop_path_rate, depth)]`. Calling
    `.item()` on a meta tensor raises
    `RuntimeError: Tensor.item() cannot be called on meta tensors` during model
    loading, crashing Janus load entirely. The fix replaces it with a pure-Python
    schedule (`drop_path_rate * i / (depth - 1)`) that is numerically identical
    and touches no tensors during `__init__`.
    """
    import torch

    # Force the janus package import so its monkey-patches run.
    import janus  # noqa: F401
    siglip = importlib.import_module("janus.models.siglip_vit")
    VisionTransformer = getattr(siglip, "VisionTransformer")
    assert VisionTransformer is not None, "VisionTransformer not found in siglip_vit"

    # 1. meta-device instantiation must NOT crash (this is the issue #19 failure).
    try:
        with torch.device("meta"):
            vt = VisionTransformer(
                img_size=384, patch_size=14, embed_dim=1024,
                depth=24, num_heads=16, mlp_ratio=4,
            )
    except Exception as exc:
        raise AssertionError(
            f"VisionTransformer crashed under torch.device('meta') (issue #19 "
            f"regression): {type(exc).__name__}: {exc}"
        ) from exc
    assert len(vt.blocks) == 24, f"expected 24 blocks, got {len(vt.blocks)}"

    # 2. normal instantiation still works and builds the expected structure.
    vt_real = VisionTransformer(
        img_size=384, patch_size=14, embed_dim=1024,
        depth=24, num_heads=16, mlp_ratio=4,
    )
    assert len(vt_real.blocks) == 24

    # 3. the dpr schedule is numerically identical to the old linspace form.
    depth, rate = 24, 0.1
    old = [x.item() for x in torch.linspace(0, rate, depth)]
    new = [rate * i / (depth - 1) for i in range(depth)]
    maxdiff = max(abs(a - b) for a, b in zip(old, new))
    assert maxdiff < 1e-6, f"dpr schedule diverged from linspace: max|old-new|={maxdiff}"

    print("[PASS] test_siglip_vit_meta_device_init "
          "(meta init OK, normal init OK, dpr numerically identical)")


def main():
    failures = []
    test_funcs = [
        test_each_config_importable,
        test_instantiate_each_config_with_no_kwargs,
        test_instantiate_each_config_with_params_kwarg,
        test_registration_block_executes,
        test_siglip_vit_meta_device_init,
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
