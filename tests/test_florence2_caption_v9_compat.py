"""
Round-7 TDD: Florence2 plugin image processor for transformers 5.x.

Symptom (post-round-6, V9.0 still failing in API workflow):
    File ".../modeling_florence2.py", line 2857, in _encode_image
    AssertionError: only support square feature maps for now

Root cause:
  transformers >= 5.0 no longer pulls `do_resize` / `size` / `resample` from
  the preprocessor config defaults. Worse, the Florence2Processor wrapper
  does NOT forward a `size=` kwarg to its image_processor -- so passing
  `size=...` raises TypeError. The robust fix is to resize the PIL image
  ourselves before calling the processor with `do_resize=False`.

Test scope (standalone -- no real model load):
  1. The describe_single_image function MUST resize non-square images to a
     square (HxW) before passing to the processor.
  2. The image's target size must come from `processor.image_processor.size`,
     not be hardcoded.
  3. The processor must be called with `do_resize=False` to prevent the
     image_processor from re-applying its (broken on 5.x) resize path.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
import traceback


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_florence2_caption_module():
    PKG = "_florence2_caption_round7"
    for n in list(sys.modules):
        if n == PKG or n.startswith(PKG + "."):
            sys.modules.pop(n, None)
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [_REPO_ROOT]
    sys.modules[PKG] = pkg
    spec = importlib.util.spec_from_file_location(
        f"{PKG}.florence2_caption",
        os.path.join(_REPO_ROOT, "florence2_caption.py"),
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"{PKG}.florence2_caption"] = m
    try:
        spec.loader.exec_module(m)
    except Exception as e:
        print(f"NOTE: florence2_caption.py import raised: {type(e).__name__}: {e}")
    return m


def test_describe_single_image_resizes_pil_before_processor():
    """The describe_single_image function must resize the PIL image
    to a square (HxW) size derived from `processor.image_processor.size`
    BEFORE calling the processor. This bypasses the 5.x Florence2Processor's
    `size` kwarg not being forwarded.
    """
    import inspect
    fc = _load_florence2_caption_module()
    if not hasattr(fc, "describe_single_image"):
        raise AssertionError("florence2_caption module missing describe_single_image")
    src = inspect.getsource(fc.describe_single_image)
    assert "img_proc" in src and "size" in src, (
        "describe_single_image must derive size from processor.image_processor"
    )
    assert "pil_image" in src and "resize" in src, (
        "describe_single_image must call pil_image.resize() to pre-resize the image"
    )
    print("[PASS] test_describe_single_image_resizes_pil_before_processor")


def test_describe_single_image_calls_processor_with_do_resize_false():
    """The describe_single_image function must call the processor with
    `do_resize=False` to prevent the image_processor from re-applying its
    (broken-on-5.x) default resize path.
    """
    import inspect
    fc = _load_florence2_caption_module()
    src = inspect.getsource(fc.describe_single_image)
    assert "do_resize=False" in src, (
        "describe_single_image must pass do_resize=False to the processor"
    )
    print("[PASS] test_describe_single_image_calls_processor_with_do_resize_false")


def test_describe_single_image_passes_resized_image_to_processor():
    """End-to-end check: with a non-square input image, the PIL image that
    reaches the processor must already be the configured square size.
    """
    import torch
    from PIL import Image

    fc = _load_florence2_caption_module()
    if not hasattr(fc, "describe_single_image"):
        raise AssertionError("florence2_caption module missing describe_single_image")

    # We track what PIL image actually reaches the processor.
    received_pil_size = {}

    class _StubImageProcessor:
        do_resize = True
        size = {"height": 768, "width": 768}
        resample = 3

    class _StubBatchFeature(dict):
        def to(self, *_a, **_kw):
            return self

    class _StubProcessor:
        image_processor = _StubImageProcessor()

        def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
            if isinstance(images, Image.Image):
                received_pil_size["size"] = images.size  # (W, H)
            bf = _StubBatchFeature()
            bf["input_ids"] = torch.tensor([[1, 2, 3]], dtype=torch.long)
            bf["pixel_values"] = torch.zeros(1, 3, 768, 768)
            return bf

        def batch_decode(self, ids, skip_special_tokens=False):
            return ["<stub_decoded_text>"]

    class _StubModel:
        def to(self, *_a, **_kw):
            return self
        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    # Provide a non-square PIL image via the patchable converter.
    non_square = Image.new("RGB", (896, 1344), color=(128, 64, 32))
    fc.image_to_pil_image = lambda _: non_square

    out = fc.describe_single_image(
        torch.zeros(1, 3, 1344, 896),  # fake "image" tensor
        _StubModel(),
        _StubProcessor(),
        "<MORE_DETAILED_CAPTION>",
        device="cpu",
        dtype=torch.float32,
        num_beams=1,
        max_new_tokens=4,
        do_sample=False,
    )
    assert "size" in received_pil_size, (
        "processor must have been called with a PIL image"
    )
    w, h = received_pil_size["size"]
    assert w == h, (
        f"PIL image reaching the processor must be square (HxW equal), got {w}x{h}"
    )
    print(f"[PASS] test_describe_single_image_passes_resized_image_to_processor (received={w}x{h})")


def main():
    failures = []
    test_funcs = [
        test_describe_single_image_resizes_pil_before_processor,
        test_describe_single_image_calls_processor_with_do_resize_false,
        test_describe_single_image_passes_resized_image_to_processor,
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
