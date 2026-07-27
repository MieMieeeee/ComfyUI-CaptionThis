"""
Regression tests for issue #13: "Cannot figure out path format (linux)".

Root cause was in common.py: get_image_files() probed every entry with
PIL.Image.open (slow + fragile on NFS / in containers, fails on mixed content
like .DS_Store / partial transfers), paths were not normalized (whitespace,
trailing separator, ~), and the only failure message was a generic
"No image files found in <dir>" that gave the user no way to tell whether the
path was wrong, the mount was down, or the directory simply had no images.

These tests cover the common.py changes directly without needing ComfyUI
runtime (describe_images_core is exercised with a stub describe_function).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import tempfile
import types
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_common():
    """Load common.py standalone by stubbing the ComfyUI runtime deps it imports
    (`nodes`, and the relative `.utils`). We only exercise the pure path/file
    helpers, so the stubs don't need real behavior -- just importable names.
    """
    PKG = "_captionthis_common_test"
    for n in list(sys.modules):
        if n == PKG or n.startswith(PKG + ".") or n == "nodes":
            sys.modules.pop(n, None)

    # stub ComfyUI `nodes` module (provides node_helpers, ImageSequence, ImageOps)
    nodes_stub = types.ModuleType("nodes")
    nodes_stub.node_helpers = types.SimpleNamespace(pillow=lambda fn, *a, **k: fn(*a, **k))
    nodes_stub.ImageSequence = type("ImageSequence", (), {"Iterator": object})
    nodes_stub.ImageOps = types.SimpleNamespace(exif_transpose=lambda img: img)
    sys.modules["nodes"] = nodes_stub

    # stub the plugin's own `utils` as a package member so `from .utils import mie_log` resolves
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [_REPO_ROOT]
    sys.modules[PKG] = pkg
    utils_stub = types.ModuleType(f"{PKG}.utils")
    utils_stub.mie_log = lambda *a, **k: None
    sys.modules[f"{PKG}.utils"] = utils_stub

    spec = importlib.util.spec_from_file_location(
        f"{PKG}.common", os.path.join(_REPO_ROOT, "common.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"{PKG}.common"] = m
    spec.loader.exec_module(m)
    return m


_common = _load_common()
get_image_files = _common.get_image_files
normalize_directory_path = _common.normalize_directory_path
describe_images_core = _common.describe_images_core
IMAGE_EXTENSIONS = _common.IMAGE_EXTENSIONS

failures = []


# --------------------------------------------------------------------------- #
# normalize_directory_path
# --------------------------------------------------------------------------- #
def test_normalize_strips_whitespace():
    assert normalize_directory_path("  /tmp/x  ") == os.path.normpath("/tmp/x")
    # also ensure no surrounding whitespace survives
    assert " " not in normalize_directory_path("  /tmp/x  ")
    print("[PASS] test_normalize_strips_whitespace")


def test_normalize_handles_trailing_separator():
    # Trailing slash should not break glob; normpath collapses it.
    assert normalize_directory_path("/tmp/x/").rstrip(os.sep) == "/tmp/x".replace("/", os.sep).rstrip(os.sep) \
        or normalize_directory_path("/tmp/x/") == normalize_directory_path("/tmp/x")
    # Stronger invariant: normalized form never ends with a lone separator (except root).
    n = normalize_directory_path(os.path.join(tempfile.gettempdir(), "foo") + os.sep)
    assert not n.endswith(os.sep), f"trailing sep present: {n!r}"
    print("[PASS] test_normalize_handles_trailing_separator")


def test_normalize_expands_user():
    home = os.path.expanduser("~")
    assert normalize_directory_path("~") == home
    print(f"[PASS] test_normalize_expands_user (-> {home})")


def test_normalize_none_passthrough():
    assert normalize_directory_path(None) is None
    print("[PASS] test_normalize_none_passthrough")


# --------------------------------------------------------------------------- #
# get_image_files - the core #13 fix (extension filter, not per-file open)
# --------------------------------------------------------------------------- #
def _make_mixed_dir():
    """A directory that mirrors the #13 failure: images mixed with junk that
    used to make the old Image.open probe choke or slow down."""
    d = tempfile.mkdtemp()
    Image.new("RGB", (4, 4)).save(os.path.join(d, "a.png"))
    Image.new("RGB", (4, 4)).save(os.path.join(d, "b.JPG"))  # uppercase ext
    Image.new("RGB", (4, 4)).save(os.path.join(d, "c.webp"))
    # junk that the OLD code would Image.open one by one:
    open(os.path.join(d, "notes.txt"), "w").write("hi")
    open(os.path.join(d, ".DS_Store"), "w").write("garbage")
    open(os.path.join(d, "partial.dat"), "wb").write(b"\x00\x01notimage")
    os.makedirs(os.path.join(d, "subdir"))
    return d


def test_get_image_files_extension_filter_picks_images_only():
    d = _make_mixed_dir()
    found = sorted(os.path.basename(f) for f in get_image_files(d))
    assert found == ["a.png", "b.JPG", "c.webp"], f"unexpected files: {found}"
    print(f"[PASS] test_get_image_files_extension_filter_picks_images_only ({found})")


def test_get_image_files_case_insensitive_extension():
    d = _make_mixed_dir()
    found = [os.path.basename(f) for f in get_image_files(d)]
    assert "b.JPG" in found, "uppercase .JPG must be recognized"
    print("[PASS] test_get_image_files_case_insensitive_extension")


def test_get_image_files_empty_dir_returns_empty():
    d = tempfile.mkdtemp()
    assert get_image_files(d) == []
    print("[PASS] test_get_image_files_empty_dir_returns_empty")


def test_get_image_files_nonexistent_returns_empty_no_crash():
    # glob on a missing dir returns [] without raising -- must stay that way.
    assert get_image_files(os.path.join(tempfile.gettempdir(), "definitely_missing_xyz")) == []
    print("[PASS] test_get_image_files_nonexistent_returns_empty_no_crash")


# --------------------------------------------------------------------------- #
# describe_images_core - actionable diagnostics for the 3 failure modes
# --------------------------------------------------------------------------- #
def _stub_describe(image, *argv):
    return "stub"


def test_describe_core_nonexistent_directory_message():
    msg = describe_images_core(
        os.path.join(tempfile.gettempdir(), "nope_xyz"), False, None, _stub_describe
    )
    text = msg[0]
    assert "does not exist" in text, f"wrong message: {text!r}"
    print(f"[PASS] test_describe_core_nonexistent_directory_message")


def test_describe_core_path_is_file_message():
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    f.close()
    try:
        msg = describe_images_core(f.name, False, None, _stub_describe)
        assert "not a directory" in msg[0], f"wrong message: {msg[0]!r}"
        print("[PASS] test_describe_core_path_is_file_message")
    finally:
        os.unlink(f.name)


def test_describe_core_empty_dir_lists_extensions_seen():
    d = tempfile.mkdtemp()
    # put only non-image files so the user can see why nothing matched
    open(os.path.join(d, "readme.md"), "w").write("x")
    open(os.path.join(d, "data.csv"), "w").write("y")
    msg = describe_images_core(d, False, None, _stub_describe)
    text = msg[0]
    assert "No image files" in text
    assert ".md" in text and ".csv" in text, f"should list seen extensions: {text!r}"
    print(f"[PASS] test_describe_core_empty_dir_lists_extensions_seen ({text[:80]}...)")


def test_describe_core_processes_real_images():
    """describe_images_core iterates exactly the image files and calls the
    describe function once per image. We stub load_image_core on the module so
    we don't need the full ComfyUI runtime to exercise the loop + save path.
    """
    d = _make_mixed_dir()
    seen = []

    def fake_load(image_path):
        # return a 1-tuple matching load_image_core's (image, mask) shape
        seen.append(image_path)
        return ("fake-image",)

    orig = _common.load_image_core
    _common.load_image_core = fake_load
    try:
        msg = describe_images_core(d, False, None, _stub_describe)
    finally:
        _common.load_image_core = orig
    assert "Described 3 images" in msg[0], f"unexpected: {msg[0]!r}"
    assert len(seen) == 3, f"expected 3 load calls, got {len(seen)}"
    # .txt sidecars must have been written next to each image (a/b/c, not notes.txt)
    txts = sorted(f for f in os.listdir(d) if f.endswith(".txt"))
    expected_txts = ["a.txt", "b.txt", "c.txt"]
    assert all(t in txts for t in expected_txts), f"missing sidecars: {txts}"
    print(f"[PASS] test_describe_core_processes_real_images (loaded {len(seen)}, wrote {expected_txts})")


def main():
    tests = [
        test_normalize_strips_whitespace,
        test_normalize_handles_trailing_separator,
        test_normalize_expands_user,
        test_normalize_none_passthrough,
        test_get_image_files_extension_filter_picks_images_only,
        test_get_image_files_case_insensitive_extension,
        test_get_image_files_empty_dir_returns_empty,
        test_get_image_files_nonexistent_returns_empty_no_crash,
        test_describe_core_nonexistent_directory_message,
        test_describe_core_path_is_file_message,
        test_describe_core_empty_dir_lists_extensions_seen,
        test_describe_core_processes_real_images,
    ]
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            failures.append((fn.__name__, str(e)))
            print(f"[FAIL] {fn.__name__}: {e}")
        except Exception as e:
            failures.append((fn.__name__, f"unexpected {type(e).__name__}: {e}"))
            print(f"[FAIL] {fn.__name__}: unexpected {type(e).__name__}: {e}")
    print()
    print(f"Summary: {len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
