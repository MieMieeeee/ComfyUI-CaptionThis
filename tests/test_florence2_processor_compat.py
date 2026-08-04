# coding=utf-8
"""Tests for the Florence2 processor loader (Issue #21 regression).

Verifies that ``florence2_processor.load_florence2_processor`` builds the
processor WITHOUT going through ``AutoProcessor -> AutoConfig`` -- the chain that,
on transformers 5.x, executes the model repo's remote ``configuration_florence2.py``
and hits the unguarded ``self.forced_bos_token_id`` read.

Two test layers
---------------
1. ``MockedAssemblyTests`` -- fast unit tests that mock the three external entry
   points (``BartTokenizerFast.from_pretrained``, ``CLIPImageProcessor.from_pretrained``,
   ``get_class_from_dynamic_module``) to verify the loader's *assembly logic*:
   correct class-reference resolution, explicit ``__init__`` construction (not
   ``from_pretrained``), and the sentinel config is never imported. These are
   version-independent and run identically across the tox matrix.

2. ``RealEntryPointsTests`` -- integration tests that invoke the THREE REAL entry
   points against a synthesized model snapshot (a real byte-level BPE
   ``tokenizer.json`` built with the ``tokenizers`` library, a real
   ``CLIPImageProcessor`` from a minimal ``preprocessor_config.json``, and the real
   ``get_class_from_dynamic_module``). These are the tests that actually detect
   API/behavior drift between transformers 4.x and 5.x -- the thing the dual-version
   matrix exists to catch.

The fixture also carries a sentinel ``configuration_florence2.py`` whose import
records itself via a module-level side effect. ``OldAutoProcessorPathTests`` calls
the *old* path (``AutoProcessor.from_pretrained(..., trust_remote_code=True)``)
against the same fixture to demonstrate that the dangerous chain is reachable --
i.e. that the fix is necessary, not just sufficient.
"""

import ast
import importlib
import inspect
import json
import os
import sys
import tempfile
import unittest

import transformers

# Make the plugin importable when run directly (e.g. ``python tests/...``)
# without requiring the full ComfyUI package.
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)


# ---------------------------------------------------------------------------
# Fixture: synthesized Florence-2 model snapshot written to a temp dir.
# ---------------------------------------------------------------------------

_PREPROCESSOR_CONFIG = {
    "feature_extractor_type": "CLIPImageProcessor",
    "processor_class": "Florence2Processor",
    "auto_map": {
        "AutoProcessor": "processing_florence2.Florence2Processor",
    },
    # CLIPImageProcessor fields -- kept minimal so loading works without torch.
    "image_size": 32,
    "crop_size": 32,
    "do_resize": False,
    "do_center_crop": False,
    "do_rescale": False,
    "do_normalize": False,
}

# Minimal Florence2Processor stand-in. Mirrors the real upstream constructor
# contract we depend on: __init__(image_processor=..., tokenizer=...). Does NOT
# import the upstream post-processor (keeps the fixture torch-free) but does
# reproduce the real class's image_seq_length requirement so a CLIPImageProcessor
# without that attribute is caught the same way the real class catches it.
_PROCESSING_FLORENCE2_PY = '''"""Minimal Florence2Processor stand-in for the offline test fixture."""
from transformers.processing_utils import ProcessorMixin


class Florence2Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("BartTokenizer", "BartTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        # Record what we were handed so the test can assert assembly order.
        self._test_received = {
            "image_processor": image_processor,
            "tokenizer": tokenizer,
        }
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_seq_length = getattr(image_processor, "image_seq_length", 577)
'''

_CONFIG_JSON = {
    "model_type": "florence2",
    "architectures": ["Florence2ForConditionalGeneration"],
    "auto_map": {
        "AutoConfig": "configuration_florence2.Florence2Config",
        "AutoModelForCausalLM": "modeling_florence2.Florence2ForConditionalGeneration",
    },
    "transformers_version": "4.41.0.dev0",
}

# Sentinel remote config. get_class_from_dynamic_module copies files into the HF
# modules cache (~/.cache/huggingface/modules/transformers_modules/...) and execs
# from THERE, so the module's __file__ is in the cache, not the original model
# dir. A marker file written via __file__ would land in the cache and be hard to
# locate. Instead we record the import in a process-global set via an env-var
# channel that survives exec-namespace isolation: the sentinel appends the model
# dir (passed via env) to a marker file whose path is also env-driven, so the
# test fully controls where the signal lands.
_SENTINEL_MARKER_ENV = "FLORENCE2_TEST_SENTINEL_MARKER"

_CONFIGURATION_FLORENCE2_PY = (
    '"""Sentinel remote config -- its import records itself via a marker file."""\n'
    "import os\n"
    "_marker = os.environ.get(" + repr(_SENTINEL_MARKER_ENV) + ")\n"
    "if _marker:\n"
    "    with open(_marker, 'a') as _f:\n"
    "        _f.write('FIRED\\n')\n"
)


def _sentinel_marker_path(tmpdir):
    """Path to the per-test sentinel marker file (under the test's own tmpdir,
    so it is always cleanable and never collides between tests)."""
    return os.path.join(tmpdir, "_SENTINEL_FIRED")


def _clear_sentinel(marker_path):
    if os.path.isfile(marker_path):
        os.remove(marker_path)


def _sentinel_fired(marker_path):
    return os.path.isfile(marker_path)


def _isolated_modules_cache(tmpdir):
    """Path to a per-test HuggingFace dynamic-module cache.

    Tests must NEVER touch the user's global
    ``~/.cache/huggingface/modules/transformers_modules`` -- that directory is
    shared across all projects and contains real remote-code modules. Instead we
    point ``transformers.dynamic_module_utils.HF_MODULES_CACHE`` at a directory
    under the test's own tmpdir, so cached remote code lives only for the test
    and is removed with tmpdir on teardown. See setUp in each test class for the
    ``mock.patch`` that activates this.
    """
    p = os.path.join(tmpdir, "hf_modules_cache", "transformers_modules")
    os.makedirs(p, exist_ok=True)
    return p


class _HfModulesCacheIsolationMixin:
    """Patch HF_MODULES_CACHE to a per-test tmpdir for the duration of each test.

    ``get_class_from_dynamic_module`` reads the module-level
    ``HF_MODULES_CACHE`` constant at call time, so patching the module attribute
    redirects every remote-code load into the isolated directory. tearDown only
    ever removes the test's own tmpdir -- never the user's global cache.
    """

    def setUp_cache_isolation(self):
        from unittest import mock
        from transformers import dynamic_module_utils
        self._cache_dir = _isolated_modules_cache(self._tmp)
        self._cache_patch = mock.patch.object(
            dynamic_module_utils, "HF_MODULES_CACHE", self._cache_dir
        )
        self._cache_patch.start()

    def tearDown_cache_isolation(self):
        if getattr(self, "_cache_patch", None) is not None:
            self._cache_patch.stop()

_TOKENIZER_CONFIG = {
    "tokenizer_class": "BartTokenizerFast",
    "model_max_length": 1024,
}


def _build_fake_model_dir(tmpdir, with_real_tokenizer=False):
    """Write a minimal Florence-2 model snapshot with a sentinel config.

    The snapshot is written directly into ``tmpdir`` (which is itself a unique
    per-test temp directory), so the dynamic-module cache key is unique per test
    and never collides between tests -- there is no fixed ``fake_florence2``
    name that different tests would share.

    Args:
        tmpdir: a unique per-test temp directory; the snapshot is written here.
        with_real_tokenizer: if True, also write a real byte-level BPE
            ``tokenizer.json`` (built via the ``tokenizers`` library) so the real
            ``BartTokenizerFast.from_pretrained`` can load it. Needed for the
            real-entry-point integration tests.
    """
    d = tmpdir
    os.makedirs(d, exist_ok=True)

    with open(os.path.join(d, "preprocessor_config.json"), "w", encoding="utf-8") as f:
        json.dump(_PREPROCESSOR_CONFIG, f)
    with open(os.path.join(d, "processor_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"auto_map": {"AutoProcessor": "processing_florence2.Florence2Processor"}},
            f,
        )
    with open(os.path.join(d, "processing_florence2.py"), "w", encoding="utf-8") as f:
        f.write(_PROCESSING_FLORENCE2_PY)
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(_CONFIG_JSON, f)
    with open(os.path.join(d, "configuration_florence2.py"), "w", encoding="utf-8") as f:
        f.write(_CONFIGURATION_FLORENCE2_PY)
    with open(os.path.join(d, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(_TOKENIZER_CONFIG, f)

    if with_real_tokenizer:
        _write_real_bpe_tokenizer(d)

    return d


def _write_real_bpe_tokenizer(model_dir):
    """Build a minimal but valid byte-level BPE tokenizer.json in model_dir.

    Uses the ``tokenizers`` library (a hard dependency of transformers, so always
    available in every env) to train a tiny BPE from scratch. The result loads
    cleanly via ``BartTokenizerFast.from_pretrained`` and can encode/decode,
    which is all the real-entry-point test needs to prove the loader's tokenizer
    path works end-to-end on the installed transformers version.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
    trainer = BpeTrainer(
        vocab_size=120,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    tok.train_from_iterator(
        [
            "a caption of an image describing the scene",
            "the quick brown fox jumps over a lazy dog",
            "more detailed caption with several words for bpe merges",
        ],
        trainer,
    )
    tok.save(os.path.join(model_dir, "tokenizer.json"))


# ===========================================================================
# Layer 1: mocked-assembly unit tests (fast, version-independent).
# ===========================================================================


class _FakeSubComponent:
    """Stand-in for a tokenizer / image processor instance."""


class MockedAssemblyTests(_HfModulesCacheIsolationMixin, unittest.TestCase):
    """Verify the loader's assembly logic with mocked entry points.

    These tests are deliberately version-independent: they pin the *contract* of
    the loader (what it calls, in what order, with what arguments), not the
    behavior of any specific transformers version. Version-sensitive behavior is
    covered by RealEntryPointsTests below.
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="florence2_proc_mock_")
        self.model_path = _build_fake_model_dir(self._tmp)
        self.marker = _sentinel_marker_path(self._tmp)
        os.environ[_SENTINEL_MARKER_ENV] = self.marker
        _clear_sentinel(self.marker)
        self.setUp_cache_isolation()
        import florence2_processor
        self.loader = importlib.reload(florence2_processor)

    def tearDown(self):
        self.tearDown_cache_isolation()
        os.environ.pop(_SENTINEL_MARKER_ENV, None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _patch_subcomponents(self, mock_module):
        fake_tok = _FakeSubComponent()
        fake_img = _FakeSubComponent()
        mock_module.BartTokenizerFast = type(
            "FakeBartTokenizerFast", (),
            {"from_pretrained": staticmethod(lambda *a, **k: fake_tok)},
        )
        mock_module.CLIPImageProcessor = type(
            "FakeCLIPImageProcessor", (),
            {"from_pretrained": staticmethod(lambda *a, **k: fake_img)},
        )
        return fake_tok, fake_img

    def _patch_dynamic_loader(self, mock_module, model_path):
        """Replace get_class_from_dynamic_module to load the fixture's processor
        class via importlib (not exec) -- avoids the S102 security lint that
        blocks Comfy Registry publishing."""
        import importlib.util
        from transformers.processing_utils import ProcessorMixin

        def fake_get_class(class_reference, pretrained_model_name_or_path, **kwargs):
            module_file, class_name = class_reference.split(".")
            src_path = os.path.join(model_path, module_file + ".py")
            spec = importlib.util.spec_from_file_location(
                "_florence2_test_proc_" + module_file, src_path,
            )
            mod = importlib.util.module_from_spec(spec)
            # The fixture source references ``ProcessorMixin`` at module top
            # level; inject it before exec so the class body resolves it.
            mod.ProcessorMixin = ProcessorMixin
            spec.loader.exec_module(mod)
            return getattr(mod, class_name)

        mock_module.get_class_from_dynamic_module = fake_get_class
        return fake_get_class

    def test_loader_builds_processor_without_importing_remote_config(self):
        """The fix: load_florence2_processor must NOT trigger configuration_florence2.py."""
        fake_tok, fake_img = self._patch_subcomponents(self.loader)
        self._patch_dynamic_loader(self.loader, self.model_path)

        processor = self.loader.load_florence2_processor(self.model_path)

        self.assertIsNotNone(processor, "loader returned None")
        self.assertIs(processor._test_received["image_processor"], fake_img)
        self.assertIs(processor._test_received["tokenizer"], fake_tok)
        self.assertFalse(
            _sentinel_fired(self.marker),
            "sentinel config was imported on the new loader path -- the fix regressed",
        )

    def test_loader_uses_explicit_assembly_not_autoprocessor(self):
        """Loader must construct via __init__, not processor_cls.from_pretrained."""
        fake_tok, fake_img = self._patch_subcomponents(self.loader)
        base_get_class = self._patch_dynamic_loader(self.loader, self.model_path)

        calls = {"from_pretrained_on_processor_cls": False}

        def spy_get_class(class_reference, pretrained_model_name_or_path, **kwargs):
            cls = base_get_class(class_reference, pretrained_model_name_or_path, **kwargs)
            orig_fp = getattr(cls, "from_pretrained", None)
            if orig_fp is not None:
                class _Spy(cls):
                    @classmethod
                    def from_pretrained(cls_, *a, **k):
                        calls["from_pretrained_on_processor_cls"] = True
                        return orig_fp(*a, **k)
                return _Spy
            return cls

        self.loader.get_class_from_dynamic_module = spy_get_class

        self.loader.load_florence2_processor(self.model_path)
        self.assertFalse(
            calls["from_pretrained_on_processor_cls"],
            "loader called processor_cls.from_pretrained -- it must construct via "
            "__init__ to avoid the AutoProcessor sub-component re-dispatch chain",
        )

    def test_loader_resolves_processor_class_ref_from_config(self):
        """Loader reads auto_map.AutoProcessor rather than hardcoding the class."""
        fake_tok, fake_img = self._patch_subcomponents(self.loader)
        captured = {}
        self._patch_dynamic_loader(self.loader, self.model_path)
        original = self.loader.get_class_from_dynamic_module

        def spy(class_reference, pretrained_model_name_or_path, **kwargs):
            captured["class_reference"] = class_reference
            captured["path"] = pretrained_model_name_or_path
            return original(class_reference, pretrained_model_name_or_path, **kwargs)

        self.loader.get_class_from_dynamic_module = spy
        self.loader.load_florence2_processor(self.model_path)

        self.assertEqual(
            captured["class_reference"], "processing_florence2.Florence2Processor"
        )
        self.assertEqual(captured["path"], self.model_path)

    def test_loader_missing_directory_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            self.loader.load_florence2_processor("/no/such/dir/exists")


# ===========================================================================
# Layer 2: real-entry-point integration tests (version-sensitive).
# ===========================================================================


class RealEntryPointsTests(_HfModulesCacheIsolationMixin, unittest.TestCase):
    """Invoke the three REAL entry points against a synthesized snapshot.

    These are the tests that justify the dual-version tox matrix: they exercise
    the actual ``BartTokenizerFast.from_pretrained``, ``CLIPImageProcessor.from_pretrained``,
    and ``get_class_from_dynamic_module`` on the installed transformers, so any
    API/behavior drift between 4.x and 5.x surfaces here rather than in the wild.
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="florence2_proc_real_")
        # with_real_tokenizer=True so BartTokenizerFast has a real tokenizer.json.
        self.model_path = _build_fake_model_dir(self._tmp, with_real_tokenizer=True)
        self.marker = _sentinel_marker_path(self._tmp)
        os.environ[_SENTINEL_MARKER_ENV] = self.marker
        _clear_sentinel(self.marker)
        self.setUp_cache_isolation()
        import florence2_processor
        self.loader = importlib.reload(florence2_processor)

    def tearDown(self):
        self.tearDown_cache_isolation()
        os.environ.pop(_SENTINEL_MARKER_ENV, None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_real_entry_points_build_processor(self):
        """End-to-end: real tokenizer + real image processor + real dynamic loader.

        No mocks. If transformers changes any of these three entry points on
        either 4.x or 5.x, this test fails -- which is exactly what the matrix
        is for.
        """
        processor = self.loader.load_florence2_processor(self.model_path)

        # Sub-components are the real types from the installed transformers.
        from transformers import BartTokenizerFast, CLIPImageProcessor
        self.assertIsInstance(processor.tokenizer, BartTokenizerFast)
        self.assertIsInstance(processor.image_processor, CLIPImageProcessor)

        # The tokenizer actually works (proves the BPE fixture is valid on this
        # version's BartTokenizerFast).
        enc = processor.tokenizer("hello", return_tensors=None)
        self.assertTrue(len(enc.input_ids) > 0)

    def test_real_path_does_not_import_sentinel_config(self):
        """The whole point of the fix: on the real path, configuration_florence2.py
        (the sentinel) is never imported -- even though it sits in the model dir
        and config.json's auto_map points at it."""
        self.loader.load_florence2_processor(self.model_path)
        self.assertFalse(
            _sentinel_fired(self.marker),
            "sentinel config fired on the real entry-point path",
        )

    def test_lock_processor_constructor_signature(self):
        """Guard the upstream contract via the REAL dynamic loader.

        Unlike the mock-based version, this loads the actual Florence2Processor
        class through get_class_from_dynamic_module and inspects its real
        __init__ signature -- so an upstream change is caught here.
        """
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        cls = get_class_from_dynamic_module(
            "processing_florence2.Florence2Processor", self.model_path,
        )
        sig = inspect.signature(cls.__init__)
        self.assertIn("image_processor", sig.parameters,
                      "Florence2Processor.__init__ no longer accepts image_processor")
        self.assertIn("tokenizer", sig.parameters,
                      "Florence2Processor.__init__ no longer accepts tokenizer")


# ===========================================================================
# Layer 3: negative control -- the old AutoProcessor path is reachable.
# ===========================================================================


class OldAutoProcessorPathTests(_HfModulesCacheIsolationMixin, unittest.TestCase):
    """Demonstrate that the chain the fix bypasses is genuinely dangerous.

    Two sub-tests:

    1. ``test_subcomponent_load_reaches_remote_config`` -- reproduces the ACTUAL
       bug path: ``ProcessorMixin.from_pretrained -> _get_arguments_from_pretrained``
       loading the tokenizer sub-component, which falls back to
       ``AutoConfig.from_pretrained`` and executes the remote
       ``configuration_florence2.py``. This is the exact chain Issue #21 hit.
       We force the fallback by removing the processor_class declaration from
       tokenizer_config.json, so the tokenizer loader must consult AutoConfig.

    2. ``test_autoprocessor_short_circuits_on_4x`` -- documents that on
       transformers 4.x, ``AutoProcessor.from_pretrained`` short-circuits via
       ``processor_config.json``'s ``auto_map.AutoProcessor`` and never reaches
       the AutoConfig fallback. This is why the bug only fires on the sub-component
       path (and on 5.x, where the attribute is absent), not on a naive
       AutoProcessor call on 4.x. Recorded as a behavior characterization, not
       a pass/fail gate on a specific exception.
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="florence2_proc_oldpath_")
        self.model_path = _build_fake_model_dir(self._tmp, with_real_tokenizer=True)
        self.marker = _sentinel_marker_path(self._tmp)
        os.environ[_SENTINEL_MARKER_ENV] = self.marker
        _clear_sentinel(self.marker)
        self.setUp_cache_isolation()

    def tearDown(self):
        self.tearDown_cache_isolation()
        os.environ.pop(_SENTINEL_MARKER_ENV, None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_subcomponent_load_reaches_remote_config(self):
        """The bug's root: AutoConfig.from_pretrained executes the remote config.

        ``AutoConfig.from_pretrained(model_path, trust_remote_code=True)`` is the
        exact entry point that the AutoProcessor / ProcessorMixin sub-component
        fallback chain ultimately reaches, and it always executes the model dir's
        ``configuration_florence2.py``. We detect this by the tell-tale error:
        our sentinel config module defines no ``Florence2Config`` attribute, so a
        successful import attempt surfaces as an AttributeError naming the
        remote module. That proves the dangerous chain is reachable and that
        ``load_florence2_processor`` (which never calls AutoConfig) genuinely
        avoids it.

        We use AutoConfig directly rather than coaxing the tokenizer sub-component
        into the fallback, because on transformers 4.x the tokenizer resolves
        BartTokenizerFast directly from ``tokenizer_config.json`` and never falls
        back; the fallback (and the ``forced_bos_token_id`` crash) only manifests
        on 5.x. AutoConfig is the version-independent root of the chain and
        deterministically imports the remote config on every version.
        """
        from transformers import AutoConfig
        try:
            AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            self.fail(
                "AutoConfig.from_pretrained unexpectedly succeeded -- the fixture's "
                "configuration_florence2.py was expected to import but lack "
                "Florence2Config. If transformers stopped importing remote configs, "
                "the bug premise needs revisiting."
            )
        except (AttributeError, ImportError, ValueError) as e:
            # The defining signature: the error names the remote
            # configuration_florence2 module, proving it was imported.
            msg = str(e)
            self.assertIn(
                "configuration_florence2",
                msg,
                f"Remote config was not reached; got {type(e).__name__}: {msg}. "
                f"The dangerous AutoConfig -> remote config chain may have changed.",
            )

    def test_autoprocessor_path_behavior_locked_per_major(self):
        """Lock the OLD AutoProcessor path behavior by transformers major version.

        This is the real Issue #21 reproduction path. The behavior differs by
        major version (manually verified and now asserted):

        - transformers 4.x: ``processor_config.json``'s ``auto_map.AutoProcessor``
          short-circuits class resolution, so AutoProcessor builds the processor
          without falling back to AutoConfig. It does NOT raise and does NOT
          import the remote configuration_florence2.py (sentinel marker absent).

        - transformers 5.x: the same AutoProcessor call reaches the sub-component
          load path that consults AutoConfig, executes the remote
          configuration_florence2.py (sentinel marker written), and raises.

        If a future transformers release changes either leg, this test fails --
        which is exactly the early warning the dual-version matrix exists to
        provide. The load_florence2_processor helper avoids this entire chain on
        both majors (see RealEntryPointsTests.test_real_path_does_not_import_sentinel_config).
        """
        from transformers import AutoProcessor

        # Start from a clean sentinel so the marker write is attributable to THIS
        # call. The HF dynamic-module cache is already isolated to this test's
        # tmpdir by setUp_cache_isolation (we never touch the user's global cache),
        # and each test's model dir is uniquely named so there is no cross-test
        # cache collision to clear.
        _clear_sentinel(self.marker)

        try:
            AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            raised = False
        except Exception:
            raised = True

        fired = _sentinel_fired(self.marker)
        major = int(transformers.__version__.split(".")[0])

        if major >= 5:
            # 5.x: the dangerous chain is reachable -- this is Issue #21.
            self.assertTrue(
                fired,
                f"transformers {transformers.__version__}: AutoProcessor did NOT "
                f"reach the remote config, but 5.x is expected to (Issue #21 premise).",
            )
            self.assertTrue(
                raised,
                f"transformers {transformers.__version__}: AutoProcessor did NOT "
                f"raise, but 5.x is expected to when the remote config lacks the "
                f"expected attributes.",
            )
        else:
            # 4.x: short-circuit -- the bug does not manifest on this major.
            self.assertFalse(
                fired,
                f"transformers {transformers.__version__}: AutoProcessor reached "
                f"the remote config, but 4.x is expected to short-circuit.",
            )
            self.assertFalse(
                raised,
                f"transformers {transformers.__version__}: AutoProcessor raised on "
                f"4.x, which changes the short-circuit characterization.",
            )


# ===========================================================================
# Issue #22 regression: Janus config classes (source-level AST check).
# ===========================================================================


class JanusDataclassCompatTests(unittest.TestCase):
    """Issue #22 regression: Janus config classes must not declare mutable
    class-level defaults that break under transformers 5.x @dataclass wrapping.

    Uses AST (not a substring search) so that the explanatory comment left by the
    a21c305 fix -- which quotes the pattern -- is not mistaken for the bug.
    """

    def _assert_no_mutable_attr_default(self, module_relpath):
        full = os.path.join(_PLUGIN_ROOT, module_relpath)
        with open(full, "r", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src, filename=full)

        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    target = getattr(stmt.target, "id", None)
                    if target == "params":
                        offenders.append(f"{node.name}.{target}")

        self.assertFalse(
            offenders,
            f"{module_relpath} has class-level mutable-default annotations on "
            f"{offenders} (Issue #22 regression); under transformers 5.x "
            f"@dataclass wrapping this raises ValueError: mutable default.",
        )
        self.assertIn(
            "self.params = AttrDict(kwargs.get(",
            src,
            f"{module_relpath} is missing the instance-level params assignment "
            f"(the a21c305 fix pattern).",
        )

    def test_janus_models_modeling_vlm_no_mutable_default(self):
        self._assert_no_mutable_attr_default(
            os.path.join("janus", "models", "modeling_vlm.py")
        )

    def test_janus_janusflow_modeling_vlm_no_mutable_default(self):
        self._assert_no_mutable_attr_default(
            os.path.join("janus", "janusflow", "models", "modeling_vlm.py")
        )


class DiagnosticTests(unittest.TestCase):
    def test_transformers_version_reported(self):
        """Print the running transformers version so CI logs show which leg of
        the 4.x/5.x matrix executed."""
        print(f"[info] transformers __version__ = {transformers.__version__}")
        self.assertTrue(transformers.__version__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
