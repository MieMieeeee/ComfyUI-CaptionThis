"""Florence2 processor loader that bypasses the AutoProcessor -> AutoConfig chain.

Why this exists
---------------
Issue #21: on transformers 5.x, the unconditional
``AutoProcessor.from_pretrained(model_path, trust_remote_code=True)`` call in
``florence2_caption.py`` loaded the model repo's remote ``configuration_florence2.py``,
which reads ``self.forced_bos_token_id`` without a getattr guard. transformers 5.x
no longer binds generation-default attributes onto the config instance
(pop-and-discard in ``__post_init__``), so that read raises ``AttributeError``.

This module assembles the processor explicitly:

    CLIPImageProcessor.from_pretrained(model_path)      # no AutoConfig involved
    BartTokenizerFast.from_pretrained(model_path, ...)  # tokenizer_config.json
    Florence2Processor(image_processor=..., tokenizer=...)  # __init__ only

By building the two sub-components ourselves and calling the dynamically-loaded
``Florence2Processor`` class via its constructor (not its ``from_pretrained``), we
skip the ``ProcessorMixin.from_pretrained -> _get_arguments_from_pretrained`` path
that re-dispatches through AutoConfig and re-triggers the remote config bug.

This module deliberately imports nothing from ComfyUI (no ``folder_paths``, no
``comfy.model_management``) so it can be unit-tested in an isolated venv.
"""

import json
import os

from transformers import BartTokenizerFast, CLIPImageProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import logging

logger = logging.get_logger(__name__)

# The processor class reference shipped by every Florence-2 model repo we support.
# Matches ``auto_map.AutoProcessor`` in the repo's preprocessor_config.json /
# processor_config.json. Centralised so a future change is a one-line edit.
_DEFAULT_PROCESSOR_CLASS_REF = "processing_florence2.Florence2Processor"


def _read_processor_auto_map(model_path):
    """Return the ``auto_map.AutoProcessor`` value for the model repo.

    Looks in ``processor_config.json`` then ``preprocessor_config.json`` (the two
    files AutoProcessor itself checks, in that order). Falls back to the known
    default for Florence-2 if neither file declares it -- this keeps loading
    working for repos that rely on the conventional filename.
    """
    for fname in ("processor_config.json", "preprocessor_config.json"):
        fpath = os.path.join(model_path, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Failed to read {fpath} while loading Florence2 processor: {e}"
            ) from e
        auto_map = cfg.get("auto_map") or {}
        ref = auto_map.get("AutoProcessor")
        if ref:
            return ref
    # Conventional default for Florence-2 repos -- avoids a hard failure when the
    # repo simply omits the auto_map but ships processing_florence2.py.
    return _DEFAULT_PROCESSOR_CLASS_REF


def load_florence2_processor(model_path):
    """Build a Florence2Processor without going through AutoProcessor.

    Steps:
      1. Resolve the processor class reference from the repo's config files.
      2. Build ``CLIPImageProcessor`` (image_processing_utils never touches
         AutoConfig -- always safe).
      3. Build ``BartTokenizerFast`` (safe when tokenizer_config.json declares
         ``tokenizer_class``, which Florence-2 repos do).
      4. Load the ``Florence2Processor`` *class* via the dynamic-module API and
         construct it directly with the two sub-components.

    .. warning::
        This loader **executes the model repo's ``processing_florence2.py``**.
        That is inherent to Florence-2 (the processor is custom code shipped with
        the weights, not a built-in transformers class). ``get_class_from_dynamic_module``
        does not have a ``trust_remote_code`` parameter (it accepts and ignores
        arbitrary kwargs) and it *always* runs the resolved module -- so unlike
        ``AutoProcessor.from_pretrained``, there is no opt-in gate to expose. The
        security-relevant difference from the old path is *what* code runs: this
        loader runs only ``processing_florence2.py`` and the two named component
        classes, never the dangerous ``configuration_florence2.py`` that reads
        ``forced_bos_token_id``.

    Args:
        model_path: Local directory of the downloaded Florence-2 model snapshot.

    Returns:
        An assembled processor instance.

    Raises:
        FileNotFoundError: if ``model_path`` does not exist.
        RuntimeError: if the processor class reference cannot be resolved.
        TypeError: if the loaded processor's constructor does not accept
            ``image_processor`` / ``tokenizer`` (upstream signature change).
    """
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Florence2 model directory does not exist: {model_path}"
        )

    processor_class_ref = _read_processor_auto_map(model_path)
    if not processor_class_ref or "." not in processor_class_ref:
        raise RuntimeError(
            f"Could not resolve a Florence2 processor class reference for "
            f"{model_path!r} (got {processor_class_ref!r}). The model snapshot "
            f"may be incomplete; expected processing_florence2.Florence2Processor."
        )

    # Step 2: image processor -- no AutoConfig involvement, always safe.
    image_processor = CLIPImageProcessor.from_pretrained(model_path)  # nosec B615

    # Step 3: tokenizer. Florence-2 repos' tokenizer_config.json declares
    # ``tokenizer_class: BartTokenizerFast``, so the tokenizer loader resolves
    # the class directly and never falls back to AutoConfig.from_pretrained (the
    # fallback that would re-trigger the remote config).
    tokenizer = BartTokenizerFast.from_pretrained(model_path)  # nosec B615

    # Step 4: load the processor *class* (not an instance) and construct it
    # directly. This skips ProcessorMixin.from_pretrained, which would otherwise
    # re-dispatch sub-component loading and re-trigger the remote config.
    processor_cls = get_class_from_dynamic_module(
        processor_class_ref,
        model_path,
    )

    # Guard the upstream constructor contract: if a future Florence2Processor
    # drops/renames these params we want a clear error, not a silent misbuild.
    try:
        processor = processor_cls(
            image_processor=image_processor, tokenizer=tokenizer
        )
    except TypeError as e:
        raise TypeError(
            f"The loaded {processor_cls.__name__} from {model_path!r} rejected "
            f"(image_processor=..., tokenizer=...): {e}. The upstream processor "
            f"constructor signature may have changed; this loader needs updating."
        ) from e

    return processor
