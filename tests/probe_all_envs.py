"""
Cross-environment import probe for `comfyui_caption_this` / `ComfyUI-CaptionThis`.

Runs the same import smoke test against three independent Python environments
(one Python process per env) and prints a pass/fail summary.

    python tests/probe_all_envs.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The inner script is built as a normal string template (NOT an f-string) so that
# it survives the trip through `python -c <script>` without any interpolation
# ambiguity. The three {PARENT} / {REPO} / {PLUGIN} placeholders below are
# substituted via `.format(...)`.
INNER_SCRIPT_TEMPLATE = r'''
import sys, importlib, traceback
print("python:", sys.version.split()[0])
print("transformers:", __import__("transformers").__version__)

sys.path.insert(0, r"{REPO}")
plugin_name = "{PLUGIN}"
sys.path.insert(0, r"{PARENT}")

for n in list(sys.modules):
    if (n.startswith("janus")
        or n == plugin_name
        or n.startswith(plugin_name + ".")
        or n == "ComfyUI-CaptionThis"
        or n.startswith("ComfyUI-CaptionThis.")):
        sys.modules.pop(n, None)

print()

try:
    plugin = importlib.import_module(plugin_name)
    print(f"  plugin {plugin_name} OK")
except Exception as e:
    print(f"  plugin import FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(2)

try:
    mm = importlib.import_module(plugin_name + ".janus.models")
    mvm = importlib.import_module(plugin_name + ".janus.models.modeling_vlm")
    classes = [n for n in ("VisionConfig","AlignerConfig","GenVisionConfig","GenAlignerConfig","GenHeadConfig","MultiModalityConfig") if hasattr(mvm, n)]
    print(f"  Config classes available on modeling_vlm: {classes}")
    vc = mvm.VisionConfig()
    print(f"  VisionConfig().params type: {type(vc.params).__name__}")
    print(f"  VLChatProcessor exists: {hasattr(mm, 'VLChatProcessor')}")
    print(f"  MultiModalityCausalLM exists: {hasattr(mm, 'MultiModalityCausalLM')}")
except Exception as e:
    print(f"  modeling_vlm FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(3)

print("RESULT: PASS")
'''


ENVS = [
    {
        "label": "V8.0 (transformers 4.56.2, attrdict installed)",
        "py": r"E:\FF\ComfyUI_Mie_2026_V8.0\python_embeded\python.exe",
        "parent": r"E:\FF\ComfyUI_Mie_2026_V8.0\ComfyUI\custom_nodes",
        "plugin": "ComfyUI-CaptionThis",
    },
    {
        "label": "V9.0 (transformers 5.9.0, attrdict installed)",
        "py": r"E:\HH\Package\ComfyUI_Mie_2026_V9.0\python_embeded\python.exe",
        "parent": r"E:\HH\Package\ComfyUI_Mie_2026_V9.0\ComfyUI\custom_nodes",
        "plugin": "comfyui_caption_this",
    },
    {
        "label": "V9.0_cu126 (transformers 5.9.0, attrdict installed)",
        "py": r"E:\HH\Package\ComfyUI_Mie_2026_V9.0_cu126\python_embeded\python.exe",
        "parent": r"E:\HH\Package\ComfyUI_Mie_2026_V9.0_cu126\ComfyUI\custom_nodes",
        "plugin": "comfyui_caption_this",
    },
]


def run_env(env):
    inner = (INNER_SCRIPT_TEMPLATE
             .replace("{REPO}", str(REPO))
             .replace("{PARENT}", env["parent"])
             .replace("{PLUGIN}", env["plugin"]))
    print("=" * 80)
    print(f"Running env probe: {env['label']}")
    print("=" * 80)
    proc = subprocess.run(
        [env["py"], "-c", inner],
        capture_output=True, text=True, timeout=120,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        clean_err = "\n".join(
            ln for ln in proc.stderr.splitlines()
            if "FutureWarning" not in ln
            and "warnings.warn" not in ln
            and "_warn_unsupported_code" not in ln
            and "If you want to use the NVIDIA" not in ln
            and "compute capability" not in ln
            and "Please follow the instructions" not in ln
            and "The current PyTorch install supports" not in ln
            and "Please install pytorch" not in ln
        )
        if clean_err.strip():
            print("--- stderr ---")
            print(clean_err, end="")
    return proc.returncode == 0


def main():
    rows = []
    for env in ENVS:
        ok = run_env(env)
        rows.append((env["label"], "PASS" if ok else "FAIL"))
        print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for label, status in rows:
        print(f"  [{status}] {label}")
    failed = [r for r in rows if r[1] == "FAIL"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
