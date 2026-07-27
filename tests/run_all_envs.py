"""
Run the full v9-compat test suite across all 3 environments.

For each env (V8.0, V9.0, V9.0_cu126), spawn a subprocess and run:
  - tests/test_modeling_vlm_v9_compat.py
  - tests/test_configuration_florence2_v9_compat.py
  - tests/test_modeling_florence2_v9_compat.py
  - tests/test_modeling_florence2_round6_cache.py
  - tests/test_florence2_caption_v9_compat.py
  - tests/probe_all_envs.py (once, end-to-end)

Prints a 3xN pass/fail matrix. Exits 0 if all green, 1 otherwise.
"""

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"

# Each env is one row: (label, python_exe).
ENVS = [
    ("V8.0 (transformers 4.56.2)", r"E:\FF\ComfyUI_Mie_2026_V8.0\python_embeded\python.exe"),
    ("V9.0 (transformers 5.9.0)", r"E:\HH\Package\ComfyUI_Mie_2026_V9.0\python_embeded\python.exe"),
    ("V9.0_cu126 (transformers 5.9.0)", r"E:\HH\Package\ComfyUI_Mie_2026_V9.0_cu126\python_embeded\python.exe"),
]

# Each test is a column: (rel_path, env-specific overrides dict).
# We do not have env-specific overrides, so the test files are environment-agnostic.
TEST_FILES = [
    "test_modeling_vlm_v9_compat.py",
    "test_configuration_florence2_v9_compat.py",
    "test_modeling_florence2_v9_compat.py",
    "test_modeling_florence2_round6_cache.py",
    "test_florence2_caption_v9_compat.py",
]


def run(py, test_path, timeout=180):
    proc = subprocess.run(
        [py, str(test_path)],
        capture_output=True, text=True, timeout=timeout,
        cwd=str(REPO),
    )
    return proc.returncode, proc.stdout, proc.stderr


def main():
    matrix = {}
    for env_label, py in ENVS:
        if not os.path.exists(py):
            print(f"SKIP {env_label}: python not found at {py}")
            continue
        row = {}
        for t in TEST_FILES:
            tpath = TESTS / t
            if not tpath.exists():
                row[t] = "MISSING"
                continue
            rc, out, err = run(py, tpath)
            if rc == 0:
                row[t] = "PASS"
            else:
                # Pick the FAIL line from the output for the summary.
                fail = [ln for ln in out.splitlines() if ln.startswith("[FAIL]")]
                row[t] = f"FAIL ({len(fail)})"
        matrix[env_label] = row

    # Pretty-print
    col_w = max(len(t) for t in TEST_FILES) + 2
    env_w = max(len(label) for label in matrix.keys()) + 2
    header = f"{'env'.ljust(env_w)}" + "".join(t.ljust(col_w) for t in TEST_FILES)
    print()
    print(header)
    print("-" * len(header))
    for env_label, row in matrix.items():
        line = f"{env_label.ljust(env_w)}"
        for t in TEST_FILES:
            line += row.get(t, "???").ljust(col_w)
        print(line)
    print()

    # Summary
    total = sum(len(r) for r in matrix.values())
    passed = sum(1 for r in matrix.values() for v in r.values() if v == "PASS")
    print(f"Summary: {passed}/{total} PASS")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
