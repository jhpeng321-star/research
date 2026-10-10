"""Run the pre-restructure ``wms`` package in a subprocess (it shares the name ``wms``).

The legacy package lives in ``output/wd01/wm_n1``; importing it in-process would clash
with the new ``wms``.  Tests that compare against it skip when torch or the tree is absent.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
OLD_ROOT = ROOT / "output" / "wd01" / "wm_n1"


def run_old(code: str) -> Any:
    """Execute ``code`` in the legacy tree; it must ``print`` one JSON document."""
    if not (ROOT / "archive" / "wms_era" / "wm_n1" / "wms" / "train.py").is_file():
        pytest.skip("legacy wms package not present in this checkout")
    result = subprocess.run([sys.executable, "-c", code], cwd=OLD_ROOT, capture_output=True,
                            text=True, timeout=300)
    if result.returncode != 0:
        if "No module named 'torch'" in result.stderr:
            pytest.skip("torch not importable for the legacy package")
        raise AssertionError(result.stderr[-2000:])
    return json.loads(result.stdout)
