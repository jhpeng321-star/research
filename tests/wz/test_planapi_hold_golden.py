"""WM-A: 'hold'-only planning calls stay bit-identical after the 'prog' role was added
(golden captured on 2fb9d454 before any prog code; see planapi_hold_golden.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("torch")

from tests.wz import planapi_hold_golden as G  # noqa: E402


def test_hold_only_evaluate_is_bit_identical():
    expected = json.loads(G.GOLDEN.read_text(encoding="utf-8"))
    got = json.loads(json.dumps(G.compute(), sort_keys=True))
    bad = [k for k in sorted(set(expected) | set(got)) if expected.get(k) != got.get(k)]
    assert not bad, f"hold golden drift: {bad}"
