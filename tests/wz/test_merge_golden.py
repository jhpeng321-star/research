"""WM-A: the map port keeps the ORIGINAL merge-map WM-S bit-identical (golden captured pre-port).

``tests/wz/data/merge_golden.json`` was written by ``python tests/wz/golden_merge.py --capture`` on
the unmodified merged code (c4/wz-wm-port bd9bd5d6 = clean3 base + c4/mainstart).  This test
recomputes everything with the CURRENT code under the merge-map road (explicitly scoped AND via
the default process road) and requires every sha256 / float.hex / int / string to match exactly:
pack conversion, decoded training windows, a fresh round-16 loop-FT model's forward/backward loss
record, one optimizer step, and the physics primitives.  CPU only; ~3 min.
"""
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

from tests.wz import golden_merge  # noqa: E402


def _diff(expected: dict, got: dict) -> list[str]:
    bad = []
    for key in sorted(set(expected) | set(got)):
        if expected.get(key, "<missing>") != got.get(key, "<missing>"):
            bad.append(f"{key}: {expected.get(key, '<missing>')!r} != {got.get(key, '<missing>')!r}")
    return bad


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(golden_merge.GOLDEN.read_text(encoding="utf-8"))


def test_merge_map_pipeline_is_bit_identical(tmp_path, golden):
    from wms.core import road as R

    R._reset_for_tests()
    with R.road_scope(R.MERGE_ORIGINAL):
        got = golden_merge.compute(tmp_path)
    bad = _diff(golden, json.loads(json.dumps(got, sort_keys=True)))
    assert not bad, "merge-map golden drift:\n" + "\n".join(bad[:40])
