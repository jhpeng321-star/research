"""New ``wms.eval`` vs the legacy ``wms_eval`` on the C1 checkpoint (real data; skipped if absent).

The new side runs the REAL ``wms.model`` / ``wms.data`` stack (``wms.eval.backend.load_backend``);
since WP-I no legacy network is involved on that side.  Legacy results come from a subprocess
(name collision of the ``wms`` package) and are cached under ``runs/tmp_eq`` (git-ignored).
The 20-state subset is gated by ``WMS_EQ_FULL=1`` (about an hour of CPU).
"""
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
STATES = Path("E:/rd-wt/ws-wallc1/_out/wallc1_states_v2/wall_same.json")
CKPT = Path("E:/research-dense/runs/active/WM-S-C1FAST-20261006/box/full_struct/wm_s.pt")
RECEIPT = CKPT.with_name("TRAIN_RECEIPT.json")
CACHE = REPO / "runs" / "tmp_eq"
HORIZONS = (10.0, 30.0, 60.0)
QUICK_60S_MEDIAN_M = 11.9576   # bisect_stages SUMMARY C1_stage120 (final ckpt ~ stage 120)

pytestmark = pytest.mark.skipif(
    not (STATES.is_file() and CKPT.is_file() and RECEIPT.is_file()),
    reason="C1 checkpoint / wall_same states not available")


def _legacy(mode: str) -> list[dict]:
    out = CACHE / f"legacy_{mode}.json"
    if not out.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        staged = CACHE / "wm_s.pt"      # the legacy loader requires the ckpt inside the repo
        if not staged.is_file():
            staged.write_bytes(CKPT.read_bytes())
        subprocess.run([sys.executable, str(Path(__file__).with_name("legacy_eval_run.py")),
                        str(STATES), str(staged), str(RECEIPT), mode, str(out)],
                       check=True, cwd=REPO)
    return json.loads(out.read_text(encoding="utf-8"))


def _new(selector, *args) -> list[dict]:
    from wms.eval.predictor import WmsPredictor
    from wms.eval.score import ScoreSpec, evaluate_state

    physics = json.loads(RECEIPT.read_text(encoding="utf-8"))["physics"]
    predictor = WmsPredictor.from_checkpoint(CKPT, physics=physics)
    spec = ScoreSpec(horizons_s=HORIZONS, predict_horizon_s=60.0)
    return [evaluate_state(rec, predictor.predict_states, spec) for rec in selector(STATES, *args)]


def _assert_equal(new_rows, old_rows, tol=1e-6):
    assert [(r["world"], r["t0_s"], r["key"]) for r in new_rows] == \
        [(r["world"], r["t0_s"], r["key"]) for r in old_rows]
    for new, old in zip(new_rows, old_rows):
        assert sorted(new["arms"]) == sorted(old["arms"])
        for arm, block in old["arms"].items():
            assert new["arms"][arm]["error"] is None
            for h in HORIZONS:
                for stratum in ("all", "hdv", "cav"):
                    a = new["arms"][arm]["car"][str(h)][stratum]
                    b = block["car"][str(h)][stratum]
                    for field, value in b.items():
                        assert a[field] == pytest.approx(value, abs=tol, rel=1e-4), (arm, h, field)


def test_first_three_states_match_legacy():
    from wms.eval.subsets import first_states

    new_rows = _new(first_states, 3)
    _assert_equal(new_rows, _legacy("first3"))


@pytest.mark.skipif(os.environ.get("WMS_EQ_FULL") != "1", reason="set WMS_EQ_FULL=1 (slow)")
def test_quick_subset_matches_legacy_and_bisect_median():
    from wms.eval.subsets import first_per_world

    new_rows = _new(first_per_world, 20)
    # float32 thread-count nondeterminism across processes reaches ~2e-5 m on this subset
    _assert_equal(new_rows, _legacy("per_world20"), tol=1e-3)
    medians = [r["arms"]["wall"]["car"]["60.0"]["hdv"]["ds_med"] for r in new_rows]
    assert len(medians) == 20
    assert statistics.median(medians) == pytest.approx(QUICK_60S_MEDIAN_M, abs=0.001)
