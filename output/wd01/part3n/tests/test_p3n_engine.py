"""Part 3n integration tests (require WD01_DATA_ROOT; skipped
otherwise): engine default-neutrality is inherited from part 3m (the
engine is UNCHANGED in part 3n), and the whole part 3n raw basis is
pinned by re-rolling one committed TASK A window exactly."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import p3n_lib as N

HERE = Path(__file__).resolve().parent
PART3N = HERE.parent


def _data_root():
    return os.environ.get("WD01_DATA_ROOT", "").strip()


@pytest.mark.skipif(not _data_root(), reason="WD01_DATA_ROOT not set")
def test_task_a_window_reproduces_committed_row():
    """Re-roll cv20_7002m2000 t0=120 with the p3m member-0 config and
    require EXACT equality with the committed TASK A raw row (same
    engine, same seeds -> byte-identical metrics)."""
    raw = PART3N / "results" / "raw" / "raw_train_internal_p3n.jsonl"
    if not raw.exists():
        pytest.skip("TASK A raw rows not present")
    want = None
    for ln in raw.read_text(encoding="utf-8").splitlines():
        r = json.loads(ln)
        if (r["variant"] == "L6" and r["member"] == 0 and
                r["root"] == "cv20_7002m2000" and r["t0"] == 120 and
                r["H"] == 30):
            want = r
            break
    if want is None:
        pytest.skip("target row not present")
    import p3lib as P3
    fit = N.load_json(PART3N.parent / "part3m" / "results" /
                      "fit_p3m.json")
    m0 = fit["members"][0]
    dev = P3.device()
    r = N.P3_roll("cv20_7002m2000", 120.0, dict(m0["merge_params"]),
                  float(m0["lc_scale"]), dev,
                  float(m0.get("dawdle_scale", 1.0)), seed=int(m0["seed"]))
    got = N.window_raw_rows(r, "L6", 0, "cv20_7002m2000", 120, 1.0,
                            "test")[0]
    for k in ("speed_rmse", "pos_rmse", "collisions_per_1000",
              "pred_lc", "true_lc", "delay_phys_pred",
              "delay_phys_true", "pred_ttc3_exposure",
              "true_ttc3_exposure", "pred_vehicle_steps",
              "matched_vehicle_steps"):
        assert got[k] == pytest.approx(float(want[k]), rel=1e-12,
                                       abs=1e-9), k


@pytest.mark.skipif(not _data_root(), reason="WD01_DATA_ROOT not set")
def test_holdout_roots_refused_without_registration():
    """The physical holdout block is authoritative (inherited)."""
    import p3lib as P3
    sp = N.L3.splits_3l()
    for root in sp["holdout"]:
        with pytest.raises(Exception):
            P3.root_arrays(root)
