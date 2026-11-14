"""Part 3m aggregation tests: the generalised coverage/breakdown logic
(synthetic ladder rows, no data, no rollouts) + the freeze-gate
behaviour of the calibration sidecar."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "part3l"))
import aggregate_p3l as AG  # noqa: E402
import aggregate_p3m as AGM  # noqa: E402


def mkrow(variant, member, root, t0, H, speed=4.0, coll=0.0, lc_p=10,
          lc_t=10, dp=100.0, dpt=100.0, t3=5.0, t3t=5.0, dexp=100.0,
          dexp_t=100.0):
    return dict(variant=variant, member=member, root=root, t0=t0, H=H,
                speed_rmse=speed, pos_rmse=1.0,
                collisions_per_1000=coll,
                pred_lc=lc_p, true_lc=lc_t,
                delay_pred=dexp, delay_true=dexp_t,
                pred_vehicle_steps=1000, matched_vehicle_steps=1000,
                delay_phys_pred=dp, delay_phys_true=dpt,
                phys_steps_pred=60, phys_steps_true=60,
                pred_ttc3_exposure=t3, true_ttc3_exposure=t3t,
                pred_hard_brake_s=0.0, true_hard_brake_s=0.0,
                any_ttc3_pred=t3 > 0, any_ttc3_true=t3t > 0,
                any_hb_pred=False, any_hb_true=False, lane_frac=1.0,
                dawdle_scale=1.0, engine="synthetic")


def ens_l6(spread, truth, t3_truth=5.0, t3_centre=5.0, root="r1", t0=120,
           H=30, centre=None):
    """5 L6 members: delay/dexp around `centre` (default `truth`), ttc3
    around `t3_centre`, with the ttc3 truth at `t3_truth`."""
    c = truth if centre is None else centre
    return [mkrow("L6", mi, root, t0, H, dp=c + off * spread,
                  dpt=truth, t3=max(0.0, t3_centre + off * spread),
                  t3t=t3_truth, dexp=c + off * spread, dexp_t=truth)
            for mi, off in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0))]


# ------------------------------------------------- coverage + breakdown --

def test_coverage_and_breakdown_counts():
    # window 1: ttc3 truth INSIDE the raw band; window 2: BELOW it (the
    # real failure direction); window 3: BELOW with truth == 0 exactly
    rows = (ens_l6(1.0, 100.0, root="r1", t0=120)
            + ens_l6(1.0, 100.0, t3_truth=2.0, root="r2", t0=120)
            + ens_l6(1.0, 100.0, t3_truth=0.0, root="r3", t0=120))
    cov = AGM.coverage_and_crps_m(rows, horizons=(30,), widen=None,
                                  variants=("L6",))
    c = cov[("L6", 30)]
    assert c["n_windows"] == 3
    assert c["ttc3_cov_frac"] == pytest.approx(1 / 3, abs=1e-3)
    assert c["ttc3_below_frac"] == pytest.approx(2 / 3, abs=1e-3)
    assert c["ttc3_above_frac"] == 0.0
    assert c["ttc3_below_truth0"] == 1      # the truth == 0 tie
    # the delay views are centred on their truth in this fixture, so
    # only the ttc3 truth moves below the band
    assert c["delay_phys_cov_frac"] == 1.0
    assert c["s_out_ttc3"] is None          # no widening map given
    # CRPS is finite and positive on biased windows
    assert c["crps_phys_mean"] > 0


def test_coverage_widening_restores_membership():
    rows = ens_l6(1.0, 100.0, root="r1", t0=120)
    cov0 = AGM.coverage_and_crps_m(rows, horizons=(30,), widen=None,
                                   variants=("L6",))
    assert cov0[("L6", 30)]["ttc3_cov_frac"] == 1.0   # centred on truth
    # ttc3 ensemble off the truth: raw band misses, widened band covers
    rows_sh = ens_l6(1.0, 100.0, t3_truth=5.0, t3_centre=8.0,
                     root="r1", t0=120)
    cov_raw = AGM.coverage_and_crps_m(rows_sh, horizons=(30,),
                                      widen=None, variants=("L6",))
    cov_wid = AGM.coverage_and_crps_m(rows_sh, horizons=(30,),
                                      widen={"ttc3": {30: 4.0},
                                             "delay_phys": {30: 4.0}},
                                      variants=("L6c",))
    assert cov_raw[("L6", 30)]["ttc3_cov_frac"] == 0.0
    assert cov_wid[("L6c", 30)]["ttc3_cov_frac"] == 1.0


def test_gate_v2_verdict_l6c_uses_same_functions():
    # a PASSing L6c ensemble through the SHARED part 3l verdict function
    rows = ens_l6(1.0, 100.0, root="r1", t0=120)
    cov = AGM.coverage_and_crps_m(rows, horizons=(30,), widen=None,
                                  variants=("L6",))
    summ = AG.summarise(rows)[("L6", 30)]
    l0 = [mkrow("L0", 0, "r1", 120, 30, speed=4.0, dp=1000.0,
                dpt=1200.0)]
    l0_cov = AGM.coverage_and_crps_m(l0, horizons=(30,), widen=None,
                                     variants=())
    g = AG.gate_v2_verdict(summ, cov[("L6", 30)], l0_cov[("L0", 30)],
                           4.0)
    assert g["a_collisions_pass"] and g["b_lc_pass"]
    assert g["c_coverage_crps_pass"] and g["d_per_horizon_pass"]
    assert g["v2_PASS"] and not g["v2_failing"]


def test_widen_map_from_p3l_format():
    p = Path(AGM.__file__).parent / "_tmp_cal.json"
    try:
        p.write_text(json.dumps(dict(
            s_out={"delay_phys": {"30": 1.5, "60": None},
                   "ttc3": {"30": 2.0, "60": 2.5}})), encoding="utf-8")
        w = AGM.widen_map_from(p, horizons=(30, 60))
        assert w == {"delay_phys": {30: 1.5},
                     "ttc3": {30: 2.0, 60: 2.5}}
    finally:
        p.unlink(missing_ok=True)


# --------------------------------------------------------- freeze gate --

def test_freeze_sidecar_roundtrip(tmp_path):
    cal = tmp_path / "calibration_p3m.json"
    text = json.dumps({"s_out": {"ttc3": {"30": 1.0}}})
    cal.write_text(text, encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    side = tmp_path / "calibration_p3m.FROZEN.sha256"
    side.write_text(json.dumps(dict(sha256=sha, frozen_utc="2026-09-27")),
                    encoding="utf-8")
    frozen = json.loads(side.read_text(encoding="utf-8"))
    assert hashlib.sha256(
        cal.read_text(encoding="utf-8").encode("utf-8")).hexdigest() == \
        frozen["sha256"]
    # any later edit breaks the gate
    cal.write_text(text + " ", encoding="utf-8")
    assert hashlib.sha256(
        cal.read_text(encoding="utf-8").encode("utf-8")).hexdigest() != \
        frozen["sha256"]


def test_truthttc3_exposure_synthetic():
    """The added fit target counts over ALL truth rows (no matched-set
    dependence): one flagged pair per tick over 2 ticks -> 1.0 veh s."""
    from types import SimpleNamespace

    import p3m_lib as M

    # rows 0-1 = tick 101, rows 2-3 = tick 102; each tick has one
    # same-lane pair with gap 8, closing 5 -> ttc 1.6 (< 3)
    lane = np.array([1, 1, 1, 1])
    s_by_row = np.array([110.0, 102.0, 120.0, 112.0])
    all_v = np.array([20.0, 25.0, 20.0, 25.0])
    idx = {101: np.array([0, 1]), 102: np.array([2, 3])}

    def tick_rows(k):
        return idx.get(k, np.array([], dtype=np.int64))

    ra = SimpleNamespace(tick_rows=tick_rows, lane_m=lane, s=s_by_row,
                         v=all_v)
    assert M.truth_ttc3_exposure(ra, 100, 2) == pytest.approx(1.0)
    # horizon 0 steps -> zero
    assert M.truth_ttc3_exposure(ra, 100, 0) == 0.0
