"""Part 3m engine-knob tests: the pure knob helpers, the default
neutrality of the part 3m engine changes (end_force_s=0.0 /
v_merge_blend=0.0 / run(n_ticks) must leave the shipped behaviour
exactly unchanged, pinned against a committed part 3l raw row), and the
knobs' intended effect (end_force_s increases accepted merges).

The integration tests are skipped when WD01_DATA_ROOT is not set (the
untracked data/weights tree is absent); the pure tests always run."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import engine as E  # noqa: E402
import p3lib as P3  # noqa: E402


# ------------------------------------------------------- pure helpers --

def test_end_force_params_default_off():
    # end_force_s <= 0 is the exact shipped behaviour at ANY s
    assert E.end_force_params(400.0, 1.9, 3.0, 0.0) == (1.9, 3.0)
    assert E.end_force_params(454.09, 1.9, 3.0, 0.0) == (1.9, 3.0)
    assert E.end_force_params(100.0, 2.4, 5.0, -1.0) == (2.4, 5.0)


def test_end_force_params_relaxes_only_near_end():
    # 150 m zone before ACCEL_END_S = 454.10: outside it, unchanged
    assert E.end_force_params(300.0, 1.9, 3.0, 150.0) == (1.9, 3.0)
    # inside it: tau capped at 0.5, b floored at 6.0 (never tightened)
    assert E.end_force_params(310.0, 1.9, 3.0, 150.0) == (0.5, 6.0)
    assert E.end_force_params(310.0, 0.3, 8.0, 150.0) == (0.3, 8.0)
    assert E.end_force_params(454.09, 1.9, 3.0, 150.0) == (0.5, 6.0)


def test_local_target_speed_window_and_empty():
    # d coordinates: two target-lane vehicles at d 90 and 110, mean 20
    d_all = np.array([90.0, 110.0, 500.0])
    v_all = np.array([10.0, 30.0, 99.0])
    on1 = np.array([True, True, False])
    out = E.local_target_speed(d_all, v_all, on1, np.array([100.0]))
    assert out[0] == pytest.approx(20.0)
    # outside the 50 m window of every target vehicle -> -1 sentinel
    out2 = E.local_target_speed(d_all, v_all, on1, np.array([200.0]))
    assert out2[0] == -1.0
    # window edge is inclusive (|d - dq| <= 50)
    out3 = E.local_target_speed(d_all, v_all, on1, np.array([160.0]))
    assert out3[0] == pytest.approx(30.0)


def test_merge_params_default_contains_neutral_knobs():
    # the shipped defaults keep the p3k/p3l behaviour exactly
    assert E.MERGE_PARAMS_DEFAULT["end_force_s"] == 0.0
    assert E.MERGE_PARAMS_DEFAULT["v_merge_blend"] == 0.0


# ---------------------------------------------- integration (data) ----

def _data_available():
    return bool(os.environ.get("WD01_DATA_ROOT", "").strip())


P3L_RAW = (Path(__file__).resolve().parents[1] / "results" / "raw" /
           "raw_p3l_metrics.jsonl")
MEM0 = dict(seed=1, lc_scale=0.00075,
            merge_params=dict(tau_accept=1.9, b_accept=3.0,
                              p_coop_scale=0.0),
            dawdle_scale=1.0)


@pytest.mark.skipif(not _data_available(),
                    reason="WD01_DATA_ROOT not set (untracked data absent)")
def test_default_params_reproduce_p3l_raw_row():
    """Default-neutrality of every part 3m engine change: the L5 member-0
    config rolled on the CURRENT engine must reproduce the committed
    part 3l raw row for (cv20_7005m2000, t0=120, H=30) exactly -- the
    rollout is deterministic, so any behaviour drift would show here."""
    ref = None
    for ln in P3L_RAW.read_text(encoding="utf-8").splitlines():
        r = json.loads(ln)
        if (r["variant"] == "L5" and r["member"] == 0
                and r["root"] == "cv20_7005m2000" and int(r["t0"]) == 120
                and int(r["H"]) == 30):
            ref = r
            break
    assert ref is not None, "reference raw row missing"
    dev = P3.device()
    r = E.Rollout("cv20_7005m2000", 120.0, kind="idm_merge",
                  member_seed=MEM0["seed"], lc_scale=MEM0["lc_scale"],
                  merge_params=dict(MEM0["merge_params"]),
                  dawdle_scale=MEM0["dawdle_scale"],
                  dev=dev).run(n_ticks=60)
    m = E.window_metrics(r.snapshots, r.ra, r.k0, H=30)
    assert m["pred_lc"] == ref["pred_lc"]
    assert m["true_lc"] == ref["true_lc"]
    assert m["pred_ttc3_exposure"] == pytest.approx(
        ref["pred_ttc3_exposure"])
    assert m["true_ttc3_exposure"] == pytest.approx(
        ref["true_ttc3_exposure"])
    assert m["speed_rmse"] == pytest.approx(ref["speed_rmse"], abs=1e-12)
    assert m["collisions_per_1000"] == pytest.approx(
        ref["collisions_per_1000"])


@pytest.mark.skipif(not _data_available(),
                    reason="WD01_DATA_ROOT not set (untracked data absent)")
def test_end_force_increases_accepted_merges():
    """The knob's intended effect: relaxing acceptance near the lane end
    strictly weakens the M2 rejection terms, so accepted merges can only
    grow (same seeds)."""
    dev = P3.device()
    mp0 = dict(MEM0["merge_params"])
    mp150 = dict(MEM0["merge_params"], end_force_s=150.0)
    acc = {}
    for tag, mp in (("off", mp0), ("on", mp150)):
        tot = 0
        for t0 in (120.0, 300.0):
            r = E.Rollout("cv20_7005m2000", float(t0), kind="idm_merge",
                          member_seed=1, lc_scale=0.00075,
                          merge_params=dict(mp), dev=dev).run(n_ticks=240)
            tot += r.merge_stats["accepted"]
        acc[tag] = tot
    assert acc["on"] >= acc["off"]
    assert acc["on"] > 0
