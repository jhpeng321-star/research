"""WD-01-PA2G-CNT1 registered injector tests (REGISTRATION_PA2G_CNT1.md
section 7): insertion schedule and safe-gap rule, lane alternation, the
no-count-after-w guard, FIFO queue, queue-delay convention, and the
nominal byte-identity with the injector OFF."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from partA_v2g import arrivals as ARR
from partA_v2g import metrics as MX
from partA_v2g.engine import PartAV2GEngine, load_params

PARAMS = load_params()


def mk_engine(vehicles, t0=0.0):
    return PartAV2GEngine(PARAMS, t0=t0, vehicles=vehicles)


# ------------------------------------------------------------- schedule

def test_schedule_evenly_spaced_lambda_count_over_W():
    counts = {"main": 10, "ramp": 0}
    w, H = 100.0, 120.0
    sched = ARR.insertion_schedule(counts, w, H)
    lam = counts["main"] / 60.0
    dues = [d["due"] - w for d in sched if d["entrance"] == "main"]
    # (k - 0.5)/lambda <= H  ->  (k - 0.5)*6 <= 120 -> k <= 20.5 -> 20
    assert len(dues) == 20
    for k, d in enumerate(dues, start=1):
        assert abs(d - (k - 0.5) / lam) < 1e-9
    assert np.allclose(np.diff(dues), 60.0 / counts["main"])
    # ramps carry their own lambda from their own count
    sched2 = ARR.insertion_schedule({"main": 0, "ramp": 4}, 0.0, 120.0)
    dues_r = [d["due"] for d in sched2]
    assert [d["k"] for d in sched2] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert np.allclose(np.diff(dues_r), 60.0 / 4.0)


def test_schedule_boundary_and_zero_rate():
    # due exactly at w + H is kept ((k - 0.5)/lam == H)
    s = ARR.insertion_schedule({"main": 1, "ramp": 0}, 0.0, 30.0)
    assert [d["k"] for d in s] == [1]
    assert abs(s[0]["due"] - 30.0) < 1e-9
    # beyond w + H is not scheduled
    assert ARR.insertion_schedule({"main": 1, "ramp": 0}, 0.0, 29.9) == []
    # zero count -> zero rate -> no vehicles
    assert ARR.insertion_schedule({"main": 0, "ramp": 0}, 0.0, 120.0) == []
    # sorted by due time across entrances (main first on exact ties)
    s2 = ARR.insertion_schedule({"main": 6, "ramp": 6}, 0.0, 60.0)
    dues = [d["due"] for d in s2]
    assert dues == sorted(dues)
    assert s2[0]["entrance"] == "main"


def test_lane_alternation_main_from_lane1_ramp_single():
    assert [ARR.lane_for("main", k) for k in (1, 2, 3, 4, 5, 6)] == \
        [1, 0, 1, 0, 1, 0]
    assert all(ARR.lane_for("ramp", k) == 2 for k in range(1, 10))
    with pytest.raises(ValueError):
        ARR.lane_for("bogus", 1)


# ------------------------------------------------------- counter reader

def test_entrance_classification_network_edges():
    assert ARR.classify_departure("1191554636") == "main"
    assert ARR.classify_departure("11553048#1") == "ramp"
    assert ARR.classify_departure("1191554636 ") == "main"
    assert ARR.classify_departure("somewhere_else") is None


def test_no_count_after_w_is_ever_read():
    w = 100.0
    deps = [(10.0, "main"), (40.0, "main"),      # 10.0 and the half-open
            (59.9, "ramp"), (60.0, "main"),      # lower edge 40.0 excluded
            (100.0, "main")]
    base = ARR.counts_from_departures(deps, w, 60.0)
    assert base == {"main": 2, "ramp": 1}
    # sentinel future rows (any epsilon after w, and far future) must
    # leave every count unchanged for any W: future departures are
    # filtered BEFORE counting
    polluted = deps + [(w + 1e-6, "main"), (w + 0.5, "ramp"),
                       (150.0, "main"), (1e9, "ramp")]
    for W in (60.0, 30.0, 1.0):
        assert ARR.counts_from_departures(polluted, w, W) == \
            ARR.counts_from_departures(deps, w, W)


# ------------------------------------------------------ insertion rules

def test_insertion_speed_and_safe_gap_accept():
    eng = mk_engine({"lead": dict(lane=1, s=40.0, v=10.0)})
    sched = ARR.insertion_schedule({"main": 6, "ramp": 0}, 0.0, 60.0)
    inj = ARR.ArrivalInjector(eng, sched)
    inj._pump(1)          # due step of k=1 is ceil(5/0.5) = 10
    assert not inj.inserted
    inj._pump(10)
    assert len(inj.inserted) == 1
    rec = inj.inserted[0]
    assert rec["lane"] == 1 and rec["v_ins"] == 10.0   # min(31.29, 10)
    # the vehicle is at s = 0 in lane 1 with the inserted speed
    r = len(eng.names) - 1
    assert eng.names[r] == ARR.injected_vid("main", 1)
    assert eng.lane[r] == 1 and eng.s[r] == 0.0 and eng.v[r] == 10.0
    assert eng.cav[r] == False  # noqa: E712 (generic HDV)


def test_insertion_safe_gap_block_and_retry():
    # gap 15 < s0 + v_ins*T = 2 + 10*1.5 = 17 -> blocked, retried later
    eng = mk_engine({"lead": dict(lane=1, s=20.0, v=10.0)})
    sched = ARR.insertion_schedule({"main": 6, "ramp": 0}, 0.0, 60.0)
    inj = ARR.ArrivalInjector(eng, sched)
    inj._pump(10)
    assert not inj.inserted and inj.blocked_steps == 1
    assert len(inj._queues["main"]) == 1
    # the leader drives on; the head is retried each step until the gap
    # reaches s0 + v_ins*T (the engine stepping is simulated here by
    # moving the leader the engine's own way: +v*dt per step)
    inserted_at = None
    for j in range(11, 61):
        eng.s[0] += eng.v[0] * PARAMS["dt_s"]
        inj._pump(j)
        if inj.inserted:
            inserted_at = j
            break
    assert inserted_at is not None
    gap_now = eng.s[0] - eng.length[0]
    assert gap_now >= PARAMS["idm"]["s0_m"] + \
        inj.inserted[0]["v_ins"] * PARAMS["idm"]["T_s"] - 0.5 * 10.0 + 1e-9


def test_fifo_head_blocks_queue_no_overtaking():
    # the RAMP is single-lane, so queued vehicles share one target lane:
    # a blocked head must block the whole queue (strict FIFO)
    eng = mk_engine({"blk": dict(lane=2, s=6.0, v=0.0)})
    # lambda = 12/60 = 0.2 -> dues 2.5, 7.5, 12.5 -> due steps 5, 15, 25
    sched = ARR.insertion_schedule({"main": 0, "ramp": 12}, 0.0, 60.0)
    inj = ARR.ArrivalInjector(eng, sched)
    inj._pump(4)
    assert not inj.inserted and not inj._queues["ramp"]
    inj._pump(5)    # k=1 due: gap 1 < s0 + v_ins*T = 2 -> blocked
    assert not inj.inserted and len(inj._queues["ramp"]) == 1
    inj._pump(15)
    assert len(inj._queues["ramp"]) == 2 and not inj.inserted, \
        "k=2 must queue behind the blocked head (strict FIFO)"
    inj._pump(25)
    assert len(inj._queues["ramp"]) == 3 and not inj.inserted
    # free the gap: the head (k=1) inserts first, v_ins = min(22.22, 0)=0
    eng.s[0] = 60.0
    inj._pump(26)
    assert [r["k"] for r in inj.inserted] == [1]
    # now k=2 faces k=1 at s=0 in the SAME lane (bumper gap -5 < s0)
    # and stays queued
    inj._pump(27)
    assert [r["k"] for r in inj.inserted] == [1]
    assert len(inj._queues["ramp"]) == 2
    eng.s[1] = 60.0     # k=1's row; move it out of the way
    inj._pump(28)
    assert [r["k"] for r in inj.inserted] == [1, 2]


def test_ramp_insertion_single_lane_and_limit():
    eng = mk_engine({"lead2": dict(lane=2, s=50.0, v=15.0)})
    sched = ARR.insertion_schedule({"main": 0, "ramp": 6}, 0.0, 60.0)
    inj = ARR.ArrivalInjector(eng, sched)
    inj._pump(10)
    assert len(inj.inserted) == 1
    rec = inj.inserted[0]
    assert rec["lane"] == 2
    assert rec["v_ins"] == 15.0        # min(ramp limit 22.22, 15)
    r = len(eng.names) - 1
    assert eng.names[r] == ARR.injected_vid("ramp", 1)
    assert eng.lane[r] == 2 and eng.s[r] == 0.0


def test_empty_lane_inserts_at_limit():
    eng = mk_engine({})
    sched = ARR.insertion_schedule({"main": 6, "ramp": 0}, 0.0, 60.0)
    inj = ARR.ArrivalInjector(eng, sched)
    inj._pump(10)
    assert inj.inserted[0]["v_ins"] == pytest.approx(31.29)


# ------------------------------------------------- nominal byte identity

def test_nominal_byte_identity_with_injector_off():
    vehicles = {f"v{i}": dict(lane=i % 3, s=100.0 + 17.0 * i,
                              v=20.0 + (i % 5)) for i in range(9)}
    e1 = PartAV2GEngine(PARAMS, t0=0.0, vehicles=vehicles)
    e1.run(40)
    e2 = PartAV2GEngine(PARAMS, t0=0.0, vehicles=vehicles)
    ARR.ArrivalInjector(e2, []).run(40)   # empty schedule: degenerate
    assert e1.snapshots == e2.snapshots
    assert e1.collisions == e2.collisions
    assert e1.collisions_by_step == e2.collisions_by_step
    assert e1.lc_counts == e2.lc_counts
    assert e1.exited == e2.exited


def test_queue_delay_accrues_only_after_insertion():
    eng = mk_engine({})
    sched = ARR.insertion_schedule({"main": 6, "ramp": 0}, 0.0, 60.0)
    inj = ARR.ArrivalInjector(eng, sched)
    inj.run(20)                            # k=1 inserted at step 10
    vid = ARR.injected_vid("main", 1)
    snaps = MX.snaps_list(eng)
    assert all(vid not in sn for sn in snaps[:10])
    assert vid in snaps[10]
    # no delay before the first physical step (the queue convention:
    # identical to the truth, which counts delay from the physics rows
    # a vehicle acquires only once SUMO has inserted it)
    ser = MX.WindowSeries(snaps, [vid])
    assert ser.stats(9, PARAMS)["delay"] == 0.0
    assert ser.stats(20, PARAMS)["delay"] > 0.0


def test_injected_vehicles_counted_in_vehicle_steps():
    eng = mk_engine({})
    sched = ARR.insertion_schedule({"main": 6, "ramp": 0}, 0.0, 60.0)
    inj = ARR.ArrivalInjector(eng, sched)
    inj.run(20)
    snaps = MX.snaps_list(eng)
    # steps 1..9: engine empty; k=1 inserted at step 10 -> present in
    # snapshots 10..20 = 11 injected vehicle-steps in the denominator
    vsteps = sum(len(snaps[jj]) for jj in range(1, 21))
    assert vsteps == 11
    s = inj.summary(20)
    assert s["inserted"] == {"main": 1, "ramp": 0}
    # nothing waits: k=2 (due step 30) is not due yet at step 20
    assert s["still_queued"] == {"main": 0, "ramp": 0}


def test_injector_takes_no_type_input():
    """No vType / CAV flag / penetration ever enters the injector: the
    engine hook's signature carries identity + state only, and the
    injected record has no type field."""
    sig = inspect.signature(PartAV2GEngine.insert_vehicle)
    assert set(sig.parameters) == {"self", "vid", "lane", "s", "v",
                                   "length_m"}
    src = inspect.getsource(ARR.ArrivalInjector)
    for bad in ("vtype", "penetration", "is_cav", "type_id"):
        assert bad not in src
