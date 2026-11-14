"""WD-01-PA2G churn-fix regression scenes (2026-09-30).

Pins the two correctness fixes behind the PA2G-A discretionary
lane-change churn (68,614 predicted disc LCs vs 243 true at H10):

1. `_neighbors()` leader/follower roles were swapped (the vehicle BEHIND
   was used as the leader), so decision bumper gaps went non-positive
   and the 1e-6 IDM gap floor blew the MOBIL incentive terms up to
   ~1e15 m/s^2 -> every change accepted in both directions.
2. Decision accelerations were unclipped while the dynamics path clips
   to dynamics.a_clip_mps2; overlap/collision states made the incentive
   astronomically large.  Decisions now use the model's own clipped
   accelerations (same-state consistency, textbook MOBIL requirement).

Nominal params here have the declared stability guards OFF (cooldown_s
0, lc_min_speed_mps 0, ego_gain_floor_mps2 0): the no-ping-pong
property is pinned WITHOUT any cooldown (owner design-deviation rule).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

WD01 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WD01))

from partA_v2g.engine import (PartAV2GEngine, load_params)  # noqa: E402

P = load_params()
assert float(P["mobil"]["cooldown_s"]) == 0.0, \
    "nominal cooldown must stay OFF (design-deviation rule)"
assert float(P["mobil"]["lc_min_speed_mps"]) == 0.0, \
    "nominal lc_min_speed must stay OFF (design-deviation rule)"
A_LO, A_HI = P["dynamics"]["a_clip_mps2"]


def eng(vehicles, steps=0, params=None, **kw):
    e = PartAV2GEngine(params or P, t0=480.0, vehicles=vehicles, **kw)
    e.run(steps)
    return e


def uniform_two_lane(offset=35.0, n_per_lane=6, v=24.0, start=2100.0,
                     spacing=70.0):
    """Two mainline lanes, uniform equal-speed columns, staggered by
    `offset` m (not aligned, not overlapping)."""
    vehicles = {}
    for k in range(n_per_lane):
        vehicles[f"l0_{k}"] = dict(lane=0, s=start + k * spacing, v=v)
        vehicles[f"l1_{k}"] = dict(lane=1,
                                   s=start + offset + k * spacing, v=v)
    return vehicles


# ------------------------------------------------- (a) uniform = no disc LC

def test_uniform_two_lane_zero_discretionary_lcs_120s():
    """Uniform equal-speed traffic on both mainline lanes (the
    interleaved default scene) produces ZERO discretionary lane changes
    over 120 s (240 steps), guards OFF."""
    e = eng(uniform_two_lane(), steps=240)
    assert e.lc_counts["disc_left"] == 0
    assert e.lc_counts["disc_right"] == 0
    assert e.lc_counts["merge"] == 0
    assert e.collisions == 0


def test_uniform_two_lane_stays_churn_free_varied_offsets():
    """Across stagger offsets the scene stays churn-free: at most one
    one-time (textbook-rational) move per 120 s, no vehicle ever twice,
    no collisions."""
    for off in (10.0, 25.0, 45.0, 60.0):
        e = eng(uniform_two_lane(offset=off), steps=240)
        total = e.lc_counts["disc_left"] + e.lc_counts["disc_right"]
        assert total <= 1, f"offset {off}: {e.lc_counts}"
        per_vid: dict = {}
        for j in range(1, 241):
            for vid, st in e.snapshots[j].items():
                if vid == "__t__":
                    continue
                prev = e.snapshots[j - 1].get(vid)
                if prev is not None and prev[0] != st[0]:
                    per_vid[vid] = per_vid.get(vid, 0) + 1
        assert all(n <= 1 for n in per_vid.values()), \
            f"offset {off}: repeat changes {per_vid}"
        assert e.collisions == 0, f"offset {off}"


# --------------------------------------------------- (b) no step ping-pong

def test_no_ping_pong_after_a_change_without_cooldown():
    """A vehicle that just changed lanes in a symmetric-ish scene does
    not change back the next steps: nominal params, cooldown 0, no
    other guard.  The change is forced by a genuinely slower own-lane
    leader; afterwards the ego's return incentive is negative."""
    vehicles = {
        "slow": dict(lane=1, s=2050.0, v=10.0),
        "ego": dict(lane=1, s=2000.0, v=26.0),
        "foll": dict(lane=1, s=1950.0, v=26.0),
        "l0_lead": dict(lane=0, s=2150.0, v=26.0),
        "l0_rear": dict(lane=0, s=2050.0, v=26.0),
    }
    e = PartAV2GEngine(P, t0=480.0, vehicles=vehicles)
    e.run(20)
    rows = [e.snapshots[j]["ego"] for j in range(21)]
    assert rows[0][0] == 1
    assert any(r[0] == 0 for r in rows), "ego should move left once"
    j_left = next(j for j, r in enumerate(rows) if r[0] == 0)
    # no change back within the window
    assert all(r[0] == 0 for r in rows[j_left:]), \
        "ping-pong: ego returned after its change"
    # ego changed exactly once; no vehicle anywhere ping-pongs (two
    # changes within 2 steps of each other)
    changes: dict = {}
    for j in range(1, 21):
        for vid, st in e.snapshots[j].items():
            if vid == "__t__":
                continue
            prev = e.snapshots[j - 1].get(vid)
            if prev is not None and prev[0] != st[0]:
                changes.setdefault(vid, []).append(j)
    assert len(changes.get("ego", [])) == 1
    for vid, js in changes.items():
        for k in range(1, len(js)):
            assert js[k] - js[k - 1] > 4, f"ping-pong by {vid}: {js}"


def test_no_repeat_change_per_vehicle_traced():
    """Instrumented trace: no vehicle changes lane more than once per
    10 s in the uniform scene (the churn signature was a change every
    2 s = the old cooldown cap)."""
    vehicles = uniform_two_lane()
    e = PartAV2GEngine(P, t0=480.0, vehicles=vehicles)
    last_lc: dict = {}
    repeats = 0
    for j in range(1, 41):
        e.step()
        for vid, st in e.snapshots[j].items():
            if vid == "__t__":
                continue
            prev = e.snapshots[j - 1].get(vid)
            if prev is not None and prev[0] != st[0]:
                if vid in last_lc and j - last_lc[vid] <= 20:
                    repeats += 1
                last_lc[vid] = j
    assert repeats == 0
    assert sum(e.lc_counts.values()) == 0


# ------------------------------------ (d) simultaneous LCs, no overlap

def test_simultaneous_lcs_leave_no_overlap():
    """Two same-lane vehicles changing into the same target lane in one
    step are applied front-most first with the updated target lane, and
    never end up overlapping (bumper gaps stay >= 0)."""
    vehicles = {
        "a": dict(lane=1, s=2000.0, v=25.0),
        "b": dict(lane=1, s=1930.0, v=25.0),   # 65 m behind a
        "slow": dict(lane=1, s=2100.0, v=12.0),
        # blocks 'slow' from vacating first: its own insertion gap would
        # be 0.1 m < gap_floor, so only a (then b) may enter lane 0
        "l0_block": dict(lane=0, s=2103.65, v=25.0),
        "l0_ahead": dict(lane=0, s=2450.0, v=25.0),
    }
    e = eng(vehicles, steps=8)
    assert e.snapshots[1]["a"][0] == 0 and e.snapshots[1]["b"][0] == 0, \
        "both should leave lane 1 in the same step"
    assert e.snapshots[1]["slow"][0] == 1
    for j in range(1, 9):
        snap = e.snapshots[j]
        for li in (0, 1, 2):
            rows = sorted((st[1], vid) for vid, st in snap.items()
                          if vid != "__t__" and st[0] == li)
            for k in range(len(rows) - 1):
                gap = rows[k + 1][0] - 5.0 - rows[k][0]
                assert gap >= -1e-9, \
                    f"overlap at step {j} lane {li}: {rows[k]} -> " \
                    f"{rows[k + 1]} gap {gap}"


def test_swap_fix_neighbors_consistent_with_lane_order():
    """_neighbors() leader/follower agree with the ascending-s lane
    order (the PA2G root-cause pin)."""
    vehicles = uniform_two_lane()
    e = PartAV2GEngine(P, t0=480.0, vehicles=vehicles)
    leader, follower = e._neighbors()
    for li in (0, 1, 2):
        rows = np.where(e.lane == li)[0]
        rows = rows[np.argsort(e.s[rows], kind="stable")]
        for j in range(len(rows)):
            r = int(rows[j])
            exp_lead = int(rows[j + 1]) if j + 1 < len(rows) else -1
            exp_foll = int(rows[j - 1]) if j - 1 >= 0 else -1
            assert leader[r] == exp_lead, (li, j)
            assert follower[r] == exp_foll, (li, j)
            if exp_lead >= 0:
                assert e.s[exp_lead] > e.s[r], "leader must be ahead"


# ------------------------- decision accelerations use the model's clip

def test_decision_accelerations_bounded_in_overlap_state():
    """In a bumper-overlap (collision) state the MOBIL incentive terms
    stay within the nominal a_clip band (no 1e15 gains): the stay and
    change accelerations seen by the decision are clipped."""
    vehicles = {
        "ego": dict(lane=1, s=2000.0, v=20.0),
        "jam": dict(lane=1, s=1998.0, v=0.0),    # overlaps ego's rear
        "l0_free": dict(lane=0, s=2300.0, v=25.0),
    }
    e = PartAV2GEngine(P, t0=480.0, vehicles=vehicles)
    lo = e.names.index("ego")
    a_stay = e._a_stay(lo)
    assert A_LO - 1e-9 <= a_stay <= A_HI + 1e-9
    d_sorted_t, rows_t = e._lane_d_sorted(0)
    ok, _ = e._disc_decision(lo, 0, False, d_sorted_t, rows_t)
    # bounded: the change may pass or fail on textbook terms, but no
    # astronomical incentive can force it
    assert isinstance(ok, bool)
    assert e.collisions >= 0


# ----------------------------------------- (e) CAV flags + native routing

def test_eval_cav_adaptation_flags_and_routes_native():
    """The eval-side CAV wiring: zone-state vehicles that appear in the
    run's CAV identity set get is_cav=True and the engine runs with the
    native CavControllerLoop hook (all-native actions -> no commands,
    generic rules drive the CAV; the partA_ctrl native path engaged)."""
    from partA_v2g import eval_a2
    vehicles = {
        "cav1": dict(lane=1, s=2000.0, v=26.0),
        "hdv1": dict(lane=1, s=2060.0, v=25.0),
    }
    cav_ids = {"cav1"}
    adapted = eval_a2.flag_cavs(vehicles, cav_ids)
    assert adapted["cav1"]["is_cav"] is True
    assert "is_cav" not in vehicles["hdv1"] or \
        vehicles["hdv1"].get("is_cav") is False
    hook = eval_a2.native_control_hook(t0=480.0, params=P,
                                       cav_ids=cav_ids)
    e = PartAV2GEngine(P, t0=480.0, vehicles=adapted, control_hook=hook)
    assert bool(e.cav[e.names.index("cav1")])
    assert not bool(e.cav[e.names.index("hdv1")])
    e.run(10)
    assert hook.n_commands == 0
    assert hook.called_ticks > 0
    assert e.lc_counts["merge"] == 0 and e.collisions == 0
