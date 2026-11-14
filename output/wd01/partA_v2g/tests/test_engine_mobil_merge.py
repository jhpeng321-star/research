"""MOBIL + mandatory-merge rule scenes, and collision accounting."""
import sys
from pathlib import Path

import numpy as np
import pytest

WD01 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WD01))

from partA_v2g.engine import (PartAV2GEngine, load_params, merge_b_safe,
                              merge_threshold, merge_urgency, mobil_gain,
                              mobil_passes, mobil_safe)

P = load_params()
MOBIL = P["mobil"]
MERGE = P["merge"]
Z0 = MERGE["zone_s0_m"]
Z1 = MERGE["zone_s1_m"]
LZ = MERGE["zone_len_m"]


def eng(vehicles, steps=0, **kw):
    e = PartAV2GEngine(P, t0=480.0, vehicles=vehicles, **kw)
    e.run(steps)
    return e


# ------------------------------------------------------------------ MOBIL

def test_mobil_gain_politeness_weight():
    gain = mobil_gain(np.array([0.3]), np.array([0.0]),
                      np.array([-0.2]), np.array([0.0]),
                      np.array([0.0]), np.array([0.0]), MOBIL)
    # 0.3 + 0.5 * (-0.2 + 0.0)
    assert float(gain[0]) == pytest.approx(0.3 - 0.1)


def test_politeness_blocks_harmful_change():
    """Ego gains slightly; the new follower would lose a lot.  With the
    nominal p = 0.5 the change is blocked; a selfish p = 0 passes."""
    d_new_foll = -1.0
    p_nominal = dict(MOBIL, politeness_p=0.5)
    p_selfish = dict(MOBIL, politeness_p=0.0)
    g_nom = mobil_gain(np.array([0.25]), np.array([0.0]),
                       np.array([d_new_foll]), np.array([0.0]),
                       np.array([0.0]), np.array([0.0]), p_nominal)
    g_slf = mobil_gain(np.array([0.25]), np.array([0.0]),
                       np.array([d_new_foll]), np.array([0.0]),
                       np.array([0.0]), np.array([0.0]), p_selfish)
    assert not bool(mobil_passes(g_nom, p_nominal, to_right=True)[0])
    assert bool(mobil_passes(g_slf, p_selfish, to_right=True)[0])


def test_safety_rejects_hard_new_follower_braking():
    assert not bool(mobil_safe(np.array([-4.5]), MOBIL)[0])
    assert bool(mobil_safe(np.array([-3.9]), MOBIL)[0])


def test_keep_right_bias_asymmetry():
    """The same gain passes rightward (bias earned) and fails leftward
    (bias paid)."""
    gain = np.array([0.15])
    assert bool(mobil_passes(gain, MOBIL, to_right=True)[0])   # >= -0.2
    assert not bool(mobil_passes(gain, MOBIL, to_right=False)[0])  # < 0.4


def test_discretionary_move_to_free_lane():
    """A slow leader on main_right, free main_left: the follower moves
    left (disc_left) within a few steps."""
    vehicles = {
        "lead": dict(lane=1, s=2000.0, v=5.0),
        "ego": dict(lane=1, s=1950.0, v=28.0),
    }
    e = eng(vehicles, steps=6)
    assert e.lc_counts["disc_left"] >= 1


# ------------------------------------------------------- mandatory merge

def test_urgency_and_b_safe_ramp():
    u0 = float(merge_urgency(np.array([Z0]), MERGE)[0])
    u1 = float(merge_urgency(np.array([Z1]), MERGE)[0])
    assert u0 == pytest.approx(0.0)
    assert u1 == pytest.approx(1.0)
    assert float(merge_b_safe(np.array([0.0]), MERGE)[0]) == pytest.approx(4.0)
    assert float(merge_b_safe(np.array([1.0]), MERGE)[0]) == pytest.approx(6.0)
    assert float(merge_threshold(np.array([0.0]), MOBIL)[0]) == \
        pytest.approx(0.1)
    assert float(merge_threshold(np.array([1.0]), MOBIL)[0]) == \
        pytest.approx(0.0)


def _merge_scene(ego_s, foll_gap, foll_v, ego_v=20.0):
    """Accel vehicle in the zone + a lane-1 follower behind its d."""
    d_i = 378.66 - ego_s                       # accel-lane d of the ego
    # same d on lane 1: s = OFFSETS[1] - d
    foll_s = 2644.22 - (d_i + foll_gap)
    vehicles = {
        "ego": dict(lane=2, s=ego_s, v=ego_v),
        "foll": dict(lane=1, s=foll_s, v=foll_v),
    }
    return vehicles


def test_urgent_merger_accepted_near_lane_end():
    """Near the lane end the relaxed b_safe (up to 6) accepts a follower
    required deceleration of ~5.2 m/s^2; early in the zone b_safe ~ 4.2
    rejects it."""
    # follower at 22 m/s behind a 20 m/s merger, 10 m d-gap:
    # b_req = (22^2-20^2)/(2*(10-2)) = 5.25
    vehicles = _merge_scene(ego_s=448.0, foll_gap=10.0, foll_v=22.0)
    e = eng(vehicles, steps=2)
    assert e.lc_counts["merge"] >= 1          # accepted (b_safe ~ 5.84)
    vehicles_early = _merge_scene(ego_s=385.0, foll_gap=10.0, foll_v=22.0)
    e2 = eng(vehicles_early, steps=2)
    # early: b_safe ~ 4.17 < 5.25 -> rejected this step
    assert e2.snapshots[1]["ego"][0] == 2


def _stopped_queues(ego_s, ego_v, ahead_m=120.0, spacing=6.0):
    """Dense STOPPED queues on both mainline lanes starting exactly at
    the ego's d, ANCHORED at the front (the queue cannot dissolve: the
    anchor stays stopped because its bumper gap < s*)."""
    d_ego = 378.66 - ego_s
    vehicles = {"ego": dict(lane=2, s=ego_s, v=ego_v)}
    for li in (0, 1):
        off = P["geometry"]["lane_offsets_m"][li]
        s0q = (2644.22 - d_ego) - (off - 2644.22) if li == 0 \
            else (2644.22 - d_ego)
        s = s0q
        k = 0
        while s < s0q + ahead_m:
            vehicles[f"q{li}_{k}"] = dict(lane=li, s=s, v=0.0)
            s += spacing
            k += 1
    return vehicles


def test_stop_at_lane_end_when_no_gap():
    """Dense stopped queues on BOTH mainline lanes from the ego d: no
    slot opens while the ego decelerates, it stops AT the lane end (the
    mandatory-merge 'no gap' outcome).  The queue eventually discharges
    (the back of a standing jam does), so the assertion window is the
    30 s it takes the ego to arrive and stop."""
    e = eng(_stopped_queues(430.0, 12.0), steps=60)
    assert e.lc_counts["merge"] == 0
    ego_rows = [e.snapshots[j]["ego"] for j in range(61)]
    assert all(r[0] == 2 for r in ego_rows)          # never left lane 2
    assert ego_rows[-1][2] < 0.5                     # stopped
    assert ego_rows[-1][1] <= Z1 + 1e-6              # at / before the end
    assert ego_rows[-1][1] > Z1 - 8.0                # actually AT the end
    assert e.anomalies.get("accel_end_clamped", 0) == 0


def test_gap_acceptance_merges_into_free_slot():
    vehicles = _merge_scene(ego_s=400.0, foll_gap=60.0, foll_v=20.0)
    e = eng(vehicles, steps=4)
    assert e.lc_counts["merge"] == 1
    rows = [e.snapshots[j]["ego"] for j in range(5)]
    assert any(r[0] != 2 for r in rows)      # the ego left the accel lane
    j_star = next(j for j, r in enumerate(rows) if r[0] != 2)
    assert j_star >= 1
    # no teleport: the merge step advances by at most one step of travel
    prev = e.snapshots[j_star - 1]["ego"]
    d_before = 378.66 - prev[1]
    d_after = 2644.22 - rows[j_star][1] if rows[j_star][0] == 1 \
        else 2645.67 - rows[j_star][1]
    assert d_after <= d_before + 0.5 * prev[2] + 1e-6


# ------------------------------------------------------------ collisions

def test_collision_counted_never_repaired():
    """A follower placed overlapping the leader counts collisions; the
    model never teleports them apart (lane changes frozen; a tight
    stopped leader keeps the front vehicle at rest)."""
    vehicles = {
        "lead2": dict(lane=1, s=2001.5, v=0.0),   # gap 1.5 < s*: keeps
        "lead": dict(lane=1, s=2000.0, v=0.0),    # 'lead2' stopped
        "foll": dict(lane=1, s=1997.0, v=0.0),    # bumper gap -2 m
    }
    e = PartAV2GEngine(P, t0=480.0, vehicles=vehicles,
                       lane_changes_enabled=False)
    e.run(4)
    assert e.collisions >= 4
    s_f = e.snapshots[4]["foll"][1]
    s_l = e.snapshots[4]["lead"][1]
    assert s_f >= 1996.0          # no teleport backwards
    assert s_l - 5.0 - s_f <= 0.0 + 1e-9   # still overlapping (not repaired)


def test_accel_end_never_overshot():
    """A lane-2 vehicle with blocked target slots stops at/before the
    lane end; the numerics never carry s past it."""
    e = eng(_stopped_queues(440.0, 12.0), steps=60)
    assert e.snapshots[60]["ego"][0] == 2
    assert e.snapshots[60]["ego"][1] <= Z1
    assert e.snapshots[60]["ego"][2] < 0.5
    assert e.anomalies.get("accel_end_clamped", 0) == 0
