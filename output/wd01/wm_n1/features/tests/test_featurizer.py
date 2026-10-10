"""Featurizer unit tests on synthetic physics fixtures (no SUMO)."""
from __future__ import annotations

import numpy as np
import pytest

from features import constants as C
from features import featurizer as F
from synth_features import (ACCEL, MAIN_LEFT, MAIN_RIGHT, plan_row, row, veh,
                   write_run)

EMPTY_LABELS = dict(outcomes_root=None)


def slot_at(sh, k, vid):
    vidx = sh["slot_vid"].index(vid)
    cols = np.flatnonzero(sh["slot_index"][k] == vidx)
    return int(cols[0]) if len(cols) else -1


def _build(tmp_path, rows, arm="base", t0=0.0, **kw):
    root = write_run(tmp_path / arm, rows, **kw)
    return F.build_shard("chunk4", "wsynth", arm, root, split="TRAIN",
                         plan_row=plan_row(t0), outcomes_root=None)


def test_tensor_dtypes_match_format(tmp_path):
    rows = [row(0.5, {"mainline.0": veh(MAIN_RIGHT, 10, 30)})]
    sh = _build(tmp_path, rows)
    expected = {
        "veh_step": np.float16, "veh_mask": np.uint8,
        "slot_index": np.int32, "plan": np.float16,
        "decision_t": np.float64, "wall": np.float16,
        "wall_mask": np.uint8, "counter": np.float16,
        "accel_next": np.float16, "target_mask": np.uint8,
        "lane_next": np.int8, "arrivals": np.int16,
    }
    for name, dt in expected.items():
        assert sh[name].dtype == dt, (name, sh[name].dtype, dt)


def test_slot_assignment_stable_and_reused(tmp_path):
    rows = [
        row(0.5, {"mainline.0": veh(MAIN_RIGHT, 10, 30),
                  "mainline.1": veh(MAIN_RIGHT, 20, 28)}),
        row(1.0, {"mainline.0": veh(MAIN_RIGHT, 10.5, 30),
                  "mainline.1": veh(MAIN_RIGHT, 20.5, 28)}),
        row(1.5, {"mainline.0": veh(MAIN_RIGHT, 11, 30)}),
        row(2.0, {"mainline.0": veh(MAIN_RIGHT, 11.5, 30),
                  "mainline.2": veh(MAIN_RIGHT, 30, 27)},
            departed=["mainline.2"]),
    ]
    sh = _build(tmp_path, rows)
    si = sh["slot_index"]
    slot_vid = sh["slot_vid"]

    def slot_at(k, vid):
        vidx = slot_vid.index(vid)
        cols = np.flatnonzero(si[k] == vidx)
        return int(cols[0]) if len(cols) else -1

    # rows start at t=0.5 -> tick 1 (t_origin = 0)
    sa = [slot_at(k, "mainline.0") for k in (1, 2, 3, 4)]
    assert len(set(sa)) == 1 and sa[0] >= 0
    assert slot_at(1, "mainline.1") == slot_at(2, "mainline.1") >= 0
    # leaving vehicle frees its slot; the new vehicle reuses it
    assert slot_at(3, "mainline.1") == -1
    assert slot_at(4, "mainline.2") == slot_at(1, "mainline.1")


def test_lane_and_action_one_hot_and_junction(tmp_path):
    rows = [
        row(0.5, {"mainline.0": veh(MAIN_RIGHT, 10, 30),
                  "ramp.0": veh(ACCEL, 10, 20)},
            departed=["ramp.0"]),
        row(1.0, {"mainline.0": veh(MAIN_RIGHT, 10.5, 30),
                  "ramp.0": veh(":1780739017_1_1", 3, 20)}),
    ]
    sh = _build(tmp_path, rows)
    v0 = slot_at(sh, 1, "mainline.0")
    r0 = slot_at(sh, 1, "ramp.0")
    assert sh["veh_step"][1, v0, C.LANE_RIGHT] == 1.0
    assert sh["veh_step"][1, r0, C.LANE_ACCEL] == 1.0
    r0b = slot_at(sh, 2, "ramp.0")
    assert sh["veh_step"][2, r0b, C.JUNCTION_IDX] == 1.0
    # junction row carries the last physical s_down
    assert sh["veh_step"][2, r0b, C.S_DOWN] == sh["veh_step"][1, r0, C.S_DOWN]
    assert sh["veh_step"][1, v0, C.PRESENT_IDX] == 1.0


def test_step_targets_hand_computed(tmp_path):
    rows = [
        row(0.5, {"mainline.0": veh(MAIN_RIGHT, 10, 30, a=1.0),
                  "mainline.1": veh(MAIN_RIGHT, 20, 25, a=-0.5)}),
        row(1.0, {"mainline.0": veh(MAIN_RIGHT, 10.5, 30, a=2.0),
                  "mainline.1": veh(MAIN_LEFT, 20.5, 24, a=0.0)},
            departed=["mainline.2"]),
        row(1.5, {"mainline.0": veh(MAIN_RIGHT, 11.0, 30, a=0.0)}),
    ]
    sh = _build(tmp_path, rows)
    a = slot_at(sh, 1, "mainline.0")
    b = slot_at(sh, 1, "mainline.1")
    assert sh["accel_next"][1, a] == pytest.approx(2.0, abs=1e-3)
    assert sh["target_mask"][1, a] == 1
    assert sh["lane_next"][1, a] == C.LANE_STAY
    assert sh["lane_next"][1, b] == C.LANE_LEFT_CLS
    # mainline departure at t=1.0 (tick 2)
    assert int(sh["arrivals"][2, 0]) == 1
    assert int(sh["arrivals"][:, 1].sum()) == 0


def test_counter_bins_only_past(tmp_path):
    rows = [
        row(0.5, {"mainline.0": veh(MAIN_RIGHT, 10, 30)},
            departed=["mainline.0"]),
        row(1.0, {"mainline.0": veh(MAIN_RIGHT, 10.5, 30),
                  "ramp.0": veh(ACCEL, 5, 20)},
            departed=["ramp.0"]),
        row(1.5, {"mainline.0": veh(MAIN_RIGHT, 11, 30)}),
    ]
    sh = _build(tmp_path, rows)
    # at t=1.5, window (t-300, t]: mainline dep at 0.5 -> age 1.0 -> bin 0
    c = sh["counter"][3]
    assert c[0, 0] == 1.0
    assert c[1, 0] == 1.0
    # no departure has ever occurred at a future tick, so no bin anomaly
    assert c.sum() == sh["arrivals"][:4].sum()


def test_pi_token_is_planned_not_damping(tmp_path):
    sch = dict(run="wsynth__pi", schedule=dict(events=[
        dict(vehicle_id="mainline.9", time_s=1.0, action="pi-external-u")]))
    cmds = [dict(kind="command", timestamp_s=1.5,
                 vehicle_id="mainline.9", owner="damping")]
    rows = [
        row(1.5, {"mainline.9": veh(MAIN_RIGHT, 100, 30)},
            cav_ids=["mainline.9"]),
        row(2.0, {"mainline.9": veh(MAIN_RIGHT, 100.5, 30)},
            cav_ids=["mainline.9"]),
    ]
    root = write_run(tmp_path / "pi", rows, controller=None, schedule=sch,
                     commands=cmds)
    sh = F.build_shard("chunk4", "wsynth", "pi", root, split="TRAIN",
                       plan_row=plan_row(1.0), outcomes_root=None)
    s = slot_at(sh, 1, "mainline.9")
    assert sh["veh_step"][1, s, C.ACT_PI] == 1.0
    assert sh["veh_step"][1, s, C.ACT_DAMPING] == 0.0
    # plan at decision t0 = 1.0, segment 0 -> pi one-hot (t0+0.5 state)
    assert sh["plan"][0, s, 0, C.PLAN_ACTION_SLOTS + C.ACTIONS.index("pi")] == 1.0


def test_wall_token_and_member_override(tmp_path):
    walls = [
        dict(kind="wall_request", wall_id="W1", timestamp_s=1.0,
             members=["mainline.9"], wall_speed_mps=20.0, hold_s=50.0),
        dict(kind="wall_event", wall_id="W1", timestamp_s=1.0,
             event="assembly_start"),
        dict(kind="wall_event", wall_id="W1", timestamp_s=1.5,
             event="assembled"),
    ]
    rows = [
        row(0.5, {"mainline.9": veh(MAIN_RIGHT, 100, 30)},
            cav_ids=["mainline.9"]),
        row(1.0, {"mainline.9": veh(MAIN_RIGHT, 100.5, 30)},
            cav_ids=["mainline.9"]),
        row(2.0, {"mainline.9": veh(MAIN_RIGHT, 101, 30)},
            cav_ids=["mainline.9"]),
    ]
    root = write_run(tmp_path / "wall", rows, controller=None, walls=walls)
    sh = F.build_shard("chunk4", "wsynth", "wall", root, split="TRAIN",
                       plan_row=plan_row(0.0), outcomes_root=None)
    s = slot_at(sh, 4, "mainline.9")
    assert sh["wall_mask"][4, 0] == 1
    assert sh["veh_step"][4, s, C.WALL_MEMBER_IDX] == 1.0
    assert sh["veh_step"][4, s, C.ACT_WALL] == 1.0
    assert sh["wall"][4, 0, 2] == 1.0               # holding phase one-hot


def test_no_forbidden_inputs_change_features(tmp_path):
    base = {"mainline.0": veh(MAIN_RIGHT, 10, 30, a=0.5)}
    r1 = [row(0.5, base)]
    r2 = [row(0.5, {"mainline.0": veh(MAIN_RIGHT, 10, 30, a=0.5,
                                      type_id="aggressive_car",
                                      speed_mode=27)})]
    sh1 = _build(tmp_path / "a", r1)
    sh2 = F.build_shard("chunk4", "wsynth_other", "base",
                        write_run(tmp_path / "b" / "base", r2),
                        split="TRAIN", plan_row=plan_row(0.0),
                        outcomes_root=None)
    assert np.array_equal(sh1["veh_step"], sh2["veh_step"])
    assert np.array_equal(sh1["veh_mask"], sh2["veh_mask"])


@pytest.mark.parametrize("bad", ["type_id", "speed_mode", "pre_fix", "seed",
                                 "t0", "world", "vtype", "penetration"])
def test_feature_names_are_deployment_honest(bad):
    assert all(bad not in name for name in C.VEH_FEATURE_NAMES)
    assert C.VEH_F == len(C.VEH_FEATURE_NAMES)
