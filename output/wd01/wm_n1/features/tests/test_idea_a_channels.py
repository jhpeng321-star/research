"""P9 shard channels: veh_kin / hdv_base / cav_plan_ch / events + leak rule.

Acceptance checks on a synthetic in-memory shard (K = 128 slots, no data
reads): the channels carry no CAV/action/strategy information, the HDV
baseline rows come from the frozen nominal replay, CAV plan channels are
teacher-forced from recorded CAV motion, event channels are all-zero masks
for the current data, and every channel at tick ``t`` is unchanged when the
recorded data AFTER ``t`` is perturbed (inputs use data <= t only).
"""
from __future__ import annotations

import numpy as np
import pytest

from features import constants as C
from features.idea_a import (CAV_PLAN_CH_F, CAV_ACCEL_SCALE,
                             CAV_TGT_SPEED_SCALE, EVENT_MAX, EVENT_TOK_F,
                             HDV_BASE_F, KIN_F, idea_a_channels_from_shard,
                             kinematics_view)


def _wall_row(phase_idx, speed_mps=24.0, hold_s=240.0):
    row = np.zeros(C.WALL_F, np.float16)
    row[phase_idx] = 1.0
    row[5] = np.float16(speed_mps / 40.0)
    row[7] = np.float16(hold_s / 600.0)
    return row


def _shard(n_ticks: int = 6, *, aborted_tick: int = 4) -> dict:
    veh = np.zeros((n_ticks, C.K_MAX, C.VEH_F), np.float16)
    mask = np.zeros((n_ticks, C.K_MAX), np.uint8)
    for t in range(n_ticks):
        # slot 0: CAV on main_right (member of the wall while active)
        veh[t, 0, C.S_DOWN] = 100.0
        veh[t, 0, C.LANE_RIGHT] = 1.0
        veh[t, 0, C.V_IDX] = 20.0
        veh[t, 0, C.A_IDX] = 0.6
        veh[t, 0, C.LENGTH_IDX] = C.NOMINAL_LENGTH_M
        veh[t, 0, C.IS_CAV_IDX] = 1.0
        veh[t, 0, C.ACT_WALL] = 1.0
        veh[t, 0, C.WALL_MEMBER_IDX] = 1.0 if t < aborted_tick else 0.0
        veh[t, 0, C.PRESENT_IDX] = 1.0
        mask[t, 0] = 1
        # slot 1: HDV ahead on main_right
        veh[t, 1, C.S_DOWN] = 160.0
        veh[t, 1, C.LANE_RIGHT] = 1.0
        veh[t, 1, C.V_IDX] = 18.0
        veh[t, 1, C.LENGTH_IDX] = C.NOMINAL_LENGTH_M
        veh[t, 1, C.PRESENT_IDX] = 1.0
        mask[t, 1] = 1
        # slot 2: HDV on the ramp inside the merge zone
        veh[t, 2, C.S_DOWN] = 30.0
        veh[t, 2, C.LANE_ACCEL] = 1.0
        veh[t, 2, C.V_IDX] = 21.0
        veh[t, 2, C.LENGTH_IDX] = C.NOMINAL_LENGTH_M
        veh[t, 2, C.PRESENT_IDX] = 1.0
        mask[t, 2] = 1

    wall = np.zeros((n_ticks, C.MAX_WALLS, C.WALL_F), np.float16)
    wall_mask = np.zeros((n_ticks, C.MAX_WALLS), np.uint8)
    for t in range(n_ticks):
        if t < aborted_tick:
            wall[t, 0] = _wall_row(2)          # holding
            wall_mask[t, 0] = 1
        elif t == aborted_tick:
            wall[t, 0] = _wall_row(4)          # aborted (terminal tick)
            wall_mask[t, 0] = 1

    meta = dict(chunk="chunk4", world="wsmoke", arm="w-lo-early",
                split="TRAIN", t0_s=420.0, n_ticks=n_ticks)
    return dict(veh_step=veh, veh_mask=mask, wall=wall, wall_mask=wall_mask,
                meta=meta)


def test_channels_shapes_and_kin_view():
    built = idea_a_channels_from_shard(_shard(4))
    for name, shape in (("veh_kin", (4, C.K_MAX, KIN_F)),
                        ("hdv_base", (4, C.K_MAX, HDV_BASE_F)),
                        ("cav_plan_ch", (4, C.K_MAX, CAV_PLAN_CH_F)),
                        ("event_tok", (4, EVENT_MAX, EVENT_TOK_F)),
                        ("event_mask", (4, EVENT_MAX))):
        assert built[name].shape == shape
    assert built["veh_kin"].dtype == np.float16
    assert built["event_mask"].dtype == np.uint8
    assert built["event_tok"].sum() == 0 and built["event_mask"].sum() == 0
    sh = _shard(4)
    assert np.array_equal(built["veh_kin"], kinematics_view(
        np.asarray(sh["veh_step"])))


def test_hdv_base_rows_and_cav_zeroing():
    built = idea_a_channels_from_shard(_shard(3))
    base = built["hdv_base"]
    assert np.all(base[:, 0, :] == 0.0)         # CAV slot zeroed
    assert np.abs(base[:, 1, :]).max() > 0.0    # HDV rows populated
    assert np.abs(base[:, 2, 3]).max() > 0.0    # ramp merge urgency
    assert np.all(base[:, 3:, :] == 0.0)        # absent slots zero


def test_cav_plan_channels_teacher_forced():
    built = idea_a_channels_from_shard(_shard(3))
    plan = built["cav_plan_ch"]
    assert np.all(plan[:, 1:, :] == 0.0)        # non-CAV slots zero
    # recorded CAV motion stands in for the command pair (data <= t)
    assert np.allclose(plan[:, 0, 0], 20.0 / CAV_TGT_SPEED_SCALE, atol=1e-3)
    assert np.allclose(plan[:, 0, 1], 0.6 / CAV_ACCEL_SCALE, atol=1e-3)
    # wall speed/hold come from the active wall token while a member
    assert np.allclose(plan[:3, 0, 2], 24.0 / 40.0, atol=1e-3)
    assert np.allclose(plan[:3, 0, 4], 240.0 / 600.0, atol=1e-3)
    assert np.allclose(plan[:3, 0, 3], 0.0, atol=1e-6)     # episode 0 -> 0/4
    # after the terminal tick the member is handed back: no wall (-1/4)
    assert np.allclose(plan[4:, 0, 3], -0.25, atol=1e-6)
    assert np.all(plan[:4, 0, 2] >= 0)
    # teacher-forced lane target = current lane one-hot (main_right)
    assert np.allclose(plan[:, 0, 5:8], [0.0, 1.0, 0.0], atol=1e-6)


def test_relabel_counts_aborted_tokens():
    built = idea_a_channels_from_shard(_shard(6, aborted_tick=4))
    assert built["relabel"]["n_aborted_tokens"] == 1
    assert idea_a_channels_from_shard(_shard(6, aborted_tick=99))[
        "relabel"]["n_aborted_tokens"] == 0


def test_channels_use_only_data_up_to_t():
    """Perturbing every row AFTER t leaves all channels at <= t unchanged."""
    sh_a = _shard(6, aborted_tick=4)
    sh_b = _shard(6, aborted_tick=4)
    veh_b = np.asarray(sh_b["veh_step"])
    veh_b[3:, :, C.V_IDX] = 99.0
    veh_b[3:, :, C.A_IDX] = -7.0
    veh_b[3:, :, C.S_DOWN] = -900.0
    sh_b["veh_step"] = veh_b
    wall_b = np.asarray(sh_b["wall"]).copy()
    wall_b[3:, :, :] = 0.0
    wall_b[3:, 1, 2] = 1.0                      # a different wall episode
    sh_b["wall"] = wall_b
    built_a = idea_a_channels_from_shard(sh_a)
    built_b = idea_a_channels_from_shard(sh_b)
    for name in ("veh_kin", "hdv_base", "cav_plan_ch"):
        assert np.array_equal(built_a[name][:3], built_b[name][:3]), name
    assert np.array_equal(built_a["event_tok"], built_b["event_tok"])
    assert np.array_equal(built_a["event_mask"], built_b["event_mask"])
    # sanity: the perturbation DID change tick 3
    assert not np.array_equal(built_a["hdv_base"][3], built_b["hdv_base"][3])


def test_two_wall_episodes_get_distinct_ordinals():
    sh = _shard(6, aborted_tick=99)             # no aborted terminal
    wall = np.zeros((6, C.MAX_WALLS, C.WALL_F), np.float16)
    wall_mask = np.zeros((6, C.MAX_WALLS), np.uint8)
    for t in range(0, 3):
        wall[t, 0] = _wall_row(2)
        wall_mask[t, 0] = 1
    for t in range(4, 6):
        wall[t, 0] = _wall_row(1)               # second episode, same row
        wall_mask[t, 0] = 1
    sh["wall"], sh["wall_mask"] = wall, wall_mask
    built = idea_a_channels_from_shard(sh)
    plan = built["cav_plan_ch"]
    assert np.allclose(plan[0:3, 0, 3], 0.0, atol=1e-6)     # episode 0
    assert np.allclose(plan[4:6, 0, 3], 0.25, atol=1e-6)    # episode 1 -> 1/4


def test_rejects_non_128_slot_shard():
    sh = _shard(2)
    sh["veh_mask"] = sh["veh_mask"][:, :16]
    with pytest.raises(ValueError):
        idea_a_channels_from_shard(sh)
