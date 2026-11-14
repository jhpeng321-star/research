"""Junction-transit presence: carry through, drop only on exit/identity change."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[5]
PKG = ROOT / "output" / "wd01" / "wm_n1"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from wms.model import PhysicsConfig, WmSModel  # noqa: E402
from wms.presence import junction_carry_presence  # noqa: E402
from wms.train import TrajectoryDataset  # noqa: E402
from hdv.config import HdvConfig  # noqa: E402


def small_model(**kwargs):
    return WmSModel(
        physics=PhysicsConfig(max_gap_steps=8),
        hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# helper
# --------------------------------------------------------------------------- #
def test_junction_step_keeps_the_vehicle_and_is_flagged():
    present = np.array([[1], [1], [0], [1], [1]], dtype=bool)
    junction = np.array([[0], [0], [1], [0], [0]], dtype=bool)
    occupancy, junction_step = junction_carry_presence(present, junction)
    assert occupancy[:, 0].tolist() == [True, True, True, True, True]
    assert junction_step[:, 0].tolist() == [False, False, True, False, False]


def test_true_exit_and_identity_change_still_drop_and_never_revive():
    # registered occupancy ends at t1 (real exit): no junction step, and the
    # later rows (a slot reused by a new vehicle) must not revive it.
    present = np.array([[1], [1], [0], [1]], dtype=bool)
    junction = np.zeros_like(present)
    occupancy, junction_step = junction_carry_presence(present, junction)
    assert occupancy[:, 0].tolist() == [True, True, False, False]
    assert not junction_step.any()


def test_no_future_leakage_presence_is_a_prefix_function():
    present = np.array([[1], [0], [1], [1]], dtype=bool)
    junction = np.array([[0], [0], [1], [0]], dtype=bool)
    full, _ = junction_carry_presence(present, junction)
    for cut in range(1, present.shape[0] + 1):
        prefix, _ = junction_carry_presence(present[:cut], junction[:cut])
        assert np.array_equal(prefix, full[:cut])


def test_helper_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        junction_carry_presence(np.zeros((2, 2), bool), np.zeros((2, 3), bool))


# --------------------------------------------------------------------------- #
# training dataset builder
# --------------------------------------------------------------------------- #
def _junction_states(n=4, lane=0):
    states = np.zeros((n, 1, 7), dtype=np.float32)
    states[:, :, 1 + lane] = 1.0
    states[:, :, 4] = 10.0
    return states


def test_from_arrays_junction_keeps_vehicle_and_excludes_only_that_step():
    states = _junction_states(4)
    states[2, :, 1:4] = 0.0                       # t2 is a junction internal lane
    dataset = TrajectoryDataset.from_arrays({
        "states": states,
        "present": np.array([[[1], [1], [0], [1]]], dtype=bool),
        "junction": np.array([[[0], [0], [1], [0]]], dtype=bool),
        "identities": np.array([[[5], [5], [5], [5]]], dtype=np.int64),
    })
    record = dataset[0]
    assert record["present"][:, 0].tolist() == [True, True, True, True]
    assert record["loss_valid"][:, 0].tolist() == [True, True, False, True]


def test_from_arrays_identity_change_on_a_normal_lane_still_drops():
    states = _junction_states(4)
    dataset = TrajectoryDataset.from_arrays({
        "states": states,
        "present": np.array([[[1], [1], [0], [1]]], dtype=bool),
        "identities": np.array([[[5], [5], [-1], [9]]], dtype=np.int64),
    })
    record = dataset[0]
    assert record["present"][:, 0].tolist() == [True, True, False, False]


def test_from_arrays_default_presence_is_unchanged_without_junction():
    states = _junction_states(3)
    dataset = TrajectoryDataset.from_arrays(
        {"states": states, "present": np.ones((1, 3, 1), bool)})
    record = dataset[0]
    assert record["present"][:, 0].tolist() == [True, True, True]
    assert record["loss_valid"][:, 0].tolist() == [True, True, True]


# --------------------------------------------------------------------------- #
# WM-S rollout presence
# --------------------------------------------------------------------------- #
def _one_slot_state():
    kin = torch.zeros(1, 1, 7)
    kin[0, 0, 0] = 0.0
    kin[0, 0, 1] = 1.0
    kin[0, 0, 4] = 10.0
    return {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
            "a_mps2": kin[..., 5], "length_m": kin[..., 6],
            "lane": torch.zeros(1, 1, dtype=torch.long),
            "present": torch.ones(1, 1, dtype=torch.bool),
            "is_cav": torch.zeros(1, 1, dtype=torch.bool)}


def test_rollout_junction_seq_carries_the_slot_through_a_dropped_step():
    model = small_model()
    n = 3
    present_seq = torch.ones(1, n + 1, 1, dtype=torch.bool)
    present_seq[:, 2, 0] = False                  # recorded presence drops at t2
    junction_seq = torch.zeros(1, n + 1, 1, dtype=torch.bool)
    junction_seq[:, 2, 0] = True                  # ...because it is a junction
    roll = model.rollout(_one_slot_state(), n_steps=n,
                         present_seq=present_seq, junction_seq=junction_seq)
    assert roll["masks"][0, :, 0].tolist() == [True, True, True]


def test_rollout_without_junction_still_drops():
    model = small_model()
    n = 3
    present_seq = torch.ones(1, n + 1, 1, dtype=torch.bool)
    present_seq[:, 2, 0] = False
    roll = model.rollout(_one_slot_state(), n_steps=n, present_seq=present_seq)
    assert roll["masks"][0, :, 0].tolist() == [True, False, False]


# --------------------------------------------------------------------------- #
# wms_eval teacher / matcher
# --------------------------------------------------------------------------- #
def _eval_state():
    def frame(t, rows):
        return {"time_s": t, "vehicles": rows}
    lane_rows = [
        frame(0.0, [{"id": "a", "s": 10.0, "lane": 0, "v": 10.0, "present": True, "is_cav": False},
                    {"id": "b", "s": 20.0, "lane": 1, "v": 10.0, "present": True, "is_cav": False},
                    {"id": "c", "s": 30.0, "lane": 2, "v": 10.0, "present": True, "is_cav": False}]),
        frame(0.5, [{"id": "a", "s": 15.0, "lane": 0, "v": 10.0, "present": True, "is_cav": False},
                    {"id": "b", "s": 25.0, "lane": -1, "v": 10.0, "present": True, "is_cav": False},
                    {"id": "c", "s": 35.0, "lane": 2, "v": 10.0, "present": True, "is_cav": False}]),
        frame(1.0, [{"id": "a", "s": 20.0, "lane": 0, "v": 10.0, "present": True, "is_cav": False},
                    {"id": "b", "s": 30.0, "lane": -1, "v": 10.0, "present": True, "is_cav": False}]),
        frame(1.5, [{"id": "a", "s": 25.0, "lane": 0, "v": 10.0, "present": True, "is_cav": False},
                    {"id": "b", "s": 35.0, "lane": 1, "v": 10.0, "present": True, "is_cav": False}]),
    ]
    return {"world": "w", "t0_s": 0.0, "recorded": {"native": lane_rows}}


def test_eval_teacher_carries_junction_but_drops_a_true_exit():
    from wms_eval.wms_plugin import WmsCheckpointModel

    model = WmsCheckpointModel.__new__(WmsCheckpointModel)
    model._torch = torch
    ids = ["a", "b", "c"]
    kin = np.zeros((3, 7), dtype=np.float32)
    kin[0, 0], kin[1, 0], kin[2, 0] = 10.0, 20.0, 30.0
    kin[0, 1], kin[1, 2], kin[2, 3] = 1.0, 1.0, 1.0
    kin[:, 4] = 10.0
    is_cav = np.zeros(3, dtype=bool)
    teacher, present, junction = model._teacher(
        _eval_state(), 0.0, ids, kin, is_cav, 3, "native")

    # b is a junction transit at steps 1 and 2: kept present, lane carried at 1.
    assert present[0, :, 1].tolist() == [True, True, True, True]
    assert junction[0, :, 1].tolist() == [False, True, True, False]
    assert int(teacher[0, 1, 1, 1]) == 1
    # a stays on lane 0 throughout; c truly exits after step 1.
    assert present[0, :, 0].tolist() == [True, True, True, True]
    assert present[0, :, 2].tolist() == [True, True, False, False]


def test_closed_loop_entrant_uses_plan_default_and_gets_spawn_local_controller(monkeypatch):
    from types import SimpleNamespace
    from wms_eval.wms_plugin import WmsCheckpointModel

    created = []

    class Circles:
        def __init__(self, _cfg):
            created.append(self)

        def decide(self, _frame, *, now_s):
            return SimpleNamespace(target_speed_mps=None)

    class PI:
        def __init__(self, **_kwargs):
            pass

    class UReport:
        def __init__(self, *args):
            pass

    class PIParams:
        pass

    cfg = SimpleNamespace(sensor_range_m=100.0, control_dt_s=1.0,
                          acceleration_max_mps2=2.0,
                          acceleration_min_mps2=-4.0)
    model = WmsCheckpointModel.__new__(WmsCheckpointModel)
    model.model = SimpleNamespace(
        physics=SimpleNamespace(speed_limit_mps=30.0, dt_s=0.5),
        physics_accel=lambda state: torch.zeros_like(state["v_mps"]))
    model._controller_dependencies = lambda: (
        cfg, Circles, lambda *args: (None, None), lambda *args: SimpleNamespace(
            gap_m=args[0], relative_speed_mps=args[1]),
        lambda **kwargs: SimpleNamespace(**kwargs), PI, UReport, PIParams)
    callback = model._closed_loop_callback(
        ["t0", "__entry_0"], np.asarray([False, False]),
        {"arm": "native", "default_entrant_controller": "damping", "t0_s": 0.0},
        2, initial_slots=1)
    assert created == []
    state = {
        "s_m": torch.tensor([[10.0, 20.0]]),
        "v_mps": torch.tensor([[10.0, 10.0]]),
        "lane": torch.tensor([[0, 0]]),
        "present": torch.tensor([[True, True]]),
        "is_cav": torch.tensor([[False, True]]),
    }
    callback(state)
    assert len(created) == 1  # created at spawn, not before t0
    callback(state)
    assert len(created) == 1  # same entrant retains its controller instance
