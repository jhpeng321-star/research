"""Fake plug-in tests for the WM-S closed-loop predictor."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_WM = Path(__file__).resolve().parents[2]
_REPO = Path(__file__).resolve().parents[5]
for p in (str(_WM), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from closedloop.a11_predictor import LiveSample  # noqa: E402
from closedloop.live_state import LiveState  # noqa: E402
from closedloop.wm_s_predictor import (  # noqa: E402
    PredictedStates,
    WMSPredictor,
    WMSPredictorError,
    window_label_delay,
)
from traffic_shepherd.planner.interface import PlanCandidate  # noqa: E402


def _state():
    kin = np.zeros((3, 7), np.float32)
    kin[:, 0] = (-100.0, -40.0, 20.0)
    kin[:, 2] = 1.0
    kin[:, 4] = 20.0
    kin[:, 6] = 5.0
    return LiveState(
        time_s=10.0,
        veh_kin=kin,
        veh_mask=np.ones(3, bool),
        is_cav=np.array([True, True, False]),
        slot_id=np.arange(3, dtype=np.int64),
        counter=np.zeros((2, 30), np.float32),
        slot_vid=["c0", "c1", "h0"],
    )


def _plugin_factory(calls):
    def predict_states(state, cav_controllers, horizon_s, seed):
        calls.append((dict(cav_controllers), float(horizon_s), int(seed)))
        n = int(round(horizon_s / 0.5))
        t = np.arange(n + 1, dtype=float)[:, None]
        speeds = np.repeat(state.veh_kin[:, 4][None, :], n + 1, axis=0)
        for slot, cav in enumerate(("c0", "c1")):
            shift = {"native": 0.0, "damping": 2.0,
                     "pi-external-u": 4.0}[cav_controllers[cav]]
            speeds[1:, slot] -= shift          # index 0 stays the live state
        positions = state.veh_kin[:, 0][None, :] + t * speeds * 0.5
        return {
            "positions": positions,
            "speeds": speeds,
            "lanes": np.ones_like(speeds, dtype=np.int64),
            "masks": np.ones_like(speeds, dtype=bool),
            "accelerations": np.zeros((n, 3)),
            "slot_ids": np.repeat(state.slot_id[None, :], n + 1, axis=0),
            "lengths": np.full(3, 5.0),
            "speed_limits_mps": np.full((n + 1, 3), 30.0),
        }

    return predict_states


def test_wms_pairs_all_vehicle_delay_from_same_rollouts():
    calls = []
    predictor = WMSPredictor(
        predict_states=_plugin_factory(calls), tau_s=300.0, max_horizon_s=4.0
    )
    sample = LiveSample(_state(), meta={"seed": 77})
    plans = [PlanCandidate({"c0": "native", "c1": "native"}),
             PlanCandidate({"c0": "damping", "c1": "native"})]
    outputs = predictor.predict_many([sample], plans, horizon_s=300.0)

    assert [h for _controllers, h, _seed in calls] == [4.0, 4.0]
    assert {seed for _controllers, _h, seed in calls} == {77}
    assert outputs[0].delay_vs_native_s[0, 0] == pytest.approx(0.0)
    assert outputs[1].delay_vs_native_s[0, 0] > 0.0
    assert outputs[1].meta["delay_metric"] == "v0.4a_window_label"
    assert outputs[1].meta["delay_scope"] == "all_vehicles"
    assert outputs[1].meta["decision_tau_s"] == 4.0
    assert set(outputs[1].trajectories_tracked) == {"c0", "c1", "h0"}
    assert np.isfinite(outputs[1].native_delay_s[0, 0])


def test_wms_registry_names_and_fleet_policy_use_rollout_delay():
    from closedloop import policies as policies_module
    from closedloop.policies import DecisionContext

    assert {"pl01_wms", "fleet_wms"} <= set(policies_module.POLICIES)
    state = _state()
    calls = []
    policy = policies_module.make_policy(
        "fleet_wms",
        predictor=WMSPredictor(predict_states=_plugin_factory(calls)),
    )
    ctx = DecisionContext(
        world="train", seed=9, t0_s=10.0, now_s=10.0, decision_index=0,
        state=state, live_cavs=("c0", "c1"), prev_assignments={},
    )
    result = policy.decide(ctx)
    assert result.assignments == {"c0": "native", "c1": "native"}
    assert result.record["delay_vs_native_s"]["native"] == pytest.approx(0.0)
    assert len(calls) == 3


def _single_state_plugin(state, cav_controllers, horizon_s, seed):
    """A plug-in that returns only the live state (T+1 = 1 sample)."""
    del cav_controllers, horizon_s, seed
    return {
        "positions": state.veh_kin[:, 0][None, :],
        "speeds": state.veh_kin[:, 4][None, :],
        "lanes": np.ones((1, state.veh_kin.shape[0]), dtype=np.int64),
        "masks": np.ones((1, state.veh_kin.shape[0]), dtype=bool),
    }


def test_closed_loop_predictor_rejects_a_single_state_prediction():
    predictor = WMSPredictor(predict_states=_single_state_plugin, max_horizon_s=4.0)
    sample = LiveSample(_state(), meta={"seed": 3})
    plans = [PlanCandidate({"c0": "native", "c1": "native"})]
    with pytest.raises(WMSPredictorError):
        predictor.predict_many([sample], plans, horizon_s=300.0)


def test_closed_loop_predictor_rejects_prediction_shorter_than_horizon():
    def short_plugin(state, cav_controllers, horizon_s, seed):
        del cav_controllers, horizon_s, seed
        n = 2
        return {
            "positions": np.repeat(state.veh_kin[:, 0][None, :], n + 1, axis=0),
            "speeds": np.repeat(state.veh_kin[:, 4][None, :], n + 1, axis=0),
            "lanes": np.ones((n + 1, state.veh_kin.shape[0]), dtype=np.int64),
            "masks": np.ones((n + 1, state.veh_kin.shape[0]), dtype=bool),
        }

    predictor = WMSPredictor(predict_states=short_plugin)
    sample = LiveSample(_state(), meta={"seed": 3})
    plans = [PlanCandidate({"c0": "native", "c1": "native"})]
    with pytest.raises(WMSPredictorError):
        predictor.predict_many([sample], plans, horizon_s=10.0)


def test_closed_loop_predictor_rejects_a_misaligned_t0():
    def shifted_plugin(state, cav_controllers, horizon_s, seed):
        del cav_controllers, seed
        n = int(round(horizon_s / 0.5))
        speeds = np.repeat(state.veh_kin[:, 4][None, :], n + 1, axis=0)
        positions = np.repeat(state.veh_kin[:, 0][None, :], n + 1, axis=0)
        positions[0, :] += 5.0
        return {
            "positions": positions, "speeds": speeds,
            "lanes": np.ones((n + 1, state.veh_kin.shape[0]), dtype=np.int64),
            "masks": np.ones((n + 1, state.veh_kin.shape[0]), dtype=bool),
        }

    predictor = WMSPredictor(predict_states=shifted_plugin, max_horizon_s=2.0)
    sample = LiveSample(_state(), meta={"seed": 3})
    plans = [PlanCandidate({"c0": "native", "c1": "native"})]
    with pytest.raises(WMSPredictorError):
        predictor.predict_many([sample], plans, horizon_s=300.0)


def test_closed_loop_predictor_rejects_missing_live_slot_at_t0():
    """Dropping a live on-lane slot from the plug-in output must fail closed."""

    def missing_plugin(state, cav_controllers, horizon_s, seed):
        del cav_controllers, seed
        n = int(round(horizon_s / 0.5))
        t = np.arange(n + 1, dtype=float)[:, None]
        masks = np.ones((n + 1, state.veh_kin.shape[0]), dtype=bool)
        masks[:, 1] = False  # live slot c1 omitted from every sample
        return {
            "positions": state.veh_kin[:, 0][None, :] + t * state.veh_kin[:, 4][None, :] * 0.5,
            "speeds": np.repeat(state.veh_kin[:, 4][None, :], n + 1, axis=0),
            "lanes": np.ones((n + 1, state.veh_kin.shape[0]), dtype=np.int64),
            "masks": masks,
        }

    predictor = WMSPredictor(predict_states=missing_plugin, max_horizon_s=2.0)
    sample = LiveSample(_state(), meta={"seed": 3})
    plans = [PlanCandidate({"c0": "native", "c1": "native"})]
    with pytest.raises(WMSPredictorError):
        predictor.predict_many([sample], plans, horizon_s=300.0)


def test_closed_loop_predictor_rejects_nonuniform_time_grid():
    """A strictly increasing but nonuniform time axis must fail closed."""

    def nonuniform_plugin(state, cav_controllers, horizon_s, seed):
        del cav_controllers, seed
        n = int(round(horizon_s / 0.5))
        times = np.arange(n + 1, dtype=float) * 0.5
        times[1:] += 0.25  # e.g. 0, 0.75, 1.25, ... : first gap is 0.75 s
        return {
            "positions": state.veh_kin[:, 0][None, :] + times[:, None] * state.veh_kin[:, 4][None, :],
            "speeds": np.repeat(state.veh_kin[:, 4][None, :], n + 1, axis=0),
            "lanes": np.ones((n + 1, state.veh_kin.shape[0]), dtype=np.int64),
            "masks": np.ones((n + 1, state.veh_kin.shape[0]), dtype=bool),
            "times": times + state.time_s,
        }

    predictor = WMSPredictor(predict_states=nonuniform_plugin, max_horizon_s=2.0)
    sample = LiveSample(_state(), meta={"seed": 3})
    plans = [PlanCandidate({"c0": "native", "c1": "native"})]
    with pytest.raises(WMSPredictorError):
        predictor.predict_many([sample], plans, horizon_s=300.0)


def test_closed_loop_predictor_accepts_explicit_uniform_dt_grid():
    """An explicit uniform DT_S time axis is accepted and priced consistently."""

    def explicit_plugin(state, cav_controllers, horizon_s, seed):
        del cav_controllers
        n = int(round(horizon_s / 0.5))
        times = float(state.time_s) + 0.5 * np.arange(n + 1)
        speeds = np.repeat(state.veh_kin[:, 4][None, :], n + 1, axis=0)
        positions = (
            state.veh_kin[:, 0][None, :]
            + (times[:, None] - state.time_s) * speeds
        )
        return {
            "positions": positions,
            "speeds": speeds,
            "lanes": np.ones((n + 1, state.veh_kin.shape[0]), dtype=np.int64),
            "masks": np.ones((n + 1, state.veh_kin.shape[0]), dtype=bool),
            "times": times,
        }

    predictor = WMSPredictor(predict_states=explicit_plugin, max_horizon_s=4.0)
    sample = LiveSample(_state(), meta={"seed": 3})
    plans = [PlanCandidate({"c0": "native", "c1": "native"})]
    outputs = predictor.predict_many([sample], plans, horizon_s=300.0)
    assert np.isfinite(outputs[0].native_delay_s[0, 0])


def _predicted(speeds, limits, times):
    speeds = np.asarray(speeds, dtype=float)[:, None]
    shape = speeds.shape
    return PredictedStates(
        positions=np.zeros(shape),
        speeds=speeds,
        lanes=np.zeros(shape, dtype=np.int64),
        masks=np.ones(shape, dtype=bool),
        speed_limits_mps=np.asarray(limits, dtype=float)[:, None],
        times=np.asarray(times, dtype=float),
    )


def test_window_label_delay_integrates_actual_deltas_and_refuses_nonuniform():
    states = _predicted([10.0, 20.0, 20.0], [30.0, 30.0, 30.0], [0.0, 0.5, 1.0])
    assert window_label_delay(states, 1.0) == pytest.approx(1.0 / 3.0)

    bad = _predicted([10.0, 20.0, 20.0], [30.0, 30.0, 30.0], [0.0, 0.4, 1.0])
    with pytest.raises(WMSPredictorError):
        window_label_delay(bad, 1.0)
