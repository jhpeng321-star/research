"""Integration of action-conditioned rollouts and live controller dispatch."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "output" / "wd01" / "wm_n1"))
from wms.action_timeline import ActionTimeline, CONTROLLERS  # noqa: E402
from wms.model import PhysicsConfig, WmSModel  # noqa: E402
from wms_eval.wms_plugin import WmsCheckpointError, WmsCheckpointModel  # noqa: E402


def _plugin(tmp_path, mode="residual", lane="off"):
    torch.manual_seed(7)
    model = WmSModel(physics=PhysicsConfig(), rollout_mode=mode, lane_head=lane)
    ckpt = tmp_path / "switch.pt"
    torch.save({"format": "wm_s/1", "model": model.state_dict(),
                "config": {"rollout_mode": mode, "lane_head": lane}}, ckpt)
    return WmsCheckpointModel(ckpt)


def _record():
    kin = np.zeros((4, 2, 7), np.float32)
    kin[..., 1] = 1
    kin[..., 4] = 20
    kin[..., 6] = 5
    return {"initial_state": {"time_s": 10.0, "vehicles": [
        {"id": "h1", "s": 60.0, "lane": 0, "v": 20.0, "is_cav": False},
        {"id": "c1", "s": 40.0, "lane": 0, "v": 21.0, "is_cav": True},
    ]}, "history": {"ids": ["h1", "c1"], "kin": kin.tolist(),
                    "mask": np.ones((4, 2), bool).tolist(),
                    "target_mask": np.ones((3, 2), bool).tolist()}}


def _timeline(labels):
    ids = np.asarray([[CONTROLLERS.index(label)] for label in labels])
    timeline = ActionTimeline(ids, np.full(ids.shape, -1.0, np.float32),
                              np.full(ids.shape, 30.0, np.float32)).to_dict()
    return dict(timeline, vehicle_ids=["c1"])


def _fake_damping(monkeypatch, plugin):
    deps = list(plugin._controller_dependencies())
    events = []

    class Damping:
        def __init__(self, _cfg):
            events.append("create")

        def decide(self, frame, *, now_s):
            events.append(now_s)
            return SimpleNamespace(target_speed_mps=frame.speed_mps - 1.0)

    deps[1] = Damping
    pi_type = deps[5]

    class PI(pi_type):
        def reset(self, vid, **kwargs):
            events.append(("pi_reset", vid))
            return super().reset(vid, **kwargs)

    deps[5] = PI
    monkeypatch.setattr(plugin, "_controller_dependencies", lambda: tuple(deps))
    return events


@pytest.mark.parametrize("mode,lane", [
    ("residual", "off"), ("driver_params", "off"), ("hybrid", "off"),
    ("driver_params", "history"), ("hybrid", "history"),
])
def test_constant_timeline_is_byte_identical_to_fixed_live_controller(
        tmp_path, monkeypatch, mode, lane):
    plugin = _plugin(tmp_path, mode, lane)
    events = _fake_damping(monkeypatch, plugin)
    controllers = {"arm": "damping", "cav_mode": "closed_loop"}
    fixed = plugin.predict_states(_record(), controllers, 2.0, 0)
    constant = plugin.predict_states(
        _record(), dict(controllers, action_timeline=_timeline(["damping"] * 2)), 2.0, 0)
    assert fixed == constant
    assert events == ["create", 10.0, 11.0, "create", 10.0, 11.0]
    speeds = [next(row["v"] for row in frame["vehicles"] if row["id"] == "c1")
              for frame in fixed]
    assert speeds == [21.0, 20.5, 20.0, 19.5, 19.0]


def test_live_timeline_switches_at_one_second_and_retains_lifecycle(tmp_path, monkeypatch):
    plugin = _plugin(tmp_path)
    events = _fake_damping(monkeypatch, plugin)
    labels = ["native", "pi-external-u", "native", "damping", "direct",
              "damping", "pi-external-u"]
    frames = plugin.predict_states(_record(), {
        "arm": "native", "cav_mode": "closed_loop",
        "action_timeline": _timeline(labels)}, 7.0, 0)
    assert events.count("create") == 1
    assert events.count(("pi_reset", "c1")) == 2
    assert [event for event in events if isinstance(event, float)] == [13.0, 15.0]
    speeds = [next(row["v"] for row in frame["vehicles"] if row["id"] == "c1")
              for frame in frames]
    # The direct command begins at t0+4, then holds over both physics steps.
    assert speeds[9] == pytest.approx(speeds[8] - 0.5, abs=2e-6)
    assert speeds[10] == pytest.approx(speeds[8] - 1.0, abs=2e-6)


def test_action_roster_rejects_unknown_id_and_pads_entrant_default(tmp_path):
    plugin = _plugin(tmp_path)
    raw = _timeline(["direct"])
    raw["vehicle_ids"] = ["missing"]
    with pytest.raises(WmsCheckpointError, match="t0 identities"):
        plugin._action_timeline({}, {"action_timeline": raw}, ["c1", "h1"], 2)
    packed = plugin._action_timeline({}, {
        "arm": "native", "action_timeline": _timeline(["direct"]),
        "default_entrant_controller": "pi"}, ["c1", "h1"], 2)
    assert packed["controller"].tolist() == [[[3, 0, 2, 2]]]


def test_mixed_initial_and_entrant_cavs_dispatch_once_per_physics_tick(tmp_path):
    model = _plugin(tmp_path)._load()
    state = {"s_m": torch.tensor([[0.0, 30.0]]), "v_mps": torch.full((1, 2), 10.0),
             "a_mps2": torch.zeros((1, 2)),
             "lane": torch.zeros((1, 2), dtype=torch.long),
             "present": torch.ones((1, 2), dtype=torch.bool),
             "is_cav": torch.ones((1, 2), dtype=torch.bool),
             "entrant": torch.tensor([[False, True]])}
    calls = []

    def callback(current):
        calls.append(current)
        return model.hold_teacher(current)

    model.rollout(state, n_steps=4, cav_controller=callback)
    assert len(calls) == 4


def test_delayed_entrant_controller_uses_episode_physics_clock(tmp_path, monkeypatch):
    plugin = _plugin(tmp_path)
    model = plugin._load()
    events = _fake_damping(monkeypatch, plugin)
    callback = plugin._closed_loop_callback(
        ["entrant"], np.asarray([False]),
        {"arm": "native", "default_entrant_controller": "damping", "t0_s": 10.0},
        10, initial_slots=0)
    state = {"s_m": torch.zeros((1, 1)), "v_mps": torch.full((1, 1), 10.0),
             "a_mps2": torch.zeros((1, 1)), "lane": torch.zeros((1, 1), dtype=torch.long),
             "present": torch.ones((1, 1), dtype=torch.bool),
             "is_cav": torch.ones((1, 1), dtype=torch.bool),
             "entrant": torch.ones((1, 1), dtype=torch.bool)}
    model.rollout(state, n_steps=2, k0=6, cav_controller=callback)
    assert events == ["create", 13.0]
