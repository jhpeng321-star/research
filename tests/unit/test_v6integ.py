"""V6 P4 execution invariants; synthetic CPU scenes, no simulator or data packs."""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from wms.cli.assemble import build_physics
from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.model.checkpoint import load_checkpoint, save_checkpoint
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.accel import baseline_accel
from wms.physics.config import PhysicsConfig
from wms.physics.idm import IDMParams, idm_accel
from wms.physics.lane_conflict import resolve_lane_conflicts
from wms.physics.neighbours import lead_neighbour
from wms.physics.projection import FeasibilityProjection, ProjectionConfig
from wms.physics.state import state_kin
from wms.train.settings import TrainSettings


def _state(s, v, lane, length=None):
    s = torch.tensor([s], dtype=torch.float32)
    return dict(s_m=s, v_mps=torch.tensor([v], dtype=torch.float32),
                a_mps2=torch.zeros_like(s), lane=torch.tensor([lane]),
                length_m=torch.full_like(s, 5.0) if length is None else torch.tensor([length]),
                present=torch.ones_like(s, dtype=torch.bool))


def _fixed_model(physics, accel, lanes=None, resolve=False):
    def backend(state):
        choice = state["lane"] if lanes is None else torch.tensor([lanes])
        return {"accel_mps2": torch.zeros_like(state["s_m"]) + accel,
                "lane_logits": F.one_hot(choice, 3).float() * 8.0}
    return WmSModel(ModelSpec(lane_conflict_resolve=resolve), physics=physics,
                    physics_step=backend).eval()


@pytest.mark.parametrize("enabled,expected", [(False, 0.5), (True, 0.25)])
def test_stop_within_step(enabled, expected):
    state = _state([0.0], [2.0], [0])
    accel = torch.tensor([[-8.0]], requires_grad=True)
    model = _fixed_model(PhysicsConfig(stop_time_integration=enabled), accel)
    out = model.step(state)
    assert out["next_s_m"].item() == expected
    assert out["next_v_mps"].item() == 0.0
    if enabled:
        out["next_s_m"].sum().backward()
        assert accel.grad.item() == pytest.approx(0.03125)


@pytest.mark.parametrize("v,a", [(0.0, 0.0), (2.0, -4.0), (3.0, -2.0), (3.0, 2.0),
                                  (1e-9, -8e-9)])
def test_ballistic_motion_boundary_and_gradients(v, a):
    state = _state([0.0], [v], [0])
    accel = torch.tensor([[a]], requires_grad=True)
    model = _fixed_model(PhysicsConfig(stop_time_integration=True), accel)
    out = model.step(state)
    travel_time = min(0.5, -v / a) if a < 0 else 0.5
    expected = v * travel_time + 0.5 * a * travel_time**2
    assert out["next_s_m"].item() == pytest.approx(expected, abs=1e-16)
    out["next_s_m"].sum().backward()
    assert torch.isfinite(accel.grad).all()


def _params(state):
    return {key: torch.full_like(state["s_m"], value) for key, value in
            dict(v0=30.0, time_headway=1.1, min_gap=2.5, accel=2.6, decel=2.0).items()}


@pytest.mark.parametrize("inferred", [False, True])
def test_emergency_stopped_leader_no_overlap(inferred):
    # 8 m/s with 4 m of usable stopping distance needs 8 m/s^2, above comfortable b.
    for limit in (None, 8.0):
        state = _state([11.5, 0.0], [0.0, 8.0], [0, 0])
        state["is_cav"] = torch.tensor([[True, False]])
        params = _params(state) if inferred else None
        physics = PhysicsConfig(stop_time_integration=True, emergency_decel_mps2=limit)

        def backend(current):
            return dict(accel_mps2=baseline_accel(state_kin(current), current["present"],
                                                physics, params),
                        lane_logits=F.one_hot(current["lane"], 3).float() * 8.0)

        model = WmSModel(physics=physics, physics_step=backend).eval()
        for _ in range(2):
            out = model.step(state, cav_teacher=model.hold_teacher(state))
            state = {**state, "s_m": out["next_s_m"], "v_mps": out["next_v_mps"],
                     "a_mps2": out["accel_mps2"], "lane": out["next_lane"]}
            if limit is not None:
                assert -limit <= out["physics_accel_mps2"][0, 1] < -physics.idm()[1]
                assert state["s_m"][0, 0] - 5.0 - state["s_m"][0, 1] >= 2.5
        if limit is not None:
            assert state["v_mps"][0, 1] == 0.0
        elif inferred:
            assert state["s_m"][0, 0] - 5.0 - state["s_m"][0, 1] < 0.0


def test_emergency_only_unlocks_when_comfortable_stopping_is_insufficient():
    state = _state([30.0, 0.0], [0.0, 8.0], [0, 0])
    params = _params(state)
    kin = state_kin(state)
    old = baseline_accel(kin, state["present"], PhysicsConfig(), params)
    new = baseline_accel(kin, state["present"], PhysicsConfig(emergency_decel_mps2=8.0), params)
    assert torch.equal(new, old)
    assert new[0, 1] >= -params["decel"][0, 1]
    # An unavoidably close leader cannot request more than the configured plant limit.
    state["s_m"][0, 0] = 6.0
    state["present"][0, 0] = True
    accel = baseline_accel(state_kin(state), state["present"],
                           PhysicsConfig(emergency_decel_mps2=8.0), params)
    assert accel[0, 1] == -8.0


def test_baseline_off_matches_legacy_formula_exactly():
    from tests.unit.test_gaprule import fuzz_scene
    kin, present, params = fuzz_scene()
    physics = PhysicsConfig()
    gap, leader_v = lead_neighbour(kin, present, physics)
    for values in (None, params):
        p = physics.idm_params() if values is None else IDMParams.from_driver_params(values)
        expected = idm_accel(kin[..., 4], leader_v, gap, p, accel_min=-9.0, accel_max=3.0)
        if values is not None:
            expected = torch.minimum(torch.maximum(expected, -values["decel"]), values["accel"])
        expected = torch.where(present, expected, torch.zeros_like(expected))
        assert torch.equal(baseline_accel(kin, present, physics, values), expected)


def test_emergency_limit_also_bounds_merge_yield():
    state = _state([50.0, 56.0], [10.0, 4.0], [1, 2])
    physics = PhysicsConfig(merge_cal=True, emergency_decel_mps2=1.0)
    accel = baseline_accel(state_kin(state), state["present"], physics, _params(state))
    assert (accel >= -1.0).all()


def test_unequal_length_projection_and_gradient():
    s = torch.tensor([[20.0, 14.0]], requires_grad=True)
    length = torch.tensor([[10.0, 5.0]], requires_grad=True)
    projection = FeasibilityProjection(ProjectionConfig(min_gap_m=2.0))
    out = projection(s, torch.zeros_like(s), torch.zeros_like(s).long(),
                     torch.ones_like(s).bool(), length)
    assert torch.equal(out["s_m"], torch.tensor([[20.0, 8.0]]))
    assert out["active"]["gap_before"] == out["active"]["s_clamped"] == 1
    out["s_m"][0, 1].backward()
    assert torch.equal(s.grad, torch.tensor([[1.0, 0.0]]))
    assert torch.equal(length.grad, torch.tensor([[-1.0, 0.0]]))


def test_unequal_projection_chain_and_anchor():
    s = torch.tensor([[40.0, 35.0, 30.0]])
    length = torch.tensor([[10.0, 5.0, 7.0]])
    projection = FeasibilityProjection()
    for cav, expected in [(None, [[40.0, 28.0, 21.0]]),
                          (torch.tensor([[False, True, False]]), [[40.0, 35.0, 28.0]])]:
        out = projection(s, torch.zeros_like(s), torch.zeros_like(s).long(),
                         torch.ones_like(s).bool(), length, is_cav=cav)
        assert torch.equal(out["s_m"], torch.tensor(expected))


@pytest.mark.parametrize("resolve", [False, True])
def test_two_lane_changes_into_same_gap(resolve):
    state = _state([50.0, 49.0, 90.0, 10.0], [0.0] * 4, [0, 2, 1, 1])
    model = _fixed_model(PhysicsConfig(), 0.0, lanes=[1, 1, 1, 1], resolve=resolve)
    out = model.step(state)
    assert out["next_lane"].tolist() == [[1, 2 if resolve else 1, 1, 1]]
    assert torch.equal(out["next_s_m"], state["s_m"])
    if resolve:
        projection = FeasibilityProjection()
        projection(out["next_s_m"], out["next_v_mps"], out["next_lane"],
                   out["next_present"], state["length_m"])
        assert projection.stats["gap_before"] == 0


def test_conflicts_use_post_motion_and_teacher_destinations():
    # Initially disjoint footprints converge during the step; a teacher has priority.
    state = _state([50.0, 42.0], [0.0, 16.0], [0, 2])
    state["is_cav"] = torch.tensor([[False, True]])
    model = _fixed_model(PhysicsConfig(), 0.0, lanes=[1, 1], resolve=True)
    teacher = torch.tensor([[[50.0, 0.0, 0.0, 0.0], [50.0, 1.0, 16.0, 0.0]]])
    out = model.step(state, cav_teacher=teacher)
    assert out["next_lane"].tolist() == [[0, 1]]
    assert torch.equal(out["next_s_m"], teacher[..., 0])


def test_conflicts_ties_absent_slots_and_reserved_source_space():
    # B would collide in lane 1 and stays in lane 2; C must not enter B's source space.
    state = _state([50.0, 50.0, 49.0, 50.0], [0.0] * 4, [0, 2, 1, 1])
    present = torch.tensor([[True, True, True, False]])
    result = resolve_lane_conflicts(state["s_m"], state["length_m"], state["lane"],
                                    torch.tensor([[1, 1, 2, 1]]), present, present, 0.5)
    # C initially reserves lane 1, so neither conflicting inbound proposal is admitted.
    assert result.tolist() == [[0, 2, 1, 1]]
    state = _state([50.0, 50.0, 10.0], [0.0] * 3, [0, 2, 0])
    model = _fixed_model(PhysicsConfig(), 0.0, lanes=[1, 1, 1], resolve=True)
    assert model.step(state)["next_lane"].tolist() == [[1, 2, 1]]


def test_config_roundtrip_and_legacy_defaults():
    keys = {"stop_time_integration", "emergency_decel_mps2", "lane_conflict_resolve"}
    old = load_train_config("configs/c4_wm_v5.toml")
    assert not keys & old.to_legacy_flat().keys()
    assert not build_physics(old).stop_time_integration
    cfg = load_train_config("configs/c4_wm_v6_integ.toml")
    physics = build_physics(cfg)
    assert physics.stop_time_integration and physics.emergency_decel_mps2 == 9.0
    flat = cfg.to_legacy_flat()
    assert TrainConfig.from_legacy_flat(flat).to_legacy_flat() == flat
    assert TrainSettings.from_flat(flat).lane_conflict_resolve
    assert ModelSpec.from_mapping(flat).lane_conflict_resolve
    assert cfg.model.projection.enabled and cfg.runtime.threads == 2
    assert build_physics(TrainConfig.from_dict({"model": {"merge_cal": True},
                                              "physics": {"stop_time_integration": True}})
                         ).stop_time_integration


def test_v6_config_live_step_backward():
    cfg = load_train_config("configs/c4_wm_v6_integ.toml")
    torch.manual_seed(0)
    model = WmSModel(ModelSpec.from_mapping(cfg.to_legacy_flat()), physics=build_physics(cfg))
    state = _state([50.0, 49.0, 90.0, 10.0], [2.0, 2.0, 0.0, 4.0], [0, 2, 1, 1])
    history = torch.zeros(1, 4, model.driver_history_len, 9)
    history[:, :, -1, 0] = state["v_mps"] / 30.0
    history[:, :, -1, 2:5] = F.one_hot(state["lane"], 3).float()
    mask = torch.zeros(history.shape[:-1], dtype=torch.bool)
    mask[:, :, -1] = True
    out = model.step(state, k_steps=5, lane_uniform=torch.full((1, 4), 0.5),
                     driver_history=history, driver_history_mask=mask,
                     projection=FeasibilityProjection())
    out["next_s_m"].sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert (out["next_v_mps"] >= 0).all()


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), 9.1, True])
def test_emergency_limit_validation(value):
    with pytest.raises(ValueError):
        PhysicsConfig(emergency_decel_mps2=value)
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"physics": {"emergency_decel_mps2": value}})


@pytest.mark.parametrize("enabled", [False, True])
def test_physics_json_checkpoint_compatibility(tmp_path, enabled):
    cfg = TrainConfig.from_dict({"model": {"lane_conflict_resolve": enabled},
                                "physics": {"stop_time_integration": enabled,
                                            "emergency_decel_mps2": 8.0 if enabled else None}})
    physics = build_physics(cfg)
    model = WmSModel(ModelSpec.from_mapping(cfg.to_legacy_flat()), physics=physics)
    path = tmp_path / "wm_s.pt"
    save_checkpoint(path, model, cfg)
    raw = asdict(physics)
    if not enabled:
        raw.pop("stop_time_integration")
        raw.pop("emergency_decel_mps2")
    (tmp_path / "physics.json").write_text(json.dumps(raw), encoding="utf-8")
    loaded = load_checkpoint(path).model
    assert loaded.physics == physics
    assert loaded.lane_conflict_resolve == enabled
    assert all(torch.equal(value, loaded.state_dict()[key])
               for key, value in model.state_dict().items())
