"""Checkpoint-agnostic R13E runtime contracts: synthetic CPU inputs only."""
from dataclasses import asdict, replace
import json
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from wms.cli.assemble import build_model, build_physics
from wms.config.train import PhysicsOptions, TrainConfig
from wms.eval.backend import load_backend
from wms.model import ModelSpec, WmSModel
from wms.model.checkpoint import load_checkpoint
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.merge_pivot import FirstPassageNet, advance_longitudinal, geometry, hard_st
from wms.physics.accel import baseline_accel
from wms.physics.config import PhysicsConfig
from wms.physics.emergency import leader_inputs, safe_accel
from wms.physics.idm import IDMParams, idm_accel
from wms.physics.neighbours import lead_neighbour
from wms.physics.state import state_kin


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def scene(follower_speed=15.0):
    s = torch.tensor([[30., 5., 90.]], dtype=torch.float64)
    return dict(s_m=s, v_mps=s.new_tensor([[5., follower_speed, 8.]]),
                a_mps2=torch.zeros_like(s), length_m=torch.full_like(s, 5.),
                lane=torch.tensor([[2, 1, 1]]), present=torch.ones_like(s, dtype=torch.bool),
                is_cav=torch.zeros_like(s, dtype=torch.bool))


def drivers():
    return {"time_headway": torch.tensor([[1., 1.2, .7]], dtype=torch.float64),
            "decel": torch.tensor([[2., 3., 5.]], dtype=torch.float64),
            "min_gap": torch.tensor([[2., 1.5, 3.]], dtype=torch.float64)}


def model(enabled=False, hybrid=False, **spec):
    torch.manual_seed(17)
    return WmSModel(ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5",
                              merge_decision="first_passage", lane_decision="sample",
                              merge_passage_law="hybrid" if hybrid else "strict", **spec),
                    physics=PhysicsConfig(secure_gap_veto=enabled)).double().eval()


def fixed_forward(state, **kwargs):
    lane, s = state["lane"], state["s_m"]
    logits = s.new_full((*lane.shape, 3), -20.)
    logits.scatter_(-1, lane.unsqueeze(-1), 20.)
    settings = drivers()
    values = torch.stack([settings.get(name, torch.ones_like(s)) for name in PARAMETER_NAMES], -1)
    result = dict(accel_mps2=torch.zeros_like(s), lane_logits=logits,
                  lane_feasible=torch.ones_like(logits, dtype=torch.bool),
                  hdv_mask=state["present"] & ~state["is_cav"], driver_parameters=values,
                  lane_age=torch.zeros_like(lane), lane_sample=torch.zeros_like(state["present"]))
    if kwargs.get("lane_v5_acc") is not None:
        result["lane_v5_acc"] = kwargs["lane_v5_acc"]
    return result


def native_step(m, state, params=None, **kwargs):
    with patch.object(m, "forward", side_effect=fixed_forward):
        return m.step(state, driver_params=params,
                      lane_uniform=torch.full_like(state["s_m"], .999999), **kwargs)


def oracle_margin(g, d):
    """Independent pair calculation: ramp=0, target follower=1, target leader=2."""
    v, vf, vl = g["v"][0, 0], g["v_f"][0, 0], g["v_l"][0, 0]
    b = d["decel"][0]
    rf = d["min_gap"][0, 1] + vf * d["time_headway"][0, 1] + F.relu(
        vf**2 / (2 * b[1]) - v**2 / (2 * b[0]))
    rl = d["min_gap"][0, 0] + v * d["time_headway"][0, 0] + F.relu(
        v**2 / (2 * b[0]) - vl**2 / (2 * b[2]))
    return torch.minimum((g["gap_f"][0, 0] / rf).log(), (g["gap_l"][0, 0] / rl).log())


def test_schema_roundtrip_assembly_and_legacy_default_omission():
    defaults = TrainConfig()
    assert not defaults.physics.emergency_braking and not defaults.physics.secure_gap_veto
    assert not {"emergency_braking", "secure_gap_veto"} & defaults.to_legacy_flat().keys()
    cfg = TrainConfig.from_dict({
        "model": {"rollout_mode": "driver_params", "lane_model": "hazard_v5",
                  "merge_decision": "first_passage"},
        "physics": {"emergency_braking": True, "secure_gap_veto": True,
                    "inference": {"secure_gap_veto": False}}})
    assert TrainConfig.from_legacy_flat(cfg.to_legacy_flat()) == cfg
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
    physics = build_physics(cfg)
    assert physics.emergency_braking and physics.secure_gap_veto
    assert not replace(physics, **cfg.physics.inference).secure_gap_veto
    for constructor in (PhysicsConfig, PhysicsOptions):
        with pytest.raises(ValueError, match="cannot combine"):
            constructor(emergency_braking=True, emergency_decel_mps2=9.)
    with pytest.raises(ValueError, match="first_passage"):
        TrainConfig.from_dict({"physics": {"secure_gap_veto": True}})
    with pytest.raises(ValueError, match="first_passage"):
        WmSModel(physics=PhysicsConfig(secure_gap_veto=True))


@pytest.mark.parametrize("name", ["emergency_braking", "secure_gap_veto"])
@pytest.mark.parametrize("value", [1, "true", None])
def test_boolean_switches_reject_non_boolean_values(name, value):
    for constructor in (PhysicsConfig, PhysicsOptions):
        with pytest.raises(ValueError, match=name):
            constructor(**{name: value})


@pytest.mark.parametrize("storage", ["embedded", "sidecar", "override"])
@pytest.mark.parametrize("enabled", [False, True])
def test_strict_checkpoint_loading_preserves_supplied_physics_without_new_tensors(
        tmp_path, storage, enabled):
    cfg = TrainConfig.from_dict({"model": {"rollout_mode": "driver_params",
                                          "driver_param_mode": "physical",
                                          "lane_model": "hazard_v5",
                                          "merge_decision": "first_passage"},
                                 "physics": {"emergency_braking": enabled,
                                             "secure_gap_veto": enabled}})
    physics = build_physics(cfg)
    torch.manual_seed(19)
    original = build_model(cfg, physics).eval()
    torch.manual_seed(19)
    other = build_model(cfg, replace(physics, emergency_braking=not enabled,
                                     secure_gap_veto=not enabled)).eval()
    weights = original.state_dict()
    assert weights.keys() == other.state_dict().keys()
    assert all(torch.equal(value, other.state_dict()[key]) for key, value in weights.items())
    flat, block = cfg.to_legacy_flat(), asdict(physics)
    if storage == "embedded":
        flat["physics"] = block
    elif storage == "sidecar":
        (tmp_path / "physics.json").write_text(json.dumps(block), encoding="utf-8")
    path = tmp_path / "synthetic.pt"
    torch.save({"format": "wm_s/1", "model": weights, "config": flat}, path)
    # Embedded full PhysicsConfig blocks use the low-level loader: they are not
    # legacy-flat TrainConfig keys accepted by load_backend's config_factory.
    loaded = (load_checkpoint(path, device="cpu", strict=True).model if storage == "embedded"
              else load_backend(path, device="cpu",
                                physics=block if storage == "override" else None).model)
    assert asdict(loaded.physics) == block
    loaded.load_state_dict(weights, strict=True)
    state = {key: value.float() if value.is_floating_point() else value
             for key, value in scene().items()}
    with torch.no_grad():
        rolled = loaded.rollout(state, n_steps=2, lane_seed=3)
    assert torch.isfinite(rolled["positions"]).all()
    assert torch.isfinite(rolled["speeds"]).all()
    # A pre-R13E block with the same model tensors defaults to the old execution path.
    legacy_block = {key: value for key, value in block.items()
                    if key not in ("emergency_braking", "secure_gap_veto")}
    old = load_checkpoint(path, device="cpu", strict=True, physics=legacy_block).model
    assert not old.physics.emergency_braking and not old.physics.secure_gap_veto
    old.load_state_dict(weights, strict=True)


@pytest.mark.parametrize("hybrid", [False, True])
def test_flags_off_boundary_and_st_match_legacy_arithmetic(hybrid):
    m, state = model(hybrid=hybrid), scene()
    g = geometry(state_kin(state), state["present"], m.physics)
    p, sides = m.merge_pivot.params(), []
    for side in ("f", "l"):
        vf, vl = ((g[f"v_{side}"], g["v"]) if side == "f" else (g["v"], g[f"v_{side}"]))
        req = p[f"s_{side}"] + p[f"T_{side}"] * vf + F.relu(vf**2 - vl**2) / (2*p[f"b_{side}"])
        sides.append(torch.where(g[f"has_{side}"], g[f"gap_{side}"].log() - req.log(),
                                 torch.full_like(vf, torch.inf)))
    expected = torch.where(g["eligible"], torch.minimum(*sides), torch.full_like(g["v"], -torch.inf))
    assert torch.equal(m.merge_pivot(g), expected)
    c, z = expected[:, :1], expected.new_tensor([[.3]])
    soft = torch.sigmoid((c + p["tau"] * z) / m.merge_st_width)
    old_st = (c + p["tau"] * z >= 0).to(c) + (soft - soft.detach())
    assert torch.equal(hard_st(c, p["tau"], z, m.merge_st_width,
                               torch.ones_like(c, dtype=torch.bool)), old_st)


@pytest.mark.parametrize("variant", [{}, {"temporal_accel_lane_bias": True},
                                    {"temporal_accel_state_table": True}])
def test_flags_off_strict_loading_and_seeded_rollout_preserve_temporal_variants(variant):
    off, on = model(**variant), model(True, **variant)
    on.load_state_dict(off.state_dict(), strict=True)
    on.physics = replace(on.physics, secure_gap_veto=False, emergency_braking=False)
    off.float()
    on.float()
    state = {key: value.float() if value.is_floating_point() else value
             for key, value in scene().items()}
    with torch.no_grad():
        a = off.rollout(state, n_steps=4, lane_seed=7)
        b = on.rollout(state, n_steps=4, lane_seed=7)
    assert a.keys() == b.keys()
    for key, value in a.items():
        if torch.is_tensor(value):
            assert torch.equal(value, b[key]), key
    if variant.get("temporal_accel_lane_bias"):
        assert off.temporal_lane_bias is not None
    if variant.get("temporal_accel_state_table"):
        assert off.temporal_state_mean is not None


@pytest.mark.parametrize("speed,accepted", [(2., True), (15., False)])
@pytest.mark.parametrize("hybrid", [False, True])
def test_rollout_veto_and_ungated_replay_boundary(speed, accepted, hybrid):
    m, state = model(True, hybrid), scene(speed)
    with torch.no_grad():
        m.merge_pivot.tau_raw.copy_(torch.log(torch.expm1(torch.tensor(2.))))
    out = native_step(m, state)
    assert out["merge_pivot_eligible"][0, 0]
    assert bool(out["merge_pivot_committed"][0, 0]) is accepted
    assert out["next_lane"][0, 0] == (1 if accepted else 2)
    assert bool(torch.isneginf(out["merge_pivot_c"][0, 0])) is (not accepted)
    # Recorded replay and courtesy intent deliberately call without driver settings.
    g = m.pivot_decision_geometry(state, torch.zeros_like(state["s_m"]))
    before = m.merge_pivot(g)
    grads = torch.autograd.grad(before[0, 0], list(m.merge_pivot.parameters()), allow_unused=True)
    m.physics.secure_gap_veto = False
    after = m.merge_pivot(g)
    assert torch.equal(before, after)
    again = torch.autograd.grad(after[0, 0], list(m.merge_pivot.parameters()), allow_unused=True)
    assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(grads, again))
    assert native_step(m, state)["merge_pivot_committed"][0, 0]


def test_secure_gap_uses_min_gap_asymmetric_braking_and_neighbor_indices():
    state, d, net = scene(), drivers(), FirstPassageNet().double()
    kin = state_kin(state)
    g = geometry(kin, state["present"], PhysicsConfig())
    actual = net.secure_gap_margin(g, d)[0, 0]
    torch.testing.assert_close(actual, oracle_margin(g, d), rtol=1e-14, atol=1e-14)
    perm = torch.tensor([2, 0, 1])
    gp = geometry(kin[:, perm], state["present"][:, perm], PhysicsConfig())
    dp = {name: val[:, perm] for name, val in d.items()}
    torch.testing.assert_close(net.secure_gap_margin(gp, dp)[:, 1], actual.unsqueeze(0))
    kin[..., 4] = 0.
    kin[0, 1, 0], kin[0, 2, 0] = 23.5, 37.
    g = geometry(kin, state["present"], PhysicsConfig())
    assert net.secure_gap_margin(g, d)[0, 0] == 0.  # both exactly at their learned s0
    d["min_gap"][0, 1] += .1
    assert net.secure_gap_margin(g, d)[0, 0] < 0.


@pytest.mark.parametrize("speed", [8., 15.])
def test_secure_gap_st_has_finite_driver_and_state_credit_on_accepted_and_rejected_rows(speed):
    net, state = FirstPassageNet().double(), scene(speed)
    d = {key: value.requires_grad_() for key, value in drivers().items()}
    kin = state_kin(state).requires_grad_()
    g = geometry(kin, state["present"], PhysicsConfig())
    c, margin = net(g), net.secure_gap_margin(g, d)
    tau = c.new_tensor(2.)
    st = hard_st(c, tau, torch.zeros_like(c), 1., g["eligible"], secure_gap_margin=margin)
    grads = torch.autograd.grad(st[0, 0], [kin, *d.values()])
    assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)
    # Finite differences certify the labeled smooth estimator, not the hard veto.
    def smooth(t):
        values = {**d, "time_headway": t}
        return torch.sigmoid(c.detach()[0, 0]) * torch.sigmoid(oracle_margin(g, values) / tau)
    t = d["time_headway"].detach().clone().requires_grad_()
    assert torch.autograd.gradcheck(smooth, (t,), eps=1e-5, atol=2e-8, rtol=2e-5)
    reference = torch.autograd.grad(smooth(t), t)[0]
    torch.testing.assert_close(grads[1], reference)


def test_callback_receives_inferred_settings_and_explicit_settings_take_precedence():
    state, m = scene(), model(True)
    state["is_cav"][0, 2] = True
    received = []
    def controller(current):
        received.append(current)
        return torch.stack((current["s_m"], current["lane"].to(current["s_m"]),
                            current["v_mps"], torch.zeros_like(current["s_m"])), -1)
    for explicit in (None, {key: value + .1 for key, value in drivers().items()}):
        native_step(m, state, explicit, cav_controller=controller)
        expected = drivers() if explicit is None else explicit
        for key, name in (("secure_gap_T_s", "time_headway"),
                          ("secure_gap_b_mps2", "decel"), ("secure_gap_s0_m", "min_gap")):
            assert torch.equal(received[-1][key], expected[name])
            assert torch.equal(received[-1]["inferred_driver_parameters"][..., PARAMETER_NAMES.index(name)],
                               expected[name])
        assert not any(key.startswith("secure_gap_") for key in state)
    m.physics.secure_gap_veto = False
    native_step(m, state, cav_controller=controller)
    assert not any(key.startswith("secure_gap_") for key in received[-1])
    assert received[-1]["inferred_idm_delta"] == m.physics.idm()[4]
    for name, value in drivers().items():
        assert torch.equal(received[-1]["inferred_driver_parameters"][..., PARAMETER_NAMES.index(name)], value)


def test_emergency_envelope_gradcheck_and_bounded_infeasible_state():
    def envelope(x):
        return safe_accel(x[..., 0], x[..., 1], x[..., 2], x[..., 4], x[..., 3],
                          dt=.5, emergency_decel=9.)
    x = torch.tensor([[20., 15., 20., 2.5, 2.], [10., 15., 80., 2.5, -.4]],
                     dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(envelope, (x,), eps=1e-5, atol=2e-6, rtol=3e-4)
    grad = torch.autograd.grad(envelope(x).sum(), x)[0]
    assert torch.isfinite(grad).all() and torch.all(grad.abs().sum(0) > 1e-5)
    unsafe = torch.tensor([[30., 0., .5, 2.5, 3.]], dtype=torch.float64)
    assert envelope(unsafe).item() == pytest.approx(-9.)


def test_flags_off_baseline_acceleration_is_exact_legacy_idm():
    kin, present = state_kin(scene()), scene()["present"]
    physics = PhysicsConfig()
    d = {**drivers(), "v0": torch.full_like(kin[..., 4], 31.29),
         "accel": torch.full_like(kin[..., 4], 2.6)}
    gap, leader_v = lead_neighbour(kin, present, physics)
    expected = idm_accel(kin[..., 4], leader_v, gap, IDMParams.from_driver_params(d),
                         accel_min=physics.accel_min_mps2, accel_max=physics.accel_max_mps2)
    expected = torch.minimum(torch.maximum(expected, -d["decel"]), d["accel"])
    assert torch.equal(baseline_accel(kin, present, physics, d), expected)


@pytest.mark.parametrize("obstacle", ["leader", "lane_end"])
def test_emergency_baseline_preserves_feasible_approach_for_80_seconds(obstacle):
    physics = PhysicsConfig(emergency_braking=True, lane_rule="forced_merge_only")
    kin = torch.zeros(1, 2, 7, dtype=torch.float64)
    kin[..., 2], kin[..., 6] = 1., 5.
    kin[0, 0, 4] = 18.
    kin[0, 1, 0] = 26.  # 21 m bumper gap > s0 + stopping distance at B=9
    present = torch.ones(1, 2, dtype=torch.bool)
    if obstacle == "lane_end":
        kin[0, 0, 0] = physics.merge_zone_s1_m - 21.
        kin[0, 0, 1:4] = kin.new_tensor([0., 0., 1.])
        kin[0, 1, 0] = physics.merge_zone_s1_m + 5.
        present[0, 1] = False
    settings = {"time_headway": torch.full((1, 2), 1.1), "min_gap": torch.full((1, 2), 2.5),
                "decel": torch.ones(1, 2), "accel": torch.full((1, 2), 2.6),
                "v0": torch.full((1, 2), 31.29)}
    emergency_used = False
    for _ in range(160):
        accel = baseline_accel(kin, present, physics, settings)
        emergency_used |= bool((accel[:, 0] < -1.01).any())
        assert torch.isfinite(accel).all() and torch.all(accel >= -9.)
        commands = torch.stack((accel[:, 0], torch.full_like(accel[:, 0], -9.)), -1)
        moved = advance_longitudinal(kin, commands, .5, stop_time_integration=False)
        assert torch.all(moved[:, 1, 0] - 5. - moved[:, 0, 0] >= 2.5 - 1e-7)
        kin = moved
    assert emergency_used and kin[0, 0, 4] < .02
    # No real leader on an otherwise free mainline creates no phantom constraint.
    _, _, has_leader = leader_inputs(kin, torch.tensor([[True, False]]), physics)
    if obstacle == "leader":
        assert not has_leader.any()
