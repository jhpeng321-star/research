"""R13E synthetic CPU contracts: driver-specific veto, replay and rollout adjoint.

Finite differences below certify the explicitly labeled smooth ST surrogate,
not a derivative of the discontinuous hard decision. No datasets/checkpoints.
"""
import io
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from wms.cli.assemble import build_physics
from wms.config.train import PhysicsOptions, TrainConfig
from wms.model import ModelSpec, WmSModel
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.merge_pivot import FirstPassageNet, geometry, hard_st, update_state
from wms.physics.config import PhysicsConfig
from wms.physics.state import state_kin


DTYPE = torch.float64


def scene(follower_speed=15.0):
    s = torch.tensor([[30., 5., 90.]], dtype=DTYPE)
    return dict(s_m=s, v_mps=s.new_tensor([[5., follower_speed, 8.]]),
                a_mps2=torch.zeros_like(s), length_m=torch.full_like(s, 5.),
                lane=torch.tensor([[2, 1, 1]]), present=torch.ones_like(s, dtype=torch.bool),
                is_cav=torch.zeros_like(s, dtype=torch.bool))


def drivers():
    # Deliberately heterogeneous, including different braking of each pair member.
    return {"time_headway": torch.tensor([[1., 1.2, .7]], dtype=DTYPE),
            "decel": torch.tensor([[2., 3., 5.]], dtype=DTYPE),
            "min_gap": torch.tensor([[2., 1.5, 3.]], dtype=DTYPE)}


def model(enabled=False, hybrid=False):
    torch.manual_seed(17)
    return WmSModel(ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5",
                              merge_decision="first_passage", lane_decision="sample",
                              merge_passage_law="hybrid" if hybrid else "strict"),
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


def native_step(m, state, params=None, uniform=.999999):
    with patch.object(m, "forward", side_effect=fixed_forward):
        return m.step(state, driver_params=params,
                      lane_uniform=torch.full_like(state["s_m"], uniform))


def oracle_margin(g, d):
    # Independent pair arithmetic for this fixture: ramp=0, follower=1, leader=2.
    v, vf, vl = g["v"][0, 0], g["v_f"][0, 0], g["v_l"][0, 0]
    b = d["decel"][0]
    rf = d["min_gap"][0, 1] + vf * d["time_headway"][0, 1] + F.relu(
        vf**2 / (2 * b[1]) - v**2 / (2 * b[0]))
    rl = d["min_gap"][0, 0] + v * d["time_headway"][0, 0] + F.relu(
        v**2 / (2 * b[0]) - vl**2 / (2 * b[2]))
    return torch.minimum((g["gap_f"][0, 0] / rf).log(), (g["gap_l"][0, 0] / rl).log())


def test_config_roundtrip_assembly_default_omission_and_emergency_preserved():
    assert not PhysicsConfig().secure_gap_veto
    assert "secure_gap_veto" not in TrainConfig().to_legacy_flat()
    cfg = TrainConfig.from_dict(dict(
        model=dict(rollout_mode="driver_params", lane_model="hazard_v5",
                   merge_decision="first_passage"),
        physics=dict(secure_gap_veto=True, emergency_braking=True,
                     inference={"secure_gap_veto": False})))
    assert TrainConfig.from_legacy_flat(cfg.to_legacy_flat()) == cfg
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
    p = build_physics(cfg)
    assert p.secure_gap_veto and p.emergency_braking
    assert not replace(p, **cfg.physics.inference).secure_gap_veto
    with pytest.raises(ValueError, match="first_passage"):
        TrainConfig.from_dict({"physics": {"secure_gap_veto": True}})
    with pytest.raises(ValueError, match="first_passage"):
        WmSModel(physics=PhysicsConfig(secure_gap_veto=True))


@pytest.mark.parametrize("value", [1, "true", None])
def test_boolean_validation(value):
    with pytest.raises(ValueError, match="secure_gap_veto"):
        PhysicsConfig(secure_gap_veto=value)
    with pytest.raises(ValueError, match="secure_gap_veto"):
        PhysicsOptions(secure_gap_veto=value)


@pytest.mark.parametrize("hybrid", [False, True])
def test_off_is_bit_identical_to_legacy_boundary_st_rng_and_checkpoint_tensors(hybrid):
    m = model(hybrid=hybrid)
    g = geometry(state_kin(scene()), scene()["present"], m.physics)
    p = m.merge_pivot.params()
    sides = []
    for side in ("f", "l"):
        vf, vl = ((g[f"v_{side}"], g["v"]) if side == "f" else
                  (g["v"], g[f"v_{side}"]))
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
    # Same seeded construction on/off creates exactly the same tensor bytes and RNG.
    off_rng = torch.random.get_rng_state().clone()
    on = model(enabled=True, hybrid=hybrid)
    assert torch.equal(off_rng, torch.random.get_rng_state())
    a, b = io.BytesIO(), io.BytesIO()
    torch.save(m.state_dict(), a)
    torch.save(on.state_dict(), b)
    assert a.getvalue() == b.getvalue()
    on.physics.secure_gap_veto = False
    on.load_state_dict(m.state_dict(), strict=True)
    m.float()
    on.float()
    torch.random.set_rng_state(off_rng)
    x = m.step(scene())
    rng_x = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(off_rng)
    y = on.step(scene())
    assert torch.equal(rng_x, torch.random.get_rng_state())
    for key, val in x.items():
        if torch.is_tensor(val):
            assert torch.equal(val, y[key]), key


@pytest.mark.parametrize("speed,accepted", [(2., True), (15., False)])
@pytest.mark.parametrize("hybrid", [False, True])
def test_native_slow_fast_follower_accept_reject_despite_willing_latent(speed, accepted, hybrid):
    m, state = model(True, hybrid), scene(speed)
    with torch.no_grad():
        m.merge_pivot.tau_raw.copy_(torch.log(torch.expm1(torch.tensor(2., dtype=DTYPE))))
    rng = torch.random.get_rng_state().clone()
    out = native_step(m, state)
    assert out["merge_pivot_eligible"][0, 0]  # physical eligibility is unchanged
    assert bool(out["merge_pivot_committed"][0, 0]) is accepted
    assert out["next_lane"][0, 0] == (1 if accepted else 2)
    assert bool(torch.isneginf(out["merge_pivot_c"][0, 0])) is (not accepted)
    m.physics.secure_gap_veto = False
    torch.random.set_rng_state(rng)
    assert native_step(m, state)["merge_pivot_committed"][0, 0]


def test_exact_asymmetric_braking_formula_and_neighbor_permutation():
    state, d, net = scene(), drivers(), FirstPassageNet().double()
    kin = state_kin(state)
    g = geometry(kin, state["present"], PhysicsConfig())
    actual = net.secure_gap_margin(g, d)[0, 0]
    torch.testing.assert_close(actual, oracle_margin(g, d), rtol=1e-14, atol=1e-14)
    perm = torch.tensor([2, 0, 1])
    gp = geometry(kin[:, perm], state["present"][:, perm], PhysicsConfig())
    dp = {name: val[:, perm] for name, val in d.items()}
    torch.testing.assert_close(net.secure_gap_margin(gp, dp)[:, 1], actual.unsqueeze(0))
    # Unequal b must matter even when follower is slower than its leader.
    kin[0, 0, 4], kin[0, 1, 4] = 10., 8.
    d["decel"][0, 0], d["decel"][0, 1] = 6., 1.
    g = geometry(kin, state["present"], PhysicsConfig())
    torch.testing.assert_close(net.secure_gap_margin(g, d)[0, 0], oracle_margin(g, d))


def test_leader_side_can_veto_and_equal_gap_is_secure():
    net, d, state = FirstPassageNet().double(), drivers(), scene(2.)
    d["time_headway"][0, 0] = 20.
    g = geometry(state_kin(state), state["present"], PhysicsConfig())
    assert g["gap_f"][0, 0] > 0 and net.secure_gap_margin(g, d)[0, 0] < 0
    assert torch.isneginf(net(g, d)[0, 0])
    # At rest both sides require just the follower's learned standstill gap.
    kin = state_kin(state)
    kin[..., 4] = 0.
    kin[0, 1, 0], kin[0, 2, 0] = 23.5, 37.
    g = geometry(kin, state["present"], PhysicsConfig())
    assert net.secure_gap_margin(g, d)[0, 0] == 0
    assert torch.isfinite(net(g, d)[0, 0])


@pytest.mark.parametrize("overlap", [False, True])
def test_absence_invalid_padding_and_overlap_remain_exact_with_zero_adjoint(overlap):
    state, d, net = scene(), drivers(), FirstPassageNet().double()
    kin, present = state_kin(state), state["present"].clone()
    present[0, 1:] = False
    kin[0, 1:] = torch.nan
    if overlap:
        present[0, 1] = True
        kin[0, 1] = kin[0, 0].clone()
        kin[0, 1, 1:4] = kin.new_tensor([0., 1., 0.])
    d = {name: value.clone().requires_grad_() for name, value in d.items()}
    with torch.no_grad():
        for value in d.values():
            value[~present] = torch.nan
    kin.requires_grad_()
    g = geometry(kin, present, PhysicsConfig())
    c = net(g, d)
    assert bool(torch.isneginf(c[0, 0])) is overlap
    assert bool(g["overlap"][0, 0]) is overlap
    st = hard_st(net(g), net.params()["tau"], torch.zeros_like(c), 1., g["eligible"],
                 secure_gap_margin=net.secure_gap_margin(g, d))
    grads = torch.autograd.grad(st.sum(), [kin, *d.values()])
    for grad in grads:
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) == 0


@pytest.mark.parametrize("name", ["time_headway", "decel", "min_gap"])
def test_invalid_present_driver_setting_vetoes_without_nan_gradients(name):
    state, d, net = scene(), drivers(), FirstPassageNet().double()
    d[name][0, 1] = torch.nan
    d = {key: value.requires_grad_() for key, value in d.items()}
    g = geometry(state_kin(state), state["present"], PhysicsConfig())
    c = net(g, d)
    assert torch.isneginf(c[0, 0])
    st = hard_st(net(g), net.params()["tau"], torch.zeros_like(c), 1., g["eligible"],
                 secure_gap_margin=net.secure_gap_margin(g, d))
    for grad in torch.autograd.grad(st.sum(), list(d.values())):
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad) == 0


@pytest.mark.parametrize("speed", [8., 15.])
@pytest.mark.parametrize("parameter", ["tau_raw", "time_headway", "decel", "min_gap", "speed"])
def test_native_rollout_surrogate_fd_is_nonzero_on_accepted_and_rejected_gaps(speed, parameter):
    m, state, d = model(True), scene(speed), drivers()
    # Keep the existing learned scale away from sigmoid saturation for a meaningful FD.
    with torch.no_grad():
        m.merge_pivot.tau_raw.copy_(torch.log(torch.expm1(torch.tensor(2., dtype=DTYPE))))
    d = {name: value.requires_grad_() for name, value in d.items()}
    state["v_mps"].requires_grad_()
    target = (m.merge_pivot.tau_raw if parameter == "tau_raw" else
              state["v_mps"] if parameter == "speed" else d[parameter])
    index = () if parameter == "tau_raw" else (0, 1)
    out = native_step(m, state, d, uniform=.5)  # z=0 isolates the veto-scale derivative
    ad = torch.autograd.grad(out["merge_pivot_lane_weights"][0, 0, 1], target)[0][index]
    assert torch.isfinite(ad) and ad.abs() > 1e-5

    def smooth_reference():
        # FD the defining continuous surrogate including the native one-step motion.
        g = m.pivot_decision_geometry(state, torch.zeros_like(state["s_m"]))
        c = m.merge_pivot(g)[0, 0]
        return torch.sigmoid(c / m.merge_st_width) * torch.sigmoid(
            oracle_margin(g, d) / m.merge_pivot.params()["tau"])

    eps = 1e-5
    with torch.no_grad():
        original = target[index].clone()
        target[index] = original + eps
        plus = smooth_reference()
        target[index] = original - eps
        minus = smooth_reference()
        target[index] = original
    fd = (plus - minus) / (2*eps)
    torch.testing.assert_close(ad, fd, atol=2e-8, rtol=2e-5)


def test_rejected_tick_retains_latent_and_can_merge_when_gap_becomes_secure():
    m, state = model(True), scene()
    first = native_step(m, state)
    assert not first["merge_pivot_cross"][0, 0] and not first["merge_pivot_terminal"][0, 0]
    carry = {key.removeprefix("merge_pivot_"): val for key, val in first.items()
             if key in {"merge_pivot_z", "merge_pivot_maximum", "merge_pivot_present",
                        "merge_pivot_identity", "merge_pivot_started", "merge_pivot_terminal"}}
    state = scene(2.)
    g = geometry(state_kin(state), state["present"], m.physics)
    result = update_state(m.merge_pivot(g, drivers()), m.merge_pivot.params()["tau"],
                          g["ramp"], state["present"], carry=carry)
    assert torch.equal(result["z"], first["merge_pivot_z"])
    assert result["cross"][0, 0]


@pytest.mark.parametrize("hybrid", [False, True])
def test_recorded_replay_geometry_likelihood_and_gradients_are_ungated(hybrid):
    from wms.train.inputs import build_targets, prepare_inputs
    from wms.train.losses.base import LossContext
    from wms.train.losses.merge_pivot import MergePivotConfig, MergePivotTerm
    from wms.train.settings import TrainSettings

    results = []
    for enabled in (False, True):
        m = model(enabled, hybrid)
        # Pre-zone row certifies entry, then an event in an intentionally insecure gap.
        kin = state_kin(scene()).float()
        states = kin[:, None].repeat(1, 4, 1, 1)
        states[0, :, 0, 0] = torch.tensor([-1., 1., 3., 5.])
        states[0, :, 1, 0] = torch.tensor([-20., -22., -30., -14.])
        states[0, -1, 0, 2:4] = torch.tensor([1., 0.])
        present = torch.ones(states.shape[:-1], dtype=torch.bool)
        batch = dict(states=states, present=present, is_cav=torch.zeros_like(present))
        settings = TrainSettings(rollout_mode="driver_params", lane_model="hazard_v5")
        inp = prepare_inputs(batch, m, settings, torch.device("cpu"), 3)
        targets = build_targets(inp, batch, settings, .5, torch.device("cpu"), physics=m.physics)
        targets.prepared["merge_pivot_source"] = (m, inp)
        term = MergePivotTerm(MergePivotConfig(tf_weight=1.,
                                               passage_law="hybrid" if hybrid else "strict"))
        with patch.object(m, "forward", side_effect=fixed_forward):
            term.prepare(targets)
            # An explicit rollout call on this same replay geometry really is vetoed.
            g = {key: val[:, 1] for key, val in targets.prepared["merge_pivot"]["geometry"].items()}
            assert torch.isneginf(m.merge_pivot(g, drivers())[0, 0])
            value = term(LossContext(pred={}, current=inp.initial, targets=targets, c0=0, c1=3))
        assert torch.isfinite(value.value)
        grads = torch.autograd.grad(value.value, list(m.merge_pivot.parameters()))
        results.append((value.value.detach(), grads))
    assert torch.equal(results[0][0], results[1][0])
    assert results[0][0] > 0
    for a, b in zip(results[0][1], results[1][1]):
        assert torch.isfinite(a).all() and torch.equal(a, b)
