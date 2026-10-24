"""Focused R13Y contracts: synthetic tensors only, CPU only, no training or simulator."""
import math

import pytest
import torch
import torch.nn.functional as F

from wms.model.merge_yield import LearnedMergeYield
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.physics.idm import IDMParams, idm_accel
from wms.physics.mobil import merge_end_blocked
from wms.train.inputs import state_from_kin


@pytest.fixture(autouse=True)
def cpu_only():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.device("cpu"):
        yield
    torch.set_num_threads(previous)


def physics():
    return PhysicsConfig(lane_learnable=True, lane_rule="forced_merge_only",
                         lane_gap_rule="speed", stop_time_integration=True)


def scene(positions=(75., 15.), speeds=(0., 25.), lanes=(2, 1), *, dtype=torch.float32):
    kin = torch.zeros(1, len(positions), 7, dtype=dtype, device="cpu")
    kin[..., 0] = kin.new_tensor(positions)
    kin[..., 1:4] = F.one_hot(torch.tensor([lanes], device="cpu"), 3).to(dtype)
    kin[..., 4] = kin.new_tensor(speeds)
    kin[..., 6] = 5.
    return kin


def drivers(kin, **overrides):
    values = dict(v0=31.29, time_headway=1.1, min_gap=2.5, accel=2.6, decel=4.5,
                  politeness=0.2, mobil_threshold=0.1, merge_gap=6., merge_urgency=1.)
    values.update(overrides)
    return {key: torch.full_like(kin[..., 0], value) for key, value in values.items()}


def state(kin, *, present=None, cav=None):
    present = (torch.ones(kin.shape[:2], dtype=torch.bool, device="cpu")
               if present is None else present)
    cav = torch.zeros_like(present) if cav is None else cav
    return {**state_from_kin(kin, present), "is_cav": cav}


def model(*, enabled=True, courtesy=False):
    # Explicit driver parameters remove encoder noise; no residual acceleration is emitted.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1319)
        return WmSModel(ModelSpec(rollout_mode="driver_params",
                                  mainline_merge_yield_learnable=enabled,
                                  follower_courtesy=courtesy,
                                  follower_courtesy_intent="none"), physics=physics()).cpu().eval()


def held_lane_logits(st):
    logits = st["veh_kin"].new_full((*st["present"].shape, 3), -1.e6)
    return logits.scatter(-1, st["lane"].unsqueeze(-1), 0.)


def advance(m, st, params):
    """Exercise the real step integrator; hold lane decisions and the stopped CAV ramp."""
    out = m.step(st, driver_params=params, baseline_lane_logits=held_lane_logits(st),
                 cav_teacher=m.hold_teacher(st))
    kin = torch.cat((out["next_s_m"].unsqueeze(-1),
                     F.one_hot(out["next_lane"], 3).to(st["veh_kin"].dtype),
                     out["next_v_mps"].unsqueeze(-1), out["accel_mps2"].unsqueeze(-1),
                     st["length_m"].unsqueeze(-1)), -1)
    return state(kin, present=out["next_present"], cav=st["is_cav"]), out


def scalar_fd(fn, param, h=1.e-3):
    original = param.detach().clone()
    try:
        with torch.no_grad():
            param.copy_(original + h)
            plus = fn().detach().clone()
            param.copy_(original - h)
            minus = fn().detach().clone()
        return (plus - minus) / (2. * h)
    finally:
        with torch.no_grad():
            param.copy_(original)


def test_shared_positive_parameters_and_initialization():
    layer = LearnedMergeYield().cpu()
    assert set(dict(layer.named_parameters())) == {"log_decel", "log_distance"}
    assert all(p.shape == () and p.device.type == "cpu" for p in layer.parameters())
    assert layer.decel_mps2.item() == pytest.approx(2.)
    assert layer.distance_m.item() == pytest.approx(60.)
    with torch.no_grad():
        layer.log_decel.add_(math.log(1.5))
        layer.log_distance.add_(math.log(0.5))
    assert layer.decel_mps2.item() == pytest.approx(3.)
    assert layer.distance_m.item() == pytest.approx(30.)


@pytest.mark.parametrize("invalid", ["fixed_driver", "fixed_lanes", "black_box", "backend"])
def test_enable_hook_rejects_incompatible_cores(invalid):
    spec = ModelSpec(rollout_mode="residual" if invalid == "fixed_driver" else "driver_params",
                     black_box=invalid == "black_box")
    p = physics()
    p.lane_learnable = invalid != "fixed_lanes"
    kwargs = {"physics_step": lambda st: {}} if invalid == "backend" else {}
    m = WmSModel(spec, physics=p, **kwargs).cpu()
    with pytest.raises(ValueError, match="learned-driver lane_learnable"):
        m.enable_mainline_merge_yield_learnable()


def test_enable_hook_is_idempotent_and_does_not_draw_rng():
    m = model(enabled=False)
    assert not m.mainline_merge_yield_learnable and m.merge_yield is None
    before = torch.random.get_rng_state().clone()
    m.enable_mainline_merge_yield_learnable()
    layer = m.merge_yield
    m.enable_mainline_merge_yield_learnable()
    assert m.merge_yield is layer and m.mainline_merge_yield_learnable
    assert torch.equal(before, torch.random.get_rng_state())


def test_only_nearest_eligible_lane_one_follower_is_selected():
    # The closer lane-0 car, an overlapping lane-1 body, an absent car and a car
    # ahead of the merger must not replace the nearest strictly-behind follower.
    kin = scene((75., 15., -25., 65., 73., 80., 40.),
                (0., 25., 25., 25., 0., 0., 25.), (2, 1, 1, 0, 1, 1, 1))
    st = state(kin)
    st["present"][0, -1] = False
    raw = torch.full_like(kin[..., 0], 1.5)
    out, selected = LearnedMergeYield()(kin, st["present"], raw, physics(), drivers(kin))
    assert selected.tolist() == [[False, True, False, False, False, False, False]]
    assert not selected.requires_grad and selected.dtype == torch.bool
    assert out[0, 1] < raw[0, 1]
    assert torch.equal(out[~selected], raw[~selected])


def test_both_log_parameter_gradients_are_nonzero_and_match_finite_differences():
    layer = LearnedMergeYield().double()
    kin = scene(dtype=torch.float64)
    st, params, p = state(kin), drivers(kin), physics()
    raw = torch.full_like(kin[..., 0], 1.5)
    assert merge_end_blocked(kin, st["present"], p)[0, 0]
    virtual = idm_accel(kin[:, 1, 4], kin[:, 0, 4], kin.new_tensor([52.5]),
                       IDMParams.from_driver_params({k: v[:, 1] for k, v in params.items()}),
                       accel_min=p.accel_min_mps2, accel_max=p.accel_max_mps2)
    assert virtual.item() < -2. and params["decel"][0, 1] > 2.

    def evaluate():
        out, selected = layer(kin, st["present"], raw, p, params)
        assert selected.tolist() == [[False, True]]
        return out[0, 1]

    # Front distance is 60 m: both the sigmoid slope and the 2 m/s² cap are live.
    grads = torch.autograd.grad(evaluate(), (layer.log_decel, layer.log_distance))
    for param, grad in zip((layer.log_decel, layer.log_distance), grads):
        assert torch.isfinite(grad) and abs(grad.item()) > 0.1
        assert grad.item() == pytest.approx(scalar_fd(evaluate, param, 1.e-5).item(),
                                          rel=1.e-6, abs=1.e-7)


def test_selected_gap_and_speed_state_adjoint_matches_finite_differences():
    layer = LearnedMergeYield().double()
    # Unsaturated IDM here: ramp/follower speeds AND gathered bumper gap are live.
    kin = scene(speeds=(0., 17.), dtype=torch.float64).requires_grad_(True)
    present = torch.ones(1, 2, dtype=torch.bool)
    params = drivers(kin)
    raw = torch.full_like(kin[..., 0], 1.5)

    def evaluate(x):
        out, selected = layer(x, present, raw, physics(), params)
        assert selected.tolist() == [[False, True]]
        return out[0, 1]

    grad, = torch.autograd.grad(evaluate(kin), kin)
    assert torch.isfinite(grad).all()
    for slot, channel in ((0, 0), (1, 0), (0, 6), (0, 4), (1, 4)):
        plus, minus = kin.detach().clone(), kin.detach().clone()
        plus[0, slot, channel] += 1.e-5
        minus[0, slot, channel] -= 1.e-5
        fd = (evaluate(plus) - evaluate(minus)) / 2.e-5
        actual = grad[0, slot, channel].item()
        assert abs(actual) > 1.e-3, (slot, channel, actual)
        assert actual == pytest.approx(fd.item(), rel=1.e-6, abs=1.e-7), (slot, channel)


def test_virtual_gap_and_range_use_the_followers_learned_min_gap():
    layer = LearnedMergeYield().double()
    kin = scene(speeds=(0., 17.), dtype=torch.float64)
    st, p = state(kin), physics()
    raw = torch.full_like(kin[..., 0], 1.5)
    outputs = []
    for min_gap in (1.5, 3.5):
        params = drivers(kin, min_gap=min_gap)
        # Offset from 60 m makes the gate width observable independently of the cap.
        shifted = kin.clone()
        shifted[0, 1, 0] -= 2.
        out, selected = layer(shifted, st["present"], raw, p, params)
        assert selected[0, 1]
        gap = shifted[0, 0, 0] - shifted[0, 0, 6] - shifted[0, 1, 0] - min_gap
        own = {k: v[:, 1] for k, v in params.items()}
        virtual = idm_accel(shifted[:, 1, 4], shifted[:, 0, 4], gap,
                            IDMParams.from_driver_params(own),
                            accel_min=p.accel_min_mps2, accel_max=p.accel_max_mps2)
        assert -2. < virtual.item() < 0.  # check actual IDM gap, not only the capped branch
        weight = torch.sigmoid((layer.distance_m - 62.) / min_gap)
        expected = 1.5 + weight * (virtual[0] - 1.5)
        torch.testing.assert_close(out[0, 1], expected)
        outputs.append(out[0, 1])
    assert abs((outputs[0] - outputs[1]).item()) > 0.05


def test_own_learned_deceleration_caps_yield():
    layer = LearnedMergeYield().double()
    kin = scene(dtype=torch.float64)
    raw = torch.full_like(kin[..., 0], 1.5)
    with torch.no_grad():
        layer.log_decel.fill_(math.log(6.))
    params = drivers(kin, decel=0.8)
    out, selected = layer(kin, state(kin)["present"], raw, physics(), params)
    assert selected[0, 1]
    weight = torch.sigmoid((layer.distance_m - 60.) / params["min_gap"][0, 1])
    torch.testing.assert_close(out[0, 1], 1.5 + weight * (-0.8 - 1.5))
    grad, = torch.autograd.grad(out[0, 1], layer.log_decel)
    assert grad.item() == 0.  # the driver's b, rather than shared b, is binding


def test_direct_physics_and_forward_apply_yield_once_from_raw_baseline():
    off, on = model(enabled=False), model()
    kin = scene()
    st, params = state(kin), drivers(kin)
    raw = off.physics_accel(st, params)
    assert raw[0, 1] > 0.
    expected, selected = on.merge_yield(kin, st["present"], raw, on.physics, params)
    assert selected[0, 1] and expected[0, 1] < raw[0, 1]
    assert torch.equal(on._physics_step(st, params)[0], raw)
    assert torch.equal(on.physics_accel(st, params), expected)
    out = on(st, driver_params=params)
    assert torch.equal(out["accel_mps2"], expected)


def test_short_step_rollout_pathwise_gradients_match_fd_for_both_log_parameters():
    m = model()
    kin = scene()
    initial = state(kin, cav=torch.tensor([[True, False]]))
    params = drivers(kin)

    def terminal():
        st = initial
        for _ in range(3):
            st, _ = advance(m, st, params)
            assert st["lane"].tolist() == [[2, 1]]
        return st["v_mps"][0, 1] + 0.1 * st["s_m"][0, 1]

    parameters = (m.merge_yield.log_decel, m.merge_yield.log_distance)
    grads = torch.autograd.grad(terminal(), parameters)
    for param, grad in zip(parameters, grads):
        assert torch.isfinite(grad) and abs(grad.item()) > 1.e-2
        for h in (1.e-3, 2.e-3):
            assert grad.item() == pytest.approx(scalar_fd(terminal, param, h).item(),
                                              rel=1.e-2, abs=2.e-3)


@pytest.mark.parametrize("case", ["absent", "merged", "upstream", "fast", "secure"])
def test_no_blocked_ramp_is_bitexact_noop(case):
    off, on = model(enabled=False), model()
    kin = scene()
    present = torch.ones(1, 2, dtype=torch.bool)
    if case == "absent":
        present[0, 0] = False
    elif case == "merged":
        kin[0, 0, 1:4] = kin.new_tensor([0., 1., 0.])
    elif case == "upstream":
        kin[0, 0, 0] = 30.
        kin[0, 1, 0] = -30.
    elif case == "fast":
        kin[0, 0, 4] = 5.
    else:
        kin[0, 1, 0] = -40.
        kin[0, 1, 4] = 5.
    st, params = state(kin, present=present), drivers(kin)
    assert not merge_end_blocked(kin, present, on.physics).any()
    assert torch.equal(on.physics_accel(st, params), off.physics_accel(st, params))
    a, b = off(st, driver_params=params), on(st, driver_params=params)
    for key in ("accel_mps2", "lane_logits"):
        assert torch.equal(a[key], b[key]), (case, key)


def test_cav_follower_is_unchanged_in_direct_forward_and_step_paths():
    off, on = model(enabled=False), model()
    kin = scene((75., 15., -20.), (0., 25., 25.), (2, 1, 1))
    st = state(kin, cav=torch.tensor([[False, True, False]]))
    params = drivers(kin)
    raw = off.physics_accel(st, params)
    bound, selected = on.merge_yield(kin, st["present"], raw, on.physics, params,
                                    is_cav=st["is_cav"])
    assert not selected.any()  # do not reassign a CAV's yield to the second follower
    assert torch.equal(bound, raw)
    assert torch.equal(on.physics_accel(st, params), raw)
    assert torch.equal(on(st, driver_params=params)["accel_mps2"],
                       off(st, driver_params=params)["accel_mps2"])
    a, _ = advance(off, st, params)
    b, _ = advance(on, st, params)
    for key in ("s_m", "v_mps", "lane"):
        assert torch.equal(a[key][st["is_cav"]], b[key][st["is_cav"]]), key


@pytest.mark.parametrize("offset", [-0.5, -5., 3.])
def test_courtesy_and_yield_compete_by_minimum_without_double_count(offset):
    m = model(courtesy=True)
    kin = scene()
    st, params = state(kin), drivers(kin)
    raw = m._physics_step(st, params)[0]
    bound, selected, weight = m.merge_yield(kin, st["present"], raw, m.physics, params,
                                            return_influence=True)
    with torch.no_grad():
        m.follower_courtesy.net[-1].bias.fill_(offset)
    out = m(st, driver_params=params)
    assert selected[0, 1] and out["courtesy_applies"][0, 1]
    assert out["courtesy_accel_mps2"][0, 1].item() == pytest.approx(offset)
    assert torch.equal(out["accel_pre_courtesy_mps2"], raw)
    cap = bound[0, 1] + (1. - weight[0, 1]) * max(offset, 0.)
    expected = torch.minimum(raw[0, 1] + offset, cap)
    torch.testing.assert_close(out["accel_mps2"][0, 1], expected)
    if offset == -5.:
        assert raw[0, 1] + offset < bound[0, 1]  # stronger courtesy wins
    else:
        assert cap < raw[0, 1] + offset  # yield constrains positive courtesy too


def test_distant_blocked_merger_does_not_cancel_positive_courtesy():
    m = model(courtesy=True)
    # A very fast follower makes the legacy secure-gap mask blocked despite the
    # large distance; the learned influence is numerically zero at this range.
    kin = scene((75., -925.), (0., 100.))
    st, params = state(kin), drivers(kin, v0=200.)
    assert merge_end_blocked(kin, st["present"], m.physics)[0, 0]
    with torch.no_grad():
        m.follower_courtesy.net[-1].bias.fill_(3.)
    out = m(st, driver_params=params)
    assert out["courtesy_applies"][0, 1]
    m.mainline_merge_yield_learnable = False
    off = m(st, driver_params=params)
    assert torch.equal(out["accel_mps2"], off["accel_mps2"])


def blocked_rollout(m, *, chain=False):
    # Lane 1 is free ahead. Approach until the follower crawls beside the
    # blocked merger; then replay the CAV merge/removal in the release test.
    kin = (scene((75., 45., -5., -55.), (0., 8., 8., 8.), (2, 1, 1, 1))
           if chain else scene((75., 45.), (0., 8.), (2, 1)))
    cav = torch.zeros(kin.shape[:2], dtype=torch.bool)
    cav[0, 0] = True  # controlled stopped ramp, lane 1 otherwise free ahead
    st, params = state(kin, cav=cav), drivers(kin)
    history = [st]
    with torch.no_grad():
        for _ in range(9):
            st, _ = advance(m, st, params)
            history.append(st)
    return history, params


@pytest.mark.parametrize("release", ["gone", "merged"])
def test_blocked_ramp_causes_near_end_crawl_and_releases(release):
    m = model()
    history, params = blocked_rollout(m)
    assert history[-1]["v_mps"][0, 1].item() < 2.
    assert 55. < history[-1]["s_m"][0, 1].item() < 75.
    assert merge_end_blocked(history[-1]["veh_kin"], history[-1]["present"], m.physics)[0, 0]
    free_history, _ = blocked_rollout(model(enabled=False))
    assert free_history[-1]["v_mps"][0, 1] > 15.
    kin = history[-1]["veh_kin"].clone()
    present = history[-1]["present"].clone()
    if release == "gone":
        present[0, 0] = False
    else:
        # A merger that has entered lane 1 and pulled clear is an ordinary IDM leader.
        kin[0, 0, 0] += 35.
        kin[0, 0, 1:4] = kin.new_tensor([0., 1., 0.])
        kin[0, 0, 4] = 15.
    released = state(kin, present=present, cav=history[-1]["is_cav"])
    raw = m._physics_step(released, params)[0]
    assert torch.equal(m.physics_accel(released, params), raw)
    assert raw[0, 1] > 1.
    before = released["v_mps"][0, 1].item()
    with torch.no_grad():
        for _ in range(4):
            released, _ = advance(m, released, params)
    assert released["v_mps"][0, 1].item() > before + 2.


def test_controlled_follow_chain_develops_no_new_same_lane_overlap():
    history, _ = blocked_rollout(model(), chain=True)
    for step, st in enumerate(history):
        assert torch.isfinite(st["veh_kin"]).all(), step
        assert (st["v_mps"] >= 0.).all(), step
        assert st["lane"].tolist() == [[2, 1, 1, 1]], step
        gaps = st["s_m"][:, 1:-1] - st["length_m"][:, 1:-1] - st["s_m"][:, 2:]
        assert (gaps >= 0.).all(), (step, gaps)
    assert history[-1]["v_mps"][0, 1] < 2.


def test_synthetic_state_dict_roundtrip_preserves_learned_yield():
    source, restored = model(), model()
    with torch.no_grad():
        source.merge_yield.log_decel.fill_(math.log(3.2))
        source.merge_yield.log_distance.fill_(math.log(58.))
    result = restored.load_state_dict(source.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    kin = scene()
    st, params = state(kin), drivers(kin)
    for a, b in zip(source.merge_yield.parameters(), restored.merge_yield.parameters()):
        assert torch.equal(a, b)
    assert torch.equal(source(st, driver_params=params)["accel_mps2"],
                        restored(st, driver_params=params)["accel_mps2"])


def test_legacy_state_dict_initializes_yield_and_partial_keys_fail_closed():
    old, new = model(enabled=False), model()
    result = new.load_state_dict(old.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert new.merge_yield.decel_mps2.item() == pytest.approx(2.)
    assert new.merge_yield.distance_m.item() == pytest.approx(60.)
    partial = new.state_dict()
    del partial["merge_yield.log_distance"]
    with pytest.raises(RuntimeError, match="log_distance"):
        new.load_state_dict(partial, strict=True)


def test_checkpoint_records_runtime_enable_and_reloads_strictly(tmp_path):
    from dataclasses import asdict
    from wms.model.checkpoint import load_checkpoint, save_checkpoint
    source = model(enabled=False)
    source.enable_mainline_merge_yield_learnable()
    with torch.no_grad():
        source.merge_yield.log_decel.fill_(math.log(3.2))
    path = tmp_path / "yield.pt"
    save_checkpoint(path, source, {"rollout_mode": "driver_params",
                                   "physics": asdict(source.physics)})
    loaded = load_checkpoint(path, strict=True)
    assert loaded.legacy_config["mainline_merge_yield_learnable"] is True
    assert loaded.model.mainline_merge_yield_learnable
    assert torch.equal(source.merge_yield.log_decel, loaded.model.merge_yield.log_decel)


def test_activation_checkpoint_preserves_yield_pathwise_gradients():
    m = model()
    initial = state(scene(), cav=torch.tensor([[True, False]]))
    params = drivers(initial["veh_kin"])
    values, gradients = [], []
    for flag in (False, True):
        m.checkpoint_forward = flag
        st = initial
        for _ in range(3):
            st, _ = advance(m, st, params)
        value = st["v_mps"][0, 1] + 0.1 * st["s_m"][0, 1]
        values.append(value.detach())
        gradients.append(torch.autograd.grad(value, tuple(m.merge_yield.parameters())))
    assert torch.equal(*values)
    for a, b in zip(*gradients):
        assert torch.isfinite(a) and a.abs() > 0
        assert torch.equal(a, b)
