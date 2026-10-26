"""R16 (S-RAMPSPEED) contracts: synthetic tensors only, CPU only, no training or simulator."""
import json
import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from wms.model.ramp_speed_adapt import INIT_PATH, LearnedRampSpeedAdapt, load_init, target_lane_leader
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.physics.idm import IDMParams, idm_accel
from wms.train.inputs import state_from_kin


@pytest.fixture(autouse=True)
def cpu_only():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.device("cpu"):
        yield
    torch.set_num_threads(previous)


def physics(**kw):
    return PhysicsConfig(lane_learnable=True, lane_rule="forced_merge_only", lane_gap_rule="speed",
                         stop_time_integration=True, **kw)


def pivot_physics():
    """The 13E / R14 inference physics (first-passage pivot core)."""
    return PhysicsConfig(lane_learnable=True, lane_rule="learned_discretionary", lane_gap_rule="speed",
                         mainline_lane_mode="courtesy", stop_time_integration=True,
                         ramp_lane_end_stop=True, ramp_risk_retain=True, emergency_braking=True,
                         secure_gap_veto=True, lane_change_secure_gap=True, emergency_onset="driver_b")


def scene(positions, speeds, lanes, *, dtype=torch.float32):
    kin = torch.zeros(1, len(positions), 7, dtype=dtype)
    kin[..., 0] = kin.new_tensor(positions)
    kin[..., 1:4] = F.one_hot(torch.tensor([lanes]), 3).to(dtype)
    kin[..., 4] = kin.new_tensor(speeds)
    kin[..., 6] = 5.
    return kin


def drivers(kin, **overrides):
    values = dict(v0=31.29, time_headway=1.1, min_gap=2.5, accel=2.6, decel=4.5,
                  politeness=0.2, mobil_threshold=0.1, merge_gap=6., merge_urgency=1.)
    values.update(overrides)
    return {key: torch.full_like(kin[..., 0], value) for key, value in values.items()}


def state(kin, *, present=None, cav=None):
    present = torch.ones(kin.shape[:2], dtype=torch.bool) if present is None else present
    cav = torch.zeros_like(present) if cav is None else cav
    return {**state_from_kin(kin, present), "is_cav": cav}


def model(*, enabled=True, pivot=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1616)
        if pivot:
            spec = ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5", lane_decision="sample",
                             merge_decision="first_passage", merge_passage_law="hybrid",
                             mainline_merge_yield_learnable=True, ramp_speed_adapt_learnable=enabled)
            return WmSModel(spec, physics=pivot_physics()).eval()
        return WmSModel(ModelSpec(rollout_mode="driver_params", ramp_speed_adapt_learnable=enabled,
                                  follower_courtesy_intent="none"), physics=physics()).eval()


def scalar_fd(fn, param, h=1.e-4):
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


# --------------------------------------------------------------------------- parameters / wiring
def test_three_shared_scalars_start_at_the_train_fitted_init():
    doc = json.loads(INIT_PATH.read_text(encoding="utf-8"))
    assert "INITIALIZATION" in doc["provenance"] and len(doc["fit_evidence_sha256"]) == 64
    layer = LearnedRampSpeedAdapt()
    assert set(dict(layer.named_parameters())) == {"log_lookahead", "strength_logit", "log_headway_scale"}
    assert all(p.shape == () for p in layer.parameters())
    init = load_init()
    assert layer.lookahead_m.item() == pytest.approx(init["lookahead_m"])
    assert layer.strength.item() == pytest.approx(init["strength"])
    assert layer.headway_scale.item() == pytest.approx(init["headway_scale"])


def test_flag_off_creates_no_module_tensor_or_output():
    m = model(enabled=False)
    assert m.ramp_speed_adapt is None and not m.ramp_speed_adapt_learnable
    assert not any(k.startswith("ramp_speed_adapt.") for k in m.state_dict())
    kin = scene((40., 50.), (15., 5.), (2, 1))
    out = m(state(kin), driver_params=drivers(kin))
    assert "ramp_speed_adapt_weight" not in out


def test_enable_hook_is_idempotent_does_not_draw_rng_and_disable_restores_bitexact():
    m = model(enabled=False)
    kin = scene((40., 50., 20.), (15., 5., 5.), (2, 1, 1))
    st, d = state(kin), drivers(kin)
    before_out = m(st, driver_params=d)["accel_mps2"]
    rng = torch.random.get_rng_state().clone()
    m.enable_ramp_speed_adapt_learnable()
    layer = m.ramp_speed_adapt
    m.enable_ramp_speed_adapt_learnable()
    assert m.ramp_speed_adapt is layer and torch.equal(rng, torch.random.get_rng_state())
    on = m(st, driver_params=d)["accel_mps2"]
    assert on[0, 0] < before_out[0, 0]
    m.ramp_speed_adapt_learnable = False
    assert torch.equal(m(st, driver_params=d)["accel_mps2"], before_out)


@pytest.mark.parametrize("invalid", ["fixed_driver", "black_box", "backend"])
def test_enable_hook_rejects_incompatible_cores(invalid):
    spec = ModelSpec(rollout_mode="residual" if invalid == "fixed_driver" else "driver_params",
                     black_box=invalid == "black_box")
    kwargs = {"physics_step": lambda st: {}} if invalid == "backend" else {}
    m = WmSModel(spec, physics=physics(), **kwargs)
    with pytest.raises(ValueError, match="learned-driver structured core"):
        m.enable_ramp_speed_adapt_learnable()


def test_legacy_checkpoint_loads_with_init_and_new_checkpoint_round_trips():
    legacy = model(enabled=False).state_dict()
    m = model(enabled=True)
    result = m.load_state_dict(legacy, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert m.ramp_speed_adapt.lookahead_m.item() == pytest.approx(load_init()["lookahead_m"])
    with torch.no_grad():
        m.ramp_speed_adapt.log_lookahead.fill_(math.log(30.))
    fresh = model(enabled=True)
    fresh.load_state_dict(m.state_dict(), strict=True)
    assert fresh.ramp_speed_adapt.lookahead_m.item() == pytest.approx(30.)


def test_config_flag_reaches_spec_settings_and_checkpoint(tmp_path):
    from wms.config.legacy_flat import FLAT_TO_PATH
    from wms.model.checkpoint import load_checkpoint, save_checkpoint
    assert FLAT_TO_PATH["ramp_speed_adapt_learnable"] == "model.ramp_speed_adapt_learnable"
    m = model(enabled=True)
    path = tmp_path / "wm_s.pt"
    save_checkpoint(path, m, {"rollout_mode": "driver_params", "follower_courtesy_intent": "none"})
    loaded = load_checkpoint(path, physics=json.loads(json.dumps(physics().__dict__)))
    assert loaded.model.ramp_speed_adapt_learnable and loaded.legacy_config["ramp_speed_adapt_learnable"]


# --------------------------------------------------------------------------- selection / formula
def test_target_lane_leader_matches_pivot_convention():
    # ego ramp at 40; lane-1 cars at 40 (tie -> leader), 55, 20; lane-0 car at 41 ignored.
    kin = scene((40., 55., 40., 20., 41.), (10., 5., 6., 5., 30.), (2, 1, 1, 1, 0))
    has, gap, lead_v = target_lane_leader(kin, torch.ones(1, 5, dtype=torch.bool), physics())
    assert has[0, 0] and gap[0, 0].item() == pytest.approx(-5.) and lead_v[0, 0].item() == 6.


@pytest.mark.parametrize("case", ["cav", "upstream", "past_end", "no_lane1_leader", "absent_leader",
                                  "lane1_car", "faster_leader"])
def test_unselected_slots_are_bitexact(case):
    layer = LearnedRampSpeedAdapt()
    kin = scene((40., 52., 20.), (15., 5., 5.), (2, 1, 1))
    present = torch.ones(1, 3, dtype=torch.bool)
    learned = present.clone()
    if case == "cav":
        learned[0, 0] = False
    elif case == "upstream":
        kin[0, 0, 0] = -1.
    elif case == "past_end":
        kin[0, 0, 0] = 75.5
    elif case == "no_lane1_leader":
        kin[0, 1, 0] = 30.
    elif case == "absent_leader":
        present[0, 1] = False
    elif case == "faster_leader":
        kin[0, 1, 0], kin[0, 1, 4] = 120., 30.
    raw = torch.tensor([[0.7, 0.3, -0.2]])
    out, selected, weight = layer(kin, present, raw, physics(), drivers(kin), learned=learned)
    if case == "lane1_car":
        assert not selected[0, 1] and out[0, 1] == raw[0, 1]
        return
    if case == "faster_leader":
        assert selected[0, 0] and out[0, 0] == raw[0, 0]   # deceleration-only bound
        return
    assert not selected.any() and torch.equal(out, raw) and torch.equal(weight, torch.zeros_like(raw))


def test_formula_uses_the_egos_learned_settings():
    layer = LearnedRampSpeedAdapt().double()
    kin = scene((40., 60., 20.), (14., 6., 5.), (2, 1, 1), dtype=torch.float64)
    d = drivers(kin, time_headway=1.3, min_gap=3.0, decel=3.5)
    raw = kin.new_tensor([[0.8, 0., 0.]])
    p = physics()
    out, selected, weight = layer(kin, torch.ones(1, 3, dtype=torch.bool), raw, p, d,
                                  learned=torch.ones(1, 3, dtype=torch.bool))
    own = {k: v[:, 0] for k, v in d.items()}
    params = IDMParams.from_driver_params(own)
    params = replace(params, time_headway=params.time_headway * layer.headway_scale)
    a_v = idm_accel(kin[:, 0, 4], kin[:, 1, 4], kin.new_tensor([15.]), params,
                    accel_min=p.accel_min_mps2, accel_max=p.accel_max_mps2).clamp_min(-3.5)
    w = layer.strength * torch.sigmoid((layer.lookahead_m - (75.44 - 40.)) / 3.0)
    torch.testing.assert_close(out[0, 0], raw[0, 0] + w * torch.clamp_max(a_v[0] - raw[0, 0], 0.))
    torch.testing.assert_close(weight[0, 0], w)
    assert out[0, 0] < raw[0, 0] and selected.tolist() == [[True, False, False]]


# --------------------------------------------------------------------------- gradients (pathwise)
def _grad_scene():
    # Unsaturated IDM (gap 12 m, closing 5 m/s), d_end inside the sigmoid slope of D.
    kin = scene((11.44, 28.44, 0.), (12., 7., 7.), (2, 1, 1), dtype=torch.float64)
    return kin, drivers(kin, time_headway=1.0, min_gap=4.0, decel=6.0, accel=1.5)


def test_shared_scalar_gradients_match_finite_differences():
    layer = LearnedRampSpeedAdapt().double()
    with torch.no_grad():
        layer.log_headway_scale.fill_(0.)
    kin, d = _grad_scene()
    raw = kin.new_tensor([[0.5, 0., 0.]])
    ones = torch.ones(1, 3, dtype=torch.bool)

    def evaluate():
        out, selected, _ = layer(kin, ones, raw, physics(), d, learned=ones)
        assert selected[0, 0]
        return out[0, 0]

    params = (layer.log_lookahead, layer.strength_logit, layer.log_headway_scale)
    grads = torch.autograd.grad(evaluate(), params)
    for param, grad in zip(params, grads):
        assert torch.isfinite(grad) and abs(grad.item()) > 1.e-3, param
        assert grad.item() == pytest.approx(scalar_fd(evaluate, param, 1.e-6).item(), rel=1.e-5, abs=1.e-8)


def test_state_and_driver_setting_adjoints_match_finite_differences():
    layer = LearnedRampSpeedAdapt().double()
    with torch.no_grad():
        layer.log_headway_scale.fill_(0.)
    kin0, d0 = _grad_scene()
    kin = kin0.clone().requires_grad_(True)
    d = {k: v.clone().requires_grad_(True) for k, v in d0.items()}
    raw = kin0.new_tensor([[0.5, 0., 0.]]).requires_grad_(True)
    ones = torch.ones(1, 3, dtype=torch.bool)

    def evaluate(x, dp, r):
        return layer(x, ones, r, physics(), dp, learned=ones)[0][0, 0]

    names = ("time_headway", "min_gap", "accel", "v0")
    grads = torch.autograd.grad(evaluate(kin, d, raw), (kin, raw, *(d[n] for n in names)))
    for slot, channel in ((0, 0), (0, 4), (1, 0), (1, 4), (1, 6)):
        plus, minus = kin0.clone(), kin0.clone()
        plus[0, slot, channel] += 1.e-6
        minus[0, slot, channel] -= 1.e-6
        fd = (evaluate(plus, d0, raw.detach()) - evaluate(minus, d0, raw.detach())) / 2.e-6
        assert abs(grads[0][0, slot, channel].item()) > 1.e-4, (slot, channel)
        assert grads[0][0, slot, channel].item() == pytest.approx(fd.item(), rel=1.e-5, abs=1.e-8)
    assert 0. < grads[1][0, 0].item() < 1.          # d out / d raw = 1 - w on the binding branch
    for name, grad in zip(names, grads[2:]):
        plus = {**d0, name: d0[name] + 1.e-6}
        minus = {**d0, name: d0[name] - 1.e-6}
        fd = (evaluate(kin0, plus, raw.detach()) - evaluate(kin0, minus, raw.detach())) / 2.e-6
        assert grad[0, 0].item() == pytest.approx(fd.item(), rel=1.e-5, abs=1.e-8), name


def _held(st):
    logits = st["veh_kin"].new_full((*st["present"].shape, 3), -1.e6)
    return logits.scatter(-1, st["lane"].unsqueeze(-1), 0.)


def test_short_rollout_pathwise_gradients_match_fd_for_all_three_scalars():
    # The WM forward runs in float32 (state_kin(...).float()); FD tolerances as in R13Y.
    m = model(enabled=True)
    with torch.no_grad():
        m.ramp_speed_adapt.log_headway_scale.fill_(0.)
    kin, d = _grad_scene()
    kin, d = kin.float(), {k: v.float() for k, v in d.items()}
    cav = torch.tensor([[False, True, True]])

    def terminal():
        st = state(kin, cav=cav)
        for _ in range(3):
            out = m.step(st, driver_params=d, baseline_lane_logits=_held(st),
                         cav_teacher=m.hold_teacher(st))
            nk = torch.cat((out["next_s_m"].unsqueeze(-1), F.one_hot(out["next_lane"], 3).to(kin.dtype),
                            out["next_v_mps"].unsqueeze(-1), out["accel_mps2"].unsqueeze(-1),
                            st["length_m"].unsqueeze(-1)), -1)
            st = state(nk, present=out["next_present"], cav=cav)
            assert st["lane"][0, 0].item() == 2
        return st["v_mps"][0, 0] + 0.1 * st["s_m"][0, 0]

    layer = m.ramp_speed_adapt
    params = (layer.log_lookahead, layer.strength_logit, layer.log_headway_scale)
    grads = torch.autograd.grad(terminal(), params)
    for param, grad in zip(params, grads):
        assert torch.isfinite(grad) and abs(grad.item()) > 1.e-2
        for h in (1.e-3, 2.e-3):
            assert grad.item() == pytest.approx(scalar_fd(terminal, param, h).item(), rel=1.e-2, abs=2.e-3)


# --------------------------------------------------------------------------- synthetic ramp approach
def _approach(enabled, steps=80):
    """One ramp HDV enters the acceleration lane at 14 m/s beside a congested lane-1 platoon of CAVs
    (constant 4 m/s, 22 m spacing).  The first-passage pivot (13E veto, R14 physics) decides the merge;
    the same lane / latent draws (``lane_seed``) are used in both arms."""
    m = model(enabled=enabled, pivot=True)
    lane1 = [float(x) for x in range(-150, 200, 22)]
    kin = scene([-5.] + lane1, [14.] + [4.] * len(lane1), [2] + [1] * len(lane1))
    cav = torch.tensor([[False] + [True] * len(lane1)])
    d = drivers(kin, decel=3.0, time_headway=1.2, min_gap=2.5)
    st = state(kin, cav=cav)
    with torch.no_grad():
        res = m.rollout(st, n_steps=steps, driver_params=d, cav_controller=m.hold_teacher,
                        lane_seed=77)
    s0, v0, l0 = kin[0, 0, 0].item(), kin[0, 0, 4].item(), 2
    trace = [(s0, v0, l0)] + [(res["positions"][0, i, 0].item(), res["speeds"][0, i, 0].item(),
                               int(res["lanes"][0, i, 0])) for i in range(steps)]
    return trace


def test_synthetic_ramp_approach_adapts_speed_and_merges_while_moving():
    """At the shipped TRAIN-fitted init, 10 common-random-number seeds (the pivot latent draws use the
    global RNG): ON slows toward the lane-1 stream inside the zone, and the UNCHANGED pivot accepts it
    more often while moving, earlier and further before the lane end.  OFF mostly merges from standstill."""
    def merge(trace):
        return next(((i, s, v) for i, ((s, v, lane), nxt) in enumerate(zip(trace, trace[1:]))
                     if lane == 2 and nxt[2] == 1), None)

    def at(trace, x):   # speed when first at or past x metres into the zone
        return next(v for s, v, lane in trace if s >= x and lane == 2)
    arms = {}
    for enabled in (False, True):
        rows = []
        for seed in range(10):
            torch.manual_seed(seed)
            trace = _approach(enabled)
            rows.append((at(trace, 30.), *merge(trace)))
        arms[enabled] = rows
    off, on = arms[False], arms[True]
    assert all(a[0] < b[0] for a, b in zip(on, off))              # slower at 30 m in every seed
    moving = {k: sum(r[3] > 0.5 for r in v) for k, v in arms.items()}
    assert moving[True] > moving[False]
    mean = {k: [sum(r[j] for r in v) / len(v) for j in (1, 2)] for k, v in arms.items()}
    assert mean[True][0] < mean[False][0]                         # earlier (steps)
    assert mean[True][1] < mean[False][1]                         # further before the lane end
