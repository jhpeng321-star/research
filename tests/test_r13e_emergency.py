"""Emergency braking contracts: synthetic CPU states, unchanged trapezoidal plant.

Safety claims below are restricted to initially B=9-feasible approaches. There
is no projection, lane-end pinning, future observation, or training-run fixture.
"""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from wms.cli.assemble import build_model, build_physics
from wms.config.train import TrainConfig
from wms.core.errors import ConfigError
from wms.model.checkpoint import load_checkpoint, save_checkpoint
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.merge_pivot import advance_longitudinal
from wms.physics.accel import baseline_accel
from wms.physics.config import PhysicsConfig
from wms.physics.emergency import safe_accel

from test_r10b_courtesy import sequence, state_at_step
from test_r11_courtesy import model_v

ROOT = Path(__file__).resolve().parents[1]
DT = 0.5
B = 9.0


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_default_opt_in_and_nested_legacy_config_round_trip():
    cfg = TrainConfig.from_dict({})
    assert PhysicsConfig().emergency_braking is False
    assert cfg.physics.emergency_braking is False
    assert build_physics(cfg).emergency_braking is False
    assert "emergency_braking" not in cfg.to_legacy_flat()
    for flag in (False, True):
        configured = TrainConfig.from_dict({"physics": {"emergency_braking": flag}})
        assert configured.physics.emergency_braking is flag
        assert build_physics(configured).emergency_braking is flag
        again = TrainConfig.from_legacy_flat(configured.to_legacy_flat())
        assert again.physics.emergency_braking is flag
        if flag:
            assert configured.to_legacy_flat()["emergency_braking"] is True
    # The new branch must not require the old constant-speed emergency switch.
    assert build_physics(TrainConfig.from_dict({"physics": {"emergency_braking": True}})
                         ).emergency_decel_mps2 is None


@pytest.mark.parametrize("bad", [0, 1, -1, 0.5, "true", "false", None, [], {}])
def test_emergency_flag_rejects_non_booleans(bad):
    with pytest.raises(ValueError, match="emergency_braking"):
        PhysicsConfig(emergency_braking=bad)
    with pytest.raises(ConfigError, match="emergency_braking"):
        TrainConfig.from_dict({"physics": {"emergency_braking": bad}})


@pytest.fixture
def artifact_dir():
    # Keep even checkpoint/identity test artifacts inside the workspace.
    import tempfile
    root = ROOT / "runs/r13e_emergency_tests"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="test-", dir=root))


@pytest.mark.parametrize("storage", ["embedded", "sidecar"])
@pytest.mark.parametrize("flag", [False, True])
def test_strict_checkpoint_physics_round_trip_adds_no_network_tensors(artifact_dir, storage, flag):
    cfg = TrainConfig.from_dict({"model": {"rollout_mode": "driver_params",
                                          "driver_param_mode": "physical"},
                                 "physics": {"emergency_braking": flag}})
    physics = build_physics(cfg)
    torch.manual_seed(19)
    model = build_model(cfg, physics).eval()
    torch.manual_seed(19)
    other = build_model(cfg, replace(physics, emergency_braking=not flag)).eval()
    weights = model.state_dict()
    assert weights.keys() == other.state_dict().keys()
    for key, value in weights.items():
        assert torch.equal(value, other.state_dict()[key]), key
    assert list(model.named_parameters()) and not any("emergency" in k for k in weights)
    path = artifact_dir / "synthetic.pt"
    flat = cfg.to_legacy_flat()
    block = asdict(physics)
    if storage == "embedded":
        flat["physics"] = block
    else:
        (artifact_dir / "physics.json").write_text(json.dumps(block), encoding="utf-8")
    save_checkpoint(path, model, flat)
    before = path.read_bytes()
    loaded = load_checkpoint(path, strict=True)
    assert asdict(loaded.model.physics) == block
    assert loaded.model.physics.emergency_braking is flag
    result = loaded.model.load_state_dict(weights, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    for key, value in loaded.model.state_dict().items():
        assert torch.equal(value, weights[key]), key
    assert path.read_bytes() == before
    # Pre-R13E physics blocks do not contain the new key and still load strictly.
    block.pop("emergency_braking")
    legacy = artifact_dir / "legacy.pt"
    save_checkpoint(legacy, model, {**flat, "physics": block})
    assert load_checkpoint(legacy, strict=True).model.physics.emergency_braking is False


def _safe(x):
    """x columns: speed, leader speed, gap, minimum gap, nominal acceleration."""
    return safe_accel(x[..., 0], x[..., 1], x[..., 2], x[..., 4], x[..., 3],
                      dt=DT, emergency_decel=B)


def test_finite_difference_gradients_for_all_five_continuous_inputs():
    # Active safe-speed branch, plus an unconstrained row carrying the nominal derivative.
    x = torch.tensor([[20., 15., 20., 2.5, 2.], [10., 15., 80., 2.5, -0.4]],
                     dtype=torch.float64, requires_grad=True)
    result = _safe(x)
    assert -B < result[0] < -4.5 and result[1].item() == pytest.approx(-0.4, abs=1e-6)
    assert torch.autograd.gradcheck(_safe, (x,), eps=1e-5, atol=2e-6, rtol=3e-4)
    gradient, = torch.autograd.grad(result.sum(), x)
    assert torch.isfinite(gradient).all()
    assert torch.all(gradient.abs().sum(0) > 1e-5), gradient
    assert gradient[0, 0] < 0 and gradient[0, 1] > 0
    assert gradient[0, 2] > 0 and gradient[0, 3] < 0
    assert gradient[1, 4].item() == pytest.approx(1., abs=1e-5)


def test_nominal_safe_speed_join_is_smooth_at_dtype_scaled_width():
    x = torch.tensor([20., 15., 20., 2.5, 3.], dtype=torch.float64)
    cap = _safe(x).detach()
    width = torch.finfo(x.dtype).eps ** 0.5 * B
    derivatives = []
    for fraction in (-0.25, 0., 0.25):
        nominal = (cap + fraction * width).requires_grad_()
        value = safe_accel(x[0], x[1], x[2], nominal, x[3], dt=DT, emergency_decel=B)
        grad, = torch.autograd.grad(value, nominal)
        h = width * 0.02
        plus = safe_accel(x[0], x[1], x[2], nominal.detach() + h, x[3],
                          dt=DT, emergency_decel=B)
        minus = safe_accel(x[0], x[1], x[2], nominal.detach() - h, x[3],
                           dt=DT, emergency_decel=B)
        assert grad.item() == pytest.approx(((plus - minus) / (2 * h)).item(), abs=2e-3)
        assert 0.02 < grad < 0.98, "hard min would give 0 or 1 beside the join"
        derivatives.append(grad.item())
    assert derivatives[0] > derivatives[1] > derivatives[2]


def test_learned_driver_head_has_a_finite_difference_gradient_with_emergency_active():
    model = model_v(version=0, rollout_mode="driver_params", driver_param_mode="physical")
    model.physics = replace(model.physics, emergency_braking=True)
    batch = sequence(cav=False)
    batch["states"][:, :, 1, 0] = 25.0  # 20 m bumper gap; active, unsaturated safe-speed constraint
    batch["states"][:, :, 0, 4] = 20.0
    batch["states"][:, :, 1, 4] = 15.0
    state = state_at_step(batch, 0)
    torch.manual_seed(310)
    history = torch.randn(1, 5, 6, 6) * 0.1
    mask = torch.ones(1, 5, 6, dtype=torch.bool)

    def objective():
        return model(state, driver_history=history, driver_history_mask=mask)["accel_mps2"][0, 0]

    value = objective()
    with torch.no_grad():
        model.physics = replace(model.physics, emergency_braking=False)
        off = objective()
        model.physics = replace(model.physics, emergency_braking=True)
    assert -B + 0.1 < value < off - 0.2, "must exercise an active, unsaturated emergency row"
    head = model.driver_encoder.head[-1]
    grad_bias, grad_weight = torch.autograd.grad(value, (head.bias, head.weight))
    row = PARAMETER_NAMES.index("min_gap")
    column = int(grad_weight[row].abs().argmax())
    for parameter, index, analytic in ((head.bias, (row,), grad_bias[row]),
                                        (head.weight, (row, column), grad_weight[row, column])):
        assert torch.isfinite(analytic) and analytic.abs() > 1e-5
        original = parameter[index].detach().clone()
        h = 1e-2
        try:
            with torch.no_grad():
                parameter[index] = original + h
                plus = objective().item()
                parameter[index] = original - h
                minus = objective().item()
        finally:
            with torch.no_grad():
                parameter[index] = original
        assert analytic.item() == pytest.approx((plus - minus) / (2 * h), rel=0.03, abs=2e-4)


def braking_distance(v):
    """Independent oracle: sum actual dt=.5 full-brake trapezoids until rest."""
    speed = v.clone()
    distance = torch.zeros_like(v)
    for _ in range(32):
        next_speed = (speed - B * DT).clamp_min(0)
        distance += (speed + next_speed) * (DT / 2)
        speed = next_speed
    assert torch.all(speed == 0)
    return distance


def sample_clearance(kin, moved):
    """Check the whole step, including a pessimistic ballistic-braking leader.

    The follower follows the quadratic joining its actual trapezoid endpoints.
    For the leader, take the smaller displacement of that same interpolation and
    immediate physical B-braking with exact stop time. This also checks between
    the plant's sample times when the leader stops during a tick.
    """
    t = torch.linspace(0, DT, 33, dtype=kin.dtype).reshape(1, -1)
    v, next_v = kin[..., 4], moved[..., 4]
    displacement = v[..., None] * t + (next_v - v)[..., None] * t.square() / (2 * DT)
    stop_t = torch.minimum(t, v[:, 1, None] / B)
    ballistic_lead = v[:, 1, None] * stop_t - B * stop_t.square() / 2
    leader = torch.minimum(displacement[:, 1], ballistic_lead)
    gap = kin[:, 1, 0] - kin[:, 1, 6] - kin[:, 0, 0]
    return gap[:, None] + leader - displacement[:, 0]


def approaches(dtype):
    grid = torch.cartesian_prod(torch.tensor([0., 2., 4.4, 4.6, 9., 13.7, 22., 31.]),
                                torch.tensor([0., 2., 7., 14., 25., 31.]),
                                torch.tensor([0.5, 2.5, 5.]),
                                torch.tensor([0.05, 0.5, 3., 15.])).to(dtype)
    v, lv, min_gap, slack = grid.unbind(-1)
    # Feasibility uses actual follower stopping trapezoids and the shorter,
    # conservative ballistic leader stop. It is deliberately not just TTC > 0.
    gap = min_gap + (braking_distance(v) - lv.square() / (2 * B)).clamp_min(0) + slack
    kin = torch.zeros(len(grid), 2, 7, dtype=dtype)
    kin[:, :, 2] = 1  # both in mainline lane 1
    kin[..., 6] = 5
    kin[:, 1, 0] = gap + 5
    kin[:, 0, 4], kin[:, 1, 4] = v, lv
    return kin, min_gap


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_feasible_approaches_survive_80_seconds_and_within_step_samples(dtype):
    kin, min_gap = approaches(dtype)
    initial_gap = kin[:, 1, 0] - 5 - kin[:, 0, 0]
    assert torch.all(initial_gap >= min_gap)
    worst = float("inf")
    tolerance = 3e-4 if dtype == torch.float32 else 2e-8
    with torch.no_grad():
        for step in range(160):  # 80 seconds, including sustained stationary-obstacle approaches
            gap = kin[:, 1, 0] - 5 - kin[:, 0, 0]
            nominal = torch.full_like(gap, 2.6)  # continued desire to accelerate, even at rest
            accel = safe_accel(kin[:, 0, 4], kin[:, 1, 4], gap, nominal, min_gap,
                               dt=DT, emergency_decel=B)
            assert torch.isfinite(accel).all() and torch.all(accel >= -B)
            assert torch.all(accel <= nominal + tolerance)
            commands = torch.stack((accel, torch.full_like(accel, -B)), -1)
            moved = advance_longitudinal(kin, commands, DT, stop_time_integration=False)
            clearance = sample_clearance(kin, moved)
            assert torch.all(clearance >= -tolerance), f"body collision at step {step}"
            next_gap = moved[:, 1, 0] - 5 - moved[:, 0, 0]
            worst = min(worst, float(next_gap.min()))
            assert torch.all(next_gap >= min_gap - tolerance), f"standstill gap at step {step}"
            assert torch.all(moved[..., 0] >= kin[..., 0])
            reserve = next_gap - min_gap + moved[:, 1, 4].square() / (2 * B)
            assert torch.all(braking_distance(moved[:, 0, 4]) <= reserve + tolerance), step
            kin = moved
    assert worst >= -tolerance
    assert kin[:, 1, 4].count_nonzero() == 0
    assert kin[:, 0, 4].max() < 0.02


@pytest.mark.parametrize("obstacle", ["leader", "lane_end"])
def test_real_baseline_handles_feasible_emergency_approach_without_projection(obstacle):
    physics = PhysicsConfig(emergency_braking=True, stop_time_integration=False,
                            lane_rule="forced_merge_only")
    v = torch.tensor([4.6, 9., 18., 25., 31.], dtype=torch.float64)
    lv = torch.zeros_like(v) if obstacle == "lane_end" else torch.tensor([0., 4., 8., 15., 25.])
    gap = 2.5 + (braking_distance(v) - lv.square() / (2 * B)).clamp_min(0) + 0.5
    kin = torch.zeros(len(v), 2, 7, dtype=v.dtype)
    kin[..., 6] = 5
    kin[..., 2] = 1
    kin[:, 0, 4], kin[:, 1, 4] = v, lv
    kin[:, 1, 0] = gap + 5
    present = torch.ones(len(v), 2, dtype=torch.bool)
    if obstacle == "lane_end":
        kin[:, 0, 0] = physics.merge_zone_s1_m - gap
        kin[:, 0, 1:4] = torch.tensor([0., 0., 1.])
        # Slot 1 is just an oracle marker for the obstacle, not a neighbour.
        kin[:, 1, 0] = physics.merge_zone_s1_m + 5
        present[:, 1] = False
    # Comfortable braking is deliberately much weaker than the physical B=9.
    defaults = dict(zip(PARAMETER_NAMES, (31.29, 1.1, 2.5, 2.6, 1., .2, .1, 6., 1.)))
    params = {key: torch.full_like(kin[..., 4], value) for key, value in defaults.items()}
    used_emergency = False
    for step in range(160):
        accel = baseline_accel(kin, present, physics, params)
        assert torch.isfinite(accel).all() and torch.all(accel >= -B)
        assert torch.all(accel <= 3.)
        used_emergency |= bool((accel[:, 0] < -1.01).any())
        if obstacle == "lane_end":
            assert torch.equal(accel[:, 1], torch.zeros_like(accel[:, 1]))
        commands = torch.stack((accel[:, 0], torch.full_like(v, -B)), -1)
        moved = advance_longitudinal(kin, commands, DT, stop_time_integration=False)
        assert torch.all(sample_clearance(kin, moved) >= -2e-8), step
        assert torch.all(moved[:, 1, 0] - 5 - moved[:, 0, 0] >= 2.5 - 2e-8), step
        kin = moved
    assert used_emergency
    assert kin[:, 0, 4].max() < 0.02


def test_infeasible_state_is_bounded_but_not_claimed_preventable():
    x = torch.tensor([[30., 0., .5, 2.5, 3.], [0., 0., 0., 2.5, 3.],
                      [0., 0., 2.5, 2.5, 0.]], dtype=torch.float64)
    output = _safe(x)
    assert torch.isfinite(output).all()
    assert torch.all(output >= -B) and torch.all(output <= 3.)
    assert output[0].item() == pytest.approx(-B, abs=1e-6)
    # Even immediate maximum braking travels farther than the available space.
    assert braking_distance(x[:1, 0]).item() > x[0, 2].item()
    first_step_min_distance = (30. + max(0., 30. - B * DT)) * DT / 2
    assert first_step_min_distance > .5


def test_emergency_off_is_bit_identical_to_actual_c8dd7162_forward_and_rollout():
    path = ROOT / "tools/r13e_checks/bit_identity.py"
    spec = importlib.util.spec_from_file_location("r13e_bit_identity", path)
    identity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    report = identity.run_check()
    assert report["status"] == "PASS"
    assert report["base_commit"] == "c8dd71620d866620532eded754dd467649358175"
    assert report["cases"] == 4 and report["rollout_steps"] == 6
    assert all(count > 100 for count in report["tensor_comparisons"].values())
