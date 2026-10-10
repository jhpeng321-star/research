"""Long-horizon contracts on tiny, in-memory CPU fixtures only.

No pack reads, fit calls, optimizer updates, GPU or SUMO.  The flags-off oracle
    loads the committed model/trainer source with ``git show bf298cf:path`` directly
into memory; it never writes a reference module or reads external worktrees.
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from output.wd01.wm_n1.hdv.config import HdvConfig
from output.wd01.wm_n1.wms import cli
from output.wd01.wm_n1.wms import gate_stages
from output.wd01.wm_n1.wms.long_horizon import (
    LongHorizonSelection,
    lane_rate_loss,
    residual_mag_loss,
    soft_bound,
    validate_bounds,
    validation_horizons,
)
from output.wd01.wm_n1.wms.model import PhysicsConfig, WmSModel
from output.wd01.wm_n1.wms.train import (
    TrajectoryDataset,
    WmSTrainConfig,
    WmSTrainer,
    collate,
)


LONG_HORIZON_DEFAULTS = {
    "residual_soft_bound_mps2": None,
    "residual_soft_bound_neg_mps2": None,
    "residual_mag_weight": 0.0,
    "lane_rate_weight": 0.0,
    "lane_rate_target": 0.0123,
    "select_by_long_horizon": False,
    "val_every": 200,
}
MONITOR_KEYS = (
    "predicted_lane_change_rate",
    "raw_residual_abs_mean_mps2",
    "raw_residual_abs_p95_mps2",
)


@pytest.fixture(autouse=True)
def _tiny_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _small_model(model_cls=WmSModel, physics_cls=PhysicsConfig, **kwargs):
    return model_cls(
        physics=physics_cls(max_gap_steps=8),
        hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16, dropout=0.0),
        history_hidden=8,
        k_hidden=4,
        **kwargs,
    )


def _tiny_batch():
    """Four transitions, one CAV, two HDVs, one padding slot and an exit."""
    states = np.zeros((1, 5, 4, 7), dtype=np.float32)
    states[..., 1] = 1.0
    states[..., 6] = 5.0
    states[..., 0] = (
        np.array([0.0, 40.0, 80.0, 120.0], dtype=np.float32)[None, None]
        + np.arange(5, dtype=np.float32)[None, :, None] * 5.0
    )
    states[..., 4] = 10.0
    states[:, 2:, 1, 4] = 9.5
    present = np.ones((1, 5, 4), dtype=bool)
    present[..., 3] = False
    present[:, 3:, 2] = False
    cav = np.array([[True, False, False, False]])
    dataset = TrajectoryDataset.from_arrays(
        {"states": states, "present": present, "is_cav": cav}
    )
    batch = collate([dataset[0]])
    # An undefined target must be excluded even when the slot remains present.
    batch["loss_valid"][:, 2, 1] = False
    return batch


def _train_config(config_cls=WmSTrainConfig, **kwargs):
    return config_cls(
        device="cpu", threads=1, num_workers=0, batch_size=1,
        steps_per_epoch=1, unroll_curriculum_s=(2.0,),
        truncation_steps=kwargs.pop("truncation_steps", 4), **kwargs,
    )


# ---------------------------------------------------------------- soft bounds
def test_soft_bound_identity_values_symmetry_and_unit_slope():
    x = torch.tensor([-5.0, -1.5, -0.01, 0.0, 0.01, 1.5, 5.0], requires_grad=True)
    assert torch.equal(soft_bound(x, None), x)
    y = soft_bound(x, 1.5)
    torch.testing.assert_close(y, 1.5 * torch.tanh(x / 1.5))
    torch.testing.assert_close(y, -y.flip(0))
    assert bool((y.abs() <= 1.5).all())
    y.sum().backward()
    expected_grad = torch.cosh(x.detach().double() / 1.5).reciprocal().square().float()
    torch.testing.assert_close(x.grad, expected_grad, rtol=1e-5, atol=1e-7)
    assert x.grad[3].item() == pytest.approx(1.0)


@pytest.mark.parametrize("neg_bound", [None, 1.5, 3.0])
def test_soft_bound_saturated_float32_backward_is_nonzero(neg_bound):
    x = torch.tensor([-26.0, 26.0], dtype=torch.float32, requires_grad=True)
    y = soft_bound(x, 1.5, neg_bound)
    bounds = torch.tensor([neg_bound or 1.5, 1.5], dtype=torch.float64)
    expected = bounds * torch.tanh(x.detach().double() / bounds)
    torch.testing.assert_close(y, expected.float(), rtol=1e-6, atol=1e-7)
    y.sum().backward()
    assert x.grad.dtype == torch.float32
    assert bool(torch.isfinite(x.grad).all()) and bool((x.grad > 0).all())
    # A float32 tanh backward using its rounded output returns exactly zero.
    # Pin the true derivative so a straight-through or arbitrary floor fails.
    derivative = torch.cosh(x.detach().double() / bounds).reciprocal().square().float()
    torch.testing.assert_close(x.grad, derivative, rtol=0.05, atol=0.0)


def test_asymmetric_soft_bound_uses_each_side_and_is_continuous_at_zero():
    x = torch.tensor([-8.0, -1.0, -1e-6, 0.0, 1e-6, 1.0, 8.0], requires_grad=True)
    y = soft_bound(x, 1.5, 3.0)
    bound = torch.where(x.detach() < 0, 3.0, 1.5)
    torch.testing.assert_close(y, bound * torch.tanh(x / bound))
    assert y.min().item() >= -3.0 and y.max().item() <= 1.5
    y.sum().backward()
    torch.testing.assert_close(x.grad[2:5], torch.ones(3), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("bounds", [
    (None, None, None, None), (1.5, None, None, None),
    (1.5, 3.0, None, None), (None, None, 1.5, None),
    (None, None, 1.5, 3.0),
])
def test_validate_bounds_accepts_exclusive_valid_families(bounds):
    validate_bounds(*bounds)


@pytest.mark.parametrize("bounds", [
    (1.5, None, 1.5, None), (1.5, None, None, 3.0),
    (1.5, 3.0, 1.5, 3.0), (None, 3.0, None, None),
])
def test_validate_bounds_rejects_mixed_families_and_orphan_soft_negative(bounds):
    with pytest.raises(ValueError):
        validate_bounds(*bounds)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_validate_bounds_rejects_invalid_magnitudes_on_either_family(index, bad):
    bounds = [1.5, None, None, None] if index < 2 else [None, None, 1.5, None]
    bounds[index] = bad
    with pytest.raises(ValueError):
        validate_bounds(*bounds)


# ------------------------------------------------------- penalties and masks
def test_residual_mag_loss_is_weighted_masked_squared_mean_with_exact_gradients():
    raw = torch.tensor([[[2.0, 200.0], [-4.0, -300.0]]], requires_grad=True)
    valid = torch.tensor([[[True, False], [True, False]]])
    loss = residual_mag_loss(raw, valid, 0.5)
    assert loss.ndim == 0 and loss.item() == pytest.approx(5.0)
    loss.backward()
    torch.testing.assert_close(raw.grad, torch.tensor([[[1.0, 0.0], [-2.0, 0.0]]]),
                               rtol=0.0, atol=0.0)


@pytest.mark.parametrize("target", [0.0123, 0.1])
def test_lane_rate_loss_uses_probabilities_masks_weight_and_target(target):
    probability = torch.tensor([[[0.2, 0.99], [0.2, 0.8]]], requires_grad=True)
    valid = torch.tensor([[[True, False], [True, False]]])
    loss = lane_rate_loss(probability, valid, 0.4, target=target)
    assert loss.item() == pytest.approx(0.4 * (0.2 - target) ** 2, rel=1e-6)
    loss.backward()
    expected = torch.tensor([[[0.4 * (0.2 - target), 0.0],
                              [0.4 * (0.2 - target), 0.0]]])
    torch.testing.assert_close(probability.grad, expected, rtol=1e-6, atol=1e-8)


def test_lane_rate_default_target_matches_explicit_target():
    p = torch.tensor([[[0.2, 0.2]]], requires_grad=True)
    valid = torch.ones_like(p, dtype=torch.bool)
    torch.testing.assert_close(lane_rate_loss(p, valid, 0.4),
                               lane_rate_loss(p, valid, 0.4, target=0.0123),
                               rtol=0.0, atol=0.0)


@pytest.mark.parametrize("helper", [residual_mag_loss, lane_rate_loss])
@pytest.mark.parametrize("shape", [(1, 2, 3), (1, 0, 3)])
def test_empty_penalties_are_connected_scalar_zero_with_zero_gradients(helper, shape):
    x = torch.full(shape, 0.2, requires_grad=True)
    loss = helper(x, torch.zeros(shape, dtype=torch.bool), 0.7)
    assert loss.ndim == 0 and loss.device.type == "cpu" and loss.item() == 0.0
    loss.backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.zeros_like(x), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("helper", [residual_mag_loss, lane_rate_loss])
def test_zero_weight_penalties_have_no_gradient_effect(helper):
    x = torch.full((1, 2, 3), 0.2, requires_grad=True)
    loss = helper(x, torch.ones_like(x, dtype=torch.bool), 0.0)
    assert loss.item() == 0.0
    loss.backward()
    assert x.grad is not None and not bool(x.grad.any())


# ----------------------------------------------------------- config and CLI
def test_config_defaults_are_off_omitted_and_legacy_receipt_round_trips():
    cfg = WmSTrainConfig()
    for key, value in LONG_HORIZON_DEFAULTS.items():
        assert getattr(cfg, key) == value
        assert key not in cfg.to_dict()
    assert WmSTrainConfig.from_dict(cfg.to_dict()) == cfg


def test_nondefault_config_fields_serialize_and_round_trip():
    overrides = {
        "residual_soft_bound_mps2": 1.5,
        "residual_soft_bound_neg_mps2": 3.0,
        "residual_mag_weight": 0.2,
        "lane_rate_weight": 0.3,
        "lane_rate_target": 0.02,
        "select_by_long_horizon": True,
        "val_every": 17,
    }
    cfg = WmSTrainConfig(**overrides)
    serialized = cfg.to_dict()
    assert {key: serialized[key] for key in overrides} == overrides
    assert WmSTrainConfig.from_dict(serialized) == cfg


@pytest.mark.parametrize("kwargs", [
    {"residual_soft_bound_neg_mps2": 3.0},
    {"residual_soft_bound_mps2": 1.5, "residual_cap_mps2": 1.5},
    {"residual_soft_bound_mps2": 1.5, "residual_cap_neg_mps2": 3.0},
    {"residual_mag_weight": -0.1}, {"lane_rate_weight": -0.1},
    {"lane_rate_target": -0.1}, {"lane_rate_target": 1.1},
    {"val_every": 0}, {"val_every": -1},
])
def test_config_rejects_invalid_combinations_weights_targets_and_cadence(kwargs):
    with pytest.raises(ValueError):
        WmSTrainConfig(**kwargs)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("key", [
    "residual_soft_bound_mps2", "residual_soft_bound_neg_mps2",
    "residual_mag_weight", "lane_rate_weight", "lane_rate_target",
])
def test_config_rejects_nonfinite_new_values(key, bad):
    kwargs = {key: bad}
    if key == "residual_soft_bound_neg_mps2":
        kwargs["residual_soft_bound_mps2"] = 1.5
    with pytest.raises(ValueError):
        WmSTrainConfig(**kwargs)


@pytest.mark.parametrize("key", ["residual_soft_bound_mps2", "residual_soft_bound_neg_mps2"])
@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_config_rejects_nonpositive_soft_bounds(key, bad):
    kwargs = {key: bad}
    if key == "residual_soft_bound_neg_mps2":
        kwargs["residual_soft_bound_mps2"] = 1.5
    with pytest.raises(ValueError):
        WmSTrainConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"residual_soft_bound_mps2": 0.0},
    {"residual_soft_bound_mps2": math.nan},
    {"residual_soft_bound_mps2": math.inf},
    {"residual_soft_bound_neg_mps2": 3.0},
    {"residual_soft_bound_mps2": 1.5, "residual_soft_bound_neg_mps2": -1.0},
    {"residual_soft_bound_mps2": 1.5, "residual_cap_mps2": 1.5},
    {"residual_soft_bound_mps2": 1.5, "residual_cap_neg_mps2": 3.0},
])
def test_model_constructor_rejects_invalid_soft_bounds_and_mixed_caps(kwargs):
    with pytest.raises(ValueError):
        _small_model(**kwargs)


@pytest.mark.parametrize("command", ["train", "dry-run"])
def test_cli_parser_defaults_match_config_without_launch(command):
    args = cli._parser().parse_args([command, "--pack", "synthetic-unused"])
    for key, value in LONG_HORIZON_DEFAULTS.items():
        assert getattr(args, key) == value
    assert args.device == "cpu"


@pytest.mark.parametrize("command", ["train", "dry-run"])
def test_cli_parser_and_actual_config_keyword_plumbing_without_launch(command):
    args = cli._parser().parse_args([
        command, "--pack", "synthetic-unused", "--threads", "1",
        "--residual-soft-bound-mps2", "1.5", "--residual-soft-bound-neg-mps2", "3",
        "--residual-mag-weight", "0.2", "--lane-rate-weight", "0.3",
        "--lane-rate-target", "0.02", "--select-by-long-horizon", "--val-every", "17",
    ])
    # Evaluate only the seven actual keyword expressions from cli.main.  This
    # checks its wiring, including swapped names, without calling main, loading
    # a dataset, building a large model, or invoking a training entry point.
    tree = ast.parse(textwrap.dedent(inspect.getsource(cli.main)))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "WmSTrainConfig"]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    overrides = {}
    for key in LONG_HORIZON_DEFAULTS:
        assert key in keywords, f"CLI does not pass {key} to the trainer config"
        expression = ast.fix_missing_locations(ast.Expression(keywords[key]))
        overrides[key] = eval(compile(expression, "<cli-config-keyword>", "eval"),
                              {"args": args, "float": float, "int": int, "bool": bool})
    expected = {
        "residual_soft_bound_mps2": 1.5, "residual_soft_bound_neg_mps2": 3.0,
        "residual_mag_weight": 0.2, "lane_rate_weight": 0.3, "lane_rate_target": 0.02,
        "select_by_long_horizon": True, "val_every": 17,
    }
    assert overrides == expected
    cfg = WmSTrainConfig(**overrides)
    assert {key: getattr(cfg, key) for key in expected} == expected


# --------------------------------------------------------- horizon selection
def _record(horizon=60.0, score=10.0, stage=60.0, step=1, valid=1):
    return {"horizon_s": horizon, "stage_s": stage, "mean_position": score,
            "step": step, "valid_targets": valid}


def test_selection_starts_unselected_and_prefers_longer_then_lower_score():
    selection = LongHorizonSelection()
    assert selection.horizon_s == 0 and selection.score == math.inf and selection.best is None
    first = _record()
    selection.consider(first)
    assert selection.best == first and selection.horizon_s == 60 and selection.score == 10
    longer = _record(horizon=120, score=100, stage=120, step=2)
    selection.consider(longer)
    assert selection.best == longer and selection.horizon_s == 120 and selection.score == 100
    # A much better 60 s score cannot displace a selected 120 s checkpoint.
    selection.consider(_record(horizon=60, score=0, stage=120, step=3))
    assert selection.best == longer
    lower = _record(horizon=120, score=90, stage=120, step=4)
    selection.consider(lower)
    assert selection.best == lower and selection.score == 90
    selection.consider(_record(horizon=120, score=90, stage=120, step=5))
    selection.consider(_record(horizon=120, score=91, stage=120, step=6))
    assert selection.best == lower  # ties preserve the first eligible checkpoint


@pytest.mark.parametrize("record", [
    _record(horizon=59.5), _record(stage=59.5), _record(valid=0), _record(valid=-1),
    _record(score=math.nan), _record(score=math.inf), _record(score=-math.inf),
    _record(horizon=math.nan), _record(horizon=math.inf),
    _record(stage=math.nan), _record(stage=math.inf),
])
def test_ineligible_selection_records_cannot_initialize_or_replace(record):
    empty = LongHorizonSelection()
    empty.consider(record)
    assert empty.best is None and empty.horizon_s == 0 and empty.score == math.inf
    existing = LongHorizonSelection()
    accepted = _record()
    existing.consider(accepted)
    existing.consider(record)
    assert existing.best == accepted and existing.horizon_s == 60 and existing.score == 10


@pytest.mark.parametrize("stage,expected", [
    (10, (10,)), (30, (30,)), (59.5, (59.5,)), (60, (60,)),
    (60.5, (60, 60.5)), (120, (60, 120)), (300, (60, 300)),
])
def test_validation_horizons_keep_stage_and_deduplicate_sixty(stage, expected):
    assert validation_horizons(stage) == expected


# ------------------------------------------------------- model/rollout hooks
def test_model_exposes_prebound_raw_residual_and_trainer_applies_config_bounds():
    torch.manual_seed(7)
    plain = _small_model().eval()
    bounded = _small_model().eval()
    bounded.load_state_dict(plain.state_dict())
    trainer = WmSTrainer(bounded, _train_config(
        residual_soft_bound_mps2=1.5, residual_soft_bound_neg_mps2=3.0,
    ))
    assert trainer.model.residual_soft_bound_mps2 == 1.5
    assert trainer.model.residual_soft_bound_neg_mps2 == 3.0
    batch = _tiny_batch()
    kin = batch["states"][:, 0]
    current = {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
               "a_mps2": kin[..., 5], "lane": kin[..., 1:4].argmax(-1),
               "length_m": kin[..., 6], "present": batch["present"][:, 0],
               "is_cav": batch["is_cav"][:, 0]}
    a = plain(current, k_steps=4)
    b = bounded(current, k_steps=4)
    assert b["raw_residual_mps2"].shape == (1, 4)
    torch.testing.assert_close(b["raw_residual_mps2"], a["raw_residual_mps2"],
                               rtol=0.0, atol=0.0)
    learned = b["hdv_mask"]
    assert bool(b["raw_residual_mps2"][learned].abs().sum() > 0)
    expected = torch.where(
        learned,
        b["physics_accel_mps2"] + soft_bound(b["raw_residual_mps2"], 1.5, 3.0),
        b["physics_accel_mps2"],
    ).clamp(bounded.physics.accel_min_mps2, bounded.physics.accel_max_mps2)
    torch.testing.assert_close(b["accel_mps2"], expected)


@pytest.mark.parametrize("history_head", [False, True])
def test_rollout_lane_change_probability_is_relative_to_predicted_pre_step_lane(
    monkeypatch, history_head,
):
    model = _small_model(**({"rollout_mode": "hybrid", "lane_head": "history"}
                            if history_head else {}))
    kin = torch.zeros(1, 4, 7)
    initial_lane = torch.tensor([[1, 1, 1, 1]] if history_head else [[0, 1, 2, 0]])
    kin[..., 1:4] = torch.nn.functional.one_hot(initial_lane, 3).float()
    kin[..., 0] = torch.arange(4) * 40.0
    kin[..., 4] = 10.0
    kin[..., 6] = 5.0
    current = {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
               "a_mps2": kin[..., 5], "length_m": kin[..., 6], "lane": initial_lane,
               "present": torch.ones(1, 4, dtype=torch.bool),
               "is_cav": torch.zeros(1, 4, dtype=torch.bool)}
    expected = []
    raw_steps = []

    def scripted_forward(state, **kwargs):
        index = len(expected)
        probabilities = torch.tensor([0.1, 0.7, 0.2] if history_head and index == 0
                                     else [0.8, 0.1, 0.1] if history_head
                                     else [0.1, 0.2, 0.7]).view(1, 1, 3).expand(1, 4, 3)
        if history_head:
            expected.append(1.0 - probabilities[..., 0])
        else:
            expected.append(1.0 - probabilities.gather(
                -1, state["lane"].unsqueeze(-1)).squeeze(-1))
        raw = torch.full((1, 4), float(index + 1), requires_grad=True)
        raw_steps.append(raw)
        zero = torch.zeros(1, 4)
        out = {"accel_mps2": zero, "physics_accel_mps2": zero,
               "accel_delta_mps2": zero, "raw_residual_mps2": raw,
               "lane_logits": probabilities.log(), "hdv_mask": state["present"]}
        if history_head:
            out.update(lane_probabilities=probabilities,
                       driver_parameters=torch.ones(1, 4, 9),
                       driver_embedding=torch.zeros(1, 4, model.driver_embed_dim))
        return out

    monkeypatch.setattr(model, "forward", scripted_forward)
    out = model.rollout(current, n_steps=2)
    assert out["raw_residual_mps2"].shape == (1, 2, 4)
    assert out["lane_change_probabilities"].shape == (1, 2, 4)
    torch.testing.assert_close(out["raw_residual_mps2"], torch.stack(raw_steps, 1))
    torch.testing.assert_close(out["lane_change_probabilities"], torch.stack(expected, 1))
    # The absolute head changes lanes on its first step.  Its second rate must
    # use the rolled lane (2), rather than repeatedly using the lane at t0 or
    # the just-selected next lane for the first-step probability.
    if not history_head:
        torch.testing.assert_close(out["lane_change_probabilities"][0, 0],
                                   torch.tensor([0.9, 0.8, 0.3, 0.9]))
        torch.testing.assert_close(out["lane_change_probabilities"][0, 1],
                                   torch.full((4,), 0.3))
    out["raw_residual_mps2"].sum().backward()
    assert all(raw.grad is not None and bool((raw.grad == 1).all()) for raw in raw_steps)


# ----------------------------------------------------------- HEAD parity oracle
@pytest.fixture(scope="module")
def _head_legacy_sources():
    root = Path(__file__).resolve().parents[5]
    package = "output.wd01.wm_n1.wms"
    modules = []
    try:
        for filename in ("model", "train"):
            source_path = f"output/wd01/wm_n1/wms/{filename}.py"
            source = subprocess.run(
                ["git", "show", f"bf298cf:{source_path}"], cwd=root,
                check=True, capture_output=True, text=True, encoding="utf-8",
            ).stdout
            name = f"{package}._long_horizon_head_{filename}"
            module = types.ModuleType(name)
            module.__file__ = str(root / source_path)
            module.__package__ = package
            sys.modules[name] = module  # dataclasses needs its module in sys.modules
            modules.append(module)
            exec(compile(source, f"<git HEAD:{source_path}>", "exec"), module.__dict__)
        yield modules[0], modules[1]
    finally:
        for module in modules:
            sys.modules.pop(module.__name__, None)


@pytest.mark.parametrize("truncation_steps", [2, 4])
def test_flags_off_trainer_matches_committed_loss_and_every_parameter_gradient(
    _head_legacy_sources, truncation_steps,
):
    legacy_model, legacy_train = _head_legacy_sources
    torch.manual_seed(19)
    current_model = _small_model()
    before_model = _small_model(legacy_model.WmSModel, legacy_model.PhysicsConfig)
    before_model.load_state_dict(current_model.state_dict(), strict=True)
    current_cfg = _train_config(truncation_steps=truncation_steps)
    before_cfg = _train_config(legacy_train.WmSTrainConfig, truncation_steps=truncation_steps)
    assert current_cfg.to_dict() == before_cfg.to_dict()
    current = WmSTrainer(current_model, current_cfg)
    before = legacy_train.WmSTrainer(before_model, before_cfg)
    batch = _tiny_batch()
    parameter_snapshot = copy.deepcopy(current_model.state_dict())
    a = current._forward_backward(batch, unroll_steps=4)
    b = before._forward_backward(batch, unroll_steps=4)
    assert a["loss"] == b["loss"] and a["valid_targets"] == b["valid_targets"] > 0
    for key, value in b.items():
        assert a[key] == value, key
    assert "residual_mag" not in a and "lane_rate" not in a
    current_parameters = dict(current_model.named_parameters())
    before_parameters = dict(before_model.named_parameters())
    assert current_parameters.keys() == before_parameters.keys()
    nonzero = False
    for name, p in current_parameters.items():
        q = before_parameters[name]
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0.0, atol=0.0, msg=name)
            assert bool(torch.isfinite(p.grad).all()), name
            nonzero |= bool(p.grad.abs().sum() > 0)
    assert nonzero  # excludes vacuous parity on a loss with no learned gradient
    for name, value in current_model.state_dict().items():
        torch.testing.assert_close(value, parameter_snapshot[name], rtol=0.0, atol=0.0)
    assert current.optimizer.state == {} and before.optimizer.state == {}


# ------------------------------------------------------ trainer integration
def test_enabled_penalties_log_masked_monitors_and_add_to_existing_loss(monkeypatch):
    torch.manual_seed(23)
    model = _small_model()
    trainer = WmSTrainer(model, _train_config(residual_mag_weight=0.4, lane_rate_weight=0.7))
    rolls = []
    original_rollout = model.rollout

    def capture_rollout(*args, **kwargs):
        result = original_rollout(*args, **kwargs)
        rolls.append(result)
        return result

    monkeypatch.setattr(model, "rollout", capture_rollout)
    batch = _tiny_batch()
    snapshot = copy.deepcopy(model.state_dict())
    record = trainer._forward_backward(batch, unroll_steps=4)
    assert len(rolls) == 2  # deterministic no-grad rate pass, then chunk BPTT
    valid = batch["loss_valid"][:, 1:] & ~batch["is_cav"][:, 1:]
    assert int(valid.sum()) == record["valid_targets"] == 5
    raw = rolls[0]["raw_residual_mps2"]
    probabilities = rolls[0]["lane_change_probabilities"]
    assert raw.shape == probabilities.shape == valid.shape == (1, 4, 4)
    assert record["residual_mag"] == pytest.approx(
        residual_mag_loss(raw.detach(), valid, 0.4).item(), rel=1e-6, abs=1e-9,
    )
    assert record["lane_rate"] == pytest.approx(
        lane_rate_loss(probabilities.detach(), valid, 0.7).item(), rel=1e-6, abs=1e-9,
    )
    selected = raw.detach()[valid].abs()
    assert record["predicted_lane_change_rate"] == pytest.approx(
        probabilities.detach()[valid].mean().item(), rel=1e-6,
    )
    assert record["raw_residual_abs_mean_mps2"] == pytest.approx(selected.mean().item(), rel=1e-6)
    assert record["raw_residual_abs_p95_mps2"] == pytest.approx(
        torch.quantile(selected, 0.95).item(), rel=1e-6,
    )
    expected_loss = sum(record[key] for key in ("position", "speed", "lane", "residual_mag", "lane_rate"))
    assert record["loss"] == pytest.approx(expected_loss, rel=1e-6)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    assert any(bool(g.abs().sum() > 0) for g in gradients)
    assert all(math.isfinite(record[key]) for key in MONITOR_KEYS)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, snapshot[name], rtol=0.0, atol=0.0)
    assert trainer.optimizer.state == {}


@pytest.mark.parametrize("capture_backward", [False, True])
def test_penalties_have_nonzero_gradient_norm_shares_without_optimizer_update(capture_backward):
    torch.manual_seed(29)
    model = _small_model()
    trainer = WmSTrainer(model, _train_config(
        residual_mag_weight=0.4, lane_rate_weight=0.7, loss_share_logging=True,
    ))
    if capture_backward:
        trainer._capture_loss_shares = True
        record = trainer._forward_backward(_tiny_batch(), unroll_steps=4)
        for name in ("residual_mag", "lane_rate"):
            assert math.isfinite(record["loss_grad_norm"][name])
            assert record["loss_grad_norm"][name] > 0.0
    else:
        record = trainer._forward_backward(
            _tiny_batch(), unroll_steps=4, backward=False, share=True,
        )
        assert all(p.grad is None for p in model.parameters())
    for name in ("residual_mag", "lane_rate"):
        assert 0.0 < record["loss_share"][name] <= 1.0
    assert sum(record["loss_share"].values()) == pytest.approx(1.0, abs=1e-6)
    assert trainer.optimizer.state == {}


def test_trainer_empty_supervision_has_zero_penalties_monitors_and_finite_gradients():
    trainer = WmSTrainer(_small_model(), _train_config(
        residual_mag_weight=0.4, lane_rate_weight=0.7,
    ))
    batch = _tiny_batch()
    batch["loss_valid"].zero_()
    record = trainer._forward_backward(batch, unroll_steps=4)
    assert record["valid_targets"] == 0 and record["loss"] == 0.0
    for key in ("residual_mag", "lane_rate") + MONITOR_KEYS:
        assert record[key] == 0.0, key
    for parameter in trainer.model.parameters():
        if parameter.grad is not None:
            assert bool(torch.isfinite(parameter.grad).all())
            assert not bool(parameter.grad.any())
    assert trainer.optimizer.state == {}


@pytest.mark.parametrize("chunk", [1, 2, 4])
def test_rate_penalty_is_one_batch_square_and_gradient_across_bptt_chunks(monkeypatch, chunk):
    model = _small_model()
    model.rate_logit = torch.nn.Parameter(torch.tensor(-1.0))
    cfg = _train_config(truncation_steps=chunk, lane_rate_weight=0.7)
    cfg.position_weight = cfg.speed_weight = cfg.lane_weight = 0.0
    trainer = WmSTrainer(model, cfg)
    original = model.rollout

    def rollout(*args, **kwargs):
        out = original(*args, **kwargs)
        t = torch.arange(kwargs["k0"], kwargs["k0"] + kwargs["n_steps"]).view(1, -1, 1)
        out["lane_change_probabilities"] = torch.sigmoid(model.rate_logit + t).expand_as(out["speeds"])
        return out

    monkeypatch.setattr(model, "rollout", rollout)
    batch = _tiny_batch()
    record = trainer._forward_backward(batch, unroll_steps=4)
    reference = torch.tensor(-1.0, requires_grad=True)
    p = torch.sigmoid(reference + torch.arange(4).view(1, -1, 1)).expand(1, 4, 4)
    valid = batch["loss_valid"][:, 1:] & ~batch["is_cav"][:, 1:]
    expected = lane_rate_loss(p, valid, .7)
    expected.backward()
    assert record["lane_rate"] == pytest.approx(expected.item(), rel=1e-6)
    torch.testing.assert_close(model.rate_logit.grad, reference.grad, rtol=1e-6, atol=1e-7)


def test_mocked_fit_periodic_selection_saves_longest_best_and_final_separately(tmp_path, monkeypatch):
    """Exercise fit orchestration without optimizer updates or real rollouts."""
    cfg = _train_config(select_by_long_horizon=True, val_every=2, out_dir=str(tmp_path / "select"))
    cfg.unroll_curriculum_s = (30., 60., 120.)
    cfg.stage_steps = (2, 3, 3)
    trainer = WmSTrainer(_small_model(), cfg)
    counter = []
    evaluations = []

    def fake_train(_batch, *, unroll_steps):
        counter.append(unroll_steps)
        with torch.no_grad():
            trainer.model.gate_increments.fill_(len(counter))
        return {"loss": 1., "finite": True}

    def fake_evaluate(_dataset, *, unroll_steps, require_full_horizon=False):
        assert require_full_horizon
        step = len(counter)
        horizon = unroll_steps * .5
        evaluations.append((step, horizon))
        score = (100 if step == 6 else 110) if horizon == 120 else 10 / step
        return {"n_batches": 1, "valid_targets": 1, "mean_position": score,
                "reached_steps": unroll_steps, "unroll_steps": unroll_steps}

    monkeypatch.setattr(trainer, "train_batch", fake_train)
    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)
    dataset = TrajectoryDataset.from_arrays({"states": np.zeros((1, 2, 1, 7), np.float32)})
    receipt = trainer.fit(dataset, val_dataset=dataset)
    assert evaluations == [(2, 30), (4, 60), (5, 60), (6, 60), (6, 120), (8, 60), (8, 120)]
    best = torch.load(tmp_path / "select" / "wm_s_best.pt", weights_only=False)
    final = torch.load(tmp_path / "select" / "wm_s.pt", weights_only=False)
    assert best["step"] == 6 and best["val"]["horizon_s"] == 120
    assert bool((best["model"]["gate_increments"] == 6).all())
    assert bool((final["model"]["gate_increments"] == 8).all())
    selection = receipt["long_horizon_selection"]
    assert selection["best"]["step"] == 6 and selection["longest_horizon_s"] == 120
    assert selection["final_checkpoint_policy"] == "last-optimizer-step"
    saved = json.loads((tmp_path / "select" / "TRAIN_RECEIPT.json").read_text())
    assert saved["long_horizon_selection"] == selection
    assert trainer.optimizer.state == {}


def test_selection_requires_val_and_strict_evaluation_never_labels_short_data_sixty(tmp_path):
    trainer = WmSTrainer(_small_model(), _train_config(
        select_by_long_horizon=True, out_dir=str(tmp_path / "unused")))
    batch = _tiny_batch()
    arrays = {key: value.numpy() for key, value in batch.items()
              if torch.is_tensor(value) and key in ("states", "present", "is_cav")}
    dataset = TrajectoryDataset.from_arrays(arrays)
    with pytest.raises(ValueError, match="validation dataset"):
        trainer.fit(dataset)
    assert not (tmp_path / "unused").exists()
    row = trainer.evaluate(dataset, unroll_steps=120, require_full_horizon=True)
    assert row["n_batches"] == 0 and row["available_steps"] == 4
    assert row["reached_steps"] is None
    assert not LongHorizonSelection().consider({**row, "stage_s": 60, "horizon_s": 60})


@pytest.mark.parametrize("explicit", [False, True])
def test_smoke_gate_requires_new_terms_on_every_step(tmp_path, explicit):
    path = tmp_path / "train_log.jsonl"
    row = {"stage_s": 60., "step": 1, "finite": True, "gate_monotone": True,
           "gate_g0": 0., "position": 1., "residual_mag": .2, "lane_rate": .01}
    path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "step": 2}) + "\n")
    if not explicit:
        (tmp_path / "train_config.json").write_text(json.dumps(
            {"residual_mag_weight": .2, "lane_rate_weight": .3}))
    terms = ("position", "residual_mag", "lane_rate") if explicit else ("position",)
    result = gate_stages.evaluate(path, stages=(60.,), terms=terms)
    assert result["passed"] and set(result["terms"]) == {"position", "residual_mag", "lane_rate"}
    del row["lane_rate"]
    path.write_text(json.dumps(row) + "\n")
    result = gate_stages.evaluate(path, stages=(60.,), terms=terms)
    assert not result["passed"] and any("lane_rate" in f for f in result["failures"])


def test_checkpoint_replay_restores_soft_bounds_from_config(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from wms_eval.wms_plugin import WmsCheckpointModel

    # Use the public checkpoint architecture, since its loader restores that
    # architecture rather than the deliberately smaller helper used above.
    model = WmSModel()
    path = tmp_path / "wm_s_best.pt"
    torch.save({"format": "wm_s/1", "model": model.state_dict(),
                "config": {"residual_soft_bound_mps2": 1.5,
                           "residual_soft_bound_neg_mps2": 3.0}}, path)
    loaded = WmsCheckpointModel(path)._load()
    assert loaded.residual_soft_bound_mps2 == 1.5
    assert loaded.residual_soft_bound_neg_mps2 == 3.0
    assert loaded.residual_cap_mps2 is None and loaded.residual_cap_neg_mps2 is None
