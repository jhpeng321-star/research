"""SECUREGAP merge core: geometry, latent persistence, gradients and checkpoint loading."""
from pathlib import Path

import pytest
import torch

from wms.model import ModelSpec, WmSModel
from wms.model.checkpoint import load_checkpoint
from wms.model.merge_core import (MergeCoreNet, load_init, secure_gap_margin,
                                  update_core_alpha)
from wms.physics.config import PhysicsConfig
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.merge_core import MergeCoreConfig, MergeCoreTerm


def _params(core: MergeCoreNet):
    p = core.params()
    return p["s0"], p["T"], p["b"], p["tau_a"]


def _scene(v, v_l, v_f, gap_l, gap_f, has_l=True, has_f=True, alpha=1.0, residual=0.0):
    one = torch.ones(1)
    return secure_gap_margin(
        torch.tensor([gap_l], dtype=torch.float32), torch.tensor([gap_f], dtype=torch.float32),
        torch.tensor([v], dtype=torch.float32), torch.tensor([v_l], dtype=torch.float32),
        torch.tensor([v_f], dtype=torch.float32), one * 0.0, one * 0.0, one * 0.0,
        torch.tensor([has_l]), torch.tensor([has_f]), *[x.view(()) for x in _params(
            MergeCoreNet())], torch.tensor([alpha]), torch.tensor([residual]))


def test_margins_closing_follower_rejects_large_gap():
    margin = _scene(10.0, 10.0, 15.0, 100.0, 30.0)
    assert margin.item() < 0.0
    assert torch.sigmoid(margin / 0.75).item() < 0.5


def test_margins_matched_speed_small_gap_accepted():
    margin = _scene(10.0, 10.0, 10.0, 20.0, 20.0)
    assert margin.item() > 0.0
    assert torch.sigmoid(margin / 0.75).item() > 0.5


def test_margins_absent_side_is_free_space():
    margin = _scene(10.0, 10.0, 10.0, 20.0, 20.0, has_f=False)
    assert margin.item() > 0.0


def test_alpha_persists_and_resets_on_slot_reuse():
    location = torch.tensor([[0.2, 0.0]])
    sigma = torch.full_like(location, 0.3)
    ramp = torch.tensor([[True, True]])
    present = torch.tensor([[True, True]])
    draw = torch.tensor([[1.0, 0.0]])
    first = update_core_alpha(location, sigma, ramp, present, normal=draw)
    alpha0 = first["alpha"][0, 0].item()
    assert first["has_alpha"].all()
    assert alpha0 == pytest.approx(torch.exp(torch.tensor(0.5)).item(), rel=1e-6)
    second = update_core_alpha(location, sigma, ramp, present,
                               alpha=first["alpha"], has_alpha=first["has_alpha"],
                               prev_present=first["prev_present"],
                               normal=torch.zeros_like(draw))
    assert second["alpha"][0, 0].item() == pytest.approx(alpha0)
    gone = update_core_alpha(location, sigma, ramp, torch.tensor([[False, True]]),
                             alpha=second["alpha"], has_alpha=second["has_alpha"],
                             prev_present=second["prev_present"], normal=torch.zeros_like(draw))
    assert not gone["has_alpha"][0, 0]
    back = update_core_alpha(location, sigma, ramp, present,
                             alpha=gone["alpha"], has_alpha=gone["has_alpha"],
                             prev_present=gone["prev_present"], normal=torch.zeros_like(draw))
    assert back["has_alpha"][0, 0]
    assert back["alpha"][0, 0].item() == pytest.approx(torch.exp(torch.tensor(0.2)).item(),
                                                      rel=1e-6)


def test_core_gradients_are_finite():
    torch.manual_seed(0)
    core = MergeCoreNet()
    params = core.params()
    gap_l = torch.tensor([20.0, 40.0])
    gap_f = torch.tensor([20.0, 5.0])
    zero = torch.zeros(2)
    ten = torch.full((2,), 10.0)
    margin = secure_gap_margin(gap_l, gap_f, ten, ten, ten, zero, zero, zero,
                               torch.tensor([True, True]), torch.tensor([True, True]),
                               params["s0"], params["T"], params["b"], params["tau_a"],
                               torch.tensor([1.1, 0.9]),
                               core.margin_residual(gap_l, gap_f, ten, ten, ten, gap_l,
                                                    gap_f, None))
    p = torch.sigmoid(margin / params["temperature"])
    yield_a = core.yield_accel(gap_f, torch.tensor([0.4, 0.6]), ten, ten, ten, gap_f,
                               params["temperature"])
    (p.sum() + margin.sum() + yield_a.sum()).backward()
    for name, parameter in core.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def _targets_and_pred():
    b, k, n = 1, 3, 4
    states = torch.zeros(b, n + 1, k, 7)
    for t in range(n + 1):
        states[0, t, 0, 0] = 20.0 + 5.0 * t
        states[0, t, 1, 0] = 40.0 + 5.0 * t
        states[0, t, 2, 0] = 5.0 + 5.0 * t
        for j in range(k):
            states[0, t, j, 4] = 8.0
            states[0, t, j, 6] = 5.0
            lane = 2 if (j == 0 and t <= 2) else 1
            states[0, t, j, 1 + lane] = 1.0
    weight = torch.ones(1, n, 1)
    targets = UnrollTargets(
        states=states, n_steps=n, weights=weight, hdv_mask=torch.ones(b, n, k),
        aux_mask=torch.ones(b, n, k), aux_count=b * n * k, denom=torch.tensor(1.0),
        lane_target=torch.zeros(b, n + 1, k), loss_valid=torch.ones(b, n + 1, k),
        is_cav=torch.zeros(b, n + 1, k), dt_s=0.5, position_delta_m=5.0,
        speed_delta_mps=1.0, history_lane_head=False)
    init = load_init()
    m = n - 1
    alpha_loc = torch.zeros(b, m, k, requires_grad=True)
    zeros = torch.zeros(b, m, k)
    pred = {
        "lanes": torch.tensor([[[2, 1, 1], [2, 1, 1], [1, 1, 1]]]),
        "masks": torch.ones(b, m, k, dtype=torch.bool),
        "positions": states[:, 1:n, :, 0],
        "speeds": states[:, 1:n, :, 4],
        "accel": torch.zeros(b, m, k),
        "lane_v5_acts": torch.ones(b, m, k, dtype=torch.bool),
        "merge_core_gap_l": torch.full((b, m, k), 30.0),
        "merge_core_gap_f": torch.full((b, m, k), 15.0),
        "merge_core_v_l": torch.full((b, m, k), 8.0),
        "merge_core_v_f": torch.full((b, m, k), 8.0),
        "merge_core_a_l": zeros,
        "merge_core_a_f": zeros,
        "merge_core_has_l": torch.ones(b, m, k, dtype=torch.bool),
        "merge_core_has_f": torch.ones(b, m, k, dtype=torch.bool),
        "merge_core_alpha_log_location": alpha_loc,
        "merge_core_alpha_scale": torch.full((b, m, k), 0.3),
        "merge_core_residual": zeros,
        "merge_core_s0": torch.full((m,), init["s0_m"]),
        "merge_core_T": torch.full((m,), init["T_s"]),
        "merge_core_b": torch.full((m,), init["b_mps2"]),
        "merge_core_tau_a": torch.full((m,), init["tau_a_s"]),
        "merge_core_temperature": torch.full((m,), init["temperature_m"]),
    }
    current = {"lane": states[:, 0, :, 1:4].argmax(-1), "present": torch.ones(b, k, dtype=torch.bool),
               "s_m": states[:, 0, :, 0], "v_mps": states[:, 0, :, 4],
               "a_mps2": torch.zeros(b, k)}
    return targets, pred, current


def test_merge_core_loss_is_finite_and_differentiable():
    targets, pred, current = _targets_and_pred()
    term = MergeCoreTerm(MergeCoreConfig(weight=1.0, tf_weight=1.0))
    term.prepare(targets)
    value = term(LossContext(pred=pred, current=current, targets=targets, c0=0, c1=3))
    assert value is not None
    assert torch.isfinite(value.value)
    value.value.backward()
    grad = pred["merge_core_alpha_log_location"].grad
    assert grad is not None and torch.isfinite(grad).all()
    metrics = __import__("wms.train.losses.merge_core", fromlist=["merge_core_metrics"]
                         ).merge_core_metrics(targets.prepared["merge_core"])
    assert metrics["merge_core_events"] >= 1.0
    assert metrics["merge_core_tf_events"] >= 1.0


def test_default_switch_is_off_and_yield_is_zero_init():
    default = WmSModel()
    assert default.merge_decision == "hazard"
    assert default.merge_core is None
    model = WmSModel(ModelSpec(lane_model="hazard_v5", lane_decision="gated",
                               merge_decision="secure_gap",
                               rollout_mode="driver_params"), physics=PhysicsConfig()).eval()
    kin = torch.zeros(1, 3, 7)
    kin[0, 0, 0], kin[0, 0, 4] = 40.0, 8.0
    kin[0, 0, 2] = 1.0                     # ramp lane 2
    kin[0, 1, 0], kin[0, 1, 4], kin[0, 1, 3] = 60.0, 8.0, 1.0   # leader lane 1
    kin[0, 2, 0], kin[0, 2, 4], kin[0, 2, 3] = 20.0, 8.0, 1.0   # follower lane 1
    kin[..., 6] = 5.0
    state = {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
             "a_mps2": torch.zeros(1, 3), "lane": torch.tensor([[2, 1, 1]]),
             "length_m": kin[..., 6], "present": torch.ones(1, 3, dtype=torch.bool),
             "is_cav": torch.zeros(1, 3, dtype=torch.bool),
             "lane_age": torch.zeros(1, 3, dtype=torch.long)}
    torch.manual_seed(0)
    with torch.no_grad():
        with_core = model.step(state, k_steps=0)
    assert torch.allclose(with_core["merge_core_yield_accel"],
                          torch.zeros_like(with_core["merge_core_yield_accel"]), atol=1e-7)
    saved = model.merge_core
    model.merge_core = None
    try:
        with torch.no_grad():
            without = model.forward(state, k_steps=0)
    finally:
        model.merge_core = saved
    assert torch.allclose(with_core["accel_mps2"], without["accel_mps2"], atol=1e-7)
    assert torch.allclose(with_core["lane_logits"], without["lane_logits"], atol=1e-7)


ROUND2 = Path("E:/Temp/dc_laptop/ck_round2/wm_s_stage240.pt")


@pytest.mark.skipif(not ROUND2.is_file(), reason="round-2 checkpoint is not available")
def test_round2_loads_with_switch_on_at_fitted_init():
    loaded = load_checkpoint(ROUND2, strict=False,
                            model_config={"merge_decision": "secure_gap"})
    model = loaded.model
    assert model.merge_core is not None
    init = load_init()
    p = model.merge_core.params()
    assert p["s0"].item() == pytest.approx(init["s0_m"], rel=1e-5)
    assert p["T"].item() == pytest.approx(init["T_s"], rel=1e-5)
    assert p["b"].item() == pytest.approx(init["b_mps2"], rel=1e-5)
    assert p["tau_a"].item() == pytest.approx(init["tau_a_s"], rel=1e-5)
