"""Round-5 switches: FIX 1 (hybrid + driver grad across chunks), FIX 2 (per-stage
truncation + step activation checkpointing), FIX 3 (lane gate/flip-rate), plus the
bit-identity regression proving every switch defaults OFF (existing configs, flat
forms and default-model numerics unchanged).
"""
from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.train.losses.compose import build_loss
from wms.train.losses.flip_rate import FlipRateLoss, flip_rate_loss
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

# --- golden baselines captured on 2026-10-12 BEFORE the round-5 changes -------------
GOLDEN_FLAT_R4 = {
    "batch_size": 8, "black_box": False, "checkpoint_residual": False,
    "conservation_enabled": False, "conservation_weight": 0.0, "controller_embed_dim": 16,
    "device": "cuda", "driver_sidecar": None, "entries_v2": False, "entry_pad": 8,
    "epochs_per_stage": 1, "exclude_window_list": ["lpwo_v3/excluded_runs_PI_wall_v3.json"],
    "gate_kind": "saturating", "grad_clip": 1.0, "history_features": 6, "history_s": 60.0,
    "history_stride_s": 1.0, "lane_head": "off", "lane_logit_cap": None,
    "lane_rate_target": 0.00124327163, "lane_rate_weight": 1.0, "lane_weight": 1.0,
    "loss_share_logging": True, "lr": 0.001, "n_controllers": 7, "num_workers": 0,
    "out_dir": "runs/active/wms", "position_huber_delta_m": 5.0, "position_weight": 1.0,
    "projection_enabled": False, "projection_min_gap_m": 2.0, "residual_cap_mps2": None,
    "residual_mag_weight": 0.01, "residual_magnitude_weight": None,
    "residual_soft_bound_mps2": 4.0, "rollout_mode": "residual", "seed": 101,
    "select_by_long_horizon": True, "speed_huber_delta_mps": 1.0, "speed_weight": 1.0,
    "stability_margin": 1.0, "stability_probes": 1, "stability_weight": 0.0,
    "stage_steps": [2500, 2500, 1875, 1250], "state_noise_std_m": 0.0,
    "state_noise_std_mps": 0.0, "step_weight_decay": 0.01, "steps_per_epoch": 2000,
    "switch_window_fraction": 0.0, "threads": 2, "traffic_bin_s": 30.0,
    "traffic_free_speed_mps": 13.89, "traffic_loss": False, "traffic_ramp_end_s": 120.0,
    "traffic_ramp_start_s": 30.0, "traffic_weight": 0.1, "truncation_steps": 20,
    "unroll_curriculum_s": [10.0, 30.0, 60.0, 120.0], "val_batches": 8,
    "val_window_list": "lpwo_v3/windows_val_noPI_noWall.json", "weight_decay": 0.0,
    "window_list": "lpwo_v3/windows_train_noPI_noWall.json",
}
GOLDEN_FLAT_DEFAULT = {
    "batch_size": 8, "black_box": False, "checkpoint_residual": False,
    "conservation_enabled": False, "conservation_weight": 0.0, "controller_embed_dim": 16,
    "device": "cpu", "driver_sidecar": None, "entries_v2": False, "entry_pad": 8,
    "epochs_per_stage": 1, "exclude_window_list": None, "grad_clip": 1.0,
    "history_features": 6, "lane_head": "off", "lane_logit_cap": None, "lane_weight": 1.0,
    "loss_share_logging": False, "lr": 0.001, "n_controllers": 7, "num_workers": 0,
    "out_dir": "runs/active/wms", "position_huber_delta_m": 5.0, "position_weight": 1.0,
    "projection_enabled": False, "projection_min_gap_m": 2.0, "residual_cap_mps2": None,
    "residual_magnitude_weight": None, "rollout_mode": "residual", "seed": 101,
    "speed_huber_delta_mps": 1.0, "speed_weight": 1.0, "stability_margin": 1.0,
    "stability_probes": 1, "stability_weight": 0.0, "stage_steps": None,
    "state_noise_std_m": 0.0, "state_noise_std_mps": 0.0, "step_weight_decay": 0.01,
    "steps_per_epoch": 2000, "switch_window_fraction": 0.0, "threads": 6,
    "traffic_bin_s": 30.0, "traffic_free_speed_mps": 13.89, "traffic_loss": False,
    "traffic_ramp_end_s": 120.0, "traffic_ramp_start_s": 30.0, "traffic_weight": 0.1,
    "truncation_steps": 20, "unroll_curriculum_s": [10.0, 30.0, 60.0, 120.0],
    "val_batches": 8, "val_window_list": None, "weight_decay": 0.0, "window_list": None,
}
GOLDEN_RECORD = {
    "entrants": 0, "gate_g0": 0.0, "gate_g_last": 0.03357674181461334,
    "gate_monotone": True, "lane": 1.1563764810562134, "loss": 222.7676239013672,
    "position": 215.51055908203125, "predicted_lane_change_rate": 0.09352046251296997,
    "raw_residual_abs_mean_mps2": 0.0026664503384381533,
    "raw_residual_abs_p95_mps2": 0.007273766677826643, "speed": 6.1006975173950195,
    "unroll_steps": 6, "valid_hdv_slots_t0": 4, "valid_targets": 24,
}


def _default_batch():
    """Deterministic synthetic batch (identical RNG order to the golden capture)."""
    B, T, K = 2, 7, 4
    lane_onehot = torch.zeros(B, T, K, 3)
    lane_onehot[..., 1] = 1.0
    states = torch.zeros(B, T, K, 7)
    states[..., 0] = torch.rand(B, T, K) * 80 + torch.arange(K) * 22
    states[..., 1:4] = lane_onehot
    states[..., 4] = torch.rand(B, T, K) * 15 + 5
    states[..., 5] = 0.1
    states[..., 6] = 5.0
    present = torch.ones(B, T, K, dtype=torch.bool)
    present[:, :, 3] = False
    is_cav = torch.zeros(B, T, K, dtype=torch.bool)
    is_cav[:, :, 0] = True
    teacher = torch.zeros(B, T, K, 4)
    hist = torch.zeros(B, K, 10, 6)
    hist[..., 0] = 40.0
    hist[..., 4] = 18.0
    hist_mask = torch.ones(B, K, 10, dtype=torch.bool)
    return {"states": states, "present": present, "is_cav": is_cav, "cav_teacher": teacher,
            "hist": hist, "hist_mask": hist_mask}


def _run_default(steps: int) -> dict:
    torch.manual_seed(0)
    model = WmSModel(ModelSpec())
    model.train()
    batch = _default_batch()
    settings = TrainSettings.from_config(TrainConfig())
    composer = build_loss(settings, physics=model.physics)
    runner = UnrollRunner(model, settings, composer, projection=None,
                          device=torch.device("cpu"))
    return runner.run(batch, unroll_steps=steps, backward=True)


# --- defaults bit-identical --------------------------------------------------------

def test_existing_r4_flat_form_is_bit_identical():
    cfg = load_train_config("r4")
    assert cfg.to_legacy_flat() == GOLDEN_FLAT_R4
    assert TrainConfig.from_legacy_flat(GOLDEN_FLAT_R4).to_legacy_flat() == GOLDEN_FLAT_R4


def test_default_config_flat_form_is_bit_identical():
    assert TrainConfig().to_legacy_flat() == GOLDEN_FLAT_DEFAULT


def test_default_unroll_record_is_bit_identical():
    record = _run_default(6)
    for key, value in GOLDEN_RECORD.items():
        assert record[key] == value, key


def test_default_composer_and_state_dict_unchanged():
    settings = TrainSettings()
    assert build_loss(settings).names == ("position", "speed", "lane")
    torch.manual_seed(1)
    keys = WmSModel(ModelSpec(gate_kind="saturating")).state_dict().keys()
    assert not [key for key in keys if key.startswith("lane_gate")]
    assert not [key for key in keys if "checkpoint_step" in key]


# --- FIX 1: hybrid + driver_grad_across_chunks (encode once at t0 + leaf grads) -----


def _hybrid_parts(driver_grad: bool):
    torch.manual_seed(3)
    model = WmSModel(ModelSpec(rollout_mode="hybrid", history_s=2.0,
                               gate_kind="saturating"))
    model.train()
    batch = _default_batch()
    B, K = batch["states"].shape[0], batch["states"].shape[2]
    batch["driver_hist"] = torch.zeros(B, K, 3, 9)
    batch["driver_hist"][..., 0] = 0.5
    batch["driver_hist"][..., 2] = 1.0
    batch["driver_hist_mask"] = torch.ones(B, K, 3, dtype=torch.bool)
    settings = TrainSettings(rollout_mode="hybrid", history_s=2.0, truncation_steps=2,
                             lane_rate_weight=0.0, driver_grad_across_chunks=driver_grad)
    composer = build_loss(settings, physics=model.physics)
    return model, UnrollRunner(model, settings, composer, projection=None,
                               device=torch.device("cpu")), batch


def _encoder_grad_norm(model) -> float:
    total = 0.0
    for parameter in model.driver_encoder.parameters():
        if parameter.grad is None:  # e.g. history_proj is unused in long-history mode
            continue
        assert bool(torch.isfinite(parameter.grad).all())
        total += float(parameter.grad.detach().square().sum())
    return total ** 0.5


def _encoder_grad_vector(model) -> dict[str, torch.Tensor]:
    return {name: p.grad.detach().clone()
            for name, p in model.driver_encoder.named_parameters()
            if p.grad is not None}


class _LateOnlyComposer:
    """Delegating composer that zeroes the optimised total of chunks below ``min_c0``.

    ``total * 0`` keeps the graph connected with an exactly-zero gradient, so every
    surviving encoder gradient provably comes from chunks at/after ``min_c0``.
    """

    def __init__(self, composer, min_c0: int) -> None:
        self._composer = composer
        self.min_c0 = int(min_c0)

    def __call__(self, ctx):
        breakdown = self._composer(ctx)
        if ctx.c0 < self.min_c0:
            breakdown.total = breakdown.total * 0.0
        return breakdown

    def __getattr__(self, name):
        return getattr(self._composer, name)


def test_fix1_encoder_grad_from_late_chunks_only():
    """A loss confined to chunks >= 2 reaches the encoder only with the flag on."""
    model, runner, batch = _hybrid_parts(driver_grad=True)
    runner.composer = _LateOnlyComposer(runner.composer, min_c0=4)  # chunks 2 and 3 of 4
    runner.run(batch, unroll_steps=8, backward=True)
    assert _encoder_grad_norm(model) > 0.0

    model_off, runner_off, batch_off = _hybrid_parts(driver_grad=False)
    runner_off.composer = _LateOnlyComposer(runner_off.composer, min_c0=4)
    runner_off.run(batch_off, unroll_steps=8, backward=True)
    # Flag OFF: the detached carry-over cuts the encoder after chunk 0 -> zero grad.
    assert _encoder_grad_norm(model_off) == 0.0


def test_fix1_leaf_gradient_equals_full_bptt():
    """With one chunk covering the unroll, the leaf mechanism is exact full BPTT."""
    results = {}
    for label, driver_grad, truncation in (("on_1chunk", True, 8),
                                           ("on_4chunks", True, 2),
                                           ("off_1chunk", False, 8),
                                           ("off_4chunks", False, 2)):
        model, runner, batch = _hybrid_parts(driver_grad=driver_grad)
        settings = dataclasses.replace(runner.settings, truncation_steps=truncation)
        composer = build_loss(settings, physics=model.physics)
        runner = UnrollRunner(model, settings, composer, projection=None,
                              device=torch.device("cpu"))
        record = runner.run(batch, unroll_steps=8, backward=True)
        results[label] = (_encoder_grad_vector(model), record["loss"])
    on_full, loss_full = results["on_1chunk"]
    on_chunked, loss_chunked = results["on_4chunks"]
    off_full, loss_off = results["off_1chunk"]
    off_chunked, _ = results["off_4chunks"]
    # Same forward numerics: the t0 encoding is used identically in all runs
    # (differences are float32 accumulation order across chunk boundaries).
    assert loss_chunked == pytest.approx(loss_full, rel=1e-5)
    assert loss_off == pytest.approx(loss_full, abs=1e-6)
    assert set(on_full) == set(on_chunked) == set(off_full)
    # One chunk == full BPTT: the leaf path reproduces the direct-encode gradient.
    for name, grad in on_full.items():
        assert torch.allclose(grad, off_full[name], atol=1e-6), name
    # Chunked (4 chunks): every chunk still contributes (the detached carry of the
    # flag-OFF path would keep only chunk 0's contribution); the remaining gap to
    # the single-chunk gradient is the truncated-BPTT state detach itself.
    chunked_total = sum(float(g.abs().sum()) for g in on_chunked.values())
    off_chunked_total = sum(float(g.abs().sum()) for g in off_chunked.values())
    full_total = sum(float(g.abs().sum()) for g in on_full.values())
    assert off_chunked_total < chunked_total < full_total


def test_fix1_leaf_mechanism_is_lossless_on_a_feedback_free_toy():
    """Toy without state feedback: chunked leaf grads == the full-BPTT gradient."""
    w = torch.nn.Parameter(torch.tensor(3.0))
    x = torch.tensor(2.0)
    encoding = w * x  # encode once
    leaf = encoding.detach().requires_grad_(True)
    state = torch.tensor(1.0)  # exogenous carry (no feedback onto the encoding)
    losses = []
    for chunk in range(3):
        step = leaf * state
        losses.append(step.square().sum() * (chunk + 1))
        state = (state * 0.5).detach()
    torch.autograd.backward(losses)  # chunks accumulate into the leaf
    torch.autograd.backward([encoding], [leaf.grad])
    leaf_based = w.grad.detach().clone()

    w.grad = None
    state = torch.tensor(1.0)
    total = 0.0
    for chunk in range(3):
        step = (w * x) * state
        total = total + step.square().sum() * (chunk + 1)
        state = state * 0.5
    total.backward()
    assert torch.allclose(leaf_based, w.grad, atol=1e-7)


def test_fix1_every_chunk_consumes_the_t0_encoding():
    seen = []

    for flag in (False, True):
        model, runner, batch = _hybrid_parts(driver_grad=flag)
        original = model.rollout

        def spy(state, *, n_steps, **kwargs):
            seen.append((flag, kwargs.get("driver_params") is not None))
            return original(state, n_steps=n_steps, **kwargs)

        model.rollout = spy
        runner.run(batch, unroll_steps=6, backward=False)
    # flag OFF: chunk 0 encodes internally, later chunks receive the detached carry;
    # flag ON: the t0 encoding (detached leaf tensors) is handed to every chunk.
    assert seen == [(False, False), (False, True), (False, True),
                    (True, True), (True, True), (True, True)]


def test_fix1_flag_on_is_bit_identical_forward_to_flag_off():
    model_on, runner_on, batch_on = _hybrid_parts(driver_grad=True)
    model_off, runner_off, batch_off = _hybrid_parts(driver_grad=False)
    torch.manual_seed(11)
    record_on = runner_on.run(batch_on, unroll_steps=6, backward=False)
    torch.manual_seed(11)
    record_off = runner_off.run(batch_off, unroll_steps=6, backward=False)
    assert set(record_on) == set(record_off)
    for key in ("loss", "position", "speed", "lane", "predicted_lane_change_rate"):
        assert record_on[key] == pytest.approx(record_off[key], abs=1e-9), key


def test_fix1_hybrid_physics_uses_encoder_settings():
    torch.manual_seed(4)
    model = WmSModel(ModelSpec(rollout_mode="hybrid", history_s=2.0, gate_kind="saturating"))
    batch = _default_batch()
    B, K = batch["states"].shape[0], batch["states"].shape[2]
    batch["driver_hist"] = torch.zeros(B, K, 3, 9)
    batch["driver_hist_mask"] = torch.ones(B, K, 3, dtype=torch.bool)
    state = {"s_m": batch["states"][:, 0, :, 0], "lane": torch.full((B, K), 1),
             "v_mps": batch["states"][:, 0, :, 4], "a_mps2": torch.zeros(B, K),
             "present": torch.ones(B, K, dtype=torch.bool)}
    out = model(state, k_steps=5, driver_history=batch["driver_hist"],
                driver_history_mask=batch["driver_hist_mask"])
    assert "driver_parameters" in out and "driver_embedding" in out
    assert model.residual_net.cond_proj is not None
    physics_before = out["physics_accel_mps2"].detach().clone()
    with torch.no_grad():
        model.driver_encoder.head[-1].bias.add_(0.5)
    out2 = model(state, k_steps=5, driver_history=batch["driver_hist"],
                 driver_history_mask=batch["driver_hist_mask"])
    # The encoder's settings reach the physics in hybrid mode (they were inert in
    # residual mode; R4-DIAG-HIST).
    assert not torch.equal(physics_before, out2["physics_accel_mps2"].detach())


# --- FIX 2: per-stage truncation + activation checkpointing ------------------------

def test_fix2_stage_plan_carries_per_stage_truncation():
    from wms.train.curriculum import stage_plan
    settings = TrainSettings(truncation_steps=20, truncation_steps_per_stage=(20, 60, 120, 240))
    plans = stage_plan(settings, 0.5)
    assert [p.truncation_steps for p in plans] == [20, 60, 120, 240]
    assert [p.unroll_steps for p in plans] == [20, 60, 120, 240]
    fallback = stage_plan(TrainSettings(truncation_steps=7), 0.5)
    assert [p.truncation_steps for p in fallback] == [7, 7, 7, 7]


def test_fix2_runner_honours_truncation_override():
    model, runner, batch = _hybrid_parts(driver_grad=False)
    calls = []
    original = model.rollout

    def spy(state, *, n_steps, **kwargs):
        calls.append(n_steps)
        return original(state, n_steps=n_steps, **kwargs)

    model.rollout = spy
    runner.run(batch, unroll_steps=6, backward=False, truncation_steps=4)
    assert calls == [4, 2]
    calls.clear()
    runner.run(batch, unroll_steps=6, backward=False)  # settings default: 2
    assert calls == [2, 2, 2]


def test_fix2_step_checkpoint_is_transparent_and_differentiable():
    torch.manual_seed(5)
    model = WmSModel(ModelSpec(gate_kind="saturating"))
    model.train()
    batch = _default_batch()
    state = {"s_m": batch["states"][:, 0, :, 0].clone(),
             "lane": torch.full((2, 4), 1),
             "v_mps": batch["states"][:, 0, :, 4].clone(),
             "a_mps2": torch.zeros(2, 4), "present": torch.ones(2, 4, dtype=torch.bool)}
    model.checkpoint_step = False
    plain = model.rollout(state, n_steps=3)
    model.checkpoint_step = True
    checked = model.rollout(state, n_steps=3)
    for key in ("positions", "speeds", "lanes", "lane_logits", "accel"):
        assert torch.equal(plain[key], checked[key]), key
    checked["speeds"].sum().backward()
    grads = [p.grad for p in model.residual_net.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)


def test_fix2_closed_loop_controllers_are_never_checkpointed():
    model = WmSModel(ModelSpec(gate_kind="saturating"))
    model.checkpoint_step = True
    model.train()
    state = {"s_m": torch.tensor([[0.0, 30.0]]), "lane": torch.tensor([[1, 1]]),
             "v_mps": torch.tensor([[20.0, 18.0]]), "a_mps2": torch.zeros(1, 2),
             "present": torch.ones(1, 2, dtype=torch.bool)}
    calls = []

    def controller(callback_state):
        calls.append(1)
        return torch.zeros(1, 2, 4)

    cav = torch.zeros(1, 3, 2, dtype=torch.bool)
    cav[:, :, 0] = True  # slot 0 is a stateful-controller CAV on every step
    model.rollout(state, n_steps=2, is_cav_seq=cav, cav_controller=controller)
    assert len(calls) == 2  # one per step, no recomputation replay


# --- FIX 3: lane gate + flip-rate loss ---------------------------------------------

def test_fix3_separate_lane_gate_params_and_independence():
    torch.manual_seed(6)
    model = WmSModel(ModelSpec(gate_kind="saturating", lane_gate_kind="saturating"))
    assert torch.isclose(torch.sigmoid(model.lane_gate_g_max_logit), torch.tensor(0.16))
    assert torch.isclose(torch.nn.functional.softplus(model.lane_gate_tau_raw),
                         torch.tensor(30.0))
    k = torch.tensor([10, 60, 120, 240])
    assert not torch.equal(model.lane_gate(k), model.gate(k))
    state = {"s_m": torch.tensor([[0.0, 30.0]]), "lane": torch.tensor([[1, 1]]),
             "v_mps": torch.tensor([[20.0, 18.0]]), "a_mps2": torch.zeros(1, 2),
             "present": torch.ones(1, 2, dtype=torch.bool)}
    out = model(state, k_steps=60)
    out["lane_logits"].sum().backward()
    assert model.lane_gate_g_max_logit.grad is not None
    assert float(model.lane_gate_g_max_logit.grad.abs()) > 0
    assert model.lane_gate_tau_raw.grad is not None
    # The lane loss does not touch the acceleration gate of a separate-gate model.
    assert model.gate_g_max_logit.grad is None or float(model.gate_g_max_logit.grad.abs()) == 0


def test_fix3_shared_lane_gate_default_is_bit_identical():
    torch.manual_seed(7)
    model = WmSModel(ModelSpec(gate_kind="saturating"))
    k = torch.tensor([0, 1, 10, 60, 240])
    gate = model.gate(k)
    assert torch.equal(model.lane_gate(k, shared=gate), gate)
    assert torch.equal(model.lane_gate(k), gate)


def test_fix3_flip_rate_loss_value_and_gradient():
    allowance = 0.00124327163
    probabilities = torch.tensor([0.5, 0.001, 0.0, 0.002], requires_grad=True)
    valid = torch.tensor([True, True, False, True])
    value = flip_rate_loss(probabilities, valid, 2.0, allowance)
    hinge = torch.relu(probabilities.detach() - allowance)
    expected = 2.0 * ((hinge[0] + 0.0 + hinge[3]) / 3.0)
    assert torch.isclose(value, expected)
    value.backward()
    grad = probabilities.grad
    assert grad is not None
    assert grad[0] == pytest.approx(2.0 / 3.0)
    assert grad[1] == 0.0
    assert grad[2] == 0.0
    assert grad[3] == pytest.approx(2.0 / 3.0)
    # everything below the allowance: exact zero value and gradient
    low = torch.tensor([0.0, 0.0005, 0.001], requires_grad=True)
    zero = flip_rate_loss(low, torch.ones(3, dtype=torch.bool), 2.0, allowance)
    assert float(zero.detach()) == 0.0
    zero.backward()
    assert float(low.grad.abs().sum()) == 0.0


def test_fix3_flip_rate_term_composes_and_logs():
    settings = TrainSettings(rollout_mode="hybrid", history_s=2.0, truncation_steps=2,
                             lane_rate_weight=0.0, flip_rate_weight=1.0,
                             driver_grad_across_chunks=True)
    composer = build_loss(settings)
    assert "flip_rate" in composer.names
    torch.manual_seed(8)
    model, runner, batch = _hybrid_parts(driver_grad=True)
    # rebuild the runner with the flip-rate composer but keep model/batch identical
    runner = UnrollRunner(model, settings, composer, projection=None,
                          device=torch.device("cpu"))
    record = runner.run(batch, unroll_steps=4, backward=True)
    assert record["flip_rate"] >= 0.0
    assert any(isinstance(term, FlipRateLoss) for term in composer.chunk_terms)


# --- r5 configs flow into the model spec -------------------------------------------

@pytest.mark.parametrize("name,keys", [
    ("r5_base", {"rollout_mode": "hybrid", "driver_grad_across_chunks": True}),
    ("r5_f2", {"checkpoint_step": True,
               "truncation_steps_per_stage": [20, 60, 120, 240]}),
    ("r5_f3", {"lane_gate_kind": "saturating", "lane_logit_cap": 1.5,
               "flip_rate_weight": 1.0, "lane_rate_weight": 0.0}),
    ("r5_f23", {"checkpoint_step": True, "lane_gate_kind": "saturating",
                "flip_rate_weight": 1.0, "truncation_steps_per_stage": [20, 60, 120, 240]}),
])
def test_r5_configs_reach_settings_and_spec(name, keys):
    cfg = load_train_config(name)
    settings = TrainSettings.from_config(cfg)
    for key, value in keys.items():
        actual = getattr(settings, key)
        if isinstance(value, list):
            actual = list(actual)
        assert actual == value, (name, key)
    spec = ModelSpec.from_mapping(cfg.to_legacy_flat())
    assert spec.rollout_mode == "hybrid"
    if "lane_gate_kind" in keys:
        assert spec.lane_gate_kind == "saturating"
    if name in ("r5_f2", "r5_f23"):
        assert spec.checkpoint_step is True
    assert cfg.model.init_from == "ckpt_r4"
    # No driver-type vocabulary anywhere in the recipe.
    text = json.dumps(cfg.to_dict()).lower()
    for banned in ("aggressive", "cautious", "vtype", "driver_type", "true_type"):
        assert banned not in text


def test_settings_field_inventory_matches_flat():
    cfg = TrainConfig()
    flat = cfg.to_legacy_flat()
    names = {f.name for f in dataclasses.fields(TrainSettings)}
    assert set(flat) <= names


# --- round-5 inference-time diagnostic switches (default OFF) -----------------------

def _saturating_gate_model():
    torch.manual_seed(12)
    model = WmSModel(ModelSpec(rollout_mode="hybrid", history_s=2.0, gate_kind="saturating",
                               lane_gate_kind="saturating", lane_logit_cap=1.5))
    model.eval()
    state = {"s_m": torch.tensor([[0.0, 30.0]]), "lane": torch.tensor([[1, 1]]),
             "v_mps": torch.tensor([[20.0, 18.0]]), "a_mps2": torch.zeros(1, 2),
             "present": torch.ones(1, 2, dtype=torch.bool)}
    driver_hist = torch.zeros(1, 2, 3, 9)
    driver_hist[..., 0] = 0.5
    driver_hist[..., 2] = 1.0
    driver_hist_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    rollout_kwargs = {"driver_history": driver_hist,
                      "driver_history_mask": driver_hist_mask}
    return model, state, rollout_kwargs


def test_diag_switches_default_off_bit_identical():
    model, state, kwargs = _saturating_gate_model()
    assert model.diag_encoder_population is False
    base = model.rollout(state, n_steps=3, **kwargs)
    # Defaults re-applied explicitly change nothing (same call path).
    again = model.rollout(state, n_steps=3, lane_teacher=None, **kwargs)
    assert torch.equal(base["lanes"], again["lanes"])
    assert torch.equal(base["lane_logits"], again["lane_logits"])


def test_diag_lane_gate_shared_switch():
    model, state, kwargs = _saturating_gate_model()
    with torch.no_grad():  # a sizeable lane correction so the two gates measurably differ
        model.temporal[-1].bias.fill_(3.0)
    separate = model.rollout(state, n_steps=3, **kwargs)
    model.lane_gate_kind = "shared"  # scorer switch: lane correction uses accel gate
    shared = model.rollout(state, n_steps=3, **kwargs)
    assert not torch.equal(separate["lane_logits"], shared["lane_logits"])
    model.lane_gate_kind = "saturating"
    restored = model.rollout(state, n_steps=3, **kwargs)
    assert torch.equal(separate["lane_logits"], restored["lane_logits"])


def test_diag_lane_cap_off_switch():
    model, state, kwargs = _saturating_gate_model()
    with torch.no_grad():  # correction (gate * delta ~ 0.16 * 6) crosses the 1.5 cap
        model.temporal[-1].bias.fill_(3.0)
    capped = model.rollout(state, n_steps=3, **kwargs)
    model.lane_logit_cap = None  # scorer switch: unbounded lane correction
    uncapped = model.rollout(state, n_steps=3, **kwargs)
    # Removing the cap must let the correction exceed the capped magnitude.
    diff = (uncapped["lane_logits"] - capped["lane_logits"]).abs().max()
    assert float(diff.detach()) > 0.0


def test_diag_encoder_population_switch():
    torch.manual_seed(13)
    model = WmSModel(ModelSpec(rollout_mode="hybrid", history_s=2.0, gate_kind="saturating"))
    model.eval()
    batch = _default_batch()
    B, K = batch["states"].shape[0], batch["states"].shape[2]
    driver_hist = torch.zeros(B, K, 3, 9)
    driver_hist[..., 0] = 0.5
    mask = torch.ones(B, K, 3, dtype=torch.bool)
    kin = batch["states"][:, 0]
    kin_state = {"veh_kin": kin, "present": batch["present"][:, 0]}
    defaults = model._default_driver_parameters(kin, B, K)
    out = model(kin_state, k_steps=3, driver_history=driver_hist, driver_history_mask=mask)
    assert not torch.allclose(out["driver_parameters"], defaults)
    model.diag_encoder_population = True
    out_pop = model(kin_state, k_steps=3, driver_history=driver_hist,
                    driver_history_mask=mask)
    assert torch.allclose(out_pop["driver_parameters"], defaults)
    assert float(out_pop["driver_shrinkage"].abs().max()) == 0.0


def test_diag_recorded_lanes_teacher_forcing():
    model, state, kwargs = _saturating_gate_model()
    recorded = torch.tensor([[[1, 1], [2, 1], [0, 1], [1, 2]]])  # [B, T+1, K], slot 1 moved
    forced = model.rollout(state, n_steps=3, lane_teacher=recorded, **kwargs)
    assert torch.equal(forced["lanes"], recorded[:, 1:].long())
    # A -1 entry frees only that step: the model's own decision there must match the
    # same model given the identical (forced) earlier state.
    free_after_forced = recorded.clone()
    free_after_forced[:, 2:, 0] = -1  # steps 1 and 2 free for slot 0
    partial = recorded.clone()
    partial[:, 2, 0] = -1  # step 1 free, step 2 forced again
    mixed = model.rollout(state, n_steps=3, lane_teacher=partial, **kwargs)
    reference = model.rollout(state, n_steps=3, lane_teacher=free_after_forced, **kwargs)
    assert torch.equal(mixed["lanes"][:, 1, 0], reference["lanes"][:, 1, 0])
    assert torch.equal(mixed["lanes"][:, 2, 0], recorded[:, 3, 0])


def test_diag_flags_map_to_predictor_and_config():
    from types import SimpleNamespace

    from wms.eval.predictor import RolloutFlags, WmsPredictor
    from wms.eval.run import DIAG_SWITCHES, rollout_flags

    predictor = WmsPredictor.__new__(WmsPredictor)
    predictor.model = SimpleNamespace(lane_gate_kind="saturating", lane_logit_cap=1.5,
                                      diag_encoder_population=False)
    predictor._apply_diag_flags(RolloutFlags(diag_lane_gate_shared=True,
                                             diag_lane_cap_off=True,
                                             diag_encoder_population=True))
    assert predictor.model.lane_gate_kind == "shared"
    assert predictor.model.lane_logit_cap is None
    assert predictor.model.diag_encoder_population is True

    from wms.config.eval import EvalConfig
    from wms.config.load import load_eval_config

    flags = rollout_flags(EvalConfig())
    assert flags["entries"] is False
    assert all(flags[name] is False for name in DIAG_SWITCHES)
    cfg = load_eval_config("eval_unseen", ["flags.diag_recorded_lanes=true"])
    assert rollout_flags(cfg)["diag_recorded_lanes"] is True


def test_r5b_lr_per_stage_and_grad_logging_config():
    """Round 5b: per-stage learning rate parses, defaults off, and must match the stage count."""
    from wms.config.load import load_train_config
    from wms.train.settings import TrainSettings

    s = TrainSettings.from_config(load_train_config("configs/r5b.toml"))
    assert s.lr_per_stage == (3.0e-4, 1.0e-4)
    assert s.unroll_curriculum_s == (60.0, 120.0) and s.batch_size == 32
    assert TrainSettings.from_config(load_train_config("configs/r4.toml")).lr_per_stage is None
    import pytest
    with pytest.raises(Exception):
        TrainSettings.from_config(load_train_config("configs/r5b.toml",
                                                    ["optim.lr_per_stage=[0.001]"]))


def test_r5b_grad_scale_is_forward_exact_and_damps_backward():
    import torch
    from wms.model.wms import _scale_grad

    x = torch.tensor([1.5, -2.0], requires_grad=True)
    y = _scale_grad(x, 0.9)
    assert torch.equal(y.detach(), x.detach())
    (y * torch.tensor([2.0, 3.0])).sum().backward()
    assert torch.allclose(x.grad, torch.tensor([1.8, 2.7]))


def test_r5b_grad_decay_default_off():
    from wms.config.load import load_train_config
    from wms.train.settings import TrainSettings

    assert TrainSettings.from_config(load_train_config("configs/r4.toml")).grad_decay_per_step == 1.0
