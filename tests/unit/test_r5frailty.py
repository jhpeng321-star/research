"""R5FR: persistent per-driver frailty hazard and censored discrete-time survival loss.

Covers the latent distribution (quadrature normalisation), the decision mechanics (one latent
per car across chunks, reset on slot reuse, sigma -> 0 == per-step hazard), the integrated
censored likelihood (event / right-censoring / delayed entry, positive likelihood for every
recorded merge), finite gradients, on-policy censoring, flags-off identity, config plumbing
and a CPU 2-step wiring proof.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.model.factory import build_model
from wms.model.merge_frailty import (MergeFrailtyNet, gauss_hermite, update_frailty_state)
from wms.model.spec import ModelSpec
from wms.physics.config import PhysicsConfig
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.compose import build_loss
from wms.train.losses.merge_frailty import (MergeFrailtyConfig, MergeFrailtyTerm,
                                            merge_frailty_metrics)
from wms.train.settings import TrainSettings

from .test_wmlane import _batch

NEW_FLAT_KEYS = ("merge_frailty_weight", "merge_frailty_n_quad", "merge_frailty_metrics",
                 "merge_frailty_pos_tol_m", "merge_frailty_speed_tol_mps",
                 "merge_frailty_gap_tol_m", "merge_frailty_gap_speed_tol_mps",
                 "merge_frailty_location_reg_weight")


# ------------------------------------------------------------------ quadrature
def test_quadrature_weights_normalise_and_integrate_moments():
    nodes, weights = gauss_hermite(8, dtype=torch.float64)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-12)
    assert float((weights * nodes.square()).sum()) == pytest.approx(1.0, abs=1e-10)
    # A constant integrand is integrated exactly (probability normalisation).
    assert float((weights * torch.ones_like(nodes)).sum()) == pytest.approx(1.0, abs=1e-12)
    nodes4, weights4 = gauss_hermite(4, dtype=torch.float64)
    assert float((weights4 * nodes4.pow(4)).sum()) == pytest.approx(3.0, abs=1e-10)


# ------------------------------------------------------------------ decision mechanics
def test_latent_persists_across_steps_and_resets_on_slot_reuse():
    z = torch.zeros(1, 2)
    ramp = torch.tensor([[True, False]])
    feasible = torch.tensor([[True, False]])
    loc = torch.zeros(1, 2)
    sigma = torch.ones(1, 2)
    present = torch.ones(1, 2, dtype=torch.bool)
    never = torch.ones(1, 2)
    first = update_frailty_state(z, ramp, feasible, present, loc, sigma,
                                 normal=torch.tensor([[1.0, 0.0]]), uniform=never)
    assert first["has_w"][0, 0].item()
    assert first["w"][0, 0].item() == pytest.approx(1.0)
    assert not first["has_w"][0, 1].item()
    # The latent is not redrawn while the car stays present, even with a new normal draw.
    second = update_frailty_state(z, ramp, feasible, present, loc, sigma,
                                  w=first["w"], has_w=first["has_w"],
                                  prev_present=first["prev_present"],
                                  normal=torch.tensor([[-9.0, 0.0]]), uniform=never)
    assert second["w"][0, 0].item() == pytest.approx(1.0)
    # Slot 0 leaves the road, then a new car reuses it: a fresh latent is drawn.
    gone = update_frailty_state(z, torch.zeros_like(ramp), torch.zeros_like(feasible),
                                torch.zeros_like(present), loc, sigma,
                                w=second["w"], has_w=second["has_w"],
                                prev_present=second["prev_present"], uniform=never)
    assert not gone["has_w"][0, 0].item()
    reuse = update_frailty_state(z, ramp, feasible, present, loc, sigma,
                                 w=gone["w"], has_w=gone["has_w"],
                                 prev_present=gone["prev_present"],
                                 normal=torch.tensor([[-2.0, 0.0]]), uniform=never)
    assert reuse["has_w"][0, 0].item()
    assert reuse["w"][0, 0].item() == pytest.approx(-2.0)


def test_sigma_zero_rollout_matches_the_per_step_hazard():
    """With sigma -> 0 the latent is loc (0), so the decision is exactly Bernoulli(sigmoid z).

    This is the no-training reduction check: a rollout of the decision state (one latent per
    car, at-risk from its first ramp step) produces the same merge count as the per-step
    hazard draw with the same uniforms.
    """
    torch.manual_seed(3)
    B, K, T = 2, 4, 60
    present = torch.ones(B, K, dtype=torch.bool)
    ramp = torch.ones(B, K, dtype=torch.bool)
    feasible = torch.ones(B, K, dtype=torch.bool)
    loc = torch.zeros(B, K)
    sigma = torch.zeros(B, K)
    w = torch.zeros(B, K)
    has = torch.zeros(B, K, dtype=torch.bool)
    prev = torch.zeros(B, K, dtype=torch.bool)
    frailty = torch.zeros(B, K, dtype=torch.bool)
    hazard = torch.zeros(B, K, dtype=torch.bool)
    for _ in range(T):
        z = torch.randn(B, K) * 2.0
        uniform = torch.rand(B, K)
        carry = update_frailty_state(z, ramp, feasible, present, loc, sigma, w=w, has_w=has,
                                     prev_present=prev, normal=torch.zeros(B, K),
                                     uniform=uniform)
        w, has, prev = carry["w"], carry["has_w"], carry["prev_present"]
        step_cross = feasible & ~frailty & (uniform < torch.sigmoid(z))
        frailty = frailty | carry["cross"]
        hazard = hazard | step_cross
    assert int(frailty.sum()) == int(hazard.sum())
    assert torch.equal(frailty, hazard)


def test_frailty_net_is_zero_centred_with_a_positive_scale():
    net = MergeFrailtyNet(input_dim=0)
    loc, sigma = net(None, (2, 3))
    assert loc.shape == (2, 3) and sigma.shape == (2, 3)
    assert float(loc.detach().abs().max()) == 0.0
    assert float(sigma.detach().min()) > 0.0
    with torch.no_grad():
        net.location_bias.fill_(100.0)
    assert float(net(None, (1, 1))[0].detach().abs().max()) <= net.location_bound + 1e-6
    net2 = MergeFrailtyNet(input_dim=8)
    assert float(net2(torch.randn(2, 3, 8), (2, 3))[0].detach().abs().max()) == 0.0


# ------------------------------------------------------------------ loss fixtures
def _targets(n=8, k=3, event=3, entry=None, slot=0):
    states = torch.zeros(1, n + 1, k, 7)
    states[..., 0] = torch.tensor([50.0, 30.0, 60.0][:k])
    states[..., 4] = 10.0
    states[..., 6] = 5.0
    lanes = torch.full((1, n + 1, k), 2, dtype=torch.long)
    if entry is not None:
        lanes[:, :entry + 1, slot] = 1
    if event is not None:
        lanes[:, event + 1:, slot] = 1
    states[..., 1:4] = F.one_hot(lanes, 3).float()
    present = torch.ones(1, n + 1, k, dtype=torch.bool)
    return UnrollTargets(
        states=states, n_steps=n, weights=torch.ones(1, n, 1), hdv_mask=present[:, 1:],
        aux_mask=present[:, :n], aux_count=int(present[:, :n].sum()),
        denom=torch.tensor(float(present[:, 1:].sum())), lane_target=lanes[:, 1:],
        loss_valid=present, is_cav=torch.zeros_like(present), dt_s=0.5,
        position_delta_m=5.0, speed_delta_mps=1.0, history_lane_head=False), lanes


def _ctx(targets, head, acts=None, lanes=None, positions=None, loc=None, sigma=None,
         entrant=None, c0=0, c1=None):
    n = targets.n_steps
    c1 = n if c1 is None else c1
    b, _, k = targets.states.shape[:3]
    s = targets.states[:, :, :, 0]
    lanes = (targets.states[:, :, :, 1:4].argmax(-1) if lanes is None else lanes)
    acts = torch.ones(b, n, k, dtype=torch.bool) if acts is None else acts
    pred = {
        "lanes": lanes[:, c0 + 1:c1 + 1],
        "masks": torch.ones(b, c1 - c0, k, dtype=torch.bool),
        "positions": s[:, c0 + 1:c1 + 1] if positions is None else positions[:, c0 + 1:c1 + 1],
        "speeds": targets.states[:, c0 + 1:c1 + 1, :, 4],
        "accel": torch.zeros(b, c1 - c0, k),
        "lane_logits": torch.zeros(b, c1 - c0, k, 3),
        "lane_v5_head_logit": head[:, c0:c1],
        "lane_v5_acts": acts[:, c0:c1],
        "entrant_seq": (torch.zeros(b, c1 - c0, k, dtype=torch.bool)
                        if entrant is None else entrant[:, c0:c1, :k].bool()),
    }
    if loc is not None:
        pred["merge_frailty_location"] = loc[:, c0:c1]
    if sigma is not None:
        pred["merge_frailty_sigma"] = sigma[:, c0:c1]
    current = {"lane": lanes[:, c0], "present": torch.ones(b, k, dtype=torch.bool),
               "s_m": s[:, c0], "v_mps": targets.states[:, c0, :, 4],
               "a_mps2": targets.states[:, c0, :, 5],
               "length_m": targets.states[:, c0, :, 6]}
    return LossContext(pred=pred, current=current, targets=targets, c0=c0, c1=c1)


def _run(targets, head, cfg=None, **kw):
    term = MergeFrailtyTerm(cfg or MergeFrailtyConfig(weight=1.0))
    term.prepare(targets)
    value = term(_ctx(targets, head, **kw))
    return term, value, targets.prepared["merge_frailty"]


def _expected(nll_parts):
    return sum(nll_parts)


# ------------------------------------------------------------------ loss behaviour
def test_event_and_censored_likelihood_match_hand_computation():
    targets, _ = _targets(event=3)
    head = torch.zeros(1, 8, 3)
    loc = torch.zeros(1, 8, 3)
    sigma = torch.ones(1, 8, 3)
    _, value, prep = _run(targets, head, loc=loc, sigma=sigma,
                          cfg=MergeFrailtyConfig(weight=1.0, n_quad=8))
    nodes, weights = gauss_hermite(8, dtype=torch.float64)
    h = torch.sigmoid(nodes)
    event_q = torch.log1p(-h) * 3 + torch.log(h)          # steps 0,1,2 + event at 3
    cens_q = torch.log1p(-h) * 8                          # slots 1, 2 never merge
    expect = -(torch.logsumexp(torch.log(weights) + event_q, 0)
               + 2.0 * torch.logsumexp(torch.log(weights) + cens_q, 0)) / 3.0
    torch.testing.assert_close(value.value.detach(), expect.to(torch.float32),
                               rtol=2e-4, atol=1e-5)
    metrics = merge_frailty_metrics(prep)
    assert metrics["merge_frailty_events"] == 1.0
    assert metrics["merge_frailty_zero_likelihood_events"] == 0.0


def test_every_recorded_merge_has_positive_likelihood_even_below_a_previous_max():
    # A strongly negative constant score: the first-passage model would have zero mass,
    # the frailty hazard keeps h_T > 0 for every car.
    targets, _ = _targets(event=5)
    head = torch.full((1, 8, 3), -40.0)
    _, value, prep = _run(targets, head, loc=torch.zeros(1, 8, 3),
                          sigma=torch.ones(1, 8, 3))
    metrics = merge_frailty_metrics(prep)
    assert metrics["merge_frailty_events"] == 1.0
    assert metrics["merge_frailty_zero_likelihood_events"] == 0.0
    assert torch.isfinite(value.value).all()


def test_infeasible_steps_give_no_merge_opportunity():
    targets, _ = _targets(event=3)
    head = torch.zeros(1, 8, 3)
    acts = torch.ones(1, 8, 3, dtype=torch.bool)
    acts[0, 1, 0] = False                                  # one infeasible pre-event step
    _, value, prep = _run(targets, head, acts=acts, loc=torch.zeros(1, 8, 3),
                          sigma=torch.ones(1, 8, 3))
    nodes, weights = gauss_hermite(8, dtype=torch.float64)
    h = torch.sigmoid(nodes)
    event_q = torch.log1p(-h) * 2 + torch.log(h)   # step 1 drops out of survival
    cens_q = torch.log1p(-h) * 8
    expect = -(torch.logsumexp(torch.log(weights) + event_q, 0)
               + 2.0 * torch.logsumexp(torch.log(weights) + cens_q, 0)) / 3.0
    torch.testing.assert_close(value.value.detach(), expect.to(torch.float32),
                               rtol=2e-4, atol=1e-5)
    assert merge_frailty_metrics(prep)["merge_frailty_events"] == 1.0


def test_delayed_entry_does_not_accumulate_pre_entry_survival():
    targets, _ = _targets(event=6, entry=4, slot=1)
    head = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3) * 0.1
    term, value, prep = _run(targets, head, loc=torch.zeros(1, 8, 3),
                             sigma=torch.ones(1, 8, 3))
    assert torch.isfinite(value.value).all()
    assert merge_frailty_metrics(prep)["merge_frailty_events"] == 1.0


def test_gradient_is_finite_and_shapes_the_event_hazard():
    targets, _ = _targets(event=3)
    head = (torch.arange(24, dtype=torch.float32).reshape(1, 8, 3) * 0.1
            ).requires_grad_()
    loc = torch.zeros(1, 8, 3, requires_grad=True)
    sigma = torch.ones(1, 8, 3, requires_grad=True)
    _, value, _ = _run(targets, head, loc=loc, sigma=sigma)
    value.value.backward()
    for tensor in (head, loc, sigma):
        assert tensor.grad is not None and bool(torch.isfinite(tensor.grad).all())
    assert float(head.grad[0, 3, 0]) < 0.0            # raise the event hazard
    assert float(head.grad[0, 0:3, 0].abs().sum()) > 0.0   # lower pre-event hazards


def test_compatibility_censors_at_last_compatible_step():
    targets, _ = _targets(event=5)
    head = torch.zeros(1, 8, 3)
    positions = targets.states[:, :, :, 0].clone()
    positions[:, 3:, 0] += 100.0
    _, value, prep = _run(targets, head, positions=positions,
                          loc=torch.zeros(1, 8, 3), sigma=torch.ones(1, 8, 3))
    metrics = merge_frailty_metrics(prep)
    assert metrics["merge_frailty_censored_divergence"] >= 1.0
    assert metrics["merge_frailty_events"] == 0.0
    assert torch.isfinite(value.value).all()


def test_early_model_merge_is_censored_and_counted():
    targets, lanes = _targets(event=5)
    head = torch.zeros(1, 8, 3)
    model_lanes = lanes.clone()
    model_lanes[:, 1:, 0] = 1                    # the model merges slot 0 at step 0
    _, value, prep = _run(targets, head, lanes=model_lanes,
                          loc=torch.zeros(1, 8, 3), sigma=torch.ones(1, 8, 3))
    metrics = merge_frailty_metrics(prep)
    assert metrics["merge_frailty_early_merges"] == 1.0
    assert metrics["merge_frailty_events"] == 0.0
    assert torch.isfinite(value.value).all()


def test_non_ramp_slots_are_not_counted():
    targets, lanes = _targets(event=3)
    mainline = torch.zeros_like(lanes)
    targets.states[..., 1:4] = F.one_hot(mainline, 3).float()
    targets.lane_target = mainline[:, 1:]
    _, value, prep = _run(targets, torch.zeros(1, 8, 3), loc=torch.zeros(1, 8, 3),
                          sigma=torch.ones(1, 8, 3))
    metrics = merge_frailty_metrics(prep)
    assert metrics["merge_frailty_events"] == 0.0
    assert metrics["merge_frailty_censored_divergence"] == 0.0
    assert torch.isfinite(value.value).all()


# ------------------------------------------------------------------ flags / config
def test_default_switch_is_off_and_flat_keys_absent():
    flat = TrainConfig().to_legacy_flat()
    assert not set(NEW_FLAT_KEYS) & set(flat)
    settings = TrainSettings()
    assert settings.merge_decision == "hazard" and not settings.merge_frailty_on
    assert "merge_frailty" not in build_loss(settings).names
    cfg = TrainConfig()
    assert cfg.loss.merge_frailty.weight == 0.0 and cfg.loss.merge_frailty.n_quad == 8


def test_config_and_model_wiring():
    cfg = TrainConfig.from_legacy_flat({
        "lane_model": "hazard_v5", "rollout_mode": "hybrid",
        "merge_decision": "frailty", "merge_frailty_weight": 1.0,
        "merge_frailty_metrics": True, "merge_frailty_n_quad": 6,
        "merge_timing_weight": 1.0, "merge_survival_weight": 1.0})
    assert cfg.model.merge_decision == "frailty"
    assert cfg.loss.merge_frailty.n_quad == 6
    settings = TrainSettings.from_config(cfg)
    assert settings.merge_frailty_on and settings.merge_frailty_n_quad == 6
    names = build_loss(settings).names
    assert "merge_frailty" in names
    assert "merge_timing" not in names and "merge_first_passage" not in names
    flat = cfg.to_legacy_flat()
    spec = ModelSpec.from_mapping(flat)
    assert spec.merge_decision == "frailty"
    model = build_model(spec, PhysicsConfig())
    assert model.merge_frailty is not None and model.merge_threshold is None
    with pytest.raises(ValueError):
        TrainConfig.from_legacy_flat({"merge_decision": "bogus"})
    with pytest.raises(ValueError):
        TrainConfig.from_legacy_flat({"merge_frailty_n_quad": 0})
    with pytest.raises(ValueError):
        ModelSpec(merge_decision="bogus")
    with pytest.raises(ValueError):
        TrainSettings(merge_decision="bogus")
    with pytest.raises(ValueError):
        TrainSettings(merge_frailty_n_quad=0)


def test_cpu_two_step_wiring_proof():
    cfg = load_train_config("configs/c4_wm.toml", [
        "runtime.device=cpu", 'model.lane_model="hazard_v5"',
        "physics.lane_learnable=true", 'model.merge_decision="frailty"',
        "loss.merge_frailty.weight=1.0", "loss.merge_frailty.metrics=true"])
    flat = cfg.to_legacy_flat()
    settings = TrainSettings.from_flat(flat)
    physics = PhysicsConfig(sumo_fidelity=True, lane_learnable=True,
                            lane_rule="learned_discretionary", lane_gap_rule="speed")
    torch.manual_seed(0)
    model = build_model(ModelSpec.from_mapping(flat), physics)
    assert model.merge_frailty is not None
    composer = build_loss(settings, physics=physics)
    from wms.train.unroll import UnrollRunner

    runner = UnrollRunner(model, settings, composer, projection=None,
                          device=torch.device("cpu"))
    optim = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0e-3)
    losses = []
    for _ in range(2):
        optim.zero_grad()
        rec = runner.run(_batch(), unroll_steps=2, backward=True, truncation_steps=2)
        assert math.isfinite(rec["loss"])
        assert rec["merge_frailty_n_cars"] > 0.0
        assert math.isfinite(rec["merge_frailty_nll"])
        assert rec["merge_frailty_zero_likelihood_events"] == 0.0
        optim.step()
        losses.append(rec["loss"])
    assert all(math.isfinite(x) for x in losses)
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.merge_frailty.parameters())
