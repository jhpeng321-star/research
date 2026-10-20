"""R5THR: persistent learned gap-acceptance threshold and integrated censored first passage.

Covers the decision mechanics (one threshold per car across chunks / slot reuse, envelope
feasible running maximum, first crossing), the integrated likelihood (normalisation,
infeasible steps, constant score, empty feasibility, delayed entry, interval censoring,
stable gradients, timestep refinement), on-policy censoring (compatibility, early model
merges), flags-off identity, config plumbing and a CPU 2-step wiring proof.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.model.factory import build_model
from wms.model.merge_threshold import (MergeThresholdNet, first_passage_probs,
                                       log_event_mass, log_survival, update_threshold_state)
from wms.model.spec import ModelSpec
from wms.physics.config import PhysicsConfig
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.compose import build_loss
from wms.train.losses.merge_first_passage import (MergeFirstPassageConfig,
                                                  MergeFirstPassageTerm,
                                                  merge_first_passage_metrics)
from wms.train.settings import TrainSettings

from .test_wmlane import _batch

NEW_FLAT_KEYS = ("merge_decision", "merge_first_passage_weight",
                 "merge_first_passage_rank_weight", "merge_first_passage_metrics",
                 "merge_first_passage_pos_tol_m", "merge_first_passage_speed_tol_mps",
                 "merge_first_passage_gap_tol_m", "merge_first_passage_gap_speed_tol_mps",
                 "merge_first_passage_location_reg_weight")


# ------------------------------------------------------------------ likelihood helpers
def test_normalisation_event_mass_plus_survival():
    loc = torch.tensor(0.3)
    m = torch.tensor([[-1.0, 0.2, 0.2, 1.5, 2.0]])
    p, surv = first_passage_probs(m, loc)
    assert float(p.sum() + surv) == pytest.approx(1.0, abs=1e-6)
    assert float(p.min()) >= 0.0
    # the stable log mass agrees with the plain CDF difference
    prev = torch.cat((torch.tensor([[-math.inf]]), m[:, :-1]), 1)
    plain = torch.sigmoid(m - loc) - torch.sigmoid(prev - loc)
    stable = log_event_mass(prev, m, loc.expand_as(m))
    torch.testing.assert_close(stable.exp(), plain, rtol=1e-6, atol=1e-12)


def test_timestep_refinement_does_not_accumulate_ticks():
    loc = torch.tensor(0.0)
    m = torch.tensor([[0.0, 0.5, 2.0]])
    refined = m.repeat_interleave(3, dim=1)          # same scores, 3x finer steps
    p, surv = first_passage_probs(m, loc)
    pr, survr = first_passage_probs(refined, loc)
    assert float(survr) == pytest.approx(float(surv), rel=1e-9)
    assert float(p.sum() + surv) == pytest.approx(float(pr.sum() + survr), rel=1e-9)


def test_delayed_entry_conditions_on_survival_at_entry():
    loc = torch.tensor(-0.5)
    m_prev = torch.tensor([1.0])
    m_cur = torch.tensor([2.0])
    conditional = log_event_mass(m_prev, m_cur, loc) - log_survival(m_prev, loc)
    mass = math.exp(float(log_event_mass(m_prev, m_cur, loc)))
    surv = math.exp(float(log_survival(m_prev, loc)))
    assert math.exp(float(conditional)) == pytest.approx(mass / surv, rel=1e-8)
    # first at-risk step (no earlier feasible maximum): conditioning is the identity
    none_prev = torch.tensor([-math.inf])
    assert float(log_survival(none_prev, loc)) == pytest.approx(0.0, abs=1e-9)
    torch.testing.assert_close(log_event_mass(none_prev, m_cur, loc),
                               F.logsigmoid(m_cur - loc), rtol=1e-6, atol=1e-12)


# ------------------------------------------------------------------ decision mechanics
def test_threshold_persists_and_resets_on_slot_reuse():
    z = torch.tensor([[0.5, 0.5]])
    ramp = torch.tensor([[True, False]])
    feasible = torch.tensor([[True, False]])
    loc = torch.zeros(1, 2)
    present = torch.ones(1, 2, dtype=torch.bool)
    first = update_threshold_state(z, ramp, feasible, present, loc,
                                   uniform=torch.tensor([[0.3, 0.3]]))
    assert first["cross"][0, 0].item()                  # u = logit(0.3) < 0.5
    assert not first["has_u"][0, 1].item()
    second = update_threshold_state(
        z + 1.0, ramp, feasible, present, loc, u=first["u"], has_u=first["has_u"],
        m=first["m"], prev_present=first["prev_present"],
        uniform=torch.tensor([[0.9, 0.9]]))
    assert torch.equal(second["u"][0, 0], first["u"][0, 0])   # not re-drawn
    assert second["cross"][0, 0].item()                       # z = 1.5 > u
    # slot 0 leaves the road, then comes back: a new car gets a fresh threshold / maximum
    gone = update_threshold_state(z, torch.zeros_like(ramp), torch.zeros_like(feasible),
                                  torch.zeros_like(present), loc,
                                  u=second["u"], has_u=second["has_u"], m=second["m"],
                                  prev_present=second["prev_present"])
    assert not gone["has_u"][0, 0].item()
    assert torch.isinf(gone["m"][0, 0]) and gone["m"][0, 0] < 0
    reuse = update_threshold_state(z, ramp, feasible, present, loc,
                                   u=gone["u"], has_u=gone["has_u"], m=gone["m"],
                                   prev_present=gone["prev_present"],
                                   uniform=torch.tensor([[0.9, 0.9]]))
    assert reuse["has_u"][0, 0].item()
    assert reuse["u"][0, 0] == pytest.approx(math.log(0.9 / 0.1))
    assert not reuse["cross"][0, 0].item()               # u = +2.2 > z = 0.5


def test_merge_threshold_net_starts_centred_and_is_bounded():
    net = MergeThresholdNet(input_dim=0)
    out = net(None, (2, 3))
    assert out.shape == (2, 3) and out.abs().max().item() == 0.0
    with torch.no_grad():
        net.location_bias.fill_(100.0)
    assert net(None, (1, 1)).abs().max().item() <= net.location_bound + 1e-6
    features = torch.randn(2, 3, 8)
    net2 = MergeThresholdNet(input_dim=8)
    assert net2(features, (2, 3)).abs().max().item() == 0.0


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


def _ctx(targets, head, acts=None, lanes=None, positions=None, speeds=None, loc=None,
         c0=0, c1=None):
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
        "speeds": (targets.states[:, c0 + 1:c1 + 1, :, 4] if speeds is None
                   else speeds[:, c0 + 1:c1 + 1]),
        "accel": torch.zeros(b, c1 - c0, k),
        "lane_logits": torch.zeros(b, c1 - c0, k, 3),
        "lane_v5_head_logit": head[:, c0:c1],
        "lane_v5_acts": acts[:, c0:c1],
        "entrant_seq": torch.zeros(b, c1 - c0, k, dtype=torch.bool),
    }
    if loc is not None:
        pred["merge_threshold_location"] = loc[:, c0:c1]
    current = {"lane": lanes[:, c0], "present": torch.ones(b, k, dtype=torch.bool),
               "s_m": s[:, c0], "v_mps": targets.states[:, c0, :, 4],
               "a_mps2": targets.states[:, c0, :, 5],
               "length_m": targets.states[:, c0, :, 6]}
    return LossContext(pred=pred, current=current, targets=targets, c0=c0, c1=c1)


def _run(targets, head, cfg=None, **kw):
    term = MergeFirstPassageTerm(cfg or MergeFirstPassageConfig(weight=1.0))
    term.prepare(targets)
    value = term(_ctx(targets, head, **kw))
    return term, value, targets.prepared["merge_first_passage"]


# ------------------------------------------------------------------ loss behaviour
def test_on_policy_event_and_censoring_match_hand_computation():
    targets, _ = _targets(event=3)
    head = torch.zeros(1, 8, 3)
    head[0, :, 0] = torch.arange(8, dtype=torch.float32)
    term, value, prep = _run(targets, head)
    z = head[0, :, 0]
    m = torch.cummax(z, 0).values
    expect0 = -(torch.log(torch.sigmoid(m[3]) - torch.sigmoid(m[2])))
    expect12 = -(torch.log1p(-torch.sigmoid(torch.tensor(0.0)))) * 2
    torch.testing.assert_close(value.value.detach(),
                               (expect0 + expect12) / 3.0, rtol=1e-5, atol=1e-6)
    metrics = merge_first_passage_metrics(prep)
    assert metrics["merge_fp_events"] == 1.0
    assert metrics["merge_fp_unsupported_events"] == 0.0
    assert metrics["merge_fp_early_merges"] == 0.0


def test_infeasible_event_step_has_zero_mass_and_is_diagnosed():
    targets, _ = _targets(event=3)
    head = torch.zeros(1, 8, 3)
    head[0, :, 0] = torch.arange(8, dtype=torch.float32)
    acts = torch.ones(1, 8, 3, dtype=torch.bool)
    acts[0, 3, 0] = False                       # the event step is envelope-infeasible
    _, value, prep = _run(targets, head, acts=acts)
    metrics = merge_first_passage_metrics(prep)
    assert metrics["merge_fp_unsupported_events"] == 1.0
    assert torch.isfinite(value.value).all()
    # an infeasible step never enters the maximum: M_4.. stay at M_2
    assert metrics["merge_fp_nll"] > 0.0


def test_constant_score_and_empty_feasibility_are_finite():
    targets, _ = _targets(event=0)
    head = torch.zeros(1, 8, 3)
    _, value, prep = _run(targets, head)
    assert torch.isfinite(value.value).all()
    assert merge_first_passage_metrics(prep)["merge_fp_events"] == 1.0
    # constant score and a later event: zero mass, diagnosed, no NaN
    targets2, _ = _targets(event=4)
    _, value2, prep2 = _run(targets2, torch.zeros(1, 8, 3))
    assert torch.isfinite(value2.value).all()
    assert merge_first_passage_metrics(prep2)["merge_fp_unsupported_events"] == 1.0
    # empty feasibility
    targets3, _ = _targets(event=4)
    acts = torch.zeros(1, 8, 3, dtype=torch.bool)
    _, value3, prep3 = _run(targets3, torch.ones(1, 8, 3), acts=acts)
    assert torch.isfinite(value3.value).all()
    metrics3 = merge_first_passage_metrics(prep3)
    assert metrics3["merge_fp_events"] == 1.0
    assert metrics3["merge_fp_unsupported_events"] == 1.0   # no feasible step: zero mass


def test_gradient_is_finite_on_supported_events():
    targets, _ = _targets(event=3)
    head = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3).requires_grad_()
    _, value, _ = _run(targets, head, cfg=MergeFirstPassageConfig(weight=1.0, rank_weight=1.0))
    value.value.backward()
    assert head.grad is not None and bool(torch.isfinite(head.grad).all())
    assert float(head.grad[0, 3, 0]) < 0.0       # raise the event score
    assert float(head.grad[0, 2, 0]) > 0.0       # lower the previous feasible maximum
    assert float(head.grad[0, 0:2, 0].abs().sum()) == 0.0   # never entered the maximum


def test_chunked_first_passage_telescopes():
    targets, _ = _targets(event=5)
    head = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)
    term = MergeFirstPassageTerm(MergeFirstPassageConfig(weight=1.0))
    term.prepare(targets)
    whole = term(_ctx(targets, head)).value
    first = term(_ctx(targets, head, c0=0, c1=3)).value
    second = term(_ctx(targets, head, c0=3, c1=8)).value
    torch.testing.assert_close(first + second, whole, rtol=1e-5, atol=1e-6)


def test_delayed_entry_does_not_accumulate_pre_entry_survival():
    targets, _ = _targets(event=6, entry=4, slot=1)
    head = torch.zeros(1, 8, 3)
    head[0, :, 1] = torch.arange(8, dtype=torch.float32)   # slot 1 enters at 4, merges at 6
    term, value, prep = _run(targets, head)
    z = head[0, :, 1]
    m = torch.cummax(z, 0).values
    expect = -(torch.log(torch.sigmoid(m[6]) - torch.sigmoid(m[5])))
    # normalised by the two cars in the ramp lane at some point (slots 0, 1)
    assert torch.isfinite(value.value).all()
    assert merge_first_passage_metrics(prep)["merge_fp_events"] == 1.0
    assert float(expect) > 0.0


def test_compatibility_censors_at_last_compatible_step():
    targets, _ = _targets(event=5)
    head = torch.zeros(1, 8, 3)
    head[0, :, 0] = torch.arange(8, dtype=torch.float32)
    positions = targets.states[:, :, :, 0].clone()
    positions[:, 3:, 0] += 100.0                 # ego diverges from the record after step 2
    _, value, prep = _run(targets, head, positions=positions)
    metrics = merge_first_passage_metrics(prep)
    assert metrics["merge_fp_censored_divergence"] >= 1.0
    assert metrics["merge_fp_events"] == 0.0
    assert torch.isfinite(value.value).all()


def test_early_model_merge_is_censored_with_the_recorded_survival():
    targets, lanes = _targets(event=5)
    head = torch.zeros(1, 8, 3)
    head[0, :, 0] = torch.arange(8, dtype=torch.float32)
    model_lanes = lanes.clone()
    model_lanes[:, 1:, 0] = 1                    # model merges slot 0 at step 0
    _, value, prep = _run(targets, head, lanes=model_lanes)
    metrics = merge_first_passage_metrics(prep)
    assert metrics["merge_fp_early_merges"] == 1.0
    assert metrics["merge_fp_events"] == 0.0
    z = head[0, :, 0]
    m = torch.cummax(z, 0).values
    expect = -(torch.log1p(-torch.sigmoid(m[0])))          # survival at the early merge
    assert float(value.value) > 0.0
    assert float(expect) > 0.0


# ------------------------------------------------------------------ flags / config
def test_default_switch_is_off_and_flat_keys_absent():
    flat = TrainConfig().to_legacy_flat()
    assert not set(NEW_FLAT_KEYS) & set(flat)
    settings = TrainSettings()
    assert settings.merge_decision == "hazard" and not settings.merge_first_passage_on
    assert "merge_first_passage" not in build_loss(settings).names
    cfg = TrainConfig()
    assert cfg.model.merge_decision == "hazard"
    assert cfg.loss.merge_first_passage.weight == 0.0


def test_config_and_model_wiring():
    cfg = TrainConfig.from_legacy_flat({
        "lane_model": "hazard_v5", "rollout_mode": "hybrid",
        "merge_decision": "threshold", "merge_first_passage_weight": 1.0,
        "merge_first_passage_rank_weight": 0.5, "merge_first_passage_metrics": True})
    assert cfg.model.merge_decision == "threshold"
    assert cfg.loss.merge_first_passage.rank_weight == 0.5
    settings = TrainSettings.from_config(cfg)
    assert settings.merge_first_passage_on
    assert "merge_first_passage" in build_loss(settings).names
    spec = ModelSpec.from_mapping(cfg.to_legacy_flat())
    assert spec.merge_decision == "threshold"
    model = build_model(spec, PhysicsConfig())
    assert model.merge_threshold is not None and model.merge_decision == "threshold"
    with pytest.raises(ValueError):
        TrainConfig.from_legacy_flat({"merge_decision": "threshold"})   # residual lane model
    with pytest.raises(ValueError):
        ModelSpec(merge_decision="bogus")
    with pytest.raises(ValueError):
        TrainSettings(merge_decision="bogus")


def test_cpu_two_step_wiring_proof():
    cfg = load_train_config("configs/c4_wm.toml", [
        "runtime.device=cpu", 'model.lane_model="hazard_v5"',
        "physics.lane_learnable=true", 'model.merge_decision="threshold"',
        "loss.merge_first_passage.weight=1.0", "loss.merge_first_passage.rank_weight=0.5",
        "loss.merge_first_passage.metrics=true"])
    flat = cfg.to_legacy_flat()
    settings = TrainSettings.from_flat(flat)
    physics = PhysicsConfig(sumo_fidelity=True, lane_learnable=True,
                            lane_rule="learned_discretionary", lane_gap_rule="speed")
    torch.manual_seed(0)
    model = build_model(ModelSpec.from_mapping(flat), physics)
    assert model.merge_threshold is not None
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
        assert rec["merge_fp_n_cars"] > 0.0
        assert math.isfinite(rec["merge_fp_nll"])
        optim.step()
        losses.append(rec["loss"])
    assert all(math.isfinite(x) for x in losses)
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.merge_threshold.parameters())
    assert rec["merge_fp_events"] >= 0.0
