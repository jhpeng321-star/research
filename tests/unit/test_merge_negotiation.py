"""V10NEGOT: merge-negotiation block, censored survival loss, config plumbing."""
from __future__ import annotations

import pytest
import torch

from wms.config.train import TrainConfig
from wms.model import ModelSpec, WmSModel
from wms.model.merge_negotiation import (MergeNegotiationNet, N_EXT_FEATURES, N_PAIR_FEATURES,
                                         pair_features, scatter_follower_accel)
from wms.physics.config import PhysicsConfig
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.merge_negotiation import (MergeNegotiationLossConfig,
                                                MergeNegotiationTerm, MergeYieldLossConfig,
                                                RampAdaptHuberTerm, SafetyBCETerm,
                                                YieldHuberTerm)
from wms.train.losses.merge_timing import MergeTimingConfig, MergeTimingTerm
from wms.train.settings import TrainSettings
from wms.train.inputs import state_from_kin

from .test_gaprule import _kin
from .test_lanefix import _initial
from .test_wmlane import _batch, _c4


def _learnable() -> PhysicsConfig:
    return PhysicsConfig(sumo_fidelity=True, lane_learnable=True,
                         lane_rule="learned_discretionary", lane_gap_rule="speed",
                         merge_zone_s1_m=75.44, merge_zone_s0_m=0.0)


def _v5_flat(*, negotiation: bool):
    _, flat, _ = _c4(['model.lane_model="hazard_v5"', "physics.lane_learnable=true"])
    flat = dict(flat)
    flat["merge_negotiation"] = bool(negotiation)
    return flat


def test_pair_features_ramp_only_and_follower_index():
    physics = _learnable()
    kin = _kin([(40.0, 2, 8.0), (20.0, 1, 14.0), (100.0, 1, 10.0)])
    present = kin[..., 1:4].sum(-1) > 0
    out = pair_features(kin, present, physics, merge_prob=torch.tensor([[0.0, 0.7, 0.0]]))
    assert out["features"].shape == (1, 3, N_PAIR_FEATURES)
    assert out["ramp"].tolist() == [[True, False, False]]
    assert out["valid_pair"].tolist() == [[True, False, False]]
    assert int(out["follower_index"][0, 0]) == 1
    feats = out["features"][0, 0]
    assert feats[0] == 1.0                       # ramp flag
    assert feats[6] == 1.0                       # follower present
    assert feats[7] > 0.0                        # follower gap
    assert out["features"][0, 1].abs().max() == 0.0
    assert out["features"][0, 2].abs().max() == 0.0


def test_zero_initialised_negotiation_is_an_exact_noop():
    torch.manual_seed(0)
    base_spec = ModelSpec.from_mapping(_v5_flat(negotiation=False))
    on_spec = ModelSpec.from_mapping(_v5_flat(negotiation=True))
    base = WmSModel(base_spec, physics=_learnable()).eval()
    on = WmSModel(on_spec, physics=_learnable()).eval()
    on.load_state_dict(base.state_dict(), strict=False)
    assert on.merge_negotiation is not None
    assert on.merge_negotiation.net[-1].weight.abs().max().item() == 0.0
    b = _batch()
    state = _initial(b)
    kwargs = dict(driver_history=b["driver_hist"],
                  driver_history_mask=b["driver_hist_mask"])
    with torch.no_grad():
        a = base.forward(state, **kwargs)
        c = on.forward(state, **kwargs)
    assert torch.equal(a["accel_mps2"], c["accel_mps2"])
    assert torch.equal(a["lane_logits"], c["lane_logits"])
    assert c["merge_negotiation_yield_logit"].abs().max().item() == 0.0
    assert c["merge_negotiation_merge_delta"].abs().max().item() == 0.0


def test_nonzero_negotiation_moves_the_follower_and_merge_logit():
    physics = _learnable()
    kin = _kin([(40.0, 2, 8.0), (20.0, 1, 14.0)])
    present = kin[..., 1:4].sum(-1) > 0
    feats = pair_features(kin, present, physics, merge_prob=torch.tensor([[0.0, 0.8]]))
    net = MergeNegotiationNet()
    with torch.no_grad():
        net.net[-1].weight[0].fill_(1.0)
        net.net[-1].weight[1].fill_(-1.0)
        net.net[-1].bias.copy_(torch.tensor([2.0, 3.0]))
    out = net(feats["features"], follower_index=feats["follower_index"],
              valid_pair=feats["valid_pair"], ramp=feats["ramp"])
    assert out["yield_accel"][0, 0] < 0.0        # follower yields
    assert out["merge_logit_delta"][0, 0] > 0.0  # ramp merge logit opens
    moved = scatter_follower_accel(out["yield_accel"], out["follower_index"],
                                   out["valid_pair"], k=2)
    assert moved[0, 1] < 0.0 and moved[0, 0] == 0.0


def _survival_targets():
    b, t, k = 1, 2, 3
    states = torch.zeros(b, t + 1, k, 7)
    lanes = torch.tensor([[[2, 2, 0], [2, 2, 0], [1, 2, 0]]])   # car 0 merges, car 1 stuck
    states[..., 1:4] = torch.nn.functional.one_hot(lanes, 3).float()
    states[..., 0] = torch.tensor([[[50.0, 30.0, 0.0], [55.0, 30.0, 0.0], [60.0, 30.0, 0.0]]])
    states[..., 4] = 10.0
    present = torch.ones(b, t + 1, k, dtype=torch.bool)
    present[..., 2] = False
    return UnrollTargets(
        states=states, n_steps=t, weights=torch.ones(1, t, 1),
        hdv_mask=present[:, 1:], aux_mask=present[:, :t], aux_count=int(present[:, :t].sum()),
        denom=torch.tensor(float(present[:, 1:].sum())), lane_target=lanes[:, 1:].argmax(-1),
        loss_valid=present, is_cav=torch.zeros_like(present), dt_s=0.5,
        position_delta_m=5.0, speed_delta_mps=1.0, history_lane_head=False)


def _negotiation_targets(n_steps: int, k: int, seed: int = 0) -> UnrollTargets:
    g = torch.Generator().manual_seed(seed)
    b = 3
    states = torch.zeros(b, n_steps + 1, k, 7)
    lanes = torch.randint(-1, 3, (b, n_steps + 1, k), generator=g)
    onehot = torch.zeros(b, n_steps + 1, k, 3)
    keep = lanes >= 0
    onehot[keep] = torch.nn.functional.one_hot(lanes[keep], 3).float()
    states[..., 1:4] = onehot
    pos = torch.randint(0, 150, (b, n_steps + 1, k), generator=g).float()
    # Ties: two lane-1 vehicles at the same position -> equal gaps for the min reduction.
    lanes[2, :, 2] = 1
    lanes[2, :, 3] = 1
    onehot[2, :, 2] = torch.tensor([0.0, 1.0, 0.0])
    onehot[2, :, 3] = torch.tensor([0.0, 1.0, 0.0])
    states[2, :, :, 1:4] = onehot[2]
    pos[2, :, 2] = pos[2, :, 3]
    states[..., 0] = pos
    states[..., 5] = torch.rand(b, n_steps + 1, k, generator=g) * 4.0 - 2.5
    present = torch.rand(b, n_steps + 1, k, generator=g) > 0.15
    present[0] = False                                   # empty mask batch row
    return UnrollTargets(
        states=states, n_steps=n_steps, weights=torch.ones(1, n_steps, 1),
        hdv_mask=present[:, 1:], aux_mask=present[:, :n_steps],
        aux_count=int(present[:, :n_steps].sum()),
        denom=torch.tensor(float(present[:, 1:].sum())),
        lane_target=lanes[:, 1:].clamp_min(0), loss_valid=present,
        is_cav=torch.zeros_like(present), dt_s=300.0,
        position_delta_m=5.0, speed_delta_mps=1.0, history_lane_head=False)


def _old_prepare(targets: UnrollTargets, cfg: MergeNegotiationLossConfig) -> dict:
    """The pre-chunking dense [B, T, K, K] implementation, kept as the reference."""
    n = targets.n_steps
    rec = targets.states[:, :n + 1].float()
    k = rec.shape[2]
    onehot = rec[..., 1:4]
    lane = onehot.argmax(-1)
    lanes = torch.where(onehot.sum(-1) > 0, lane, torch.full_like(lane, -1))
    valid = targets.loss_valid[:, :n + 1, :k].bool() & ~targets.is_cav[:, :n + 1, :k]
    s, a = rec[..., 0], rec[..., 5]
    ramp = valid[:, :n] & (lanes[:, :n] == cfg.merge_lane)
    near = ramp & ((75.44 - s[:, :n]) < float(cfg.dend_m))
    target = valid[:, :n] & (lanes[:, :n] == cfg.target_lane)
    ds = s[:, :n].unsqueeze(-1) - s[:, :n].unsqueeze(-2)
    behind = target.unsqueeze(-2) & (ds >= 0.0) & (ds <= float(cfg.behind_m))
    gap = torch.where(behind, ds, torch.full_like(ds, float("inf")))
    gap_min, idx = gap.min(-1)
    has = torch.isfinite(gap_min) & near
    safe_idx = idx.clamp(0, max(k - 1, 0))
    f_acc = a[:, 1:n + 1].gather(-1, safe_idx)
    yield_label = (f_acc < float(cfg.yield_decel_mps2)) & has
    merged = (lanes[:, 1:n + 1] == cfg.target_lane) & ramp
    return {"valid": has, "yield": yield_label, "merged": merged,
            "n_valid": float(has.sum()), "n_pos": float(yield_label.sum()),
            "n_merge_pos": float((yield_label & merged).sum()),
            "n_neg": float((has & ~yield_label).sum())}


@pytest.mark.parametrize("n_steps", [23, 0])
def test_chunked_prepare_is_bit_identical_to_dense(n_steps):
    for seed in (0, 1):
        targets = _negotiation_targets(n_steps, k=6, seed=seed)
        cfg = MergeNegotiationLossConfig(weight=1.0)
        want = _old_prepare(targets, cfg)
        for chunk in (1, 5, 7, 50, 1000):
            term = MergeNegotiationTerm(MergeNegotiationLossConfig(
                weight=1.0, pair_chunk_steps=chunk))
            term.prepare(targets)
            prep = targets.prepared["merge_negotiation"]
            for key in ("valid", "yield", "merged"):
                assert torch.equal(prep[key], want[key]), (seed, chunk, key)
            for key in ("n_valid", "n_pos", "n_merge_pos", "n_neg"):
                assert prep[key] == want[key], (seed, chunk, key)
            targets.prepared.clear()


def test_survival_loss_is_correctly_censored():
    targets = _survival_targets()
    term = MergeTimingTerm(MergeTimingConfig(weight=1.0, metrics=True))
    term.prepare(targets)
    prep = targets.prepared["merge_timing"]
    assert prep["event"].shape == (1, 2, 3)
    assert prep["event"][0, 1, 0]                 # car 0 leaves lane 2 at step 1
    assert not prep["event"][0, 0, 0]
    assert prep["survive"][0, 0, 0] and prep["survive"][0, 0, 1]
    assert prep["rec_merge"][0, 1].item() == -1   # car 1 is right-censored
    assert prep["n_vehicles"] == 2.0
    pred = {"lanes": torch.tensor([[[2, 2, 0], [1, 2, 0]]]),
            "masks": torch.ones(1, 2, 3, dtype=torch.bool),
            "positions": targets.states[:, 1:, :, 0],
            "lane_logits": torch.zeros(1, 2, 3, 3)}
    pred["lane_logits"][..., 1] = 1.0
    pred["lane_logits"][..., 0] = 0.0
    ctx = LossContext(pred=pred, current=state_from_kin(targets.states[:, 0],
                                                        targets.loss_valid[:, 0]),
                      targets=targets, c0=0, c1=2)
    value = term(ctx)
    assert value is not None and torch.isfinite(value.logged)


def test_config_keys_default_off_and_round_trip():
    default = TrainConfig()
    assert default.model.merge_negotiation is False
    assert default.loss.merge_survival.weight == 0.0
    assert default.curriculum.merge_teacher_frac_schedule == ()
    flat = default.to_legacy_flat()
    assert not any(k in flat for k in ("merge_negotiation", "merge_survival_weight",
                                       "merge_teacher_frac_schedule"))
    cfg = TrainConfig.from_legacy_flat({
        "lane_model": "hazard_v5", "rollout_mode": "hybrid",
        "merge_negotiation": True,
        "merge_survival_weight": 1.5, "merge_teacher_frac_schedule": [0.5, 0.0]})
    assert cfg.model.merge_negotiation is True
    assert cfg.loss.merge_survival.weight == 1.5
    assert cfg.curriculum.merge_teacher_frac_schedule == (0.5, 0.0)
    settings = TrainSettings.from_config(cfg)
    assert settings.merge_negotiation_on is False   # needs negative_weight > 0
    assert settings.merge_timing_on is True
    with pytest.raises(ValueError):
        TrainConfig.from_legacy_flat({"merge_teacher_frac_schedule": [1.5]})
    with pytest.raises(ValueError):
        TrainConfig.from_legacy_flat({"lane_model": "residual", "merge_negotiation": True})


def test_model_spec_requires_hazard_v5():
    with pytest.raises(ValueError):
        ModelSpec(merge_negotiation=True, lane_model="residual")


def test_merge_teacher_forces_only_ramp_cars():
    torch.manual_seed(0)
    model = WmSModel(ModelSpec.from_mapping(_v5_flat(negotiation=True)),
                     physics=_learnable()).eval()
    b = _batch()
    state = _initial(b)
    kwargs = dict(driver_history=b["driver_hist"],
                  driver_history_mask=b["driver_hist_mask"],
                  cav_teacher=b["cav_teacher"][:, :5])
    recorded = b["states"][..., 1:4].argmax(-1)
    present0 = b["present"][:, 0]
    ramp = (recorded[:, 0] == int(model.physics.merge_lane)) & present0
    assert bool(ramp.any())
    teacher = recorded.clone()
    teacher[:, 1:][ramp.unsqueeze(1).expand_as(teacher[:, 1:])] = 1
    other = ~ramp & present0
    teacher[:, 1:][other.unsqueeze(1).expand_as(teacher[:, 1:])] = 0
    with torch.no_grad():
        forced = model.rollout(state, n_steps=4, merge_lane_teacher=teacher,
                               merge_teacher_frac=1.0, lane_seed=7, **kwargs)
        free = model.rollout(state, n_steps=4, lane_seed=7, **kwargs)
        zero = model.rollout(state, n_steps=4, merge_lane_teacher=teacher,
                             merge_teacher_frac=0.0, lane_seed=7, **kwargs)
    assert torch.equal(forced["lanes"][0, 0][ramp[0]], torch.ones(int(ramp[0].sum()),
                                                                 dtype=torch.long))
    assert torch.equal(forced["lanes"][:, :, other[0]], free["lanes"][:, :, other[0]])
    assert torch.equal(free["lanes"], zero["lanes"])


def test_extended_pair_features_include_detached_intent_trend_and_next_gap():
    physics = _learnable()
    kin = _kin([(40.0, 2, 8.0), (20.0, 1, 14.0), (10.0, 1, 12.0), (100.0, 1, 10.0)])
    present = kin[..., 1:4].sum(-1) > 0
    merge_logit = torch.tensor([[1.0, 0.0, -1.0, 0.0]])
    prev_logit = torch.tensor([[0.5, 0.0, -2.0, 0.0]])
    out = pair_features(
        kin, present, physics, merge_prob=torch.tensor([[0.0, 0.7, 0.2, 0.1]]),
        merge_logit=merge_logit, prev_merge_logit=prev_logit)
    assert out["ext_features"].shape == (1, 4, N_EXT_FEATURES)
    ext = out["ext_features"][0, 0]
    assert ext[N_PAIR_FEATURES].item() == pytest.approx(1.0 / 10.0)
    assert ext[N_PAIR_FEATURES + 1].item() == pytest.approx((1.0 - 0.5) / 5.0)
    assert ext[N_PAIR_FEATURES + 2].item() == 1.0          # next-behind present
    assert ext[N_PAIR_FEATURES + 3].item() > 0.0           # next-behind gap
    # The frozen base features are unchanged by the extension.
    assert torch.equal(out["features"], out["ext_features"][..., :N_PAIR_FEATURES])


def test_new_heads_receive_gradients():
    torch.manual_seed(0)
    physics = _learnable()
    kin = _kin([(40.0, 2, 8.0), (20.0, 1, 14.0), (10.0, 1, 12.0)])
    present = kin[..., 1:4].sum(-1) > 0
    feats = pair_features(kin, present, physics, merge_prob=torch.tensor([[0.0, 0.8, 0.2]]),
                          merge_logit=torch.tensor([[0.0, 1.0, 0.0]]))
    net = MergeNegotiationNet()
    out = net(feats["features"], ext_features=feats["ext_features"],
              follower_index=feats["follower_index"], valid_pair=feats["valid_pair"],
              ramp=feats["ramp"])
    loss = (out["yield_accel"].sum() + out["ramp_accel_delta"].sum()
            + out["required_brake"].sum() + out["conflict_logit"].sum()
            + out["merge_logit_delta"].sum())
    loss.backward()
    for head in (net.yield_head, net.ramp_head, net.safety_head, net.merge_safety_head):
        grad = head[-1].weight.grad
        assert grad is not None and float(grad.abs().sum()) > 0.0


def _yield_targets() -> UnrollTargets:
    b, n, k = 1, 6, 2
    states = torch.zeros(b, n + 1, k, 7)
    lanes = torch.tensor([[[2, 1], [2, 1], [2, 1], [2, 1], [1, 1], [1, 1], [1, 1]]])
    states[..., 1:4] = torch.nn.functional.one_hot(lanes, 3).float()
    states[..., 0] = torch.tensor(
        [[[[50.0, 30.0], [55.0, 35.0], [60.0, 40.0], [65.0, 45.0],
           [70.0, 50.0], [75.0, 55.0], [80.0, 60.0]]]]).squeeze(1)
    states[..., 4] = 10.0
    states[..., 5] = torch.tensor(
        [[[[-0.2, -0.5], [-0.3, -1.5], [-0.4, -2.0], [-0.5, -2.5],
           [-0.6, -3.0], [-0.7, -2.0], [-0.8, -1.0]]]]).squeeze(1)
    states[..., 6] = 5.0
    present = torch.ones(b, n + 1, k, dtype=torch.bool)
    return UnrollTargets(
        states=states, n_steps=n, weights=torch.ones(1, n, 1),
        hdv_mask=present[:, 1:], aux_mask=present[:, :n], aux_count=int(present[:, :n].sum()),
        denom=torch.tensor(float(present[:, 1:].sum())),
        lane_target=lanes[:, 1:].argmax(-1), loss_valid=present,
        is_cav=torch.zeros_like(present), dt_s=0.5, position_delta_m=5.0,
        speed_delta_mps=1.0, history_lane_head=False)


def test_merge_yield_masks_are_recorded_only_and_terms_get_gradients():
    targets = _yield_targets()
    cfg = MergeYieldLossConfig(yield_huber=1.0, ramp_adapt_huber=1.0, safety_bce=1.0,
                               yield_window_s=2.0, gap_window_s=2.0, safety_window_s=1.0)
    terms = [YieldHuberTerm(cfg), RampAdaptHuberTerm(cfg), SafetyBCETerm(cfg)]
    for term in terms:
        term.prepare(targets)
    prep = targets.prepared["merge_yield"]
    assert not prep["yield_mask"].requires_grad
    assert bool(prep["yield_mask"][0, :3, 0].all())       # before the recorded merge
    assert not bool(prep["yield_mask"][0, 4:, 0].any())   # after the recorded merge
    assert bool(prep["safety_pos"][0, 3, 0])              # the recorded merge step
    assert prep["n_premerge"] >= 3.0 and prep["n_safety_pos"] >= 1.0

    pred = {
        "physics_accel_mps2": torch.zeros(1, 6, 2),
        "merge_negotiation_yield_accel": torch.full((1, 6, 2), -0.1, requires_grad=True),
        "merge_negotiation_ramp_delta": torch.full((1, 6, 2), -0.1, requires_grad=True),
        "merge_negotiation_required_brake": torch.full((1, 6, 2), 0.5, requires_grad=True),
        "merge_negotiation_conflict_logit": torch.zeros(1, 6, 2, requires_grad=True),
    }
    ctx = LossContext(pred=pred, current={}, targets=targets, c0=0, c1=6)
    total = 0.0
    for term in terms:
        value = term(ctx)
        assert value is not None and torch.isfinite(value.logged)
        total = total + value.value
    total.backward()
    for key in ("merge_negotiation_yield_accel", "merge_negotiation_ramp_delta",
                "merge_negotiation_required_brake", "merge_negotiation_conflict_logit"):
        assert pred[key].grad is not None
        assert torch.isfinite(pred[key].grad).all()


def test_old_negotiation_state_dict_loads_with_new_heads():
    net = MergeNegotiationNet()
    old = {key: value for key, value in net.state_dict().items()
           if not any(key.startswith(prefix) for prefix in (
               "yield_head.", "ramp_head.", "safety_head.", "merge_safety_head."))}
    fresh = MergeNegotiationNet()
    missing = fresh.load_state_dict(old, strict=False)
    assert not missing.unexpected_keys
    for head in (fresh.yield_head, fresh.ramp_head, fresh.safety_head,
                 fresh.merge_safety_head):
        assert float(head[-1].weight.detach().abs().max()) == 0.0


def test_merge_yield_config_defaults_and_round_trip():
    default = TrainConfig()
    assert default.loss.merge_yield.enabled is False
    flat = default.to_legacy_flat()
    assert not any(key.startswith("merge_yield") or key.startswith("merge_safety")
                   or key.startswith("merge_ramp") or key.startswith("merge_gap")
                   or key.startswith("merge_conflict") for key in flat)
    cfg = TrainConfig.from_legacy_flat({
        "lane_model": "hazard_v5", "rollout_mode": "hybrid",
        "merge_negotiation": True,
        "merge_yield_huber": 0.5, "merge_ramp_adapt_huber": 0.25,
        "merge_safety_bce": 0.4, "merge_yield_window_s": 4.0})
    assert cfg.loss.merge_yield.yield_huber == 0.5
    assert cfg.loss.merge_yield.ramp_adapt_huber == 0.25
    assert cfg.loss.merge_yield.safety_bce == 0.4
    assert cfg.loss.merge_yield.yield_window_s == 4.0
    settings = TrainSettings.from_config(cfg)
    assert settings.merge_yield_on is True
    with pytest.raises(ValueError):
        MergeYieldLossConfig(yield_huber=-1.0)

def test_two_start_two_step_wiring_proof_with_new_losses():
    from dataclasses import replace

    from wms.train.losses.compose import build_loss
    from wms.train.unroll import UnrollRunner
    from .test_wmlane import _batch, _c4, _model

    _, flat, settings = _c4(['model.lane_model="hazard_v5"',
                             "physics.lane_learnable=true"])
    flat = dict(flat)
    flat["merge_negotiation"] = True
    settings = replace(settings, merge_negotiation=True,
                       merge_yield_huber=0.5, merge_ramp_adapt_huber=0.25,
                       merge_safety_bce=0.4, merge_survival_weight=0.0,
                       merge_survival_negative_weight=0.0,
                       aggregate_weight=0.0, agg_delay_weight=0.0,
                       agg_delay_paired_weight=0.0, paired_diff_weight=0.0,
                       lane_flow_weight=0.0, lane_event_weight=0.0,
                       merge_loss_speed_weight=0.0, merge_loss_discharge_weight=0.0,
                       traffic_loss=False)
    model, physics = _model(flat)
    composer = build_loss(settings, physics=physics)
    assert {"yield_huber", "ramp_adapt_huber", "safety_bce"} <= set(composer.names)
    runner = UnrollRunner(model, settings, composer, projection=None,
                          device=torch.device("cpu"))
    model.train()
    rec = runner.run(_batch(), unroll_steps=2, backward=True)
    assert torch.isfinite(torch.tensor(rec["loss"]))
    for name in ("yield_huber", "ramp_adapt_huber", "safety_bce"):
        assert name in rec and torch.isfinite(torch.tensor(rec[name]))
    for head in (model.merge_negotiation.yield_head, model.merge_negotiation.ramp_head,
                 model.merge_negotiation.safety_head,
                 model.merge_negotiation.merge_safety_head):
        assert head[-1].weight.grad is not None
