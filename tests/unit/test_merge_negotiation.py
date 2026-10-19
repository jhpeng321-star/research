"""V10NEGOT: merge-negotiation block, censored survival loss, config plumbing."""
from __future__ import annotations

import pytest
import torch

from wms.config.train import TrainConfig
from wms.model import ModelSpec, WmSModel
from wms.model.merge_negotiation import (MergeNegotiationNet, N_PAIR_FEATURES,
                                         pair_features, scatter_follower_accel)
from wms.physics.config import PhysicsConfig
from wms.train.losses.base import LossContext, UnrollTargets
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
                               merge_teacher_frac=1.0, **kwargs)
        free = model.rollout(state, n_steps=4, **kwargs)
        zero = model.rollout(state, n_steps=4, merge_lane_teacher=teacher,
                             merge_teacher_frac=0.0, **kwargs)
    assert torch.equal(forced["lanes"][0, 0][ramp[0]], torch.ones(int(ramp[0].sum()),
                                                                 dtype=torch.long))
    assert torch.equal(forced["lanes"][:, :, other[0]], free["lanes"][:, :, other[0]])
    assert torch.equal(free["lanes"], zero["lanes"])

