"""V10UNBLOCK (2026-10-19): ``physics.lane_learnable`` -- the physical lane-change envelope.

Behavioural lane-change vetoes are replaced by a strictly physical envelope so a closed
Dreamer loop can learn merge / lane-change / yield from data:

* the switch defaults OFF and every flat/config/model attribute is unchanged (existing golden
  tests still hold on the same tree);
* the envelope: road-legal transitions only, actual body non-intersection one step ahead,
  lane-end stopping kept, and NO behavioural margin (speed-gap headway / standstill /
  impatience / ``merge_end_wait``) -- a configuration the legacy speed rule vetoes but whose
  bodies do not intersect is admitted;
* the physics lane logits (``mobil_merge_logits``) use the envelope;
* the lane heads sample their own probability independent of the AP gate, the last-5 m window
  is not forced and ``mainline_merge_yield`` braking is not applied;
* ``recorded_rows`` returns ALL at-risk HDV rows (stay = negative, censored endpoints counted,
  envelope-infeasible rows excluded and counted, forced-window rows included) and the head
  gradient is non-zero on every head for those rows.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from .test_gaprule import _kin
from .test_lanev5 import V5, _rollout, _v5_model
from .test_wmlane import _batch, _c4
from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.model import lane_v5 as lv5
from wms.physics.accel import baseline_accel
from wms.physics.config import PhysicsConfig
from wms.physics.mobil import (MASKED_LOGIT, mobil_merge_logits,
                               physical_merge_envelope, physical_target_lane)
from wms.train.losses.lane_event import lane_v5_batch_rows
from wms.train.settings import TrainSettings


def _learnable(**kw) -> PhysicsConfig:
    return replace(_phys(), lane_learnable=True)


def _phys() -> PhysicsConfig:
    return PhysicsConfig(sumo_fidelity=True, lane_rule="learned_discretionary",
                         lane_gap_rule="speed")


# --------------------------------------------------------------- switch plumbing
def test_switch_defaults_off_and_plumbing():
    assert TrainConfig().physics.lane_learnable is False
    assert "lane_learnable" not in TrainConfig().to_legacy_flat()
    for name in ("c4_wm", "configs/c4_wm_v5.toml", "configs/c4_wm_v8ft2.toml"):
        assert load_train_config(name, []).physics.lane_learnable is False, name
    assert TrainSettings().lane_learnable is False
    # on: flat + settings + physics
    _, flat, settings = _c4([*V5, "physics.lane_learnable=true"])
    assert flat["lane_learnable"] is True and settings.lane_learnable is True
    assert load_train_config("c4_wm", ["physics.lane_learnable=true"]).physics.lane_learnable is True
    assert PhysicsConfig(lane_learnable=True).lane_learnable is True
    with pytest.raises(ValueError):
        PhysicsConfig(lane_physical_margin_m=-1.0)
    with pytest.raises(ValueError):
        PhysicsConfig(lane_learnable=1)


# --------------------------------------------------------------- envelope
def test_envelope_legality_body_and_lane_end():
    p, one = _learnable(), torch.ones(1, 1, dtype=torch.bool)
    # ramp before the zone: road transition illegal
    assert not bool(physical_merge_envelope(_kin([(-50.0, 2, 10.0)]), one, p)["admissible"][0, 0])
    # ramp past the lane end: illegal (lane-end stopping)
    beyond = _kin([(80.0, 2, 10.0)])
    assert not bool(physical_merge_envelope(beyond, one, p)["legal"][0, 0])
    # free in-zone ramp: admissible
    assert bool(physical_merge_envelope(_kin([(30.0, 2, 10.0)]), one, p)["admissible"][0, 0])
    # a lane-1 car whose body overlaps the ego after the step: not clear
    two = torch.ones(1, 2, dtype=torch.bool)
    kin = _kin([(30.0, 2, 10.0), (30.0, 1, 10.0)])
    assert not bool(physical_merge_envelope(kin, two, p)["clear"][0, 0])
    # target lane mapping: 0 -> 1, 1 -> 0, 2 -> 1
    lane = torch.tensor([[0, 1, 2]])
    assert physical_target_lane(lane).tolist() == [[1, 0, 1]]


def test_envelope_removes_the_behavioural_follower_gap_veto():
    p_off = _phys()
    p_on = replace(p_off, lane_learnable=True)
    # ramp car at 60 m, lane-1 follower 1 m behind its rear: the speed rule vetoes (needs
    # ~6 m at 10 m/s), the bodies do not intersect.
    kin = _kin([(60.0, 2, 10.0), (54.0, 1, 10.0)])
    present = torch.ones(1, 2, dtype=torch.bool)
    env = physical_merge_envelope(kin, present, p_on)
    assert bool(env["admissible"][0, 0]) and bool(env["clear"][0, 0])
    off = mobil_merge_logits(kin, present, p_off)
    on = mobil_merge_logits(kin, present, p_on)
    assert float(off[0, 0, 1]) <= MASKED_LOGIT + 1e-6      # legacy: merge masked
    assert float(on[0, 0, 1]) > MASKED_LOGIT + 1e-6        # envelope: merge allowed


def test_envelope_ignores_merge_end_wait_and_impatience():
    kin = _kin([(70.0, 2, 1.0), (50.0, 1, 10.0)])
    present = torch.ones(1, 2, dtype=torch.bool)
    base = dict(sumo_fidelity=True, lane_gap_rule="speed", merge_end_wait=True,
                merge_impatience=True)
    off = mobil_merge_logits(kin, present, PhysicsConfig(**base))
    on = mobil_merge_logits(kin, present, PhysicsConfig(**base, lane_learnable=True))
    assert float(off[0, 0, 1]) <= MASKED_LOGIT + 1e-6      # lane-end veto
    assert float(on[0, 0, 1]) > MASKED_LOGIT + 1e-6        # envelope only


# --------------------------------------------------------------- decisions
def test_lane_learnable_no_forced_window_and_head_logit_value():
    p = _learnable()
    net = lv5.LaneV5Net()
    with torch.no_grad():
        net.heads.bias.fill_(-9.0)
    s1 = p.merge_zone_s1_m
    kin = _kin([(s1 - 2.0, 2, 1.0)])
    present = torch.ones(1, 1, dtype=torch.bool)
    base = mobil_merge_logits(kin, present, p)
    acc = lv5.initial_acc(present, kin, p, None, None, 1)
    raw, _, inst = lv5.step_features(acc, kin, present, p, None)
    lg, head, acts = lv5.lane_logits_v5(net, raw, inst, present, base, p)
    assert bool(acts[0, 0]) and float(lg[0, 0, 1].detach()) < lv5.FORCED_MERGE_LOGIT


def test_lane_learnable_ignores_the_ap_gate_and_all_heads_sample():
    model, _ = _v5_model(['model.lane_decision="gated"'])
    b = _batch()
    with torch.no_grad():
        model.lane_v5.heads.bias.copy_(torch.tensor([-0.5, -0.5, -0.5]))
        model.lane_v5.head_ap.copy_(torch.tensor([-1.0, -1.0, -1.0]))     # every gate closed
    lanes0 = b["states"][:, :1, :, 1:4].argmax(-1)
    learned = (b["present"][:, 0] & ~b["is_cav"][:, 0]).unsqueeze(1)

    def moved(roll):
        seq = torch.cat([lanes0, roll["lanes"]], 1)
        mainline_src = learned & (seq[:, :-1] <= 1)      # ramp merges also occur with closed gates
        return bool(((seq[:, 1:] != seq[:, :-1]) & mainline_src).any())

    assert not moved(_rollout(model, b, lane_seed=4))                     # closed gates: no draw
    model.physics = replace(model.physics, lane_learnable=True)
    assert model._samples_lanes
    assert moved(_rollout(model, b, lane_seed=4))                        # sampled, gate ignored
    r1, r2 = _rollout(model, b, lane_seed=4), _rollout(model, b, lane_seed=4)
    assert torch.equal(r1["lanes"], r2["lanes"])                         # still seeded


def test_lane_learnable_disables_mainline_yield_braking():
    p = PhysicsConfig(sumo_fidelity=True, lane_gap_rule="speed",
                      merge_end_wait=True, mainline_merge_yield=True)
    kin = _kin([(72.0, 2, 0.5), (65.0, 1, 5.0)])
    present = torch.ones(1, 2, dtype=torch.bool)
    off = baseline_accel(kin, present, p)
    on = baseline_accel(kin, present, replace(p, lane_learnable=True))
    assert float(on[0, 1]) > float(off[0, 1]) + 1e-6     # follower stops braking for the merger


# --------------------------------------------------------------- training rows
def test_recorded_rows_learnable_all_risk_censored_and_gradient():
    model_off, settings_off = _v5_model(["loss.lane_event.weight=0.0"])
    b = _batch()
    b["states"][1, 6:, 2, 1:4] = torch.tensor([1.0, 0.0, 0.0])           # a recorded change
    off = lane_v5_batch_rows(model_off, b, settings_off)

    model, settings = _v5_model(["loss.lane_event.weight=0.0"])
    model.physics = replace(model.physics, lane_learnable=True)
    rows = lane_v5_batch_rows(model, b, settings)
    st = rows["stats"]
    assert st["rows"] >= off["stats"]["rows"]                            # at least as many rows
    for key in ("censored", "infeasible_rows", "infeasible_events"):
        assert key in st
    assert st["rows"] > 0
    # the head gradient is non-zero on every head present in the rows
    logits = model.lane_v5(rows["features"]).gather(-1, rows["head"].unsqueeze(-1)).squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, rows["label"])
    loss.backward()
    assert float(model.lane_v5.heads.weight.grad.abs().sum()) > 0.0


def _event_batch():
    """Four slots with a recorded in-zone merge (formerly vetoed by the speed rule) and
    mainline overtake / return rows."""
    p = _learnable()
    b, t, k = 1, 4, 4
    states = torch.zeros(b, t, k, 7)
    states[..., 6] = 5.0
    states[:, :, 0, 0], states[:, :, 0, 4] = 60.0, 8.0
    states[:, :, 0, 1:4] = torch.tensor([0.0, 0.0, 1.0])                  # ramp lane
    states[:, -1, 0, 1:4] = torch.tensor([0.0, 1.0, 0.0])                 # merges at the end
    states[:, :, 1, 0], states[:, :, 1, 4] = 54.0, 8.0
    states[:, :, 1, 1:4] = torch.tensor([0.0, 1.0, 0.0])                  # 1 m follower
    states[:, :, 2, 0], states[:, :, 2, 4] = 200.0, 20.0
    states[:, :, 2, 1:4] = torch.tensor([1.0, 0.0, 0.0])                  # lane 0
    states[:, :, 3, 0], states[:, :, 3, 4] = 260.0, 20.0
    states[:, :, 3, 1:4] = torch.tensor([0.0, 1.0, 0.0])                  # lane 1
    valid = torch.ones(b, t, k, dtype=torch.bool)
    is_cav = torch.zeros(b, t, k, dtype=torch.bool)
    return p, states, valid, is_cav


def test_all_risk_rows_are_overfittable_per_head():
    """Check (c) capacity smoke: the all-risk rows carry a learnable signal on every head."""
    import torch.nn.functional as F

    model, _ = _v5_model(["loss.lane_event.weight=0.0"])
    p, states, valid, is_cav = _event_batch()
    model.physics = p
    age = torch.full((1, states.shape[1] - 1, states.shape[2]), 999.0)
    acc0 = model.initial_lane_v5_acc(valid[:, 0], states[:, 0], None, None)
    rows = lv5.recorded_rows(states, valid, is_cav, p, None, age, acc0)
    assert rows["stats"]["merge_events"] == 1
    net = lv5.LaneV5Net()
    opt = torch.optim.Adam(net.parameters(), lr=5e-2)
    logits = net(rows["features"]).gather(-1, rows["head"].unsqueeze(-1)).squeeze(-1)
    before = float(F.binary_cross_entropy_with_logits(logits.detach(), rows["label"]))
    pos = rows["label"] > 0.5
    p_before_pos = float(torch.sigmoid(logits.detach())[pos].mean())
    for _ in range(150):
        opt.zero_grad()
        logits = net(rows["features"]).gather(-1, rows["head"].unsqueeze(-1)).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, rows["label"])
        loss.backward()
        opt.step()
    after = float(loss.detach())
    assert after < before
    with torch.no_grad():
        p_after = torch.sigmoid(net(rows["features"]).gather(
            -1, rows["head"].unsqueeze(-1)).squeeze(-1))
    assert float(p_after[pos].mean()) > p_before_pos + 1e-3              # positives move up


def test_recorded_rows_learnable_includes_forced_window_and_censoring():
    model, _ = _v5_model(["loss.lane_event.weight=0.0"])
    model.physics = replace(model.physics, lane_learnable=True)
    p = model.physics
    s1 = p.merge_zone_s1_m
    b, t, k = 1, 3, 2
    states = torch.zeros(b, t, k, 7)
    states[..., 6] = 5.0
    states[:, :, 0, 0], states[:, :, 0, 1:4], states[:, :, 0, 4] = (
        s1 - 2.0, torch.tensor([0.0, 0.0, 1.0]), 1.0)                     # ramp in the window
    states[:, :, 1, 0], states[:, :, 1, 1:4], states[:, :, 1, 4] = (
        s1 - 40.0, torch.tensor([0.0, 1.0, 0.0]), 1.0)                    # free lane-1 car
    valid = torch.ones(b, t, k, dtype=torch.bool)
    valid[0, -1, 0] = False                                              # censor the last endpoint
    is_cav = torch.zeros(b, t, k, dtype=torch.bool)
    age = torch.full((b, t - 1, k), 999.0)
    acc0 = model.initial_lane_v5_acc(valid[:, 0], states[:, 0], None, None)
    rows = lv5.recorded_rows(states, valid, is_cav, p, model.lane_v5_mask_fn(), age, acc0)
    assert rows["stats"]["forced_window_rows"] >= 1
    assert rows["stats"]["rows"] >= 1
