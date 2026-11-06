"""V8LANE (2026-10-17): mainline overtake / return decisions and the on-policy lane-flow loss.

* every new switch defaults OFF (flat forms, configs, composer, model attribute);
* ``lane_v5.budget_choice``: the population-matched count (systematic accumulator), the
  top-hazard / hazard-proportional pick, the carried accumulator and untouched non-mainline rows;
* ``lane_v5_mainline='sample'`` equals the gated rule with only the mainline gates open, and the
  budget rule runs in a hazard_v5 rollout (seeded, accumulator carried in the state);
* the lane-flow loss: exact pooled Poisson value, gradient sign, occupancy feedback, gradient on a
  hazard_v5 rollout reaching the overtake / return heads only; config plumbing.
"""
from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from .test_lanev5 import V5, _rollout, _v5_model
from .test_v8wm import _ramp_batch
from .test_wmlane import _batch, _c4
from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.model import lane_v5 as lv5
from wms.model.spec import ModelSpec
from wms.train.losses import lane_flow as lf
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.compose import build_loss
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

NEW_FLAT_KEYS = ("lane_v5_mainline", "lane_flow_weight", "lane_flow_balance")


# ----------------------------------------------------------------------- defaults
def test_switches_default_off_and_flat_forms_unchanged():
    assert not set(NEW_FLAT_KEYS) & set(TrainConfig().to_legacy_flat())
    for name in ("c4_wm", "configs/c4_wm_v5.toml", "configs/c4_wm_v7ft.toml",
                 "configs/c4_wm_v8ft_short2.toml"):
        assert not set(NEW_FLAT_KEYS) & set(load_train_config(name).to_legacy_flat()), name
    s = TrainSettings()
    assert s.lane_v5_mainline == "default" and s.lane_flow_weight == 0.0
    assert "lane_flow" not in build_loss(s).names
    assert ModelSpec().lane_v5_mainline == "default"
    model, _ = _v5_model()
    assert model.lane_v5_mainline == "default" and model._mainline_mode == "default"
    assert not model._samples_lanes


# ----------------------------------------------------------------------- budget_choice
def _rows(lanes, logit):
    """One batch row of slots in ``lanes`` with target-lane logit ``logit`` (stay = 0)."""
    lane = torch.tensor([lanes])
    k = lane.shape[1]
    logits = torch.zeros(1, k, 3)
    target = lv5._lookup(lv5.TARGET_OF_LANE, lane)
    logits.scatter_(-1, target.unsqueeze(-1), torch.tensor([logit]).unsqueeze(-1))
    feasible = torch.zeros(1, k, 3, dtype=torch.bool)
    feasible.scatter_(-1, lane.unsqueeze(-1), True)
    feasible.scatter_(-1, target.unsqueeze(-1), True)
    return lane, logits, feasible


def test_budget_choice_count_top_pick_and_carry():
    # four lane-1 cars (overtake) with p = sigmoid(logit), one lane-0 car, one ramp car
    lane, logits, feasible = _rows([1, 1, 1, 1, 0, 2],
                                   [0.0, -1.0, 1.0, -3.0, -2.0, 5.0])
    p = torch.sigmoid(torch.tensor([0.0, -1.0, 1.0, -3.0]))
    learned = torch.ones(1, 6, dtype=torch.bool)
    choice0 = torch.tensor([[1, 1, 1, 1, 0, 1]])          # ordinary rule (ramp car merges)
    u = torch.full((1, 6), 0.5)
    budget = torch.tensor([[0.2, 0.0]])
    choice, new = lv5.budget_choice(choice0, logits, feasible, lane, learned, budget, u, "budget")
    acc = 0.2 + float(p.sum())                              # 0.2 + 1.5 + ... -> floor = 1 or 2
    n = math.floor(acc)
    fired = (choice[0, :4] == 0).nonzero().flatten().tolist()
    assert len(fired) == n
    assert fired == sorted(p.argsort(descending=True)[:n].tolist())   # highest hazards
    assert float(new[0, 0]) == pytest.approx(acc - n, abs=1e-5)
    # return: p = sigmoid(-2) = 0.12 < 1 -> no fire, accumulated
    assert int(choice[0, 4]) == 0
    assert float(new[0, 1]) == pytest.approx(float(torch.sigmoid(torch.tensor(-2.0))), abs=1e-5)
    assert int(choice[0, 5]) == 1                           # non-mainline row untouched
    # CAV / not-learned rows are never fired and keep the ordinary choice
    learned[0, 2] = False
    choice, _ = lv5.budget_choice(choice0, logits, feasible, lane, learned, budget, u, "budget")
    assert int(choice[0, 2]) == 1
    # the expected count over many steps equals sum P (systematic sampling)
    acc_state, total = torch.tensor([[0.37, 0.0]]), 0
    for _ in range(200):
        c, acc_state = lv5.budget_choice(choice0, logits, feasible, lane,
                                         torch.ones(1, 6, dtype=torch.bool), acc_state, u, "budget")
        total += int((c[0, :4] == 0).sum())
    assert abs(total - 200 * float(p.sum())) <= 1.0


def test_budget_prop_uses_the_draws_and_infeasible_rows_never_fire():
    lane, logits, feasible = _rows([1, 1, 1], [2.0, 2.0, 2.0])
    feasible[0, 1, 0] = False                               # car 1: overtake masked
    learned = torch.ones(1, 3, dtype=torch.bool)
    choice0 = lane.clone()
    seen = set()
    for seed in range(20):
        u = torch.rand(1, 3, generator=torch.Generator().manual_seed(seed))
        c, _ = lv5.budget_choice(choice0, logits, feasible, lane, learned,
                                 torch.tensor([[0.0, 0.0]]), u, "budget_prop")
        fired = (c[0] == 0).nonzero().flatten().tolist()
        assert len(fired) == 1 and 1 not in fired           # floor(2 * 0.88) = 1
        seen.add(fired[0])
    assert seen == {0, 2}                                   # identity drawn, not fixed
    # no budget yet -> initialised from the row's first draw
    _, new = lv5.budget_choice(choice0, logits, feasible, lane, learned, None,
                               torch.full((1, 3), 0.25), "budget")
    assert float(new[0, 1]) == pytest.approx(0.25)


# ----------------------------------------------------------------------- rollouts
def test_mainline_sample_equals_gated_with_only_mainline_open():
    model, _ = _v5_model(['model.lane_decision="gated"'])
    model.eval()
    b = _batch()
    with torch.no_grad():
        model.lane_v5.heads.bias.copy_(torch.tensor([-0.5, -0.5, -0.5]))
        model.lane_v5.head_ap.copy_(torch.tensor([0.9, 0.9, -1.0]))       # mainline open
    gated = _rollout(model, b, lane_seed=5)
    with torch.no_grad():
        model.lane_v5.head_ap.copy_(torch.tensor([-1.0, -1.0, -1.0]))     # all closed
    model.lane_v5_mainline = "sample"
    sampled = _rollout(model, b, lane_seed=5)
    assert torch.equal(gated["lanes"], sampled["lanes"])
    model.lane_v5_mainline = "default"
    assert not torch.equal(_rollout(model, b, lane_seed=5)["lanes"], sampled["lanes"])


def test_budget_rollout_seeded_and_carries_the_accumulator():
    model, _ = _v5_model(['model.lane_decision="gated"'])
    model.eval()
    b = _batch()
    with torch.no_grad():
        model.lane_v5.heads.bias.copy_(torch.tensor([-1.0, -1.0, -0.5]))
    model.lane_v5_mainline = "budget"
    assert model._samples_lanes
    r1, r2 = _rollout(model, b, lane_seed=3), _rollout(model, b, lane_seed=3)
    assert torch.equal(r1["lanes"], r2["lanes"])
    acc = r1["state"]["lane_v5_budget"]
    assert acc.shape == (2, 2) and bool(((acc >= 0) & (acc <= 1)).all())
    lanes0 = b["states"][:, 0, :, 1:4].argmax(-1)
    seq = torch.cat([lanes0.unsqueeze(1), r1["lanes"]], 1)
    learned = b["present"][:, 0] & ~b["is_cav"][:, 0]
    moved = (seq[:, 1:] != seq[:, :-1]) & (seq[:, :-1] <= 1) & learned.unsqueeze(1)
    assert int(moved.sum()) >= 1                            # p ~ 0.27 per feasible row
    # truncated continuation: the accumulator is carried in the state
    half = _rollout(model, b, n=6, lane_seed=3)
    assert "lane_v5_budget" in half["state"]


# ----------------------------------------------------------------------- lane-flow loss
def _flow_targets(rec_lanes):
    b, t1, k = rec_lanes.shape
    states = torch.zeros(b, t1, k, 7)
    states[..., 0] = torch.arange(k).float() * 30.0
    states[..., 1:4] = F.one_hot(rec_lanes, 3).float()
    present = torch.ones(b, t1, k, dtype=torch.bool)
    is_cav = torch.zeros(b, t1, k, dtype=torch.bool)
    return UnrollTargets(
        states=states, n_steps=t1 - 1, weights=torch.ones(1, t1 - 1, 1) / (t1 - 1),
        hdv_mask=present[:, 1:], aux_mask=present[:, 1:], aux_count=0, denom=torch.tensor(1.0),
        lane_target=rec_lanes[:, 1:], loss_valid=present, is_cav=is_cav, dt_s=0.5,
        position_delta_m=5.0, speed_delta_mps=1.0, history_lane_head=False)


def _flow_ctx(targets, model_lanes, head, c0=0, c1=None):
    c1 = targets.n_steps if c1 is None else c1
    b, _, k = model_lanes.shape
    logits = torch.zeros(b, c1 - c0, k, 3)
    pred = {"lanes": model_lanes[:, c0 + 1:c1 + 1],
            "masks": torch.ones(b, c1 - c0, k, dtype=torch.bool), "lane_logits": logits,
            "lane_v5_head_logit": head[:, c0:c1],
            "lane_v5_acts": torch.ones(b, c1 - c0, k, dtype=torch.bool)}
    current = {"lane": model_lanes[:, c0], "present": torch.ones(b, k, dtype=torch.bool)}
    return LossContext(pred=pred, current=current, targets=targets, c0=c0, c1=c1)


def test_lane_flow_value_gradient_and_chunks():
    # record: slot 0 overtakes (1 -> 0) after step 2; slot 1 stays in lane 1, slot 2 in lane 0
    rec = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1] * 7, [0] * 7]).T.unsqueeze(0)   # [1, 7, 3]
    targets = _flow_targets(rec)
    model_lanes = torch.tensor([[1] * 7, [1] * 7, [0] * 7]).T.unsqueeze(0)
    head = torch.full((1, 6, 3), -3.0).requires_grad_()
    term = lf.LaneFlowTerm(lf.LaneFlowConfig(weight=1.0))
    term.prepare(targets)
    value = term(_flow_ctx(targets, model_lanes, head)).value
    h = float(torch.sigmoid(torch.tensor(-3.0)))
    e_ovt, e_ret = 12 * h, 6 * h                            # two lane-1 cars, one lane-0 car
    expect = (e_ovt - 1.0 * math.log(e_ovt + lf._EPS)) + (e_ret - 0.0)
    assert float(value) == pytest.approx(expect, rel=1e-5)
    value.backward()
    g = head.grad
    assert float(g[0, :, :2].max()) < 0.0                   # E_ovt = 0.57 < 1 recorded: raise
    assert float(g[0, :, 2].min()) > 0.0                    # no recorded return: lower it
    m = lf.lane_flow_metrics(targets.prepared["lane_flow"])
    assert m["lane_flow_overtake_recorded"] == 1.0 and m["lane_flow_return_recorded"] == 0.0
    # chunked: stats per chunk add up to the whole unroll
    term2 = lf.LaneFlowTerm(lf.LaneFlowConfig(weight=1.0))
    term2.prepare(targets)
    for c0, c1 in ((0, 3), (3, 6)):
        term2(_flow_ctx(targets, model_lanes, head.detach(), c0, c1))
    m2 = lf.lane_flow_metrics(targets.prepared["lane_flow"])
    assert m2["lane_flow_overtake_expected"] == pytest.approx(e_ovt, rel=1e-5)
    assert m2["lane_flow_overtake_recorded"] == 1.0


def test_lane_flow_balance_adds_the_lane0_excess_to_the_return_target():
    rec = torch.tensor([[1] * 5, [1] * 5]).T.unsqueeze(0)
    targets = _flow_targets(rec)
    model_lanes = torch.tensor([[0] * 5, [0] * 5]).T.unsqueeze(0)   # both in lane 0: excess 2
    head = torch.full((1, 4, 2), -2.0)                                 # E_ret = 8 * 0.12 = 0.95
    signs = {}
    for balance in (0.0, 1.0):
        term = lf.LaneFlowTerm(lf.LaneFlowConfig(weight=1.0, balance=balance))
        term.prepare(targets)
        h = head.clone().requires_grad_()
        term(_flow_ctx(targets, model_lanes, h)).value.backward()
        signs[balance] = float(h.grad.sum())
        assert targets.prepared["lane_flow"]["stats"][0]["lane0_excess"] == 2.0 * (balance > 0)
    assert signs[0.0] > 0.0          # no recorded return: lower the return hazard
    assert signs[1.0] < 0.0          # target 0 + 1.0 * 2 > 0.95: raise it
    with pytest.raises(ValueError):
        lf.LaneFlowConfig(weight=-1.0)


def _flow_grads(extra):
    model, settings = _v5_model(extra)
    model.lane_hazard_detach = True
    model.train()
    runner = UnrollRunner(model, settings, build_loss(settings, physics=model.physics),
                          projection=None, device=torch.device("cpu"))
    rec = runner.run(_ramp_batch(), unroll_steps=12, backward=True, truncation_steps=5)
    return model, rec, model.lane_v5.heads.weight.grad.clone()


def test_lane_flow_on_hazard_v5_rollout_reaches_mainline_heads_only():
    _, off_rec, off = _flow_grads([])
    model, rec, on = _flow_grads(["loss.lane_flow.weight=1.0", "loss.lane_flow.balance=0.5"])
    assert model.emit_merge_hazard and "lane_flow" not in off_rec
    assert math.isfinite(rec["lane_flow"]) and rec["lane_flow_overtake_expected"] > 0.0
    assert float((on[0] - off[0]).abs().sum()) > 0.0       # overtake head row
    torch.testing.assert_close(on[2], off[2])               # merge head: only the event loss
    for key in ("position", "speed", "lane_event"):
        if key in off_rec:
            assert rec[key] == off_rec[key], key


def _flow_grads_learnable(extra):
    """V10UNBLOCK: the same hazard_v5 unroll as ``_flow_grads`` with ``lane_learnable`` on
    (every head samples, every present HDV step is an at-risk row)."""
    model, settings = _v5_model(["loss.lane_event.weight=0.0", *extra])
    model.physics = replace(model.physics, lane_learnable=True)
    model.lane_hazard_detach = True
    model.train()
    runner = UnrollRunner(model, settings, build_loss(settings, physics=model.physics),
                          projection=None, device=torch.device("cpu"))
    rec = runner.run(_ramp_batch(), unroll_steps=12, backward=True, truncation_steps=5)
    return model, rec, model.lane_v5.heads.weight.grad


def test_lane_flow_counts_and_gradient_under_lane_learnable():
    # all heads sample and every present HDV row is at risk; lane_event is off so the only
    # mainline-head objective is the count-matching lane_flow term.
    off_model, off_rec, off = _flow_grads_learnable(["loss.lane_flow.weight=0.0"])
    model, rec, on = _flow_grads_learnable(["loss.lane_flow.weight=1.0",
                                            "loss.lane_flow.balance=0.5"])
    assert model.physics.lane_learnable and model._samples_lanes
    assert off_model._samples_lanes and "lane_flow" not in off_rec
    assert model.emit_merge_hazard and math.isfinite(rec["lane_flow"])
    # per-step P over the at-risk mainline rows: lane-1 cars can overtake, none can return
    assert rec["lane_flow_overtake_expected"] > 0.0
    assert rec["lane_flow_overtake_recorded"] == 0.0
    assert rec["lane_flow_return_recorded"] == 0.0 and rec["lane_flow_return_expected"] >= 0.0
    # the count term trains the overtake head (row 0) and leaves the merge head (row 2) alone;
    # with lane_event off the count term is the ONLY objective on the mainline heads
    off_g = torch.zeros_like(on) if off is None else off
    assert float(on[0].abs().sum()) > 0.0 and float(off_g[0].abs().sum()) == 0.0
    torch.testing.assert_close(on[2], off_g[2])


def test_plumbing_and_validation():
    _, flat, settings = _c4([*V5, 'model.lane_v5_mainline="budget"',
                             "loss.lane_flow.weight=0.5", "loss.lane_flow.balance=0.3"])
    assert flat["lane_v5_mainline"] == "budget" and flat["lane_flow_weight"] == 0.5
    assert settings.lane_flow_balance == 0.3 and settings.lane_v5_mainline == "budget"
    assert ModelSpec.from_mapping(flat).lane_v5_mainline == "budget"
    with pytest.raises(Exception):
        _c4(['model.lane_v5_mainline="budget"'])                        # needs hazard_v5
    with pytest.raises(Exception):
        _c4([*V5, 'model.lane_v5_mainline="often"'])
    with pytest.raises(ValueError):
        ModelSpec(lane_v5_mainline="sample")


def test_v8lane_precheck_config_differs_from_v8ft_short2_only_in_the_lane_keys():
    a = load_train_config("configs/c4_wm_v8ft_short2.toml", []).to_legacy_flat()
    b = load_train_config("configs/c4_wm_v8lane_short2.toml", []).to_legacy_flat()
    diff = {k: b.get(k) for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"lane_v5_mainline": "budget", "lane_flow_weight": 1.0,
                    "lane_flow_balance": 0.5, "out_dir": "runs/active/c4_wm_v8lane_short2"}
    s = TrainSettings.from_flat(b)
    assert "lane_flow" in build_loss(s).names and s.lane_model == "hazard_v5"



def test_v8lane3_configs_differ_from_v8ft3_only_in_the_lane_keys():
    # V8INT: the lane fix on top of the V8WM3 merge-zone rules (merge_impatience etc. kept).
    for base, name in (("c4_wm_v8ft3", "c4_wm_v8lane3"),
                       ("c4_wm_v8ft_short3", "c4_wm_v8lane_short3")):
        a = load_train_config(f"configs/{base}.toml", []).to_legacy_flat()
        b = load_train_config(f"configs/{name}.toml", []).to_legacy_flat()
        diff = {k: b.get(k) for k in set(a) | set(b) if a.get(k) != b.get(k)}
        assert diff == {"lane_v5_mainline": "budget", "lane_flow_weight": 1.0,
                        "lane_flow_balance": 0.5, "out_dir": f"runs/active/{name}"}
        s = TrainSettings.from_flat(b)
        assert {"lane_flow", "merge_timing"} <= set(build_loss(s).names)
        assert s.lane_model == "hazard_v5"

def test_budget_training_unroll_with_lane_flow_is_finite():
    model, rec, grad = _flow_grads(['model.lane_v5_mainline="budget"',
                                    "loss.lane_flow.weight=1.0"])
    assert model._mainline_mode == "budget"
    assert math.isfinite(rec["lane_flow"]) and bool(torch.isfinite(grad).all())
