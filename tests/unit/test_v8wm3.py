"""V8WM3 (2026-10-17): waiting-time impatience of lane-end mergers (``physics.merge_impatience``).
Default off = legacy (bit-identity: the golden lane tests and test_v8wm2 stay green)."""
from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from .test_gaprule import _kin
from .test_lanefix import _phys
from .test_lanev5 import _batch, _rollout, _single_cars, _v5_model
from wms.cli.assemble import build_physics
from wms.config.load import load_train_config
from wms.model import lane_v5 as lv5
from wms.model.merge_wait import initial_merge_wait, recorded_merge_wait
from wms.physics.accel import baseline_accel
from wms.physics.config import PhysicsConfig
from wms.physics.merge_v6 import merge_feasible_among
from wms.physics.mobil import (MASKED_LOGIT, merge_end_blocked, merge_impatience_floor,
                               merge_impatience_level, mobil_merge_logits, secure_gaps,
                               speed_gap_safe, update_merge_wait)
from wms.physics.neighbours import lane_of

END = 75.44


def _imp(**kw):
    return _phys(merge_impatience=True, **kw)


def _scene(seed=7, b=64, k=8):
    g = torch.Generator().manual_seed(seed)
    kin = torch.zeros(b, k, 7)
    kin[..., 0] = torch.rand(b, k, generator=g) * 100.0 - 10.0
    lane = 1 + (torch.rand(b, k, generator=g) < 0.4).long()
    kin.scatter_(-1, (1 + lane).unsqueeze(-1), 1.0)
    kin[..., 4] = torch.rand(b, k, generator=g) * 12.0 * (torch.rand(b, k, generator=g) > 0.3)
    kin[..., 5] = torch.rand(b, k, generator=g) * 4.0 - 2.0
    kin[..., 6] = 4.0 + 2.0 * torch.rand(b, k, generator=g)
    present = torch.rand(b, k, generator=g) > 0.1
    level = torch.rand(b, k, generator=g) * (lane_of(kin) == 2)
    return kin, present, level


# --------------------------------------------------------------- config
def test_defaults_off_and_validation():
    p = PhysicsConfig()
    assert not p.merge_impatience
    assert merge_impatience_level(torch.ones(1, 1), p) is None
    assert merge_impatience_level(None, _imp()) is None
    with pytest.raises(ValueError):          # needs the speed gap rule
        PhysicsConfig(merge_impatience=True)
    for bad in (dict(merge_impatience_gap_tau_s=0.0), dict(merge_impatience_hazard_tau_s=-1.0),
                dict(merge_impatience_s0_min_m=-0.1), dict(merge_impatience_distance_m=0.0)):
        with pytest.raises(ValueError):
            _imp(**bad)
    assert _imp(merge_impatience_s0_min_m=0.0, merge_impatience_t_min_s=0.0).merge_impatience


def test_configs_v8ft3_rules_on_in_training_and_inference():
    for name, steps in (("c4_wm_v8ft3", (300, 250)), ("c4_wm_v8ft_short3", (80, 60))):
        cfg = load_train_config(f"configs/{name}.toml")
        assert tuple(cfg.curriculum.stage_steps) == steps
        assert cfg.loss.merge_timing.weight == 6.0 and cfg.loss.loss_share_logging is False
        for phys in (build_physics(cfg, resolver=lambda n: n),):
            infer = replace(phys, **cfg.physics.inference)
            for p in (phys, infer):
                assert p.merge_end_wait and p.mainline_merge_yield and p.merge_impatience
                assert p.forced_window_mode == "hazard" and p.forced_window_min_hazard == 0.05
                assert p.mainline_merge_yield_scope == "zone"
                assert p.merge_end_follow_constants() == (1.0, 0.6)
            assert infer == phys
    with pytest.raises(Exception):
        load_train_config("configs/c4_wm_v8ft.toml", ['physics.overrides={merge_impatience=1.0}'])


# --------------------------------------------------------------- waiting time
def test_update_merge_wait_accumulates_holds_and_resets():
    p = _imp()
    present = torch.ones(1, 4, dtype=torch.bool)
    # waiting (stopped near the end), creeping (2 m/s near the end), far upstream, lane 1
    kin = _kin([(END - 3.0, 2, 0.0), (END - 10.0, 2, 2.0), (END - 60.0, 2, 0.0), (END - 3.0, 1, 0.0)])
    w = update_merge_wait(None, kin, present, p)
    assert w.tolist() == [[0.5, 0.0, 0.0, 0.0]]
    w = update_merge_wait(torch.tensor([[3.0, 4.0, 5.0, 6.0]]), kin, present, p)
    assert w.tolist() == [[3.5, 4.0, 5.0, 0.0]]                     # held in lane 2, reset off
    w = update_merge_wait(torch.tensor([[3.0, 4.0, 5.0, 6.0]]), kin,
                          torch.tensor([[False, True, True, True]]), p)
    assert float(w[0, 0]) == 0.0                                     # absent -> reset


def test_initial_merge_wait_from_long_history():
    p = _imp()
    kin = _kin([(END - 2.0, 2, 0.0), (END - 2.0, 1, 0.0)])
    present = torch.ones(1, 2, dtype=torch.bool)
    hist = torch.zeros(1, 2, 5, 9)
    hist[..., 2 + 2] = 1.0                       # lane 2 for car 0
    hist[0, 1, :, 2 + 2], hist[0, 1, :, 2 + 1] = 0.0, 1.0             # car 1 in lane 1
    mask = torch.ones(1, 2, 5, dtype=torch.bool)
    w = initial_merge_wait(present, kin, p, hist, mask, stride_steps=2)
    # 4 pre-t0 samples x 1 s + the t0 step 0.5 s
    assert w.tolist() == [[4.5, 0.0]]
    assert initial_merge_wait(present, kin, p, None, None, 2).tolist() == [[0.5, 0.0]]
    seq = recorded_merge_wait(torch.stack([kin, kin, kin], 1), torch.ones(1, 3, 2, dtype=torch.bool),
                              p, w)
    assert seq[0, :, 0].tolist() == [4.5, 5.0, 5.5]


# --------------------------------------------------------------- gap relaxation
def test_speed_rule_margins_shrink_but_never_overlap():
    p = _imp()
    present = torch.ones(1, 2, dtype=torch.bool)
    tl = torch.ones(1, 2, dtype=torch.long)
    # stopped merger, stopped lane-1 leader 0.8 m ahead (bumper): legacy needs s0_lead 1.0 m
    kin = _kin([(END - 2.0, 2, 0.0), (END - 2.0 + 5.8, 1, 0.0)])
    assert not bool(speed_gap_safe(kin, present, tl, p)[0, 0])
    zero = torch.zeros(1, 2)
    assert torch.equal(speed_gap_safe(kin, present, tl, p, zero), speed_gap_safe(kin, present, tl, p))
    assert bool(speed_gap_safe(kin, present, tl, p, torch.ones(1, 2))[0, 0])
    # overlap is never relaxed
    kin = _kin([(END - 2.0, 2, 0.0), (END - 4.0, 1, 0.0)])
    assert not bool(speed_gap_safe(kin, present, tl, p, torch.ones(1, 2))[0, 0])


def test_merge_end_wait_secure_gaps_relax_leader_and_follower():
    p = _imp(merge_end_wait=True, merge_end_wait_s0_follow_m=1.0, merge_end_wait_t_follow_s=0.6)
    v, lv = torch.tensor([0.0]), torch.tensor([0.0])
    gl, gf, fv = torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([0.0])
    assert not bool(secure_gaps(v, gl, lv, gf, fv, p))
    assert not bool(secure_gaps(v, gl, lv, gf, fv, p, torch.tensor([0.5])))
    assert bool(secure_gaps(v, gl, lv, gf, fv, p, torch.tensor([1.0])))
    # the follower's kinematic term stays: a 6 m/s follower 1 m behind is never secure
    assert not bool(secure_gaps(v, gl, lv, gf, torch.tensor([6.0]), p, torch.tensor([1.0])))
    kin = _kin([(END - 1.0, 2, 0.0), (END - 1.0 + 6.0, 1, 0.0), (END - 7.0, 1, 0.0)])
    present = torch.ones(1, 3, dtype=torch.bool)
    assert bool(merge_end_blocked(kin, present, p)[0, 0])
    assert not bool(merge_end_blocked(kin, present, p, torch.tensor([[1.0, 0.0, 0.0]]))[0, 0])


@pytest.mark.parametrize("wait,relax", [(False, True), (True, True), (True, False)])
def test_impatient_feasible_among_equals_physics_mask(wait, relax):
    kin, present, level = _scene()
    phys = _imp(merge_end_wait=wait, gap_stopped_relax=relax)
    lane1 = present & (lane_of(kin) == 1)
    cand = lane1.unsqueeze(-2).expand(*lane1.shape, lane1.shape[-1])
    mask = (mobil_merge_logits(kin, present, phys, impatience=level)[..., 1] > MASKED_LOGIT) \
        & present & (lane_of(kin) == 2)
    assert torch.equal(merge_feasible_among(kin, present, phys, cand, level), mask)
    legacy = (mobil_merge_logits(kin, present, phys)[..., 1] > MASKED_LOGIT) \
        & present & (lane_of(kin) == 2)
    assert bool((mask | ~legacy).all()) and int(mask.sum()) > int(legacy.sum())


def test_discretionary_rows_unchanged_and_fast_mask_matches():
    kin, present, level = _scene(seed=11)
    phys = _imp(merge_end_wait=True, gap_stopped_relax=True)
    lo = mobil_merge_logits(kin, present, phys)
    li = mobil_merge_logits(kin, present, phys, impatience=level)
    ramp = lane_of(kin) == 2
    assert torch.equal(lo[~ramp], li[~ramp])
    inst = lv5.instant_features(kin, present, phys, None, level)
    tq = inst["tq"]
    ref = li.gather(-1, tq.unsqueeze(-1)).squeeze(-1) > MASKED_LOGIT
    assert torch.equal(inst["target_feasible"], ref & present)
    # features (ml / mf) keep the legacy rule
    base = lv5.instant_features(kin, present, phys, None)
    assert torch.equal(inst["ml"], base["ml"]) and torch.equal(inst["mf"], base["mf"])


def test_yield_accel_unchanged_without_impatience_level():
    kin, present, level = _scene(seed=5)
    phys = _imp(merge_end_wait=True, mainline_merge_yield=True, mainline_merge_yield_scope="zone")
    a0 = baseline_accel(kin, present, phys)
    a1 = baseline_accel(kin, present, phys, impatience=torch.zeros_like(level))
    assert torch.allclose(a0, a1)
    assert torch.isfinite(baseline_accel(kin, present, phys, impatience=level)).all()


# --------------------------------------------------------------- hazard floor
def test_forced_window_floor_rises_with_waiting_time():
    p = _imp(forced_window_mode="hazard", forced_window_min_hazard=0.05)
    w = torch.tensor([[0.0, 12.5, 25.0, 100.0]])
    h = merge_impatience_floor(w, p)
    assert h[0].tolist() == pytest.approx([0.05, 0.525, 1.0, 1.0])
    kin = _single_cars([(END - 2.0, 2, 0.0)] * 4)
    net = lv5.LaneV5Net()
    with torch.no_grad():
        net.heads.bias.fill_(-9.0)
    present = torch.ones(kin.shape[:2], dtype=torch.bool)
    base = mobil_merge_logits(kin, present, p)
    acc = lv5.initial_acc(present, kin, p, None, None, 1)
    raw, _, inst = lv5.step_features(acc, kin, present, p, None)
    lg, head, acts = lv5.lane_logits_v5(net, raw, inst, present, base, p, None,
                                         w.reshape(4, 1))
    got = lg[:, 0, 1].tolist()
    assert got[0] == pytest.approx(math.log(0.05 / 0.95), abs=1e-5)
    assert got[1] == pytest.approx(math.log(0.525 / 0.475), abs=1e-4)
    assert got[2] == pytest.approx(lv5.FORCED_MERGE_LOGIT) and got[3] == got[2]
    assert acts.all()
    # without a carried wait (or switch off) the constant floor applies
    lg0, _, _ = lv5.lane_logits_v5(net, raw, inst, present, base, p)
    assert lg0[:, 0, 1].tolist() == pytest.approx([math.log(0.05 / 0.95)] * 4, abs=1e-5)


# --------------------------------------------------------------- rollout carry
def test_rollout_carries_wait_and_chunks_match():
    model, _ = _v5_model()
    model.eval()
    model.physics = replace(model.physics, merge_impatience=True, merge_end_wait=True,
                            forced_window_mode="hazard", forced_window_min_hazard=0.05)
    assert model._tracks_merge_wait
    b = _batch()
    full = _rollout(model, b, n=8, lane_seed=3)
    assert "merge_wait_s" in full["state"]
    w = full["state"]["merge_wait_s"]
    assert (w >= 0).all() and (w <= 0.5 * 8 + 1e3).all()
    first = _rollout(model, b, n=8, lane_seed=3)
    assert torch.equal(first["positions"], full["positions"])        # deterministic
    # off: no key carried
    model.physics = replace(model.physics, merge_impatience=False)
    assert "merge_wait_s" not in _rollout(model, b, n=2)["state"]
