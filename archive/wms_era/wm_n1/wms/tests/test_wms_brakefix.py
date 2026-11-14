"""Braking-response fix: asymmetric loss, decel weighting, propagation, asym cap.

Pins (1) flags-off parity with the pre-fix trainer, (2) the sign/weighting of
each term on synthetic cases, (3) finite gradients, (4) the vectorised
implementations against slow Python references, (5) the real model's braking
reach versus ``min_accel_reach``, and (6) CLI/config/gate plumbing.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from output.wd01.wm_n1.hdv.config import HdvConfig
from output.wd01.wm_n1.wms import brakefix as bf
from output.wd01.wm_n1.wms.model import PhysicsConfig, WmSModel
from output.wd01.wm_n1.wms.stability import bounded_value
from output.wd01.wm_n1.wms.long_horizon import MONITORS
from output.wd01.wm_n1.wms.train import (BRAKEFIX_CONFIG_KEYS, TrajectoryDataset,
                                         WmSTrainConfig, WmSTrainer, collate)


# ------------------------------------------------------------------- config / caps
def test_defaults_are_off_and_validated():
    cfg = bf.BrakefixConfig()
    assert not cfg.any_on and cfg.log_terms == ()
    assert bf.BrakefixConfig(asym_alpha=2.0).log_terms == ("asym_speed",)
    full = bf.BrakefixConfig(asym_alpha=2.0, decel_weight=5.0, prop_weight=1.0)
    assert full.log_terms == ("asym_speed", "decel_weight", "propagation")
    assert bf.BrakefixConfig(prop_window_s=3.0, dt_s=0.5).window_steps == 6
    for bad in (dict(asym_alpha=0.5), dict(decel_weight=0.9), dict(decel_threshold_mps2=0.0),
                dict(prop_weight=-1.0), dict(prop_window_s=0.0), dict(prop_max_gap_m=0.0)):
        with pytest.raises(ValueError):
            bf.BrakefixConfig(**bad)
    with pytest.raises(ValueError):
        WmSTrainConfig(asym_speed_alpha=0.5)
    with pytest.raises(ValueError):
        WmSTrainConfig(residual_cap_neg_mps2=0.0)


def test_train_config_default_to_dict_omits_braking_keys_and_round_trips():
    d = WmSTrainConfig().to_dict()
    assert not any(k in d for k in BRAKEFIX_CONFIG_KEYS)
    on = WmSTrainConfig(asym_speed_alpha=2.0, residual_cap_neg_mps2=3.0)
    d2 = on.to_dict()
    assert d2["asym_speed_alpha"] == 2.0 and d2["residual_cap_neg_mps2"] == 3.0
    assert "decel_weight" not in d2
    assert WmSTrainConfig.from_dict(d2).to_dict() == d2
    assert WmSTrainConfig.from_dict(d).to_dict() == d


def test_bounded_value_asym_identity_symmetric_and_asymmetric_limits():
    x = torch.linspace(-20.0, 20.0, 81)
    assert torch.equal(bf.bounded_value_asym(x, None, None), x)
    torch.testing.assert_close(bf.bounded_value_asym(x, 1.5, 1.5), bounded_value(x, 1.5))
    y = bf.bounded_value_asym(x, 1.5, 3.0)
    assert float(y.max()) <= 1.5 + 1e-6 and float(y.min()) >= -3.0 - 1e-6
    assert float(y[-1]) == pytest.approx(1.5, abs=1e-4)
    assert float(y[0]) == pytest.approx(-3.0, abs=1e-4)
    # one side uncapped
    z = bf.bounded_value_asym(x, None, 3.0)
    assert float(z.max()) == 20.0 and float(z.min()) >= -3.0 - 1e-6
    with pytest.raises(ValueError):
        bf.bounded_value_asym(x, 0.0, 1.0)


def test_bounded_value_asym_gradients_are_finite_and_continuous_at_zero():
    x = torch.tensor([-50.0, -1.0, -1e-6, 0.0, 1e-6, 1.0, 50.0], requires_grad=True)
    bf.bounded_value_asym(x, 1.5, 3.0).sum().backward()
    assert bool(torch.isfinite(x.grad).all())
    assert float(x.grad[3]) == pytest.approx(1.0)       # unit slope on both sides
    assert float(x.grad[2]) == pytest.approx(1.0, abs=1e-4)
    assert float(x.grad[4]) == pytest.approx(1.0, abs=1e-4)
    assert float(x.grad[0]) < 1e-6 and float(x.grad[-1]) < 1e-6


# ----------------------------------------------------------------- weight helpers
def test_asym_multiplier_penalises_too_fast_only():
    pred = torch.tensor([1.0, 3.0, 2.0])
    rec = torch.tensor([2.0, 2.0, 2.0])
    torch.testing.assert_close(bf.asym_multiplier(pred, rec, 4.0),
                               torch.tensor([1.0, 4.0, 1.0]))   # ties are not "too fast"
    p = torch.tensor([3.0], requires_grad=True)
    assert not bf.asym_multiplier(p, torch.tensor([2.0]), 4.0).requires_grad


def test_decel_multiplier_is_strict_below_minus_threshold():
    a = torch.tensor([-4.0, -3.0, -2.99, 0.5, -5.0])
    valid = torch.tensor([True, True, True, True, False])
    torch.testing.assert_close(bf.decel_multiplier(a, valid, 6.0, 3.0),
                               torch.tensor([6.0, 1.0, 1.0, 1.0, 1.0]))


# ------------------------------------------------------ braking_terms vs slow ref
def _rand_terms_inputs(seed, shape=(2, 5, 6)):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    rec_pos, rec_spd = r(*shape) * 5, r(*shape) * 3 + 10
    pred_pos, pred_spd = rec_pos + r(*shape) * 2, rec_spd + r(*shape)
    pos = F.huber_loss(pred_pos, rec_pos, delta=5.0, reduction="none")
    spd = F.huber_loss(pred_spd, rec_spd, delta=1.0, reduction="none")
    a_rec = r(*shape) * 2 - 1.0
    mask = torch.rand(*shape, generator=g) > 0.2
    weights = torch.exp(-0.01 * torch.arange(shape[1], dtype=torch.float32)).view(1, -1, 1)
    denom = (weights * mask.float()).sum().clamp_min(1.0)
    return dict(pos=pos, spd=spd, pred_pos=pred_pos, rec_pos=rec_pos, pred_spd=pred_spd,
                rec_spd=rec_spd, a_rec=a_rec, hdv_mask=mask, weights=weights, denom=denom)


def _slow_terms(cfg, inp, wp, ws):
    asym = dec = 0.0
    n_b, n_t, n_k = inp["pos"].shape
    for b in range(n_b):
        for t in range(n_t):
            for k in range(n_k):
                if not bool(inp["hdv_mask"][b, t, k]):
                    continue
                w = float(inp["weights"][0, t, 0])
                base_p = wp * float(inp["pos"][b, t, k]) * w
                base_s = ws * float(inp["spd"][b, t, k]) * w
                m_ap = cfg.asym_alpha if (cfg.asym_on and float(inp["pred_pos"][b, t, k])
                                          > float(inp["rec_pos"][b, t, k])) else 1.0
                m_as = cfg.asym_alpha if (cfg.asym_on and float(inp["pred_spd"][b, t, k])
                                          > float(inp["rec_spd"][b, t, k])) else 1.0
                m_d = cfg.decel_weight if (cfg.decel_on and float(inp["a_rec"][b, t, k])
                                           < -cfg.decel_threshold_mps2) else 1.0
                asym += (m_ap - 1.0) * base_p + (m_as - 1.0) * base_s
                dec += (m_d - 1.0) * (m_ap * base_p + m_as * base_s)
    d = float(inp["denom"])
    return asym / d, dec / d


@pytest.mark.parametrize("alpha,w", [(3.0, 1.0), (1.0, 6.0), (3.0, 6.0)])
def test_braking_terms_match_slow_reference_and_total_identity(alpha, w):
    cfg = bf.BrakefixConfig(asym_alpha=alpha, decel_weight=w, decel_threshold_mps2=1.2)
    inp = _rand_terms_inputs(7)
    out = bf.braking_terms(cfg, position_weight=1.0, speed_weight=0.7, **inp)
    asym, dec = _slow_terms(cfg, inp, 1.0, 0.7)
    assert ("asym_speed" in out) == cfg.asym_on and ("decel_weight" in out) == cfg.decel_on
    if cfg.asym_on:
        assert float(out["asym_speed"]) == pytest.approx(asym, rel=1e-5)
    if cfg.decel_on:
        assert float(out["decel_weight"]) == pytest.approx(dec, rel=1e-5)
    # base + terms == sum(w * m_asym * m_decel * base) / denom
    m = inp["hdv_mask"].float()
    base = ((1.0 * inp["pos"] + 0.7 * inp["spd"]) * inp["weights"] * m).sum() / inp["denom"]
    assert float(base + sum(out.values())) >= float(base) - 1e-9
    assert float(sum(out.values())) == pytest.approx(asym + dec, rel=1e-5)


def test_asym_term_weights_too_fast_alpha_times_the_too_slow_error():
    # Equal-magnitude +/- errors: too-fast must contribute alpha x too-slow (total).
    shape = (1, 1, 2)
    rec = torch.zeros(shape)
    pred_spd = torch.tensor([[[0.5, -0.5]]])             # fast, slow (both |e| < delta)
    pred_pos = torch.zeros(shape)
    spd = F.huber_loss(pred_spd, rec, delta=1.0, reduction="none")
    w = torch.ones(1, 1, 1)
    mask = torch.ones(shape, dtype=torch.bool)
    cfg = bf.BrakefixConfig(asym_alpha=5.0)
    terms = bf.braking_terms(cfg, pos=torch.zeros(shape), spd=spd, pred_pos=pred_pos,
                             rec_pos=rec, pred_spd=pred_spd, rec_spd=rec,
                             a_rec=torch.zeros(shape), hdv_mask=mask, weights=w,
                             denom=torch.tensor(2.0), position_weight=1.0, speed_weight=1.0)
    too_slow_mass = float(spd[0, 0, 1]) / 2.0            # base contribution of the slow slot
    base_total = float(spd.sum()) / 2.0
    total = base_total + float(terms["asym_speed"])
    assert total == pytest.approx(too_slow_mass + 5.0 * float(spd[0, 0, 0]) / 2.0)


def test_braking_terms_gradients_flow_only_through_predictions():
    inp = _rand_terms_inputs(3)
    pred_pos = inp["pred_pos"].clone().requires_grad_(True)
    pred_spd = inp["pred_spd"].clone().requires_grad_(True)
    rec_pos, rec_spd = inp["rec_pos"], inp["rec_spd"]
    inp["pos"] = F.huber_loss(pred_pos, rec_pos, delta=5.0, reduction="none")
    inp["spd"] = F.huber_loss(pred_spd, rec_spd, delta=1.0, reduction="none")
    inp["pred_pos"], inp["pred_spd"] = pred_pos, pred_spd
    cfg = bf.BrakefixConfig(asym_alpha=3.0, decel_weight=5.0, decel_threshold_mps2=1.0)
    out = bf.braking_terms(cfg, position_weight=1.0, speed_weight=1.0, **inp)
    sum(out.values()).backward()
    for g in (pred_pos.grad, pred_spd.grad):
        assert g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0
    # masked-out slots receive no gradient
    off = ~inp["hdv_mask"]
    assert float(pred_spd.grad[off].abs().sum()) == 0.0


# ------------------------------------------------- propagation vs slow references
def _rand_scene(seed, shape=(2, 7, 7)):
    n_b, t_n, k_n = shape
    g = torch.Generator().manual_seed(seed)
    s = torch.rand(n_b, t_n + 1, k_n, generator=g) * 120.0 + torch.arange(k_n) * 0.001
    lane = torch.randint(0, 3, (n_b, t_n + 1, k_n), generator=g)
    valid = torch.rand(n_b, t_n + 1, k_n, generator=g) > 0.15
    length = torch.full((n_b, t_n + 1, k_n), 5.0)
    a_next = torch.rand(n_b, t_n, k_n, generator=g) * 7 - 5.0
    return s, lane, valid, length, a_next


def _slow_events(s, lane, valid, length, a_next, thr, max_gap):
    n_b, t1, k_n = s.shape
    ev = torch.zeros(n_b, t1 - 1, k_n, dtype=torch.bool)
    for b in range(n_b):
        for t in range(t1 - 1):
            for i in range(k_n):
                if not bool(valid[b, t, i]):
                    continue
                best, bj = math.inf, -1
                for j in range(k_n):
                    if j == i or not bool(valid[b, t, j]) or int(lane[b, t, j]) != int(lane[b, t, i]):
                        continue
                    d = float(s[b, t, j]) - float(s[b, t, i])
                    if d > 0 and d < best:
                        best, bj = d, j
                if bj < 0:
                    continue
                gap = best - float(length[b, t, bj])
                ev[b, t, i] = (bool(valid[b, t + 1, bj]) and gap <= max_gap
                               and float(a_next[b, t, bj]) < -thr)
    return ev


@pytest.mark.parametrize("budget", [4_000_000, 1])    # one chunk vs one step per chunk
def test_leader_brake_events_match_slow_reference(budget):
    s, lane, valid, length, a_next = _rand_scene(11)
    fast = bf.leader_brake_events(s, lane, valid, length, a_next, threshold=2.0,
                                  max_gap_m=40.0, budget=budget)
    slow = _slow_events(s, lane, valid, length, a_next, 2.0, 40.0)
    assert fast.dtype == torch.bool and fast.shape == slow.shape
    assert torch.equal(fast, slow)
    assert int(slow.sum()) > 0           # the scene actually exercises the code path


def test_propagation_window_matches_loop_reference():
    g = torch.Generator().manual_seed(5)
    ev = torch.rand(2, 12, 5, generator=g) > 0.8
    for length in (1, 3, 6, 20):
        fast = bf.propagation_window(ev, length)
        slow = torch.zeros_like(ev)
        for b in range(2):
            for u in range(12):
                for k in range(5):
                    slow[b, u, k] = any(bool(ev[b, t, k]) for t in range(max(0, u - length), u))
        assert torch.equal(fast, slow)
    # an onset at t affects t+1..t+L, never t itself
    one = torch.zeros(1, 10, 1, dtype=torch.bool)
    one[0, 2, 0] = True
    got = bf.propagation_window(one, 3)[0, :, 0].tolist()
    assert got == [False, False, False, True, True, True, False, False, False, False]


def test_nearest_recorded_leader_has_no_default_distance_cutoff():
    s = torch.tensor([[[0.0, 200.0], [0.0, 200.0]]])
    lane = torch.zeros_like(s, dtype=torch.long)
    valid = torch.ones_like(s, dtype=torch.bool)
    length = torch.full_like(s, 5.0)
    a = torch.tensor([[[0.0, -4.0]]])
    assert bf.leader_brake_events(s, lane, valid, length, a, threshold=3.0)[0, 0, 0]
    assert not bf.leader_brake_events(s, lane, valid, length, a, threshold=3.0,
                                     max_gap_m=80.0)[0, 0, 0]


def test_propagation_terms_sign_mask_and_gradients():
    w = torch.ones(1, 3, 1)
    mask = torch.tensor([[[True], [True], [False]]])
    rec_v = torch.full((1, 3, 1), 10.0)
    rec_a = torch.full((1, 3, 1), -2.0)
    pred_v = torch.tensor([[[12.0], [10.0], [50.0]]], requires_grad=True)
    pred_a = torch.tensor([[[-2.0], [0.0], [9.0]]], requires_grad=True)
    t = bf.propagation_terms(pred_v, pred_a, rec_v, rec_a, mask, w, torch.tensor(2.0),
                             speed_delta=1.0, accel_delta=1.0)
    exp = (F.huber_loss(pred_v[0, 0], rec_v[0, 0], delta=1.0)        # |e|=2 -> 1.5
           + F.huber_loss(pred_a[0, 1], rec_a[0, 1], delta=1.0)) / 2.0   # |e|=2 -> 1.5
    assert float(t.detach()) == pytest.approx(float(exp.detach()), rel=1e-6)
    t.backward()
    assert float(pred_v.grad[0, 2, 0]) == 0.0 and float(pred_a.grad[0, 2, 0]) == 0.0
    assert bool(torch.isfinite(pred_v.grad).all()) and bool(torch.isfinite(pred_a.grad).all())
    zero = bf.propagation_terms(pred_v, pred_a, rec_v, rec_a, torch.zeros_like(mask), w,
                                torch.tensor(2.0), speed_delta=1.0, accel_delta=1.0)
    assert float(zero.detach()) == 0.0


# -------------------------------------------------- real model: braking reach/caps
def _small_model(**kw):
    return WmSModel(physics=PhysicsConfig(max_gap_steps=12, sumo_fidelity=True),
                    hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16), **kw)


class _Stub(torch.nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, kin, present, conditioning=None):
        z = torch.zeros(kin.shape[:2])
        return {"accel_delta_mps2": z + self.delta,
                "lane_logit_delta": torch.zeros(*kin.shape[:2], 3), "accel_log_sigma": z}


def _braking_scene():
    kin = torch.zeros(1, 2, 7)
    kin[..., 1] = 1.0
    kin[0, 0, 0], kin[0, 0, 4] = 6.0, 0.0     # stopped leader, 1 m bumper gap
    kin[0, 1, 0], kin[0, 1, 4] = 0.0, 25.0    # fast follower
    kin[..., 6] = 5.0
    state = {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4], "a_mps2": kin[..., 5],
             "length_m": kin[..., 6], "lane": kin[..., 1:4].argmax(-1),
             "present": torch.ones(1, 2, dtype=torch.bool),
             "is_cav": torch.zeros(1, 2, dtype=torch.bool)}
    decel = 4.2
    params = {"v0": 31.0, "time_headway": 1.1, "min_gap": 2.0, "accel": 2.5, "decel": decel,
              "politeness": 0.2, "mobil_threshold": 0.1, "merge_gap": 6.0,
              "merge_urgency": 1.0}
    return state, {k: torch.full((1, 2), v) for k, v in params.items()}, decel


@pytest.mark.parametrize("cap_pos,cap_neg", [(None, None), (1.5, None), (1.5, 3.0),
                                             (1.5, 4.5), (None, 2.0)])
def test_model_min_accel_equals_audit_reach(cap_pos, cap_neg):
    torch.manual_seed(0)
    model = _small_model(rollout_mode="hybrid", residual_cap_mps2=cap_pos,
                         residual_cap_neg_mps2=cap_neg)
    model.residual_net = _Stub(-50.0)             # saturating braking residual
    state, params, decel = _braking_scene()
    out = model(state, k_steps=10, driver_params=params)
    assert float(out["physics_accel_mps2"][0, 1]) == pytest.approx(-decel, abs=1e-5)
    reach = bf.min_accel_reach(rollout_mode="hybrid", decel_i=decel, cap_pos=cap_pos,
                               cap_neg=cap_neg)
    got = float(out["accel_mps2"][0, 1].detach())
    assert got >= reach - 1e-4                      # never below the audited reach
    gate = float(model.gate(10).detach())
    if cap_pos is None and cap_neg is None:
        assert got == pytest.approx(max(-9.0, -decel - 50.0 * gate), abs=0.05)
    elif gate * 50.0 / (cap_neg or cap_pos) > 5.0:
        assert got == pytest.approx(reach, abs=0.02)   # tanh saturated at the cap


def test_asym_cap_none_and_equal_cap_are_byte_identical_to_symmetric_path():
    torch.manual_seed(1)
    ref = _small_model(rollout_mode="hybrid", residual_cap_mps2=1.5)
    asym_same = _small_model(rollout_mode="hybrid", residual_cap_mps2=1.5,
                             residual_cap_neg_mps2=1.5)
    asym_same.load_state_dict(ref.state_dict())
    state, params, _ = _braking_scene()
    a = ref(state, k_steps=5, driver_params=params)["accel_mps2"]
    b = asym_same(state, k_steps=5, driver_params=params)["accel_mps2"]
    torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
    with pytest.raises(ValueError):
        _small_model(residual_cap_neg_mps2=0.0)


def test_min_accel_reach_formula():
    r = lambda **k: bf.min_accel_reach(**k)  # noqa: E731
    assert r(rollout_mode="residual", decel_i=4.2, cap_pos=1.5, cap_neg=None) == -9.0
    assert r(rollout_mode="driver_params", decel_i=4.2, cap_pos=1.5, cap_neg=3.0) == -4.2
    assert r(rollout_mode="hybrid", decel_i=4.2, cap_pos=1.5, cap_neg=None) == pytest.approx(-5.7)
    assert r(rollout_mode="hybrid", decel_i=4.2, cap_pos=1.5, cap_neg=3.0) == pytest.approx(-7.2)
    assert r(rollout_mode="hybrid", decel_i=4.2, cap_pos=1.5, cap_neg=9.0) == -9.0
    assert r(rollout_mode="hybrid", decel_i=4.2, cap_pos=None, cap_neg=None) == -9.0


# ------------------------------------------------------------- trainer integration
def _brake_dataset(n=2, t1=21, slots=4):
    """Lane-0 platoon; slot 0 (leader) brakes at transitions 5..8, then holds."""
    states = np.zeros((n, t1, slots, 7), dtype=np.float32)
    states[..., 1] = 1.0
    states[..., 6] = 5.0
    v = np.full(t1, 12.0, dtype=np.float32)
    for t in range(6, 10):
        v[t] = v[t - 1] - 2.0                               # -4 m/s^2 at dt=0.5
    v[10:] = v[9]
    a = np.zeros(t1, dtype=np.float32)
    a[1:] = (v[1:] - v[:-1]) / 0.5
    base_s = np.array([70.0, 45.0, 20.0, 0.0], dtype=np.float32)
    for slot in range(slots):
        vv = v if slot == 0 else np.full(t1, 12.0, dtype=np.float32)
        aa = a if slot == 0 else np.zeros(t1, dtype=np.float32)
        states[:, :, slot, 0] = base_s[slot] + np.concatenate(([0.0], np.cumsum(vv[1:] * 0.5)))
        states[:, :, slot, 4] = vv
        states[:, :, slot, 5] = aa
    present = np.ones((n, t1, slots), dtype=bool)
    present[:, :, -1] = False
    return TrajectoryDataset.from_arrays({"states": states, "present": present})


def _trainer(**cfg_kw):
    cfg = WmSTrainConfig(device="cpu", threads=1, steps_per_epoch=1,
                         unroll_curriculum_s=(10.0,), batch_size=2, **cfg_kw)
    return WmSTrainer(_small_model(), cfg), cfg


def _batch(ds):
    return collate([ds[0], ds[1]])


BASE_KEYS_OFF = {"loss", "gate_curve", "position", "speed", "lane", "gate_g0", "gate_g_last",
                 "gate_monotone", "unroll_steps", "valid_targets", "driver_param_stats",
                 "entrants", "valid_hdv_slots_t0", *MONITORS}


def test_flags_off_parity_record_has_no_new_keys_and_reproduces_base_formula():
    ds = _brake_dataset()
    torch.manual_seed(2)
    tr, cfg = _trainer()
    rec = tr._forward_backward(_batch(ds), unroll_steps=20, backward=False)
    assert set(rec) == BASE_KEYS_OFF
    base = cfg.position_weight * rec["position"] + cfg.speed_weight * rec["speed"] \
        + cfg.lane_weight * rec["lane"]
    assert rec["loss"] == pytest.approx(base, rel=1e-6)
    # explicitly neutral values are the same as the defaults: identical loss
    torch.manual_seed(2)
    tr2, _ = _trainer(asym_speed_alpha=1.0, decel_weight=1.0, propagation_weight=0.0,
                      decel_threshold_mps2=2.0, prop_window_s=9.0)
    rec2 = tr2._forward_backward(_batch(ds), unroll_steps=20, backward=False)
    assert rec2["loss"] == rec["loss"] and set(rec2) == set(rec)


def test_all_flags_on_adds_terms_logs_events_and_keeps_gradients_finite():
    ds = _brake_dataset()
    torch.manual_seed(2)
    tr, cfg = _trainer(asym_speed_alpha=3.0, decel_weight=6.0, decel_threshold_mps2=3.0,
                       propagation_weight=2.0, prop_window_s=3.0,
                       residual_cap_mps2=1.5, residual_cap_neg_mps2=3.0)
    assert tr.model.residual_cap_neg_mps2 == 3.0
    tr.optimizer.zero_grad(set_to_none=True)
    rec = tr._forward_backward(_batch(ds), unroll_steps=20)
    for name in ("asym_speed", "decel_weight", "propagation"):
        assert name in rec and math.isfinite(rec[name]) and rec[name] >= 0.0
    assert rec["decel_weight"] > 0.0 and rec["propagation"] > 0.0
    # leader brakes at 4 transitions in 2 windows -> 8 braking steps in the loss
    assert rec["brake_steps"] == 8
    # follower (slot 1) window = transitions 6..(8+6)=14 -> 9 steps x 2 windows (lane-0
    # followers 2 and 3? slot 2's leader is slot 1 which does NOT brake -> not an event)
    assert rec["prop_steps"] == 2 * 9
    base = cfg.position_weight * rec["position"] + cfg.speed_weight * rec["speed"] \
        + cfg.lane_weight * rec["lane"]
    extra = rec["asym_speed"] + rec["decel_weight"] + rec["propagation"]
    assert rec["loss"] == pytest.approx(base + extra, rel=1e-5)
    grads = [p.grad for p in tr.model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    # optimizer step stays finite
    tr.train_batch(_batch(ds), unroll_steps=20)


def test_each_flag_alone_adds_only_its_own_term():
    ds = _brake_dataset()
    cases = {"asym_speed": dict(asym_speed_alpha=2.0),
             "decel_weight": dict(decel_weight=4.0),
             "propagation": dict(propagation_weight=1.0)}
    for name, kw in cases.items():
        torch.manual_seed(2)
        tr, _ = _trainer(**kw)
        rec = tr._forward_backward(_batch(ds), unroll_steps=20, backward=False)
        new = set(rec) - BASE_KEYS_OFF
        assert name in new
        assert not ({"asym_speed", "decel_weight", "propagation"} - {name}) & new


def test_heavier_weights_raise_loss_monotonically():
    ds = _brake_dataset()
    losses = []
    for w in (1.0, 3.0, 9.0):
        torch.manual_seed(2)
        tr, _ = _trainer(decel_weight=w)
        losses.append(tr._forward_backward(_batch(ds), unroll_steps=20,
                                           backward=False)["loss"])
    assert losses[0] < losses[1] < losses[2]


def test_loss_share_logging_includes_enabled_braking_terms():
    ds = _brake_dataset()
    torch.manual_seed(2)
    tr, _ = _trainer(asym_speed_alpha=3.0, decel_weight=6.0, propagation_weight=2.0,
                     loss_share_logging=True)
    rec = tr._forward_backward(_batch(ds), unroll_steps=20, backward=False, share=True)
    share = rec["loss_share"]
    for name in ("asym_speed", "decel_weight", "propagation"):
        assert name in share and 0.0 <= share[name] <= 1.0
    assert sum(share.values()) == pytest.approx(1.0, abs=1e-6)


def test_entries_padding_slots_do_not_break_propagation():
    ds = _brake_dataset()
    torch.manual_seed(2)
    tr, _ = _trainer(propagation_weight=1.0, entries_v2=True, entry_pad=3)
    from output.wd01.wm_n1.wms.entries import EntranceHistory

    batch = _batch(ds)
    batch["entry_feat"] = torch.as_tensor(np.tile(
        EntranceHistory.defaults().to_features().astype(np.float32), (2, 1)))
    rec = tr._forward_backward(batch, unroll_steps=20, backward=False)
    assert rec["prop_steps"] == 18 and math.isfinite(rec["propagation"])


# ------------------------------------------------------------ CLI / gate / audit
def test_cli_plumbs_every_braking_flag_into_the_trainer_config(monkeypatch, tmp_path):
    from output.wd01.wm_n1.wms import cli

    ds = _brake_dataset()
    seen = {}

    class _Fake:
        def __init__(self, model, cfg):
            seen["cfg"], seen["model"] = cfg, model

        def fit(self, dataset, val_dataset=None):
            return {"status": "ok"}

    monkeypatch.setattr(cli, "_dataset", lambda *a, **k: ds)
    monkeypatch.setattr(cli, "WmSTrainer", _Fake)
    rc = cli.main(["train", "--pack", "p", "--identities", "i", "--out", str(tmp_path),
                   "--asym-speed-loss", "3", "--decel-weight", "6", "--decel-threshold", "2.5",
                   "--propagation-loss", "1.5", "--prop-window", "4",
                   "--residual-cap-mps2", "1.5", "--residual-cap-neg-mps2", "3"])
    assert rc == 0
    cfg = seen["cfg"]
    assert (cfg.asym_speed_alpha, cfg.decel_weight, cfg.decel_threshold_mps2) == (3.0, 6.0, 2.5)
    assert (cfg.propagation_weight, cfg.prop_window_s) == (1.5, 4.0)
    assert (cfg.residual_cap_mps2, cfg.residual_cap_neg_mps2) == (1.5, 3.0)
    assert cfg.brakefix_config().window_steps == 8
    # defaults: everything off
    seen.clear()
    cli.main(["train", "--pack", "p", "--identities", "i", "--out", str(tmp_path)])
    assert not seen["cfg"].brakefix_config().any_on and seen["cfg"].residual_cap_neg_mps2 is None


def _gate_rows(extra_keys):
    rows = []
    for i, stage in enumerate((10.0, 30.0)):
        row = {"step": i + 1, "stage_s": stage, "finite": True, "gate_monotone": True,
               "gate_g0": 0.0, "gate_curve": [0.0, 0.1, 0.2], "valid_targets": 5,
               "position": 1.0, "speed": 0.5, "lane": 0.1, "loss": 1.6}
        row.update({k: 0.01 for k in extra_keys})
        rows.append(row)
    return rows


def test_smoke_gate_requires_enabled_braking_terms(tmp_path):
    from output.wd01.wm_n1.wms import gate_s2, gate_stages

    log = tmp_path / "train_log.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in _gate_rows(
        ("asym_speed", "decel_weight", "propagation"))), encoding="utf-8")
    ok = gate_s2.check(log, stages=(10, 30), steps_per_stage=1, required_terms=gate_s2.BRAKEFIX_TERMS,
                       driver_params=False)
    assert ok["status"] == "passed"
    res = gate_stages.evaluate(log, stages=(10.0, 30.0),
                               terms=("position", "speed", "lane") + gate_stages.BRAKEFIX_TERMS)
    assert res["passed"], res["failures"]
    missing = tmp_path / "missing.jsonl"
    missing.write_text("".join(json.dumps(r) + "\n" for r in _gate_rows(("asym_speed",))),
                       encoding="utf-8")
    with pytest.raises(SystemExit):
        gate_s2.check(missing, stages=(10, 30), steps_per_stage=1,
                      required_terms=gate_s2.BRAKEFIX_TERMS, driver_params=False)
    res2 = gate_stages.evaluate(missing, stages=(10.0, 30.0),
                                terms=("position",) + gate_stages.BRAKEFIX_TERMS)
    assert not res2["passed"] and any("decel_weight" in f for f in res2["failures"])


def test_audit_accumulator_counts_unreachable_braking():
    from output.wd01.wm_n1.wms import brakefix_audit as au

    t1, k = 5, 3
    states = np.zeros((t1, k, 7), dtype=np.float32)
    states[..., 1] = 1.0
    states[..., 6] = 5.0
    states[..., 0] = np.arange(k, 0, -1)[None] * 50.0
    states[1:, 0, 5] = [-1.0, -3.0, -5.0, -6.5]           # slot 0 recorded braking
    valid = np.ones((t1, k), dtype=bool)
    cav = np.zeros((t1, k), dtype=bool)
    cav[:, 2] = True
    acc = au.Accumulator()
    acc.add(states, valid, cav)
    rep = acc.report()
    assert rep["hdv_steps"] == 8 and rep["steps_below"]["-4"] == 2
    by = {s["setting"]: s for s in rep["settings"]}
    hyb = by["hybrid / cap 1.5"]
    assert hyb["global_lower_envelope_mps2"] == pytest.approx(-5.7)
    assert hyb["within_global_lower_envelope_share_of_braking"]["a<-4"] == pytest.approx(0.5)
    wide = by["hybrid / negative cap 3"]
    assert wide["within_global_lower_envelope_share_of_braking"]["a<-4"] == pytest.approx(1.0)
    assert au.pick_indices(10, 4) == [0, 3, 6, 9] and au.pick_indices(0, 4) == []


def test_eval_plugin_replays_the_asymmetric_cap(tmp_path, monkeypatch):
    pkg = Path(__file__).resolve().parents[2]            # output/wd01/wm_n1
    monkeypatch.syspath_prepend(str(pkg))
    from wms_eval.wms_plugin import WmsCheckpointModel

    model = WmSModel(physics=PhysicsConfig(), history_features=6)
    ckpt = tmp_path / "wm_s.pt"
    torch.save({"format": "wm_s/1", "model": model.state_dict(),
                "config": {"history_features": 6, "residual_cap_mps2": 1.5,
                           "residual_cap_neg_mps2": 3.0}}, ckpt)
    loaded = WmsCheckpointModel(ckpt)._load()
    assert loaded.residual_cap_mps2 == 1.5 and loaded.residual_cap_neg_mps2 == 3.0
    plain = tmp_path / "plain.pt"
    torch.save({"format": "wm_s/1", "model": model.state_dict(),
                "config": {"history_features": 6, "residual_cap_mps2": 1.5}}, plain)
    old = WmsCheckpointModel(plain)._load()
    # main's eval2 integration already restores the symmetric receipt cap.
    assert old.residual_cap_neg_mps2 is None and old.residual_cap_mps2 == 1.5


@pytest.mark.parametrize("mode", ["residual", "driver_params", "hybrid"])
@pytest.mark.parametrize("baseline", ["6ebcf95", "ec6f948"])
def test_flags_off_matches_pre_fix_source_exactly(mode, baseline, monkeypatch):
    """Compare loss, deterministic log fields and gradients to the actual base commit."""
    pkg = "output.wd01.wm_n1.wms"
    root = Path(__file__).resolve().parents[5]
    source = subprocess.check_output(
        ["git", "show", f"{baseline}:output/wd01/wm_n1/wms/train.py"], cwd=root,
        text=True, encoding="utf-8")
    module = types.ModuleType(pkg + "._brakefix_base_train")
    module.__file__ = str(Path(__file__).resolve().parents[1] / "train.py")
    module.__package__ = pkg
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    kwargs = dict(device="cpu", threads=1, batch_size=2, unroll_curriculum_s=(10.0,),
                  truncation_steps=5, rollout_mode=mode, traffic_loss=True,
                  traffic_ramp_start_s=0.0, traffic_ramp_end_s=1.0,
                  residual_cap_mps2=1.5, loss_share_logging=True)
    torch.manual_seed(23)
    current = WmSTrainer(_small_model(rollout_mode=mode), WmSTrainConfig(**kwargs))
    torch.manual_seed(23)
    before = module.WmSTrainer(_small_model(rollout_mode=mode), module.WmSTrainConfig(**kwargs))
    current_config = current.config.to_dict()
    if baseline == "6ebcf95":
        # Earlier hybrid source predates main's existing sampler receipt field.
        assert current_config.pop("switch_window_fraction") == 0.0
    assert current_config == before.config.to_dict()
    batch = _batch(_brake_dataset())
    a = current._forward_backward(batch, unroll_steps=20, backward=False, share=True)
    b = before._forward_backward(batch, unroll_steps=20, backward=False, share=True)
    # R3 adds mandatory always-on monitoring; legacy loss/diagnostic fields
    # and every gradient must still match exactly.
    assert {k: v for k, v in a.items() if k not in MONITORS} == b
    assert all(math.isfinite(a[k]) for k in MONITORS)
    current._forward_backward(batch, unroll_steps=20)
    before._forward_backward(batch, unroll_steps=20)
    for p, q in zip(current.model.parameters(), before.model.parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad)


@pytest.mark.parametrize("initial,raw,projected", [(1.0, -4.0, None), (10.0, -2.0, None),
                                                   (10.0, -2.0, 7.0)])
def test_propagation_acceleration_uses_realized_stopping_and_projection(initial, raw, projected):
    start = torch.tensor([[initial]])
    predicted = (start + 0.5 * raw).clamp_min(0.0)
    if projected is not None:
        predicted = predicted.clamp_max(projected)
    speed = predicted.unsqueeze(1).requires_grad_(True)
    realized = bf.realized_acceleration(speed, start, 0.5)
    target_a = (predicted - start).unsqueeze(1) / 0.5
    term = bf.propagation_terms(speed, realized, speed.detach(), target_a,
                                torch.ones_like(speed, dtype=torch.bool),
                                torch.ones_like(speed), torch.tensor(1.0),
                                speed_delta=1.0, accel_delta=1.0)
    assert term.item() == 0.0
    term.backward()
    assert torch.isfinite(speed.grad).all()


def test_window_dt_and_cross_chunk_lookback():
    assert WmSTrainConfig(prop_window_s=3.0).brakefix_config(dt_s=0.25).window_steps == 12
    ev = torch.zeros(1, 12, 1, dtype=torch.bool)
    ev[:, 4] = True
    full = bf.propagation_window(ev, 6)
    assert full[:, 5:11].all() and not full[:, 4].any() and not full[:, 11].any()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_new_hyperparameters_refuse(bad):
    with pytest.raises(ValueError, match="finite"):
        bf.BrakefixConfig(prop_weight=bad)
    with pytest.raises(ValueError, match="finite"):
        bf.bounded_value_asym(torch.ones(1), 1.5, bad)
    with pytest.raises(ValueError, match="finite"):
        WmSTrainConfig(residual_cap_neg_mps2=bad)
    with pytest.raises(ValueError, match="finite"):
        _small_model(residual_cap_neg_mps2=bad)


@pytest.mark.parametrize("enabled", [("asym_speed_alpha", "asym_speed"),
                                     ("decel_weight", "decel_weight"),
                                     ("propagation_weight", "propagation")])
def test_gates_auto_require_enabled_terms_on_every_row(tmp_path, enabled):
    from output.wd01.wm_n1.wms import gate_s2, gate_stages
    flag, term = enabled
    (tmp_path / "BRAKEFIX_CONFIG.json").write_text(json.dumps({flag: 2.0}), encoding="utf-8")
    path = tmp_path / "train_log.jsonl"
    rows = _gate_rows((term,))
    rows[0][term] = 0.0  # no braking in a batch is legitimate
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert gate_stages.evaluate(path, stages=(10, 30), terms=("position",))["passed"]
    assert gate_s2.check(path, stages=(10, 30), steps_per_stage=1,
                         required_terms=(), driver_params=False)["status"] == "passed"
    del rows[1][term]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert not gate_stages.evaluate(path, stages=(10, 30), terms=("position",))["passed"]
    with pytest.raises(SystemExit, match=term):
        gate_s2.check(path, stages=(10, 30), steps_per_stage=1,
                      required_terms=(), driver_params=False)


def test_cli_train_cpu_all_stages_synthetic_smoke(tmp_path, monkeypatch):
    """Actual cli.main(train argv) and fit/optimizer/gates; synthetic input only."""
    from output.wd01.wm_n1.wms import cli, gate_s2, gate_stages
    ds = _brake_dataset(n=2, t1=601)
    monkeypatch.setattr(cli, "_dataset", lambda *a, **k: ds)
    monkeypatch.setattr(cli, "_build_model", lambda args, physics, cfg: _small_model(
        rollout_mode=cfg.rollout_mode))
    out = tmp_path / "smoke"
    argv = ["train", "--pack", "synthetic", "--identities", "synthetic",
            "--out", str(out), "--device", "cpu", "--threads", "1", "--num-workers", "0",
            "--batch-size", "1", "--stage-steps", "1,1,1,1,1", "--rollout-mode", "hybrid",
            "--asym-speed-loss", "2", "--decel-weight", "5", "--decel-threshold", "3",
            "--propagation-loss", "1", "--prop-window", "3", "--residual-cap-mps2", "1.5",
            "--residual-cap-neg-mps2", "3", "--loss-share-logging"]
    for stage in (10, 30, 60, 120, 300):
        argv.extend(["--unroll-s", str(stage)])
    assert cli.main(argv) == 0
    receipt = json.loads((out / "TRAIN_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["steps"] == 5 and receipt["device"] == "cpu"
    path = out / "train_log.jsonl"
    assert gate_stages.evaluate(path, terms=("position", "speed", "lane"))["passed"]
    assert gate_s2.check(path, stages=(10, 30, 60, 120, 300), steps_per_stage=1,
                         required_terms=())["status"] == "passed"
