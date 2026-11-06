"""A5 long-horizon aggregate delay loss: config plumbing, loss algebra and a CPU smoke.

No SUMO, no GPU, no evaluation seeds: the loss-module tests build synthetic windows and the
smoke runs a few Adam steps on 2 small deterministic TRAIN-val-style windows, asserting the
aggregate loss decreases and its gradient reaches the WM dynamics (``residual_net``).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from wms.config.load import load_train_config
from wms.config.legacy_flat import FLAT_TO_PATH, OMIT_WHEN_DEFAULT
from wms.config.train import TrainConfig
from wms.model.residual_net import HdvConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.train.losses import agg_horizon as ah
from wms.train.losses.aggregate import road_targets
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.compose import build_loss
from wms.train.losses.paired_diff import PairTargets
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

NEW_FLAT_KEYS = ("agg_delay_weight", "agg_delay_paired_weight", "agg_delay_start_s",
                 "agg_delay_horizon_s", "agg_delay_fade_s", "agg_delay_bin_s",
                 "agg_delay_sections_m")


# --------------------------------------------------------------------------- fixtures
def _recorded(b=2, t=40, k=9, seed=0):
    """Recorded window spanning upstream / merge / ramp / downstream, with a slow ramp queue."""
    g = torch.Generator().manual_seed(seed)
    s0 = torch.linspace(-260.0, 360.0, k).repeat(b, 1)
    v = 4.0 + 22.0 * torch.rand(b, 1, k, generator=g).expand(b, t + 1, k).clone()
    v[:, :, k - 2:] = 1.0                                     # ramp queue
    s = s0[:, None, :] + torch.cumsum(
        torch.cat((torch.zeros(b, 1, k), v[:, :-1] * 0.5), 1), 1)
    lane = torch.zeros(b, t + 1, k, dtype=torch.long)
    lane[..., k - 2:] = 2                                     # ramp lane
    lane[..., 4] = 1
    s = torch.where(lane == 2, s.clamp(max=70.0), s)
    states = torch.zeros(b, t + 1, k, 7)
    states[..., 0] = s
    states[..., 1:4] = torch.nn.functional.one_hot(lane, 3).float()
    states[..., 4] = v
    states[..., 6] = 5.0
    road = torch.ones(b, t + 1, k, dtype=torch.bool)
    cont = road[:, :-1] & road[:, 1:]
    return states, road, cont, lane


def _targets(states, road, cont, cohort, *, paired=False):
    b, t1, k = cohort.shape
    n = t1 - 1
    batch = {"states": states}
    rt = road_targets(batch, states, cohort, k, entries_on=False, device="cpu")
    targets = UnrollTargets(
        states=states, n_steps=n, weights=torch.ones(1, n, 1) / n,
        hdv_mask=cohort[:, 1:], aux_mask=cohort[:, 1:], aux_count=0,
        denom=torch.tensor(1.0), lane_target=torch.zeros(b, n, k, dtype=torch.long),
        loss_valid=cohort, is_cav=torch.zeros_like(cohort), dt_s=0.5, position_delta_m=5.0,
        speed_delta_mps=1.0, history_lane_head=False, road=rt)
    if paired:
        ref = torch.zeros(b, dtype=torch.long)                # row 0 is everyone's reference
        arm = torch.arange(b) != ref
        targets.pair = PairTargets(ref=ref, arm=arm, road=rt)
    return targets


def _ctx(targets, s, v, lane, mask, c0=0, c1=None):
    c1 = targets.n_steps if c1 is None else c1
    pred = {"positions": s[:, c0 + 1:c1 + 1], "speeds": v[:, c0 + 1:c1 + 1],
            "lanes": lane[:, c0 + 1:c1 + 1], "masks": mask[:, c0 + 1:c1 + 1]}
    current = {"s_m": s[:, c0], "present": mask[:, c0], "lane": lane[:, c0]}
    return LossContext(pred=pred, current=current, targets=targets, c0=c0, c1=c1)


def _cfg(**kw):
    base = dict(dt_s=0.5, weight=1.0, start_s=0.0, horizon_s=0.0, bin_s=2.0,
                queue_scale_veh=2.0, throughput_scale_veh=2.0, delay_scale_veh_s=1.0)
    base.update(kw)
    return ah.AggHorizonConfig(**base)


# ------------------------------------------------------------------- config plumbing
def test_config_defaults_off_and_flat_unchanged():
    flat = TrainConfig().to_legacy_flat()
    assert not set(NEW_FLAT_KEYS) & set(flat)
    assert "agg_delay_weight" in OMIT_WHEN_DEFAULT
    assert FLAT_TO_PATH["agg_delay_weight"] == "loss.agg_delay.weight"
    assert build_loss(TrainSettings()).names == ("position", "speed", "lane")
    assert TrainSettings().agg_delay_weight == 0.0 and not TrainSettings().agg_delay_on


def test_config_reaches_settings_and_composer():
    s = TrainSettings(agg_delay_weight=1.0, agg_delay_start_s=120.0, agg_delay_horizon_s=300.0)
    names = build_loss(s).names
    assert names[-3:] == ("agg_delay_region", "agg_queue", "agg_throughput")
    s2 = TrainSettings(agg_delay_paired_weight=0.5)
    assert build_loss(s2).names[-1:] == ("agg_pair",)
    with pytest.raises(ValueError):
        TrainSettings(agg_delay_weight=1.0, agg_delay_bin_s=0.0)
    with pytest.raises(ValueError):
        TrainSettings(agg_delay_merge_s1_m=-1.0)


def test_to_legacy_flat_roundtrips_the_new_keys():
    cfg = load_train_config("configs/c4_wm_v5m_refresh.toml",
                            ["loss.agg_delay.weight=0.2", "loss.agg_delay.sections_m=[-400.0, 0.0]"])
    flat = cfg.to_legacy_flat()
    assert flat["agg_delay_weight"] == 0.2 and flat["agg_delay_sections_m"] == [-400.0, 0.0]
    back = TrainConfig.from_legacy_flat(flat)
    assert back.loss.agg_delay == cfg.loss.agg_delay
    settings = TrainSettings.from_config(cfg)
    assert settings.agg_delay_weight == 0.2 and settings.agg_delay_sections_m == (-400.0, 0.0)


def test_required_log_terms_lists_the_new_terms():
    cfg = load_train_config("configs/c4_wm_v5m_refresh.toml", ["loss.agg_delay.weight=1.0"])
    terms = set(cfg.required_log_terms())
    assert {"agg_delay_region", "agg_queue", "agg_throughput"} <= terms
    assert "agg_pair" not in terms
    cfg2 = load_train_config("configs/c4_wm_v5m_refresh.toml",
                             ["loss.agg_delay.paired_weight=1.0"])
    assert "agg_pair" in cfg2.required_log_terms()


# --------------------------------------------------------------------------- algebra
def test_window_weight_zero_outside_the_long_window():
    cfg = _cfg(start_s=120.0, horizon_s=300.0)
    idx = torch.tensor([0, 200, 240, 600, 601, 700])
    w = cfg.window_weight(idx)
    np.testing.assert_allclose(w.numpy(), [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    faded = _cfg(start_s=120.0, horizon_s=300.0, fade_s=40.0).window_weight(
        torch.tensor([200, 240, 280, 320, 600]))              # t = 100, 120, 140, 160, 300
    assert float(faded[0]) == 0.0 and float(faded[-1]) == 0.0
    assert 0.0 < float(faded[2]) < 1.0 and float(faded[3]) == pytest.approx(1.0)


def test_regions_partition_the_road():
    s = torch.tensor([[-150.0, 40.0, 40.0, 200.0, 40.0]]).unsqueeze(1)   # [B,T,K]
    lane = torch.tensor([[0, 0, 2, 1, 2]]).unsqueeze(1)
    present = torch.ones(1, 1, 5, dtype=torch.bool)
    out = ah.region_delay_queue(s, torch.full_like(s, 10.0), lane, present,
                                cfg=_cfg(region_temp_m=0.01))
    counts = out["count"][0, 0]
    assert float(counts.sum()) == pytest.approx(5.0, abs=1e-2)   # 5 on-road vehicles
    np.testing.assert_allclose(counts.numpy(), [1.0, 1.0, 2.0, 1.0], atol=1e-2)


@pytest.mark.parametrize("paired", [False, True])
def test_zero_on_identical_trajectories(paired):
    states, road, cont, lane = _recorded()
    cohort = road.clone()
    targets = _targets(states, road, cont, cohort, paired=paired)
    cfg = _cfg(paired_weight=1.0)
    loss = ah.AggHorizonLoss(cfg)
    loss.prepare(targets)
    out = loss.components(_ctx(targets, states[..., 0], states[..., 4], lane, road))
    expected = ah.TERM_NAMES if paired else ah.TERM_NAMES[:3]
    assert set(out) == set(ah.TERM_NAMES)
    for name in expected:
        assert float(out[name]) == 0.0, name


@pytest.mark.parametrize("paired", [False, True])
def test_positive_and_gradients_reach_the_prediction(paired):
    states, road, cont, lane = _recorded()
    cohort = road.clone()
    targets = _targets(states, road, cont, cohort, paired=paired)
    cfg = _cfg(paired_weight=1.0)
    loss = ah.AggHorizonLoss(cfg)
    loss.prepare(targets)
    g = torch.Generator().manual_seed(1)
    s = (states[..., 0] + 10.0 * torch.randn(states[..., 0].shape, generator=g)).requires_grad_()
    v = (states[..., 4] * 0.6).requires_grad_()
    total = 0.0
    for c0, c1 in ((0, 20), (20, 40)):                        # chunked like truncated BPTT
        out = loss.components(_ctx(targets, s, v, lane, road, c0, c1))
        assert all(float(x.detach()) >= 0.0 for x in out.values())
        total = total + sum(out.values())
    assert float(total.detach()) > 0.0
    total.backward()
    for leaf in (s, v):
        assert leaf.grad is not None and bool(torch.isfinite(leaf.grad).all())
        assert float(leaf.grad.abs().sum()) > 0.0


def test_chunk_sum_equals_single_chunk_with_aligned_bins():
    states, road, cont, lane = _recorded(t=40)
    cohort = road.clone()
    targets = _targets(states, road, cont, cohort)
    loss = ah.AggHorizonLoss(_cfg(bin_s=2.0))                 # 4-step bins
    loss.prepare(targets)
    s = states[..., 0] + 4.0
    v = states[..., 4] * 0.8
    whole = loss.components(_ctx(targets, s, v, lane, road))
    parts = [loss.components(_ctx(targets, s, v, lane, road, a, b))
             for a, b in ((0, 20), (20, 40))]                 # bins align with chunks
    for name in ah.TERM_NAMES[:3]:
        torch.testing.assert_close(parts[0][name] + parts[1][name], whole[name])


def test_paired_term_is_zero_without_pair_groups():
    states, road, cont, lane = _recorded()
    cohort = road.clone()
    targets = _targets(states, road, cont, cohort)            # no targets.pair
    loss = ah.AggHorizonLoss(_cfg(paired_weight=1.0))
    loss.prepare(targets)
    out = loss.components(_ctx(targets, states[..., 0], states[..., 4], lane, road))
    assert float(out["agg_pair"]) == 0.0


# --------------------------------------------------------------------------- CPU smoke
def _smoke_batch(b=2, t=30, k=6):
    """Two deterministic TRAIN-val-style windows: lane 0/1/2 spread over the four regions."""
    g = torch.Generator().manual_seed(0)
    lane = torch.zeros(b, t, k, dtype=torch.long)
    lane[..., 1:4] = 0
    lane[..., 4:] = 2
    lane[..., 5] = 1
    states = torch.zeros(b, t, k, 7)
    states[..., 0] = torch.linspace(-120.0, 320.0, k).repeat(b, 1).unsqueeze(1) \
        + 12.0 * torch.rand(b, t, k, generator=g)
    states[..., 1:4] = torch.nn.functional.one_hot(lane, 3).float()
    states[..., 4] = 4.0 + 20.0 * torch.rand(b, t, k, generator=g)
    states[..., 6] = 5.0
    present = torch.ones(b, t, k, dtype=torch.bool)
    return {"states": states, "present": present,
            "is_cav": torch.zeros(b, t, k, dtype=torch.bool),
            "cav_teacher": torch.zeros(b, t, k, 4),
            "hist": torch.zeros(b, k, 10, 6), "hist_mask": torch.ones(b, k, 10, dtype=torch.bool)}


def _smoke_settings(**kw):
    base = dict(agg_delay_weight=1.0, agg_delay_start_s=0.0, agg_delay_horizon_s=0.0,
                agg_delay_bin_s=1.0, agg_delay_queue_scale_veh=2.0,
                agg_delay_throughput_scale_veh=2.0)
    base.update(kw)
    return TrainSettings(**base)


def test_cpu_smoke_aggregate_loss_decreases_and_reaches_dynamics():
    torch.manual_seed(0)
    model = WmSModel(ModelSpec(), hdv=HdvConfig(d_model=16, n_heads=2, n_layers=1, ffn=32))
    settings = _smoke_settings(position_weight=0.0, speed_weight=0.0, lane_weight=0.0)
    runner = UnrollRunner(model, settings, build_loss(settings, physics=model.physics),
                          projection=None, device=torch.device("cpu"))
    batch = _smoke_batch(b=2, t=20)
    opt = torch.optim.Adam(model.parameters(), lr=5.0e-3)
    losses = []
    for step in range(5):
        opt.zero_grad()
        record = runner.run(batch, unroll_steps=20, backward=True)
        agg = float(record["agg_delay_region"] + record["agg_queue"]
                    + record["agg_throughput"])
        assert np.isfinite(agg)
        losses.append(agg)
        opt.step()
        if step == 0:
            # the aggregate gradient reaches the WM dynamics (the residual network)
            dyn = [p.grad for p in model.residual_net.parameters() if p.grad is not None]
            assert dyn and any(float(g.abs().sum()) > 0.0 for g in dyn)
    assert losses[-1] < losses[0]
    assert min(losses[1:]) < losses[0]
