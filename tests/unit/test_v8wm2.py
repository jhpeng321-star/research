"""V8WM2 (2026-10-17): lane-end hazard window (``physics.forced_window_mode``), zipper yield
(``physics.mainline_merge_yield_scope='zone'``) and the ``merge_end_wait`` follower constants.
Every switch defaults to the legacy behaviour (bit-identity: the golden lane tests)."""
from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from .test_gaprule import _kin
from .test_lanefix import _phys
from .test_lanev5 import _batch, _rollout, _sc, _single_cars, _v5_model
from wms.cli.assemble import build_physics
from wms.config.load import load_train_config
from wms.model import lane_v5 as lv5
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.physics.merge_v6 import mainline_merge_yield, merge_feasible_among, zone_mergers
from wms.physics.mobil import (MASKED_LOGIT, merge_end_blocked, mobil_merge_logits,
                               secure_gaps)
from wms.physics.neighbours import lane_of
from wms.train.losses.merge_timing import MergeTimingConfig

END = 75.44


def _hz(**kw):
    return _phys(forced_window_mode="hazard", **kw)


def _v5_logits(phys, kin, bias):
    net = lv5.LaneV5Net()
    with torch.no_grad():
        net.heads.bias.fill_(bias)
    present = torch.ones(kin.shape[:2], dtype=torch.bool)
    base = mobil_merge_logits(kin, present, phys)
    acc = lv5.initial_acc(present, kin, phys, None, None, 1)
    raw, _, inst = lv5.step_features(acc, kin, present, phys, None)
    return net, lv5.lane_logits_v5(net, raw, inst, present, base, phys)


# --------------------------------------------------------------- config / validation
def test_defaults_are_legacy_and_validation():
    p = PhysicsConfig()
    assert (p.forced_window_mode, p.mainline_merge_yield_scope) == ("deterministic", "lane_end")
    assert p.merge_end_wait_s0_follow_m is None and p.merge_end_wait_t_follow_s is None
    road = _phys()
    assert road.merge_end_follow_constants() == (road.idm()[3], road.idm()[2])
    for bad in (dict(forced_window_mode="x"), dict(forced_window_min_hazard=0.0),
                dict(forced_window_min_hazard=1.0), dict(mainline_merge_yield_scope="x"),
                dict(merge_end_wait_s0_follow_m=-1.0)):
        with pytest.raises(ValueError):
            PhysicsConfig(**bad)
    with pytest.raises(ValueError):          # zone needs the speed gap rule
        PhysicsConfig(mainline_merge_yield=True, mainline_merge_yield_scope="zone")
    with pytest.raises(ValueError):          # hazard window needs the lane-v5 model
        WmSModel(ModelSpec(), physics=_hz())


def test_configs_v8ft2_train_hazard_and_infer_zone():
    for name, steps in (("c4_wm_v8ft2", (300, 250)), ("c4_wm_v8ft_short2", (80, 60))):
        cfg = load_train_config(f"configs/{name}.toml")
        assert tuple(cfg.curriculum.stage_steps) == steps
        train = build_physics(cfg, resolver=lambda n: n)
        assert train.forced_window_mode == "hazard" and train.forced_window_min_hazard == 0.02
        assert not train.merge_end_wait and not train.mainline_merge_yield
        assert train.merge_end_follow_constants() == (1.0, 0.6)
        infer = replace(train, **cfg.physics.inference)
        assert infer.merge_end_wait and infer.mainline_merge_yield
        assert infer.mainline_merge_yield_scope == "zone" and infer.forced_window_mode == "hazard"
    old = build_physics(load_train_config("configs/c4_wm_v8ft.toml"), resolver=lambda n: n)
    assert old.forced_window_mode == "deterministic" and old.mainline_merge_yield_scope == "lane_end"
    with pytest.raises(Exception):
        load_train_config("configs/c4_wm_v8ft.toml",
                          ['physics.overrides={forced_window_mode="always"}'])


# --------------------------------------------------------------- hazard window
def test_hazard_window_uses_floored_head():
    s1 = END
    kin = _single_cars([(s1 - 2.0, 2, 1.0), (s1 - 20.0, 2, 1.0)])
    _, (lg, _, acts) = _v5_logits(_phys(), kin, -9.0)
    assert lg[0, 0, 1] >= lv5.FORCED_MERGE_LOGIT and not acts[0, 0]        # legacy: forced
    net, (lg, head, acts) = _v5_logits(_hz(), kin, -9.0)
    floor = math.log(0.02 / 0.98)
    assert float(lg[0, 0, 1]) == pytest.approx(floor, abs=1e-5) and acts[0, 0]
    assert float(head[0, 0]) == pytest.approx(floor, abs=1e-5)
    assert float(lg[1, 0, 1]) == pytest.approx(_sc(net, -9.0), abs=1e-4) and acts[1, 0]
    _, (lg, _, _) = _v5_logits(_hz(), kin, 3.0)                           # above the floor
    assert float(lg[0, 0, 1]) == pytest.approx(_sc(net, 3.0), abs=1e-4)
    # An infeasible forced-window merge stays masked (lane-1 car alongside).
    kin = _kin([(s1 - 2.0, 2, 0.0), (s1 - 2.5, 1, 0.0)])
    _, (lg, _, acts) = _v5_logits(_hz(), kin, 9.0)
    assert float(lg[0, 0, 1]) <= MASKED_LOGIT and not acts[0, 0]


def test_recorded_rows_include_window_only_in_hazard_mode():
    # One ramp car standing in the window (t0, t1), merging at t1 -> t2.
    states = torch.cat([_kin([(END - 2.0, 2, 0.0)]), _kin([(END - 2.0, 2, 0.0)]),
                        _kin([(END - 2.0, 1, 0.0)])], 0).unsqueeze(0)       # [1, 3, 1, 7]
    valid = torch.ones(1, 3, 1, dtype=torch.bool)
    cav = torch.zeros_like(valid)
    lane_age = torch.ones(1, 2, 1)
    for phys, rows, events in ((_phys(), 0, 0), (_hz(), 2, 1)):
        acc0 = lv5.initial_acc(valid[:, 0], states[:, 0], phys, None, None, 1)
        out = lv5.recorded_rows(states, valid, cav, phys, None, lane_age, acc0)
        assert (out["stats"]["rows"], out["stats"]["events"]) == (rows, events)
        assert out["stats"]["merge_rows"] == rows


def test_hazard_rollout_draws_in_window_with_argmax_decision():
    model, _ = _v5_model(['model.lane_decision="argmax"'])
    model.eval()
    assert not model._samples_lanes
    model.physics = replace(model.physics, forced_window_mode="hazard")
    assert model._samples_lanes
    b = _batch()
    roll = _rollout(model, b, lane_seed=3)
    lanes = torch.cat([b["states"][:, :1, :, 1:4].argmax(-1), roll["lanes"]], 1)
    hdv = (b["present"][:, 0] & ~b["is_cav"][:, 0]).unsqueeze(1)
    lanes = torch.where(hdv, lanes, torch.zeros_like(lanes))
    assert (lanes[:, 1:] - lanes[:, :-1]).abs().max() <= 1
    assert not ((lanes[:, :-1] == 1) & (lanes[:, 1:] == 2)).any()


def test_merge_timing_window_conditional():
    assert not MergeTimingConfig().forced_window_hazard
    assert MergeTimingConfig(forced_window_hazard=True).forced_window_hazard


# --------------------------------------------------------------- merge_end_wait follower
def test_merge_end_follow_constants_override():
    v, lv = torch.tensor([0.0]), torch.tensor([0.0])
    gl, fv = torch.tensor([50.0]), torch.tensor([2.0])
    road = _phys(merge_end_wait=True)
    # road IDM: 2.6 + 1.135 * 2 + 4 / 8.4 = 5.35 m needed; relaxed 1.0 + 0.6 * 2 + 0.48 = 2.68 m
    gf = torch.tensor([4.0])
    assert not bool(secure_gaps(v, gl, lv, gf, fv, road))
    relaxed = replace(road, merge_end_wait_s0_follow_m=1.0, merge_end_wait_t_follow_s=0.6)
    assert bool(secure_gaps(v, gl, lv, gf, fv, relaxed))
    # leader side unchanged (road s0 2.6 m)
    assert not bool(secure_gaps(v, torch.tensor([2.0]), lv, torch.tensor([50.0]), fv, relaxed))
    kin = _kin([(END - 3.0, 2, 0.0), (END - 3.0 - 5.0 - 4.0, 1, 2.0)])
    present = torch.ones(1, 2, dtype=torch.bool)
    assert bool(merge_end_blocked(kin, present, road)[0, 0])
    assert not bool(merge_end_blocked(kin, present, relaxed)[0, 0])


# --------------------------------------------------------------- zone yield
@pytest.mark.parametrize("wait,relax", [(False, True), (True, True), (True, False)])
def test_feasible_among_all_equals_physics_mask(wait, relax):
    g = torch.Generator().manual_seed(7)
    b, k = 64, 8
    kin = torch.zeros(b, k, 7)
    kin[..., 0] = torch.rand(b, k, generator=g) * 100.0 - 10.0
    lane = 1 + (torch.rand(b, k, generator=g) < 0.4).long()
    kin.scatter_(-1, (1 + lane).unsqueeze(-1), 1.0)
    kin[..., 4] = torch.rand(b, k, generator=g) * 12.0 * (torch.rand(b, k, generator=g) > 0.3)
    kin[..., 5] = torch.rand(b, k, generator=g) * 4.0 - 2.0
    kin[..., 6] = 4.0 + 2.0 * torch.rand(b, k, generator=g)
    present = torch.rand(b, k, generator=g) > 0.1
    phys = _phys(merge_end_wait=wait, gap_stopped_relax=relax)
    lane1 = present & (lane_of(kin) == 1)
    cand = lane1.unsqueeze(-2).expand(b, k, k)
    mask = (mobil_merge_logits(kin, present, phys)[..., 1] > MASKED_LOGIT) \
        & present & (lane_of(kin) == 2)
    assert torch.equal(merge_feasible_among(kin, present, phys, cand), mask)


def _yield(phys, kin):
    present = torch.ones(kin.shape[:2], dtype=torch.bool)
    params = phys.idm_params()
    from wms.physics.idm import idm_accel
    from wms.physics.neighbours import lead_neighbour
    gap, lv = lead_neighbour(kin, present, phys)
    base = idm_accel(kin[..., 4], lv, gap, params, accel_min=phys.accel_min_mps2,
                     accel_max=phys.accel_max_mps2)
    return base, mainline_merge_yield(kin, present, base, phys, params)


ON = dict(mainline_merge_yield=True, gap_stopped_relax=True)


def test_zone_follower_yields_mid_zone_and_lane_end_scope_does_not():
    # merger at 30 m (10 m/s), lane-1 follower 3 m behind its rear at 12 m/s: blocked by the
    # follower gap only (no leader).
    kin = _kin([(30.0, 2, 10.0), (22.0, 1, 12.0)])
    base, old = _yield(_phys(**ON), kin)
    assert torch.equal(old, base)
    zphys = _phys(**ON, mainline_merge_yield_scope="zone")
    assert zone_mergers(kin, torch.ones(1, 2, dtype=torch.bool), zphys).tolist() == [[True, False]]
    _, new = _yield(zphys, kin)
    assert float(new[0, 1]) < float(base[0, 1]) and float(new[0, 1]) >= -2.0 - 1e-6
    assert float(new[0, 0]) == float(base[0, 0])


def test_zone_no_yield_when_blocked_ahead():
    # Same, plus a lane-1 car overlapping the merger from ahead (front 1 m ahead of its front).
    kin = _kin([(30.0, 2, 10.0), (22.0, 1, 12.0), (31.0, 1, 10.0)])
    zphys = _phys(**ON, mainline_merge_yield_scope="zone")
    assert not zone_mergers(kin, torch.ones(1, 3, dtype=torch.bool), zphys).any()
    base, new = _yield(zphys, kin)
    assert torch.equal(new, base)


def test_zone_overlap_yield_only_for_faster_merger():
    zphys = _phys(**ON, mainline_merge_yield_scope="zone")
    kin = _kin([(30.0, 2, 10.0), (28.0, 1, 8.0)])            # overlapping, slower than merger
    base, new = _yield(zphys, kin)
    assert float(new[0, 1]) == pytest.approx(-2.0) and float(new[0, 0]) == float(base[0, 0])
    kin = _kin([(30.0, 2, 0.0), (28.0, 1, 3.0)])             # stopped merger: pass, no brake
    base, new = _yield(zphys, kin)
    assert torch.equal(new, base)


def test_zone_never_raises_accel_and_extends_lane_end():
    g = torch.Generator().manual_seed(3)
    b, k = 32, 10
    kin = torch.zeros(b, k, 7)
    kin[..., 0] = torch.rand(b, k, generator=g) * 120.0 - 30.0
    lane = 1 + (torch.rand(b, k, generator=g) < 0.4).long()
    kin.scatter_(-1, (1 + lane).unsqueeze(-1), 1.0)
    kin[..., 4] = torch.rand(b, k, generator=g) * 10.0
    kin[..., 6] = 5.0
    for wait in (False, True):
        base, old = _yield(_phys(**ON, merge_end_wait=wait), kin)
        _, new = _yield(_phys(**ON, merge_end_wait=wait, mainline_merge_yield_scope="zone"), kin)
        assert bool((old <= base + 1e-6).all()) and bool((new <= old + 1e-6).all())
        assert bool((new >= torch.minimum(base, torch.full_like(base, -2.0)) - 1e-5).all())
