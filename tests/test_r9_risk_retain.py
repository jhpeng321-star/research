"""R9LANE2: ``physics.ramp_risk_retain`` - a crawling queue head at the zone start keeps its chain."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from wms.cli.assemble import build_physics
from wms.config.load import load_train_config
from wms.model import merge_pivot as pivot
from wms.physics.config import PhysicsConfig
from wms.train.losses.merge_pivot import MergePivotConfig, certify_pre_window

ROOT = Path(__file__).resolve().parents[1]


def b(*rows):
    return torch.tensor([[[bool(v)] for v in rows]]).transpose(0, 1).reshape(1, len(rows), 1)


def test_close_risk_bridges_only_a_started_chain_that_stays_open():
    risk, opened = b(0, 1, 0, 1, 1), b(1, 1, 1, 1, 1)
    assert pivot.close_risk(risk, opened).flatten().tolist() == [False, True, True, True, True]
    # never started: an open car is not at risk
    assert pivot.close_risk(b(0, 0, 1), b(1, 1, 1)).flatten().tolist() == [False, False, True]
    # a real exit (not open: lane change / past the end) ends the chain for good
    assert pivot.close_risk(b(1, 0, 0, 1), b(1, 0, 1, 1)).flatten().tolist() == [True, False, False, True]
    # a fresh identity ends the chain
    assert pivot.close_risk(b(1, 0, 1), b(1, 1, 1), reset=b(0, 1, 0)).flatten().tolist() == [True, False, True]
    # no change when nothing flickers
    assert pivot.close_risk(b(1, 1, 0), b(1, 1, 0)).flatten().tolist() == [True, True, False]


def kin(s, lane=2, v=0.3, length=5.0):
    row = torch.zeros(1, 1, 7)
    row[..., 0], row[..., 1 + lane], row[..., 4], row[..., 6] = s, 1.0, v, length
    return row


def test_geometry_ramp_open_has_no_start_bound_but_keeps_lane_and_end():
    physics = PhysicsConfig()
    present = torch.ones(1, 1, dtype=torch.bool)
    end = physics.merge_zone_s1_m
    for s, lane, ramp, opened in ((-0.004, 2, False, True), (0.02, 2, True, True),
                                  (end + 0.05, 2, False, False), (-0.004, 1, False, False)):
        geo = pivot.geometry(kin(s, lane), present, physics)
        assert bool(geo["ramp"][0, 0]) is ramp and bool(geo["ramp_open"][0, 0]) is opened, (s, lane)
    # open never makes a car eligible
    geo = pivot.geometry(kin(-0.004), present, physics)
    assert not bool(geo["eligible"][0, 0])


def test_update_state_retain_keeps_a_started_chain_and_never_starts_one():
    tau, c = torch.tensor(1.0), torch.tensor([[-0.5]])
    present, yes, no = torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.bool)
    u = torch.tensor([[0.5]])
    first = pivot.update_state(c, tau, yes, present, uniform=u)
    assert bool(first["started"]) and not bool(first["terminal"])
    gap = torch.full_like(c, -torch.inf)  # the flicker tick is physically ineligible
    dropped = pivot.update_state(gap, tau, no, present, carry=first, uniform=u)
    assert bool(dropped["terminal"])  # legacy: absorbing exit
    kept = pivot.update_state(gap, tau, no, present, carry=first, uniform=u, retain=yes)
    assert not bool(kept["terminal"]) and torch.equal(kept["maximum"], first["maximum"])
    after = pivot.update_state(c, tau, yes, present, carry=kept, uniform=u)
    assert not bool(after["terminal"]) and torch.equal(after["z"], first["z"])
    never = pivot.update_state(gap, tau, no, present, uniform=u, retain=yes)
    assert not bool(never["started"]) and not bool(never["terminal"])
    gone = pivot.update_state(gap, tau, no, present, carry=first, uniform=u, retain=no)
    assert bool(gone["terminal"])  # a real exit still terminates


def synthetic_prefix(retain):
    """Rows: on the ramp, flicker (WM position < 0, recorded still before the zone), back, t0."""
    h = 5
    rec = torch.zeros(1, h + 1, 1, 7)
    for i, s in enumerate((-3.0, -0.1, -0.05, 0.04, 0.5, 1.0)):
        rec[:, i, 0, 0], rec[:, i, 0, 3], rec[:, i, 0, 6] = s, 1.0, 5.0
    present = torch.ones(1, h + 1, 1, dtype=torch.bool)
    cont = torch.ones(1, h, 1, dtype=torch.bool)
    ramp = b(0, 1, 0, 1, 1)  # predicted next position crosses 0 only on some ticks
    geo = {"ramp": ramp.clone(), "ramp_open": b(1, 1, 1, 1, 1), "eligible": ramp.clone()}
    cfg = MergePivotConfig()
    return certify_pre_window(rec, present, cont, geo, cfg, retain=retain)


def test_certify_pre_window_chain_survives_the_flicker_only_with_the_flag():
    off, on = synthetic_prefix(False), synthetic_prefix(True)
    assert off["history_mask"].flatten().tolist() == [False, True, False, False, False]
    assert not bool(off["certified"])  # the chain was absorbed at the flicker
    assert on["history_mask"].flatten().tolist() == [False, True, False, True, True]
    assert bool(on["certified"])  # the chain (and its prior maximum) reaches t0
    assert not bool(off["history_mask"][0, 3:].any())


def test_flag_default_off_and_round9_preset_pin():
    assert PhysicsConfig().ramp_risk_retain is False
    physics = build_physics(load_train_config(str(ROOT / "configs/c4_wm_v10round9.toml"), []))
    assert physics.ramp_risk_retain is True and physics.ramp_lane_end_stop is True
    stored = json.loads((ROOT / "configs/r9_physics_round9.json").read_text(encoding="utf-8"))
    assert stored == asdict(physics)
    assert build_physics(load_train_config(str(ROOT / "configs/c4_wm_v10round8.toml"), [])).ramp_risk_retain is False
    with pytest.raises(ValueError):
        PhysicsConfig(ramp_risk_retain=1)
