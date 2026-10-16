"""GAPRULE: speed-dependent gap acceptance of the physics lane-change feasibility mask.

``PhysicsConfig.lane_gap_rule`` / ``model.lane_gap_rule``: ``fixed`` (default, legacy
6 m / merge_gap + IDM follower braking, bit-identical) or ``speed`` (SUMO-LC2013-like
secure gaps ``s0 + T v + kinematic braking term`` plus a length-aware no-overlap check)
for discretionary changes and the ramp merge.

Covers: switch-off bit-identity of the mask (golden numbers captured on the pre-GAPRULE
code, c4/int 683e83f) and of one C4 training step, the jam merge at 4 m / 0 m/s, the
overlap / non-adjacent / ramp-entry / outside-zone / high-speed-cut-in vetoes, the config
plumbing and a training step with the speed rule on.
"""
from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from wms.cli.assemble import build_physics
from wms.config.load import load_train_config
from wms.config.train import ModelConfig, TrainConfig
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.spec import ModelSpec
from wms.physics.config import PhysicsConfig
from wms.physics.mobil import (MASKED_LOGIT, anticipated_kin, mobil_merge_logits,
                               target_overlap)
from wms.train.settings import TrainSettings

ALL = torch.ones(1, 4, dtype=torch.bool)


def _kin(rows):
    """``[1, K, 7]`` kinematics from ``(s, lane, v)`` triples, 5 m cars."""
    kin = torch.zeros(1, len(rows), 7)
    for k, (s, lane, v) in enumerate(rows):
        kin[0, k, 0] = s
        kin[0, k, 1 + lane] = 1.0
        kin[0, k, 4] = v
        kin[0, k, 6] = 5.0
    return kin


def fuzz_scene(seed: int = 3, b: int = 4, k: int = 24):
    """Random dense 3-lane scene (jams and free flow) with random driver parameters."""
    g = torch.Generator().manual_seed(seed)
    kin = torch.zeros(b, k, 7)
    kin[..., 0] = torch.rand(b, k, generator=g) * 300 - 150
    lane = torch.randint(0, 3, (b, k), generator=g)
    kin[..., 1:4] = F.one_hot(lane, 3).float()
    kin[..., 4] = torch.rand(b, k, generator=g) * 30 * (torch.rand(b, k, generator=g) > 0.4)
    kin[..., 5] = torch.randn(b, k, generator=g)
    kin[..., 6] = 5.0
    present = torch.rand(b, k, generator=g) > 0.1
    low = torch.tensor([20.0, 0.8, 1.5, 1.5, 3.0, 0.0, 0.0, 2.0, 0.5])
    high = torch.tensor([35.0, 1.6, 3.0, 3.0, 5.0, 0.6, 0.4, 12.0, 1.5])
    u = torch.rand(b, k, len(PARAMETER_NAMES), generator=g)
    values = low + (high - low) * u
    params = {name: values[..., i] for i, name in enumerate(PARAMETER_NAMES)}
    return kin, present, params


def _phys(rule="learned_discretionary", gap="speed"):
    return PhysicsConfig(sumo_fidelity=True, lane_rule=rule, lane_gap_rule=gap)


def _open(logits, slot, target):
    return float(logits[0, slot, target]) > MASKED_LOGIT + 1e-6


# ----------------------------------------------------------------- bit identity
#: ``float(logits.double().sum())`` and the open-entry count of mobil_merge_logits on
#: ``fuzz_scene()`` captured with the pre-GAPRULE mobil.py (c4/int 683e83f).
GOLDEN_FIXED = {
    ("mobil", False): (-1049.0573828220367, 164),
    ("mobil", True): (-1067.8043104112148, 162),
    ("forced_merge_only", False): (-1263.3503922224045, 130),
    ("forced_merge_only", True): (-1266.32508918643, 130),
    ("learned_discretionary", False): (-1195.3503922224045, 164),
    ("learned_discretionary", True): (-1202.32508918643, 162),
}


def test_default_is_fixed_and_validates():
    assert PhysicsConfig().lane_gap_rule == "fixed"
    assert ModelConfig().lane_gap_rule == "fixed"
    assert TrainSettings().lane_gap_rule == "fixed"
    with pytest.raises(ValueError):
        PhysicsConfig(lane_gap_rule="bogus")
    with pytest.raises(ValueError):
        PhysicsConfig(lane_gap_rule="speed", gap_s0_lead_m=0.0)
    with pytest.raises(ValueError):
        TrainSettings.from_flat({"lane_gap_rule": "bogus"})
    with pytest.raises(Exception):
        TrainConfig.from_dict({"model": {"lane_gap_rule": "bogus"}})


@pytest.mark.parametrize("rule", ["mobil", "forced_merge_only", "learned_discretionary"])
@pytest.mark.parametrize("with_params", [False, True])
def test_fixed_mask_bit_identical_to_pre_gaprule_code(rule, with_params):
    kin, present, params = fuzz_scene()
    logits = mobil_merge_logits(kin, present, _phys(rule, "fixed"),
                                params if with_params else None)
    total, n_open = GOLDEN_FIXED[(rule, with_params)]
    assert float(logits.double().sum()) == total
    assert int((logits > MASKED_LOGIT + 1e-6).sum()) == n_open


def _run(overrides=(), config="configs/c4_wm.toml"):
    """test_wmlane's one-step C4 unroll, but with the physics built THROUGH the config."""
    from tests.unit.test_wmlane import _batch

    from wms.model.factory import build_model
    from wms.train.losses.compose import build_loss
    from wms.train.unroll import UnrollRunner

    cfg = load_train_config(config, ["runtime.device=cpu", *overrides])
    flat = cfg.to_legacy_flat()
    settings = TrainSettings.from_flat(flat)
    physics = build_physics(cfg)
    torch.manual_seed(0)
    model = build_model(ModelSpec.from_mapping(flat), physics)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    model.train()
    composer = build_loss(settings, physics=physics)
    runner = UnrollRunner(model, settings, composer, projection=None, device=torch.device("cpu"))
    rec = runner.run(_batch(), unroll_steps=12, backward=True, truncation_steps=5)
    return model, rec, flat, physics


def test_one_c4_training_step_bit_identical_with_switch_off():
    """The C4 recipe's training step (test_wmlane golden, pre-WMLANE numbers) is unchanged
    by GAPRULE, also with the switch spelled out explicitly."""
    from tests.unit.test_wmlane import GOLDEN_C4_SWITCH_OFF

    for overrides in ((), ('model.lane_gap_rule="fixed"',)):
        model, rec, flat, physics = _run(overrides)
        assert "lane_gap_rule" not in flat and physics.lane_gap_rule == "fixed"
        got = {"loss": rec["loss"], "position": rec["position"], "speed": rec["speed"],
               "lane": rec["lane"], "flip_rate": rec["flip_rate"],
               "rate": rec["predicted_lane_change_rate"]}
        assert got == {k: GOLDEN_C4_SWITCH_OFF[k] for k in got}
        grad = sum(float(p.grad.double().abs().sum()) for p in model.parameters()
                   if p.grad is not None)
        assert grad == pytest.approx(GOLDEN_C4_SWITCH_OFF["grad_abs_sum"], rel=1e-9)


# ------------------------------------------------------------------ speed rule
def _jam_merge(gap_lead=4.0, gap_follow=4.0, v=0.0):
    """Lane-2 car at 50 m (inside the merge zone) between a lane-1 leader and follower."""
    return _kin([(50.0, 2, v), (50.0 + gap_lead + 5.0, 1, v), (50.0 - 5.0 - gap_follow, 1, v),
                 (200.0, 0, 25.0)])


def test_speed_mode_allows_a_jam_merge_at_4m_0mps():
    kin = _jam_merge()
    assert not _open(mobil_merge_logits(kin, ALL, _phys(gap="fixed")), 0, 1)
    assert _open(mobil_merge_logits(kin, ALL, _phys(gap="speed")), 0, 1)
    # Also under the driver-parameter path (hybrid training), whatever merge_gap says.
    params = {n: torch.full((1, 4), 8.0 if n == "merge_gap" else 1.0) for n in PARAMETER_NAMES}
    assert _open(mobil_merge_logits(kin, ALL, _phys(gap="speed"), params), 0, 1)


def test_speed_mode_allows_a_jam_overtake_but_not_into_the_ramp():
    # lane-1 car crawling at 2 m/s; lane-0 neighbours 3 m ahead / 3 m behind, same speed.
    kin = _kin([(-300.0, 1, 2.0), (-292.0, 0, 2.0), (-308.0, 0, 2.0), (-280.0, 2, 2.0)])
    logits = mobil_merge_logits(kin, ALL, _phys(gap="speed"))
    assert _open(logits, 0, 0)
    assert not _open(logits, 0, 2)                      # never into the ramp lane
    assert not _open(mobil_merge_logits(kin, ALL, _phys(gap="fixed")), 0, 0)


@pytest.mark.parametrize("ds", [0.0, 1.0, -1.0, 4.9, -4.9])
def test_speed_mode_forbids_overlap(ds):
    kin = _kin([(50.0, 2, 0.0), (50.0 + ds, 1, 0.0), (300.0, 0, 25.0), (-300.0, 0, 25.0)])
    assert bool(target_overlap(kin, ALL, torch.ones(1, 4, dtype=torch.long), 0.5)[0, 0])
    assert not _open(mobil_merge_logits(kin, ALL, _phys(gap="speed")), 0, 1)


def test_speed_mode_forbids_non_adjacent_ramp_entry_and_outside_zone():
    # lane-0 car with an empty road: lane 2 is two lanes away and is the ramp.
    kin = _kin([(50.0, 0, 10.0), (60.0, 2, 10.0), (-500.0, 2, 5.0), (400.0, 1, 25.0)])
    logits = mobil_merge_logits(kin, ALL, _phys(gap="speed"))
    assert not _open(logits, 0, 2)                      # non-adjacent 0 -> 2
    assert not _open(logits, 1, 0)                      # non-adjacent 2 -> 0
    assert _open(logits, 1, 1)                          # in-zone merge, empty lane 1
    assert not _open(logits, 2, 1)                      # ramp lane upstream of the zone
    assert not _open(logits, 3, 2)                      # mainline -> ramp


def test_speed_mode_forbids_a_fast_follower_cut_in():
    # ego at 20 m/s merges 10 m ahead of a 30 m/s follower: needs (900-400)/9 + ... >> 10 m.
    kin = _kin([(50.0, 2, 20.0), (300.0, 1, 30.0), (35.0, 1, 30.0), (400.0, 0, 25.0)])
    assert not _open(mobil_merge_logits(kin, ALL, _phys(gap="speed")), 0, 1)
    # ... and into a slow leader 6 m ahead at 25 m/s vs 0 m/s.
    kin = _kin([(50.0, 1, 25.0), (61.0, 0, 0.0), (-300.0, 0, 25.0), (400.0, 2, 25.0)])
    assert not _open(mobil_merge_logits(kin, ALL, _phys(gap="speed")), 0, 0)


def test_speed_mode_is_monotone_in_gap():
    """Widening a gap never closes a speed-rule change (random scenes, discretionary;
    anticipation off so that the neighbour order cannot change with the spread)."""
    kin, present, _ = fuzz_scene(seed=11)
    phys = PhysicsConfig(sumo_fidelity=True, lane_rule="learned_discretionary",
                         lane_gap_rule="speed", gap_anticipation_s=0.0)
    base = mobil_merge_logits(kin, present, phys) > MASKED_LOGIT + 1e-6
    spread = kin.clone()
    spread[..., 0] = spread[..., 0] * 1.5
    wider = mobil_merge_logits(spread, present, phys) > MASKED_LOGIT + 1e-6
    lane = kin[..., 1:4].argmax(-1)
    mainline = (lane != 2).unsqueeze(-1).expand_as(base)     # merge zone moves with s
    assert not (base & ~wider & mainline).any()


def test_anticipated_kin_constant_acceleration_and_stop():
    kin = _kin([(10.0, 1, 10.0), (20.0, 0, 2.0), (30.0, 0, 0.0), (40.0, 1, 5.0)])
    kin[0, :, 5] = torch.tensor([2.0, -4.0, -1.0, 0.0])
    out = anticipated_kin(kin, 0.5)
    assert torch.allclose(out[0, :, 0], torch.tensor([15.25, 20.5, 30.0, 42.5]))
    assert torch.allclose(out[0, :, 4], torch.tensor([11.0, 0.0, 0.0, 5.0]))
    assert torch.equal(out[..., 1:4], kin[..., 1:4]) and torch.equal(out[..., 5:], kin[..., 5:])
    assert anticipated_kin(kin, 0.0) is kin


def test_speed_mode_judges_the_gap_after_the_step():
    """A fast car side by side NOW is ahead after the 0.5 s step (SUMO moves, then changes):
    feasible with anticipation, masked (overlap) without it."""
    kin = _kin([(50.0, 1, 8.0), (52.0, 0, 30.0), (-400.0, 0, 30.0), (400.0, 2, 25.0)])
    assert _open(mobil_merge_logits(kin, ALL, _phys(gap="speed")), 0, 0)
    off = PhysicsConfig(sumo_fidelity=True, lane_rule="learned_discretionary",
                        lane_gap_rule="speed", gap_anticipation_s=0.0)
    assert not _open(mobil_merge_logits(kin, ALL, off), 0, 0)


# ------------------------------------------------------------------- plumbing
def test_v3_config_turns_the_rule_on_and_roundtrips():
    cfg = load_train_config("configs/c4_wm_v3.toml", ["runtime.device=cpu"])
    assert cfg.model.lane_gap_rule == "speed"
    v2 = load_train_config("configs/c4_wm_v2.toml", ["runtime.device=cpu"])
    flat, flat2 = cfg.to_legacy_flat(), v2.to_legacy_flat()
    assert flat["lane_gap_rule"] == "speed" and "lane_gap_rule" not in flat2
    assert {k: v for k, v in flat.items() if k not in ("lane_gap_rule", "out_dir")} \
        == {k: v for k, v in flat2.items() if k != "out_dir"}
    assert TrainConfig.from_legacy_flat(flat).to_legacy_flat() == flat
    assert TrainSettings.from_flat(flat).lane_gap_rule == "speed"
    physics = build_physics(cfg)
    assert physics.lane_gap_rule == "speed" and physics.lane_rule == "learned_discretionary"
    assert PhysicsConfig(**asdict(physics)) == physics             # physics.json roundtrip


def test_unroll_with_speed_rule_is_finite_and_differs():
    model, rec, flat, physics = _run(['model.lane_gap_rule="speed"'])
    assert physics.lane_gap_rule == "speed" and flat["lane_gap_rule"] == "speed"
    assert torch.isfinite(torch.tensor(rec["loss"]))
    grad = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert grad > 0
