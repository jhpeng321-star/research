"""C4 lane rule: PhysicsConfig.lane_rule = 'mobil' (default, unchanged) vs
'forced_merge_only' (no discretionary MOBIL changes; only the mandatory ramp
merge lane 2 -> 1 inside the merge zone with the existing gap safety + urgency).

Covers the physics masking, the model-side re-mask (the learned residual lane
delta must never unmask a forbidden change, even under the lane-logit cap), the
config plumbing (ModelConfig / legacy flat with OMIT_WHEN_DEFAULT /
TrainSettings / ModelSpec / build_physics) and the default-mode bit-identity.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from wms.cli.assemble import build_physics
from wms.config.train import TrainConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.physics.mobil import MASKED_LOGIT, mobil_merge_logits
from wms.train.settings import TrainSettings


def _kin(rows: list[tuple[float, int, float]]) -> torch.Tensor:
    """``[1, K, 7]`` kinematics from ``(s, lane, v)`` triples (deterministic)."""
    kin = torch.zeros(1, len(rows), 7)
    for k, (s, lane, v) in enumerate(rows):
        kin[0, k, 0] = s
        kin[0, k, 1:4] = 0.0
        kin[0, k, 1 + lane] = 1.0
        kin[0, k, 4] = v
        kin[0, k, 5] = 0.0
        kin[0, k, 6] = 5.0
    return kin


def _scene() -> torch.Tensor:
    """A ramp scene with one discretionary MOBIL change and one forced merge.

    Slot layout (merge zone [0, 75.44] m, lane 2 -> 1 mandatory inside it):
      0: lane-1 car at 100 m, 25 m/s  (mainline, free to change discretely)
      1: slow lane-1 leader at 112 m, 5 m/s (gives slot 0 a strong incentive)
      2: lane-2 car at 60 m, 20 m/s   (inside the merge zone, clear lane 1)
      3: lane-0 car at 150 m, 25 m/s  (mainline left lane, free)
    """
    return _kin([(100.0, 1, 25.0), (112.0, 1, 5.0), (60.0, 2, 20.0), (150.0, 0, 25.0)])


def _logits(lane_rule: str) -> torch.Tensor:
    physics = PhysicsConfig(lane_rule=lane_rule)
    assert physics.lane_rule == lane_rule
    return mobil_merge_logits(_scene(), torch.ones(1, 4, dtype=torch.bool), physics)


# --------------------------------------------------------------------- physics
def test_mobil_is_the_default_and_validates():
    assert PhysicsConfig().lane_rule == "mobil"
    with pytest.raises(ValueError):
        PhysicsConfig(lane_rule="bogus")


def test_mobil_default_logits_are_unchanged_golden():
    """Default-mode logits are bit-identical to the pre-lane_rule MOBIL path."""
    logits = _logits("mobil")
    explicit = mobil_merge_logits(
        _scene(), torch.ones(1, 4, dtype=torch.bool), PhysicsConfig(lane_rule="mobil"))
    assert torch.equal(logits, explicit)
    # Golden values captured on the legacy path (2026-10-12): the lane-1 car
    # behind a slow leader wants the discretionary move to lane 0; the lane-2
    # car inside the merge zone wants the forced merge; stay logits are 0.
    assert torch.equal(logits[0, 0, 1], torch.tensor(0.0))
    assert float(logits[0, 0, 0]) > -2.5          # discretionary left change open
    assert float(logits[0, 2, 1]) > 0.0           # forced merge open (urgency > 0)
    golden = (float(logits[0, 0, 0]), float(logits[0, 0, 2]), float(logits[0, 2, 1]),
              float(logits[0, 2, 0]), float(logits[0, 3, 1]), float(logits[0, 1, 0]))
    assert golden == (0.4427475929260254, -8.0, 4.270153999328613,
                      -8.0, -2.574984312057495, -2.5841727256774902)


def test_forced_merge_only_masks_every_discretionary_change():
    logits = _logits("forced_merge_only")
    # Lane-1 car: both discretionary directions (to 0 and to 2) masked.
    assert float(logits[0, 0, 0]) == MASKED_LOGIT
    assert float(logits[0, 0, 2]) == MASKED_LOGIT
    # Lane-0 and lane-1 cars can never leave their lane except lane 2 -> 1.
    assert float(logits[0, 3, 1]) == MASKED_LOGIT
    assert float(logits[0, 1, 0]) == MASKED_LOGIT
    # Stay logits are untouched (0) and the forced merge target stays open.
    assert torch.equal(logits[0, 0, 1], torch.tensor(0.0))
    assert float(logits[0, 2, 1]) > 0.0


def test_forced_merge_only_keeps_the_mobil_merge_logit_exactly():
    """Where a change stays allowed, its logit (urgency included) is unchanged."""
    mobil, forced = _logits("mobil"), _logits("forced_merge_only")
    assert float(forced[0, 2, 1]) == float(mobil[0, 2, 1])
    assert torch.equal(mobil[0, 2, 1], forced[0, 2, 1])


def test_forced_merge_only_outside_zone_and_unsafe_gaps_stay_masked():
    kin = _kin([(100.0, 2, 20.0),          # lane 2 but PAST the merge zone
                (30.0, 2, 20.0),           # in-zone merger
                (36.0, 1, 20.0)])          # blocker 6 m ahead in lane 1
    present = torch.ones(1, 3, dtype=torch.bool)
    physics = PhysicsConfig(lane_rule="forced_merge_only")
    logits = mobil_merge_logits(kin, present, physics)
    assert float(logits[0, 0, 1]) == MASKED_LOGIT   # outside the zone
    # Gap to the lane-1 blocker is 6 m - 5 m length = 1 m < lane_safe_gap 6 m.
    assert float(logits[0, 1, 1]) == MASKED_LOGIT   # unsafe gap masked


def test_forced_merge_only_fuzz_only_forced_changes_survive():
    torch.manual_seed(0)
    physics_m = PhysicsConfig()
    physics_f = PhysicsConfig(lane_rule="forced_merge_only")
    for _ in range(20):
        kin = torch.zeros(1, 6, 7)
        kin[..., 0] = torch.rand(1, 6) * 200
        lanes = torch.randint(0, 3, (1, 6))
        kin[..., 1:4] = torch.nn.functional.one_hot(lanes, 3).float()
        kin[..., 4] = torch.rand(1, 6) * 25 + 3
        kin[..., 6] = 5.0
        present = torch.ones(1, 6, dtype=torch.bool)
        mobil = mobil_merge_logits(kin, present, physics_m)
        forced = mobil_merge_logits(kin, present, physics_f)
        lane = lanes[0]
        for target in range(3):
            change = lane != target                  # the stay diagonal is not a change
            open_m = change & (mobil[0, :, target] > MASKED_LOGIT + 1.0)
            open_f = change & (forced[0, :, target] > MASKED_LOGIT + 1.0)
            # forced-merge mode only ever closes doors: open_f => open_m.
            assert not bool((open_f & ~open_m).any())
            is_forced = (lane == physics_f.merge_lane) & (target == physics_f.merge_target_lane)
            in_zone = (kin[0, :, 0] >= physics_f.merge_zone_s0_m) & \
                      (kin[0, :, 0] <= physics_f.merge_zone_s1_m)
            assert bool((~open_f | (is_forced & in_zone)).all())
            # an open forced change keeps the MOBIL value exactly
            both = open_f
            assert torch.equal(mobil[0, :, target][both], forced[0, :, target][both])


# ---------------------------------------------------------------------- model
def _biased_model(lane_rule: str, lane_logit_cap: float = 1.5) -> WmSModel:
    """Hybrid model whose temporal head pushes every lane logit up hard."""
    torch.manual_seed(2)
    spec = ModelSpec(rollout_mode="hybrid", history_s=2.0, gate_kind="saturating",
                     lane_gate_kind="saturating", lane_logit_cap=lane_logit_cap)
    model = WmSModel(spec, physics=PhysicsConfig(lane_rule=lane_rule))
    with torch.no_grad():
        model.temporal[-1].bias.fill_(6.0)   # large positive lane-logit deltas
    model.eval()
    return model


def _model_state() -> tuple[dict, dict]:
    state = {"s_m": torch.tensor([[100.0, 112.0, 60.0, 150.0]]),
             "lane": torch.tensor([[1, 1, 2, 0]]),
             "v_mps": torch.tensor([[25.0, 5.0, 20.0, 25.0]]),
             "a_mps2": torch.zeros(1, 4),
             "present": torch.ones(1, 4, dtype=torch.bool)}
    kwargs = {"driver_history": torch.zeros(1, 4, 3, 9),
              "driver_history_mask": torch.ones(1, 4, 3, dtype=torch.bool)}
    return state, kwargs


def test_residual_lane_delta_cannot_unmask_a_forbidden_change():
    model = _biased_model("forced_merge_only")
    state, kwargs = _model_state()
    out = model(state, k_steps=60, **kwargs)
    base = model.mobil_merge_logits(state)
    forbidden = base <= MASKED_LOGIT
    assert bool(forbidden.any())
    # Masked entries stay exactly masked after the residual correction and cap.
    assert torch.equal(out["lane_logits"][forbidden],
                       torch.full_like(out["lane_logits"][forbidden], MASKED_LOGIT))
    # Consequence: argmax never picks a forbidden change for present vehicles.
    choice = out["lane_logits"].argmax(-1)
    assert not bool(forbidden.gather(-1, choice.unsqueeze(-1)).any())
    assert choice[0].tolist() == [1, 1, 1, 0]


def test_residual_lane_delta_still_corrects_allowed_entries():
    model = _biased_model("forced_merge_only")
    state, kwargs = _model_state()
    out = model(state, k_steps=60, **kwargs)
    base = model.mobil_merge_logits(state)
    allowed = base > MASKED_LOGIT
    if bool(allowed.any()):
        assert not torch.equal(out["lane_logits"][allowed], base[allowed])


def test_mobil_mode_bit_identical_to_lane_rule_free_physics():
    """Default mode: an explicit lane_rule='mobil' changes no output bit."""
    torch.manual_seed(9)
    spec = ModelSpec(gate_kind="saturating", lane_logit_cap=1.5)
    plain = WmSModel(spec)
    torch.manual_seed(9)                             # identical initial weights
    explicit = WmSModel(spec, physics=PhysicsConfig(lane_rule="mobil"))
    for model in (plain, explicit):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.mul_(0.5).add_(0.1)
    state, kwargs = _model_state()
    a = plain(state, k_steps=30, **kwargs)
    b = explicit(state, k_steps=30, **kwargs)
    for key in ("lane_logits", "accel_mps2", "physics_accel_mps2", "gate"):
        assert torch.equal(a[key], b[key]), key


def test_forced_merge_only_rollout_never_leaves_a_nonramp_lane():
    model = _biased_model("forced_merge_only")
    state, kwargs = _model_state()
    result = model.rollout(state, n_steps=40, **kwargs)
    lanes = result["lanes"][0]                     # [T, K]
    start = torch.tensor([1, 1, 2, 0]).unsqueeze(0)
    # Slots starting on lanes 0/1 (and any vehicle not merging from lane 2)
    # must stay on their lane; the lane-2 slot may only move to lane 1.
    fixed = start[0] != 2
    assert torch.equal(lanes[:, fixed], start[0][fixed].expand(40, -1))
    assert not bool((lanes[:, 2] == 2).any()) or torch.equal(
        lanes[:, 2], torch.full((40,), 2))
    assert not bool((lanes[:, 2] == 0).any())      # never jumps 2 -> 0


# ------------------------------------------------------------------- plumbing
def test_model_config_flat_roundtrip_omits_default():
    default = TrainConfig()
    assert "lane_rule" not in default.to_legacy_flat()
    assert default.model.lane_rule == "mobil"
    forced = TrainConfig.from_dict({"model": {"lane_rule": "forced_merge_only"}})
    flat = forced.to_legacy_flat()
    assert flat["lane_rule"] == "forced_merge_only"
    assert TrainConfig.from_legacy_flat(flat).to_legacy_flat() == flat
    with pytest.raises(Exception):
        TrainConfig.from_dict({"model": {"lane_rule": "bogus"}})


def test_train_settings_and_spec_validate_lane_rule():
    assert TrainSettings.from_flat({"lane_rule": "forced_merge_only"}).lane_rule \
        == "forced_merge_only"
    assert TrainSettings().lane_rule == "mobil"
    with pytest.raises(ValueError):
        TrainSettings.from_flat({"lane_rule": "bogus"})
    assert ModelSpec.from_mapping({"lane_rule": "forced_merge_only"}).lane_rule \
        == "forced_merge_only"
    assert ModelSpec().lane_rule == "mobil"
    with pytest.raises(ValueError):
        ModelSpec(lane_rule="bogus")


def test_build_physics_carries_lane_rule(tmp_path: Path):
    cfg = TrainConfig.from_dict({"model": {"lane_rule": "forced_merge_only"}})
    physics = build_physics(cfg, resolver=lambda name: str(tmp_path / name))
    assert physics.lane_rule == "forced_merge_only"
    assert build_physics(TrainConfig()).lane_rule == "mobil"
    # A merge-cal params file can never inject a lane rule (caller decides).
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"lane_rule": "bogus", "mobil_threshold_mps2": 0.2}),
                      encoding="utf-8")
    cfg = TrainConfig.from_dict({"model": {"merge_cal": True,
                                           "physics_params": "merge_params",
                                           "lane_rule": "forced_merge_only"}})
    physics = build_physics(cfg, resolver=lambda name: str(params))
    assert physics.lane_rule == "forced_merge_only"
    assert physics.merge_cal is True
    assert physics.mobil_threshold_mps2 == pytest.approx(0.2)


def test_checkpoint_physics_override_accepts_lane_rule(tmp_path: Path):
    from wms.eval.backend import load_backend

    torch.manual_seed(3)
    cfg = TrainConfig.from_dict({"model": {"gate": {"kind": "saturating"}}})
    model = WmSModel(ModelSpec.from_mapping(cfg.to_legacy_flat()))
    payload = {"format": "wm_s/1", "model": model.state_dict(),
               "config": cfg.to_legacy_flat()}
    path = tmp_path / "wm_s.pt"
    torch.save(payload, path)
    backend = load_backend(path, physics={"lane_rule": "forced_merge_only"})
    assert backend.model.physics.lane_rule == "forced_merge_only"
    backend = load_backend(path)
    assert backend.model.physics.lane_rule == "mobil"


def test_lane_rule_field_not_in_model_state_dict():
    torch.manual_seed(4)
    keys = list(WmSModel(ModelSpec(lane_rule="forced_merge_only"),
                         physics=PhysicsConfig(lane_rule="forced_merge_only")).state_dict())
    assert not [key for key in keys if "lane_rule" in key]


# ------------------------------------------------ learned_discretionary
def test_learned_discretionary_physics_alone_never_changes_lane_but_opens_safe_changes():
    from wms.physics.mobil import LEARNED_PRIOR_LOGIT
    learned, forced = _logits("learned_discretionary"), _logits("forced_merge_only")
    # The forced merge is the same as forced_merge_only; physics argmax never picks a
    # discretionary change (prior < stay = 0).
    torch.testing.assert_close(learned[0, 2], forced[0, 2])
    choice = learned.argmax(-1)[0].tolist()
    assert choice[0] == 1 and choice[3] == 0
    # Slot 0 (lane 1) can move to lane 0 safely: opened at the neutral prior.
    assert learned[0, 0, 0] == LEARNED_PRIOR_LOGIT
    # Never into the ramp lane.
    assert learned[0, 0, 2] == MASKED_LOGIT


def test_learned_discretionary_residual_can_trigger_a_safe_change():
    model = _biased_model("learned_discretionary", lane_logit_cap=8.0)  # c4_wm cap
    with torch.no_grad():                      # push only the 'move to lane 0' logit
        model.temporal[-1].bias.zero_()
        model.temporal[-1].bias[1] = 80.0   # lane gate ~0.14 at k=60: 0.14*80 > 6
    state, kwargs = _model_state()
    out = model(state, k_steps=60, **kwargs)
    base = model.mobil_merge_logits(state)
    forbidden = base <= MASKED_LOGIT
    torch.testing.assert_close(out["lane_logits"][forbidden],
                               torch.full_like(out["lane_logits"][forbidden], MASKED_LOGIT))
    choice = out["lane_logits"].argmax(-1)[0].tolist()
    assert choice[0] == 0          # learned correction lifts the safe change above 'stay'
    assert not bool(forbidden.gather(-1, out["lane_logits"].argmax(-1).unsqueeze(-1)).any())
