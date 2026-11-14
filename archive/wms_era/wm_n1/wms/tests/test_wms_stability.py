"""Synthetic unit tests for the WM-S run-2 stability features (no real data).

Covers the owner-requested checks: the feasibility projection removes overlaps
and keeps gradients finite, the tanh cap is respected, the stability
regulariser is zero for an identity-safe residual and positive for an
amplifying one, and vehicle-count conservation holds with no entries/exits.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from hdv.config import HdvConfig
from hdv.net import HdvResidualNet
from output.wd01.wm_n1.wms.model import PhysicsConfig, WmSModel
from output.wd01.wm_n1.wms.train import TrajectoryDataset, WmSTrainer, collate
from output.wd01.wm_n1.wms.stability import (ConservationConfig,
                                             ConservationTracker,
                                             FeasibilityProjection,
                                             ProjectionConfig, apply_spectral_norm,
                                             bounded_value, combine_losses,
                                             error_growth_diagnostic,
                                             per_horizon_error,
                                             residual_amplification,
                                             stability_regulariser)
from output.wd01.wm_n1.wms.train import WmSTrainConfig


def small_model(**kwargs):
    return WmSModel(
        physics=PhysicsConfig(max_gap_steps=8),
        hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16),
        **kwargs,
    )


def state(batch=2, slots=4):
    lane = torch.zeros(batch, slots, dtype=torch.long)
    lane[:, slots // 2:] = 1
    one_hot = torch.nn.functional.one_hot(lane, 3).float()
    kin = torch.cat((torch.arange(slots).float().view(1, slots, 1).expand(batch, -1, -1),
                     one_hot, torch.full((batch, slots, 1), 12.0),
                     torch.zeros(batch, slots, 2)), dim=-1)
    return {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
            "a_mps2": kin[..., 5], "length_m": kin[..., 6], "lane": lane,
            "present": torch.tensor([[1, 1, 1, 0]] * batch, dtype=torch.bool),
            "is_cav": torch.zeros(batch, slots, dtype=torch.bool)}


def overlap_state():
    """Two HDVs on lane 0 whose bumper gap is negative (-4 m)."""
    s = torch.tensor([[10.0, 11.0]])
    v = torch.tensor([[5.0, 5.0]])
    lane = torch.zeros(1, 2, dtype=torch.long)
    present = torch.ones(1, 2, dtype=torch.bool)
    length = torch.full((1, 2), 5.0)
    return s, v, lane, present, length


# --------------------------------------------------------------- (A) projection
def test_projection_removes_overlap_and_keeps_gradients_finite():
    s, v, lane, present, length = overlap_state()
    s = s.requires_grad_(True)
    proj = FeasibilityProjection(ProjectionConfig(min_gap_m=2.0))
    out = proj(s, v, lane, present, length)
    sp = out["s_m"]
    gap = sp[0, 1] - sp[0, 0] - length[0, 1]
    assert float(gap.detach()) >= 2.0 - 1.0e-4
    assert out["active"]["gap_active"] >= 1
    assert torch.isfinite(out["s_m"]).all() and torch.isfinite(out["v_mps"]).all()
    (sp.sum() + out["v_mps"].sum()).backward()
    assert s.grad is not None and bool(torch.isfinite(s.grad).all())


def test_projection_leaves_feasible_state_unchanged():
    s = torch.tensor([[0.0, 40.0]])
    v = torch.tensor([[10.0, 10.0]])
    lane = torch.zeros(1, 2, dtype=torch.long)
    present = torch.ones(1, 2, dtype=torch.bool)
    length = torch.full((1, 2), 5.0)
    out = FeasibilityProjection()(s, v, lane, present, length)
    torch.testing.assert_close(out["s_m"], s)
    assert out["active"]["gap_active"] == 0
    assert out["active"]["present_changed"] == 0


def test_projection_clamps_speed_and_logs_active_constraints():
    s = torch.tensor([[0.0, 40.0]])
    v = torch.tensor([[99.0, 10.0]])           # first is far over v_max
    lane = torch.zeros(1, 2, dtype=torch.long)
    present = torch.ones(1, 2, dtype=torch.bool)
    length = torch.full((1, 2), 5.0)
    proj = FeasibilityProjection(ProjectionConfig(v_max_mps=31.29))
    out = proj(s, v, lane, present, length)
    assert float(out["v_mps"].max()) <= 31.29 + 1.0e-6
    assert out["active"]["speed_high"] == 1
    assert proj.stats["speed_high"] == 1.0
    assert proj.summary()["steps"] == 1.0


def test_projection_never_moves_a_teacher_forced_cav():
    s = torch.tensor([[10.0, 11.0]])
    v = torch.tensor([[5.0, 5.0]])
    lane = torch.zeros(1, 2, dtype=torch.long)
    present = torch.ones(1, 2, dtype=torch.bool)
    length = torch.full((1, 2), 5.0)
    is_cav = torch.tensor([[False, True]])       # CAV is the leader
    out = FeasibilityProjection()(s, v, lane, present, length, is_cav=is_cav)
    assert float(out["s_m"][0, 1]) == 11.0        # CAV untouched
    assert float(out["s_m"][0, 0]) <= 11.0 - 5.0 - 2.0 + 1.0e-4


def _overlap_rollout_state():
    current = state(batch=1, slots=2)
    current["present"] = torch.ones(1, 2, dtype=torch.bool)
    current["lane"] = torch.zeros(1, 2, dtype=torch.long)
    current["veh_kin"] = None
    current["s_m"] = torch.tensor([[10.0, 11.0]], requires_grad=True)
    current["length_m"] = torch.full((1, 2), 5.0)
    return current


def test_projected_step_is_feasible_and_gradients_flow():
    model = small_model()
    current = _overlap_rollout_state()
    stay = torch.full((1, 2, 3), -10.0)
    stay[..., 0] = 10.0                          # force both to stay on lane 0
    proj = FeasibilityProjection(ProjectionConfig(min_gap_m=2.0))
    out = model.step(current, baseline_lane_logits=stay, projection=proj)
    assert out["projection_active"]["gap_active"] >= 1
    pos = out["next_s_m"]
    gap = pos[0, 1] - pos[0, 0] - 5.0
    assert float(gap.detach()) >= 2.0 - 1.0e-3
    pos.sum().backward()
    assert bool(torch.isfinite(current["s_m"].grad).all())


def test_rollout_with_projection_reports_stats_and_keeps_gradients_finite():
    model = small_model()
    current = _overlap_rollout_state()
    proj = FeasibilityProjection(ProjectionConfig(min_gap_m=2.0))
    out = model.rollout(current, n_steps=3, projection=proj)
    assert "projection_stats" in out
    # per-lane invariant: any two present vehicles sharing a lane keep the gap
    for step in range(out["positions"].shape[1]):
        s = out["positions"][0, step]
        lane = out["lanes"][0, step]
        present = out["masks"][0, step]
        for ln in torch.unique(lane[present]).tolist():
            sel = torch.nonzero(present & (lane == ln), as_tuple=False).flatten()
            if sel.numel() < 2:
                continue
            order = sel[torch.argsort(s[sel])]
            gaps = s[order[1:]] - s[order[:-1]] - 5.0
            assert float(gaps.min().detach()) >= 2.0 - 1.0e-2
    out["positions"].sum().backward()
    assert bool(torch.isfinite(current["s_m"].grad).all())


def test_flag_off_rollout_has_no_projection_stats():
    model = small_model()
    out = model.rollout(state(), n_steps=4)
    assert "projection_stats" not in out


def test_projection_inert_on_feasible_state_matches_no_projection():
    model = small_model()
    model.eval()
    current = state(batch=1, slots=3)
    current["present"] = torch.ones(1, 3, dtype=torch.bool)
    current["lane"] = torch.zeros(1, 3, dtype=torch.long)
    current["veh_kin"] = None
    current["s_m"] = torch.tensor([[0.0, 100.0, 200.0]])
    current["v_mps"] = torch.full((1, 3), 10.0)
    current["length_m"] = torch.full((1, 3), 5.0)
    current["is_cav"] = torch.zeros(1, 3, dtype=torch.bool)
    base = model.rollout({k: v for k, v in current.items()}, n_steps=4)
    proj = FeasibilityProjection(ProjectionConfig(min_gap_m=2.0))
    with_proj = model.rollout({k: v for k, v in current.items()}, n_steps=4,
                              projection=proj)
    torch.testing.assert_close(base["positions"], with_proj["positions"])
    torch.testing.assert_close(base["speeds"], with_proj["speeds"])
    assert proj.stats["gap_active"] == 0.0 and proj.stats["speed_high"] == 0.0


# ------------------------------------------------------------------- (B) caps
def test_tanh_cap_is_respected():
    x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
    y = bounded_value(x, 1.5)
    assert float(y.abs().max()) <= 1.5 + 1.0e-6
    torch.testing.assert_close(bounded_value(x, None), x)


def test_model_residual_and_lane_caps_are_respected():
    model = small_model(residual_cap_mps2=1.5, lane_logit_cap=1.0)
    current = state()
    k = torch.ones(2, 4, dtype=torch.long)
    out = model(current, k_steps=k)
    accel_diff = (out["accel_mps2"] - out["physics_accel_mps2"]).abs()
    assert float(accel_diff.max().detach()) <= 1.5 + 1.0e-5
    base = model.mobil_merge_logits(current)
    lane_diff = (out["lane_logits"] - base).abs()
    assert float(lane_diff.max().detach()) <= 1.0 + 1.0e-5
    assert model.residual_cap_mps2 == 1.5 and model.lane_logit_cap == 1.0


def test_default_model_has_no_caps():
    model = small_model()
    assert model.residual_cap_mps2 is None and model.lane_logit_cap is None


# ------------------------------------------------- (B) stability regulariser
class _Amplifier(torch.nn.Module):
    def forward(self, kin, present):
        # strong sensitivity to the speed channel -> amplification >> 1
        return {"accel_delta_mps2": kin[..., 4] * 1000.0}


def _kin():
    return state(batch=1)["veh_kin"]


def test_regulariser_zero_for_identity_safe_net():
    net = HdvResidualNet(HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16,
                                   zero_residual=True))
    present = torch.ones(1, 4, dtype=torch.bool)
    amp = residual_amplification(net, _kin(), present)
    assert float(amp) == 0.0
    assert float(stability_regulariser(net, _kin(), present)) == 0.0


def test_regulariser_positive_for_amplifying_net():
    present = torch.ones(1, 4, dtype=torch.bool)
    gen = torch.Generator().manual_seed(0)
    amp = residual_amplification(_Amplifier(), _kin(), present, generator=gen)
    assert float(amp) > 1.0
    reg = stability_regulariser(_Amplifier(), _kin(), present, margin=1.0,
                                generator=torch.Generator().manual_seed(0))
    assert float(reg) > 0.0


def test_spectral_norm_option_runs_and_is_idempotent():
    net = HdvResidualNet(HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16))
    apply_spectral_norm(net)
    apply_spectral_norm(net)                     # must not double-wrap
    out = net(_kin(), torch.ones(1, 4, dtype=torch.bool))
    assert bool(torch.isfinite(out["accel_delta_mps2"]).all())


# ------------------------------------------------------------ (D) conservation
def test_conservation_holds_with_no_entries_or_exits():
    present = torch.ones(1, 4, dtype=torch.bool)
    present[:, 3] = False
    tracker = ConservationTracker(present)
    for _ in range(5):
        violation = tracker.update(present)
        assert int(violation.abs().sum()) == 0
    assert float(tracker.loss()) == 0.0


def test_conservation_flags_undeclared_creation():
    present = torch.zeros(1, 3, dtype=torch.bool)
    present[0, 0] = True
    tracker = ConservationTracker(present)
    appeared = present.clone()
    appeared[0, 1] = True
    undeclared = torch.zeros_like(appeared)      # entry2 mask says "no entrant"
    violation = tracker.update(appeared, entrant_mask=undeclared)
    assert int(violation[0]) == 1
    assert int(tracker.created[0]) == 1


def test_conservation_accepts_declared_entrants_and_exits():
    present = torch.zeros(1, 3, dtype=torch.bool)
    present[0, 0] = True
    tracker = ConservationTracker(present)
    step1 = present.clone()
    step1[0, 1] = True
    entrant = torch.zeros_like(step1)
    entrant[0, 1] = True
    assert int(tracker.update(step1, entrant_mask=entrant)[0]) == 0
    step2 = step1.clone()
    step2[0, 0] = False                           # an exit
    assert int(tracker.update(step2)[0]) == 0
    res = tracker.result(torch.zeros(1, dtype=torch.long), step2)
    assert res.entered_total.tolist() == [1]
    assert res.exited_total.tolist() == [1]
    assert res.violation_abs_max == 0.0


def test_conservation_track_rollout_accepts_entry2_mask():
    masks = torch.ones(1, 3, 2, dtype=torch.bool)
    entrant_seq = torch.zeros(1, 3, 2, dtype=torch.bool)
    tracker = ConservationTracker(masks[:, 0])
    res = tracker.track_rollout(masks, entrant_seq=entrant_seq)
    assert res.violation_abs_max == 0.0


def test_conservation_config_default_off():
    assert ConservationConfig().enabled is False


# ------------------------------------------------------------ (D) error growth
def test_error_growth_diagnostic_recovers_exponential_rate():
    diag = error_growth_diagnostic({10.0: 1.0, 20.0: 2.0, 30.0: 4.0})
    assert diag["ratios"] == [1.0, 2.0, 4.0]
    assert abs(diag["growth_rate"] - math.log(2.0) / 10.0) < 1.0e-6
    assert abs(diag["fit_r2"] - 1.0) < 1.0e-9


def test_error_growth_handles_zero_and_empty():
    diag = error_growth_diagnostic({10.0: 0.0, 20.0: 1.0})
    assert diag["ratios"][0] is None
    assert error_growth_diagnostic({})["growth_rate"] is None


def test_per_horizon_error_masks_and_averages():
    pred = torch.zeros(1, 3, 2)
    target = torch.tensor([[[1.0, 3.0], [2.0, 4.0], [5.0, 5.0]]])
    valid = torch.tensor([[[True, False], [True, True], [False, False]]])
    err = per_horizon_error(pred, target, valid)
    torch.testing.assert_close(err, torch.tensor([1.0, 3.0, 0.0]))


# ----------------------------------------------------------------- loss hooks
def test_combine_losses_accepts_entry2_and_trafloss_terms():
    base = torch.tensor(1.0)
    cons = torch.tensor(4.0)
    assert float(combine_losses(base, conservation=cons,
                                conservation_weight=0.25)) == 2.0
    extra = {"section_speed": torch.tensor(2.0)}
    out = combine_losses(base, extra_terms=extra,
                         extra_weights={"section_speed": 0.5})
    assert float(out) == 2.0
    assert float(combine_losses(base)) == 1.0


def test_trainer_runs_with_all_run2_flags_on():
    n, t1, slots = 2, 21, 4
    base = state(batch=n, slots=slots)["veh_kin"].numpy()
    states = np.repeat(base[:, None], t1, axis=1)
    states[..., 0] += np.arange(t1, dtype=np.float32)[None, :, None] * 6.0
    states[..., 4] = 12.0
    present = np.ones((n, t1, slots), dtype=bool)
    present[:, :, -1] = False
    dataset = TrajectoryDataset.from_arrays({"states": states, "present": present})
    cfg = WmSTrainConfig(device="cpu", threads=4, steps_per_epoch=2,
                         unroll_curriculum_s=(10.0,), batch_size=2,
                         projection_enabled=True, residual_cap_mps2=1.5,
                         lane_logit_cap=1.0, stability_weight=0.1,
                         conservation_weight=0.1)
    trainer = WmSTrainer(small_model(), cfg)
    batch = collate([dataset[0], dataset[1]])
    record = trainer.train_batch(batch, unroll_steps=20)
    assert np.isfinite(record["loss"])
    assert "stability" in record and "conservation" in record
    assert record["stability"] >= 0.0 and record["conservation"] >= 0.0


def test_trainer_receipt_unchanged_when_flags_off():
    n, t1, slots = 1, 11, 3
    base = state(batch=n, slots=slots)["veh_kin"].numpy()
    states = np.repeat(base[:, None], t1, axis=1)
    states[..., 0] += np.arange(t1, dtype=np.float32)[None, :, None] * 6.0
    states[..., 4] = 12.0
    present = np.ones((n, t1, slots), dtype=bool)
    present[:, :, -1] = False
    dataset = TrajectoryDataset.from_arrays({"states": states, "present": present})
    cfg = WmSTrainConfig(device="cpu", threads=4, steps_per_epoch=1,
                         unroll_curriculum_s=(5.0,), batch_size=1)
    trainer = WmSTrainer(small_model(), cfg)
    record = trainer.train_batch(collate([dataset[0]]), unroll_steps=10)
    assert "conservation" not in record and "stability" not in record


def test_train_config_run2_flags_default_off():
    cfg = WmSTrainConfig()
    assert cfg.projection_enabled is False
    assert cfg.residual_cap_mps2 is None and cfg.lane_logit_cap is None
    assert cfg.stability_weight == 0.0 and cfg.conservation_weight == 0.0
    # round-trips through the receipt contract
    assert WmSTrainConfig.from_dict(cfg.to_dict()) == cfg
