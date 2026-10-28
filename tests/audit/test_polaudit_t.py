"""S-POLAUDIT-T executable checks of the candidate-A policy TRAINING side (CPU, no SUMO).

Each test PASSES when the documented behaviour (including a defect) is reproduced, so the
file is an audit record, not a regression gate.  See POLAUDIT_T.md for interpretation.
"""
from __future__ import annotations

import json
import math
import types

import pytest
import torch

from wms.control.counterfactual import (KIND_ALL, CounterfactualConfig, build_variants,
                                        row_coefficients)

REGISTERED = {"individual": 8.9, "group": 9.9, "coop": 8.9, "team": 10.3, "all": 1.0}


def _scene(slots_s, cav):
    s = torch.tensor([slots_s], dtype=torch.float32)
    k = s.shape[1]
    return {"s_m": s, "v_mps": torch.full_like(s, 10.), "a_mps2": torch.zeros_like(s),
            "lane": torch.zeros((1, k), dtype=torch.long),
            "present": torch.ones((1, k), dtype=torch.bool),
            "is_cav": torch.tensor([cav], dtype=torch.bool), "length_m": torch.full_like(s, 5.)}


def _live_cf():
    # Live [counterfactual] section of resolved_A_MAIN1500 (see _audit/resolve.py output).
    return CounterfactualConfig(enabled=True, max_starts=1, n_individual=0, n_group=1, n_flip=3,
                                cover_group_members=True, group_max_size=3, group_state_fraction=0.,
                                native_accel_max_mps2=1.2, native_cav="head",
                                curriculum_updates=[0, 1], w_all=[1., 1.], w_individual=[8.9, 8.9],
                                w_group=[9.9, 9.9], w_coop=[8.9, 8.9], w_team=[10.3, 10.3],
                                curriculum_anchor="first_update")


def test_registered_weights_give_individual_rows_exactly_zero_weight():
    """Finding 2: w_coop == w_individual cancels every individual row; flips carry -w_team."""
    cf = _live_cf()
    state = _scene([0., 40., 80., 400., 900.], [True, True, True, False, True])
    variants = build_variants(state, cf, torch.Generator().manual_seed(0))
    kinds = variants.row_kind.tolist()
    assert kinds.count(2) == 3 and kinds.count(4) == 3 and kinds.count(3) == 1  # 3 indiv, 3 flip
    c = row_coefficients(variants, cf.weights(180), variants.n_rows, 1)
    for row in variants.individual_rows.values():
        assert float(c[row]) == pytest.approx(0.0, abs=1e-6)
    group = variants.groups[0]
    assert float(c[group["row"]]) == pytest.approx(9.9 + 8.9 + 3 * 10.3)      # 49.7
    for row in group["flips"].values():
        assert float(c[row]) == pytest.approx(-10.3)
    assert float(c[0]) == pytest.approx(1.0)
    # 3 of the 9 imagined rows (individual) are computed at 8 futures x 90 s but carry no credit.
    assert sum(1 for k in kinds if k == 2) / len(kinds) == pytest.approx(1 / 3)


def test_pair_groups_also_zero_individual_and_no_team_term():
    cf = _live_cf()
    state = _scene([0., 40., 900.], [True, True, True])
    variants = build_variants(state, cf, torch.Generator().manual_seed(0))
    c = row_coefficients(variants, cf.weights(180), variants.n_rows, 1)
    assert [round(float(c[r]), 6) for r in variants.individual_rows.values()] == [0.0, 0.0]
    assert float(c[variants.groups[0]["row"]]) == pytest.approx(9.9 + 8.9)


def test_real_credit_file_is_a_noop_for_candidate_a():
    """Finding: individual multiplier 0.401 is floored back to 8.9; others are 1.0."""
    from wms.control.policy14 import guarded_real_credit_weights
    multipliers = {"individual": 0.40108381909949953, "group": 1.0, "coop": 1.0, "team": 1.0}
    effective, receipt = guarded_real_credit_weights(REGISTERED, multipliers, individual_floor=True)
    assert effective == REGISTERED
    assert receipt["raw"]["individual"] == pytest.approx(3.569645990, rel=1e-8)


def _imagined(diff_by_future, coef):
    """One start, rows=[all, base]; difference[E,H,N]."""
    variants = types.SimpleNamespace(row_start=torch.tensor([0, 0]), base_row={0: 1})
    return {"n_base": 1, "variants": variants}, diff_by_future, coef


def test_q10_selects_the_bad_tail_and_only_those_futures_get_gradient():
    from wms.control.antiexploit import tail_credit
    values = torch.tensor([5., -40., 7., 9., 11., -10., 13., 15.])      # futures, all-row credit
    diff = torch.zeros(8, 2, 2)
    diff[:, 0, 0] = values
    diff[:, 1, 0] = values.flip(0)            # a different ranking at h=1
    diff.requires_grad_(True)
    imagined, d, coef = _imagined(diff, torch.tensor([1., 0.]))
    selected, metrics = tail_credit(imagined, d, coef, enabled=True)
    (selected.mean((0, 1)) * coef).sum().backward()
    g = diff.grad[:, :, 0]
    # h=0: q10 position 0.7 -> 0.3*lowest(-40, future 1) + 0.7*second lowest(-10, future 5)
    nz0 = {i for i in range(8) if g[i, 0] != 0}
    assert nz0 == {1, 5}
    assert g[1, 0] == pytest.approx(8 * 0.3 / 16) and g[5, 0] == pytest.approx(8 * 0.7 / 16)
    # h=1 selects DIFFERENT futures (selection is per (h, start), not per start)
    nz1 = {i for i in range(8) if g[i, 1] != 0}
    assert nz1 == {6, 2} and nz1 != nz0
    assert metrics["pessimistic_paired_credit"] == pytest.approx(0.3 * -40 + 0.7 * -10)
    assert metrics["mean_paired_credit"] == pytest.approx(float(values.mean()))


def test_secant_direct_credit_sign():
    """benefit = policy_return - native_return (returns are -J): >0 pulls toward the policy action."""
    from wms.control.looplong_real import secant_loss
    native = torch.zeros(1, 3)
    contrast = torch.tensor([[1., -1., 0.]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    residual = torch.zeros(1, 3, requires_grad=True)
    loss, n = secant_loss(native + residual, native, contrast, torch.tensor([100.]), mask, 1.)
    loss.backward()
    step = -residual.grad          # gradient descent direction
    assert n == 1 and step[0, 0] > 0 and step[0, 1] < 0 and step[0, 2] == 0


def test_auto_horizon_replaces_the_toml_curriculum_with_an_immediate_jump(tmp_path, monkeypatch):
    """Finding 5: POLICY.toml [horizon] curriculum_updates=[0,1] / entry_pad_max are dead;
    looplong.configure rebuilds knots [0, P] at the resume position P, so every phase update is 90 s."""
    monkeypatch.setenv("C4_ALLOWED_ROOTS", str(tmp_path))
    from wms.control import looplong
    from wms.control.coopw import GATE_SCHEMA
    wm_sha = "ab" * 32
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"schema": GATE_SCHEMA, "wm_sha256": wm_sha,
                                "horizons": {"90": {"verdict": "PASS"}}}))
    position = 180
    trainer = types.SimpleNamespace(
        config=types.SimpleNamespace(entry_pad_max=32, horizon=10),
        _curriculum_position=lambda advance=True: position, horizon_curriculum=None,
        exploration=None, update_count=180)
    doc = {"flags": {"auto_horizon": True}, "previous_horizon_s": 90, "horizon_s": 90,
           "imagination_horizon_s": 90, "gate_file": str(gate), "wm": {"sha256": wm_sha}}
    raw = {"horizon": {"curriculum_updates": [0, 1]}}
    looplong.configure(trainer, doc, raw, tmp_path)
    h = trainer.horizon_curriculum
    assert h.horizons_s == [10.0, 90.0] and h.curriculum_updates == [0.0, 180.0]
    assert h.entry_pad_max == [32, 288]
    assert [h.horizons_s[h.stage(p)] for p in (180, 181, 250, 329)] == [90.0] * 4


def test_selection_and_gate_scoring_use_base_horizon_not_training_horizon():
    """Finding 4: _imagination_config extends the horizon only for TRAINING imagination."""
    from tests.unit.test_longmem import trainer as make_trainer
    from wms.control.coopw import HorizonConfig
    t = make_trainer(horizon=10, batch=1, slots=4)
    h = HorizonConfig(enabled=True, horizons_s=[10, 90], curriculum_updates=[0, 1])
    h.allowed_stage = 1
    t.horizon_curriculum = h
    t.update_count = 180
    batch = t.start_states.sample(1, "cpu")
    t._training_imagination = True
    assert t._imagination_config(batch).horizon == 90
    t._training_imagination = False          # independent_score / native panel / signal gate
    assert t._imagination_config(batch).horizon == 10


def test_low_share_starts_never_get_the_long_horizon():
    from tests.unit.test_longmem import trainer as make_trainer
    from wms.control.coopw import HorizonConfig
    t = make_trainer(horizon=10, batch=1, slots=12)
    h = HorizonConfig(enabled=True, horizons_s=[10, 90], curriculum_updates=[0, 1])
    h.allowed_stage = 1
    t.horizon_curriculum = h
    t.update_count = 180
    batch = t.start_states.sample(1, "cpu")
    batch.state["is_cav"][:] = False
    batch.state["is_cav"][0, 0] = True          # 1/12 = 8% CAV share
    t._training_imagination = True
    assert t._imagination_config(batch).horizon == 10


def test_full_resume_silently_replaces_run_trainer_config(tmp_path, monkeypatch):
    """Finding 1: every [trainer] key outside the memory/training/anti-exploit whitelist
    (and [reward]) comes from the RESUMED checkpoint, not from the run TOML."""
    monkeypatch.setenv("C4_ALLOWED_ROOTS", str(tmp_path))
    from tests.unit.test_longmem import trainer as make_trainer
    old = make_trainer(horizon=5, checkpoint_every=100, entropy_weight=3e-4, grad_clip=100.,
                       actor_lr=5e-5)
    path = tmp_path / "c1.pt"
    old.save_checkpoint(path)
    new = make_trainer(horizon=7, checkpoint_every=1, entropy_weight=1e-3, grad_clip=5.,
                       actor_lr=1e-3, pessimistic_credit=True, future_samples=8,
                       uncertainty_weight=0.)
    new.load_checkpoint(path)
    c = new.config
    assert (c.horizon, c.checkpoint_every, c.entropy_weight, c.grad_clip, c.actor_lr) == (
        5, 100, 3e-4, 100., 5e-5)
    assert c.pessimistic_credit and c.future_samples == 8          # whitelisted survive
    assert new.actor_optimizer.param_groups[0]["lr"] == 5e-5


def test_exec_cap_blocks_positive_residual_gradient_when_binding():
    from wms.model.cav import EXEC_CAP_KEY, apply_exec_cap
    residual = torch.tensor([[0.3, -0.3]], requires_grad=True)
    native = torch.tensor([[1.0, 1.0]])
    state = {EXEC_CAP_KEY: torch.tensor([[1.0, 1.0]])}
    out = apply_exec_cap(state, native + residual)
    out.sum().backward()
    assert residual.grad.tolist() == [[0.0, 1.0]]


def _cf(**kw):
    base = dict(enabled=True, max_starts=1, n_individual=0, n_group=1, n_flip=3,
                group_max_size=3, group_min_size=2, group_radius_m=100., group_state_fraction=0.,
                curriculum_updates=[0, 1], w_all=[1., 1.], w_team=[10.3, 10.3])
    base.update(kw)
    return CounterfactualConfig(**base)


def test_equivalent_config_without_zero_weight_individual_rows_gives_same_gradient():
    """Recommendation check: ind=coop=8.9 + group 9.9 == ind=coop=0 + group 18.8 without the
    3 individual rollouts (same loss, same actor gradient, 6 instead of 9 rows)."""
    from tests.unit.test_longmem import trainer as make_trainer
    live = make_trainer(horizon=4, batch=1, slots=4)
    live.counterfactual = _cf(cover_group_members=True, w_individual=[8.9, 8.9],
                              w_group=[9.9, 9.9], w_coop=[8.9, 8.9])
    lean = make_trainer(horizon=4, batch=1, slots=4)
    lean.counterfactual = _cf(cover_group_members=False, w_individual=[0., 0.],
                              w_group=[18.8, 18.8], w_coop=[0., 0.])
    out = []
    for t in (live, lean):
        torch.manual_seed(5)
        batch = t.start_states.sample(1, "cpu")
        variants = build_variants(batch.state, t.counterfactual, torch.Generator().manual_seed(1))
        imagined = t.imagine(batch, deterministic=True, variants=variants)
        loss, _ = t.actor_objective(imagined, 100., t.counterfactual.weights(180))
        grads = torch.autograd.grad(loss, [p for p in t.actor_parameters if p.requires_grad])
        out.append((variants.n_rows, float(loss), [g.clone() for g in grads]))
    assert out[0][0] == 9 and out[1][0] == 6
    assert out[0][1] == pytest.approx(out[1][1], rel=1e-5, abs=1e-7)
    for a, b in zip(out[0][2], out[1][2]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-7)
