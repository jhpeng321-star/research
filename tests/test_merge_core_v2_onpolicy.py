"""Round-8 version-2 on-policy supported-prefix survival (agent A).

Synthetic, CPU-only checks of the repaired bookkeeping: censoring on the absolute clock,
persistent support across chunks, event/early/admin terminals, exposure accounting, the
persistent latent posterior (monolithic == split chunks) and gradient directions.
"""
from __future__ import annotations

import pytest
import torch

from wms.model.merge_core import load_init
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.merge_core import (MergeCoreConfig, MergeCoreTerm, merge_core_metrics)


def _make(n=6, k=3, event=2, pred_event="same", rec_valid=None, pred_present=None,
          acts=None, positions=None, rec_positions=None, gap_l=20.0, gap_f=20.0,
          has_l=True, has_f=True, alpha_loc=0.0, alpha_sigma=0.2, pred_lanes=None,
          params=None):
    """One batch of the ramp target in slot 0 plus lane-1 leader/follower neighbours."""
    init = load_init()
    params = {} if params is None else dict(params)
    s0 = float(params.get("base_s", 20.0))
    v = float(params.get("v", 1.0))
    t = torch.arange(n + 1, dtype=torch.float32)

    if rec_positions is None:
        rec_positions = torch.zeros(n + 1, k)
        rec_positions[:, 0] = s0 + v * t
        rec_positions[:, 1] = s0 + 40.0 + v * t
        rec_positions[:, 2] = s0 - 20.0 + v * t
    rec_positions = torch.as_tensor(rec_positions, dtype=torch.float32)

    rec_lane = torch.ones(n + 1, k, dtype=torch.long)
    rec_lane[:, 0] = 2
    if event is not None:
        rec_lane[event + 1:, 0] = 1
    if rec_valid is None:
        rec_valid = torch.ones(n + 1, k, dtype=torch.bool)
    rec_valid = torch.as_tensor(rec_valid, dtype=torch.bool)

    states = torch.zeros(1, n + 1, k, 7)
    states[0, :, :, 0] = rec_positions
    states[0, :, :, 6] = 5.0
    for lane in range(3):
        states[0, :, :, 1 + lane] = (rec_lane == lane).float()
    states[0, :, :, 4] = v + (torch.arange(k).view(1, k) * 0.0)

    if pred_event == "same":
        pred_event = event
    pred_lane = rec_lane[1:].clone()
    if pred_event is not None:
        pred_lane[:, 0] = 2
        pred_lane[pred_event:, 0] = 1
    if pred_lanes is not None:
        pred_lane = torch.as_tensor(pred_lanes, dtype=torch.long).clone()

    if positions is None:
        positions = rec_positions[1:].clone()
    positions = torch.as_tensor(positions, dtype=torch.float32).clone()
    if pred_present is None:
        pred_present = rec_valid[1:].clone()
    pred_present = torch.as_tensor(pred_present, dtype=torch.bool).clone()
    if acts is None:
        acts = torch.ones(n, k, dtype=torch.bool)
    acts = torch.as_tensor(acts, dtype=torch.bool).clone()

    m = n
    zeros = torch.zeros(1, m, k)
    alpha_loc = torch.full((1, m, k), float(alpha_loc), requires_grad=True)
    pred = {
        "lanes": pred_lane.unsqueeze(0),
        "masks": pred_present.unsqueeze(0),
        "positions": positions.unsqueeze(0),
        "speeds": torch.full((1, m, k), v),
        "accel": zeros.clone(),
        "lane_v5_acts": acts.unsqueeze(0),
        "merge_core_gap_l": torch.full((1, m, k), float(gap_l)),
        "merge_core_gap_f": torch.full((1, m, k), float(gap_f)),
        "merge_core_v_l": torch.full((1, m, k), v),
        "merge_core_v_f": torch.full((1, m, k), v),
        "merge_core_a_l": zeros.clone(),
        "merge_core_a_f": zeros.clone(),
        "merge_core_has_l": torch.full((1, m, k), bool(has_l), dtype=torch.bool),
        "merge_core_has_f": torch.full((1, m, k), bool(has_f), dtype=torch.bool),
        "merge_core_alpha_log_location": alpha_loc,
        "merge_core_alpha_scale": torch.full((1, m, k), float(alpha_sigma)),
        "merge_core_residual": zeros.clone(),
        "merge_core_s0": torch.full((m,), float(params.get("s0", init["s0_m"]))),
        "merge_core_T": torch.full((m,), float(params.get("T", init["T_s"]))),
        "merge_core_b": torch.full((m,), float(params.get("b", init["b_mps2"]))),
        "merge_core_tau_a": torch.full((m,), float(params.get("tau_a", init["tau_a_s"]))),
        "merge_core_temperature": torch.full(
            (m,), float(params.get("temperature", init["temperature_m"]))),
    }
    current = {
        "lane": rec_lane[:1].clone(),
        "present": rec_valid[:1].clone(),
        "s_m": rec_positions[:1].clone(),
        "v_mps": torch.full((1, k), v),
        "a_mps2": torch.zeros(1, k),
    }
    targets = UnrollTargets(
        states=states, n_steps=n, weights=torch.ones(1, n, 1),
        hdv_mask=torch.ones(1, n, k), aux_mask=torch.ones(1, n, k), aux_count=n * k,
        denom=torch.tensor(1.0), lane_target=torch.zeros(1, n + 1, k),
        loss_valid=rec_valid.unsqueeze(0).float(), is_cav=torch.zeros(1, n + 1, k),
        dt_s=0.5, position_delta_m=5.0, speed_delta_mps=1.0, history_lane_head=False)
    return targets, pred, current


def _term(**kwargs):
    defaults = dict(version=2, weight=1.0, tf_weight=0.0, n_quad=8,
                    location_reg_weight=0.0, sigma_reg_weight=0.0)
    defaults.update(kwargs)
    return MergeCoreTerm(MergeCoreConfig(**defaults))


def _current_at(pred, current, c0):
    if c0 == 0:
        return current
    return {
        "lane": pred["lanes"][:, c0 - 1].clone(),
        "present": pred["masks"][:, c0 - 1].clone(),
        "s_m": pred["positions"][:, c0 - 1].clone(),
        "v_mps": pred["speeds"][:, c0 - 1].clone(),
        "a_mps2": pred["accel"][:, c0 - 1].clone(),
    }


def _run(targets, pred, current, chunks, **cfg):
    term = _term(**cfg)
    term.prepare(targets)
    values = []
    for c0, c1 in chunks:
        chunk_current = _current_at(pred, current, c0)
        value = term(LossContext(pred=pred, current=chunk_current, targets=targets,
                                 c0=c0, c1=c1))
        assert value is not None
        assert torch.isfinite(value.value)
        values.append(value.value)
    metrics = merge_core_metrics(targets.prepared["merge_core"])
    return values, metrics, targets


def _stack(cases):
    """Concatenate independent single-car cases along the batch dimension."""
    targets0, pred0, current0 = cases[0]
    states = torch.cat([c[0].states for c in cases], 0)
    loss_valid = torch.cat([c[0].loss_valid for c in cases], 0)
    is_cav = torch.cat([c[0].is_cav for c in cases], 0)
    n = targets0.n_steps
    k = targets0.states.shape[2]
    targets = UnrollTargets(
        states=states, n_steps=n, weights=torch.ones(len(cases), n, 1),
        hdv_mask=torch.ones(len(cases), n, k), aux_mask=torch.ones(len(cases), n, k),
        aux_count=len(cases) * n * k, denom=torch.tensor(1.0),
        lane_target=torch.zeros(len(cases), n + 1, k), loss_valid=loss_valid,
        is_cav=is_cav, dt_s=targets0.dt_s,
        position_delta_m=targets0.position_delta_m,
        speed_delta_mps=targets0.speed_delta_mps, history_lane_head=False)
    pred = {}
    for key in pred0:
        if torch.is_tensor(pred0[key]):
            pred[key] = torch.cat([c[1][key] for c in cases], 0)
        else:
            pred[key] = pred0[key]
    current = {key: torch.cat([c[2][key] for c in cases], 0) for key in current0}
    return targets, pred, current


def test_censor_on_absolute_clock_c0_zero_and_positive():
    for n, chunks in ((5, [(0, 5)]), (10, [(0, 5), (5, 10)])):
        targets, pred, current = _make(n=n, event=None)
        _, metrics, _ = _run(targets, pred, current, chunks)
        assert metrics["merge_core_terminal_censored_admin"] == 1.0
        assert metrics["merge_core_terminal_total"] == 1.0
        assert metrics["merge_core_scored_intervals"] == float(n)
        assert metrics["merge_core_scored_seconds"] == pytest.approx(n * 0.5)


def test_recorded_event_on_chunk_boundary_and_no_post_event_survival():
    for event, chunks in ((3, [(0, 4), (4, 8)]), (4, [(0, 4), (4, 8)])):
        targets, pred, current = _make(n=8, event=event)
        _, metrics, _ = _run(targets, pred, current, chunks)
        assert metrics["merge_core_terminal_event"] == 1.0
        assert metrics["merge_core_terminal_total"] == 1.0
        assert metrics["merge_core_scored_intervals"] == float(event + 1)


def test_no_post_event_survival_changes_late_hazard():
    n = 7
    targets, pred, current = _make(n=n, event=2)
    _, base, _ = _run(targets, pred, current, [(0, n)])
    targets2, pred2, current2 = _make(n=n, event=2)
    pred2["merge_core_gap_l"] = pred2["merge_core_gap_l"].clone()
    pred2["merge_core_gap_l"][:, 3:, :] = 500.0
    _, after, _ = _run(targets2, pred2, current2, [(0, n)])
    assert after["merge_core_on_nll"] == pytest.approx(base["merge_core_on_nll"], abs=0.0)
    assert after["merge_core_scored_intervals"] == 3.0


def test_first_incompatible_step_is_terminal_and_never_recovers():
    n = 8
    targets, pred, current = _make(n=n, event=None)
    pred["positions"] = pred["positions"].clone()
    pred["positions"][:, 1, 0] += 100.0          # decision step 2 is incompatible
    _, metrics, _ = _run(targets, pred, current, [(0, n)])
    assert metrics["merge_core_terminal_support_failure_pos"] == 1.0
    assert metrics["merge_core_scored_intervals"] == 2.0
    assert metrics["merge_core_terminal_total"] == 1.0


def test_no_recovery_in_later_chunk_after_support_failure():
    n = 8
    targets, pred, current = _make(n=n, event=None)
    pred["positions"] = pred["positions"].clone()
    pred["positions"][:, 1, 0] += 100.0          # fails at step 2, inside the first chunk
    _, metrics, _ = _run(targets, pred, current, [(0, 3), (3, 8)])
    assert metrics["merge_core_terminal_support_failure_pos"] == 1.0
    assert metrics["merge_core_scored_intervals"] == 2.0
    assert metrics["merge_core_terminal_total"] == 1.0


def test_early_merge_on_compatible_step_is_nonevent_and_terminal():
    n = 8
    targets, pred, current = _make(n=n, event=5, pred_event=2)
    _, metrics, _ = _run(targets, pred, current, [(0, n)])
    assert metrics["merge_core_terminal_early_nonevent"] == 1.0
    assert metrics["merge_core_terminal_event"] == 0.0
    assert metrics["merge_core_scored_intervals"] == 3.0          # 0,1 non-event + 2 early
    assert metrics["merge_core_terminal_total"] == 1.0


def test_early_merge_on_incompatible_step_is_support_failure_not_early():
    n = 8
    targets, pred, current = _make(n=n, event=5, pred_event=2)
    pred["positions"] = pred["positions"].clone()
    pred["positions"][:, 1, 0] += 100.0          # step 2 incompatible
    _, metrics, _ = _run(targets, pred, current, [(0, n)])
    assert metrics["merge_core_terminal_early_nonevent"] == 0.0
    assert metrics["merge_core_terminal_support_failure_pos"] == 1.0
    assert metrics["merge_core_scored_intervals"] == 2.0


def test_late_model_merge_does_not_change_the_supported_recorded_event():
    n = 8
    targets, pred, current = _make(n=n, event=2, pred_event=5)
    _, metrics, _ = _run(targets, pred, current, [(0, n)])
    assert metrics["merge_core_terminal_event"] == 1.0
    assert metrics["merge_core_terminal_early_nonevent"] == 0.0
    assert metrics["merge_core_scored_intervals"] == 3.0


def test_administrative_censoring():
    n = 6
    targets, pred, current = _make(n=n, event=None)
    _, metrics, _ = _run(targets, pred, current, [(0, n)])
    assert metrics["merge_core_terminal_censored_admin"] == 1.0
    assert metrics["merge_core_terminal_zero_exposure"] == 0.0
    assert metrics["merge_core_scored_intervals"] == float(n)


def test_no_exposure_car_is_zero_exposure_and_contributes_no_survival():
    n = 6
    targets, pred, current = _make(n=n, event=None)
    acts = torch.zeros(n, 3, dtype=torch.bool)   # never envelope-feasible
    pred["lane_v5_acts"] = acts.unsqueeze(0)
    _, metrics, _ = _run(targets, pred, current, [(0, n)])
    assert metrics["merge_core_terminal_zero_exposure"] == 1.0
    assert metrics["merge_core_scored_intervals"] == 0.0
    assert metrics["merge_core_opportunity_intervals"] == 0.0
    assert metrics["merge_core_terminal_total"] == 1.0


def test_slot_reuse_resets_persistent_support_and_birth():
    n = 9
    # identity 1 present 0..2, absent 3..4, identity 2 present 5..8
    rec_valid = torch.ones(n + 1, 3, dtype=torch.bool)
    rec_valid[3:5, 0] = False
    pred_present = torch.ones(n, 3, dtype=torch.bool)
    pred_present[2:5, 0] = False
    pred_present[5:, 0] = True
    targets, pred, current = _make(n=n, event=None, rec_valid=rec_valid,
                                   pred_present=pred_present)
    pred["positions"] = pred["positions"].clone()
    pred["positions"][:, 1, 0] += 100.0          # first identity fails at step 2
    _, metrics, _ = _run(targets, pred, current, [(0, n)])
    v2 = targets.prepared["merge_core"]["v2"]
    assert metrics["merge_core_terminal_support_failure_pos"] == 1.0
    # the reused slot restarts as a new identity with a fresh birth and support
    assert int(v2["birth_step"][0, 0]) == 6
    assert bool(v2["support"][0, 0])
    assert bool(v2["started"][0, 0])
    assert metrics["merge_core_terminal_censored_admin"] == 1.0


def test_non_target_departure_is_not_an_early_merge():
    n = 8
    targets, pred, current = _make(n=n, event=5, pred_event=None)
    pred["lanes"] = pred["lanes"].clone()
    pred["lanes"][:, 2, 0] = 0                   # ramp -> lane 0, not the target lane
    pred["lanes"][:, 3:, 0] = 0
    _, metrics, _ = _run(targets, pred, current, [(0, n)])
    assert metrics["merge_core_terminal_early_nonevent"] == 0.0
    assert metrics["merge_core_terminal_support_failure_topology"] == 1.0
    assert metrics["merge_core_scored_intervals"] == 2.0


def test_terminal_categories_partition_recorded_identities():
    cases = []
    # event
    cases.append(_make(n=6, event=2))
    # admin censor
    cases.append(_make(n=6, event=None))
    # position support failure
    t, p, c = _make(n=6, event=None)
    p["positions"] = p["positions"].clone()
    p["positions"][:, 1, 0] += 100.0
    cases.append((t, p, c))
    # early non-event
    cases.append(_make(n=6, event=5, pred_event=2))
    # zero exposure
    t, p, c = _make(n=6, event=None)
    p["lane_v5_acts"] = torch.zeros(1, 6, 3, dtype=torch.bool)
    cases.append((t, p, c))
    targets, pred, current = _stack(cases)
    _, metrics, _ = _run(targets, pred, current, [(0, 6)])
    total = metrics["merge_core_terminal_total"]
    partition = sum(value for key, value in metrics.items()
                    if key.startswith("merge_core_terminal_")
                    and key != "merge_core_terminal_total")
    assert total == 5.0
    assert partition == total
    assert metrics["merge_core_terminal_event"] == 1.0
    assert metrics["merge_core_terminal_censored_admin"] == 1.0
    assert metrics["merge_core_terminal_support_failure_pos"] == 1.0
    assert metrics["merge_core_terminal_early_nonevent"] == 1.0
    assert metrics["merge_core_terminal_zero_exposure"] == 1.0


def test_monolithic_and_split_chunk_likelihood_agree_with_nonzero_sigma():
    n = 9
    targets, pred, current = _make(n=n, event=None, alpha_sigma=0.5)
    mono, _, _ = _run(targets, pred, current, [(0, n)])
    targets2, pred2, current2 = _make(n=n, event=None, alpha_sigma=0.5)
    split, _, _ = _run(targets2, pred2, current2, [(0, 4), (4, n)])
    assert float(sum(v.detach() for v in split)) == pytest.approx(float(mono[0].detach()), rel=1e-5, abs=1e-6)


def test_monolithic_and_split_chunk_likelihood_agree_with_event():
    n = 8
    targets, pred, current = _make(n=n, event=3, alpha_sigma=0.4)
    mono, _, _ = _run(targets, pred, current, [(0, n)])
    targets2, pred2, current2 = _make(n=n, event=3, alpha_sigma=0.4)
    split, _, _ = _run(targets2, pred2, current2, [(0, 5), (5, n)])
    assert float(sum(v.detach() for v in split)) == pytest.approx(float(mono[0].detach()), rel=1e-5, abs=1e-6)


def test_gradient_direction_pre_event_hazard():
    n = 6
    targets, pred, current = _make(n=n, event=None, gap_l=20.0, gap_f=20.0)
    base, _, _ = _run(targets, pred, current, [(0, n)])
    targets2, pred2, current2 = _make(n=n, event=None, gap_l=60.0, gap_f=60.0)
    higher, _, _ = _run(targets2, pred2, current2, [(0, n)])
    assert float(higher[0].detach()) > float(base[0].detach())


def test_gradient_direction_event_hazard_and_zero_post_terminal_gradient():
    n = 6
    targets, pred, current = _make(n=n, event=2)
    base, _, _ = _run(targets, pred, current, [(0, n)])
    targets2, pred2, current2 = _make(n=n, event=2)
    gaps = pred2["merge_core_gap_l"].clone()
    gaps = torch.where(torch.arange(n).view(1, n, 1) == 2,
                       torch.full_like(gaps, 80.0), gaps).requires_grad_(True)
    gap = gaps
    pred2["merge_core_gap_l"] = gap
    pred2["merge_core_gap_f"] = torch.where(
        torch.arange(n).view(1, n, 1) == 2, torch.full_like(gap, 80.0),
        pred2["merge_core_gap_f"])
    higher_event, _, _ = _run(targets2, pred2, current2, [(0, n)])
    assert float(higher_event[0].detach()) < float(base[0].detach())
    higher_event[0].backward()
    assert gap.grad is not None
    assert torch.allclose(gap.grad[:, 3:, :], torch.zeros_like(gap.grad[:, 3:, :]))
    assert bool(gap.grad[:, 2, :].abs().sum() > 0.0)


def test_zero_gradient_on_never_exposed_car():
    n = 6
    targets, pred, current = _make(n=n, event=None)
    gap = pred["merge_core_gap_l"].clone().requires_grad_(True)
    pred["merge_core_gap_l"] = gap
    pred["lane_v5_acts"] = torch.zeros(1, n, 3, dtype=torch.bool)
    values, _, _ = _run(targets, pred, current, [(0, n)])
    values[0].backward()
    if gap.grad is not None:
        assert torch.allclose(gap.grad, torch.zeros_like(gap.grad))


def test_version_one_path_is_unchanged_and_tf_weight_zero_v2_runs():
    n = 5
    targets, pred, current = _make(n=n, event=2)
    v1 = _term(version=1, weight=1.0, tf_weight=0.0)
    v1.prepare(targets)
    value = v1(LossContext(pred=pred, current=current, targets=targets, c0=0, c1=n))
    assert value is not None and torch.isfinite(value.value)
    targets2, pred2, current2 = _make(n=n, event=2)
    v2 = _term(version=2, weight=1.0, tf_weight=0.0)
    v2.prepare(targets2)
    value2 = v2(LossContext(pred=pred2, current=current2, targets=targets2, c0=0, c1=n))
    assert value2 is not None and torch.isfinite(value2.value)


def test_seqtf_contract_with_a_stub(monkeypatch):
    import sys
    import types

    module = types.ModuleType("wms.train.losses.merge_core_seqtf")

    def build_recorded_scene(targets, *, merge_lane, zone_s0_m, zone_s1_m, k):
        return {"targets": targets, "k": k, "calls": []}

    def seq_tf_chunk(scene, c0, c1, *, s0, T, b, tau_a, temperature, sigma, n_quad,
                     log_post, require_both=None):
        scene["calls"].append((c0, c1))
        nll = (s0 * 0.0 + T * 0.0 + b * 0.0 + tau_a * 0.0
               + sigma.sum() * 0.0 + 1.0)
        return nll, torch.tensor(1.0), None, {"nll": 1.0, "rows": 1.0, "events": 0.0}

    module.build_recorded_scene = build_recorded_scene
    module.seq_tf_chunk = seq_tf_chunk
    monkeypatch.setitem(sys.modules, "wms.train.losses.merge_core_seqtf", module)
    n = 6
    targets, pred, current = _make(n=n, event=2)
    term = _term(tf_weight=1.0)
    term.prepare(targets)
    value = term(LossContext(pred=pred, current=current, targets=targets, c0=0, c1=n))
    assert value is not None and torch.isfinite(value.value)
    assert targets.prepared["merge_core"]["v2"]["scene"]["calls"] == [(0, n)]


def test_config_validation_and_normalisers():
    with pytest.raises(ValueError):
        MergeCoreConfig(version=3)
    with pytest.raises(ValueError):
        MergeCoreConfig(version=2, z_on=0.0)
    with pytest.raises(ValueError):
        MergeCoreConfig(version=2, z_tf=float("nan"))


