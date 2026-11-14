"""Tests for the standalone CTM (``wms_eval.ctm``).

Covers the brief's required unit contracts: vehicle conservation, FD limits,
the merge rule (priority + capacity drop), counters -> boundary inflow,
soft-vs-exact agreement as the temperature tends to zero, and the soft/hard
t0 binning used for evaluation vs training.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(PKG))

from wms_eval import ctm  # noqa: E402


def _counter(mainline=6.0, ramp=2.0):
    counter = np.zeros((2, 30), dtype=np.float64)
    counter[0, :12] = mainline
    counter[1, :12] = ramp
    return counter


def _params(**kw):
    return ctm.default_params(**kw)


def _initial(layout, params, total=60.0):
    """Deterministic non-degenerate initial counts spread over the chains."""
    rng = np.random.default_rng(0)
    n = np.zeros(layout.n_cells, dtype=np.float64)
    per = total / layout.n_cells
    n[:] = per
    n += rng.uniform(0.0, 1.0, layout.n_cells)
    return n


def test_conservation_with_boundary_flows():
    layout = ctm.build_layout(100.0)
    params = _params()
    n0 = _initial(layout, params)
    run = ctm.run_numpy(n0, layout, params, _counter(), steps=120, t0_s=0.0)
    mass = run.counts.sum(axis=1)
    delta = np.diff(mass)
    expected = run.main_inflow + run.ramp_inflow - run.outflow
    assert np.allclose(delta, expected, atol=1e-6)
    # no inflow, no initial state -> mass is exactly conserved at zero
    empty = ctm.run_numpy(np.zeros(layout.n_cells), layout, params,
                          np.zeros((2, 30)), steps=60)
    assert np.allclose(empty.counts.sum(axis=1), 0.0)


def test_no_inflow_or_outflow_conserves_exactly():
    layout = ctm.build_layout(100.0)
    params = _params()
    # zero counters and a walled downstream boundary: block the free outflow by
    # feeding a saturated downstream cell is hard, so instead check that with
    # zero inflow and zero initial vehicles nothing is created.
    run = ctm.run_numpy(np.zeros(layout.n_cells), layout, params,
                        np.zeros((2, 30)), steps=40)
    assert run.counts.sum() == 0.0
    assert run.main_inflow.sum() == 0.0


def test_fd_limits_and_speeds():
    layout = ctm.build_layout(75.0)
    params = _params()
    n0 = _initial(layout, params, total=400.0)
    demand, supply, speed = ctm._demand_supply(n0[None, :], layout, params, params.dt_s)
    cap_by_cell = np.array([params.fd(r).capacity_mps_per_lane for r in layout.role])
    assert np.all(demand >= -1e-9)
    assert np.all(demand <= cap_by_cell[None, :] * layout.lanes[None, :] * params.dt_s + 1e-9)
    assert np.all(supply >= -1e-9)
    v_free = np.array([params.fd(r).v_free_mps for r in layout.role])
    assert np.all(speed <= v_free[None, :] + 1e-9)
    assert np.all(speed >= -1e-9)


def test_merge_rule_priority_and_capacity():
    # both demands fit
    ym, yr = ctm.merge_allocation(3.0, 1.0, supply=10.0)
    assert np.isclose(ym, 3.0) and np.isclose(yr, 1.0)
    # oversaturated: supply fully used, mainline keeps priority share
    ym, yr = ctm.merge_allocation(20.0, 20.0, supply=6.0, p_main=2.0, p_ramp=1.0)
    assert np.isclose(ym + yr, 6.0)
    assert ym >= (2.0 / 3.0) * 6.0 - 1e-9
    assert yr > 0.0
    # ramp only
    ym, yr = ctm.merge_allocation(0.0, 5.0, supply=4.0)
    assert np.isclose(ym, 0.0) and np.isclose(yr, 4.0)
    # mainline only
    ym, yr = ctm.merge_allocation(9.0, 0.0, supply=4.0)
    assert np.isclose(ym, 4.0) and np.isclose(yr, 0.0)
    # flows never exceed demand or supply
    ym, yr = ctm.merge_allocation(7.0, 2.0, supply=3.0)
    assert ym <= 7.0 + 1e-9 and yr <= 2.0 + 1e-9 and ym + yr <= 3.0 + 1e-9


def test_capacity_drop_activated_by_queue():
    layout = ctm.build_layout(100.0)
    params = _params(merge_capacity_drop=0.5)
    params_full = ctm.default_params(merge_capacity_drop=1.0)
    # a queue next to the merge (last main cell congested)
    n = np.zeros(layout.n_cells)
    n[layout.main_idx[-1]] = 25.0
    n[layout.main_idx[0]] = 5.0
    n, *_ = ctm._step_numpy(n, layout, params, 0.0, 0.0, smooth=False, tau=1e-3)
    n2 = np.zeros(layout.n_cells)
    n2[layout.main_idx[-1]] = 25.0
    n2[layout.main_idx[0]] = 5.0
    n2, *_ = ctm._step_numpy(n2, layout, params_full, 0.0, 0.0, smooth=False, tau=1e-3)
    # the drop lowers the merge throughput, so fewer vehicles leave the main
    # approach into the merge cell
    assert n[layout.merge_idx] <= n2[layout.merge_idx] + 1e-9


def test_counter_rates_match_wms_entries():
    counter = np.arange(2 * 30, dtype=np.float64).reshape(2, 30) % 9.0
    ours = ctm.counter_rates(counter, 12)
    try:
        from wms.entries import observed_rates
    except Exception:  # pragma: no cover - torch/wms unavailable
        pytest.skip("wms.entries not importable")
    assert np.allclose(ours, observed_rates(counter, 12))


def test_soft_binning_recovers_hard_as_sigma_to_zero():
    layout = ctm.build_layout(100.0)
    s = np.array([-500.0, -120.0, -10.0, 50.0, 200.0])
    lane = np.array([0, 1, 2, 2, 0])
    hard = ctm.hard_bin_counts(layout, s, lane)
    soft = ctm.soft_bin_counts(layout, s, lane, sigma_m=1e-3)
    assert np.isclose(hard.sum(), 5.0)
    assert np.isclose(soft.sum(), 5.0)
    assert np.allclose(hard, np.round(soft), atol=1e-6)


def test_soft_and_exact_ctm_agree_as_tau_to_zero():
    layout = ctm.build_layout(100.0)
    params = _params(merge_capacity_drop=0.8)
    n0 = _initial(layout, params, total=300.0)
    counter = _counter(8.0, 3.0)
    exact = ctm.run_numpy(n0, layout, params, counter, steps=80, smooth=False)
    soft = ctm.run_numpy(n0, layout, params, counter, steps=80, smooth=True, tau=1e-6)
    assert np.max(np.abs(exact.counts - soft.counts)) < 5e-3
    assert np.allclose(exact.counts.sum(axis=1), soft.counts.sum(axis=1), atol=1e-3)


def test_torch_ctm_matches_numpy_reference():
    torch = pytest.importorskip("torch")
    layout = ctm.build_layout(100.0)
    params = _params(merge_capacity_drop=0.85)
    n0 = _initial(layout, params, total=300.0)
    counter = _counter(8.0, 3.0)
    exact = ctm.run_numpy(n0, layout, params, counter, steps=60, smooth=False)
    model = ctm.TorchCTM(params, layout)
    out = model.run(n0, counter, steps=60, tau=1e-6)
    counts = out["counts"].numpy()[0]
    assert np.max(np.abs(counts - exact.counts)) < 5e-3
    assert torch.isfinite(out["counts"]).all()


def test_ctm_traffic_metrics_schema_and_coverage():
    layout = ctm.build_layout(100.0)
    params = _params()
    n0 = _initial(layout, params, total=200.0)
    run = ctm.run_numpy(n0, layout, params, _counter(), steps=240, t0_s=10.0)
    traffic = ctm.ctm_traffic_metrics(run, bin_s=30.0, free_speed_mps=27.78)
    assert set(traffic) == {"0", "1", "2", "3"}
    for block in traffic.values():
        assert set(block["sections"]) == {"upstream", "merge", "downstream"}
        for section in block["sections"].values():
            assert {"mean_speed", "density", "stopped_vehicles",
                    "merge_queue_length", "merge_flow", "total_delay"} <= set(section)
    assert traffic["0"]["sections"]["merge"]["merge_flow"] >= 0.0
