"""Part B skeleton tests -- synthetic data only (no real branch data)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import partB
from partB import (DelayEnsemble, DelayEstimator, DelayEstimatorConfig,
                   collate, count_parameters, featurize, gaussian_nll,
                   near_zero_weight, pairwise_ranking_hinge, partb_loss,
                   resolve_device, seed_everything)
from partB.featurize import (CAV_RAW, CTX_RAW, DEMAND_RAW, LMAX_BINS,
                             WALL_RAW)


# ------------------------------------------------------------ fixtures ----

def _traj(t0, dt, times, lane, s, v):
    """[T,4] contract row set: real at given times, NaN elsewhere."""
    T = int(round(600 / dt)) + 1
    arr = np.full((T, 4), np.nan)
    for t, ln, si, vi in zip(times, lane, s, v):
        k = int(round((t - t0) / dt))
        arr[k] = (t, si, ln, vi)
    return arr


@pytest.fixture()
def synth_state():
    """End-of-Part-A state at t0=120, A=60 with a few vehicles on lanes."""
    t0, A, dt = 120.0, 60.0, 1.0
    traj = {}
    # two mains + one accel vehicle alive through the window
    traj["mainline.0"] = _traj(t0, dt, np.arange(t0, t0 + 60.0, dt),
                               np.zeros(60), np.linspace(50, 650, 60),
                               np.full(60, 30.0))
    traj["mainline.1"] = _traj(t0, dt, np.arange(t0, t0 + 60.0, dt),
                               np.ones(60), np.linspace(30, 590, 60),
                               np.full(60, 25.0))
    traj["ramp.0"] = _traj(t0, dt, np.arange(t0, t0 + 60.0, dt),
                           np.full(60, 2), np.linspace(10, 250, 60),
                           np.full(60, 18.0))
    # a vehicle that exits mid-window (NaN suffix before t_end)
    traj["mainline.2"] = _traj(t0, dt, np.arange(t0, t0 + 20.0, dt),
                               np.zeros(20), np.linspace(100, 690, 20),
                               np.full(20, 31.0))
    controller_state = {"mainline.0": dict(action="damping",
                                           wall_id=None, wall_phase=None,
                                           time_in_phase_s=0.0,
                                           wall_speed_mps=None),
                        "ramp.0": dict(action="native", wall_id=None,
                                       wall_phase=None, time_in_phase_s=0.0,
                                       wall_speed_mps=None)}
    return dict(t0=t0, part_a_horizon_s=A, trajectories=traj,
                controller_state=controller_state)


@pytest.fixture()
def synth_plan():
    return dict(per_cav={"mainline.0": "damping", "ramp.0": "native"},
                walls=[dict(wall_id="w1", members=("mainline.0", "mainline.1"),
                            speed_mps=8.0, phase="assemble",
                            release_at_s=None, hold_s=90.0)])


@pytest.fixture()
def synth_arrivals():
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(120.0, 720.0, 40))
    return dict(t_s=t,
                entrance=np.where(rng.random(40) < 0.3, "ramp", "mainline"),
                is_cav=(rng.random(40) < 0.2).astype(np.int8))


def _features(state, plan, arrivals):
    return featurize(state, plan, arrivals, ctx=dict(
        time_since_last_native_branch_s=30.0, demand_regime=1.0,
        part_a_delay_s=350.0, part_a_ttc_lt3_exposure_s=2.5,
        queue_mainline=4.0, queue_ramp=1.0))


# ------------------------------------------------------------ featurizer ----

def test_feature_shapes_and_dtypes(synth_state, synth_plan, synth_arrivals):
    f = _features(synth_state, synth_plan, synth_arrivals)
    assert f.lane_bins.shape == (3, LMAX_BINS, 3)
    assert f.lane_mask.shape == (3, LMAX_BINS)
    assert f.cav_tokens.shape == (64, CAV_RAW) and CAV_RAW == 20
    assert f.wall_tokens.shape == (4, WALL_RAW) and WALL_RAW == 11
    assert f.demand.shape == (DEMAND_RAW,) and DEMAND_RAW == 21
    assert f.context.shape == (CTX_RAW,) and CTX_RAW == 8
    for name in ("lane_bins", "lane_mask", "cav_tokens", "cav_mask",
                 "wall_tokens", "wall_mask", "demand", "context"):
        assert getattr(f, name).dtype == np.float32, name
    # cavs = per_cav {mainline.0, ramp.0} + wall member mainline.1
    assert f.meta["n_cav"] == 3 and f.meta["n_walls"] == 1


def test_lane_bins_content(synth_state, synth_plan, synth_arrivals):
    f = _features(synth_state, synth_plan, synth_arrivals)
    # mainline.0's snapshot at t_end: s=650 -> bin 65, lane 0
    k = 65
    assert f.lane_mask[0, k] == 1.0
    assert f.lane_bins[0, k, 0] == pytest.approx(1.0)
    assert f.lane_bins[0, k, 1] == pytest.approx(30.0)
    assert f.lane_bins[0, k, 2] == pytest.approx(0.0)
    # mainline.2 exited at t0+19 < t_end -> its old bins stay empty
    assert f.lane_mask[0, 10] == 0.0 and f.lane_bins[0, 10, 0] == 0.0


def test_featurize_deterministic(synth_state, synth_plan, synth_arrivals):
    a = _features(synth_state, synth_plan, synth_arrivals)
    b = _features(synth_state, synth_plan, synth_arrivals)
    for name in ("lane_bins", "lane_mask", "cav_tokens", "cav_mask",
                 "wall_tokens", "wall_mask", "demand", "context"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name


def test_collate_stacks_batch(synth_state, synth_plan, synth_arrivals):
    feats = [_features(synth_state, synth_plan, synth_arrivals),
             _features(synth_state, synth_plan, synth_arrivals)]
    batch = collate(feats)
    assert batch["lane_bins"].shape == (2, 3, LMAX_BINS, 3)
    assert batch["cav_tokens"].shape == (2, 64, CAV_RAW)
    assert batch["demand"].shape == (2, DEMAND_RAW)


# ----------------------------------------------------------------- model ----

def test_model_forward_shapes_and_param_count(synth_state, synth_plan,
                                              synth_arrivals):
    cfg = DelayEstimatorConfig(d_model=64, nhead=4, n_layers=2, dim_ff=128)
    model = DelayEstimator(cfg).to(resolve_device())
    batch = collate([_features(synth_state, synth_plan, synth_arrivals)])
    out = model(batch)
    assert out.shape == (1, 4)
    assert torch.isfinite(out).all()
    big = DelayEstimator()
    n = count_parameters(big)
    assert 1_000_000 <= n <= 3_000_000, f"default width = {n} params"


def test_model_batch_of_four(synth_state, synth_plan, synth_arrivals):
    model = DelayEstimator(DelayEstimatorConfig(d_model=64, n_layers=1,
                                                dim_ff=128))
    batch = collate([_features(synth_state, synth_plan, synth_arrivals)
                     for _ in range(4)])
    assert model(batch).shape == (4, 4)


def test_model_deterministic_given_seed(synth_state, synth_plan,
                                        synth_arrivals):
    batch = collate([_features(synth_state, synth_plan, synth_arrivals)])
    outs = []
    for _ in range(2):
        seed_everything(1234)
        model = DelayEstimator(DelayEstimatorConfig(d_model=64, n_layers=1,
                                                    dim_ff=128)).eval()
        with torch.no_grad():
            outs.append(model(batch))
    assert torch.equal(outs[0], outs[1])
    seed_everything(4321)
    other = DelayEstimator(DelayEstimatorConfig(d_model=64, n_layers=1,
                                                dim_ff=128)).eval()
    with torch.no_grad():
        assert not torch.equal(other(batch), outs[0])


# ---------------------------------------------------------------- losses ----

def test_gaussian_nll_decreases_toward_target():
    torch.manual_seed(0)
    y = torch.tensor([2.0])
    mean = torch.zeros(1, requires_grad=True)
    logvar = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([mean, logvar], lr=0.05)
    first = float(gaussian_nll(mean, logvar, y).detach())
    for _ in range(300):
        opt.zero_grad()
        loss = gaussian_nll(mean, logvar, y)
        loss.backward()
        opt.step()
    assert float(loss.detach()) < 0.05 * first
    assert float(mean.detach()) == pytest.approx(2.0, abs=0.1)


def test_ranking_hinge_sign():
    # target says i before j (y_i < y_j); predictions violating the order
    # pay margin - gap; satisfying predictions pay nothing
    dr_mean = torch.tensor([1.0, 0.0])
    dr_target = torch.tensor([0.0, 1.0])
    groups = torch.zeros(2, dtype=torch.long)
    v = float(pairwise_ranking_hinge(dr_mean, dr_target, groups, margin=2.0))
    assert v == pytest.approx(2.0 + 1.0)       # margin + violation gap
    ok = float(pairwise_ranking_hinge(torch.tensor([0.0, 3.0]), dr_target,
                                      groups, margin=2.0))
    assert ok == 0.0                            # order + full margin: free
    # right order but a 0.5 gap < margin still pays margin - gap
    part = float(pairwise_ranking_hinge(torch.tensor([0.5, 1.0]), dr_target,
                                        groups, margin=2.0))
    assert part == pytest.approx(2.0 - 0.5)


def test_ranking_hinge_respects_groups():
    dr_mean = torch.tensor([1.0, 0.0, 0.0])
    dr_target = torch.tensor([0.0, 1.0, 1.0])
    groups = torch.tensor([0, 0, 1])
    # pair (0,1) violates by 1 (margin 3 + gap 1 = 4); (0,2)/(1,2)
    # cross-group excluded
    v = float(pairwise_ranking_hinge(dr_mean, dr_target, groups, margin=3.0))
    assert v == pytest.approx(4.0)
    tied = float(pairwise_ranking_hinge(dr_mean, torch.zeros(3),
                                        groups, margin=3.0))
    assert tied == 0.0                          # ties skipped


def test_partb_loss_combines_terms(synth_state, synth_plan, synth_arrivals):
    model = DelayEstimator(DelayEstimatorConfig(d_model=32, n_layers=1,
                                                dim_ff=64))
    batch = collate([_features(synth_state, synth_plan, synth_arrivals),
                     _features(synth_state, synth_plan, synth_arrivals)])
    pred = model(batch)
    target = torch.tensor([[500.0, 0.0], [520.0, -40.0]])
    groups = torch.zeros(2, dtype=torch.long)
    out = partb_loss(pred, target, groups)
    assert set(out) >= {"total", "nll_r", "nll_dr", "rank", "total_item"}
    assert float(out["total"].detach()) > 0.0
    assert not out["total"].requires_grad is False  # differentiable
    out["total"].backward()


def test_near_zero_weight_peaks_at_zero():
    w = near_zero_weight(torch.tensor([-3.0, 0.0, 3.0]), width=2.0,
                         boost=1.0)
    assert float(w[1]) == pytest.approx(2.0)
    assert float(w[0]) < 1.2 and float(w[2]) < 1.2   # exp(-(3/2)^2) ~ 0.105


# --------------------------------------------------------------- ensemble ----

def test_ensemble_output_and_device(synth_state, synth_plan, synth_arrivals):
    ens = DelayEnsemble(DelayEstimatorConfig(d_model=32, n_layers=1,
                                             dim_ff=64), n_members=3)
    batch = collate([_features(synth_state, synth_plan, synth_arrivals)])
    mean, sigma = ens.predict(batch)
    assert mean.shape == (3, 1, 2) and sigma.shape == (3, 1, 2)
    assert (sigma > 0).all()
    assert str(resolve_device()) in ("cpu", "cuda", "cuda:0")


def test_ensemble_members_differ_but_are_reproducible():
    cfg = DelayEstimatorConfig(d_model=32, n_layers=1, dim_ff=64)
    ens = DelayEnsemble(cfg, n_members=3, base_seed=99)
    p = [torch.cat([x.flatten() for x in m.parameters()])
         for m in ens.members]
    assert not torch.equal(p[0], p[1])
    ens2 = DelayEnsemble(cfg, n_members=3, base_seed=99)
    p2 = torch.cat([x.flatten() for x in ens2.members[0].parameters()])
    assert torch.equal(p[0], p2)


def test_ensemble_save_load_weights_only(tmp_path, synth_state, synth_plan,
                                         synth_arrivals):
    cfg = DelayEstimatorConfig(d_model=32, n_layers=1, dim_ff=64)
    ens = DelayEnsemble(cfg, n_members=2, base_seed=5)
    batch = collate([_features(synth_state, synth_plan, synth_arrivals)])
    with torch.no_grad():
        ref = ens(batch)
    meta = ens.save_pretrained(tmp_path / "ens")
    assert meta["n_members"] == 2
    loaded = DelayEnsemble.from_pretrained(tmp_path / "ens")
    with torch.no_grad():
        got = loaded(batch)
    assert torch.allclose(ref, got, atol=1e-6)
    # weights_only=True load path exercised inside from_pretrained
    sd = torch.load(tmp_path / "ens" / "member_0.pt", weights_only=True)
    assert isinstance(sd, dict) and len(sd) > 0


def test_cpu_run_end_to_end(synth_state, synth_plan, synth_arrivals):
    """Full synthetic forward+loss+backward on the resolved device
    (CPU here; no .cuda() calls anywhere in partB)."""
    device = resolve_device()
    if device.type != "cpu":
        pytest.skip("non-cpu WD_DEVICE set")
    torch.manual_seed(0)
    model = DelayEstimator(DelayEstimatorConfig(d_model=32, n_layers=1,
                                                dim_ff=64))
    feats = [_features(synth_state, synth_plan, synth_arrivals)
             for _ in range(3)]
    batch = collate(feats)
    target = torch.tensor([[500.0, 10.0], [480.0, -5.0], [530.0, 25.0]])
    groups = torch.zeros(3, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        out = partb_loss(model(batch), target, groups)
        out["total"].backward()
        opt.step()
        losses.append(float(out["total"]))
    assert losses[-1] < losses[0]              # optimisable on toy data
