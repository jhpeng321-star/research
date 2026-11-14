"""Acceptance unit test (B3b): the v0.3 tau conditioning distinguishes
horizons.

A synthetic dataset whose delay label depends ONLY on tau (y = 100 +
8*tau + noise) is trained for a few hundred CPU steps; the median point
forecast (A5) after per-tau de-normalisation (A1) must fit the tau-only
signal to < 5% relative MAE, and the per-tau predictions must be
monotone in tau.  v0.2 failed exactly this: its unnormalised fixed grid
squeezed all horizons toward the middle.

Smoke-sized CPU run (owner rule: no real training on CPU).
"""
from __future__ import annotations

import numpy as np
import torch

from conftest_load import make_tiny_dataset
from partB_v0.data import V0Dataset, V03TensorData, load_manifest
from partB_v0.evaluate import load_net_v03, net_mixture_probs
from partB_v0.model import PmfGrid, V03ValueConfig
from partB_v0.train_net import ConcurrentNetTrainer, TrainNetConfig

STEPS = 300


def test_tau_only_signal_fit_below_5pct(tmp_path):
    root = tmp_path / "data"
    make_tiny_dataset(root, n_train=4, n_cal=2, n_dec=14, seed=11,
                      y_fn=lambda tau, t, source: 100.0 + 8.0 * tau)
    device = torch.device("cpu")
    man = load_manifest(root, verify_files=False, log=lambda *_: None)
    ds = V0Dataset(man, log=lambda *_: None)
    data = V03TensorData(ds, device, log=lambda *_: None)
    data.set_members(20260929, 1)

    cfg = TrainNetConfig(
        data_root=str(root), out_dir=str(tmp_path / "out"), members=1,
        batch_size=128, td_batch=64, pair_batch=32, seed=5,
        max_steps=STEPS, eval_every_steps=100, patience=3,
        time_budget_min=0, resume=False, verify=False, device="cpu")
    cfg_model = V03ValueConfig(d_model=64, n_layers=1, dim_ff=128,
                               head_hidden=64, head_layers=1,
                               tau_emb_dim=16, delta_hidden=32,
                               delta_layers=1, cav_raw=17, wall_raw=6)
    tr = ConcurrentNetTrainer(cfg, data, device, "fp32",
                              log=lambda *_: None, cfg_model=cfg_model)
    tr.train(STEPS, None)
    data.save_stats(tmp_path / "out" / "stats.json")   # A4 ships with weights
    tr.write_members(tmp_path / "out")

    bundle = load_net_v03(tmp_path / "out", device,
                          log=lambda *_: None)
    cal_idx = np.flatnonzero(data.split_cal)
    grid = PmfGrid(cfg_model.sym_min, cfg_model.sym_max, cfg_model.n_bins)
    probs = net_mixture_probs(bundle, data, cal_idx)
    med_norm = grid.median(probs)
    scale = data.med_row["delay"][cal_idx]
    pred = med_norm * scale
    y = ds.rows["total_delay_ref_s"].to_numpy()[cal_idx]
    tau = data.row_tau[cal_idx]

    rel_mae = float(np.abs(pred - y).mean() / y.mean())
    print(f"tau-only signal: relative MAE {rel_mae:.4f} after "
          f"{STEPS} CPU steps (smoke-only numbers)")
    assert rel_mae < 0.05

    # the conditioning must distinguish horizons: predictions increase
    # with tau (v0.2 compressed them toward the middle)
    means = {float(t): float(pred[np.isclose(tau, t)].mean())
             for t in np.unique(tau)}
    ordered = [means[t] for t in sorted(means)]
    assert all(b > a for a, b in zip(ordered, ordered[1:])), means
    # and the normalised forecast sits near the normalised target (~1)
    y_norm = data.y_norm["delay"][torch.as_tensor(
        cal_idx, device=device)].to(torch.float64).cpu().numpy()
    assert float(np.abs(med_norm - y_norm).mean()) < 0.2
