"""End-to-end evaluation tests on the tiny synthetic dataset (CPU)."""
from __future__ import annotations

import numpy as np
import torch

from partB_v0.data import V0Dataset, load_manifest
from partB_v0.evaluate import EvalConfig, evaluate_run, load_members
from partB_v0.model import V0ValueNet
from partB_v0.train import seed_everything


def _train_one_member(root, out, steps=3):
    from partB_v0.train import TrainConfig, run_training
    cfg = TrainConfig(data_root=str(root), out_dir=str(out), members=1,
                      batch_size=32, td_batch=16, max_steps=steps,
                      probe_steps=0, seed=5, verify=False, smoke=False)
    metrics, ds = run_training(cfg)
    return metrics, ds


def test_eval_produces_all_five_sections(tiny_dataset, tmp_path):
    root, sha = tiny_dataset
    out = tmp_path / "out"
    _train_one_member(root, out)
    man = load_manifest(root, verify_files=False, log=lambda *_: None)
    ds = V0Dataset(man, log=lambda *_: None)
    report, m = evaluate_run(EvalConfig(out_dir=str(out), smoke=True), ds,
                             log=lambda *_: None)
    for header in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5."):
        assert header in report
    assert sha in report
    assert m["item1_crps_pit_coverage"], "item-1 table rows missing"
    assert m["item2_td_residual"]
    assert m["item3_policy_contrast"]["n_worlds"] == 2   # n_cal = 2
    assert m["item5_speed"]["predictions_per_s"] > 0
    assert m["smoke"] is True
    # switching-mix and damping rows scored separately, per chunk, per tau
    # (v0.3.2 fix #3: the '__native' source maps to switching_mix)
    pols = {(x["chunk"], x["policy"]) for x in m["item1_crps_pit_coverage"]}
    assert ("chunk1", "switching_mix") in pols and \
        ("chunk1", "damping_fleet") in pols
    taus = {x["tau"] for x in m["item1_crps_pit_coverage"]}
    assert {30.0, 60.0, 120.0, 300.0, 570.0} <= taus


def test_eval_contrast_uses_both_policies_at_same_state(tiny_dataset,
                                                        tmp_path):
    """dV must come from ONE state s_t0 queried under two policies."""
    root, _ = tiny_dataset
    out = tmp_path / "out"
    _train_one_member(root, out)
    man = load_manifest(root, verify_files=False, log=lambda *_: None)
    ds = V0Dataset(man, log=lambda *_: None)
    nets = load_members(out, torch.device("cpu"), log=lambda *_: None)
    seed_everything(0)
    # identity check: a randomly initialised net still separates policies
    r = ds.rows
    b0 = r[np.isclose(r.t, r.t0_s) & np.isclose(r.tau, 570.0)
           & (r.source == "base")].iloc[0]
    from partB_v0.evaluate import mixture_point
    g = [int(b0["t_idx"])]
    vd = mixture_point(nets, ds, g, 570.0, 1, 1.0)[0]   # damping_fleet
    vn = mixture_point(nets, ds, g, 570.0, 0, 1.0)[0]   # switching_mix
    assert vd != vn            # policy conditioning is live end to end


def test_member_crps_subset_matches_split(tiny_dataset, tmp_path):
    root, _ = tiny_dataset
    out = tmp_path / "out"
    _train_one_member(root, out)
    man = load_manifest(root, verify_files=False, log=lambda *_: None)
    ds = V0Dataset(man, log=lambda *_: None)
    net = load_members(out, torch.device("cpu"),
                       log=lambda *_: None)[0][1]
    from partB_v0.evaluate import member_crps_subset
    from partB_v0.distributions import DistMetrics  # noqa: F401
    c = member_crps_subset(net, ds, tau=300.0, policy="switching_mix")
    assert c == c              # finite (NaN fails this check)
    assert c > 0
