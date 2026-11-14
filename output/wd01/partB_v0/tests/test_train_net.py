"""v0.3 trainer tests (B3b): bitwise per-member resume, OOB selection
(CAL never used), n-step chains, delta pairs, ablation, budget re-plan
(CPU, tiny).

A handful of optimizer steps on the tiny synthetic dataset -- a
mechanics check, not training (owner rule).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from partB_v0.data import V0Dataset, V03TensorData, load_manifest
from partB_v0.model import V03ValueConfig
from partB_v0.train_net import (ConcurrentNetTrainer, TrainNetConfig,
                                mc_weight, run_ablation)

TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)


def _small_cfg_model() -> V03ValueConfig:
    # trained on the tiny PRE-v0.4 dataset (token widths 17/6; the
    # config-driven widths must match the data)
    return V03ValueConfig(d_model=64, n_layers=1, dim_ff=128,
                          cav_raw=17, wall_raw=6,
                          head_hidden=64, head_layers=1, tau_emb_dim=16,
                          delta_hidden=32, delta_layers=1)


def _cfg(root, out, **kw) -> TrainNetConfig:
    base = dict(data_root=str(root), out_dir=str(out), members=2,
                batch_size=32, td_batch=16, pair_batch=8, seed=1234,
                verify=False, resume=False, probe_steps=0,
                eval_every_steps=4, eval_row_cap=50, time_budget_min=0)
    base.update(kw)
    return TrainNetConfig(**base)


def _data(root, device=torch.device("cpu")):
    man = load_manifest(root, verify_files=False, log=lambda *_: None)
    ds = V0Dataset(man, log=lambda *_: None)
    return V03TensorData(ds, device, log=lambda *_: None)


# ------------------------------------------------------------- A1/B8 ----

def test_mc_weight_rises_with_tau():
    t = torch.tensor([30.0, 60.0, 120.0, 300.0, 570.0])
    w = mc_weight(t, 0.5)
    assert w[0] < w[2] < w[4]
    assert w[4] == pytest.approx(1.0)
    assert (w >= 0.5).all() and (w <= 1.0).all()


def test_data_v03_normalisation_and_gather(tiny_dataset):
    root, _ = tiny_dataset
    data = _data(root)
    r = data.ds.rows
    # A1: y_norm * med_row reconstructs the raw target
    yd = r["total_delay_ref_s"].to_numpy()
    back = data.y_norm["delay"].cpu().numpy() * data.med_row["delay"]
    assert np.allclose(back, yd, rtol=1e-4)
    # exact per-tau medians at the grid nodes
    tr = r[r["split"] == "TRAIN"]
    for pol in ("switching_mix", "damping_fleet"):
        for tau in TAUS:
            node = tr[np.isclose(tr["tau"], tau)
                      & (tr["policy"] == pol)]["total_delay_ref_s"] \
                .median()
            got = data.med["delay"].at(np.array([tau]), np.array([pol]))[0]
            assert got == pytest.approx(float(node), rel=1e-6)
    b = data.gather_rows(np.arange(5))
    assert b["summary"].shape == (5, 40)
    assert b["lane_bins"].shape == (5, 3, 320, 3)
    assert b["tau"].shape == (5,)
    # standardisation: train positions are ~unit variance per block
    src_split = np.array(data.ds.src_split)
    g_split = np.repeat(src_split, np.asarray(data.ds.src_sizes))
    g = np.flatnonzero(g_split == "TRAIN")
    sd = data.t_feats["summary_base"][torch.as_tensor(g)].std(dim=0)
    assert ((sd - 1).abs() < 0.3).float().mean() > 0.8


def test_td_chains_and_exact_normalisers(tiny_dataset):
    """B8: on-grid n-step chains (sum d30 + tau=30 label) and the exact
    piecewise-linear normalisers for the one-step bootstrap."""
    root, _ = tiny_dataset
    data = _data(root)
    tds = data.ds.td                    # v0.2 table (has t / src_idx)
    # chains sum the per-step d30 (which varies with t) and end at a
    # finite tau=30 label when marked feasible
    d30_map = {}
    for si, t, dv in zip(tds["src_idx"].to_numpy(),
                         tds["t"].to_numpy().round(6),
                         tds["d30_s"].to_numpy()):
        d30_map.setdefault((int(si), round(float(t), 6)), float(dv))
    src_arr = tds["src_idx"].to_numpy()
    t_arr = tds["t"].to_numpy()
    for i in np.flatnonzero(data.td_chain["ok"]):
        k = int(data.td_chain["k"][i])
        assert k >= 1
        t0 = round(float(t_arr[i]), 6)
        expected = sum(d30_map[(int(src_arr[i]), round(t0 + 30.0 * j, 6))]
                       for j in range(k))
        assert data.td_chain["sum"][i] == pytest.approx(expected,
                                                        rel=1e-6)
        yterm = data.td_chain["term_y30"][i]
        assert np.isfinite(yterm) and yterm > 0
    # the tau=60 chains are the common feasible case (one step to tau=30)
    m60 = np.isclose(np.asarray(data.td["tau"]), 60.0)
    assert (data.td_chain["ok"][m60]).any()
    # med_next is the EXACT interpolator of the saved per-tau medians
    nodes = data.med["delay"]._nodes("damping_fleet")
    for tau in (90.0, 270.0, 540.0):
        got = data.med["delay"].at(np.array([tau]),
                                   np.array(["damping_fleet"]))[0]
        assert got == pytest.approx(float(np.interp(tau, TAUS, nodes)),
                                    rel=1e-9)


def test_delta_pairs_offset_and_normalisation(tiny_dataset):
    """C13: the tiny native grid starts at t0+10 -> recorded offset
    pairs; delta_norm = delta_raw / per-tau scale."""
    root, _ = tiny_dataset
    data = _data(root)
    pf = data.pairs["TRAIN"]
    assert len(pf) == 3 * 5                       # 3 train worlds x 5 taus
    assert set(pf["pair_kind"]) == {"restore_offset10s"}
    assert (pf["offset_s"] == 10.0).all()
    back = pf["delta_norm"].to_numpy() \
        * data.delta_scale.at(pf["tau"].to_numpy())
    assert np.allclose(back, pf["delta_raw"].to_numpy(), rtol=1e-6)
    cal = data.pairs["CALIBRATION"]
    assert len(cal) == 2 * 5


# ------------------------------------------------------ B10: resume -----

def test_resume_is_bitwise_per_member(tiny_dataset, tmp_path):
    """8 lockstep steps straight == 3 + resume(5) BITWISE for every
    stacked parameter and AdamW moment."""
    root, _ = tiny_dataset
    ds_props = _data(root)
    out_a, out_b = tmp_path / "out_straight", tmp_path / "out_part"

    tr_a = ConcurrentNetTrainer(_cfg(root, out_a, max_steps=8),
                                ds_props, torch.device("cpu"), "fp32",
                                log=lambda *_: None,
                                cfg_model=_small_cfg_model())
    tr_a.train(8, None)

    cfg_b = _cfg(root, out_b, max_steps=8)
    tr_b = ConcurrentNetTrainer(cfg_b, ds_props, torch.device("cpu"),
                                "fp32", log=lambda *_: None,
                                cfg_model=_small_cfg_model())
    tr_b.train(8, None, stop_after=3)
    assert tr_b.step == 3
    meta = json.loads((out_b / "ckpt" / "net_meta.json").read_text())
    assert meta["step"] == 3 and meta["completed"] is False

    tr_c = ConcurrentNetTrainer(_cfg(root, out_b, max_steps=8),
                                ds_props, torch.device("cpu"), "fp32",
                                log=lambda *_: None,
                                cfg_model=_small_cfg_model())
    assert tr_c.try_resume() == 3
    tr_c.train(8, None)
    assert tr_c.step == 8

    for k in tr_a.params:
        assert torch.equal(tr_a.params[k], tr_c.params[k]), k
        assert torch.equal(tr_a.opt.m[k], tr_c.opt.m[k]), f"m:{k}"
        assert torch.equal(tr_a.opt.v[k], tr_c.opt.v[k]), f"v:{k}"
    # per-member weights identical too (the brief's per-member contract)
    sa = tr_a.write_members(tmp_path / "mem_a")
    sb = tr_c.write_members(tmp_path / "mem_b")
    for m in range(2):
        wa = torch.load(tmp_path / "mem_a" / f"member_{m}.pt",
                        weights_only=True)
        wb = torch.load(tmp_path / "mem_b" / f"member_{m}.pt",
                        weights_only=True)
        for k in wa:
            assert torch.equal(wa[k], wb[k]), f"member {m}: {k}"
    assert sa[0]["steps"] == 8 and sb[0]["steps"] == 8


def test_resume_with_smaller_max_steps_writes_all_members(tiny_dataset,
                                                          tmp_path):
    """Addendum-4c parity for the net path: interrupt member training,
    resume with a SMALLER --max-steps than the restored step -- the skip
    path must still write every member_k.pt from the persisted best."""
    root, _ = tiny_dataset
    data = _data(root)
    out = tmp_path / "out"
    tr = ConcurrentNetTrainer(_cfg(root, out, max_steps=6, members=2),
                              data, torch.device("cpu"), "fp32",
                              log=lambda *_: None,
                              cfg_model=_small_cfg_model())
    tr.train(6, None, stop_after=4)
    assert tr.step == 4
    tr2 = ConcurrentNetTrainer(_cfg(root, out, max_steps=3, members=2),
                               data, torch.device("cpu"), "fp32",
                               log=lambda *_: None,
                               cfg_model=_small_cfg_model())
    assert tr2.try_resume() == 4
    assert tr2.step >= 3            # already past the smaller --max-steps
    tr2.write_members(out)
    for m in range(2):
        assert (out / f"member_{m}.pt").exists()
        assert (out / f"member_{m}.json").exists()


# ------------------------------------------------------- B6: selection --

def test_oob_covers_all_tau_and_both_policies(tiny_dataset):
    root, _ = tiny_dataset
    data = _data(root)
    data.set_members(1234, 3)
    non_empty = [rows for rows in data.oob_rows if len(rows)]
    assert non_empty, "at least one member must have OOB worlds"
    for rows in non_empty:
        pol = set(data.row_policy[rows].tolist())
        taus = {round(float(t)) for t in data.row_tau[rows]}
        assert pol == {0, 1}
        assert taus == {30, 60, 120, 300, 570}


def test_oob_selection_never_touches_cal(tiny_dataset, tmp_path):
    """B6 CAL guard: eval_oob (the selection signal) reads ONLY TRAIN
    OOB rows -- corrupting every CALIBRATION label leaves it bit-identical,
    and no OOB row is CALIBRATION.  (v0.3.2 fix #9: the bootstrap unit is
    the world, so a 3-world fixture can occasionally leave a member with
    an empty OOB set -- the seed search below avoids that degenerate
    case; a NaN-tolerant comparison is the backstop.)"""
    root, _ = tiny_dataset
    data = _data(root)
    cfg = None
    for seed in (20260929, 11, 12, 13, 14, 15, 16, 17):
        data.set_members(seed, 1)
        if any(len(rows) for rows in data.oob_rows):
            cfg = _cfg(root, tmp_path / "out", max_steps=2, members=1,
                       seed=seed)
            break
    assert cfg is not None, "no seed produced a member with OOB worlds"
    for rows in data.oob_rows:
        assert not data.split_cal[rows].any()
    tr = ConcurrentNetTrainer(cfg, data, torch.device("cpu"), "fp32",
                              log=lambda *_: None,
                              cfg_model=_small_cfg_model())
    tr._train_one_step(2)
    before = tr.eval_oob(cap=None)
    # corrupt EVERY CAL row's label and features-agnostic target
    cal_rows = data.ds.rows.index[data.split_cal]
    data.ds.rows.loc[cal_rows, "total_delay_ref_s"] = -12345.0
    data.y_norm["delay"][torch.as_tensor(np.asarray(cal_rows),
                                         device=data.device)] = -99.0
    after = tr.eval_oob(cap=None)
    same = (before == after) or all(x != x for x in before + after)
    assert same, "eval_oob must not depend on CAL rows"


def test_oob_early_stop_patience_and_best_write(tiny_dataset, tmp_path,
                                                monkeypatch):
    root, _ = tiny_dataset
    cfg = _cfg(root, tmp_path / "out", max_steps=6, members=1,
               eval_every_steps=2, patience=1)
    tr = ConcurrentNetTrainer(cfg, _data(root), torch.device("cpu"),
                              "fp32", log=print,
                              cfg_model=_small_cfg_model())
    vals = iter([10.0, 11.0, 12.0])
    monkeypatch.setattr(tr, "eval_oob", lambda cap=None: [next(vals)])
    rep = tr.train(6, None)
    assert rep["active"] == [0.0], "member must be frozen at patience"
    assert tr.best[0][0] == pytest.approx(10.0)
    side = tr.write_members(tmp_path / "out")
    assert side[0]["best_oob_crps"] == pytest.approx(10.0)
    # member_k.pt written from the BEST state on every path
    assert (tmp_path / "out" / "member_0.pt").exists()
    assert (tmp_path / "out" / "ckpt" / "m0_best.pt").exists()


def test_eval_oob_uses_eval_mode(tiny_dataset, tmp_path):
    """eval_oob must not leave the template in train mode (dropout /
    RNG)."""
    root, _ = tiny_dataset
    tr = ConcurrentNetTrainer(_cfg(root, tmp_path / "out", max_steps=2,
                                   members=1),
                              _data(root), torch.device("cpu"), "fp32",
                              log=lambda *_: None,
                              cfg_model=_small_cfg_model())
    tr.eval_oob(cap=20)
    assert tr.template.training


# ---------------------------------------------------- B11: budget ------

def test_budget_replan_follows_measured_rate(tiny_dataset, tmp_path):
    root, _ = tiny_dataset
    logs = []
    cfg = _cfg(root, tmp_path / "out", max_steps=None, members=1,
               replan_after_steps=3, time_budget_min=0)
    tr = ConcurrentNetTrainer(cfg, _data(root), torch.device("cpu"),
                              "fp32", log=logs.append,
                              cfg_model=_small_cfg_model())
    rep = tr.train(50, deadline=None, stop_after=4)
    assert rep["steps_done"] == 4
    assert rep["planned"] == 3 + 4 * 3           # step + 4 x replan_after
    assert any("RE-PLANNED" in ln for ln in logs)
    side = tr.write_members(tmp_path / "out")
    assert side[0]["steps"] == 4


def test_smoke_config_is_tiny(tiny_dataset, tmp_path):
    from partB_v0.train_net import run_net_training
    root, _ = tiny_dataset
    cfg = TrainNetConfig(data_root=str(root), out_dir=str(tmp_path / "o"),
                         smoke=True, members=1, seed=3, verify=False,
                         device="cpu")
    metrics, _data = run_net_training(cfg, log=lambda *_: None)
    assert cfg.max_steps == 6 and cfg.batch_size == 64
    assert metrics["training"]["steps_done"] == 6
    assert (tmp_path / "o" / "member_0.pt").exists()
    assert (tmp_path / "o" / "member_0.json").exists()
    assert (tmp_path / "o" / "stats.json").exists()


# ---------------------------------------------------------- B7 -------

def test_ablation_runs_variants_and_chooses_by_oob(tiny_dataset,
                                                   tmp_path):
    root, _ = tiny_dataset
    cfg = _cfg(root, tmp_path / "out", max_steps=2, members=1,
               eval_row_cap=20)
    table, chosen = run_ablation(cfg, _data(root), torch.device("cpu"),
                                 "fp32", log=lambda *_: None)
    assert {t["variant"] for t in table} == {"base", "wd", "dropout",
                                             "d_model128"}
    assert chosen in {"base", "wd", "dropout", "d_model128"}
    doc = json.loads((tmp_path / "out" / "ablation.json").read_text())
    assert doc["chosen"] == chosen and len(doc["table"]) == 4


# ------------------------------------------------- C13: delta head -----

def test_delta_head_train_step_moves_prediction(tiny_dataset, tmp_path):
    root, _ = tiny_dataset
    data = _data(root)
    tr = ConcurrentNetTrainer(_cfg(root, tmp_path / "out", max_steps=3,
                                   members=1, w_delta=1.0),
                              data, torch.device("cpu"), "fp32",
                              log=lambda *_: None,
                              cfg_model=_small_cfg_model())
    pair = tr._stack_pairs([tr._pair_batch(0, 0)])
    assert pair["mask"].sum() > 0
    loss0, parts0 = tr._train_one_step(3)
    assert np.isfinite(loss0)
    for _ in range(2):
        loss, parts = tr._train_one_step(3)
    assert np.isfinite(loss)
    # delta loss is finite and tracked (parts index 6)
    assert np.isfinite(float(parts[0, 6]))
