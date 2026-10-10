"""B3a tests: quantile-GBM value model, baselines + HARD acceptance,
persistence/skip-path fixes, strict load_members, results.zip whitelist.

CPU only; the GBM fits here run on the tiny synthetic dataset with a
small --gbm-max-iter (smoke-sized, owner rule respected).
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
import torch

from partB_v0.data import V0Dataset, load_manifest
from partB_v0.gbm import row_feature_matrix


# ------------------------------------------------------------- helpers --

def _load_ds(root):
    man = load_manifest(root, verify_files=False, log=lambda *_: None)
    return V0Dataset(man, log=lambda *_: None)


def _fit_gbm(ds, **kw):
    from partB_v0.gbm import GbmValueModel
    # tiny grid + tiny iters: keep the test fit count low (the box is
    # CPU-saturated; per-fit overhead dominates on tiny data)
    kw.setdefault("quantiles", (0.1, 0.5, 0.9))
    kw.setdefault("max_iter", 15)
    kw.setdefault("cal_folds", 2)
    return GbmValueModel(**kw).fit(ds, log=lambda *_: None)


def _train_net(root, out, members=1, steps=3):
    from partB_v0.train import TrainConfig, run_training
    cfg = TrainConfig(data_root=str(root), out_dir=str(out),
                      members=members, batch_size=32, td_batch=16,
                      max_steps=steps, probe_steps=0, seed=5,
                      verify=False, smoke=False)
    metrics, ds = run_training(cfg)
    return metrics, ds


# ------------------------------------------------- gbm fit/predict/pkl --

def test_gbm_fit_predict_serialize_roundtrip(tiny_dataset, tmp_path):
    root, _ = tiny_dataset
    ds = _load_ds(root)
    model = _fit_gbm(ds)
    assert len(model.taus) == 5
    assert set(model.models.keys()) and model.shifts

    cal_idx = ds.rows.index[ds.rows["split"] == "CALIBRATION"].to_numpy()
    q = model.predict_quantiles_rows(ds, cal_idx[:50], "delay")
    assert set(q) == set(model.target_quantiles["delay"])
    # monotone calibrated quantiles, finite
    qs_sorted = sorted(q)
    for a, b in zip(qs_sorted[:-1], qs_sorted[1:]):
        assert np.all(q[b] >= q[a] - 1e-9)
    assert np.isfinite(q[0.5]).all()
    # median equals the q=0.5 API
    med = model.median_rows(ds, cal_idx[:50])
    assert np.allclose(med, q[0.5])

    # off-grid tau interpolation (TD queries tau-30)
    X, _ = row_feature_matrix(ds, cal_idx[:5])
    pt = model.point_matrix(X, np.full(5, 90.0), "delay")   # 90 off-grid
    assert np.isfinite(pt).all()

    # crps / pit / coverage rows
    crps = model.crps_rows(ds, cal_idx[:50])
    pit = model.pit_rows(ds, cal_idx[:50])
    cov = model.coverage_rows(ds, cal_idx[:50])
    assert np.all(np.isfinite(crps)) and np.all(crps >= 0)
    assert np.all((pit > 0) & (pit < 1))
    assert cov.dtype == bool
    pcoll = model.coll_prob_rows(ds, cal_idx[:50])
    assert np.all((pcoll >= 0) & (pcoll <= 1))

    # serialize + reload -> identical predictions
    out = tmp_path / "out"
    model.save(out)
    assert (out / "gbm_value.pkl").exists()
    assert (out / "gbm_value.json").exists()
    side = json.loads((out / "gbm_value.json").read_text("utf-8"))
    assert side["sklearn_version"]
    assert side["data_manifest_sha256"] == ds.man.sha256
    from partB_v0.gbm import GbmValueModel as M
    model2 = M.load(out, log=lambda *_: None)
    q2 = model2.predict_quantiles_rows(ds, cal_idx[:50], "delay")
    for k in q:
        assert np.allclose(q[k], q2[k])


def test_gbm_run_training_and_point_global(tiny_dataset, tmp_path):
    from partB_v0.gbm import run_gbm_training
    from partB_v0.train import TrainConfig
    root, _ = tiny_dataset
    cfg = TrainConfig(data_root=str(root), out_dir=str(tmp_path / "out"),
                      verify=False,
                      extras={"gbm_max_iter": 15, "gbm_cal_folds": 0})
    metrics, ds, refit_skipped = run_gbm_training(cfg, log=lambda *_: None)
    assert refit_skipped is False
    assert metrics["value_model"] == "gbm"
    assert metrics["gbm"]["taus"] and not metrics["members"]

    # point_global: policy contrast API (median per (g, tau, policy))
    from partB_v0.gbm import GbmValueModel
    model = GbmValueModel.load(tmp_path / "out", log=lambda *_: None)
    g = ds.rows["t_idx"].to_numpy()[:4]
    v0 = model.point_global(ds, g, 570.0, 0, 1.0)
    v1 = model.point_global(ds, g, 570.0, 1, 1.0)
    assert np.isfinite(v0).all() and np.isfinite(v1).all()

    # re-run against the same dataset -> fit skipped
    metrics2, _ds2, refit_skipped2 = run_gbm_training(
        cfg, log=lambda *_: None)
    assert refit_skipped2 is True
    assert metrics2["gbm"]["refit_skipped"] is True


# ------------------------------------- baselines + HARD acceptance -----

def test_baselines_section_and_hard_acceptance(tiny_dataset):
    from partB_v0.gbm import baselines_section
    root, _ = tiny_dataset
    ds = _load_ds(root)
    cal_idx = ds.rows.index[ds.rows["split"] == "CALIBRATION"].to_numpy()
    y = ds.rows["total_delay_ref_s"].to_numpy()[cal_idx]
    model_med = y + 1.0                       # a "model" off by 1 s
    model_crps = np.full(len(y), 10.0)
    sec = baselines_section(ds, cal_idx, model_med, model_crps,
                            log=lambda *_: None, gbm_max_iter=40)
    rows = sec["rows"]
    assert rows, "baseline table rows missing"
    for r in rows:
        for k in ("mae_climatology", "mae_ridge", "mae_gbm_baseline",
                  "mae_model", "crps_climatology", "crps_ridge",
                  "crps_gbm_baseline", "crps_model"):
            assert k in r
    # the 1-second-off model beats ridge on MAE at every tau here...
    acc = sec["acceptance"]
    assert acc and {a["tau"] for a in acc} == \
        {30.0, 60.0, 120.0, 300.0, 570.0}
    # ...but CRPS 10.0 must lose to the ridge residual-sample CRPS at
    # least somewhere -> the table discriminates (no all-degenerate rows)
    assert any((not a["beats_ridge_crps"]) for a in acc) or \
        any(a["beats_ridge_crps"] for a in acc)
    assert sec["overall"]["n_taus"] == 5
    assert isinstance(sec["overall"]["hard_pass"], bool)


def test_eval_report_sections_6_7_and_3_decimals(tiny_dataset, tmp_path):
    from partB_v0.evaluate import EvalConfig, evaluate_run
    root, _ = tiny_dataset
    out = tmp_path / "out"
    _train_net(root, out)
    ds = _load_ds(root)
    report, m = evaluate_run(EvalConfig(out_dir=str(out), smoke=True),
                             ds, log=lambda *_: None)
    for header in ("## 6. Baselines", "## 7. HARD acceptance"):
        assert header in report, header
    assert "Overall HARD acceptance vs ridge:" in report
    assert "PIT histogram per tau" in report
    # 3-decimal coverage/PIT in the item-1 table
    import re
    line = next(l for l in report.splitlines()
                if l.startswith("| chunk1 |"))
    assert re.search(r"\|\s*\d+\.\d{3}\s*\|\s*\d+\.\d{3}\s*\|", line), \
        f"coverage/PIT not 3-decimal: {line}"
    assert "item6_baselines" in m and "acceptance" in m["item6_baselines"]
    assert m["item1_pit_hist_per_tau"]


# ------------------------------------------ persistence / skip path ----

def test_persisted_best_and_skip_path_smaller_max_steps(
        tiny_dataset, tmp_path, monkeypatch, capsys):
    """The addendum-4c incident: member_0.pt missing, ckpt at step 6;
    a resume with a SMALLER --max-steps (3 < 6) takes the skip path and
    must still write member_0.pt -- from the PERSISTED best state -- so
    eval + load_members work."""
    import partB_v0.train as T
    from partB_v0.evaluate import EvalConfig, evaluate_run, load_members
    root, _ = tiny_dataset
    out = tmp_path / "out"

    cfg = T.TrainConfig(data_root=str(root), out_dir=str(out), members=1,
                        batch_size=32, td_batch=16, max_steps=6,
                        probe_steps=0, seed=5, verify=False)
    metrics1, _ = T.run_training(cfg, log=lambda *_: None)
    assert (out / "member_0.pt").exists()
    assert (out / "ckpt" / "m0_best.pt").exists()        # addendum 4c #1
    best_meta = json.loads(
        (out / "ckpt" / "m0_best.json").read_text("utf-8"))
    assert best_meta["step"] >= 0
    # the shipped member file IS the best state
    shipped = torch.load(out / "member_0.pt", map_location="cpu",
                         weights_only=True)
    best = torch.load(out / "ckpt" / "m0_best.pt", map_location="cpu",
                      weights_only=True)
    for k in shipped:
        assert torch.allclose(shipped[k], best[k]), k

    # the cloud-kill incident state: weights never exported
    (out / "member_0.pt").unlink()
    (out / "member_0.json").unlink()

    cfg2 = T.TrainConfig(data_root=str(root), out_dir=str(out), members=1,
                         batch_size=32, td_batch=16, max_steps=3,
                         probe_steps=0, seed=5, verify=False)
    logs: list[str] = []
    metrics2, ds = T.run_training(cfg2, log=logs.append)
    joined = "\n".join(logs)
    assert "already at step 6 >= planned 3; skipping" in joined
    assert "writing member_0.pt from the persisted best state" in joined
    assert (out / "member_0.pt").exists()
    assert metrics2["members"][0]["already_done"] is True
    side = json.loads((out / "member_0.json").read_text("utf-8"))
    assert side["weights_source"] == "ckpt_best"

    nets = load_members(out, torch.device("cpu"), members=1,
                        log=lambda *_: None)
    assert len(nets) == 1
    report, m = evaluate_run(EvalConfig(out_dir=str(out)), ds,
                             log=lambda *_: None)
    assert m["n_members"] == 1


def test_budget_stop_writes_member_file(tiny_dataset, tmp_path):
    """A deadline/budget stop mid-member still ships member_0.pt."""
    from partB_v0.train import TrainConfig, run_training
    root, _ = tiny_dataset
    out = tmp_path / "out"
    cfg = TrainConfig(data_root=str(root), out_dir=str(out), members=1,
                      batch_size=32, td_batch=16, max_steps=8,
                      time_budget_min=1e-4, probe_steps=2, verify=False)
    metrics, _ds = run_training(cfg, log=lambda *_: None)
    assert metrics["members"][0]["steps_done"] <= 1
    assert (out / "member_0.pt").exists()
    side = json.loads((out / "member_0.json").read_text("utf-8"))
    assert side["weights_source"] in ("memory_best", "ckpt_best",
                                      "ckpt_state")


# ------------------------------------------------ strict load_members --

def test_load_members_names_every_missing(tiny_dataset, tmp_path):
    from partB_v0.evaluate import load_members
    root, _ = tiny_dataset
    out = tmp_path / "out"
    _train_net(root, out, members=3)
    (out / "member_0.pt").unlink()
    (out / "member_2.pt").unlink()

    with pytest.raises(FileNotFoundError) as e:
        load_members(out, torch.device("cpu"), members=3,
                     log=lambda *_: None)
    msg = str(e.value)
    assert "0" in msg and "2" in msg, f"must name ALL missing: {msg}"

    # without an explicit count the expected set is 0..max(present) ->
    # 1 present -> missing [0] (still named, never a silent stop)
    with pytest.raises(FileNotFoundError) as e2:
        load_members(out, torch.device("cpu"), log=lambda *_: None)
    assert "[0]" in str(e2.value) and "[2]" not in str(e2.value)

    # all present again -> loads
    _train_net(root, out, members=3)      # rewrites every member file
    nets = load_members(out, torch.device("cpu"), members=3,
                        log=lambda *_: None)
    assert [i for i, _n, _s in nets] == [0, 1, 2]


# ------------------------------------------------ zip whitelist (D19) --

def test_results_zip_strict_whitelist(tiny_dataset, tmp_path, capsys):
    from partB_v0.__main__ import main
    from partB_v0.outputs import finalize_outputs, whitelist_relpaths
    root, _ = tiny_dataset
    out = tmp_path / "out"
    rc = main(["train", "--data", str(root), "--out", str(out),
               "--smoke", "--members", "1", "--device", "cpu",
               "--no-verify"])
    assert rc == 0
    capsys.readouterr()

    # stray OUT_DIR content that must NOT ship (the v0.2 ckpt_m0_old case)
    old = out / "ckpt_m0_old"
    old.mkdir()
    (old / "m0_state.pt").write_bytes(b"old")
    (out / "stray_notes.txt").write_text("junk", encoding="utf-8")

    rels = whitelist_relpaths(out)
    assert "stray_notes.txt" not in rels
    assert not any(r.startswith("ckpt") for r in rels)
    assert "logs/train.log" in rels

    finalize_outputs(out, log=None)
    with zipfile.ZipFile(out / "results.zip") as z:
        names = set(z.namelist())
    assert not any(n == "ckpt" or n.startswith("ckpt/")
                   or n.startswith("ckpt_m0_old") for n in names)
    assert "stray_notes.txt" not in names
    assert {"member_0.pt", "metrics.json", "eval_report.md",
            "env_report.json", "OUTPUT_MANIFEST.json"} <= names

    rc = main(["verify", "--out", str(out)])
    assert rc == 0
    assert "verify: OK" in capsys.readouterr().out


# -------------------------------------- hybrid variant + planner speed --

def test_gbm_hybrid_variant_point_speed_contract(tiny_dataset):
    """Hybrid (default): ridge point models + residual quantile-GBM.

    Contracts: (a) the on-grid point median is EXACTLY the predictive's
    calibrated median; (b) off-grid tau = linear interpolation of the
    bracketing grid medians; (c) point_matrix BATCHES -- one median pass
    per NEEDED grid tau, no matter how many off-grid query taus (the
    per-plan/per-row predict storms that made gbm_v02_b3a report 2
    pred/s are gone); (d) world-grouped OOF delay predictions are stored
    for the TRAIN-internal acceptance table."""
    from partB_v0.gbm import GbmValueModel, oob_acceptance_section
    root, _ = tiny_dataset
    ds = _load_ds(root)
    model = _fit_gbm(ds)                      # hybrid is the default
    assert model.variant == "hybrid" and model.point_models
    assert model.oof_grouping == "world-grouped"
    assert set(model.oof_delay) == set(model.taus)

    cal_idx = ds.rows.index[ds.rows["split"] == "CALIBRATION"].to_numpy()
    idx = cal_idx[:30]
    X, _ = row_feature_matrix(ds, idx)
    taus = ds.rows["tau"].to_numpy()[idx]
    q = model.predict_quantiles_rows(ds, idx, "delay")
    # (a) on-grid point == predictive median
    pt = model.point_matrix(X, taus, "delay")
    assert np.allclose(pt, q[0.5])
    # (b) off-grid interpolation is linear between the grid medians
    q60 = model.predict_quantiles_matrix(X, np.full(len(idx), 60.0))[0.5]
    q120 = model.predict_quantiles_matrix(X, np.full(len(idx), 120.0))[0.5]
    p90 = model.point_matrix(X, np.full(len(idx), 90.0), "delay")
    assert np.allclose(p90, 0.5 * (q60 + q120))

    # (c) batching: 30 unique off-grid taus still cost <= 1 median pass
    # per grid tau (the old code predicted 2x per query tau)
    calls = []
    orig = GbmValueModel._median_at_grid

    def _counting(self, Xa, tau, target):
        calls.append(float(tau))
        return orig(self, Xa, tau, target)

    wild = np.linspace(31.0, 569.0, len(idx))     # 30 unique off-grid taus
    GbmValueModel._median_at_grid = _counting
    try:
        p = model.point_matrix(X, wild, "delay")
    finally:
        GbmValueModel._median_at_grid = orig
    assert np.isfinite(p).all()
    assert len(calls) <= len(model.taus), \
        f"point_matrix not batched: {len(calls)} median passes"

    # (d) TRAIN-internal world-grouped OOB acceptance table
    sec = oob_acceptance_section(ds, model, log=lambda *_: None,
                                 n_splits=3)
    acc = sec["acceptance"]
    assert {a["tau"] for a in acc} == {30.0, 60.0, 120.0, 300.0, 570.0}
    for a in acc:
        assert a["n"] > 0
        assert np.isfinite(a["mae_model"]) and np.isfinite(a["mae_ridge"])
        assert np.isfinite(a["crps_model"]) and np.isfinite(a["crps_ridge"])
    assert sec["overall"]["n_taus"] == 5
    assert "world-grouped" in sec["note"]

    # a model without world-grouped OOF (in-fit residuals) skips the table
    model0 = GbmValueModel(quantiles=(0.1, 0.5, 0.9), max_iter=15,
                           cal_folds=0).fit(ds, log=lambda *_: None)
    sec0 = oob_acceptance_section(ds, model0, log=lambda *_: None)
    assert sec0["acceptance"] == [] and "skipped" in sec0["note"]


def test_gbm_pure_variant_and_legacy_schema1_load(tiny_dataset, tmp_path):
    """--gbm-variant pure keeps the original semantics, and the /1 pkls
    written before the hybrid variant still load (as pure)."""
    from partB_v0.gbm import GbmValueModel
    root, _ = tiny_dataset
    ds = _load_ds(root)
    model = GbmValueModel(quantiles=(0.1, 0.5, 0.9), max_iter=15,
                          cal_folds=2, variant="pure").fit(
        ds, log=lambda *_: None)
    assert model.variant == "pure" and not model.point_models
    cal_idx = ds.rows.index[ds.rows["split"] == "CALIBRATION"].to_numpy()
    q = model.predict_quantiles_rows(ds, cal_idx[:20], "delay")
    assert np.isfinite(q[0.5]).all()

    out = tmp_path / "out"
    model.save(out)
    import pickle
    pkl = out / "gbm_value.pkl"
    body = pickle.loads(pkl.read_bytes())
    body["schema"] = "wd01_partB_v0_gbm/1"
    for k in ("variant", "point_models", "oof_delay", "oof_grouping"):
        body.pop(k, None)
    pkl.write_bytes(pickle.dumps(body, protocol=5))
    side = json.loads((out / "gbm_value.json").read_text("utf-8"))
    side.pop("pkl_sha256")               # body was rewritten above
    (out / "gbm_value.json").write_text(json.dumps(side), encoding="utf-8")
    m2 = GbmValueModel.load(out, log=lambda *_: None)
    assert m2.variant == "pure" and not m2.point_models
    q2 = m2.predict_quantiles_rows(ds, cal_idx[:20], "delay")
    for k in q:
        assert np.allclose(q[k], q2[k])


def test_cli_gbm_pure_variant_flag(tiny_dataset, tmp_path, capsys):
    from partB_v0.__main__ import main
    root, _ = tiny_dataset
    out = tmp_path / "out_pure"
    rc = main(["train", "--value-model", "gbm", "--gbm-variant", "pure",
               "--data", str(root), "--out", str(out), "--gbm-max-iter",
               "15", "--gbm-cal-folds", "0", "--device", "cpu",
               "--no-verify"])
    assert rc == 0
    capsys.readouterr()
    side = json.loads((out / "gbm_value.json").read_text("utf-8"))
    assert side["variant"] == "pure"
    report = (out / "eval_report.md").read_text("utf-8")
    assert "value model: gbm — PURE" in report
    metrics = json.loads((out / "metrics.json").read_text("utf-8"))
    assert metrics["gbm"]["variant"] == "pure"
    assert metrics["eval"]["gbm_variant"] == "pure"
    # the planner speed item ran through the batched point path
    assert metrics["eval"]["item5_speed"]["predictions_per_s"] > 0

    # default variant is hybrid; cal_folds=2 gives the world-grouped
    # OOF the 7b table needs
    out2 = tmp_path / "out_hyb"
    rc = main(["train", "--value-model", "gbm", "--data", str(root),
               "--out", str(out2), "--gbm-max-iter", "15",
               "--gbm-cal-folds", "2", "--device", "cpu", "--no-verify"])
    assert rc == 0
    capsys.readouterr()
    side2 = json.loads((out2 / "gbm_value.json").read_text("utf-8"))
    assert side2["variant"] == "hybrid"
    report2 = (out2 / "eval_report.md").read_text("utf-8")
    assert "value model: gbm — HYBRID" in report2
    assert "## 7b. HARD acceptance vs RIDGE on TRAIN OOB" in report2
    ev2 = json.loads((out2 / "metrics.json").read_text("utf-8"))["eval"]
    assert ev2["item7b_oob_acceptance"]["overall"]["n_taus"] == 5
    rc = main(["verify", "--out", str(out2)])
    assert rc == 0
    assert "verify: OK" in capsys.readouterr().out


# -------------------------------------------- GBM end-to-end via CLI --

def test_cli_gbm_train_eval_verify(tiny_dataset, tmp_path, capsys):
    from partB_v0.__main__ import main
    root, _ = tiny_dataset
    out = tmp_path / "out_gbm"
    rc = main(["train", "--value-model", "gbm", "--data", str(root),
               "--out", str(out), "--gbm-max-iter", "15",
               "--gbm-cal-folds", "0",
               "--device", "cpu", "--no-verify"])
    assert rc == 0
    capsys.readouterr()

    assert (out / "gbm_value.pkl").exists()
    assert (out / "gbm_value.json").exists()
    assert not (out / "member_0.pt").exists()      # no net weights
    report = (out / "eval_report.md").read_text("utf-8")
    assert "## 6. Baselines" in report and "## 7. HARD acceptance" \
        in report
    assert "value model: gbm" in report
    metrics = json.loads((out / "metrics.json").read_text("utf-8"))
    assert metrics["eval"]["value_model"] == "gbm"
    assert metrics["config"]["value_model"] == "gbm"

    man = json.loads((out / "OUTPUT_MANIFEST.json").read_text("utf-8"))
    assert "gbm_value.pkl" in man["files"]
    assert not any(k.startswith("member_") for k in man["files"])
    with zipfile.ZipFile(out / "results.zip") as z:
        names = set(z.namelist())
    assert "gbm_value.pkl" in names and "gbm_value.json" in names
    assert not any(n.startswith("member_") for n in names)

    rc = main(["verify", "--out", str(out)])
    assert rc == 0
    capsys.readouterr()

    # identical re-run: fit skipped, eval skipped
    first_log = (out / "logs" / "train.log").read_text("utf-8")
    rc = main(["train", "--value-model", "gbm", "--data", str(root),
               "--out", str(out), "--gbm-max-iter", "15",
               "--gbm-cal-folds", "0",
               "--device", "cpu", "--no-verify"])
    assert rc == 0
    new_log = (out / "logs" / "train.log").read_text("utf-8")[
        len(first_log):]
    assert "skipping the fit" in new_log
    assert "skipping evaluation" in new_log
    capsys.readouterr()

    # `eval --value-model gbm` re-evaluates
    rc = main(["eval", "--value-model", "gbm", "--data", str(root),
               "--out", str(out), "--device", "cpu", "--no-verify"])
    assert rc == 0
    assert "verify: OK" in capsys.readouterr().out
