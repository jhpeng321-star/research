"""--features deployable tests: the loader switch on the v0.3.1 sidecar
blocks, the tiny-data baseline degradation (BUNDLE_V0.3_RECORD.md
integration note), the gbm feature_set round-trip and the CLI guard."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from conftest_load import make_tiny_dataset


def json_loads_safe(p):
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture()
def v031_dataset(tmp_path):
    """(root, sha) of a tiny synthetic dataset WITH deployable sidecars.

    The sidecars keep the state blocks and replace demand (12 observed
    columns) and context (time-to-branch zeroed) -- written independently
    of the oracle npz so the values genuinely differ."""
    from partB_v0.data import load_manifest

    root = tmp_path / "data"
    make_tiny_dataset(root)
    man = load_manifest(root, verify_files=False, log=lambda *_: None)
    rng = np.random.default_rng(99)
    for key, entry in man.entries.items():
        rel = f"worlds/{entry['chunk_id']}/{entry['world']}__" \
              f"{entry['source']}_feat.npz"
        z = np.load(root / rel, allow_pickle=False)
        n = int(z["t"].shape[0])
        payload = {k: z[k] for k in z.files}
        payload["demand"] = rng.random((n, 12)).astype(np.float32)
        ctx = np.asarray(z["context"], np.float32).copy()
        ctx[:, 2] = 0.0
        payload["context"] = ctx
        payload["feature_set"] = np.str_("deployable")
        z.close()
        dep_rel = rel.replace("_feat.npz", "_feat_dep.npz")
        np.savez_compressed(root / dep_rel, **payload)
        doc = json_loads_safe(root / "MANIFEST.json")
        doc["worlds"][key]["files"][dep_rel] = \
            hashlib.sha256((root / dep_rel).read_bytes()).hexdigest()
        (root / "MANIFEST.json").write_text(
            json.dumps(doc, indent=1), encoding="utf-8")
    return root, None


def test_loader_switch_oracle_vs_deployable(v031_dataset):
    from partB_v0.data import V0Dataset, load_manifest
    root, _sha = v031_dataset
    log = lambda *a, **k: None
    man = load_manifest(root, verify_files=True, log=log)
    ds_o = V0Dataset(man, feature_set="oracle", log=log)
    ds_d = V0Dataset(man, feature_set="deployable", log=log)
    assert ds_o.feature_set == "oracle" and ds_d.feature_set == "deployable"
    # state blocks identical, demand/context swapped
    for k in ("lane_bins", "lane_mask", "cav_tokens", "cav_mask",
              "wall_tokens", "wall_mask"):
        np.testing.assert_array_equal(ds_o.feats[k], ds_d.feats[k])
    assert ds_o.feats["demand"].shape[1] == 21
    assert ds_d.feats["demand"].shape[1] == 12
    assert not np.array_equal(ds_o.feats["demand"][:, 0],
                              ds_d.feats["demand"][:, 0])
    assert (ds_d.feats["context"][:, 2] == 0.0).all()
    np.testing.assert_allclose(ds_o.feats["context"][:, 0],
                               ds_d.feats["context"][:, 0], atol=0)
    # alignment survives: same label rows, same t_idx
    assert ds_o.rows.equals(ds_d.rows)
    assert ds_o.td.equals(ds_d.td)
    # gather paths work on both
    idx = np.arange(len(ds_d.rows))
    b = ds_d.gather(idx)
    assert b["demand"].shape[1] == 12 and b["context"].shape[1] == 9


def test_summary_feature_names_follow_feature_set(v031_dataset):
    from partB_v0.data import V0Dataset, load_manifest
    from partB_v0.gbm import DEMAND_DEP_NAMES, row_feature_matrix
    root, _sha = v031_dataset
    log = lambda *a, **k: None
    man = load_manifest(root, verify_files=False, log=log)
    ds_d = V0Dataset(man, feature_set="deployable", log=log)
    _X, names = row_feature_matrix(ds_d)
    for n in DEMAND_DEP_NAMES:
        assert n in names
    assert "demand_0" not in names


def test_unknown_feature_set_rejected():
    from partB_v0.data import V0Dataset
    with pytest.raises(ValueError, match="feature set"):
        V0Dataset(None, load_features=False, feature_set="bogus")


def test_gbm_feature_set_roundtrip(tmp_path, tiny_dataset):
    """GbmValueModel.save/load carries the feature set."""
    from partB_v0.data import V0Dataset, load_manifest
    from partB_v0.gbm import GbmValueModel
    root, _sha = tiny_dataset
    log = lambda *a, **k: None
    ds = V0Dataset(load_manifest(root, verify_files=False, log=log),
                   log=log)
    m = GbmValueModel(max_iter=5, cal_folds=0, feature_set="deployable")
    m.fit(ds, log=log)
    side = m.save(tmp_path / "m")
    assert side["feature_set"] == "deployable"
    m2 = GbmValueModel.load(tmp_path / "m", log=log)
    assert m2.feature_set == "deployable"


def test_baselines_skip_tiny_slices(tiny_dataset):
    """TRAIN slices under min_rows are SKIPPED with a note instead of
    crashing HGB's early-stopping split on n_samples=1 (the previously
    failing bundle smoke test)."""
    from partB_v0.data import V0Dataset, load_manifest
    from partB_v0.gbm import baselines_section
    root, _sha = tiny_dataset
    log = lambda *a, **k: None
    ds = V0Dataset(load_manifest(root, verify_files=False, log=log,
                                 splits=("TRAIN", "CALIBRATION")),
                   log=log)
    # shrink TRAIN to one single row total so every tau slice is tiny
    r = ds.rows
    keep = r.index[r["split"] == "TRAIN"][:1]
    rows = ds.rows.copy()
    rows.loc[rows.index.difference(keep), "split"] = "CALIBRATION"
    ds.rows = rows
    cal_idx = rows.index[rows["split"] == "CALIBRATION"].to_numpy()[:20]
    med = rows["total_delay_ref_s"].to_numpy()[cal_idx]
    sec = baselines_section(ds, cal_idx, med, med * 0 + 1.0, log=log)
    assert sec["skipped_baselines"], "tiny taus must be skipped"
    assert sec["acceptance"] and \
        all(not a["hard_pass"] for a in sec["acceptance"])
    assert all(np.isnan(a["mae_ridge"]) for a in sec["acceptance"])
    assert "SKIPPED" in (sec.get("approx") or "")


def test_cli_refuses_net_with_deployable(tiny_dataset, tmp_path, capsys):
    from partB_v0.__main__ import main
    root, _sha = tiny_dataset
    rc = main(["train", "--data", str(root), "--out", str(tmp_path / "o"),
               "--value-model", "net", "--features", "deployable",
               "--no-verify"])
    assert rc == 2
    assert "deployable" in capsys.readouterr().err


def test_v03_net_trains_on_deployable_features(v031_dataset, tmp_path):
    """B3b: the v0.3 net sizes every input projection from the SAVED
    stats, so `net --features deployable` trains on the v0.3.1 sidecar
    blocks (12 observed demand columns) with no oracle-shaped layout."""
    from partB_v0.data import V0Dataset, V03TensorData, load_manifest
    from partB_v0.model import V03ValueConfig
    from partB_v0.train_net import ConcurrentNetTrainer, TrainNetConfig
    root, _sha = v031_dataset

    def log(*a, **k):
        pass
    man = load_manifest(root, verify_files=False,
                        splits=("TRAIN", "CALIBRATION"), log=log)
    ds = V0Dataset(man, feature_set="deployable", log=log)
    data = V03TensorData(ds, torch.device("cpu"), log=log)
    assert data.stats["feature_set"] == "deployable"
    dims = data.model_dims()
    assert dims["demand_raw"] == 12
    assert dims["summary_dim"] == 3 + 3 + 3 + 8 + 12 + 2
    data.save_stats(tmp_path / "stats.json")     # A4: ships with weights
    side = json_loads_safe(tmp_path / "stats.json")
    assert side["feature_set"] == "deployable"

    cfg = TrainNetConfig(data_root=str(root), out_dir=str(tmp_path / "out"),
                         members=1, batch_size=16, td_batch=8, pair_batch=4,
                         seed=5, verify=False, resume=False,
                         eval_every_steps=100, time_budget_min=0)
    cfg_model = V03ValueConfig(d_model=32, n_layers=1, dim_ff=64,
                               head_hidden=32, head_layers=1, tau_emb_dim=16,
                               **dims)
    tr = ConcurrentNetTrainer(cfg, data, torch.device("cpu"), "fp32",
                              log=log, cfg_model=cfg_model)
    loss, parts = tr._train_one_step(2)
    assert np.isfinite(loss)
    b = data.gather_rows(np.arange(4))
    assert b["summary"].shape == (4, dims["summary_dim"])
    assert b["demand"].shape == (4, 12)
