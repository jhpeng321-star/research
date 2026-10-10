"""v0.4 action/wall input tests on the trainer side (brief items 2-3).

* config-driven token widths with a LOUD failure on a width mismatch with
  old checkpoints (v0.4 bumped CAV_RAW 17 -> 20 and WALL_RAW 6 -> 11);
* data-driven dims (V03TensorData.model_dims now also carries wall_raw);
* strict-variant field audit extended to the new fields: the action and
  wall token blocks are planner/RSU-known, so the HEADLINE deployable
  variant carries them unchanged (identical to the sensitivity variant).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

def _load_helpers():
    """Explicit loader (conftest_load.py pattern): keeps this module
    collectable even in MIXED invocations of the partB_v0* test trees,
    where a bare `conftest_load` import may bind to the other tree."""
    import importlib.util
    path = Path(__file__).resolve().parent / "conftest_load.py"
    spec = importlib.util.spec_from_file_location(
        "_partb_v0_action_wall_conftest_load", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make_tiny_dataset = _load_helpers().make_tiny_dataset


@pytest.fixture()
def v031_dataset(tmp_path):
    """(root, sha) of a tiny synthetic dataset WITH deployable sidecars
    (same construction as test_features_deployable.v031_dataset)."""
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
        det_rel = rel.replace("_feat.npz", "_feat_dep_det.npz")
        np.savez_compressed(root / det_rel, **dict(
            payload, feature_set=np.str_("deployable_detector"),
            demand=rng.random((n, 12)).astype(np.float32)))
        str_rel = rel.replace("_feat.npz", "_feat_dep_strict.npz")
        np.savez_compressed(root / str_rel, **dict(
            payload, feature_set=np.str_("deployable_strict"),
            demand=rng.random((n, 6)).astype(np.float32)))
        doc = json.loads((root / "MANIFEST.json").read_text("utf-8"))
        for extra in (dep_rel, det_rel, str_rel):
            doc["worlds"][key]["files"][extra] = \
                hashlib.sha256((root / extra).read_bytes()).hexdigest()
        (root / "MANIFEST.json").write_text(
            json.dumps(doc, indent=1), encoding="utf-8")
    return root, None


def _old_width_state():
    from partB_v0.model import V0ValueConfig, V0ValueNet
    cfg = V0ValueConfig(d_model=32, n_layers=1, dim_ff=64, head_hidden=32,
                        cav_raw=17, wall_raw=6)      # pre-v0.4 widths
    return V0ValueNet(cfg).state_dict(), cfg


def test_check_token_widths_fails_loud_on_old_checkpoint():
    from partB_v0.model import V0ValueConfig, check_token_widths
    state, _old = _old_width_state()
    with pytest.raises(ValueError) as ei:
        check_token_widths(state, V0ValueConfig(), where="member_0.pt")
    msg = str(ei.value)
    assert "token-width mismatch" in msg
    assert "17" in msg and "20" in msg and "6" in msg and "11" in msg
    assert "cfg_model" in msg          # points at the recorded-config fix


def test_check_token_widths_accepts_matching_state():
    from partB_v0.model import (V0ValueConfig, V0ValueNet,
                                check_token_widths)
    state, cfg = _old_width_state()
    check_token_widths(state, cfg)          # faithful reload: no error
    check_token_widths(V0ValueNet(V0ValueConfig()).state_dict(),
                       V0ValueConfig())


def test_load_members_loud_on_sidecarless_old_weights(tmp_path):
    """An OLD member_*.pt without a cfg_model sidecar must fail LOUDLY at
    load (never a generic torch shape error, never a silent wrong net)."""
    from partB_v0.evaluate import load_members
    state, _cfg = _old_width_state()
    torch.save(state, tmp_path / "member_0.pt")
    (tmp_path / "member_0.json").write_text(
        json.dumps({"format": "v0", "cfg_model": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="token-width mismatch"):
        load_members(tmp_path, torch.device("cpu"), log=lambda *_: None)


def test_load_members_honours_recorded_cfg_widths(tmp_path):
    """Old checkpoints that RECORD their widths reload faithfully."""
    from partB_v0.evaluate import load_members
    state, cfg = _old_width_state()
    torch.save(state, tmp_path / "member_0.pt")
    (tmp_path / "member_0.json").write_text(
        json.dumps({"format": "v0", "cfg_model": cfg.to_json()}),
        encoding="utf-8")
    nets = load_members(tmp_path, torch.device("cpu"),
                        log=lambda *_: None)
    assert len(nets) == 1
    assert nets[0][1].cfg.cav_raw == 17 and nets[0][1].cfg.wall_raw == 6


def test_config_width_roundtrip_and_forward_guard():
    from partB_v0.model import V03ValueConfig, V03ValueNet, V0ValueConfig
    for cfg_cls in (V0ValueConfig, V03ValueConfig):
        cfg = cfg_cls()                    # v0.4 defaults: 20 / 11
        assert (cfg.cav_raw, cfg.wall_raw) == (20, 11)
        cfg2 = cfg_cls.from_json(cfg.to_json())
        assert (cfg2.cav_raw, cfg2.wall_raw) == (20, 11)
    net = V03ValueNet(V03ValueConfig(d_model=32, n_layers=1, dim_ff=64,
                                     head_hidden=32, head_layers=1,
                                     tau_emb_dim=16))
    b = dict(lane_bins=torch.randn(2, 3, 320, 3),
             lane_mask=torch.ones(2, 3, 320), lane_geom=torch.ones(2, 3, 320),
             cav_tokens=torch.randn(2, 64, 17), cav_mask=torch.ones(2, 64),
             wall_tokens=torch.zeros(2, 4, 6), wall_mask=torch.zeros(2, 4),
             demand=torch.randn(2, 21), context=torch.randn(2, 8),
             summary=torch.randn(2, 40),
             tau=torch.tensor([30.0, 60.0]), policy=torch.zeros(2))
    with pytest.raises(ValueError, match="do not match the model config"):
        net(b)                             # old-width data into a v0.4 net


def test_model_dims_carry_both_token_widths(tiny_dataset, tmp_path):
    """Pre-v0.4 roots (17/6) and v0.4 roots (20/11) both size the net from
    the DATA (model_dims), so a v0.4 rebuild is a data property, never a
    code default change for old roots."""
    from partB_v0.data import V0Dataset, fit_v03_stats, load_manifest
    root, _sha = tiny_dataset
    def log(*a, **k):
        pass
    ds = V0Dataset(load_manifest(root, verify_files=False, log=log),
                   log=log)
    stats = fit_v03_stats(ds, log=log)
    assert stats["feature"]["cav_tokens"]["shape"][-1] == 17
    assert stats["feature"]["wall_tokens"]["shape"][-1] == 6

    # a v0.4-shaped root: same builder, new token widths
    root2 = tmp_path / "v04root"
    make_tiny_dataset(root2, cav_raw=20, wall_raw=11)
    ds2 = V0Dataset(load_manifest(root2, verify_files=False, log=log),
                    log=log)
    stats2 = fit_v03_stats(ds2, log=log)
    assert stats2["feature"]["cav_tokens"]["shape"][-1] == 20
    assert stats2["feature"]["wall_tokens"]["shape"][-1] == 11
    from partB_v0.model import V03ValueConfig
    dims = dict(demand_raw=21, ctx_raw=8,
                cav_raw=stats2["feature"]["cav_tokens"]["shape"][-1],
                wall_raw=stats2["feature"]["wall_tokens"]["shape"][-1],
                summary_dim=40)
    cfg = V03ValueConfig(d_model=32, n_layers=1, dim_ff=64, head_hidden=32,
                         head_layers=1, tau_emb_dim=16, **dims)
    assert (cfg.cav_raw, cfg.wall_raw) == (20, 11)


def test_strict_variant_carries_action_and_wall_blocks_unchanged(
        v031_dataset):
    """Strict-variant field audit extended to the new fields (brief item
    3): action and wall state are planner/RSU-known (the RSU issues the
    commands and owns the wall plan), so deployable_strict keeps the same
    action/wall token blocks as the sensitivity variant -- and they are
    the ONLY blocks the two variants share besides the lane field."""
    from partB_v0.data import V0Dataset, load_manifest
    root, _sha = v031_dataset
    def log(*a, **k):
        pass
    man = load_manifest(root, verify_files=True, log=log)
    ds_s = V0Dataset(man, feature_set="deployable_strict", log=log)
    ds_d = V0Dataset(man, feature_set="deployable_detector", log=log)
    np.testing.assert_array_equal(ds_s.feats["cav_tokens"],
                                  ds_d.feats["cav_tokens"])
    np.testing.assert_array_equal(ds_s.feats["wall_tokens"],
                                  ds_d.feats["wall_tokens"])
    np.testing.assert_array_equal(ds_s.feats["cav_mask"],
                                  ds_d.feats["cav_mask"])
    np.testing.assert_array_equal(ds_s.feats["wall_mask"],
                                  ds_d.feats["wall_mask"])
    # the swapped blocks remain demand/context only
    assert not np.array_equal(ds_s.feats["demand"], ds_d.feats["demand"])
    # no wall tokens in the v0.3 data yet, but the audit holds by shape
    assert ds_s.feats["wall_tokens"].shape[-1] == 6      # pre-v0.4 root
