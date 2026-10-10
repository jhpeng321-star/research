"""Tests for the offline WM-S2 VAL/TEST driver-recovery scorer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "wd01" / "wm_n1"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from wms.driver_sidecar import PARAM_ORDER  # noqa: E402
from wms_eval import driver as D  # noqa: E402


def _write_pack(root: Path, *, n_ticks=8, k=4, tick=5):
    kin = np.zeros((n_ticks, k, 7), np.float16)
    kin[:, 0, 0] = 20.0
    kin[:, 0, 1] = 1.0
    kin[:, 0, 4] = 20.0
    kin[:, 0, 6] = 5.0
    kin[:, 1, 0] = 15.0
    kin[:, 1, 1] = 1.0
    kin[:, 1, 4] = 18.0
    kin[:, 1, 6] = 5.0
    mask = np.zeros((n_ticks, k), bool)
    mask[:, 0] = True
    mask[:, 1] = True
    target = np.zeros((n_ticks, k), bool)
    np.save(root / "veh_kin.npy", kin)
    np.save(root / "veh_mask.npy", mask)
    np.save(root / "target_mask.npy", target)
    manifest = {
        "schema": "wm_n1_pack/1", "k_max": k,
        "arrays": {name: {"file": f"{name}.npy", "dtype": "float16",
                          "shape": [n_ticks, k, 7] if name == "veh_kin"
                          else [n_ticks, k]}
                   for name in ("veh_kin", "veh_mask", "target_mask")},
        "items": [{"world": "w1", "arm": "native", "t0_s": 0.0, "tick": tick,
                   "split": "VAL", "policy": "native", "pair_key": "w1@t0=0"}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    index = {"schema": "wm_n1_corpus_pack/2",
             "runs": [{"key": "hetdata/w1__native", "tick_start": 0,
                       "tick_end": n_ticks, "world": "w1", "arm": "native"}]}
    (root / "INDEX.json").write_text(json.dumps(index), encoding="utf-8")
    return root


def test_materialise_reads_snapshot_and_history(tmp_path):
    pack = _write_pack(tmp_path)
    summary = D.materialise_pack_states(pack, tmp_path / "states.json",
                                        history_steps=10)
    assert summary["n_states"] == 1
    payload = json.loads((tmp_path / "states.json").read_text(encoding="utf-8"))
    rec = payload["states"][0]
    assert rec["world"] == "w1" and rec["arm"] == "native"
    assert rec["tick"] == 5 and rec["item_position"] == 0
    assert rec["tick_start"] == 0        # run-local tick = tick - tick_start
    assert len(rec["veh_kin"]) == 4
    assert len(rec["history"]["kin"]) == 6          # ticks 0..5
    assert rec["history"]["mask"][0][0] is True


def test_history_features_shapes(tmp_path):
    D.materialise_pack_states(_write_pack(tmp_path), tmp_path / "s.json",
                              history_steps=10)
    rec = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))["states"][0]
    feat, mask = D.history_features_from_kin(rec["history"]["kin"],
                                             rec["history"]["mask"])
    assert feat.shape == (4, 6, 6)
    assert mask.shape == (4, 6)
    assert np.isfinite(feat).all()


def test_score_predictions_against_bundle(tmp_path):
    from wms.driver_sidecar import DriverBundle
    pack = _write_pack(tmp_path)
    D.materialise_pack_states(pack, tmp_path / "states.json", history_steps=10)
    records = D._load_records(tmp_path / "states.json")

    params = np.full((1, 4, 11), np.nan, np.float32)
    params[0, 0] = [1.0 + i for i in range(11)]
    params[0, 1] = [2.0 + i for i in range(11)]
    classes = np.full((1, 4), -1, dtype=np.int64)
    classes[0, 0], classes[0, 1] = 0, 1
    bundle = DriverBundle(params=params, classes=classes,
                          global_positions=np.array([0]))

    values = np.zeros((4, 9))
    uncertainty = np.ones((4, 9))
    from wms.driver_encoder import PARAMETER_NAMES
    a_enc = PARAMETER_NAMES.index("accel")
    a_truth = PARAM_ORDER.index("accel_mps2")
    values[0, a_enc] = params[0, 0, a_truth]
    values[1, a_enc] = params[0, 1, a_truth]
    predictions = [(values, uncertainty)]
    report = D.score_predictions(records, predictions,
                                 D.bundle_truth_lookup(bundle),
                                 class_names=("aggressive_car", "normal_car",
                                              "cautious_car"))
    assert report["n_scored"] == 2
    assert report["per_parameter"]["accel"]["mae"] == 0.0
    assert report["per_class"]["aggressive_car"]["n"] == 1
    assert report["per_class"]["normal_car"]["n"] == 1


def test_predict_records_with_driver_params_checkpoint(tmp_path):
    from wms.model import PhysicsConfig, WmSModel
    from wms_eval.wms_plugin import WmsCheckpointModel

    pack = _write_pack(tmp_path)
    D.materialise_pack_states(pack, tmp_path / "states.json", history_steps=10)
    records = D._load_records(tmp_path / "states.json")

    torch.manual_seed(0)
    model = WmSModel(physics=PhysicsConfig(), history_features=6,
                     rollout_mode="driver_params")
    out = ROOT / "_out" / "wms_driver_test"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "wm_s_driver.pt"
    torch.save({"format": "wm_s/1", "model": model.state_dict(),
                "config": {"history_features": 6,
                           "rollout_mode": "driver_params"}}, ckpt)
    holder = WmsCheckpointModel(ckpt)
    loaded = holder.model or holder._load()
    predictions = D.predict_records(loaded, records)
    values, uncertainty = predictions[0]
    assert values.shape == (4, 9)
    assert uncertainty.shape == (4, 9)
    assert np.isfinite(values).all()
