from __future__ import annotations

import json

import numpy as np

from wms.data.driver_truth import PARAMETER_NAMES
from wms.data.driver_truth import (DRIVER_CLASSES, PARAM_ORDER,
                                                  DriverSidecar, align_to_slots,
                                                  driver_recovery,
                                                  load_driver_bundle,
                                                  load_driver_sidecar)

ROW = {name: float(i + 1) for i, name in enumerate(PARAM_ORDER)}


def _write_truth_csv(path, rows):
    header = ["dataset_seed", "vehicle_id", "driver_class", "stream", "is_cav",
              "vtype_id", *PARAM_ORDER]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[col]) for col in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_optional_path_absent_is_none():
    assert load_driver_sidecar(None) is None
    assert load_driver_sidecar("") is None


def test_load_csv_hetdata_truth(tmp_path):
    rows = [{"dataset_seed": 7, "vehicle_id": "a.0", "driver_class": "normal_car",
             "stream": "a", "is_cav": False, "vtype_id": "hdv.a.0", **ROW},
            {"dataset_seed": 7, "vehicle_id": "a.1", "driver_class": "aggressive_car",
             "stream": "a", "is_cav": True, "vtype_id": "hdv.a.1",
             **{k: v * 2 for k, v in ROW.items()}}]
    path = tmp_path / "hetdata_truth.csv"
    _write_truth_csv(path, rows)
    sidecar = load_driver_sidecar(path)
    assert sidecar.n == 2
    assert sidecar.vehicle_ids == ("a.0", "a.1")
    assert sidecar.classes.tolist() == [DRIVER_CLASSES.index("normal_car"),
                                        DRIVER_CLASSES.index("aggressive_car")]
    np.testing.assert_allclose(sidecar.params[0], [ROW[n] for n in PARAM_ORDER])


def test_load_npz_roundtrip_and_align(tmp_path):
    params = np.arange(11, dtype=np.float64)[None, :]
    classes = np.array([0])
    path = tmp_path / "sidecar.npz"
    np.savez(path, params=params, **{"class": classes}, vehicle_ids=np.array(["v0"]))
    sidecar = load_driver_sidecar(path)
    aligned, cls, present = align_to_slots(sidecar, ["v0", "missing", "v0"])
    assert present.tolist() == [True, False, True]
    assert cls.tolist() == [0, -1, 0]
    np.testing.assert_allclose(aligned[0], params[0])
    assert np.all(np.isnan(aligned[1]))


def test_recovery_perfect_prediction_has_zero_error_and_full_coverage():
    truth = np.array([[ROW[n] for n in PARAM_ORDER]])
    values = np.zeros((1, len(PARAMETER_NAMES)))
    for name in PARAMETER_NAMES:
        if name == "time_headway":
            values[0, PARAMETER_NAMES.index(name)] = ROW["tau_s"]
        elif name == "min_gap":
            values[0, PARAMETER_NAMES.index(name)] = ROW["min_gap_m"]
        elif name == "accel":
            values[0, PARAMETER_NAMES.index(name)] = ROW["accel_mps2"]
        elif name == "decel":
            values[0, PARAMETER_NAMES.index(name)] = ROW["decel_mps2"]
    uncertainty = np.ones_like(values)
    report = driver_recovery(values, uncertainty, truth, present=np.array([True]))
    assert report["n_scored"] == 1
    for name in ("time_headway", "min_gap", "accel", "decel"):
        entry = report["per_parameter"][name]
        assert entry["mae"] == 0.0
        assert entry["coverage_1sigma"] == 1.0
        assert entry["coverage_3sigma"] == 1.0


def test_recovery_calibration_ratio_flags_overconfidence():
    truth = np.array([[ROW[n] for n in PARAM_ORDER], [ROW[n] for n in PARAM_ORDER]])
    values = np.zeros((2, len(PARAMETER_NAMES)))
    values[0, PARAMETER_NAMES.index("accel")] = ROW["accel_mps2"] + 4.0
    values[1, PARAMETER_NAMES.index("accel")] = ROW["accel_mps2"] + 2.0
    uncertainty = np.full_like(values, 0.1)  # tiny sigma, large error
    report = driver_recovery(values, uncertainty, truth, present=np.array([True, True]))
    entry = report["per_parameter"]["accel"]
    assert entry["bias"] > 0.0
    assert entry["calibration_ratio"] < 1.0  # sigma underestimates the error
    assert entry["coverage_1sigma"] == 0.0


def test_recovery_class_nearest_centre():
    centres = {cls: {"tau_s": 1.0 + i, "min_gap_m": 2.0 + i, "accel_mps2": 3.0,
                     "decel_mps2": 3.5}
               for i, cls in enumerate(DRIVER_CLASSES)}
    truth = np.array([[ROW[n] for n in PARAM_ORDER]])
    values = np.zeros((1, len(PARAMETER_NAMES)))
    # aggressive centre exactly
    values[0, PARAMETER_NAMES.index("time_headway")] = centres["aggressive_car"]["tau_s"]
    values[0, PARAMETER_NAMES.index("min_gap")] = centres["aggressive_car"]["min_gap_m"]
    values[0, PARAMETER_NAMES.index("accel")] = centres["aggressive_car"]["accel_mps2"]
    values[0, PARAMETER_NAMES.index("decel")] = centres["aggressive_car"]["decel_mps2"]
    report = driver_recovery(values, None, truth, np.array([0]), present=np.array([True]),
                             class_centres=centres)
    assert report["class"]["n"] == 1
    assert report["class"]["accuracy"] == 1.0


def test_driver_sidecar_dataclass_rejects_bad_shape():
    try:
        DriverSidecar(params=np.zeros((2, 3)), classes=np.zeros(2, dtype=np.int64))
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong parameter width")


def test_json_rows_load(tmp_path):
    rows = [{"vehicle_id": "x", "driver_class": "cautious_car", **ROW}]
    path = tmp_path / "sidecar.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    sidecar = load_driver_sidecar(path)
    assert sidecar.n == 1
    assert sidecar.classes.tolist() == [DRIVER_CLASSES.index("cautious_car")]


def _bundle_dir(tmp_path, n_items=2, k_max=4):
    params = np.arange(n_items * k_max * 11, dtype=np.float64).reshape(
        n_items, k_max, 11)
    classes = np.full((n_items, k_max), -1, dtype=np.int64)
    classes[:, 0] = [0, 1]
    bundle = tmp_path / "driver_params_hetdata_train"
    bundle.mkdir()
    np.save(bundle / "driver_params.npy", params)
    np.save(bundle / "driver_class.npy", classes)
    np.save(bundle / "global_positions.npy", np.array([3, 5], dtype=np.int64))
    (bundle / "vehicle_params.json").write_text(json.dumps(
        {"hetdata/w1/mainline.0": {}}), encoding="utf-8")
    return bundle, params, classes


def test_load_driver_bundle_dir_and_item_sidecar(tmp_path):
    bundle_dir, params, classes = _bundle_dir(tmp_path)
    bundle = load_driver_bundle(bundle_dir)
    assert bundle.n_items == 2 and bundle.k_max == 4
    assert bundle.item_index_of(5) == 1
    item = bundle.item_sidecar(0)
    assert isinstance(item, DriverSidecar)
    assert item.vehicle_ids is None
    np.testing.assert_allclose(item.params, params[0])
    assert item.classes.tolist() == classes[0].tolist()


def test_load_driver_sidecar_accepts_bundle_dir_and_3d_npy(tmp_path):
    from wms.data.driver_truth import DriverBundle
    bundle_dir, params, _ = _bundle_dir(tmp_path)
    assert isinstance(load_driver_sidecar(bundle_dir), DriverBundle)
    flat = tmp_path / "bundle.npy"
    np.save(flat, params)
    loaded = load_driver_sidecar(flat)
    assert isinstance(loaded, DriverBundle) and loaded.n_items == 2


def test_driver_recovery_rank_correlation_and_per_class():
    from wms.data.driver_truth import driver_recovery
    truth = np.array([[ROW[n] for n in PARAM_ORDER] for _ in range(4)])
    truth[:, PARAM_ORDER.index("accel_mps2")] = [1.0, 2.0, 3.0, 4.0]
    values = np.zeros((4, len(PARAMETER_NAMES)))
    values[:, PARAMETER_NAMES.index("accel")] = [1.0, 2.0, 3.0, 4.0]
    classes = np.array([0, 0, 1, 1])
    report = driver_recovery(values, None, truth, classes,
                             present=np.array([True] * 4))
    accel = report["per_parameter"]["accel"]
    assert accel["rank_correlation"] == 1.0
    assert report["per_class"]["aggressive_car"]["n"] == 2
    assert report["per_class"]["normal_car"]["accel"] == 0.0
    assert report["per_class"]["cautious_car"]["n"] == 0
