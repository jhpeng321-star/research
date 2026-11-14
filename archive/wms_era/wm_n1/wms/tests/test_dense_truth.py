"""Unit tests for the dense per-run driver-truth builder (WM-S2 C2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_WM_N1 = Path(__file__).resolve().parents[2]        # output/wd01/wm_n1
if str(_WM_N1) not in sys.path:
    sys.path.insert(0, str(_WM_N1))

from output.wd01.wm_n1.wms import dense_truth as DT  # noqa: E402
from output.wd01.wm_n1.wms.driver_sidecar import (  # noqa: E402
    DRIVER_CLASSES, PARAM_ORDER)


def _row(vid: str, cls: str, base: float) -> dict:
    return {"dataset_seed": 7, "vehicle_id": vid, "driver_class": cls,
            "stream": "mainline", "is_cav": False, "vtype_id": f"hdv.{vid}",
            **{name: base + i for i, name in enumerate(PARAM_ORDER)}}


def _write_truth(path: Path, rows) -> None:
    header = ["dataset_seed", "vehicle_id", "driver_class", "stream", "is_cav",
              "vtype_id", *PARAM_ORDER]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[col]) for col in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _physics_line(t: float, vehicles) -> str:
    return json.dumps({"time_s": t, "departed_ids": [], "cav_ids": [],
                       "vehicles": {vid: {"lane_position_m": 10.0,
                                          "speed_mps": 20.0} for vid in vehicles}})


def _make_run(root: Path, world="w8951m2600p05", arm="damping", *,
              complete=True, split="TRAIN"):
    run_root = root / f"{split}_{arm}_{world}"
    run_dir = run_root / "attempt-0001" / "run"
    (run_dir / "scenario").mkdir(parents=True, exist_ok=True)
    lines = [_physics_line(0.5, []),
             _physics_line(1.0, ["a.0"]),
             _physics_line(1.5, ["a.0", "a.1"]),
             _physics_line(2.0, ["a.1", "a.2"]),
             _physics_line(2.5, ["a.1"]),
             _physics_line(4.0, ["a.1"])]
    (run_dir / "physics_steps.jsonl").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
    _write_truth(run_dir / "scenario" / "hetdata_truth.csv",
                 [_row("a.0", "normal_car", 1.0),
                  _row("a.1", "aggressive_car", 2.0),
                  _row("a.2", "cautious_car", 3.0)])
    receipt = run_dir.parent / "receipt.json"
    receipt.write_text(json.dumps({
        "status": "complete" if complete else "failed",
        "run": {"run_id": f"{split}_{arm}_{world}", "split": split,
                "controller": arm, "world": world}}), encoding="utf-8")
    return run_root


def test_assign_slots_persistent_pool_reuse(tmp_path):
    root = _make_run(tmp_path)
    run_dir = root / "attempt-0001" / "run"
    t0, n_ticks, records, vids, dropped = DT.assign_slots(
        run_dir / "physics_steps.jsonl", k_max=8)
    assert t0 == 0.0
    assert n_ticks == 9
    assert dropped == 0
    assert vids == ["a.0", "a.1", "a.2"]
    by_tick: dict = {}
    for k, slot, vi in records:
        by_tick.setdefault(k, {})[slot] = vi
    assert by_tick[2] == {0: 0}
    assert by_tick[3] == {0: 0, 1: 1}
    # a.0 leaves and a.2 reuses slot 0 in the SAME tick (lowest free index first)
    assert by_tick[4] == {0: 2, 1: 1}
    assert by_tick[5] == {1: 1}


def test_assign_slots_matches_featurizer_parse_run(tmp_path):
    root = _make_run(tmp_path)
    run_dir = root / "attempt-0001" / "run"
    _, _, _, vids, _ = DT.assign_slots(run_dir / "physics_steps.jsonl", k_max=8)
    built = DT.build_run(run_dir, tmp_path / "dense", k_max=8)
    slot_vehicle = np.load(tmp_path / "dense" / "slot_vehicle.npy")
    from features.featurizer import parse_run
    parsed = parse_run(run_dir, 0.0, k_max=8,
                       physics_path=run_dir / "physics_steps.jsonl")
    assert list(parsed["slot_vid"]) == vids
    np.testing.assert_array_equal(slot_vehicle, np.asarray(parsed["slot_index"]))
    assert built["n_ticks"] == int(parsed["n_ticks"])


def test_build_run_and_lookup_roundtrip(tmp_path):
    _make_run(tmp_path)
    DT.build_dense_truth(tmp_path, tmp_path / "out", k_max=8, workers=1)
    sidecar = DT.DenseDriverTruth(tmp_path / "out")
    vid, params, cls = sidecar.lookup("w8951m2600p05", "damping", tick=4, slot=0)
    assert vid == "a.2"
    assert cls == DRIVER_CLASSES.index("cautious_car")
    np.testing.assert_allclose(params, [3.0 + i for i in range(11)])
    # an empty slot at an earlier tick
    assert sidecar.lookup("w8951m2600p05", "damping", tick=1, slot=0) is None


def test_build_keeps_vehicle_without_truth_as_nan(tmp_path):
    root = _make_run(tmp_path)
    run_dir = root / "attempt-0001" / "run"
    # drop a.2 from the truth table
    _write_truth(run_dir / "scenario" / "hetdata_truth.csv",
                 [_row("a.0", "normal_car", 1.0),
                  _row("a.1", "aggressive_car", 2.0)])
    DT.build_dense_truth(tmp_path, tmp_path / "out", k_max=8, workers=1)
    sidecar = DT.DenseDriverTruth(tmp_path / "out")
    vid, params, cls = sidecar.lookup("w8951m2600p05", "damping", tick=4, slot=0)
    assert vid == "a.2"
    assert cls == -1
    assert np.all(np.isnan(params))


def test_discover_runs_filters_incomplete_and_split(tmp_path):
    _make_run(tmp_path, world="w8951m2600p05", arm="damping", split="TRAIN")
    _make_run(tmp_path, world="w9000m2700p02", arm="damping", split="TRAIN",
              complete=False)
    _make_run(tmp_path, world="w8861m2600p05", arm="damping", split="VAL")
    everything = DT.discover_runs(tmp_path)
    assert {r["world"] for r in everything} == {"w8951m2600p05", "w8861m2600p05"}
    train_only = DT.discover_runs(tmp_path, splits=["TRAIN"])
    assert [r["world"] for r in train_only] == ["w8951m2600p05"]
    assert train_only[0]["run_key"] == "w8951m2600p05__damping"


def test_build_dense_truth_parallel_and_verify(tmp_path):
    _make_run(tmp_path, world="w8951m2600p05", arm="damping", split="TRAIN")
    _make_run(tmp_path, world="w8954m2400p10", arm="native", split="TRAIN")
    receipt = DT.build_dense_truth(tmp_path, tmp_path / "out", k_max=8,
                                   workers=2)
    assert receipt["n_runs"] == 2
    assert receipt["n_vehicles_with_truth"] == 6
    assert (tmp_path / "out" / "runs.jsonl").is_file()
    sidecar = DT.DenseDriverTruth(tmp_path / "out")
    assert ("w8954m2400p10", "native") in sidecar.run_keys
    # tamper with one array: verify must fail
    row = sidecar.run("w8954m2400p10", "native")
    target = tmp_path / "out" / row["dir"] / "vehicle_params.npy"
    target.write_bytes(target.read_bytes() + b"\x00")
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = DT.main(["verify", "--out", str(tmp_path / "out")])
    assert rc == 1
    assert "vehicle_params.npy" in buf.getvalue()
