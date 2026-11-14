"""WM-20 radius/penetration package tests (synthetic worlds; no SUMO, no
fitting, no sealed data).  Follows the registered CONTRACT in
output/radius_penetration_v1/REGISTRATION.md; every constant exercised here
is declared in wm20_common.CONSTANTS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "output" / "radius_penetration_v1"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

from wm20_common import (P_IMPUTED, P_OBSERVED,  # noqa: E402
                         P_UNKNOWN, WorldSpec, reporter_seed, reporter_set,
                         resolve_run)
from constructors import (construct_b, construct_c,  # noqa: E402
                          _dedup_remembered, _memory_entries,
                          precompute_ego_detections)
from metrics import score_constructed  # noqa: E402
from overlay import Track  # noqa: E402
from radius_sensor import LaneGeometry, RadiusSensorEmulator  # noqa: E402
from state_grid import lane_intervals_with_groups, valid_road_mask  # noqa: E402


# ------------------------------------------------------------ synthetic world
@pytest.fixture(scope="module")
def synthetic_geometry():
    """Straight synthetic lanes carrying the REAL documented lane ids, so the
    frozen WM-17 lane-group convention and interval table apply."""
    lanes = {
        "1191554636_0": {"length": 100.0, "shape": [[0.0, 0.0], [100.0, 0.0]]},
        "1191554636_1": {"length": 100.0, "shape": [[0.0, 4.0], [100.0, 4.0]]},
        "1487606179_0": {"length": 80.0, "shape": [[100.0, 0.0], [180.0, 0.0]]},
        "11553048#1_0": {"length": 60.0, "shape": [[0.0, -40.0], [60.0, -8.0],
                                                   [96.0, 0.0]]},
    }
    return LaneGeometry(lanes)


@pytest.fixture(scope="module")
def synthetic_intervals(synthetic_geometry):
    lengths = {k: v["length"] for k, v in synthetic_geometry.lanes.items()}
    return lane_intervals_with_groups(synthetic_geometry, lengths)


def _snapshot(vehicles):
    return {"tick_s": 599.0, "vehicles": vehicles,
            "cavs": set(vehicles)}


def _veh(x, y, lane, speed=0.0):
    return {"position_xy": (x, y), "speed_mps": speed, "lane_id": lane}


# ------------------------------------------------------------ radius contract
def test_radius_membership_50_vs_100(synthetic_geometry):
    vehicles = {
        "ego": _veh(0.0, 0.0, "1191554636_0", 10.0),
        "near": _veh(30.0, 0.0, "1191554636_0", 20.0),    # 30 m: both radii
        "mid": _veh(70.0, 0.0, "1191554636_0", 20.0),     # 70 m: 100 m only
        "far": _veh(200.0, 0.0, "1191554636_0", 20.0),    # 200 m: neither
        "cross": _veh(0.0, 4.0, "1191554636_1", 20.0),    # 4 m: cross-lane
    }
    d50 = {d.subject_local_track_id
           for d in RadiusSensorEmulator(synthetic_geometry, radius_m=50.0)
           .observe("ego", vehicles)}
    d100 = {d.subject_local_track_id
            for d in RadiusSensorEmulator(synthetic_geometry, radius_m=100.0)
            .observe("ego", vehicles)}
    assert len(d50) == 2 and len(d100) == 3     # self is never a detection
    assert len(d100 - d50) == 1                 # the 70 m vehicle joins
    # exactly-on-boundary is OUTSIDE (dist < radius), per the frozen contract
    vehicles["boundary"] = _veh(50.0, 0.0, "1191554636_0", 20.0)
    d50b = RadiusSensorEmulator(synthetic_geometry, radius_m=50.0).observe(
        "ego", vehicles)
    assert all(d.euclidean_distance_m < 50.0 for d in d50b)


def test_precompute_is_radius_parameterized(synthetic_geometry):
    snapshots = {599.0: _snapshot({
        "ego": _veh(0.0, 0.0, "1191554636_0", 10.0),
        "near": _veh(30.0, 0.0, "1191554636_0", 20.0),
        "mid": _veh(70.0, 0.0, "1191554636_0", 20.0)})}
    d50 = precompute_ego_detections(snapshots, synthetic_geometry, 50.0)
    d100 = precompute_ego_detections(snapshots, synthetic_geometry, 100.0)
    assert len(d50[599.0]["ego"]) == 1
    assert len(d100[599.0]["ego"]) == 2


# ------------------------------------------------------------------ reporters
def _partition():
    return {"cavs": [f"cav{i:03d}" for i in range(100)],
            "hdvs": [f"hdv{i:03d}" for i in range(300)],
            "requests_is_cav": set()}


def test_reporter_subsampling_deterministic_and_seeded():
    p = _partition()
    a = reporter_set(p, 2, 0, "7003m2000")
    b = reporter_set(p, 2, 0, "7003m2000")
    assert a["reporters"] == b["reporters"]
    assert a["reporters"] == sorted(a["reporters"])
    assert len(a["reporters"]) == 10        # round(2/20 * 100)
    assert set(a["reporters"]) <= set(p["cavs"])
    assert len(reporter_set(p, 10, 0, "7003m2000")["reporters"]) == 50
    assert reporter_seed("7003m2000", 10, 0) != reporter_seed(
        "7003m2000", 10, 1)
    distinct = {frozenset(reporter_set(p, 10, d, "7003m2000")["reporters"])
                for d in range(5)}
    assert len(distinct) >= 2               # draws differ (5 draws of 100C50)
    assert reporter_set(p, 20, 0, "7003m2000")["reporters"] == p["cavs"]


def test_reporter_40pct_augments_with_hdvs_sensing_only():
    p = _partition()
    info = reporter_set(p, 40, 3, "7531m2700")
    assert len(info["cav_reporters"]) == 100
    assert len(info["hdv_reporters"]) == 100   # 2x the 20% reporter count
    assert set(info["hdv_reporters"]) <= set(p["hdvs"])
    assert set(info["reporters"]) == set(p["cavs"]) | set(info["hdv_reporters"])


# ------------------------------------------------------- constructor B and C
def _grid_inputs(synthetic_geometry, synthetic_intervals, tracks,
                 covered_cells=(("cell", 0),), valid=True):
    intervals = synthetic_intervals
    mask = valid_road_mask(intervals)
    coverage = np.zeros_like(mask)
    return mask, coverage


def test_negative_evidence_and_blind_abstain(synthetic_geometry,
                                             synthetic_intervals):
    valid = valid_road_mask(synthetic_intervals)
    # a track 10 m along the pre-merge lane -> d = 90 -> bin 69, group 0
    tracks = [Track(track_key="t1", lane_id="1191554636_0",
                    position_xy=(10.0, 0.0), speed_mps=20.0)]
    coverage = np.zeros_like(valid)
    coverage[69, 0] = True          # the occupied cell is covered
    coverage[68, 0] = True          # covered-empty cell (negative evidence)
    # bin 60 is valid (lane starts at d=100..0 -> bins 60..69) but blind
    grid_b = construct_b(tracks, coverage, valid, synthetic_geometry,
                         synthetic_intervals)
    assert grid_b["count"][69, 0] == 1.0
    assert grid_b["provenance"][69, 0] == P_OBSERVED
    assert grid_b["count"][68, 0] == 0.0 and np.isfinite(grid_b["count"][68, 0])
    assert grid_b["provenance"][68, 0] == P_OBSERVED
    blind = valid & ~coverage
    assert np.all(~np.isfinite(grid_b["count"][blind]))
    assert np.all(grid_b["provenance"][blind] == P_UNKNOWN)


def test_memory_dedup_against_current_detections(synthetic_geometry,
                                                 synthetic_intervals):
    # current detection at d=50 (bin 64); remembered report of the SAME
    # vehicle state from t=599 must dedup away
    entries = [{"group": 0, "d": 51.0, "speed": 20.5, "tick": 599.0},
               {"group": 0, "d": 200.0, "speed": 15.0, "tick": 590.0}]
    placements = [{"group": 0, "d": 50.0, "speed": 20.0, "bin": 64}]
    kept, dropped = _dedup_remembered(entries, placements)
    assert dropped["dedup_current"] == 1
    assert len(kept) == 1 and kept[0]["d"] == 200.0
    # remembered-vs-remembered dedup: most recent wins
    dup = [{"group": 0, "d": 200.0, "speed": 15.0, "tick": 599.0},
           {"group": 0, "d": 201.0, "speed": 15.2, "tick": 580.0}]
    kept2, dropped2 = _dedup_remembered(dup, [])
    assert dropped2["dedup_memory"] == 1 and kept2[0]["tick"] == 599.0


def test_memory_dead_reckoning_along_lane(synthetic_geometry,
                                          synthetic_intervals):
    from radius_sensor import RadiusDetection
    # reporter sees a subject at t=590 (xy x=10, d=90 on the pre-merge lane)
    # moving 20 m/s: at t=600 it is dead-reckoned 200 m downstream along the
    # lane -> clamped at the lane end (d = 0)
    det = RadiusDetection(reporter_id="cav0", subject_local_track_id=1,
                          euclidean_distance_m=10.0, relative_dx_m=10.0,
                          relative_dy_m=0.0, lane_id="1191554636_0",
                          speed_mps=20.0, observed_position_xy=(10.0, 0.0))
    rows = {590.0: [{"reporter_id": "cav0",
                     "reporter_position_xy": (0.0, 0.0),
                     "acquired_at_s": 590.0, "received_at_s": 590.0,
                     "self_state": None, "detections": [det]}]}
    entries, dropped = _memory_entries(rows, synthetic_geometry,
                                       synthetic_intervals)
    assert len(entries) == 1 and dropped == {"projection": 0, "off_grid": 0}
    assert entries[0]["d"] == pytest.approx(0.0, abs=1e-6)
    assert entries[0]["group"] == 0


def test_blind_fill_provenance_and_200m_limit(synthetic_geometry,
                                              synthetic_intervals):
    valid = valid_road_mask(synthetic_intervals)
    # one observed track at x=55 -> pos 55 -> d=45 -> bin 64, group 0;
    # the pre-merge lane spans valid bins 60..69 (100 m), so the other nine
    # pre-merge bins are blind and within 200 m of the source -> imputed.
    tracks = [Track(track_key="t1", lane_id="1191554636_0",
                    position_xy=(55.0, 0.0), speed_mps=18.0)]
    coverage = np.zeros_like(valid)
    coverage[64, 0] = True
    memory_rows = {599.0: []}
    grid_c = construct_c(tracks, coverage, valid, memory_rows,
                         synthetic_geometry, synthetic_intervals)
    observed = [b for b in range(60, 70)
                if grid_c["provenance"][b, 0] == P_OBSERVED]
    imputed = [b for b in range(60, 70)
               if grid_c["provenance"][b, 0] == P_IMPUTED]
    assert observed == [64] and len(imputed) == 9
    for b in imputed:
        assert grid_c["count"][b, 0] == pytest.approx(1.0)
        assert grid_c["speed"][b, 0] == pytest.approx(18.0)
    # ramp group has no populated same-group cell within 200 m -> unknown
    ramp_bins = [b for b in range(300) if valid[b, 3]]
    assert ramp_bins
    assert all(grid_c["provenance"][b, 3] == P_UNKNOWN for b in ramp_bins)


# -------------------------------------------------------------------- metrics
def test_metric_arithmetic():
    count = np.full((300, 4), np.nan)
    speed = np.full((300, 4), np.nan)
    prov = np.full((300, 4), P_UNKNOWN, dtype=np.int64)
    valid = np.zeros((300, 4), dtype=bool)
    valid[10:14, 0] = True
    truth = np.zeros((300, 4))
    truth_sp = np.full((300, 4), np.nan)
    truth[10, 0], truth[11, 0], truth[12, 0] = 1, 2, 0
    truth_sp[10, 0], truth_sp[11, 0] = 10.0, 20.0
    truth_vehicles = [{"bin": 10, "group": 0}, {"bin": 11, "group": 0},
                      {"bin": 12, "group": 0}]
    # declared: bin10 pred 1 (observed), bin11 pred 1 speed 15 (observed),
    # bin12 pred 0 (covered empty).  bin13 blind -> abstain.
    count[10, 0], speed[10, 0], prov[10, 0] = 1.0, 10.0, P_OBSERVED
    count[11, 0], speed[11, 0], prov[11, 0] = 1.0, 15.0, P_OBSERVED
    count[12, 0], prov[12, 0] = 0.0, P_OBSERVED
    grid = {"count": count, "speed": speed, "provenance": prov,
            "valid": valid, "coverage": prov != P_UNKNOWN}
    s = score_constructed(grid, truth, truth_sp, truth_vehicles)
    assert s["count_mae_occupied_10m"] == pytest.approx(
        (0.0 + 1.0) / 2)          # cells 10,11 only (cell12 truth-empty)
    assert s["speed_rmse_occupied_10m"] == pytest.approx(
        (25.0 / 2) ** 0.5)        # |15-20|=5 on cell11, 0 on cell10
    assert s["signed_total_bias_declared"] == pytest.approx(-1.0)
    assert s["abs_total_bias_declared"] == pytest.approx(1.0)
    assert s["negative_evidence_accuracy"] == 1.0     # cell12 truly empty
    assert s["abstention_share"] == pytest.approx(1 / 4)
    assert s["coverage_cells_frac"] == pytest.approx(3 / 4)
    assert s["vehicles_detected_frac"] == pytest.approx(2 / 3)
    # cells 10 and 11 carry observed tracks (3 truth vehicles); cell12's
    # vehicle sits on a covered-empty cell and is not detected
    assert s["observed_mae"] == pytest.approx(0.5)
    assert s["remembered_mae"] is None and s["imputed_mae"] is None


# ------------------------------------------------------- run availability
def _spec():
    return WorldSpec(world="7003m2000", run_tag="cv20_7003m2000-retain",
                     campaign="cv20", job="7003m2000-retain",
                     f_npz=Path("f.npz"), wm19_input_npz=Path("i.npz"),
                     requests_path=Path("r.json"),
                     v5_physics_path="runs/active/x")


def _make_verified(root, ds_root, tag="cv20_7003m2000-retain", *,
                   receipt_status="complete", match=True,
                   with_receipt=True, with_target=True):
    job_dir = root / "cv20" / "7003m2000-retain"
    if with_target:
        (job_dir / "run").mkdir(parents=True, exist_ok=True)
    if with_receipt:
        (job_dir / "REGENERATION_RECEIPT.json").write_text(
            json.dumps({"exit_status": receipt_status}), encoding="utf-8")
    split = ds_root / "train"
    split.mkdir(parents=True, exist_ok=True)
    (split / f"{tag}.comparison.json").write_text(
        json.dumps({"match": match}), encoding="utf-8")
    (split / f"{tag}.jsonl").write_text("{}\n", encoding="utf-8")


def test_partial_target_without_receipt_is_never_used(tmp_path):
    primary = tmp_path / "regenerated_v1"
    ds = tmp_path / "dataset_v2"
    (primary / "cv20" / "7003m2000-retain" / "run").mkdir(parents=True)
    out = resolve_run(_spec(), roots=[(primary, ds)])
    assert out["available"] is False
    assert "no receipt" in out["reason"]


def test_unverified_statuses_are_refused(tmp_path):
    primary = tmp_path / "regenerated_v1"
    ds = tmp_path / "dataset_v2"
    _make_verified(primary, ds, receipt_status="failed")
    assert resolve_run(_spec(), roots=[(primary, ds)])["available"] is False
    _make_verified(primary, ds, receipt_status="complete", match=False)
    assert resolve_run(_spec(), roots=[(primary, ds)])["available"] is False
    (ds / "train" / "cv20_7003m2000-retain.comparison.json").unlink()
    assert resolve_run(_spec(), roots=[(primary, ds)])["available"] is False


def test_retry_root_resolution(tmp_path):
    primary = tmp_path / "regenerated_v1"
    ds = tmp_path / "dataset_v2"
    (primary / "cv20" / "7003m2000-retain" / "run").mkdir(parents=True)
    retry = tmp_path / "regenerated_v1_retry1"
    ds_retry = tmp_path / "dataset_v2_retry1"
    _make_verified(retry, ds_retry)
    out = resolve_run(_spec(), roots=[(primary, ds), (retry, ds_retry)])
    assert out["available"] is True
    assert Path(out["run_root"]).name == "regenerated_v1_retry1"
    assert Path(out["dataset_file"]).parent.parent.name == "dataset_v2_retry1"


def test_verified_primary_wins_over_retry(tmp_path):
    primary = tmp_path / "regenerated_v1"
    ds = tmp_path / "dataset_v2"
    _make_verified(primary, ds)
    retry = tmp_path / "regenerated_v1_retry1"
    ds_retry = tmp_path / "dataset_v2_retry1"
    _make_verified(retry, ds_retry)
    out = resolve_run(_spec(), roots=[(primary, ds), (retry, ds_retry)])
    assert Path(out["run_root"]).name == "regenerated_v1"
