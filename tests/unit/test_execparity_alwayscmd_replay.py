"""Offline replay (iii) row mapping on a synthetic tape+log fixture; CPU only."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "src"),
                str(Path(__file__).resolve().parents[2])]
from tools.execparity.alwayscmd_replay import world_rows


def _driver(tmp_path):
    payload = {"schema": "execparity_learned_idm/1", "fit_controller": "direct",
               "evaluation_data_used": False, "fit_worlds": ["a"], "score_worlds": ["b"],
               "parameters": {"accel": 1.1, "decel": 2., "time_headway": 1.2,
                              "min_gap": 2., "v0": 28.}}
    path = tmp_path / "driver.json"
    path.write_text(json.dumps(payload))
    return path


def _job(tmp_path, speeds, rows):
    n = len(speeds)
    z = {
        "time_s": np.array([300. + .5 * i for i in range(n)]),
        "s_m": np.array([[90. + 10. * i, 150. + 10. * i] for i in range(n)]),
        "v_mps": np.array([[speeds[i], 20.] for i in range(n)]),
        "a_mps2": np.zeros((n, 2)),
        "present": np.ones((n, 2), dtype=bool),
        "is_cav": np.array([[True, False]] * n),
        "lane": np.zeros((n, 2), dtype=np.int8),
        "junction": np.zeros((n, 2), dtype=bool),
        "vid_index": np.array([[0, 1]] * n, dtype=np.int32),
        "vehicle_ids": np.array(["cav1", "ldr1"]),
    }
    job = tmp_path / "s977001_alwaysx"
    (job / "deploy/episode").mkdir(parents=True)
    np.savez(job / "deploy/episode/episode.npz", **z)
    with gzip.open(job / "execution_log.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return job


def _row(t, vid, speed, residual=.2, base=.1, issued=.3):
    return {"t_s": t, "vehicle_id": vid, "residual_mps2": residual,
            "base_law_mps2": base, "issued_mps2": issued,
            "speed_before_mps": speed, "leader": {"gap_m": 12., "speed_mps": 20.}}


def test_replay_maps_rows_to_tape_columns_and_realised(tmp_path, monkeypatch):
    monkeypatch.setenv("C4_EXECUTION_DRIVER", str(_driver(tmp_path)))
    speeds = [20., 20.3, 20.55]
    rows = [_row(300., "cav1", 20.), _row(300., "ghost", 20.),
            _row(300., "cav1", 20.4), _row(301., "cav1", 20.)]
    job = _job(tmp_path, speeds, rows)
    out, tape_sha = world_rows(job)
    assert len(out) == 1
    row = out[0]
    assert row["t_s"] == 300. and row["speed"] == 20.
    assert row["realised"] == pytest.approx(0.3 / .5)
    assert row["residual"] == .2 and row["issued"] == .3 and row["sumo_base"] == .1
    assert tape_sha
    monkeypatch.delenv("C4_EXECUTION_DRIVER")


def test_replay_law_value_matches_scalar_executed(tmp_path, monkeypatch):
    import os
    from wms.control.alwayscmd import scalar_executed
    from wms.control.execution import load_learned_driver
    monkeypatch.setenv("C4_EXECUTION_DRIVER", str(_driver(tmp_path)))
    params = load_learned_driver(os.environ["C4_EXECUTION_DRIVER"])
    speeds = [20., 20.3, 20.55]
    job = _job(tmp_path, speeds, [_row(300., "cav1", 20., residual=.1, base=0., issued=.1)])
    out, _ = world_rows(job)
    row = out[0]
    gap = 150. - 90. - 5.  # leader ahead by 60 m, minus leader length 5 m
    expected = scalar_executed(20., 20., gap, .1, params)
    assert row["imagined"] == pytest.approx(expected, abs=1e-6)
    monkeypatch.delenv("C4_EXECUTION_DRIVER")
