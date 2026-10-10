from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1].parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from wms_eval.indexed_switch import (  # noqa: E402
    IndexedSwitchError, bind_indexed_switch_timeline,
)


def _fixture(root: Path):
    (root / "dense" / "world__switching").mkdir(parents=True)
    schedule = {
        "schema": "wms_switch_schedule/1", "control_steps": 2,
        "control_dt_s": 1.0, "t0_s": 100.0,
        "vehicle_ids": ["cav-a", "cav-b"],
        "initial": [{"vehicle_id": "cav-a", "controller": "damping"},
                    {"vehicle_id": "cav-b", "controller": "native"}],
        "switches": [{"timestamp_s": 101, "vehicle_id": "cav-b",
                      "controller": "direct"}],
    }
    (root / "switch_schedule.json").write_text(json.dumps(schedule), encoding="utf-8")
    (root / "suffix_commands.jsonl").write_text(json.dumps({
        "timestamp_s": 101, "vehicle_id": "cav-b",
        "issued_acceleration_mps2": -0.5,
        "issued_target_speed_mps": 19.0,
    }) + "\n", encoding="utf-8")
    np.save(root / "identities.npy", np.asarray([[0, 1, -1]], dtype=np.int64),
            allow_pickle=False)
    (root / "vehicle_map.json").write_text(json.dumps([
        ["chunk1", "world", "cav-a"], ["chunk1", "world", "cav-b"],
    ]), encoding="utf-8")
    dense_root = root / "dense"
    dense_rows = np.full((5, 4), -1, dtype=np.int32)
    # Dense slots deliberately move so the join cannot rely on column order.
    dense_rows[2] = [1, -1, 0, -1]
    dense_rows[4] = [0, -1, 1, -1]
    np.save(dense_root / "world__switching" / "slot_vehicle.npy", dense_rows,
            allow_pickle=False)
    (dense_root / "world__switching" / "vehicle_ids.json").write_text(
        json.dumps(["cav-a", "cav-b"]), encoding="utf-8")
    (dense_root / "runs.jsonl").write_text(json.dumps({
        "world": "world", "arm": "switching", "dir": "world__switching",
    }) + "\n", encoding="utf-8")
    (dense_root / "DENSE_TRUTH_RECEIPT.json").write_text(
        json.dumps({"runs_file": "runs.jsonl"}), encoding="utf-8")
    return root, dense_root


def test_indexed_switch_join_uses_t0_identity_then_tick_local_dense_slots(tmp_path):
    root, dense_root = _fixture(tmp_path)
    timeline = bind_indexed_switch_timeline(
        schedule_path=root / "switch_schedule.json",
        suffix_path=root / "suffix_commands.jsonl",
        identities_path=root / "identities.npy",
        vehicle_map_path=root / "vehicle_map.json",
        dense_truth_dir=dense_root, world="world", arm="switching",
        identity_row=0, dense_t0_tick=2,
    )
    assert timeline.controller[0, 0] == 1  # cav-a's stable t0 pack column
    assert timeline.controller[0, 1] == 0  # cav-b's stable t0 pack column
    assert timeline.controller[1, 0] == 1  # dense slots moved, pack column did not
    assert timeline.controller[1, 1] == 3
    assert timeline.acceleration_mps2[1, 1] == pytest.approx(-0.5)
    assert timeline.target_speed_mps[1, 1] == pytest.approx(19.0)


def test_indexed_switch_join_fails_closed_for_unmatched_t0_identity(tmp_path):
    root, dense_root = _fixture(tmp_path)
    (root / "vehicle_map.json").write_text(json.dumps([
        ["chunk1", "world", "cav-a"],
    ]), encoding="utf-8")
    with pytest.raises(IndexedSwitchError, match="cav-b"):
        bind_indexed_switch_timeline(
            schedule_path=root / "switch_schedule.json",
            suffix_path=root / "suffix_commands.jsonl",
            identities_path=root / "identities.npy",
            vehicle_map_path=root / "vehicle_map.json",
            dense_truth_dir=dense_root, world="world", arm="switching",
            identity_row=0, dense_t0_tick=2,
        )
