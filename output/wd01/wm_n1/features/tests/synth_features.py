"""Synthetic run-dir fixtures for the WM-N1 featurizer tests (no SUMO)."""
from __future__ import annotations

import json
from pathlib import Path

MAIN_LEFT = "1191554636_1"
MAIN_RIGHT = "1191554636_0"
ACCEL = "11553048#1_0"
MERGE_LEFT = "1487606179_2"
MERGE_RIGHT = "1487606179_1"
MERGE_ACCEL = "1487606179_0"


def veh(lane_id, pos, v, a=0.0, type_id="normal_car", speed_mode=31):
    return dict(road_id=lane_id.rsplit("_", 1)[0], lane_id=lane_id,
                lane_position_m=float(pos), speed_mps=float(v),
                acceleration_mps2=float(a), type_id=type_id,
                speed_mode=int(speed_mode), lane_changed=False)


def row(t, vehicles, departed=(), arrived=(), cav_ids=()):
    return dict(time_s=float(t), live_ids=sorted(vehicles),
                departed_ids=list(departed), arrived_ids=list(arrived),
                colliding_ids=[], cav_ids=list(cav_ids),
                boundary_crossings=dict(mainline_entry=0, mainline_exit=0,
                                        ramp_entry=0),
                vehicles=dict(vehicles))


def write_run(root: Path, rows, *, controller="damping", schedule=None,
              commands=None, walls=None, decoys=True) -> Path:
    root = Path(root)
    (root / "run").mkdir(parents=True, exist_ok=True)
    with open(root / "run" / "physics_steps.jsonl", "w",
              encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    if controller is not None:
        man = dict(controller=controller, scenario=dict(cav_population=0))
        (root / "run" / "manifest.json").write_text(json.dumps(man),
                                                    encoding="utf-8")
    if schedule is not None:
        (root / "run" / "switch_schedule.json").write_text(
            json.dumps(schedule), encoding="utf-8")
    if commands is not None:
        with open(root / "run" / "suffix_commands.jsonl", "w",
                  encoding="utf-8") as fh:
            for c in commands:
                fh.write(json.dumps(c) + "\n")
    if walls is not None:
        with open(root / "run" / "wall_events.jsonl", "w",
                  encoding="utf-8") as fh:
            for w in walls:
                fh.write(json.dumps(w) + "\n")
    if decoys:
        (root / "run" / "dispatcher_rows.jsonl").write_text(
            '{"kind":"dispatcher_row"}\n', encoding="utf-8")
        (root / "run" / "insertions.jsonl").write_text(
            '{"kind":"insertion"}\n', encoding="utf-8")
    return root


def plan_row(t0_s: float) -> dict:
    return dict(world="wsynth", t0_s=float(t0_s), split="TRAIN",
                base_arm=dict(run="wsynth__base", duration_s=900.0))
