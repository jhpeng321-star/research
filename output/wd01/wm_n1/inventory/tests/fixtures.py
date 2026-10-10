"""Synthetic WM-N1 inventory fixture trees (no E:/research-dense access)."""
from __future__ import annotations

import json
from pathlib import Path

EXCLUDED_WORLD = "wEXm2000p02t100"
TRAIN_WORLD = "w1m2600p02t100"
T0 = 100.0

SPEC_A = {"name": "chunk1", "chunk": "1", "plan_rel": "plans/chunk1.json",
          "root_rel": "runs/chunk1"}
SPEC_B = {"name": "truearms_chunk2", "chunk": "2",
          "plan_rel": "plans/truearms.json", "root_rel": "runs/truearms"}
SPEC_C = {"name": "walls_pilot_chunk4", "chunk": "4",
          "plan_rel": "plans/walls.json", "root_rel": "runs/walls"}
SPEC_EX = {"name": "chunk1_ex", "chunk": "1", "plan_rel": "plans/ex.json",
           "root_rel": "runs/chunk1_ex"}
ALL_SPECS = (SPEC_A, SPEC_B, SPEC_C, SPEC_EX)


def _run_dir(root, root_rel, run_name):
    return Path(root) / root_rel / run_name


def write_run(root, root_rel, run_name, *, schedule_arm=None, events=3,
              receipt_status="complete", physics_lines=0, receipt_schema="wm44_branch_receipt/1",
              saved_state_id="sst-default", rows_written=None, policy_field=None,
              base_controller=None):
    rd = _run_dir(root, root_rel, run_name)
    (rd / "run").mkdir(parents=True, exist_ok=True)
    if schedule_arm is not None:
        sched = {
            "schema": "wm44_branch_types/1",
            "arm": schedule_arm,
            "events": [{"vehicle_id": f"mainline.{i}", "time_s": 0.0,
                        "action": "native"} for i in range(events)],
        }
        (rd / "run" / "switch_schedule.json").write_text(
            json.dumps({"run": run_name, "arm": schedule_arm,
                        "schedule": sched}), encoding="utf-8")
    if receipt_schema == "wm41_run_receipt/1":
        receipt = {"schema": receipt_schema, "run": run_name, "arm": run_name.split("__", 1)[1],
                   "policy": policy_field, "exit_status": receipt_status,
                   "evaluate_summary": {"duration_s": 600.0, "controller": base_controller}}
    else:
        receipt = {"schema": receipt_schema, "run": run_name, "exit_status": receipt_status,
                   "saved_state_id": saved_state_id,
                   "suffix_summary": {"rows_written": rows_written if rows_written is not None else 1200}}
    (rd / "RUN_RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")
    if physics_lines:
        with open(rd / "run" / "physics_steps.jsonl", "w", encoding="utf-8") as fh:
            for i in range(physics_lines):
                nveh = (i % 5) + 1
                vehicles = {f"mainline.{j}": {
                    "road_id": "1191554636", "lane_id": f"1191554636_{j % 3}",
                    "lane_position_m": 100.0 + j, "speed_mps": 20.0 + j,
                    "acceleration_mps2": 0.1 * j, "type_id": "normal_car",
                    "speed_mode": 27, "lane_changed": False} for j in range(nveh)}
                row = {"time_s": TICK(i), "live_ids": list(vehicles),
                       "departed_ids": [], "arrived_ids": [],
                       "cav_ids": [f"mainline.{j}" for j in range(min(2, nveh))],
                       "vehicles": vehicles}
                fh.write(json.dumps(row) + "\n")
    return rd


def TICK(i):
    return 0.5 * (i + 1)


def _world(world, split, t0=T0, loading=2600, pen=0.02):
    return {"world": world, "seed": 1, "loading_veh_h": loading,
            "penetration": pen, "split": split, "role": "test", "t0_s": t0}


def build_fixture_root(tmp_path):
    root = Path(tmp_path)
    (root / "protocols").mkdir(parents=True, exist_ok=True)
    (root / "plans").mkdir(parents=True, exist_ok=True)

    (root / "protocols" / "holdout_exclusions.json").write_text(json.dumps({
        "schema": "holdout_exclusions/1",
        "expected_excluded_count": 1,
        "excluded": [{"world": EXCLUDED_WORLD, "reason": "test", "source": "test"}],
        "kept_disclosed": [],
    }), encoding="utf-8")

    base_arm = {"arm": "base", "run": f"{TRAIN_WORLD}__base",
                "controller_name": "damping", "duration_s": 900.0,
                "mode": "full_instrumented", "saves_state": {"at_s": T0}}
    plan_a = {
        "schema": "wm44_chunk1_world_plan/1", "chunk": 1,
        "worlds": [dict(_world(TRAIN_WORLD, "TRAIN"), base_arm=base_arm,
                        branches=[{"arm": a} for a in ("native", "pi", "fixed", "switching")])],
        "runs": [
            dict(base_arm, world=TRAIN_WORLD, split="TRAIN"),
            {"arm": "native", "run": f"{TRAIN_WORLD}__native", "world": TRAIN_WORLD,
             "split": "TRAIN", "duration_s": 600.0, "restores_state": {"at_s": T0}},
            {"arm": "pi", "run": f"{TRAIN_WORLD}__pi", "world": TRAIN_WORLD,
             "split": "TRAIN", "duration_s": 600.0, "restores_state": {"at_s": T0}},
            {"arm": "fixed", "run": f"{TRAIN_WORLD}__fixed1", "world": TRAIN_WORLD,
             "split": "TRAIN", "duration_s": 600.0, "restores_state": {"at_s": T0}},
            {"arm": "switching", "run": f"{TRAIN_WORLD}__switching1", "world": TRAIN_WORLD,
             "split": "TRAIN", "duration_s": 600.0, "restores_state": {"at_s": T0}},
        ],
    }
    (root / "plans" / "chunk1.json").write_text(json.dumps(plan_a), encoding="utf-8")

    plan_b = {
        "schema": "wm73_truearms_plan/1", "chunk": "truearms_chunk2",
        "worlds": [_world(TRAIN_WORLD, "TRAIN")],
        "runs": [
            {"arm": "native", "run": f"{TRAIN_WORLD}__native", "world": TRAIN_WORLD,
             "split": "TRAIN", "duration_s": 600.0, "restores_state": {"at_s": T0}},
            {"arm": "pi", "run": f"{TRAIN_WORLD}__pi", "world": TRAIN_WORLD,
             "split": "TRAIN", "duration_s": 600.0, "restores_state": {"at_s": T0}},
        ],
    }
    (root / "plans" / "truearms.json").write_text(json.dumps(plan_b), encoding="utf-8")

    plan_c = {
        "schema": "wm59_walls_pilot_world_plan/1",
        "worlds": [_world(TRAIN_WORLD, "TRAIN")],
        "runs": [
            {"arm": "W-lo-early", "run": f"{TRAIN_WORLD}__w-lo-early", "mode": "restore_wall",
             "duration_s": 600.0, "restores_state": {"at_s": T0}},
            {"arm": "W-mid-early", "run": f"{TRAIN_WORLD}__w-mid-early", "mode": "restore_wall",
             "duration_s": 600.0, "restores_state": {"at_s": T0}},
        ],
    }
    (root / "plans" / "walls.json").write_text(json.dumps(plan_c), encoding="utf-8")

    plan_ex = {
        "schema": "wm44_chunk1_world_plan/1", "chunk": 1,
        "worlds": [_world(EXCLUDED_WORLD, "HOLDOUT")],
        "runs": [{"arm": "base", "run": f"{EXCLUDED_WORLD}__base",
                  "world": EXCLUDED_WORLD, "split": "HOLDOUT"}],
    }
    (root / "plans" / "ex.json").write_text(json.dumps(plan_ex), encoding="utf-8")

    write_run(root, SPEC_A["root_rel"], f"{TRAIN_WORLD}__base",
              receipt_schema="wm41_run_receipt/1", base_controller="damping",
              policy_field=None, physics_lines=40, saved_state_id=None)
    write_run(root, SPEC_A["root_rel"], f"{TRAIN_WORLD}__native",
              schedule_arm="switching", events=5, saved_state_id="sst-1")
    write_run(root, SPEC_A["root_rel"], f"{TRAIN_WORLD}__pi",
              schedule_arm="switching", events=5, saved_state_id="sst-1")
    write_run(root, SPEC_A["root_rel"], f"{TRAIN_WORLD}__fixed1",
              schedule_arm="fixed", events=5, saved_state_id="sst-1")
    write_run(root, SPEC_A["root_rel"], f"{TRAIN_WORLD}__switching1",
              schedule_arm="switching", events=5, saved_state_id="sst-1")

    write_run(root, SPEC_B["root_rel"], f"{TRAIN_WORLD}__native",
              schedule_arm="native", events=29, saved_state_id="sst-1")
    write_run(root, SPEC_B["root_rel"], f"{TRAIN_WORLD}__pi",
              schedule_arm="fleet-pi", events=29, saved_state_id="sst-1")

    write_run(root, SPEC_C["root_rel"], f"{TRAIN_WORLD}__w-lo-early",
              saved_state_id="sst-1")
    write_run(root, SPEC_C["root_rel"], f"{TRAIN_WORLD}__w-mid-early",
              saved_state_id="sst-1")

    write_run(root, SPEC_EX["root_rel"], f"{EXCLUDED_WORLD}__base",
              receipt_schema="wm41_run_receipt/1", base_controller="damping")
    (_run_dir(root, SPEC_EX["root_rel"], f"{EXCLUDED_WORLD}__base") / "OPENED_MARKER").write_text("x")
    return root
