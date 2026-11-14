"""INC1 world-plan builder (deterministic; zero SUMO).

Builds ``results_plan_inc1/WORLD_PLAN.json`` exactly as registration
section 2-4 prescribe:

* 20 fresh worlds from the audited seed block 9601-9620;
* loadings 2400/2600/2700 with 7/7/6 worlds;
* penetrations p05/p10/p20/p40 round-robin within each loading;
* incident position A (s_down +40 m) / B (s_down -150 m) split evenly
  (10/10) and distributed within each loading;
* the chunk-4 scenario template (``baselines/circles/adapter.json``,
  ``scenarios/idm/driver_routes.rou.xml``, ramp 400 veh/h, speed mode 27),
  t0 = 300 s, 300 s arms.

The builder is a PURE function of the frozen constants and the audit: no
clock, no simulation, no outcome read.  It refuses to build unless the
committed seed audit's chosen block starts at the registered seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    from . import constants as C
except ImportError:  # script execution
    import constants as C  # type: ignore

from traffic_shepherd.config import load_config  # noqa: E402


class PlanError(ValueError):
    """An INC1 world-plan contract violation; never silently relaxed."""


def _scenario_binding() -> dict:
    """The chunk-4 scenario template identity (adapter.json; read-only)."""
    config = load_config()
    return {
        "adapter": "baselines/circles/adapter.json",
        "scenario_template_path": str(config.scenario_template_path),
        "network_path": str(config.network_path),
        "ramp_vph": int(config.ramp_vph),
        "episode_seconds": int(config.episode_seconds),
        "demand_end_seconds": int(config.demand_end_seconds),
        "sensor_range_m": float(config.sensor_range_m),
        "cav_speed_mode": int(C.CAV_SPEED_MODE),
        "control_dt_s": float(C.CONTROL_DT_S),
        "physics_dt_s": float(C.PHYSICS_DT_S),
    }


def _assign_loadings(seeds: list[int]) -> list[dict]:
    """Seeds -> (seed, loading, penetration, incident) in registration order."""
    worlds: list[dict] = []
    index = 0
    for loading, count in zip(C.LOADINGS, C.WORLDS_PER_LOADING):
        for position in range(count):
            seed = seeds[index]
            penetration = C.PENETRATIONS[position % len(C.PENETRATIONS)]
            incident = "A" if index % 2 == 0 else "B"
            worlds.append({
                "index": index,
                "seed": seed,
                "loading_veh_h": loading,
                "loading_position": position,
                "penetration": penetration,
                "incident_class": incident,
            })
            index += 1
    if index != len(seeds):
        raise PlanError(f"assigned {index} seeds != {len(seeds)}")
    return worlds


def world_name(seed: int, loading: int, penetration: float, t0: float) -> str:
    """Chunk-4-style run id: ``w<seed>m<loading>p<pen>t<t0>``."""
    return f"w{seed}m{loading}p{int(round(penetration * 100)):02d}t{int(round(t0))}"


def build_plan(*, seeds: list[int] | None = None,
               audit_path: Path | None = None,
               include_scenario_binding: bool = True) -> dict:
    """The deterministic INC1 plan (pure; no clock, no I/O beyond adapter)."""
    audit_path = C.SEED_AUDIT_PATH if audit_path is None else Path(audit_path)
    if seeds is None:
        if not audit_path.is_file():
            raise PlanError(
                f"the committed seed audit is missing: {audit_path}; run the "
                "seed audit before building the plan (registration section 2)")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("chosen_free"):
            raise PlanError("the committed seed audit did not clear its block")
        seeds = list(audit["chosen_block"])
    if len(seeds) != C.SEED_BLOCK_SIZE:
        raise PlanError(f"need {C.SEED_BLOCK_SIZE} seeds, got {len(seeds)}")

    binding = _scenario_binding() if include_scenario_binding else {}
    assigned = _assign_loadings(seeds)
    worlds = []
    for entry in assigned:
        seed = entry["seed"]
        loading = entry["loading_veh_h"]
        penetration = entry["penetration"]
        incident = entry["incident_class"]
        name = world_name(seed, loading, penetration, C.T0_S)
        s_down = C.INCIDENT_S_DOWN[incident]
        branches = [{
            "arm": "warmup",
            "run": f"{name}__warmup",
            "mode": "warmup_native_savestate",
            "duration_s": C.T0_S,
            "restores_state": None,
        }]
        for arm in C.ARM_NAMES:
            branches.append({
                "arm": arm,
                "run": f"{name}__{arm.lower()}",
                "mode": "restore_incident",
                "duration_s": C.ARM_HORIZON_S,
                "restores_state": {"from_run": f"{name}__warmup",
                                   "at_s": C.T0_S},
                "description": C.ARM_DESCRIPTIONS[arm],
                "incident": {
                    "class": incident,
                    "s_down_m": s_down,
                    "lane_phys": C.INCIDENT_LANE_PHYS,
                    "hold_s": C.OBSTACLE_HOLD_S,
                    "min_gap_m": C.OBSTACLE_MIN_GAP_M,
                    "search_window_m": C.OBSTACLE_SEARCH_WINDOW_M,
                },
            })
        worlds.append({
            "world": name,
            "seed": seed,
            "loading_veh_h": loading,
            "penetration": penetration,
            "split": "SEALED_INCIDENT",
            "role": "incident_demo",
            "t0_s": C.T0_S,
            "incident_class": incident,
            "incident_s_down_m": s_down,
            "warmup_run": f"{name}__warmup",
            "branches": branches,
        })

    runs = [b for w in worlds for b in w["branches"]]
    plan = {
        "schema": C.PLAN_SCHEMA,
        "wm_id": "WM-N1-INC1",
        "registration_path": str(C.REGISTRATION_PATH),
        "registration_lf_sha256": C.REGISTRATION_LF_SHA,
        "scenario": binding,
        "constants": {
            "seed_block_start": C.SEED_BLOCK_START,
            "seed_block_size": C.SEED_BLOCK_SIZE,
            "loadings": list(C.LOADINGS),
            "worlds_per_loading": list(C.WORLDS_PER_LOADING),
            "penetrations": list(C.PENETRATIONS),
            "arms": list(C.ARM_NAMES),
            "t0_s": C.T0_S,
            "arm_horizon_s": C.ARM_HORIZON_S,
            "control_dt_s": C.CONTROL_DT_S,
            "physics_dt_s": C.PHYSICS_DT_S,
            "cav_speed_mode": C.CAV_SPEED_MODE,
            "merge_start_s": C.MERGE_START_S,
            "incident_s_down": dict(C.INCIDENT_S_DOWN),
            "incident_lane_phys": C.INCIDENT_LANE_PHYS,
            "obstacle_min_gap_m": C.OBSTACLE_MIN_GAP_M,
            "obstacle_search_window_m": C.OBSTACLE_SEARCH_WINDOW_M,
            "obstacle_hold_s": C.OBSTACLE_HOLD_S,
            "r1": {"brake_mps2": C.R1_BRAKE_MPS2, "hold_s": C.R1_HOLD_S,
                   "range_m": C.R1_RANGE_M},
            "r2": {"target_mps": C.R2_TARGET_MPS,
                   "decel_max_mps2": C.R2_DECEL_MAX_MPS2,
                   "duration_s": C.R2_DURATION_S,
                   "upstream_lo_m": C.R2_UPSTREAM_LO_M,
                   "upstream_hi_m": C.R2_UPSTREAM_HI_M},
            "r4": {"speed_factor": C.R4_SPEED_FACTOR,
                   "request_window_s": C.R4_REQUEST_WINDOW_S,
                   "blocked": False,
                   "driver_path": "inc1.r4_driver",
                   "hook_patch": C.R4_HOOK_PATCH,
                   "freeze_addendum": C.R4_FREEZE_ADDENDUM,
                   "note": ("R4 (frozen wall law v2.2) is driven through the "
                            "patched frozen driver (ADDENDUM_1 A1.2; "
                            "inc1.r4_driver) with the J4 mid factor 0.85 and "
                            "the wall request at t0. A proposer refusal or an "
                            "aborted assembly is logged and flagged "
                            "infeasible.")},
            "r5": {"duration_s": C.R5_DURATION_S,
                   "min_members": C.R5_MIN_MEMBERS,
                   "generator_module": "inc1_gen/generators.py",
                   "note": ("ADDENDUM_2 A2.2: coordinated upstream speed "
                            "harmonisation; d in [100, 1500) m at t0, "
                            "staggered target for 120 s, feasible iff >= 2 "
                            "members; driven through CommandOwnership "
                            "longitudinal-only.")},
            "r6": {"duration_s": C.R6_DURATION_S,
                   "min_members": C.R6_MIN_MEMBERS,
                   "min_openers": C.R6_MIN_OPENERS,
                   "generator_module": "inc1_gen/generators.py",
                   "note": ("ADDENDUM_2 A2.3: coordinated lane-clearing and "
                            "gap opening; adjacent-lane openers track a 3.0 s "
                            "headway, blocked-lane approachers track "
                            "min(12 m/s, native) for 180 s; feasible iff >= 2 "
                            "members with >= 1 gap opener.")},
            "generators_manifest": "inc1_gen/INC1_GENERATORS.json",
            "non_members": "native",
        },
        "budget": {
            "warmup_sim_s": C.BUDGET_WARMUP_S,
            "arms_sim_s": C.BUDGET_ARMS_S,
            "parity_smoke_sim_s": C.BUDGET_PARITY_S,
            "harness_equivalence_smoke_sim_s": C.BUDGET_HARNESS_EQUIV_S,
            "total_sim_s": C.BUDGET_TOTAL_S,
            "items": dict(C.BUDGET_ITEMS),
            "max_crashed_restarts": C.MAX_CRASHED_RESTARTS,
            "retry_complete": C.RETRY_COMPLETE,
        },
        "worlds": worlds,
        "runs": runs,
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> None:
    """The registered shape; every violation is loud (never weakened)."""
    worlds = plan["worlds"]
    if len(worlds) != C.SEED_BLOCK_SIZE:
        raise PlanError(f"plan carries {len(worlds)} worlds != {C.SEED_BLOCK_SIZE}")
    per_loading: dict = {}
    per_incident: dict = {"A": 0, "B": 0}
    run_names: set = set()
    for world in worlds:
        per_loading[world["loading_veh_h"]] = per_loading.get(
            world["loading_veh_h"], 0) + 1
        per_incident[world["incident_class"]] += 1
        if world["incident_s_down_m"] != C.INCIDENT_S_DOWN[world["incident_class"]]:
            raise PlanError(f"{world['world']}: incident s_down mismatch")
        if len(world["branches"]) != 1 + len(C.ARM_NAMES):
            raise PlanError(f"{world['world']}: branch count mismatch")
        arms = [b["arm"] for b in world["branches"]]
        if arms[0] != "warmup" or sorted(arms[1:]) != sorted(C.ARM_NAMES):
            raise PlanError(f"{world['world']}: arms != warmup + {C.ARM_NAMES}")
        for branch in world["branches"]:
            if branch["run"] in run_names:
                raise PlanError(f"duplicate run name {branch['run']!r}")
            run_names.add(branch["run"])
    if [per_loading.get(x, 0) for x in C.LOADINGS] != list(C.WORLDS_PER_LOADING):
        raise PlanError(
            f"loading counts {per_loading} != {list(C.WORLDS_PER_LOADING)}")
    if per_incident != {"A": 10, "B": 10}:
        raise PlanError(f"incident split {per_incident} != 10/10")
    if len(plan["runs"]) != C.SEED_BLOCK_SIZE * (1 + len(C.ARM_NAMES)):
        raise PlanError("run count mismatch")
    if plan["budget"]["total_sim_s"] != C.BUDGET_TOTAL_S:
        raise PlanError("budget total is not the registered 49,920 sim-s")


def write_plan(plan: dict, out_dir: Path | None = None) -> Path:
    """Write WORLD_PLAN.json; a pre-existing plan is never overwritten."""
    out_dir = C.PLAN_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "WORLD_PLAN.json"
    if path.exists():
        raise PlanError(f"refusing to overwrite an existing plan: {path}")
    stamped = dict(plan)
    stamped["created_utc"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(json.dumps(stamped, indent=2) + "\n", encoding="utf-8")
    return path


def plan_sha256(path: Path | None = None) -> str:
    return hashlib.sha256((C.PLAN_PATH if path is None else Path(path)).read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="INC1 world-plan builder")
    parser.add_argument("--audit", type=Path, default=C.SEED_AUDIT_PATH)
    parser.add_argument("--out-dir", type=Path, default=C.PLAN_DIR)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(audit_path=args.audit)
    summary = {
        "worlds": len(plan["worlds"]),
        "runs": len(plan["runs"]),
        "loadings": {str(loading): sum(1 for w in plan["worlds"]
                                       if w["loading_veh_h"] == loading)
                     for loading in C.LOADINGS},
        "penetrations": {f"p{int(p*100):02d}": sum(
            1 for w in plan["worlds"] if w["penetration"] == p)
            for p in C.PENETRATIONS},
        "incident": {k: sum(1 for w in plan["worlds"]
                            if w["incident_class"] == k)
                     for k in C.INCIDENT_POSITIONS},
        "total_sim_s": plan["budget"]["total_sim_s"],
    }
    if not args.no_write:
        path = write_plan(plan, args.out_dir)
        summary["plan"] = str(path)
        summary["plan_sha256"] = plan_sha256(path)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
