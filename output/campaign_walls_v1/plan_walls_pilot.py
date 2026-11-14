"""WM-59 WALL-2: the wall-pilot PLAN builder (deterministic, zero SUMO).

Builds ``results_plan/WORLD_PLAN.json`` from the two stage-1 SOURCE plans
(chunk 2 + chunk 3, the owner-designated E:/ inputs):

- 60 worlds: 20 per p10/p20/p40, drawn deterministically from the union
  of the two chunks' mode-27 worlds under PLAN_SEED = 20261101, with the
  stage-1 49:7:14 split proportions -> 42 TRAIN / 6 CALIBRATION /
  12 HOLDOUT (per penetration block of 20: 14/2/4).  HOLDOUT worlds ARE
  included (their outcomes are never read before a registered final
  pass);
- 4 single-wall arms per world (W-lo/W-mid x early/late) = 240 restored
  600-s branches = 144,000 simulated s.  p05 and the W-2 multi-wall arm
  are DEFERRED by the owner (not built here);
- every branch RESTORES its world's chunk base t0 saved state
  (``bases_root`` recorded per world) with the SAME arrival seed as the
  world's EXISTING native sibling - the sibling is REUSED as the
  reference and only its run name/path is recorded (never re-run, and
  its outcomes are never read by this package);
- per world the plan records the base state digests (state.xml +
  STATE_MANIFEST.json sha256 + saved_state_id), re-verified by the
  registration at check time;
- per branch: a recorded branch seed (sha256-derived from the plan seed)
  and the fixed-hold draw H in {60, 120, 240} s from that seed.

The plan is a PURE function of the two source plans: rebuild and compare
bit-exact (the replay test).  ``created_utc`` is added only at write
time.  E:/ paths only; nothing outside this package is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

if __package__ in (None, ""):  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from campaign_stage1_v3.common import (  # noqa: E402 - bootstrap above
    load_json,
    sha256_file,
    sha256_text,
    utc_now,
    write_json_archiving,
)
from campaign_walls_v1.common import (  # noqa: E402
    ARM_SPEED_FACTORS,
    ARMS,
    BRANCHES_PER_WORLD,
    BRANCHES_TOTAL,
    BRANCH_LENGTH_S,
    CAV_SPEED_MODE,
    DISK_PREFLIGHT_MIN_FREE_GB,
    HOLD_CHOICES_S,
    MAINLINE_2LANE_EDGES,
    PENETRATIONS,
    PLAN_PATH,
    PLAN_SEED,
    REQUEST_WINDOW_S,
    SIMULATED_SECONDS_TOTAL,
    SMOKE_ARM,
    SMOKE_HORIZON_TICKS,
    SMOKE_MAX_SIMULATED_S,
    SMOKE_PENETRATION,
    SPEED_FLOOR_MPS,
    SOURCE_CHUNK2_BASES_ROOT,
    SOURCE_CHUNK2_PLAN,
    SOURCE_CHUNK3_BASES_ROOT,
    SOURCE_CHUNK3_PLAN,
    SPLIT_COUNTS,
    SPLIT_COUNTS_PER_PEN,
    TIMING_OFFSET_S,
    WALLS_WORKERS,
    WORLDS_PER_PENETRATION,
    WORLDS_TOTAL,
    LAUNCH_AUTHORIZATION,
    LAUNCH_OUT_DIR,
    LAUNCH_RUNS_ROOT,
    SMOKE_OUT_DIR,
    SMOKE_RUNS_ROOT,
    WM41_REGISTRATION,
    WM41_VERDICT,
    WM43_SUMMARY,
    WallsPilotError,
    arm_speed_factor,
    arm_timing,
)

PLAN_SCHEMA = "wm59_walls_pilot_world_plan/1"
SOURCE_SCHEMAS = {
    "chunk2": "wm47_chunk2_world_plan/1",
    "chunk3": "wm54_chunk3_world_plan/1",
}
SOURCE_PLAN_PATHS = {"chunk2": SOURCE_CHUNK2_PLAN, "chunk3": SOURCE_CHUNK3_PLAN}
SOURCE_BASES_ROOTS = {
    "chunk2": SOURCE_CHUNK2_BASES_ROOT,
    "chunk3": SOURCE_CHUNK3_BASES_ROOT,
}
SAVED_STATE_DIR = "saved_state"
STATE_XML = "state.xml"
STATE_MANIFEST = "STATE_MANIFEST.json"


class PlanError(WallsPilotError):
    """A wall-pilot plan contract violation."""


def branch_seed(world_label: str, run_name: str) -> int:
    """Deterministic per-branch RNG seed from the plan seed (replayable)."""
    digest = hashlib.sha256(
        f"wm59:{PLAN_SEED}:{world_label}:{run_name}".encode()).hexdigest()
    return int(digest, 16) % (10 ** 15)


def hold_draw(seed: int) -> float:
    """The fixed hold H in {60, 120, 240} s drawn from the branch seed."""
    return float(random.Random(f"wm59-hold:{seed}").choice(HOLD_CHOICES_S))


def load_source_plan(chunk: str, path: str | Path | None = None) -> dict:
    """One SOURCE plan (chunk 2 or chunk 3); schema-checked, metadata only."""
    path = Path(path) if path is not None else Path(SOURCE_PLAN_PATHS[chunk])
    if not path.is_file():
        raise PlanError(f"the {chunk} SOURCE plan is missing: {path}")
    plan = load_json(path)
    expected = SOURCE_SCHEMAS[chunk]
    if plan.get("schema") != expected:
        raise PlanError(
            f"{path} schema {plan.get('schema')!r} != {expected!r}")
    return plan


def base_state_entry(bases_root: str | Path, base_run: str) -> dict:
    """The base saved-state digests for one world (manifest metadata)."""
    state_dir = Path(bases_root) / base_run / SAVED_STATE_DIR
    state_xml = state_dir / STATE_XML
    manifest_path = state_dir / STATE_MANIFEST
    if not state_xml.is_file() or not manifest_path.is_file():
        raise PlanError(
            f"base saved state missing for {base_run!r} under {bases_root} "
            "(the wall branches restore from it; the bases must exist "
            "before the pilot plan is built)")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    state_sha = sha256_file(state_xml)
    return {
        "state_xml_sha256": state_sha,
        "state_manifest_sha256": sha256_file(manifest_path),
        "saved_state_id": sha256_text(manifest_text + state_sha),
        "manifest_branch_point_t0_s": load_json(manifest_path).get(
            "branch_point_t0_s"),
    }


def source_world_entries(chunk2_plan: dict, chunk3_plan: dict, *,
                         chunk2_path: str = SOURCE_CHUNK2_PLAN,
                         chunk3_path: str = SOURCE_CHUNK3_PLAN,
                         bases_roots: dict | None = None,
                         read_base_state: bool = True) -> list[dict]:
    """The candidate worlds (p10/p20/p40 of both chunks) with their base
    and native-sibling identity resolved.  Only plan/manifest METADATA is
    read - never outcomes.  ``bases_roots`` overrides the registered
    chunk roots (the replay tests point it at temp copies)."""
    roots = dict(SOURCE_BASES_ROOTS)
    if bases_roots is not None:
        roots.update(bases_roots)
    candidates: list[dict] = []
    seen: set[str] = set()
    for chunk, plan, path in (
            ("chunk2", chunk2_plan, chunk2_path),
            ("chunk3", chunk3_plan, chunk3_path)):
        for world in plan["worlds"]:
            if float(world["penetration"]) not in PENETRATIONS:
                continue
            name = str(world["world"])
            if name in seen:
                raise PlanError(f"world {name!r} appears in both SOURCE plans")
            seen.add(name)
            base_run = str(world["base_arm"]["run"])
            native = next((b for b in world["branches"]
                           if b["arm"] == "native"), None)
            if native is None:
                raise PlanError(f"world {name!r} has no native sibling branch")
            entry = {
                "world": name,
                "seed": int(world["seed"]),
                "loading_veh_h": world["loading_veh_h"],
                "penetration": float(world["penetration"]),
                "split": world["split"],
                "role": world["role"],
                "t0_s": float(world["t0_s"]),
                "source_chunk": chunk,
                # posix form: the plan must not depend on how the builder
                # was invoked (Path args normalize separators on Windows)
                "source_plan": Path(path).as_posix(),
                "bases_root": str(roots[chunk]),
                "base_run": base_run,
                "native_sibling": {
                    "run": str(native["run"]),
                    "chunk": chunk,
                    "runs_root": str(roots[chunk]),
                },
            }
            if read_base_state:
                entry["base_state"] = base_state_entry(
                    roots[chunk], base_run)
            candidates.append(entry)
    return candidates


def select_worlds(candidates: list[dict]) -> list[dict]:
    """The 60 pilot worlds: 20 per penetration, deterministic from
    PLAN_SEED, per-block split counts 14/2/4 (-> 42/6/12 overall)."""
    selected: list[dict] = []
    for pen in PENETRATIONS:
        for split, count in SPLIT_COUNTS_PER_PEN.items():
            pool = sorted(
                (c for c in candidates
                 if c["penetration"] == float(pen) and c["split"] == split),
                key=lambda c: c["world"])
            if len(pool) < count:
                raise PlanError(
                    f"p{int(pen * 100)} {split}: need {count} worlds, "
                    f"the SOURCE plans offer {len(pool)}")
            rng = random.Random(f"wm59:{PLAN_SEED}:select:{pen}:{split}")
            rng.shuffle(pool)
            selected.extend(pool[:count])
    if len(selected) != WORLDS_TOTAL:
        raise PlanError(f"selected {len(selected)} worlds != {WORLDS_TOTAL}")
    return sorted(selected, key=lambda c: (c["penetration"], c["world"]))


def world_plan_entry(entry: dict) -> dict:
    """One pilot world with its 4 wall-arm branches."""
    branches = []
    for arm in ARMS:
        run = f"{entry['world']}__{arm.lower()}"
        seed = branch_seed(entry["world"], run)
        factor = arm_speed_factor(arm)
        timing = arm_timing(arm)
        target = entry["t0_s"] + TIMING_OFFSET_S[timing]
        branches.append({
            "arm": arm,
            "run": run,
            "mode": "restore_wall",
            "duration_s": BRANCH_LENGTH_S,
            "restores_state": {
                "from_run": entry["base_run"],
                "at_s": entry["t0_s"],
            },
            "native_sibling_run": entry["native_sibling"]["run"],
            "branch_seed": seed,
            "wall": {
                "speed_factor": factor,
                "timing": timing,
                "target_time_s": target,
                "request_window_s": REQUEST_WINDOW_S,
                "deadline_s": target + REQUEST_WINDOW_S,
                "speed_floor_mps": SPEED_FLOOR_MPS,
                "hold_s": hold_draw(seed),
            },
        })
    return {
        "world": entry["world"],
        "seed": entry["seed"],
        "loading_veh_h": entry["loading_veh_h"],
        "penetration": entry["penetration"],
        "split": entry["split"],
        "role": entry["role"],
        "t0_s": entry["t0_s"],
        "source_chunk": entry["source_chunk"],
        "source_plan": entry["source_plan"],
        "bases_root": entry["bases_root"],
        "base_run": entry["base_run"],
        "base_state": entry["base_state"],
        "native_sibling": entry["native_sibling"],
        "branches": branches,
    }


def build_plan(chunk2_plan: dict, chunk3_plan: dict, *,
               chunk2_path: str = SOURCE_CHUNK2_PLAN,
               chunk3_path: str = SOURCE_CHUNK3_PLAN,
               bases_roots: dict | None = None) -> dict:
    """The deterministic pilot plan (PURE; no clock, no I/O beyond the
    base-state digest reads done by ``source_world_entries``)."""
    candidates = source_world_entries(
        chunk2_plan, chunk3_plan, chunk2_path=chunk2_path,
        chunk3_path=chunk3_path, bases_roots=bases_roots)
    worlds = [world_plan_entry(e) for e in select_worlds(candidates)]
    runs = [b for w in worlds for b in w["branches"]]
    smoke_world = next(w for w in worlds
                       if w["penetration"] == SMOKE_PENETRATION)
    smoke_run = next(b["run"] for b in smoke_world["branches"]
                     if b["arm"] == SMOKE_ARM)
    plan = {
        "schema": PLAN_SCHEMA,
        "plan_seed": PLAN_SEED,
        "wm_id": "WM-59",
        "stage": "walls_pilot",
        "constants": {
            "worlds": WORLDS_TOTAL,
            "worlds_per_penetration": WORLDS_PER_PENETRATION,
            "penetrations": list(PENETRATIONS),
            "arms": list(ARMS),
            "arm_speed_factors": dict(ARM_SPEED_FACTORS),
            "timing_offset_s": dict(TIMING_OFFSET_S),
            "request_window_s": REQUEST_WINDOW_S,
            "speed_floor_mps": SPEED_FLOOR_MPS,
            "hold_choices_s": list(HOLD_CHOICES_S),
            "branch_length_s": BRANCH_LENGTH_S,
            "mainline_2lane_edges": sorted(MAINLINE_2LANE_EDGES),
            "split_counts": dict(SPLIT_COUNTS),
            "split_counts_per_penetration": dict(SPLIT_COUNTS_PER_PEN),
            "non_members": "native",
            "multi_wall_W2": "deferred pending owner confirmation",
            "p05": "deferred (p10/p20/p40-only pre-screen approved by main discussion under owner delegation, 2026-09-29)",
            "runs_root_relative": "runs/campaign_walls_pilot_v1",
            "cav_speed_mode": CAV_SPEED_MODE,
            "workers": WALLS_WORKERS,
            "disk_preflight_min_free_gb": DISK_PREFLIGHT_MIN_FREE_GB,
            "smoke_max_simulated_s": SMOKE_MAX_SIMULATED_S,
            "smoke_horizon_ticks": SMOKE_HORIZON_TICKS,
            "launch_paths": {
                "runs_root": LAUNCH_RUNS_ROOT,
                "out_dir": LAUNCH_OUT_DIR,
                "authorization": LAUNCH_AUTHORIZATION,
                "smoke_runs_root": SMOKE_RUNS_ROOT,
                "smoke_out_dir": SMOKE_OUT_DIR,
            },
            "gate_paths": {
                "wm41_verdict": WM41_VERDICT,
                "wm41_registration": WM41_REGISTRATION,
                "wm43_summary": WM43_SUMMARY,
            },
            "source_plans": dict(SOURCE_PLAN_PATHS),
            "source_bases_roots": dict(SOURCE_BASES_ROOTS),
        },
        "budget": {
            "worlds": WORLDS_TOTAL,
            "branches_total": BRANCHES_TOTAL,
            "branch_length_s": BRANCH_LENGTH_S,
            "simulated_seconds_total": SIMULATED_SECONDS_TOTAL,
            "smoke_simulated_s_cap": SMOKE_MAX_SIMULATED_S,
        },
        "smoke": {
            "arm": SMOKE_ARM,
            "penetration": SMOKE_PENETRATION,
            "world": smoke_world["world"],
            "run": smoke_run,
            "max_simulated_s": SMOKE_MAX_SIMULATED_S,
            "horizon_ticks": SMOKE_HORIZON_TICKS,
            "bases_root": smoke_world["bases_root"],
            "restores_state": {
                "from_run": smoke_world["base_run"],
                "at_s": smoke_world["t0_s"],
            },
        },
        "worlds": worlds,
        "runs": runs,
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> None:
    """The registered shape; every violation is loud (never weakened)."""
    worlds = plan["worlds"]
    if len(worlds) != WORLDS_TOTAL:
        raise PlanError(f"plan carries {len(worlds)} worlds != {WORLDS_TOTAL}")
    per_pen = {str(pen): 0 for pen in PENETRATIONS}
    splits: dict[str, int] = {k: 0 for k in SPLIT_COUNTS}
    run_names: set[str] = set()
    for world in worlds:
        per_pen[str(world["penetration"])] += 1
        splits[world["split"]] += 1
        if len(world["branches"]) != BRANCHES_PER_WORLD:
            raise PlanError(
                f"world {world['world']}: {len(world['branches'])} branches "
                f"!= {BRANCHES_PER_WORLD}")
        if sorted(b["arm"] for b in world["branches"]) != sorted(ARMS):
            raise PlanError(f"world {world['world']}: arms != {ARMS}")
        for branch in world["branches"]:
            if branch["run"] in run_names:
                raise PlanError(f"duplicate run name {branch['run']!r}")
            run_names.add(branch["run"])
            hold = branch["wall"]["hold_s"]
            if hold not in HOLD_CHOICES_S:
                raise PlanError(f"{branch['run']}: hold {hold} not in "
                                f"{HOLD_CHOICES_S}")
            if branch["native_sibling_run"] is None:
                raise PlanError(f"{branch['run']}: no native sibling")
        if not world.get("base_state"):
            raise PlanError(f"world {world['world']}: no base-state digests")
        if world["native_sibling"]["run"] != world["branches"][0][
                "native_sibling_run"]:
            raise PlanError(f"world {world['world']}: sibling id mismatch")
    if any(count != WORLDS_PER_PENETRATION for count in per_pen.values()):
        raise PlanError(f"per-penetration worlds {per_pen} != "
                        f"{WORLDS_PER_PENETRATION} each")
    if splits != SPLIT_COUNTS:
        raise PlanError(f"split counts {splits} != {SPLIT_COUNTS}")
    if len(plan["runs"]) != BRANCHES_TOTAL:
        raise PlanError(f"plan carries {len(plan['runs'])} runs != "
                        f"{BRANCHES_TOTAL}")
    if plan["budget"]["simulated_seconds_total"] != SIMULATED_SECONDS_TOTAL:
        raise PlanError("budget is not the registered 144,000 sim-s")


def write_plan(plan: dict, out_dir: str | Path | None = None) -> Path:
    """Write WORLD_PLAN.json (created_utc added here, never in build_plan:
    the replay comparison is over the pure build).  A pre-existing plan is
    ARCHIVED (tag superseded) - never silently overwritten."""
    out_dir = Path(out_dir) if out_dir is not None else PLAN_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stamped = dict(plan)
    stamped["created_utc"] = utc_now()
    path = out_dir / "WORLD_PLAN.json"
    write_json_archiving(path, stamped)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk2-plan", type=Path, default=Path(SOURCE_CHUNK2_PLAN))
    parser.add_argument("--chunk3-plan", type=Path, default=Path(SOURCE_CHUNK3_PLAN))
    parser.add_argument("--out-dir", type=Path, default=PLAN_PATH.parent)
    args = parser.parse_args(argv)
    chunk2 = load_source_plan("chunk2", args.chunk2_plan)
    chunk3 = load_source_plan("chunk3", args.chunk3_plan)
    plan = build_plan(chunk2, chunk3, chunk2_path=str(args.chunk2_plan),
                      chunk3_path=str(args.chunk3_plan))
    path = write_plan(plan, args.out_dir)
    summary = {
        "plan": str(path),
        "plan_sha256": sha256_file(path),
        "worlds": len(plan["worlds"]),
        "worlds_per_penetration": {
            str(pen): sum(1 for w in plan["worlds"]
                          if w["penetration"] == pen)
            for pen in PENETRATIONS},
        "splits": {
            split: sum(1 for w in plan["worlds"] if w["split"] == split)
            for split in SPLIT_COUNTS},
        "branches": len(plan["runs"]),
        "simulated_seconds_total": plan["budget"]["simulated_seconds_total"],
        "smoke_world": plan["smoke"]["world"],
        "smoke_run": plan["smoke"]["run"],
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
