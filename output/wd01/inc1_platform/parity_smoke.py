"""WM-N1-INC1 §7 platform parity smoke (comparison logic; ZERO SUMO here).

Compares a 60 s platform (Linux box) native run against the laptop's
RECORDED per-vehicle physics for the same window, restored from the same
saved t0 state and scenario copy.  Two chunk-4 TRAIN worlds are pinned in
PARITY_INPUTS.json (HOLDOUT/CAL/chunk-5 are never referenced).

PASS rule (registration REGISTRATION_WM_N1_INC1.md §7): aggregate delay
within 2 % and zero collisions/teleports on the platform.  Bitwise
identity is NOT required: max |Δs| / |Δv| are reported descriptively.

The tool never starts SUMO itself.  The actual simulation is delegated to
an injected runner (--runner module:func) that runs on the Linux box:

    runner(scenario_dir: Path, state_xml: Path, duration_s: float,
           out_jsonl: Path) -> None

It must load the saved state into a COPY of the scenario (net-file
rewritten to the local Linux network asset, never the tarball original),
run `duration_s` at 1 s control / 0.5 s physics, and write physics rows
in the same jsonl schema as the recorded laptop runs (time_s, vehicles
with road_id/lane_id/lane_position_m/speed_mps, colliding_ids,
starting_teleports).

Delay convention: the frozen Part A window-delay step
(partA_v2g/metrics.py delay_step over act_lib): per vehicle and 0.5 s
step, delay = DT - ds/lim, ds = decrease of merge-relative d = offset
(phys lane) - s, lim = segment speed limit at the PREVIOUS position;
junction/internal-lane rows are skipped.  The chunk-4 physical-lane
topology (pdl.py) and frozen geometry (partA_v2g/params_nominal.json) are
embedded below.  Cost sign: diff % = platform - laptop; positive = the
platform is worse.

Forbidden inputs are refused by name (evaluate_summary, suffix_summary,
dispatcher_rows, insertions) and only TRAIN worlds are accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Tuple

DT = 0.5
DEFAULT_HORIZON_S = 60.0
DELAY_TOL_PCT = 2.0

FORBIDDEN_TOKENS = ("evaluate_summary", "suffix_summary",
                    "dispatcher_rows", "insertions")

# ---------------------------------------------------------------- geometry
# Physical lanes of the chunk-4 merge scenario (partA_data/pdl.py):
#   main_left  = 1191554636_1 -> 1487606179_2 -> 1487606178_1  (phys 0)
#   main_right = 1191554636_0 -> 1487606179_1 -> 1487606178_0  (phys 1)
#   accel      = 11553048#1_0 -> 1487606179_0                  (phys 2)
# (road_id, lane_index) -> (lane_phys, s offset within the physical lane)
LANE_CHAINS: Dict[Tuple[str, int], Tuple[int, float]] = {
    ("1191554636", 1): (0, 0.0),
    ("1487606179", 2): (0, 2645.67),
    ("1487606178", 1): (0, 2721.11),
    ("1191554636", 0): (1, 0.0),
    ("1487606179", 1): (1, 2644.22),
    ("1487606178", 0): (1, 2719.66),
    ("11553048#1", 0): (2, 0.0),
    ("1487606179", 0): (2, 378.66),
}
# Frozen WM-30 geometry (partA_v2g/params_nominal.json, identical for all
# chunk-4 worlds): merge-relative offsets, segment starts and limits.
OFFSETS = (2645.67, 2644.22, 378.66)
SEG_STARTS = ((0.0, 2645.67, 2721.11), (0.0, 2644.22, 2719.66),
              (0.0, 378.66))
SEG_LIMITS = ((31.29, 31.29, 31.29), (31.29, 31.29, 31.29),
              (22.22, 31.29))


class InputRefused(RuntimeError):
    """An input violates the access rules (forbidden file or split)."""


def split_lane_id(lane_id: str) -> Tuple[str, int]:
    """'1487606178_0' -> ('1487606178', 0); internal lanes -> ('', -1)."""
    if lane_id.startswith(":") or "_" not in lane_id:
        return "", -1
    road, _, idx = lane_id.rpartition("_")
    try:
        return road, int(idx)
    except ValueError:
        return "", -1


def lane_phys(road_id: str, lane_id: str) -> Optional[int]:
    road, idx = split_lane_id(lane_id)
    chain = LANE_CHAINS.get((road, idx))
    return chain[0] if chain else None


def phys_s(road_id: str, lane_id: str, lane_position_m: float) -> Optional[float]:
    road, idx = split_lane_id(lane_id)
    chain = LANE_CHAINS.get((road, idx))
    if chain is None:
        return None
    return chain[1] + float(lane_position_m)


def segment_limit(lane_phys_i: int, s: float) -> Optional[float]:
    starts = SEG_STARTS[lane_phys_i]
    limits = SEG_LIMITS[lane_phys_i]
    for k in range(len(starts) - 1, -1, -1):
        if s >= starts[k]:
            return limits[k]
    return limits[0]


def delay_step(prev: Tuple[str, str, float, float],
               cur: Tuple[str, str, float, float]) -> Optional[float]:
    """Frozen (DT - ds/lim) step; None when either row is unmapped."""
    la = lane_phys(prev[0], prev[1])
    lb = lane_phys(cur[0], cur[1])
    if la is None or lb is None or la != lb:
        return None
    sa = phys_s(prev[0], prev[1], prev[2])
    sb = phys_s(cur[0], cur[1], cur[2])
    lim = segment_limit(la, sa)
    if lim is None or lim <= 0.0:
        return None
    ds = (OFFSETS[la] - sa) - (OFFSETS[lb] - sb)
    return DT - ds / lim


# ------------------------------------------------------------------ rows

@dataclass
class PhysicsWindow:
    world: str
    t0: float
    horizon_s: float
    ticks: Dict[int, Dict[str, Tuple[str, str, float, float]]] = field(
        default_factory=dict)
    collision_events: int = 0
    collision_ids: Dict[str, int] = field(default_factory=dict)
    teleport_events: int = 0
    teleport_ids: List[str] = field(default_factory=list)


def iter_physics(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def absorb_row(win: PhysicsWindow, row: dict) -> None:
    t = float(row["time_s"])
    if t <= win.t0 or t > win.t0 + win.horizon_s + 1e-9:
        return
    tick = int(round((t - win.t0) / DT))
    win.ticks[tick] = {
        vid: (str(v.get("road_id", "")), str(v.get("lane_id", "")),
              float(v.get("lane_position_m", float("nan"))),
              float(v.get("speed_mps", float("nan"))))
        for vid, v in (row.get("vehicles") or {}).items()
    }
    colliding = row.get("colliding_ids") or []
    win.collision_events += len(colliding)
    for vid in colliding:
        win.collision_ids[vid] = win.collision_ids.get(vid, 0) + 1
    teleports = row.get("starting_teleports") or []
    win.teleport_events += len(teleports)
    win.teleport_ids.extend(str(v) for v in teleports)


def rows_to_window(rows: Iterator[dict], world: str, t0: float,
                   horizon_s: float = DEFAULT_HORIZON_S) -> PhysicsWindow:
    """Rows in (t0, t0 + horizon]; 0.5 s half-steps, tick = step index."""
    win = PhysicsWindow(world=world, t0=t0, horizon_s=horizon_s)
    for row in rows:
        absorb_row(win, row)
    return win


def load_window(path: Path, world: str, t0: float,
                horizon_s: float = DEFAULT_HORIZON_S) -> PhysicsWindow:
    return rows_to_window(iter_physics(path), world, t0, horizon_s)


def window_delay(win: PhysicsWindow) -> Tuple[float, int, Dict[str, float]]:
    """Total delay (s), counted steps, per-vehicle delay (s)."""
    total, steps = 0.0, 0
    per_vid: Dict[str, float] = {}
    ticks = sorted(win.ticks)
    for a, b in zip(ticks, ticks[1:]):
        if b - a != 1:
            continue
        for vid, pa in win.ticks[a].items():
            pb = win.ticks[b].get(vid)
            if pb is None:
                continue
            step = delay_step(pa, pb)
            if step is None:
                continue
            total += step
            steps += 1
            per_vid[vid] = per_vid.get(vid, 0.0) + step
    return total, steps, per_vid


def mean_delay_per_vehicle(win: PhysicsWindow) -> Tuple[float, int]:
    """(mean s/veh over tracked vehicles, tracked-vehicle count)."""
    total, _, per_vid = window_delay(win)
    n = len(per_vid)
    return (total / n) if n else 0.0, n


# --------------------------------------------------------------- compare

@dataclass
class WorldParity:
    world: str
    t0: float
    ticks_compared: int
    vehicle_ticks_compared: int
    road_mismatch_vehicle_ticks: int
    max_abs_ds_m: float
    max_abs_dv_mps: float
    laptop_mean_delay_s_per_veh: float
    platform_mean_delay_s_per_veh: float
    laptop_tracked_vehicles: int
    platform_tracked_vehicles: int
    delay_diff_pct: float
    laptop_collision_events: int
    platform_collision_events: int
    laptop_teleport_events: int
    platform_teleport_events: int
    passed: bool
    failures: List[str] = field(default_factory=list)


def compare_windows(laptop: PhysicsWindow,
                    platform: PhysicsWindow) -> WorldParity:
    if laptop.world != platform.world:
        raise ValueError(f"world mismatch: {laptop.world} vs {platform.world}")
    ticks = sorted(set(laptop.ticks) & set(platform.ticks))
    max_ds = max_dv = 0.0
    compared = mismatches = 0
    for tick in ticks:
        for vid, pa in laptop.ticks[tick].items():
            pb = platform.ticks[tick].get(vid)
            if pb is None:
                continue
            compared += 1
            if pa[0] == pb[0] and pa[1] == pb[1]:
                max_ds = max(max_ds, abs(pa[2] - pb[2]))
            else:
                mismatches += 1
            max_dv = max(max_dv, abs(pa[3] - pb[3]))
    lap_delay, lap_n = mean_delay_per_vehicle(laptop)
    plat_delay, plat_n = mean_delay_per_vehicle(platform)
    denom = abs(lap_delay)
    if denom < 1e-12:
        diff_pct = 0.0 if abs(plat_delay) < 1e-12 else math.inf
    else:
        diff_pct = 100.0 * (plat_delay - lap_delay) / denom
    failures: List[str] = []
    if not (abs(diff_pct) <= DELAY_TOL_PCT):
        failures.append(
            f"delay diff {diff_pct:.3f}% exceeds ±{DELAY_TOL_PCT}%")
    if platform.collision_events:
        failures.append(
            f"platform collisions: {platform.collision_events} events "
            f"(ids {sorted(platform.collision_ids)})")
    if platform.teleport_events:
        failures.append(
            f"platform teleports: {platform.teleport_events} events "
            f"(ids {sorted(set(platform.teleport_ids))})")
    return WorldParity(
        world=laptop.world, t0=laptop.t0, ticks_compared=len(ticks),
        vehicle_ticks_compared=compared,
        road_mismatch_vehicle_ticks=mismatches,
        max_abs_ds_m=max_ds, max_abs_dv_mps=max_dv,
        laptop_mean_delay_s_per_veh=lap_delay,
        platform_mean_delay_s_per_veh=plat_delay,
        laptop_tracked_vehicles=lap_n, platform_tracked_vehicles=plat_n,
        delay_diff_pct=diff_pct,
        laptop_collision_events=laptop.collision_events,
        platform_collision_events=platform.collision_events,
        laptop_teleport_events=laptop.teleport_events,
        platform_teleport_events=platform.teleport_events,
        passed=not failures, failures=failures)


# ---------------------------------------------------------------- inputs

def check_input_path(rel: str, world: str, split: str) -> None:
    low = rel.replace("\\", "/").lower()
    for token in FORBIDDEN_TOKENS:
        if token in low:
            raise InputRefused(f"{world}: forbidden input file {rel!r}")
    if split != "TRAIN":
        raise InputRefused(
            f"{world}: only TRAIN worlds are eligible (got {split!r})")


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs(inputs: Mapping, root: Path) -> None:
    """Verify every listed file exists under root with its sha256."""
    for world in inputs["worlds"]:
        check_input_path(world["reference_physics"], world["world"],
                         world["split"])
        for entry in world.get("files", []):
            check_input_path(entry["path"], world["world"], world["split"])
            got = sha256_file(root / entry["path"])
            if got != entry["sha256"]:
                raise InputRefused(
                    f"{world['world']}: sha256 mismatch for "
                    f"{entry['path']}: {got}")


def load_inputs(path: Path) -> dict:
    inputs = json.loads(Path(path).read_text(encoding="utf-8"))
    worlds = inputs.get("worlds") or []
    if len(worlds) != 2:
        raise InputRefused(
            f"expected exactly 2 parity worlds, got {len(worlds)}")
    for world in worlds:
        check_input_path(world["reference_physics"], world["world"],
                         world["split"])
    return inputs


# ---------------------------------------------------------------- runner

def load_runner(spec: str) -> Callable:
    """'package.module:function' -> the runner callable."""
    mod_name, _, func_name = spec.partition(":")
    if not mod_name or not func_name:
        raise ValueError("--runner must look like module:function")
    mod = importlib.import_module(mod_name)
    return getattr(mod, func_name)


def run_platform_side(runner: Callable, world: Mapping, run_dir: Path,
                      horizon_s: float) -> Path:
    """Delegate ONE 60 s simulation to the injected box-side runner."""
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"{world['world']}__native_parity" / "physics_steps.jsonl"
    if out.is_file():
        raise InputRefused(
            f"refusing to overwrite an existing platform run: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    scenario_dir = Path(world["scenario_dir"])
    state_xml = Path(world["state_xml"])
    runner(scenario_dir=scenario_dir, state_xml=state_xml,
           duration_s=float(horizon_s), out_jsonl=out)
    return out


# ------------------------------------------------------------------- CLI

def build_report(inputs: Mapping, laptop_root: Path,
                 platform_root: Optional[Path] = None,
                 horizon_s: float = DEFAULT_HORIZON_S,
                 platform_paths: Optional[Dict[str, Path]] = None) -> dict:
    worlds_out = []
    all_pass = True
    for world in inputs["worlds"]:
        name = world["world"]
        laptop = load_window(laptop_root / world["reference_physics"], name,
                             float(world["t0_s"]), horizon_s)
        if platform_paths is not None:
            plat_path = platform_paths[name]
        else:
            plat_rel = world.get("platform_physics") or \
                world["reference_physics"]
            plat_path = Path(platform_root) / plat_rel
        platform = load_window(plat_path, name, float(world["t0_s"]),
                               horizon_s)
        res = compare_windows(laptop, platform)
        all_pass = all_pass and res.passed
        worlds_out.append(res.__dict__)
    return {
        "schema": "inc1_parity_smoke/1",
        "purpose": inputs.get("purpose"),
        "horizon_s": horizon_s,
        "delay_tol_pct": DELAY_TOL_PCT,
        "pass_rule": "delay within ±2% AND platform collisions == 0 "
                     "AND platform teleports == 0",
        "passed": all_pass,
        "worlds": worlds_out,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="WM-N1-INC1 §7 parity smoke (comparison only; the "
                    "simulation itself is delegated to --runner on the box)")
    parser.add_argument("--inputs", required=True,
                        help="PARITY_INPUTS.json (2 chunk-4 TRAIN worlds)")
    parser.add_argument("--laptop-root", required=True,
                        help="root containing the recorded reference files")
    parser.add_argument("--platform-root",
                        help="root containing the platform physics rows")
    parser.add_argument("--runner", help="module:function delegated the "
                                         "60 s run on the Linux box")
    parser.add_argument("--run-dir",
                        help="where the runner writes platform physics")
    parser.add_argument("--horizon", type=float, default=DEFAULT_HORIZON_S)
    parser.add_argument("--out", help="write the JSON report here")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip input sha verification (not recommended)")
    args = parser.parse_args(argv)

    inputs = load_inputs(Path(args.inputs))
    laptop_root = Path(args.laptop_root)
    if not args.no_verify:
        verify_inputs(inputs, laptop_root)
        print("input shas verified OK")
    if args.runner and not (args.run_dir and args.platform_root):
        parser.error("--runner needs --run-dir and --platform-root")

    platform_paths: Dict[str, Path] = {}
    if args.runner:
        runner = load_runner(args.runner)
        run_dir = Path(args.run_dir)
        for world in inputs["worlds"]:
            scenario_dir = laptop_root / world["scenario_dir"]
            state_xml = laptop_root / world["state_xml"]
            spec_world = dict(world, scenario_dir=str(scenario_dir),
                              state_xml=str(state_xml))
            out = run_platform_side(runner, spec_world, run_dir,
                                    args.horizon)
            platform_paths[world["world"]] = out
            print(f"runner wrote {out}")
    elif not args.platform_root:
        for world in inputs["worlds"]:
            print(f"plan: {world['world']} (TRAIN, t0={world['t0_s']}): "
                  f"restore {world['state_xml']} into copy "
                  f"{world['scenario_dir']}, run {args.horizon:.0f} s")
        print("ZERO SUMO: no --runner given; nothing was simulated")
        return 2

    report = build_report(
        inputs, laptop_root,
        platform_root=None if args.runner else Path(args.platform_root),
        horizon_s=args.horizon,
        platform_paths=platform_paths if args.runner else None)
    text = json.dumps(report, indent=1)
    if args.out:
        out_path = Path(args.out)
        if out_path.exists():
            raise InputRefused(f"refusing to overwrite {out_path}")
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
