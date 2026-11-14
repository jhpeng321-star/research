"""A1.5 harness-equivalence smoke: driver path vs INC1 runner path.

R4 runs through the campaign driver path while R0-R3 run through the INC1
runner loop, so before any INC1 world this smoke compares the two paths with
NO wall: the driver path with the proposer forced to refuse (no feasible pair)
vs the INC1 runner R0, from the same warm-up state, 300 s each.

PASS = identical per-vehicle trajectories OR aggregate delay within 0.5 %,
with 0 collisions and 0 teleports.  FAIL is a blocker (A1.6 gates the launcher
on this verdict).

Zero SUMO in this build: the comparison is pure; the box launcher feeds the
two recorded ``physics_steps.jsonl`` run directories.
"""
from __future__ import annotations

import argparse
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

SCHEMA = "wm_n1_inc1_harness_equivalence/1"
VERDICT_NAME = "HARNESS_EQUIVALENCE_VERDICT.json"
DELAY_TOL = 0.005
DT_S = C.PHYSICS_DT_S


def verdict_path(out_dir) -> Path:
    return Path(out_dir) / VERDICT_NAME


def load_rows(path) -> list[dict]:
    """Load a ``physics_steps.jsonl`` (or a run dir containing one)."""
    path = Path(path)
    if path.is_dir():
        candidate = path / "physics_steps.jsonl"
        if not candidate.is_file():
            candidate = path / "run" / "physics_steps.jsonl"
        path = candidate
    if not path.is_file():
        raise FileNotFoundError(f"physics rows not found: {path}")
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def vehicle_tracks(rows: list[dict]) -> dict:
    """``vid -> [(time_s, lane_id, lane_position_m, speed_mps), ...]``."""
    tracks: dict = {}
    for row in rows:
        now = round(float(row["time_s"]), 6)
        for vid, state in (row.get("vehicles") or {}).items():
            tracks.setdefault(str(vid), []).append((
                now, state.get("lane_id"), round(
                    float(state.get("lane_position_m", 0.0)), 6),
                round(float(state.get("speed_mps", 0.0)), 6)))
    return tracks


def tracks_identical(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for vid in a:
        if a[vid] != b[vid]:
            return False
    return True


def max_speed(rows: list[dict]) -> float:
    top = 0.0
    for row in rows:
        for state in (row.get("vehicles") or {}).values():
            top = max(top, float(state.get("speed_mps", 0.0)))
    return top


def aggregate_delay(rows: list[dict], free_speed: float) -> float:
    """A simple aggregate lost-time proxy (s): ``sum((v_free - v) * dt)``.

    ``v_free`` is the common free-flow reference (the max speed seen across
    BOTH compared runs), so a uniformly slower path is detected.  Both paths
    are scored by the same formula.
    """
    free = float(free_speed)
    lost = 0.0
    for row in rows:
        for state in (row.get("vehicles") or {}).values():
            lost += max(0.0, free - float(state.get("speed_mps", 0.0))) * DT_S
    return round(lost, 9)


def _count_teleports(rows: list[dict]) -> int:
    seen = set()
    for row in rows:
        for key in ("starting_teleports", "ending_teleports"):
            for vid in row.get(key) or ():
                seen.add(str(vid))
    return len(seen)


def _count_collisions(rows: list[dict]) -> int:
    total = 0
    for row in rows:
        total += len(row.get("collisions") or ())
    return total


def evaluate(driver_rows: list[dict], r0_rows: list[dict], *,
             delay_tol: float = DELAY_TOL) -> dict:
    """The A1.5 verdict for two recorded runs (pure)."""
    a = vehicle_tracks(driver_rows)
    b = vehicle_tracks(r0_rows)
    identical = tracks_identical(a, b)
    ref = max(max_speed(driver_rows), max_speed(r0_rows))
    delay_a = aggregate_delay(driver_rows, ref)
    delay_b = aggregate_delay(r0_rows, ref)
    denom = max(abs(delay_b), 1e-9)
    rel = abs(delay_a - delay_b) / denom
    teleports = _count_teleports(driver_rows) + _count_teleports(r0_rows)
    collisions = _count_collisions(driver_rows) + _count_collisions(r0_rows)
    delay_ok = bool(identical or rel <= float(delay_tol))
    passed = bool(delay_ok and teleports == 0 and collisions == 0)
    return {
        "schema": SCHEMA,
        "verdict": "PASS" if passed else "FAIL",
        "identical_trajectories": bool(identical),
        "delay_driver_s": delay_a,
        "delay_runner_s": delay_b,
        "delay_rel_diff": rel,
        "delay_tol": float(delay_tol),
        "teleports": teleports,
        "collisions": collisions,
        "driver_vehicles": len(a),
        "runner_vehicles": len(b),
        "budget_sim_s": float(C.BUDGET_HARNESS_EQUIV_S),
    }


def write_verdict(out_dir, payload: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = verdict_path(out_dir)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def verdict_is_pass(path) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return record.get("schema") == SCHEMA and record.get("verdict") == "PASS"


def compare_run_dirs(driver_run, r0_run, *, out_dir=None,
                     delay_tol: float = DELAY_TOL) -> dict:
    driver_rows = load_rows(driver_run)
    r0_rows = load_rows(r0_run)
    payload = evaluate(driver_rows, r0_rows, delay_tol=delay_tol)
    if out_dir is not None:
        write_verdict(out_dir, payload)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="inc1.harness_equivalence",
        description="A1.5 driver-path vs INC1-runner-R0 equivalence smoke")
    parser.add_argument("--driver-run", required=True,
                        help="driver-path run dir (proposer forced to refuse)")
    parser.add_argument("--r0-run", required=True, help="INC1 runner R0 dir")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    payload = compare_run_dirs(args.driver_run, args.r0_run,
                               out_dir=args.out_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
