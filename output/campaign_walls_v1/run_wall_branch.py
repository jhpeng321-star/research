"""WM-59 WALL-1 child: one restored wall branch per fresh SUMO process.

Mirrors ``campaign_stage1_v3.run_chunk.run_branch_child`` (imported
read-only): resolve the planned run, refuse an existing run dir, START
SUMO from the BASE run's own scenario (common random numbers) unless an
api is injected, install the one-loadState guard, restore the base t0
saved state (EXACTLY ONE ``loadState`` - a second call raises), wire the
frozen dispatcher stack (or accept injected fakes), drive the wall arm
(``campaign_walls_v1.wall_driver``) and write the write-once receipt.

ZERO SUMO here: every code path runs on injected fakes in tests; the
real SUMO start (``branch_sumo_session``) is imported lazily exactly as
in the v3 child and is exercised by the integrator's registered smoke.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from campaign_stage1_v3.common import (  # noqa: E402 - path bootstrap above
    check_safe_name,
    load_json,
    utc_now,
    write_json,
)
from campaign_stage1_v3.run_chunk import (  # noqa: E402
    RunnerError,
    branch_route_departures,
    start_branch_sumo_with_lock_retry,
)
from campaign_walls_v1.common import WallsPilotError  # noqa: E402
from campaign_walls_v1.wall_driver import (  # noqa: E402
    install_load_state_guard,
    drive_wall_branch,
)

RECEIPT_SCHEMA = "wm59_wall_branch_receipt/1"


class WallChildError(WallsPilotError, RunnerError):
    """A wall-branch child contract violation."""


def resolve_run(plan: dict, run_name: str) -> tuple[dict, dict]:
    """(world, arm) for a planned run name; loud on any mismatch."""
    world = next((w for w in plan["worlds"] if any(
        b["run"] == run_name for b in w["branches"])), None)
    if world is None:
        raise WallChildError(f"run {run_name!r} is not in the plan")
    arm = next(a for a in world["branches"] if a["run"] == run_name)
    if arm.get("mode") != "restore_wall":
        raise WallChildError(
            f"run {run_name!r} is not a wall branch (mode "
            f"{arm.get('mode')!r})")
    return world, arm


def run_wall_branch_child(run, plan, runs_root, registration_path, *,
                          bases_root=None, session=None, ownership=None,
                          sensor=None, api=None,
                          horizon_ticks=None) -> dict:
    """One restored wall branch: fresh process, one loadState, one drive."""
    name = check_safe_name(run["run"], "run name")
    world, arm = resolve_run(plan, name)
    target = Path(runs_root) / name
    if target.exists():
        raise WallChildError(f"refusing to write into existing directory: {target}")
    target.mkdir(parents=True)
    run_dir = target / "run"
    run_dir.mkdir()
    # the branch RESTORES from the world's chunk bases root (a plan field
    # per world; the CLI flag wins) while writing to runs_root
    if bases_root is not None:
        base_root = Path(bases_root)
    elif world.get("bases_root"):
        base_root = Path(world["bases_root"])
    else:
        base_root = Path(runs_root)
    base_world_dir = base_root / arm["restores_state"]["from_run"]
    owns_api = False
    if api is None:
        from campaign_pilot_v1.run_pilot import branch_sumo_session

        # WM-64: SUMO starts from a REWRITTEN COPY of the base scenario
        # inside this branch's run dir; the recorded base stays read-only.
        api, _command_record = start_branch_sumo_with_lock_retry(
            branch_sumo_session, base_world_dir, world["seed"], world,
            copy_dir=run_dir / "scenario"
        )
        owns_api = True
    try:
        return _run_wall_branch_child_on_api(
            run, world, arm, target, run_dir, base_world_dir, api,
            registration_path, session=session, ownership=ownership,
            sensor=sensor, horizon_ticks=horizon_ticks
        )
    finally:
        if owns_api:
            close = getattr(api, "close", None)
            if callable(close):
                close()


def _run_wall_branch_child_on_api(run, world, arm, target, run_dir,
                                  base_world_dir, api, registration_path, *,
                                  session=None, ownership=None, sensor=None,
                                  horizon_ticks=None) -> dict:
    """The child body once a traci api is bound (see above)."""
    from campaign_pilot_v1.saved_state import (
        load_branch_state,
        parse_sumo_version,
    )
    from campaign_stage1_v3.run_chunk import default_dispatcher_components

    load_state_guard = install_load_state_guard(api)
    restore = load_branch_state(
        api,
        base_world_dir,
        expected_t_b=float(arm["restores_state"]["at_s"]),
        sumo_version=parse_sumo_version(api.getVersion()),
        scenario_digests={},
    )
    route_departures = branch_route_departures(base_world_dir / "run" / "scenario")
    roster = set(restore["manifest"]["harness_state"]["roster"]["cav_ids"])
    if session is None and ownership is None and sensor is None:
        # NO-INJECTION DEFAULT: the REAL frozen dispatcher stack, the same
        # wiring every restored stage-1 branch ran (WM-40 session +
        # CommandOwnership mode 27 + SumoLocalSensor)
        session, ownership, sensor = default_dispatcher_components(api, roster)
    elif session is None or ownership is None:
        raise WallChildError(
            "inject either nothing (the real frozen stack) or ALL of "
            "session + ownership (+ optional sensor); a partial injection "
            "would mix real and fake halves")
    result = drive_wall_branch(
        api,
        world,
        arm,
        run_dir,
        restore,
        session=session,
        ownership=ownership,
        sensor=sensor,
        route_departures=route_departures,
        horizon_ticks=horizon_ticks,
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run": run["run"],
        "world": world["world"],
        "arm": arm["arm"],
        "branch_seed": arm.get("branch_seed"),
        "saved_state_id": restore["saved_state_id"],
        "bases_root": str(base_world_dir.parent),
        "native_sibling_run": arm.get("native_sibling_run"),
        "horizon_mode": ("smoke_cap" if horizon_ticks is not None
                         else "registered_full"),
        "cav_speed_mode": result.get("cav_speed_mode"),
        "observation_unavailable_rows": result.get("observation_unavailable_rows"),
        "load_state_calls": load_state_guard["calls"],
        "exit_status": "complete",
        "finished_utc": utc_now(),
        "wall_summary": result,
    }
    write_json(target / "RUN_RECEIPT.json", receipt)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exec_branch = sub.add_parser(
        "_exec_wall_branch", help="launcher-dispatched child (one branch)")
    exec_branch.add_argument("--run", required=True)
    exec_branch.add_argument("--plan", type=Path, required=True)
    exec_branch.add_argument("--runs-root", type=Path, required=True)
    exec_branch.add_argument("--registration", type=Path, required=True)
    exec_branch.add_argument("--bases-root", type=Path, default=None)
    exec_branch.add_argument("--horizon-ticks", type=float, default=None)
    args = parser.parse_args(argv)
    plan = load_json(Path(args.plan))
    run = {"run": args.run}
    receipt = run_wall_branch_child(
        run, plan, Path(args.runs_root), Path(args.registration),
        bases_root=args.bases_root, horizon_ticks=args.horizon_ticks,
    )
    print(json.dumps({"run": receipt["run"], "exit_status": receipt["exit_status"],
                      "wall_outcome": receipt["wall_summary"]["wall_outcome"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
