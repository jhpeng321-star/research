"""Closed-loop SUMO runner for WM-N1-REPL-CL-20261005 Part C (one run / process).

Restores a world's saved t0 state exactly like a campaign branch arm
(``campaign_stage1_v3.run_chunk.run_branch_child``: SUMO from a rewritten
scenario COPY of the base run, ``load_branch_state`` loadState + identity
checks, the frozen dispatcher stack ``default_dispatcher_components`` =
MixedAssignmentSession + CommandOwnership (speed mode 27) + SumoLocalSensor,
the t0 release of every live roster CAV), then drives the branch loop of
``campaign_stage1_v3.branch_driver.drive_dispatcher_branch`` (1 s control
ticks, two 0.5 s physics substeps, per-substep physics rows / teleports /
insertions / dispatcher rows, speed-mode-27 guard) with ONE change: every
``decision_period_s`` (60 s, from t0) the POLICY chooses the per-CAV
assignment from the RSU-observable live state, which is applied through
``session.apply_plan`` (native = override released; damping / PI = the real
controller instances). Outputs use the campaign layout so the existing
labeler reads them: ``<target>/run/physics_steps.jsonl`` (gzipped to
``.gz`` after the receipt), ``insertions.jsonl``, ``suffix_commands.jsonl``,
``dispatcher_rows.jsonl``, ``branch_record.json``, ``outcome_record.json``
and ``<target>/RUN_RECEIPT.json``; plus ``decisions.jsonl`` (every
decision: candidates, groups, predictions, choice, timing) and
``closed_loop_schedule.json``.

Code roots: the campaign packages and ``traffic_shepherd`` are imported from
``--code-root`` (default ``E:/research-dense``, the tree Part R runs on, so
restore/dispatch/controllers are byte-identical to the campaign arms); the
A11 packages (features, rollout2, ctrl, model, evalh, ...) and this package
from this worktree's ``output/wd01/wm_n1``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

WM_N1_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODE_ROOT = Path(os.environ.get("WM_N1_CODE_ROOT") or "E:/research-dense")
DECISION_PERIOD_S = 60.0
CLOSED_LOOP_SCHEMA = "wm_n1_closed_loop/1"


def setup_paths(code_root: Path) -> None:
    for entry in (str(WM_N1_ROOT), str(Path(code_root) / "output"), str(code_root)):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    # final order: code_root, code_root/output, wm_n1 (A11 packages)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    return repr(o)


def write_json(path: Path, data) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def gzip_physics(physics: Path):
    """``<name>.gz`` next to the plain log (same as repl_driver.gzip_physics)."""
    if not physics.is_file():
        return None
    gz = Path(str(physics) + ".gz")
    if not gz.is_file():
        with open(physics, "rb") as src, gzip.open(gz, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
    return {"path": gz.name, "bytes": gz.stat().st_size}


def code_identity(code_root: Path) -> dict:
    files = {}
    for rel in ("traffic_shepherd/planner/loop.py", "traffic_shepherd/planner/grouping.py",
                "traffic_shepherd/planner/rules.py", "traffic_shepherd/planner/safety.py",
                "traffic_shepherd/planner/config.py", "traffic_shepherd/planner/interface.py",
                "traffic_shepherd/mixed_assignment.py", "traffic_shepherd/ownership.py",
                "output/campaign_stage1_v3/branch_driver.py",
                "output/campaign_stage1_v3/run_chunk.py"):
        p = Path(code_root) / rel
        files[rel] = sha256_file(p) if p.is_file() else None
    for rel in ("closedloop/runner.py", "closedloop/policies.py", "closedloop/a11_predictor.py",
                "closedloop/live_state.py", "closedloop/wm_s_predictor.py"):
        p = WM_N1_ROOT / rel
        files["wm_n1/" + rel] = sha256_file(p) if p.is_file() else None
    return files


# ------------------------------------------------------------------ drive
def drive_closed_loop(api, world: dict, run_name: str, run_dir: Path, restore: dict, *,
                      session, ownership, sensor, policy, route_departures: dict,
                      t0_s: float, horizon_s: float, decision_period_s: float,
                      base_world_dir: Path, model=None, log=print) -> dict:
    """``drive_dispatcher_branch`` with a policy decision every period."""
    from campaign_pilot_v1.branch_types import dispatcher_rows_from_telemetry, make_plan
    from campaign_pilot_v1.logging_schema import BOUNDARY_EDGES_FALLBACK
    from campaign_pilot_v1.suffix_runner import build_physics_row
    from campaign_stage1_v3.branch_driver import (
        BranchDriverError,
        _capture_frames,
        entrance_of,
    )
    from campaign_stage1_v3.speed_mode27 import (
        CAV_SPEED_MODE,
        SpeedMode27Guard,
        SpeedMode27GuardError,
    )

    from closedloop.live_state import DepartureHistory, live_state_from_row
    from closedloop.policies import DecisionContext

    run_dir = Path(run_dir)
    manifest = restore["manifest"]
    cav_ids = set(manifest["harness_state"]["roster"]["cav_ids"])
    if not cav_ids:
        raise BranchDriverError("restored roster is empty")
    insertions_path = run_dir / "insertions.jsonl"
    insertions_path.touch()
    decisions_path = run_dir / "decisions.jsonl"
    decisions_path.touch()
    teleports_seen: set = set()
    collisions_seen: list = []
    previous_row = manifest.get("physics_row_at_t0") or None
    if previous_row is None or abs(float(previous_row["time_s"]) - t0_s) > 1e-6:
        raise BranchDriverError("the saved state carries no physics_row_at_t0 at t0 "
                                "(the decision-0 RSU state); refusing")
    boundary_edges = dict(BOUNDARY_EDGES_FALLBACK)
    boundary_edges.update({k: str(v) for k, v in manifest["harness_state"]["roster"]
                           .get("boundary_edges", {}).items()})
    telemetry_sink: list = []
    bind = getattr(session, "bind_emit", None)
    if callable(bind):
        bind(lambda kind, row: telemetry_sink.append(row))
    pi_session = getattr(session, "pi_session", None)
    pi_retired: set = set()
    rows_written = 0
    dispatcher_rows_written = 0
    ownership_mode = getattr(ownership, "cav_speed_mode", None)
    guard = SpeedMode27Guard(api, cav_ids) if ownership_mode == CAV_SPEED_MODE else None
    unavailable: list = []
    departures = DepartureHistory(base_world_dir, t0_s)
    t0_release = []
    for cav in sorted(cav_ids & set(api.vehicle.getIDList())):
        api.vehicle.setSpeed(cav, -1)
        t0_release.append(cav)
        append_jsonl(run_dir / "suffix_commands.jsonl", {
            "kind": "t0_release", "timestamp_s": t0_s, "vehicle_id": cav,
            "owner": "native", "issued_target_speed_mps": None,
            "fallback_reason": "t0_release"})
    current = {c: "native" for c in cav_ids}      # executed assignment (released)
    plan_id_seen = None
    schedule_events = []
    decision_times = []
    n_decisions = 0
    decision_wall_s = []
    k_period = int(round(decision_period_s))
    for tick in range(int(round(t0_s)), int(round(horizon_s))):
        now = float(api.simulation.getTime())
        if abs(now - tick * 1.0) > 1e-6:
            raise BranchDriverError(f"time grid mismatch at t={now}")
        if (tick - int(round(t0_s))) % k_period == 0:
            t_dec = time.perf_counter()
            if abs(float(previous_row["time_s"]) - now) > 1e-6:
                raise BranchDriverError(
                    f"decision at {now}: latest physics row is {previous_row['time_s']}")
            state = live_state_from_row(previous_row, departures)
            live_cavs = tuple(sorted(state.cav_slots))
            ctx = DecisionContext(
                world=world["world"], seed=int(world["seed"]), t0_s=t0_s, now_s=now,
                decision_index=n_decisions, state=state, live_cavs=live_cavs,
                prev_assignments={c: current.get(c, "native") for c in live_cavs},
                model=model)
            out = policy.decide(ctx)
            if out.walls and not hasattr(policy, "apply"):
                raise BranchDriverError("policy returned walls without an apply hook")
            bad = {c: a for c, a in out.assignments.items()
                   if a not in ("native", "damping", "pi-external-u") or c not in cav_ids}
            if bad:
                raise BranchDriverError(f"invalid assignments {list(bad.items())[:5]}")
            assignments = {c: "native" for c in cav_ids}   # untracked/absent -> native
            assignments.update(out.assignments)
            plan_id = f"{run_name}:d{n_decisions}@{now:g}"
            if hasattr(policy, "apply"):
                policy.apply(session, ctx=ctx, out=out, assignments=assignments,
                             plan_id=plan_id, now_s=now)
            else:
                session.apply_plan(make_plan(assignments, plan_id), now_s=now)
            changed = sorted(c for c in cav_ids if assignments[c] != current.get(c))
            for c in changed:
                schedule_events.append({"vehicle_id": c, "time_s": round(now, 6),
                                        "action": assignments[c]})
            current = assignments
            plan_id_seen = now
            dt_dec = time.perf_counter() - t_dec
            decision_wall_s.append(round(dt_dec, 3))
            decision_times.append(now)
            counts = {}
            for c in live_cavs:
                counts[assignments[c]] = counts.get(assignments[c], 0) + 1
            append_jsonl(decisions_path, {
                "schema": CLOSED_LOOP_SCHEMA, "run": run_name, "world": world["world"],
                "policy": getattr(policy, "name", type(policy).__name__),
                "decision_index": n_decisions, "time_s": now,
                "n_live_cavs": len(live_cavs), "n_tracked": int(state.meta["n_present"]),
                "n_dropped_slot_vehicles": state.n_dropped,
                "action_counts_live": counts, "plan_id": plan_id,
                "assignments_live": {c: assignments[c] for c in live_cavs},
                "wall_s": round(dt_dec, 3), "record": out.record})
            log(f"[decision {n_decisions} t={now:g}] live_cavs={len(live_cavs)} "
                f"counts={counts} wall_s={dt_dec:.1f}")
            n_decisions += 1
        live = list(api.vehicle.getIDList())
        frames = _capture_frames(sensor, live, cav_ids, now, run_dir, unavailable)
        tick_sink_mark = len(telemetry_sink)
        records = session.step(ownership, frames, now_s=now, live_ids=live)
        if guard is not None:
            try:
                guard.check(now)
            except SpeedMode27GuardError as exc:
                raise BranchDriverError(str(exc)) from exc
        for record in records:
            append_jsonl(run_dir / "suffix_commands.jsonl", {
                "kind": "command", "timestamp_s": now, "vehicle_id": record.vehicle_id,
                "owner": record.owner,
                "issued_target_speed_mps": record.issued_target_speed_mps,
                "fallback_reason": record.fallback_reason})
        if pi_session is not None:
            pi_session.flush_evidence(
                lambda payload: append_jsonl(run_dir / "suffix_commands.jsonl", payload),
                finalize=False)
            pi_session.commit_evidence()
        pi_live = set(live)
        for _substep in range(2):
            api.simulationStep()
            after = float(api.simulation.getTime())
            if pi_session is not None:
                live_after = set(api.vehicle.getIDList())
                disappeared_after = (pi_live - live_after) & cav_ids
                arrived_after = set(api.simulation.getArrivedIDList()) & cav_ids
                pi_retired.update(disappeared_after | arrived_after)
                reappeared_after = pi_retired & live_after
                pi_retired.difference_update(reappeared_after)
                pi_session.observe_lifecycle(live_ids=live_after,
                                             disappeared_ids=disappeared_after,
                                             id_reuse_ids=reappeared_after)
                pi_live = live_after
            row = build_physics_row(api, after, previous_row=previous_row, cav_ids=cav_ids,
                                    cav_actions={}, boundary_edges=boundary_edges,
                                    live_by_tick=None)
            starting_tp = sorted(str(v) for v in api.simulation.getStartingTeleportIDList())
            ending_tp = sorted(str(v) for v in api.simulation.getEndingTeleportIDList())
            teleports_seen.update(starting_tp)
            row["starting_teleports"] = starting_tp
            row["ending_teleports"] = ending_tp
            if row.get("colliding_ids"):
                collisions_seen.append({"time_s": after, "ids": list(row["colliding_ids"])})
            append_jsonl(run_dir / "physics_steps.jsonl", row)
            departures.add_row(row)
            rows_written += 1
            previous_row = row
            telemetry = telemetry_sink[tick_sink_mark:]
            for dispatch_row in dispatcher_rows_from_telemetry(
                    telemetry, timestamp_s=after, plan_id=plan_id_seen):
                append_jsonl(run_dir / "dispatcher_rows.jsonl", dispatch_row)
                dispatcher_rows_written += 1
            for vehicle_id in sorted(str(v) for v in row.get("departed_ids") or []):
                route_info = (route_departures or {}).get(vehicle_id, {})
                try:
                    speed = float(api.vehicle.getSpeed(vehicle_id))
                except Exception:  # noqa: BLE001
                    speed = None
                insertion = {
                    "vehicle_id": vehicle_id,
                    "scheduled_depart_s": route_info.get("depart_s"),
                    "inserted_s": after,
                    "entrance": entrance_of(route_info.get("route", "")),
                    "depart_speed_mps": speed,
                    "is_cav": vehicle_id in cav_ids,
                    "type_id": route_info.get("type_id"),
                }
                append_jsonl(insertions_path, insertion)
    schedule = {"schema": CLOSED_LOOP_SCHEMA, "arm": getattr(policy, "name", "?"),
                "t0_s": t0_s, "horizon_s": horizon_s, "decision_period_s": decision_period_s,
                "decision_times_s": decision_times, "events": schedule_events}
    write_json(run_dir / "closed_loop_schedule.json", schedule)
    write_json(run_dir / "branch_record.json", {
        "schema": "wm44_branch_record/1", "run_kind": "closed_loop",
        "root_id": world["world"], "branch_id": run_name,
        "branch_type": getattr(policy, "name", "?"), "t0_s": t0_s, "seed": world["seed"],
        "saved_state_id": restore["saved_state_id"], "teleports": len(teleports_seen),
        "t0_release_count": len(t0_release), "cav_speed_mode": ownership_mode,
        "observation_unavailable_rows": len(unavailable), "created_utc": utcnow()})
    write_json(run_dir / "outcome_record.json", {
        "schema": "wm44_branch_outcome/1", "branch_id": run_name,
        "teleports": len(teleports_seen), "created_utc": utcnow()})
    return {
        "rows_written": rows_written, "dispatcher_rows_written": dispatcher_rows_written,
        "decisions": n_decisions, "decision_wall_s": decision_wall_s,
        "saved_state_id": restore["saved_state_id"], "t0_release_count": len(t0_release),
        "cav_speed_mode": ownership_mode, "observation_unavailable_rows": len(unavailable),
        "speed_mode_guard_checked": guard.checked if guard else 0,
        "speed_mode_guard_violations": len(guard.violations) if guard else 0,
        "teleports": len(teleports_seen), "collision_rows": len(collisions_seen),
        "collision_vehicle_events": sum(len(c["ids"]) for c in collisions_seen),
    }


# ------------------------------------------------------------------ child
def run_closed_loop(*, world_name: str, policy_name: str, plan_path: Path, bases_root: Path,
                    target: Path, horizon_s: float, decision_period_s: float,
                    code_root: Path, registration: Path | None, threads: int,
                    policy_modules=(), policy_kwargs=None, log=print) -> dict:
    setup_paths(code_root)
    import importlib

    import torch

    torch.set_num_threads(max(1, int(threads)))
    from campaign_stage1_v3 import run_chunk as RC
    from campaign_pilot_v1.run_pilot import branch_sumo_session
    from campaign_pilot_v1.saved_state import load_branch_state, parse_sumo_version

    from closedloop import policies as P

    for mod in policy_modules:
        importlib.import_module(mod)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    world = next(w for w in plan["worlds"] if w["world"] == world_name)
    t0_s = float(world["t0_s"])
    base_world_dir = Path(bases_root) / f"{world_name}__base"
    if not (base_world_dir / "saved_state" / "STATE_MANIFEST.json").is_file():
        raise RC.RunnerError(f"base saved state missing: {base_world_dir}")
    target = Path(target)
    if target.exists():
        raise RC.RunnerError(f"refusing to write into existing directory: {target}")
    target.mkdir(parents=True)
    run_dir = target / "run"
    run_dir.mkdir()
    run_name = f"{world_name}__cl_{policy_name}"
    started = utcnow()
    write_json(target / "RUN_STARTED.json", {
        "schema": CLOSED_LOOP_SCHEMA, "run": run_name, "world": world_name,
        "policy": policy_name, "started_utc": started, "pid": os.getpid(),
        "plan": str(plan_path), "bases_root": str(bases_root),
        "code_root": str(code_root), "wm_n1_root": str(WM_N1_ROOT)})
    model = P.ModelHandle()
    policy = P.make_policy(policy_name, **dict(policy_kwargs or {}))
    if getattr(policy, "uses_model", False):
        t = time.perf_counter()
        _ = model.checkpoint
        log(f"A11 loaded in {time.perf_counter() - t:.1f}s sha={model.checkpoint.sha256}")
    api, _cmd = RC.start_branch_sumo_with_lock_retry(
        branch_sumo_session, base_world_dir, world["seed"], world, copy_dir=run_dir / "scenario")
    t_start = time.perf_counter()
    receipt = {
        "schema": "wm44_branch_receipt/1", "run_kind": "closed_loop",
        "closed_loop_schema": CLOSED_LOOP_SCHEMA, "run": run_name, "world": world_name,
        "arm": f"cl_{policy_name}", "policy": policy_name, "split": world.get("split"),
        "t0_s": t0_s, "horizon_s": horizon_s, "decision_period_s": decision_period_s,
        "registration_sha256": (sha256_file(registration) if registration else None),
        "model_sha256": (model.checkpoint.sha256 if getattr(policy, "uses_model", False) else None),
        "code_identity": code_identity(code_root), "started_utc": started,
        "policy_kwargs": dict(policy_kwargs or {}),
        "device": os.environ.get("WM_N1_DEVICE", "cpu"),
        "model_path": os.environ.get("WM_N1_A11_MEMBER"),
        "climatology": os.environ.get("WM_N1_PAIRED_EFFECT_CLIMATOLOGY"),
        "torch_threads": int(threads),
    }
    try:
        copy_check = RC.verify_scenario_copy(run_dir / "scenario", base_world_dir)
        restore = load_branch_state(api, base_world_dir, expected_t_b=t0_s,
                                    sumo_version=parse_sumo_version(api.getVersion()),
                                    scenario_digests={})
        route_departures = RC.branch_route_departures(base_world_dir / "run" / "scenario")
        roster = set(restore["manifest"]["harness_state"]["roster"]["cav_ids"])
        session, ownership, sensor = RC.default_dispatcher_components(api, roster)
        result = drive_closed_loop(
            api, world, run_name, run_dir, restore, session=session, ownership=ownership,
            sensor=sensor, policy=policy, route_departures=route_departures, t0_s=t0_s,
            horizon_s=t0_s + float(horizon_s), decision_period_s=float(decision_period_s),
            base_world_dir=base_world_dir, model=model, log=log)
        rows = int(result["rows_written"])
        expected = int(round(2.0 * float(horizon_s)))
        receipt.update({
            "saved_state_id": restore["saved_state_id"],
            "cav_speed_mode": result.get("cav_speed_mode"),
            "physics_rows_written": rows, "physics_rows_expected": expected,
            "executed_time_s": round(rows * 0.5, 6), "scenario_copy_check": copy_check,
            "suffix_summary": result, "wall_s": round(time.perf_counter() - t_start, 1),
            "finished_utc": utcnow(),
            "exit_status": "complete" if rows == expected else "failed",
        })
        if rows != expected:
            receipt["error"] = f"physics_rows_written {rows} != {expected}"
    except BaseException as exc:  # noqa: BLE001 - failures are kept, never retried
        receipt.update({"exit_status": "failed", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-6000:],
                        "finished_utc": utcnow(),
                        "wall_s": round(time.perf_counter() - t_start, 1)})
        write_json(target / "RUN_RECEIPT.json", receipt)
        raise
    finally:
        try:
            api.close()
        except Exception:  # noqa: BLE001
            pass
    write_json(target / "RUN_RECEIPT.json", receipt)
    if receipt["exit_status"] == "complete":
        receipt["physics_gz"] = gzip_physics(run_dir / "physics_steps.jsonl")
        write_json(target / "RUN_RECEIPT.json", receipt)
    return receipt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--bases-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="target run dir (must not exist)")
    ap.add_argument("--horizon-s", type=float, default=600.0)
    ap.add_argument("--decision-period-s", type=float, default=DECISION_PERIOD_S)
    ap.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    ap.add_argument("--registration", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--policy-module", action="append", default=[])
    ap.add_argument("--native-reference", choices=("zero", "head"), default=None,
                    help="A11 dV native reference (default: predictor default 'zero')")
    ap.add_argument("--open-road", action="store_true",
                    help="open-road A11 rollout: physical exits + counter-driven entries (default off)")
    ap.add_argument("--max-decision-units", type=int, default=None,
                    help="PL01 planner cap (default None = planner default)")
    ap.add_argument("--model-path", default=None, help="A11 member .pt (env WM_N1_A11_MEMBER)")
    ap.add_argument("--model-sha256", default=None, help="pin (env WM_N1_A11_MEMBER_SHA256)")
    ap.add_argument("--climatology", default=None,
                    help="TRAIN climatology json (env WM_N1_PAIRED_EFFECT_CLIMATOLOGY)")
    ap.add_argument("--device", default=None, help="cpu|cuda for the HDV net (env WM_N1_DEVICE)")
    a = ap.parse_args(argv)
    for flag, env in ((a.model_path, "WM_N1_A11_MEMBER"), (a.model_sha256, "WM_N1_A11_MEMBER_SHA256"),
                      (a.climatology, "WM_N1_PAIRED_EFFECT_CLIMATOLOGY"), (a.device, "WM_N1_DEVICE")):
        if flag:
            os.environ[env] = str(flag)
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(a.threads))
    policy_kwargs = {}
    pk = {}
    if a.native_reference is not None:
        pk["native_reference"] = a.native_reference
    if a.open_road:
        pk["open_road"] = True
    if pk:
        policy_kwargs["predictor_kwargs"] = pk
    if a.max_decision_units is not None:
        policy_kwargs["config_overrides"] = {"max_walls": 0, "mode": "controlled_experiment",
                                             "max_decision_units": a.max_decision_units}
    receipt = run_closed_loop(
        world_name=a.world, policy_name=a.policy, plan_path=a.plan, bases_root=a.bases_root,
        target=a.out, horizon_s=a.horizon_s, decision_period_s=a.decision_period_s,
        code_root=a.code_root, registration=a.registration, threads=a.threads,
        policy_modules=a.policy_module, policy_kwargs=policy_kwargs,
        log=lambda m: print(m, flush=True))
    print(json.dumps({k: receipt.get(k) for k in ("run", "exit_status", "physics_rows_written",
                                                  "wall_s")}), flush=True)
    return 0 if receipt.get("exit_status") == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
