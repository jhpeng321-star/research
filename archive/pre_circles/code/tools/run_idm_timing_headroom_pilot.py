"""Development-only sequential timing-headroom pilot runner (Path B).

Runs, per frozen native-IDM schedule, the five matched arms (always abstain,
abstain replay, activate at t1, defer to t2, defer to t3) from the common
state at t1 to the common terminal time H, with preactivation identity
verification and the frozen P30 treatment.  Metrics are collected over the
identical [t1, H] window; utility/constraint normalization happens in the
analyzer with the frozen rate equations.  No observation model or RL.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import traci

from experiments.io import atomic_write_json
from idm.timing_headroom import common_state_horizon
from env.snapshot import get_active_snapshot
from tools.run_idm_decentralized_authority_verification import (
    _actions,
    _candidate_snapshot,
    _env,
    _merge_discharge_step,
    _nonclassic_observation_hash,
)

SCHEMA = "idm_timing_headroom_pilot/1"


def _landmark(occurrences: list[Mapping[str, Any]], index: int) -> Mapping[str, Any] | None:
    return occurrences[index] if len(occurrences) > index else None


def run_arm(spec: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    demand = int(spec["demand"])
    seed = int(spec["seed"])
    penetration = float(spec["cav_penetration"])
    schedule = spec["schedule"]
    arm = str(spec["arm"])
    output = Path(spec["output"])
    t1 = int(schedule["occurrences"][0]["step"])
    t2 = int(schedule["occurrences"][1]["step"]) if len(schedule["occurrences"]) > 1 else None
    t3 = int(schedule["occurrences"][2]["step"]) if len(schedule["occurrences"]) > 2 else None
    activation_step = {
        "activate_now": t1,
        "defer_r2": t2,
        "defer_r3": t3,
    }.get(arm)
    L = common_state_horizon(t1, t2, t3)
    H = t1 + L
    env = _env(
        demand,
        seed,
        output,
        f"timing.m{demand}.seed{seed}.{arm}",
        H + 5,
        "activation_speed_ceiling_v1",
        20,
        16,
        penetration,
    )
    target = None
    if activation_step is not None:
        landmark = _landmark(
            schedule["occurrences"],
            0 if arm == "activate_now" else 1 if arm == "defer_r2" else 2,
        )
        if landmark is None or not landmark.get("admissible", False):
            raise ValueError("activation landmark missing or inadmissible")
        target = str(landmark["candidate_id"])
    severe_total = wave_total = 0.0
    ttc_total = low_ttc_steps = arrivals = discharge = collisions = teleports = 0
    delay_rates: list[float] = []
    target_command_steps = 0
    unexpected_commands = 0
    member_changes: list[dict[str, Any]] = []
    release_rows: list[dict[str, Any]] = []
    identity: dict[str, Any] = {}
    completed = 0
    fired = False
    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(t1):
            obs, _, terminated, truncated, _ = env.step(_actions(obs))
            if terminated.get("__all__") or truncated.get("__all__"):
                break
        # common state identity at t1
        snapshot = _candidate_snapshot(env, obs, "runway", 1)
        record = _landmark(schedule["occurrences"], 0)
        if snapshot is None or record is None:
            identity["t1_identity"] = {"ok": False, "reason": "missing_snapshot"}
        else:
            candidate = snapshot["candidates"][0]
            observed = _nonclassic_observation_hash(
                snapshot["candidate_observations"][str(candidate["vehicle_id"])]
            )
            identity["t1_identity"] = {
                "ok": (
                    str(candidate["vehicle_id"]) == str(record["candidate_id"])
                    and observed == record["observation_hash"]
                ),
                "candidate_id": str(candidate["vehicle_id"]),
                "candidate_identity": str(candidate["vehicle_id"]) == str(record["candidate_id"]),
                "observation_identity": observed == record["observation_hash"],
            }
        snapshot_now = get_active_snapshot()
        previous_road = (
            {
                vehicle_id: str(road)
                for vehicle_id in snapshot_now.ordered_ids()
                for road in [snapshot_now.road(vehicle_id)]
                if road is not None
            }
            if snapshot_now is not None
            else {}
        )
        for relative_step in range(1, L + 1):
            absolute_step = t1 + relative_step
            fire = activation_step is not None and absolute_step == activation_step + 1
            if fire:
                fired = True
                # preactivation identity at the chosen landmark
                landmark_index = {
                    "activate_now": 0,
                    "defer_r2": 1,
                    "defer_r3": 2,
                }.get(arm, 0)
                pre_snapshot = _candidate_snapshot(env, obs, "runway", 1)
                pre_record = _landmark(schedule["occurrences"], landmark_index)
                if pre_snapshot is None or pre_record is None:
                    identity[f"t{landmark_index + 1}_preactivation_identity"] = {
                        "ok": False,
                        "reason": "missing_snapshot",
                    }
                else:
                    candidate = pre_snapshot["candidates"][0]
                    observed = _nonclassic_observation_hash(
                        pre_snapshot["candidate_observations"][str(candidate["vehicle_id"])]
                    )
                    identity[f"t{landmark_index + 1}_preactivation_identity"] = {
                        "ok": (
                            str(candidate["vehicle_id"]) == str(pre_record["candidate_id"])
                            and observed == pre_record["observation_hash"]
                        ),
                        "candidate_id": str(candidate["vehicle_id"]),
                        "candidate_identity": str(candidate["vehicle_id"]) == str(pre_record["candidate_id"]),
                        "observation_identity": observed == pre_record["observation_hash"],
                    }
            actions = (
                _actions(obs, target_id=target, mode=1)
                if fire
                else _actions(obs)
            )
            obs, _, terminated, truncated, info = env.step(actions)
            completed = relative_step
            idm_diag = info.get("idm_residual", {})
            commands = list(idm_diag.get("commands", []))
            member_changes.extend(
                {"relative_step": relative_step, **dict(row)}
                for row in idm_diag.get("member_changes", [])
            )
            release_rows.extend(
                {"relative_step": relative_step, **dict(row)}
                for row in idm_diag.get("release", [])
            )
            for command in commands:
                vehicle_id = str(command["vehicle_id"])
                if target is not None and vehicle_id == target:
                    target_command_steps += 1
                else:
                    unexpected_commands += 1
            severe_total += float(env._corridor_severe_braking_count() or 0.0)
            wave_total += float(env._corridor_acceleration_energy_sum_m2ps4() or 0.0)
            current_ttc = int(env._corridor_ttc_count() or 0)
            ttc_total += current_ttc
            low_ttc_steps += int(current_ttc > 0)
            delay = env._corridor_delay_rate_live()
            if delay is not None and math.isfinite(float(delay)):
                delay_rates.append(float(delay))
            discharge += _merge_discharge_step(env, previous_road)
            snapshot_now = get_active_snapshot()
            previous_road = (
                {
                    vehicle_id: str(road)
                    for vehicle_id in snapshot_now.ordered_ids()
                    for road in [snapshot_now.road(vehicle_id)]
                    if road is not None
                }
                if snapshot_now is not None
                else {}
            )
            arrivals = int(traci.simulation.getArrivedNumber())
            collisions += int(traci.simulation.getCollidingVehiclesNumber())
            teleports += int(traci.simulation.getStartingTeleportNumber())
            if terminated.get("__all__") or truncated.get("__all__"):
                break
    finally:
        env.close()
    return {
        "schema": SCHEMA,
        "state_id": spec["state_id"],
        "arm": arm,
        "demand": demand,
        "seed": seed,
        "cav_penetration": penetration,
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "L": L,
        "H": H,
        "fired": bool(fired),
        "completed_horizon_steps": completed,
        "target_command_step_count": target_command_steps,
        "unexpected_command_count": unexpected_commands,
        "severe_braking_count_sum": severe_total,
        "acceleration_energy_sum": wave_total,
        "ttc_count_sum": ttc_total,
        "low_ttc_exposure_steps": low_ttc_steps,
        "mean_corridor_delay_rate": float(np.mean(delay_rates)) if delay_rates else None,
        "arrived_vehicles": arrivals,
        "merge_discharge_count": discharge,
        "collision_observations": collisions,
        "teleport_observations": teleports,
        "member_changes": member_changes,
        "release_rows": release_rows,
        "identity": identity,
        "runtime_s": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "branches").mkdir(exist_ok=True)
    schedules = []
    for inventory in sorted(args.schedules.glob("p*/inventory.json")):
        payload = json.loads(inventory.read_text(encoding="utf-8"))
        schedules.extend(payload["states"])
    pending = []
    for state in schedules:
        if state["t1"] is None or not bool(state.get("schedule_identity", False)):
            continue
        cell = args.schedules / f"p{int(state['cav_penetration']*100):02d}_m{state['demand']}"
        schedule_path = (
            cell / "discoveries" / f"m{state['demand']}.seed{state['seed']}.json.pass0"
        )
        if not schedule_path.exists():
            continue
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        if len(schedule["occurrences"]) < 1:
            continue
        state_id = f"p{state['cav_penetration']:.2f}_m{state['demand']}_seed{state['seed']}"
        arms = ["abstain", "abstain_replay", "activate_now"]
        if state["t2"] is not None:
            arms.append("defer_r2")
        if state["t3"] is not None:
            arms.append("defer_r3")
        for arm in arms:
            path = output / "branches" / f"{state_id}.{arm}.json"
            if path.exists():
                continue
            pending.append(
                {
                    "demand": int(state["demand"]),
                    "seed": int(state["seed"]),
                    "cav_penetration": float(state["cav_penetration"]),
                    "state_id": state_id,
                    "arm": arm,
                    "schedule": schedule,
                    "output": str(output),
                    "path": str(path),
                }
            )
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_arm, {k: v for k, v in spec.items() if k != "path"}): spec
                for spec in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                atomic_write_json(Path(futures[future]["path"]), row)
                print(
                    f"[arm {index}/{len(pending)}] {row['state_id']}.{row['arm']} "
                    f"dose={row['target_command_step_count']} runtime={row['runtime_s']:.1f}s",
                    flush=True,
                )
    print(json.dumps({"pending_arms": len(pending)}, indent=1))


if __name__ == "__main__":
    main()
