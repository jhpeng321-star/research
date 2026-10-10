"""Single-trajectory timing-schedule discovery for the Path B headroom pilot.

One deterministic native-IDM trajectory per seed/cell records the ordered
qualified risk-onset schedule t1/t2/t3 with candidate observations and
admission feasibility.  Determinism is verified by running the same
discovery twice (exact schedule identity).  No treatment command is ever
issued by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from experiments.io import atomic_write_json
from idm.action_local import LandmarkContract
from tools.run_idm_decentralized_authority_verification import (
    _actions,
    _candidate_snapshot,
    _env,
    _landmark_at_state,
    _nonclassic_observation_hash,
    _raw_decision_context_fingerprint,
    _treatment_fingerprint,
)

SCHEMA = "idm_timing_schedule/1"
CONFIG = {
    "action_set": "activation_speed_ceiling_v1",
    "option_steps": 20,
    "minimum_dose_steps": 16,
    "landmark": "risk_onset",
    "candidate_ranking": "runway",
    "candidate_count": 1,
    "maximum_landmarks_per_cell": 3,
    "rsu_observation_contract": "idm_rsu_observation_legacy_configured_range_v1",
    "replay_deadband_base": 0.02,
}


def _config_hash(demand: int, cav_penetration: float) -> str:
    payload = {"demand": demand, "cav_penetration": cav_penetration, **CONFIG}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def discover(
    demand: int,
    seed: int,
    cav_penetration: float,
    output: Path,
    max_horizon_steps: int,
    pass_identity: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output)
    env = _env(
        demand,
        seed,
        output,
        f"schedule.m{demand}.seed{seed}.pass{pass_identity}",
        max_horizon_steps + 5,
        CONFIG["action_set"],
        CONFIG["option_steps"],
        CONFIG["minimum_dose_steps"],
        cav_penetration,
    )
    contract = LandmarkContract(maximum_landmarks_per_cell=3)
    occurrences: list[dict[str, Any]] = []
    previous_step: int | None = None
    persistence = 0
    persistent_key = None
    completed = 0
    obs = None
    try:
        obs, _ = env.reset(seed=seed)
        for step in range(max_horizon_steps + 1):
            state = _candidate_snapshot(env, obs, "runway", 1)
            landmark = (
                None
                if state is None
                else _landmark_at_state(state, "risk_onset", contract)
            )
            current_key = (
                landmark,
                None if state is None else state["lane_id"],
                tuple(
                    row["vehicle_id"] for row in state["candidates"]
                )
                if state is not None
                else (),
            )
            if state is not None and landmark is not None:
                persistence = persistence + 1 if current_key == persistent_key else 1
                persistent_key = current_key
                if persistence >= contract.persistence_steps:
                    separated = (
                        previous_step is None
                        or step - previous_step >= contract.minimum_landmark_separation_steps
                    )
                    if separated:
                        candidate = state["candidates"][0]
                        occurrences.append(
                            {
                                "step": step,
                                "landmark": landmark,
                                "persistence_steps": persistence,
                                "lane_id": str(state["lane_id"]),
                                "candidate_id": str(candidate["vehicle_id"]),
                                "runway_m": candidate.get("runway_m"),
                                "predicted_dose_s": candidate.get("predicted_dose_s"),
                                "admissible": True,  # frozen count-1 snapshot admits only feasible candidates
                                "observation_hash": _nonclassic_observation_hash(
                                    state["candidate_observations"][
                                        str(candidate["vehicle_id"])
                                    ]
                                ),
                                "raw_decision_context_fingerprint": (
                                    _raw_decision_context_fingerprint(state)
                                ),
                                "treatment_fingerprint": _treatment_fingerprint(state),
                            }
                        )
                        previous_step = step
                        if len(occurrences) >= 3:
                            break
                    persistence = 0
                    persistent_key = None
            else:
                persistence = 0
                persistent_key = None
            obs, _, terminated, truncated, _ = env.step(_actions(obs))
            completed = step + 1
            if terminated.get("__all__") or truncated.get("__all__"):
                break
    finally:
        env.close()
    return {
        "schema": SCHEMA,
        "pass_identity": pass_identity,
        "demand": demand,
        "seed": seed,
        "cav_penetration": cav_penetration,
        "config_hash": _config_hash(demand, cav_penetration),
        "max_horizon_steps": max_horizon_steps,
        "steps_completed": completed,
        "occurrences": occurrences,
        "missing_landmarks_reason": (
            None
            if len(occurrences) >= 3
            else {
                "reason": "insufficient_landmarks",
                "occurrences": len(occurrences),
                "last_completed_step": completed,
            }
        ),
        "runtime_s": time.perf_counter() - started,
    }


def _record(row: Mapping[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, default=str)


def schedule_identity(primary: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    def signature(row: Mapping[str, Any]) -> str:
        return _record(
            {
                "step": row["step"],
                "landmark": row["landmark"],
                "candidate_id": row["candidate_id"],
                "observation_hash": row["observation_hash"],
                "raw_decision_context_fingerprint": row["raw_decision_context_fingerprint"],
            }
        )

    primary_signatures = [signature(row) for row in primary["occurrences"]]
    replay_signatures = [signature(row) for row in replay["occurrences"]]
    same = primary_signatures == replay_signatures
    return {
        "primary_signature": _record(primary_signatures),
        "replay_signature": _record(replay_signatures),
        "exact_schedule_identity": same,
        "primary_occurrences": len(primary_signatures),
        "replay_occurrences": len(replay_signatures),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--demand", type=int, required=True)
    parser.add_argument("--cav-penetration", type=float, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-horizon-steps", type=int, default=905)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "discoveries").mkdir(exist_ok=True)
    pending = []
    for seed in sorted(set(args.seeds)):
        path = output / "discoveries" / f"m{args.demand}.seed{seed}.json"
        if path.exists():
            continue
        for pass_identity in (0, 1):
            pending.append(
                {
                    "demand": args.demand,
                    "seed": seed,
                    "cav_penetration": args.cav_penetration,
                    "output": str(output),
                    "max_horizon_steps": args.max_horizon_steps,
                    "pass_identity": pass_identity,
                    "path": str(path),
                }
            )
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(discover, **{key: value for key, value in spec.items() if key != "path"}): spec
                for spec in pending
            }
            for future in as_completed(futures):
                row = future.result()
                target = Path(futures[future]["path"])
                atomic_write_json(
                    Path(f"{target}.pass{row['pass_identity']}"),
                    row,
                )
                print(
                    f"[schedule] seed{row['seed']} pass{row['pass_identity']} "
                    f"landmarks={len(row['occurrences'])} runtime={row['runtime_s']:.1f}s",
                    flush=True,
                )
    summary = []
    for path in sorted(output.glob("discoveries/m*.json.pass0")):
        primary = json.loads(path.read_text(encoding="utf-8"))
        replay_path = Path(str(path).replace(".pass0", ".pass1"))
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        identity = schedule_identity(primary, replay)
        summary.append(
            {
                "seed": primary["seed"],
                "demand": primary["demand"],
                "cav_penetration": primary["cav_penetration"],
                "t1": primary["occurrences"][0]["step"] if primary["occurrences"] else None,
                "t2": primary["occurrences"][1]["step"] if len(primary["occurrences"]) > 1 else None,
                "t3": primary["occurrences"][2]["step"] if len(primary["occurrences"]) > 2 else None,
                "schedule_identity": identity["exact_schedule_identity"],
                "missing_reason": primary["missing_landmarks_reason"],
            }
        )
    atomic_write_json(
        output / "inventory.json",
        {
            "schema": SCHEMA,
            "config_hash": _config_hash(args.demand, args.cav_penetration),
            "count": len(summary),
            "states": summary,
        },
    )
    print(json.dumps({"count": len(summary), "items": summary}, default=str, indent=1)[:3000])


if __name__ == "__main__":
    main()
