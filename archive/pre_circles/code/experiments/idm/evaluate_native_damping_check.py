"""Four deterministic development comparisons after the bounded PPO check.

No fresh seeds, checkpoint search, significance claim, or acceptance marking.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from experiments.idm.train_native_damping import (
    CASES,
    atomic_json,
    collect_cached,
    digest,
    exclusive_run,
    load_rollout,
    run_identity,
)
from idm.damping_actor import SharedDampingActor
from idm.damping_credit import TeamCostScales
from idm.damping_ppo import centered_reward_series
from idm.damping_scenario import request_schedule
from env.native_damping import DampingContract
from policies.sensor_local_damping import CHECKPOINT_SCHEMA


def compare_summaries(policy, native):
    for key in ("scenario_identity", "seed", "population", "global_control_steps", "team_terminal"):
        if policy[key] != native[key]:
            raise ValueError(f"comparison identity mismatch: {key}")
    if not policy["team_terminal"] or policy["global_control_steps"] != 2400:
        raise ValueError("development comparison requires full episodes")
    result = {}
    for name in ("braking_vehicle_seconds", "acceleration_energy"):
        actual, baseline = policy["totals"][name], native["totals"][name]
        result[name + "_delta"] = actual - baseline
        result[name + "_improvement_percent"] = (
            100 * (1 - actual / baseline) if baseline > 0 else None
        )
    result.update(
        system_time_delta_s_per_vehicle=policy["system_seconds_per_requested_vehicle"]
        - native["system_seconds_per_requested_vehicle"],
        arrived_delta=policy["arrived"] - native["arrived"],
        source_wait_delta_vehicle_seconds=policy["source_wait_vehicle_seconds"]
        - native["source_wait_vehicle_seconds"],
        ttc_below_3_delta_vehicle_seconds=policy["ttc_below_3_vehicle_seconds"]
        - native["ttc_below_3_vehicle_seconds"],
        collision_involvement_delta=policy["totals"]["added_collision_events"]
        - native["totals"]["added_collision_events"],
        starting_teleport_delta=policy["totals"]["added_teleport_events"]
        - native["totals"]["added_teleport_events"],
    )
    result["within_existing_service_margins"] = (
        result["system_time_delta_s_per_vehicle"] <= 2 and result["arrived_delta"] >= -1
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    training, root = args.training_output.resolve(), args.output.resolve()
    completed = json.loads((training / "summary.json").read_text())
    manifest = json.loads((training / "manifest.json").read_text())
    expected_episodes = 4 if "reference_reuse" in manifest else 8
    if (
        completed["status"] != "bounded_learning_check_complete"
        or completed["policy_global_steps"] != expected_episodes * 2400
    ):
        raise ValueError("evaluate only the prespecified completed learning-check checkpoint")
    sumo = shutil.which("sumo")
    if not sumo:
        raise RuntimeError("SUMO must be on PATH")
    for key, value in run_identity(sumo).items():
        if manifest[key] != value:
            raise ValueError(f"training/evaluation implementation mismatch: {key}")
    checkpoint = training / f"inference_{expected_episodes:03}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        payload["schema"] != CHECKPOINT_SCHEMA
        or payload["controller"] != DampingContract().manifest()
    ):
        raise ValueError("inference schema mismatch")
    actor = SharedDampingActor(hidden=payload["hidden"])
    actor.load_state_dict(payload["actor_state"], strict=True)
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("evaluation output already exists")
    references = [load_rollout(training / f"native_{i}.json") for i in range(len(CASES))]
    reward = json.loads((training / "reward_reference.json").read_text())
    identity = dict(
        schema="native_damping_deterministic_dev_check/1",
        training_manifest=digest(manifest),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=[list(case) for case in CASES],
        evaluation_global_step_cap=9600,
        accepted_checkpoint=False,
        prospective=False,
    )
    rows = []
    with exclusive_run(root):
        path = root / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError("evaluation identity changed")
        atomic_json(path, identity)
        for index, (seed, loading) in enumerate(CASES):
            requests = request_schedule(seed, loading)
            scene = training / f"scenario_{index}" / "scenario.sumocfg"
            rollout = collect_cached(
                root / f"deterministic_{index}.json",
                scene,
                requests,
                seed=seed,
                sumo_binary=sumo,
                actor=actor,
            )
            if rollout["summary"]["observation_map"] != payload["observation_map"]:
                raise ValueError("inference map mismatch")
            scales = TeamCostScales(
                reward["braking_vehicle_seconds"], reward["acceleration_energy"], len(requests)
            )
            _, audit = centered_reward_series(rollout, references[index], scales)
            comparison = compare_summaries(rollout["summary"], references[index]["summary"])
            rows.append(
                dict(
                    case=[seed, loading],
                    comparison=comparison,
                    reward_audit=audit,
                    policy=rollout["summary"],
                    native=references[index]["summary"],
                )
            )
            print(json.dumps({"case": [seed, loading], "comparison": comparison}), flush=True)
        numeric = [
            key
            for key, value in rows[0]["comparison"].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        means = {
            key: float(np.mean([row["comparison"][key] for row in rows]))
            for key in numeric
            if all(row["comparison"][key] is not None for row in rows)
        }
        atomic_json(
            root / "summary.json",
            dict(
                status="deterministic_development_check_complete",
                cases=rows,
                unweighted_case_mean=means,
                evaluation_global_steps=9600,
                accepted_checkpoint=False,
                prospective=False,
                limitation="Four exposed cases, one optimizer seed, no checkpoint selection or confidence claim",
            ),
        )


if __name__ == "__main__":
    main()
