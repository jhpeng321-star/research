"""Read saved learning-check evidence without running any new traffic."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.idm.train_native_damping import CASES, atomic_json, load_rollout
from idm.damping_credit import TeamCostScales
from idm.damping_ppo import PPOContract, prepare_batch


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["ppo"] != PPOContract().manifest():
        raise ValueError("audit/learner contract mismatch; do not reinterpret a stopped learner")
    reference = json.loads((root / "reward_reference.json").read_text())
    records = []
    for path in sorted(root.glob("policy_???.json")):
        iteration = int(path.stem.split("_")[-1])
        row = load_rollout(path)
        baseline = load_rollout(root / f"native_{iteration % len(CASES)}.json")
        summary = row["summary"]
        scales = TeamCostScales(
            reference["braking_vehicle_seconds"],
            reference["acceleration_energy"],
            summary["population"],
        )
        batch = prepare_batch(row, baseline, scales)
        if not np.array_equal(
            batch["actor_advantages"], batch["team_advantages"][row["actor_times"]]
        ):
            raise ValueError("actor credit no longer equals the actual signed team advantage")
        returns = np.cumsum(batch["rewards"][::-1])[::-1]
        credit_error = float(
            np.max(np.abs(batch["team_advantages"] + row["old_values"][:-1] - returns))
        )
        last = defaultdict(lambda: -1)
        for vehicle, t in zip(row["actor_ids"], row["actor_times"]):
            last[vehicle] = max(last[vehicle], int(t))
        eligible = [t for t in last.values() if t < len(returns) - 1]
        post_departure_nonzero = sum(bool(abs(returns[t + 1]) > 1e-8) for t in eligible)
        actor_n = len(row["actor_times"])
        cost_consistent = all(
            np.isclose(sum(interval[k] for interval in row["intervals"]), v)
            for k, v in summary["totals"].items()
        )
        service_consistent = np.isclose(
            summary["totals"]["requested_not_arrived_vehicle_seconds"] / summary["population"],
            summary["system_seconds_per_requested_vehicle"],
        )
        if credit_error > 1e-5 or not cost_consistent or not service_consistent:
            raise ValueError(f"physical/credit audit failed for {path}")
        records.append(
            dict(
                iteration=iteration + 1,
                case=list(CASES[iteration % len(CASES)]),
                global_times=len(returns),
                actor_samples=actor_n,
                distinct_cavs=len(last),
                cavs_with_later_nonzero_team_cost=post_departure_nonzero,
                max_team_credit_identity_error=credit_error,
                aggregate_cost_identity=cost_consistent,
                requested_service_identity=bool(service_consistent),
                requested_action_mean=float(row["actions"].mean()),
                requested_action_std=float(row["actions"].std()),
                ramp_fraction_per_actor_decision=summary["commands"].get("slow_down", 0)
                / max(1, actor_n),
                commands=summary["commands"],
                reasons=summary["reasons"],
                reward_audit=batch["reward_audit"],
            )
        )
    return dict(
        schema="native_damping_saved_credit_audit/1",
        episodes_audited=len(records),
        records=records,
        new_simulated_seconds=0,
        limitation="Credit retention and command issuance are not proof of beneficial causal influence",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.training_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))


if __name__ == "__main__":
    main()
