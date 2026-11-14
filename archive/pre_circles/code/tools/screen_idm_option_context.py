"""Grouped development screen of causal P30 option-feasibility context."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.io import atomic_write_json
from idm.option_reward import IDM_BINARY_OPTION_REWARD_V1
from idm.paired_option_dataset import (
    OPTION_CONTEXT_NAMES,
    load_dataset,
    option_context_from_candidate,
)
from tools.audit_idm_phase5_observation_sufficiency import evaluate


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def analyze(dataset: Path) -> dict:
    records, manifest = load_dataset(dataset, split=None)
    rows = []
    for record in records:
        root = Path(record.source)
        discovery = _read(
            root / "discoveries" / f"m{record.demand}.seed{record.seed}.json"
        )["decision"]
        candidate = next(
            row for row in discovery["candidates"] if row["vehicle_id"] == record.vehicle_id
        )
        context = option_context_from_candidate(candidate, record.penetration)
        # Neutral, admissible effects are excluded from discrimination.  A
        # constraint failure is harmful for deployment regardless of its raw
        # physical component because it receives the frozen -2 reward.
        if record.constraints_pass and record.physical_utility > IDM_BINARY_OPTION_REWARD_V1.deadband:
            label = 1
            label_name = "helpful"
        elif (not record.constraints_pass) or (
            record.physical_utility < -IDM_BINARY_OPTION_REWARD_V1.deadband
        ):
            label = 0
            label_name = "harmful"
        else:
            label = None
            label_name = "neutral"
        rows.append(
            {
                "sample_id": record.sample_id,
                "seed": record.seed,
                "demand": record.demand,
                "penetration": record.penetration,
                "label": label,
                "label_name": label_name,
                "physical_observation": record.observation,
                "option_context": context,
                "activate_reward": record.activate_reward,
                "physical_utility": record.physical_utility,
                "constraints_pass": record.constraints_pass,
            }
        )
    labeled = [row for row in rows if row["label"] is not None]
    labels = np.asarray([row["label"] for row in labeled], dtype=int)
    seeds = np.asarray([row["seed"] for row in labeled], dtype=int)
    demands = np.asarray([row["demand"] for row in labeled], dtype=int)
    physical = np.asarray([row["physical_observation"] for row in labeled], dtype=float)
    context = np.asarray([row["option_context"] for row in labeled], dtype=float)
    combined = np.column_stack([physical, context])
    # Prespecified semantic blocks from the documented 134-field layout:
    # ego[0:9] | zone[9:21] | neighbors[21:121] | controller[121:134].
    controller = physical[:, 121:134]
    zone_controller = np.column_stack([physical[:, 9:21], controller])
    ego_zone_controller = np.column_stack([physical[:, 0:21], controller])
    zone_controller_option = np.column_stack([zone_controller, context])
    metrics = {
        "current_134": evaluate(physical, labels, seeds, demands, slice(None)),
        "option_context_13": evaluate(context, labels, seeds, demands, slice(None)),
        "combined_147": evaluate(combined, labels, seeds, demands, slice(None)),
        "controller_context_13": evaluate(
            controller, labels, seeds, demands, slice(None)
        ),
        "zone_controller_25": evaluate(
            zone_controller, labels, seeds, demands, slice(None)
        ),
        "ego_zone_controller_34": evaluate(
            ego_zone_controller, labels, seeds, demands, slice(None)
        ),
        "zone_controller_option_38": evaluate(
            zone_controller_option, labels, seeds, demands, slice(None)
        ),
    }
    constraint_labels = np.asarray([int(row["constraints_pass"]) for row in rows], dtype=int)
    constraint_seeds = np.asarray([row["seed"] for row in rows], dtype=int)
    constraint_demands = np.asarray([row["demand"] for row in rows], dtype=int)
    constraint_context = np.asarray([row["option_context"] for row in rows], dtype=float)
    constraint_metrics = evaluate(
        constraint_context,
        constraint_labels,
        constraint_seeds,
        constraint_demands,
        slice(None),
    )
    class_seeds: dict[str, set[int]] = defaultdict(set)
    for row in labeled:
        class_seeds[row["label_name"]].add(int(row["seed"]))
    gates = {
        "both_deployment_classes_in_at_least_three_seeds": min(
            len(class_seeds["helpful"]), len(class_seeds["harmful"])
        )
        >= 3,
        "combined_auroc_at_least_0p70": metrics["combined_147"]["auroc"] >= 0.70,
        "combined_balanced_accuracy_at_least_0p65": metrics["combined_147"][
            "balanced_accuracy"
        ]
        >= 0.65,
        "combined_auroc_gain_at_least_0p05": metrics["combined_147"]["auroc"]
        - metrics["current_134"]["auroc"]
        >= 0.05,
        "constraint_auroc_at_least_0p70": constraint_metrics["auroc"] >= 0.70,
    }
    return {
        "schema": "idm_option_context_screen/1",
        "development_only": True,
        "dataset": str(Path(dataset).resolve()),
        "dataset_sha256": manifest["dataset_sha256"],
        "feature_hypothesis": {
            "names": list(OPTION_CONTEXT_NAMES),
            "causal_time": "all fields frozen at the pre-action decision",
            "forbidden": [
                "realized dose",
                "branch outcome",
                "reward",
                "classification",
                "seed",
                "demand",
                "decision time",
            ],
        },
        "record_count": len(rows),
        "labeled_count": len(labeled),
        "label_counts": dict(Counter(row["label_name"] for row in rows)),
        "label_seed_coverage": {
            key: len(value) for key, value in sorted(class_seeds.items())
        },
        "metrics": metrics,
        "constraint_metrics_option_context_13": constraint_metrics,
        "gates": gates,
        "option_context_authorized_for_training": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.dataset.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
