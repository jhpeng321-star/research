"""Trace and screen a frozen binary IDM policy's deployment threshold.

This is a development-only diagnostic.  It does not retrain the checkpoint and
never touches the independent validation/confirmation seed partitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.io import atomic_write_json
from policies.rllib_policy import RLlibPolicy
from tools.evaluate_idm_decentralized_clean_smoke import (
    _checkpoint_gate,
    _run_episode,
    _summarize,
)


def _activation_probability(logits: Any) -> float:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("binary policy must expose two action logits")
    values = values[:2] - np.max(values[:2])
    probabilities = np.exp(values)
    probabilities /= probabilities.sum()
    return float(probabilities[1])


class BinaryThresholdRLlibPolicy(RLlibPolicy):
    """Honor the action mask and activate above a frozen probability threshold."""

    def __init__(self, checkpoint_path: str, threshold: float):
        super().__init__(checkpoint_path, explore=False)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = float(threshold)
        self.raw_local_dim = 0
        self.eligible_probabilities: list[float] = []

    def reset(self, env, seed=None) -> None:
        super().reset(env, seed=seed)
        self.raw_local_dim = int(env._raw_local_dim)
        self.eligible_probabilities = []

    def act(self, obs_dict):
        if self._policy is None:
            raise RuntimeError("policy is not loaded")
        actions = {}
        for vehicle_id, observation in obs_dict.items():
            observation = np.asarray(observation, dtype=np.float32)
            mask = observation[self.raw_local_dim : self.raw_local_dim + 2]
            if mask.shape != (2,):
                raise ValueError("binary action mask is missing from observation")
            if mask[0] > 0.5 and mask[1] <= 0.5:
                actions[vehicle_id] = 0
                continue
            if mask[1] > 0.5 and mask[0] <= 0.5:
                actions[vehicle_id] = 1
                continue
            if not np.all(mask > 0.5):
                actions[vehicle_id] = 0
                continue
            _, _, extra = self._policy.compute_single_action(
                observation, explore=False, full_fetch=True
            )
            probability = _activation_probability(extra["action_dist_inputs"])
            self.eligible_probabilities.append(probability)
            actions[vehicle_id] = int(probability >= self.threshold)
        return actions


def _probability_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "p25": None, "median": None,
                "p75": None, "maximum": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[4107, 4109, 4110])
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    invalid = sorted(set(args.seeds) - set(range(4107, 4121)))
    if invalid:
        raise ValueError(f"threshold screening is development-only: {invalid}")
    baseline_rows_dir = args.baseline_eval.resolve() / "rows"
    baseline_labels = (
        "always_abstain",
        "always_activate",
        "analytical_zero_residual",
    )
    rows = []
    for label in baseline_labels:
        for seed in args.seeds:
            path = baseline_rows_dir / f"{label}.seed{seed}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    labels = []
    probability_summaries = {}
    for threshold in args.thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold outside [0, 1]: {threshold}")
        label = f"checkpoint_threshold_{threshold:.4f}".replace(".", "p")
        labels.append(label)
        per_seed = {}
        for seed in args.seeds:
            row_path = args.output.resolve() / "rows" / f"{label}.seed{seed}.json"
            if row_path.exists():
                row = json.loads(row_path.read_text(encoding="utf-8"))
            else:
                policy = BinaryThresholdRLlibPolicy(str(checkpoint), threshold)
                row = _run_episode(
                    policy=policy,
                    label=label,
                    seed=seed,
                    output=args.output.resolve(),
                    decentralized=True,
                    centralized=True,
                )
                row["activation_threshold"] = threshold
                row["eligible_activation_probabilities"] = list(
                    policy.eligible_probabilities
                )
                row["probability_summary"] = _probability_summary(
                    policy.eligible_probabilities
                )
                row_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(row_path, row)
                print(f"completed threshold {threshold:.4f} seed {seed}", flush=True)
            rows.append(row)
            per_seed[str(seed)] = row.get("probability_summary") or _probability_summary(
                row.get("eligible_activation_probabilities", [])
            )
        all_probabilities = [
            value
            for row in rows
            if row["label"] == label
            for value in row.get("eligible_activation_probabilities", [])
        ]
        probability_summaries[label] = {
            "overall": _probability_summary(all_probabilities),
            "by_seed": per_seed,
        }
    summary = _summarize(rows)
    gates = {
        label: _checkpoint_gate(summary, label, list(baseline_labels))
        for label in labels
    }
    payload = {
        "schema": "idm_decentralized_activation_threshold_screen/1",
        "development_only": True,
        "checkpoint": str(checkpoint),
        "seeds": args.seeds,
        "thresholds": args.thresholds,
        "summary": summary,
        "probability_summaries": probability_summaries,
        "gates": gates,
        "pass": any(gate["pass"] for gate in gates.values()),
    }
    atomic_write_json(args.output.resolve() / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
