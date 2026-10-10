"""Grouped development screen of causal 30-second IDM option history."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.io import atomic_write_json
from idm.option_reward import IDM_BINARY_OPTION_REWARD_V1
from idm.paired_option_dataset import load_dataset
from idm.temporal_option_context import (
    TEMPORAL_LEVEL_NAMES,
    TEMPORAL_OPTION_CONTEXT_NAMES,
    TEMPORAL_OPTION_CONTEXT_SCHEMA,
    TEMPORAL_TREND_NAMES,
    temporal_option_context,
)
from tools.audit_idm_phase5_observation_sufficiency import evaluate


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _spatial_root_map(roots: list[Path]) -> dict[tuple[int, float], Path]:
    result = {}
    for root in roots:
        manifest = _read(root / "manifest.json")
        # Authority-verification manifests store the frozen protocol directly;
        # summaries later expose the same payload under ``protocol``.
        protocol = manifest.get("protocol", manifest)
        key = (int(protocol["demand"]), round(float(protocol["cav_penetration"]), 6))
        if key in result:
            raise ValueError(f"duplicate spatial-history root for {key}")
        if not protocol.get("diagnostic_spatial_history_contract"):
            raise ValueError(f"{root}: spatial-history diagnostic was not enabled")
        result[key] = root.resolve()
    return result


def analyze(dataset: Path, spatial_roots: list[Path], penetration: float) -> dict:
    records, manifest = load_dataset(dataset, split=None)
    root_map = _spatial_root_map(spatial_roots)
    rows = []
    for record in records:
        if abs(record.penetration - penetration) > 1e-9:
            continue
        root = root_map[(record.demand, round(record.penetration, 6))]
        matches = list(
            (root / "branches").glob(
                f"m{record.demand}.seed{record.seed}.t{record.decision_step}.abstain.json"
            )
        )
        if len(matches) != 1:
            raise ValueError(f"{record.sample_id}: expected one matched abstain branch")
        branch = _read(matches[0])
        if not branch.get("is_control") or branch.get("arm") != "abstain":
            raise ValueError(f"{matches[0]} is not the causal control branch")
        history = (branch.get("diagnostic_spatial_history") or {}).get("predecision_window")
        if not history:
            raise ValueError(f"{matches[0]} has no predecision spatial history")
        if (
            record.constraints_pass
            and record.physical_utility > IDM_BINARY_OPTION_REWARD_V1.deadband
        ):
            label, label_name = 1, "helpful"
        elif (
            not record.constraints_pass
        ) or record.physical_utility < -IDM_BINARY_OPTION_REWARD_V1.deadband:
            label, label_name = 0, "harmful"
        else:
            label, label_name = None, "neutral"
        rows.append(
            {
                "sample_id": record.sample_id,
                "seed": record.seed,
                "demand": record.demand,
                "label": label,
                "label_name": label_name,
                "physical": record.observation,
                "temporal": temporal_option_context(history),
                "history_frames": len(history),
            }
        )
    if not rows:
        raise ValueError(f"no records found at penetration {penetration}")
    labeled = [row for row in rows if row["label"] is not None]
    labels = np.asarray([row["label"] for row in labeled], dtype=int)
    seeds = np.asarray([row["seed"] for row in labeled], dtype=int)
    demands = np.asarray([row["demand"] for row in labeled], dtype=int)
    physical = np.asarray([row["physical"] for row in labeled], dtype=float)
    temporal = np.asarray([row["temporal"] for row in labeled], dtype=float)
    level_stop = len(TEMPORAL_LEVEL_NAMES)
    trend_stop = level_stop + len(TEMPORAL_TREND_NAMES)
    metrics = {
        "current_134": evaluate(physical, labels, seeds, demands, slice(None)),
        "temporal_levels_22": evaluate(
            temporal[:, :level_stop], labels, seeds, demands, slice(None)
        ),
        "temporal_trends_22": evaluate(
            temporal[:, level_stop:trend_stop], labels, seeds, demands, slice(None)
        ),
        "temporal_fronts_8": evaluate(
            temporal[:, trend_stop:], labels, seeds, demands, slice(None)
        ),
        "temporal_all_52": evaluate(temporal, labels, seeds, demands, slice(None)),
        "current_plus_temporal_186": evaluate(
            np.column_stack([physical, temporal]), labels, seeds, demands, slice(None)
        ),
    }
    class_seeds: dict[str, set[int]] = defaultdict(set)
    for row in labeled:
        class_seeds[row["label_name"]].add(int(row["seed"]))
    temporal_best_name = max(
        (name for name in metrics if name.startswith("temporal_")),
        key=lambda name: metrics[name]["auroc"],
    )
    best = metrics[temporal_best_name]
    gates = {
        "both_classes_in_at_least_three_seeds": min(
            len(class_seeds["helpful"]), len(class_seeds["harmful"])
        )
        >= 3,
        "best_temporal_auroc_at_least_0p70": best["auroc"] >= 0.70,
        "best_temporal_balanced_accuracy_at_least_0p65": best["balanced_accuracy"] >= 0.65,
        "temporal_auroc_gain_over_current_at_least_0p05": best["auroc"]
        - metrics["current_134"]["auroc"]
        >= 0.05,
    }
    return {
        "schema": "idm_temporal_option_context_screen/1",
        "development_only": True,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": manifest["dataset_sha256"],
        "penetration": penetration,
        "spatial_roots": [str(path.resolve()) for path in spatial_roots],
        "feature_contract": {
            "schema": TEMPORAL_OPTION_CONTEXT_SCHEMA,
            "names": list(TEMPORAL_OPTION_CONTEXT_NAMES),
            "causal_time": "abstain/control branch history ending at the declared decision",
            "forbidden": [
                "future frames",
                "treatment outcome",
                "reward",
                "seed",
                "configured demand",
            ],
        },
        "record_count": len(rows),
        "labeled_count": len(labeled),
        "label_counts": dict(Counter(row["label_name"] for row in rows)),
        "label_seed_coverage": {key: len(value) for key, value in sorted(class_seeds.items())},
        "history_frame_count": {
            "min": min(row["history_frames"] for row in rows),
            "max": max(row["history_frames"] for row in rows),
        },
        "metrics": metrics,
        "best_temporal_family": temporal_best_name,
        "gates": gates,
        "temporal_context_authorized_for_actor_ablation": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spatial-root", type=Path, action="append", required=True)
    parser.add_argument("--penetration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.dataset.resolve(),
        [path.resolve() for path in args.spatial_root],
        args.penetration,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
