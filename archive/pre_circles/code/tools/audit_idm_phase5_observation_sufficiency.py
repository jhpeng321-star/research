"""Grouped observation-sufficiency diagnostic for Phase-5 IDM residual learning."""

from __future__ import annotations
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

from env.core import TrafficShepherdMultiAgentEnv
from experiments.idm.run_idm_residual_checkpoint_eval import _env_config

ROOT = Path(__file__).resolve().parents[1]


def labels_from_audit(audit: Path) -> list[dict]:
    summary = json.loads((audit / "summary.json").read_text(encoding="utf-8"))
    rows = []
    for demand in (2200, 2700):
        for seed in range(4107, 4113):
            effects = [
                row
                for row in summary["paired_effects"]
                if int(row["demand"]) == demand and int(row["seed"]) == seed
            ]
            best = max(
                0.5 * (float(row["wave_reduction"]) + float(row["severe_braking_reduction"]))
                for row in effects
            )
            label = 1 if best > 0.02 else (0 if best <= 0.0 else None)
            rows.append(
                {
                    "demand": demand,
                    "seed": seed,
                    "best_safety_composite": best,
                    "label": label,
                    "label_name": "helpful"
                    if label == 1
                    else ("harmful" if label == 0 else "ambiguous"),
                }
            )
    return rows


def collect_case(case: dict) -> dict:
    demand, seed, label = int(case["demand"]), int(case["seed"]), int(case["label"])
    output = Path(case["output"])
    sample_path = output / "samples" / f"zero.m{demand}.seed{seed}.npz"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _env_config(demand, seed, output / "artifacts", "obs_zero", 1800, None)
    cfg["eval_tripinfo_output"].parent.mkdir(parents=True, exist_ok=True)
    env = TrafficShepherdMultiAgentEnv(cfg)
    samples = []
    try:
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            prior = obs
            actions = {agent_id: np.array([0.0], dtype=np.float32) for agent_id in prior}
            obs, _, terminated, truncated, info = env.step(actions)
            selected = {
                str(command["vehicle_id"])
                for command in info.get("idm_residual", {}).get("commands", [])
            }
            for agent_id in selected:
                if agent_id in prior:
                    samples.append(np.asarray(prior[agent_id], dtype=np.float32).reshape(-1))
            done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
    finally:
        env.close()
    array = np.stack(samples) if samples else np.empty((0, 0), dtype=np.float32)
    np.savez_compressed(
        sample_path,
        x=array,
        y=np.full(array.shape[0], label, dtype=np.int8),
        seed=np.full(array.shape[0], seed, dtype=np.int32),
        demand=np.full(array.shape[0], demand, dtype=np.int32),
    )
    return {
        "demand": demand,
        "seed": seed,
        "label": label,
        "samples": int(array.shape[0]),
        "features": int(array.shape[1]) if array.ndim == 2 and array.size else 0,
        "path": str(sample_path),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    positive = y == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    train = np.nan_to_num((x_train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    test = np.nan_to_num((x_test - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    train = np.column_stack([train, np.ones(len(train))])
    test = np.column_stack([test, np.ones(len(test))])
    # Equal total weight per demand/seed cell, then equalize the two classes.
    keys = np.asarray(groups_train)
    cell_counts = {key: int(np.sum(keys == key)) for key in np.unique(keys)}
    weights = np.asarray([1.0 / cell_counts[key] for key in keys], dtype=float)
    for cls in (0, 1):
        mask = y_train == cls
        weights[mask] *= 0.5 / max(weights[mask].sum(), 1e-12)
    w = np.zeros(train.shape[1], dtype=float)
    for step in range(1500):
        logits = np.clip(train @ w, -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        grad = train.T @ (weights * (prob - y_train))
        grad[:-1] += 0.01 * w[:-1]
        w -= 0.08 * grad
    return 1.0 / (1.0 + np.exp(-np.clip(test @ w, -30.0, 30.0)))


def evaluate(
    x: np.ndarray, y: np.ndarray, seeds: np.ndarray, demands: np.ndarray, feature_slice: slice
) -> dict:
    scores = np.zeros(len(y), dtype=float)
    folds = []
    cell_keys = np.asarray([f"{demand}:{seed}" for demand, seed in zip(demands, seeds)])
    for held_seed in sorted(set(seeds.tolist())):
        test = seeds == held_seed
        train = ~test
        prediction = fit_predict(
            x[train, feature_slice], y[train], cell_keys[train], x[test, feature_slice]
        )
        scores[test] = prediction
        folds.append(
            {
                "held_seed": int(held_seed),
                "samples": int(test.sum()),
                "labels": sorted(set(y[test].tolist())),
            }
        )
    predicted = scores >= 0.5
    tpr = float(np.mean(predicted[y == 1]))
    tnr = float(np.mean(~predicted[y == 0]))
    return {
        "auroc": auroc(y, scores),
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "true_positive_rate": tpr,
        "true_negative_rate": tnr,
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples-per-cell", type=int, default=300)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels = labels_from_audit(args.audit.resolve())
    cases = [row | {"output": str(output)} for row in labels if row["label"] is not None]
    collected = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(collect_case, case) for case in cases]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            collected.append(result)
            print(
                f"[{index}/{len(futures)}] m{result['demand']} seed{result['seed']} {result['samples']} samples",
                flush=True,
            )
    arrays = []
    for result in sorted(collected, key=lambda row: (row["demand"], row["seed"])):
        payload = np.load(result["path"])
        count = len(payload["y"])
        if count > args.max_samples_per_cell:
            indices = np.linspace(0, count - 1, args.max_samples_per_cell, dtype=int)
        else:
            indices = np.arange(count)
        arrays.append(
            (
                payload["x"][indices],
                payload["y"][indices],
                payload["seed"][indices],
                payload["demand"][indices],
            )
        )
    packed = np.concatenate([row[0] for row in arrays])
    y = np.concatenate([row[1] for row in arrays])
    seeds = np.concatenate([row[2] for row in arrays])
    demands = np.concatenate([row[3] for row in arrays])
    local_dim = 134
    x = packed[:, :local_dim]
    full = evaluate(x, y, seeds, demands, slice(None))
    context = evaluate(x, y, seeds, demands, slice(-13, None))
    label_seed_coverage = {
        name: len({row["seed"] for row in labels if row["label_name"] == name})
        for name in ("helpful", "harmful")
    }
    gates = {
        "full_observation_auroc_at_least_0p70": full["auroc"] >= 0.70,
        "full_observation_balanced_accuracy_at_least_0p65": full["balanced_accuracy"] >= 0.65,
        "both_labels_in_three_seeds": min(label_seed_coverage.values()) >= 3,
    }
    result = {
        "schema": "idm_phase5_observation_sufficiency/2",
        "development_only": True,
        "forbidden_inputs": ["demand", "seed", "arm", "filename", "future outcome"],
        "labels": labels,
        "collected": sorted(collected, key=lambda row: (row["demand"], row["seed"])),
        "sample_count": int(len(y)),
        "packed_feature_count": int(packed.shape[1]),
        "actor_visible_feature_count": int(x.shape[1]),
        "controller_context_indices": list(range(121, 134)),
        "label_seed_coverage": label_seed_coverage,
        "full_observation": full,
        "controller_context_only_last13": context,
        "gates": gates,
        "observation_sufficiency_pass": all(gates.values()),
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
