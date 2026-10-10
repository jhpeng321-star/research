"""Frozen nonlinear leave-one-seed-out probe for Phase-5 observations."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from tools.audit_idm_phase5_observation_sufficiency import auroc

ROOT = Path(__file__).resolve().parents[1]


def load_samples(root: Path, max_per_cell: int = 300):
    rows = []
    for path in sorted((root / "samples").glob("*.npz")):
        payload = np.load(path)
        count = len(payload["y"])
        indices = np.linspace(0, count - 1, min(count, max_per_cell), dtype=int)
        rows.append(
            (
                payload["x"][indices, :128],
                payload["y"][indices],
                payload["seed"][indices],
                payload["demand"][indices],
            )
        )
    return tuple(np.concatenate([row[index] for row in rows]) for index in range(4))


def cell_balanced_weights(y: np.ndarray, seeds: np.ndarray, demands: np.ndarray) -> np.ndarray:
    keys = np.asarray([f"{demand}:{seed}" for demand, seed in zip(demands, seeds)])
    counts = {key: int(np.sum(keys == key)) for key in np.unique(keys)}
    weights = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float32)
    for cls in (0, 1):
        mask = y == cls
        weights[mask] *= 0.5 / max(float(weights[mask].sum()), 1e-12)
    return weights


def evaluate(x: np.ndarray, y: np.ndarray, seeds: np.ndarray, demands: np.ndarray) -> dict:
    torch.set_num_threads(1)
    scores = np.zeros(len(y), dtype=float)
    folds = []
    for held_seed in sorted(set(seeds.tolist())):
        train = seeds != held_seed
        test = ~train
        mu = x[train].mean(axis=0)
        sigma = x[train].std(axis=0)
        sigma[sigma < 1e-6] = 1.0
        x_train = np.nan_to_num((x[train] - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )
        x_test = np.nan_to_num((x[test] - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )
        torch.manual_seed(20260827 + int(held_seed))
        model = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        features = torch.from_numpy(x_train)
        targets = torch.from_numpy(y[train].astype(np.float32)).view(-1, 1)
        weights = torch.from_numpy(
            cell_balanced_weights(y[train], seeds[train], demands[train])
        ).view(-1, 1)
        for _ in range(300):
            logits = model(features)
            loss = (
                nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
                * weights
            ).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            scores[test] = torch.sigmoid(model(torch.from_numpy(x_test))).squeeze(1).numpy()
        folds.append({"held_seed": int(held_seed), "samples": int(test.sum())})
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


def shuffled_cell_labels(y: np.ndarray, seeds: np.ndarray, demands: np.ndarray) -> np.ndarray:
    cells = sorted({(int(demand), int(seed)) for demand, seed in zip(demands, seeds)})
    original = [
        int(y[np.where((demands == demand) & (seeds == seed))[0][0]]) for demand, seed in cells
    ]
    rng = np.random.default_rng(20260827)
    permuted = rng.permutation(original)
    mapping = {cell: int(label) for cell, label in zip(cells, permuted)}
    return np.asarray(
        [mapping[(int(demand), int(seed))] for demand, seed in zip(demands, seeds)], dtype=np.int8
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    args = parser.parse_args()
    x, y, seeds, demands = load_samples(args.samples.resolve())
    real = evaluate(x, y, seeds, demands)
    shuffled = evaluate(x, shuffled_cell_labels(y, seeds, demands), seeds, demands)
    gates = {
        "auroc_at_least_0p70": real["auroc"] >= 0.70,
        "balanced_accuracy_at_least_0p65": real["balanced_accuracy"] >= 0.65,
        "auroc_margin_over_shuffled_at_least_0p10": real["auroc"] - shuffled["auroc"] >= 0.10,
    }
    result = {
        "schema": "idm_phase5_observation_mlp/1",
        "sample_count": int(len(y)),
        "actor_visible_feature_count": 128,
        "model": "128-64-ReLU-16-ReLU-1",
        "real_labels": real,
        "shuffled_cell_labels": shuffled,
        "auroc_margin_over_shuffled": real["auroc"] - shuffled["auroc"],
        "gates": gates,
        "observation_sufficiency_pass": all(gates.values()),
    }
    output = args.samples / "mlp_summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
