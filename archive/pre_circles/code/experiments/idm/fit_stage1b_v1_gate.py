"""Stage 1b v1 gate fit on the p20 generation data (Grouped CV, ladder).

Data: seeds 7001-7240, one row per (seed, t_d), t_d in {150, 250, 350}.
Targets: Delta_delay(W_td - D) s/veh; P(Delta_delay < 0); prevention
(D collapses, W does not). Features: the D-prefix RSU vector at t_d;
forbidden/excluded: hidden demand, absolute time, outcomes, W-arm
quantities; t_d only at rung L5 as a categorical.
Ladder L0 (near_merge_count threshold, k per training fold by training
policy value) -> L1 counts+densities -> L2 +120 s diffs -> L3 +CAV geometry
-> L4 +30 s flow/history -> L5 +t_d. Models per rung: standardized ridge
(alpha by inner grouped 3-fold CV) and HistGradientBoostingRegressor
(max_depth 3, min_samples_leaf 20); classifiers with the same features
(RidgeClassifier / HistGBMClassifier) for P(benefit). Outer 5-fold grouped
CV (group = seed), 3 fold seeds.

Policy value: engage if pred_delta - lambda * P(benefit) < 0, lambda in
{0,1,2,4} s/veh per prevented collapse; value = mean over seeds of
(-delta + lambda*prevented) when engaged else 0. Baselines: never,
always-W250, L0; oracle = per-seed best of never/W150/W250/W350.
Bootstrap CI90 (seed level) on every value difference. Timing selection
(t_d chosen by predicted value vs fixed W250) evaluated only if the Part-1
timing-matters fraction exceeded 20% in some bin.

Recording limitation: the generation schema stored 120 s backward
differences (per the generation amendment); the 10/30/60 s trends were not
recorded, so L2's trend block is the 120 s difference features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments import REPO_ROOT
from experiments.io import atomic_write_json as _atomic_json

PROBE = REPO_ROOT / "runs" / "probe"
OUTPUT = PROBE / "stage1b_generation"
FOLD_SEEDS = (7, 17, 29)
LAMBDAS = (0.0, 1.0, 2.0, 4.0)
RUNG_KEYS: Dict[str, List[str]] = {
    "L0": ["near_merge_count"],
    "L1": ["upstream_count", "approach_count", "near_merge_count",
           "merge_edge_count", "post_merge_count", "ramp_count",
           "upstream_density", "approach_density", "merge_edge_density"],
    "L2": ["upstream_count", "approach_count", "near_merge_count",
           "merge_edge_count", "post_merge_count", "ramp_count",
           "upstream_density", "approach_density", "merge_edge_density",
           "merge_density_diff_120s", "approach_std_diff_120s",
           "inflow_minus_discharge_diff_120s"],
    "L3": ["upstream_count", "approach_count", "near_merge_count",
           "merge_edge_count", "post_merge_count", "ramp_count",
           "upstream_density", "approach_density", "merge_edge_density",
           "merge_density_diff_120s", "approach_std_diff_120s",
           "inflow_minus_discharge_diff_120s",
           "cav_count_band", "cav_count_upstream"],
    "L4": ["upstream_count", "approach_count", "near_merge_count",
           "merge_edge_count", "post_merge_count", "ramp_count",
           "upstream_density", "approach_density", "merge_edge_density",
           "merge_density_diff_120s", "approach_std_diff_120s",
           "inflow_minus_discharge_diff_120s",
           "cav_count_band", "cav_count_upstream",
           "ramp_30s_entries", "upstream_inflow_30s", "merge_discharge_30s",
           "inflow_minus_discharge_30s", "h"],
    "L5": ["upstream_count", "approach_count", "near_merge_count",
           "merge_edge_count", "post_merge_count", "ramp_count",
           "upstream_density", "approach_density", "merge_edge_density",
           "merge_density_diff_120s", "approach_std_diff_120s",
           "inflow_minus_discharge_diff_120s",
           "cav_count_band", "cav_count_upstream",
           "ramp_30s_entries", "upstream_inflow_30s", "merge_discharge_30s",
           "inflow_minus_discharge_30s", "h"],
}


def _load_rows() -> List[Dict[str, Any]]:
    dirs = sorted(PROBE.glob("stage1b_generation_p20_s*/seeds"))
    if not dirs:
        raise RuntimeError("p20 artifacts missing")
    rows = []
    for path in sorted(dirs[0].glob("seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        d_inc = bool(payload["arms"]["D"]["health_collapse"]["incidence"])
        for t_d in ("150", "250", "350"):
            features = payload["features"].get(t_d)
            if features is None:
                continue
            w_inc = bool(payload["arms"][f"W{t_d}"]["health_collapse"]["incidence"])
            rows.append({
                "seed": seed, "t_d": int(t_d),
                "delta": float(payload["targets"][f"Delta_delay_W{t_d}_minus_D"]),
                "prevented": float(d_inc and not w_inc),
                "features": {k: float(v) for k, v in features.items()
                             if isinstance(v, (int, float))},
            })
    return rows


def _matrix(rows: Sequence[Dict[str, Any]], keys: List[str], include_td: bool) -> np.ndarray:
    # Recording-defect guard: drop zero-variance columns (the recorded
    # cav_count_band / cav_count_upstream / ramp_queue_length are constant 0
    # because the env has no public active_cavs attribute - reported in the
    # Part B results; the per-lane geometry and all vehicle-level features
    # vary correctly).
    effective = [k for k in keys
                 if float(np.ptp(np.asarray([r["features"].get(k, 0.0) for r in rows]))) > 0.0
                 or k == "t_d"]
    X = np.asarray([[r["features"].get(k, 0.0) for k in effective] for r in rows], dtype=float)
    if include_td:
        X = np.hstack([X, np.asarray([[float(r["t_d"])] for r in rows])])
    return X


def _auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    ranks = rankdata(list(pos) + list(neg))
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _brier(prob: Sequence[float], truth: Sequence[int]) -> float:
    return float(np.mean((np.asarray(prob) - np.asarray(truth)) ** 2))


def _reliability(prob: Sequence[float], truth: Sequence[int], bins: int = 10) -> List[Dict[str, Any]]:
    p, t = np.asarray(prob), np.asarray(truth)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        mask = (p >= edges[i]) & (p <= edges[i + 1]) if i == bins - 1 else ((p >= edges[i]) & (p < edges[i + 1]))
        out.append({
            "bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]),
            "n": int(mask.sum()),
            "pred_mean": float(np.mean(p[mask])) if mask.sum() else None,
            "actual": float(np.mean(t[mask])) if mask.sum() else None,
        })
    return out


def _fit_fold(rung: str, model_kind: str, train: List[int], test: List[int],
              rows: List[Dict[str, Any]], y: np.ndarray, y_bin: np.ndarray,
              groups_tr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_all = _matrix(rows, RUNG_KEYS[rung], include_td=(rung == "L5"))
    if rung == "L0":
        X_tr, y_tr = X_all[train], y[train]
        grid = np.percentile(X_tr[:, 0], np.linspace(0, 100, 31))
        best_k, best_val = grid[0], -np.inf
        for k in grid:
            engage = X_tr[:, 0] > k
            val = float(np.mean(np.where(engage, -y_tr, 0.0)))
            if val > best_val:
                best_k, best_val = k, val
        engage = X_all[test, 0] > best_k
        return np.zeros(len(test)), engage.astype(float)
    X_tr, X_te = X_all[train], X_all[test]
    y_tr, bin_tr = y[train], y_bin[train]
    if model_kind == "ridge":
        inner = GroupKFold(n_splits=3)
        model = GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                             {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]},
                             cv=inner, scoring="neg_mean_squared_error")
        model.fit(X_tr, y_tr, groups=groups_tr)
    else:
        model = HistGradientBoostingRegressor(max_depth=3, min_samples_leaf=20,
                                              max_iter=300, random_state=0)
        model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    if model_kind == "ridge":
        clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    else:
        clf = HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=20,
                                             max_iter=300, random_state=0)
    clf.fit(X_tr, bin_tr)
    if hasattr(clf, "decision_function"):
        score = np.clip(clf.decision_function(X_te), -40, 40)
        prob = 1.0 / (1.0 + np.exp(-score))
    else:
        prob = clf.predict_proba(X_te)[:, 1]
    return pred, prob


def _bootstrap_ci(values: Sequence[float], resamples: int = 2000,
                  rng_seed: int = 20260903) -> Dict[str, float]:
    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(values, dtype=float)
    means = [float(np.mean(arr[rng.integers(0, arr.size, size=arr.size)]))
             for _ in range(resamples)]
    return {"lo": float(np.percentile(means, 5)), "hi": float(np.percentile(means, 95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=str(OUTPUT))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    groups = np.asarray([r["seed"] for r in rows])
    y = np.asarray([r["delta"] for r in rows])
    y_bin = np.asarray([1.0 if r["delta"] < 0.0 else 0.0 for r in rows])
    by_td = {
        td: [(i, r) for i, r in enumerate(rows) if r["t_d"] == td]
        for td in (150, 250, 350)
    }
    per_row_td: Dict[int, int] = {i: r["t_d"] for i, r in enumerate(rows)}
    seeds = sorted({r["seed"] for r in rows})

    ladder: Dict[str, Dict[str, Any]] = {}
    for rung in ("L0", "L1", "L2", "L3", "L4", "L5"):
        for model_kind in ("ridge", "histgbm"):
            oof = {fs: None for fs in FOLD_SEEDS}
            for fold_seed in FOLD_SEEDS:
                preds_all = np.full(len(rows), np.nan)
                prob_all = np.full(len(rows), np.nan)
                perm = np.random.default_rng(fold_seed).permutation(len(seeds))
                permuted_index = np.concatenate(
                    [np.where(groups == s)[0] for s in np.asarray(seeds)[perm]]
                )
                kf = GroupKFold(n_splits=5)
                for train_i, test_i in kf.split(permuted_index, y[permuted_index],
                                                groups[permuted_index]):
                    train_l = list(permuted_index[train_i])
                    test_l = list(permuted_index[test_i])
                    pred, prob = _fit_fold(rung, model_kind, train_l, test_l, rows,
                                           y, y_bin,
                                           groups[train_l])
                    preds_all[test_l] = pred
                    prob_all[test_l] = prob
                oof[fold_seed] = (preds_all, prob_all, permuted_index)
            metric_sets = []
            for pred, prob, _idx in oof.values():
                metric_sets.append({
                    "delta_r2": float(np.corrcoef(pred, y)[0, 1] ** 2),
                    "delta_spearman": float(np.corrcoef(rankdata(pred), rankdata(y))[0, 1]),
                    "auc": _auc(list(prob[y_bin == 1]), list(prob[y_bin == 0])),
                    "brier": _brier(prob, y_bin),
                })
            keys = list(metric_sets[0])
            ladder.setdefault(rung, {})[model_kind] = {
                "metrics_mean": {k: float(np.mean([m[k] for m in metric_sets]))
                                 for k in keys},
                "metrics_range": {k: [float(min(m[k] for m in metric_sets)),
                                      float(max(m[k] for m in metric_sets))]
                                  for k in keys},
                "reliability": _reliability(oof[FOLD_SEEDS[0]][1], y_bin),
            }

    # Policy values: use the first fold seed's OOF predictions (seed-grouped).
    def value_table(rung: str, model_kind: str) -> Dict[str, Any]:
        pred, prob, _idx = _model_oof(rung, model_kind)
        out: Dict[str, Any] = {}
        for lam in LAMBDAS:
            engaged = (pred - lam * prob) < 0.0
            vals = np.where(engaged, -y + lam * np.asarray([r["prevented"] for r in rows]), 0.0)
            out[f"lambda_{lam:g}"] = {
                "value": float(np.mean(vals)),
                "incidence_prevention_count": int(sum(1 for j in range(len(rows))
                                                      if engaged[j] and rows[j]["prevented"])),
            }
        return out

    def _model_oof(rung: str, model_kind: str):
        # Recompute OOF for the value tables on the first fold seed.
        pred = np.full(len(rows), np.nan)
        prob = np.full(len(rows), np.nan)
        fold_seed = FOLD_SEEDS[0]
        perm = np.random.default_rng(fold_seed).permutation(len(seeds))
        permuted_index = np.concatenate(
            [np.where(groups == s)[0] for s in np.asarray(seeds)[perm]]
        )
        kf = GroupKFold(n_splits=5)
        for train_i, test_i in kf.split(permuted_index, y[permuted_index],
                                        groups[permuted_index]):
            train_l = list(permuted_index[train_i])
            test_l = list(permuted_index[test_i])
            p, pb = _fit_fold(rung, model_kind, train_l, test_l, rows, y, y_bin,
                              groups[train_l])
            pred[test_l] = p
            prob[test_l] = pb
        return pred, prob, permuted_index

    # Baselines for values
    value_baselines: Dict[str, Any] = {}
    deltas_by_seed: Dict[int, Dict[int, float]] = {}
    prevented_by_seed: Dict[int, Dict[int, float]] = {}
    for r in rows:
        deltas_by_seed.setdefault(r["seed"], {})[r["t_d"]] = r["delta"]
        prevented_by_seed.setdefault(r["seed"], {})[r["t_d"]] = r["prevented"]
    for lam in LAMBDAS:
        never = 0.0
        always_250 = float(np.mean([
            -deltas_by_seed[s][250] + lam * prevented_by_seed[s][250]
            for s in seeds if 250 in deltas_by_seed[s]
        ]))
        oracle = float(np.mean([
            max(0.0, -deltas_by_seed[s][250] + lam * prevented_by_seed[s][250],
                -deltas_by_seed[s].get(150, 0.0) + lam * prevented_by_seed[s].get(150, 0.0),
                -deltas_by_seed[s].get(350, 0.0) + lam * prevented_by_seed[s].get(350, 0.0))
            for s in seeds
        ]))
        per_seed_oracle = []
        for s in seeds:
            best = 0.0
            for td in (150, 250, 350):
                if td in deltas_by_seed[s]:
                    best = max(best, -deltas_by_seed[s][td] + lam * prevented_by_seed[s][td])
            per_seed_oracle.append(best)
        value_baselines[f"lambda_{lam:g}"] = {
            "never": never,
            "always_W250": always_250,
            "oracle": float(np.mean(per_seed_oracle)),
            "per_seed_oracle": per_seed_oracle,
        }
    for rung in ladder:
        for model_kind in ladder[rung]:
            ladder[rung][model_kind]["policy_value"] = value_table(rung, model_kind)
    summary = {
        "schema": "stage1b_v1_gate_fit/1",
        "n_rows": len(rows),
        "n_seeds": len(seeds),
        "ladder": ladder,
        "value_baselines": value_baselines,
        "limitation_note": ("10/30/60 s trend features were not recorded in the "
                            "generation schema (120 s differences only); L2's trend "
                            "block is the 120 s difference features; the fit is "
                            "descriptive on development seeds only."),
    }
    # Per-seed policy values + bootstrap CIs of policy-minus-L0 at lam 0/2.
    def per_seed_values(rung: str, model_kind: str, lam: float) -> np.ndarray:
        pred, prob, _ = _model_oof(rung, model_kind)
        engaged = (pred - lam * prob) < 0.0
        vals = np.where(engaged, -y + lam * np.asarray([r["prevented"] for r in rows]), 0.0)
        return vals

    value_extra: Dict[str, Any] = {}
    for lam in (0.0, 2.0):
        l0_vals = per_seed_values("L0", "ridge", lam)
        for rung in ("L1", "L2", "L3", "L4", "L5"):
            for model_kind in ("ridge", "histgbm"):
                vals = per_seed_values(rung, model_kind, lam)
                diffs = vals - l0_vals
                ci = _bootstrap_ci(diffs)
                value_extra.setdefault(rung, {}).setdefault(model_kind, {})[
                    f"vs_L0_lambda_{lam:g}"
                ] = {"mean_diff": float(np.mean(diffs)), "ci90": ci,
                     "mean_value": float(np.mean(vals)),
                     "mean_l0_value": float(np.mean(l0_vals))}
    summary["value_vs_L0_bootstrap"] = value_extra
    summary["headline"] = {
        "question": "does any rung L1-L5 beat L0 on OOF policy value?",
        "answer": {
            key: {
                "lambda": key.split("_")[-1],
                "best_rung_vs_L0": max(
                    ((rung, block[model]["vs_L0_lambda_0"]["mean_diff"] if key == "lambda_0"
                      else block[model]["vs_L0_lambda_2"]["mean_diff"])
                     for rung, block in value_extra.items() for model in block),
                    key=lambda kv: kv[1],
                )[0] if value_extra else None,
                "details": value_extra,
            }
            for key in ("lambda_0", "lambda_2")
        },
    }
    # Timing selection (Part 1: timing-matters > 20% in some bin -> evaluated).
    # Use the best regressor (L1 ridge) OOF predictions; per seed pick the t_d
    # with the lowest predicted Delta_delay; compare realized value vs fixed W250.
    pred_by_row, _prob, _ = _model_oof("L1", "ridge")
    per_seed_pred: Dict[int, Dict[int, float]] = {}
    for i, r in enumerate(rows):
        per_seed_pred.setdefault(r["seed"], {})[r["t_d"]] = pred_by_row[i]
    timed = []; fixed = []
    for seed in seeds:
        if seed not in per_seed_pred or 250 not in deltas_by_seed.get(seed, {}):
            continue
        best_td = min(per_seed_pred[seed], key=lambda td: per_seed_pred[seed][td])
        if best_td not in deltas_by_seed.get(seed, {}):
            continue
        timed.append(-deltas_by_seed[seed][best_td] + 2.0 * prevented_by_seed[seed].get(best_td, 0.0))
        fixed.append(-deltas_by_seed[seed][250] + 2.0 * prevented_by_seed[seed].get(250, 0.0))
    summary["timing_selection"] = {
        "evaluated": True,
        "reason": "Part-1 timing-matters fraction > 20% in bins 2400/2500/2600",
        "timed_value_lambda2": float(np.mean(timed)),
        "fixed_W250_value_lambda2": float(np.mean(fixed)),
        "mean_diff": float(np.mean(np.asarray(timed) - np.asarray(fixed))),
        "ci90": _bootstrap_ci(np.asarray(timed) - np.asarray(fixed)),
    }
    _atomic_json(output / "gate_fit_summary.json", summary)
    for rung, models in ladder.items():
        for model_kind, block in models.items():
            print(rung, model_kind, {k: round(v, 3) for k, v in block["metrics_mean"].items()})


if __name__ == "__main__":
    main()
