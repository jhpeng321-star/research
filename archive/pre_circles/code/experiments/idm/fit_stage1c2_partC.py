"""Stage 1c-2 Part C: fair threshold baseline (L0'), refit (L1'), onset AUCs.

Data: the p20 stratum (seeds 7001-7240), one row per (seed, t_d in
{150, 250, 350}); targets as in the v1 gate fit (Delta_delay s/veh,
prevented = D collapses and W_td does not, 60 s label after t = 100), with
the CORRECTED feature vector from the Part-A re-record
(stage1b_features_v2_p20_s7001_7240). Conventions reused from
fit_stage1b_v1_gate.py: outer 5-fold GroupKFold by seed, fold seeds
(7, 17, 29), metrics from all three, policy value from the first fold seed,
value per row = (-Delta + lambda * prevented) when engaged else 0.
CIs on value differences are seed-level bootstraps (2000 resamples).

C1 L0': per fold, the single feature and threshold (31-quantile grid,
either direction) maximizing TRAINING-fold policy value; evaluated OOF.
Reported per fold: chosen feature, direction, threshold; plus OOF value at
lambda 0 and 2.

C2 L1': the L1 rung's nine count/density features plus the repaired
cav_count_band, cav_count_upstream and ramp_queue_length (new < 5 m/s
definition), ridge and histgbm as in v1. Reported: OOF R2 (squared Pearson),
Spearman, AUC P(benefit), Brier; policy values at lambda 0/2 against
never / always-W250 / L0 / L0' with seed-bootstrap CI90 on every
difference. Also refits the original L1 on the v1 (defective) vector under
identical folds so the L1 -> L1' delta isolates the repaired features.

C3: onset (treatment-blind, h < 0.15 for 10 consecutive s in the D branch,
uncapped search) labels "onset within the next 100 / 200 s of t_d"
(0 < onset - t_d <= horizon). Univariate AUC of every feature at t_d
pooled over ~720 decisions, top ten, and a grouped-CV L2 logistic on the
full vector (OOF AUC).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments import REPO_ROOT
from experiments.idm.rsu_feature_capture import first_onset_step
from experiments.io import atomic_write_json as _atomic_json

PROBE = REPO_ROOT / "runs" / "probe"
FEATURES_V2_DIR = PROBE / "stage1b_features_v2_p20_s7001_7240"
OUTPUT = PROBE / "stage1c2_partC"
FOLD_SEEDS = (7, 17, 29)
L1_KEYS = ["upstream_count", "approach_count", "near_merge_count",
           "merge_edge_count", "post_merge_count", "ramp_count",
           "upstream_density", "approach_density", "merge_edge_density"]
L1P_EXTRA = ["cav_count_band", "cav_count_upstream", "ramp_queue_length"]
ONSET_HORIZONS = (100, 200)


def _load_rows() -> List[Dict[str, Any]]:
    gen_dirs = sorted(PROBE.glob("stage1b_generation_p20_s*/seeds"))
    v2_dirs = sorted(FEATURES_V2_DIR.glob("seeds"))
    if not gen_dirs or not v2_dirs:
        raise RuntimeError("p20 generation or features-v2 artifacts missing")
    rows: List[Dict[str, Any]] = []
    for path in sorted(gen_dirs[0].glob("seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        v2_path = v2_dirs[0] / f"seed{seed}.json"
        if not v2_path.exists():
            continue
        v2 = json.loads(v2_path.read_text(encoding="utf-8"))
        if not v2.get("digest_match"):
            raise RuntimeError(f"seed {seed}: features-v2 digest mismatch")
        d_h = [float(x) for x in payload["arms"]["D"]["series"]["h"]]
        onset = first_onset_step(d_h, sustain_s=10.0, max_step=10 ** 9)
        d_inc = bool(payload["arms"]["D"]["health_collapse"]["incidence"])
        for t_d in ("150", "250", "350"):
            features = v2["features"].get(t_d)
            features_v1 = payload["features"].get(t_d)
            if features is None or features_v1 is None:
                continue
            w_inc = bool(payload["arms"][f"W{t_d}"]["health_collapse"]["incidence"])
            rows.append({
                "seed": seed,
                "t_d": int(t_d),
                "delta": float(payload["targets"][f"Delta_delay_W{t_d}_minus_D"]),
                "prevented": float(d_inc and not w_inc),
                "features_v2": {k: float(v) for k, v in features.items()},
                "features_v1": {k: float(v) for k, v in features_v1.items()},
                "onset": onset,
            })
    return rows


def _matrix(rows: Sequence[Dict[str, Any]], keys: Sequence[str],
            key: str = "features_v2") -> np.ndarray:
    effective = [k for k in keys
                 if float(np.ptp(np.asarray([r[key].get(k, 0.0) for r in rows]))) > 0.0]
    return np.asarray([[r[key].get(k, 0.0) for k in effective] for r in rows], dtype=float)


def _auc(prob_or_score: Sequence[float], truth: Sequence[int]) -> float:
    """AUC via Mann-Whitney from ranks (score, higher = more likely positive)."""
    truth = np.asarray(truth)
    scores = np.asarray(prob_or_score, dtype=float)
    pos, neg = scores[truth == 1], scores[truth == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = rankdata(list(pos) + list(neg))
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _brier(prob: Sequence[float], truth: Sequence[int]) -> float:
    return float(np.mean((np.asarray(prob) - np.asarray(truth)) ** 2))


def _cv_folds(rows: Sequence[Dict[str, Any]], fold_seed: int) -> List[Tuple[List[int], List[int]]]:
    groups = np.asarray([r["seed"] for r in rows])
    seeds = sorted({r["seed"] for r in rows})
    perm = np.random.default_rng(fold_seed).permutation(len(seeds))
    permuted = np.concatenate([np.where(groups == s)[0] for s in np.asarray(seeds)[perm]])
    kf = GroupKFold(n_splits=5)
    dummy_y = np.zeros(len(permuted))
    return [
        (list(permuted[tr]), list(permuted[te]))
        for tr, te in kf.split(permuted, dummy_y, groups[permuted])
    ]


def _policy_value(engage: np.ndarray, delta: np.ndarray, prevented: np.ndarray,
                  lam: float) -> float:
    return float(np.mean(np.where(engage, -delta + lam * prevented, 0.0)))


def _matrix_exact(rows: Sequence[Dict[str, Any]], keys: Sequence[str],
                  key: str = "features_v2") -> np.ndarray:
    return np.asarray([[r[key].get(k, 0.0) for k in keys] for r in rows], dtype=float)


def _effective_keys(rows: Sequence[Dict[str, Any]], keys: Sequence[str],
                    key: str = "features_v2") -> List[str]:
    return [k for k in keys
            if float(np.ptp(np.asarray([r[key].get(k, 0.0) for r in rows]))) > 0.0]


def fit_l0prime(rows: List[Dict[str, Any]], lam: float) -> Dict[str, Any]:
    """L0': single-feature/threshold rule chosen by training policy value."""
    y = np.asarray([r["delta"] for r in rows])
    prev = np.asarray([r["prevented"] for r in rows])
    all_keys = sorted(rows[0]["features_v2"].keys())
    folds = _cv_folds(rows, FOLD_SEEDS[0])
    oof_engage = np.zeros(len(rows), dtype=bool)
    chosen_per_fold = []
    grid_quantiles = np.linspace(0, 100, 31)
    for train, test in folds:
        effective = _effective_keys([rows[i] for i in train], all_keys)
        X_tr = _matrix_exact([rows[i] for i in train], effective)
        best = (-np.inf, None, None, None)
        for j, key in enumerate(effective):
            column = X_tr[:, j]
            for k in np.percentile(column, grid_quantiles):
                for direction in (">", "<"):
                    engage_tr = column > k if direction == ">" else column < k
                    val = _policy_value(engage_tr, y[train], prev[train], lam)
                    if val > best[0]:
                        best = (float(val), key, direction, float(k))
        _, key, direction, k = best
        chosen_per_fold.append({"feature": key, "direction": direction, "threshold": k,
                                "train_value": best[0]})
        for i in test:
            v = rows[i]["features_v2"].get(key, 0.0)
            oof_engage[i] = (v > k) if direction == ">" else (v < k)
    return {
        "lambda": lam,
        "chosen_per_fold": chosen_per_fold,
        "oof_policy_value": _policy_value(oof_engage, y, prev, lam),
        "oof_engagement_rate": float(oof_engage.mean()),
        "oof_engage": oof_engage,
    }


def _fit_model_pair(kind: str, X_tr: np.ndarray, y_tr: np.ndarray, bin_tr: np.ndarray,
                    groups_tr: np.ndarray):
    if kind == "ridge":
        inner = GroupKFold(n_splits=3)
        model = GridSearchCV(
            make_pipeline(StandardScaler(), Ridge()),
            {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]},
            cv=inner, scoring="neg_mean_squared_error",
        )
        model.fit(X_tr, y_tr, groups=groups_tr)
        clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
        clf.fit(X_tr, bin_tr)
        score = np.clip(clf.decision_function(X_tr), -40, 40)  # noqa: F841
        return model, clf
    reg = HistGradientBoostingRegressor(max_depth=3, min_samples_leaf=20,
                                        max_iter=300, random_state=0)
    reg.fit(X_tr, y_tr)
    clf = HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=20,
                                         max_iter=300, random_state=0)
    clf.fit(X_tr, bin_tr)
    return reg, clf


def _oof_model(rows: List[Dict[str, Any]], keys: List[str], kind: str,
               feature_key: str = "features_v2") -> Dict[str, Any]:
    """OOF predictions/probabilities with the v1 CV machinery (fold seeds).

    Effective columns are fixed ONCE on the full row set (v1 convention) so
    every fold sees the same feature matrix shape.
    """
    y = np.asarray([r["delta"] for r in rows])
    y_bin = np.asarray([1.0 if r["delta"] < 0 else 0.0 for r in rows])
    groups = np.asarray([r["seed"] for r in rows])
    effective = _effective_keys(rows, keys, feature_key)
    X_all = _matrix_exact(rows, effective, feature_key)
    metrics_by_seed = []
    first_pred = first_prob = None
    for fold_seed in FOLD_SEEDS:
        pred_all = np.full(len(rows), np.nan)
        prob_all = np.full(len(rows), np.nan)
        for train, test in _cv_folds(rows, fold_seed):
            X_tr, X_te = X_all[train], X_all[test]
            reg, clf = _fit_model_pair(kind, X_tr, y[train], y_bin[train],
                                       groups[train])
            pred_all[test] = reg.predict(X_te)
            if kind == "ridge":
                score = np.clip(clf.decision_function(X_te), -40, 40)
                prob_all[test] = 1.0 / (1.0 + np.exp(-score))
            else:
                prob_all[test] = clf.predict_proba(X_te)[:, 1]
        if fold_seed == FOLD_SEEDS[0]:
            first_pred, first_prob = pred_all, prob_all
        metrics_by_seed.append({
            "delta_r2": float(np.corrcoef(pred_all, y)[0, 1] ** 2),
            "delta_spearman": float(np.corrcoef(rankdata(pred_all), rankdata(y))[0, 1]),
            "auc": _auc(prob_all, y_bin),
            "brier": _brier(prob_all, y_bin),
        })
    keys_out = list(metrics_by_seed[0])
    return {
        "metrics_mean": {k: float(np.mean([m[k] for m in metrics_by_seed]))
                         for k in keys_out},
        "metrics_range": {k: [float(min(m[k] for m in metrics_by_seed)),
                              float(max(m[k] for m in metrics_by_seed))] for k in keys_out},
        "pred": first_pred,
        "prob": first_prob,
    }


def _seed_bootstrap_ci(diffs_by_row: np.ndarray, seeds: Sequence[int],
                       resamples: int = 2000, seed: int = 20260905) -> Dict[str, float]:
    arr = np.asarray(diffs_by_row, dtype=float)
    uniq = sorted(set(seeds))
    by_seed = np.asarray([float(np.mean(arr[np.asarray(seeds) == s])) for s in uniq])
    rng = np.random.default_rng(seed)
    means = [float(np.mean(by_seed[rng.integers(0, by_seed.size, size=by_seed.size)]))
             for _ in range(resamples)]
    return {"lo": float(np.percentile(means, 5)), "hi": float(np.percentile(means, 95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = _load_rows()
    y = np.asarray([r["delta"] for r in rows])
    prev = np.asarray([r["prevented"] for r in rows])
    seeds_col = np.asarray([r["seed"] for r in rows])
    tds = np.asarray([r["t_d"] for r in rows])

    # ---- C1: L0' ----
    l0p = {lam: fit_l0prime(rows, lam) for lam in (0.0, 2.0)}
    l0p_out = {
        str(lam): {
            "chosen_per_fold": l0p[lam]["chosen_per_fold"],
            "oof_policy_value": l0p[lam]["oof_policy_value"],
            "oof_engagement_rate": l0p[lam]["oof_engagement_rate"],
        } for lam in l0p
    }

    # ---- baselines: never and always-W250 (seed-level, at t_d = 250) ----
    at250 = tds == 250
    seed_ids = sorted(set(seeds_col))
    d250 = {int(s): float(y[(seeds_col == s) & at250][0]) for s in seed_ids}
    p250 = {int(s): float(prev[(seeds_col == s) & at250][0]) for s in seed_ids}

    def always250_value(lam: float) -> float:
        return float(np.mean([-d250[s] + lam * p250[s] for s in seed_ids]))

    def never_value(_lam: float) -> float:
        return 0.0

    # L0 (v1): near_merge_count threshold by training value, lambda-free.
    v1_keys = L1_KEYS[:3] + ["merge_edge_count", "post_merge_count", "ramp_count",
                             "upstream_density", "approach_density",
                             "merge_edge_density"]
    near_col = np.asarray([r["features_v1"].get("near_merge_count", 0.0) for r in rows])
    folds = _cv_folds(rows, FOLD_SEEDS[0])
    l0_engage = np.zeros(len(rows), dtype=bool)
    for train, test in folds:
        grid = np.percentile(near_col[train], np.linspace(0, 100, 31))
        best_k, best_val = grid[0], -np.inf
        for k in grid:
            val = _policy_value(near_col[train] > k, y[train], prev[train], 0.0)
            if val > best_val:
                best_k, best_val = k, val
        l0_engage[test] = near_col[test] > best_k

    # ---- C2: L1 (v1 vector) and L1' (v2 vector) ----
    l1p_keys = L1_KEYS + L1P_EXTRA
    models: Dict[str, Dict[str, Any]] = {}
    for name, keys, fkey in (
        ("L1_v1vector", v1_keys, "features_v1"),
        ("L1prime_v2vector", l1p_keys, "features_v2"),
    ):
        for kind in ("ridge", "histgbm"):
            models[f"{name}:{kind}"] = _oof_model(rows, keys, kind, fkey)

    def per_seed_values(engage: np.ndarray, lam: float) -> np.ndarray:
        return np.where(engage, -y + lam * prev, 0.0)

    def model_engagement(block: Dict[str, Any], lam: float) -> np.ndarray:
        return (block["pred"] - lam * block["prob"]) < 0.0

    value_rows: Dict[str, Any] = {}
    for lam in (0.0, 2.0):
        l0p_engage = l0p[lam]["oof_engage"]
        entries: Dict[str, Any] = {
            "never": {"value": never_value(lam)},
            "always_W250": {"value": always250_value(lam)},
            "L0_v1": {"value": _policy_value(l0_engage, y, prev, lam)},
            "L0prime": {"value": _policy_value(l0p_engage, y, prev, lam)},
        }
        for name, block in models.items():
            engage = model_engagement(block, lam)
            entries[name] = {"value": _policy_value(engage, y, prev, lam)}
        # CI90 on every model's difference vs never / always-W250 / L0 / L0'
        # (seed-level bootstrap of per-row value differences).
        refs = {
            "never": np.zeros(len(rows)),
            "always_W250": per_seed_values(at250, lam),
            "L0_v1": per_seed_values(l0_engage, lam),
            "L0prime": per_seed_values(l0p_engage, lam),
        }
        for name, block in models.items():
            vals = per_seed_values(model_engagement(block, lam), lam)
            entries[name]["ci_vs"] = {
                ref_name: {
                    "mean_diff": float(np.mean(vals - ref_vals)),
                    **_seed_bootstrap_ci(vals - ref_vals, seeds_col),
                }
                for ref_name, ref_vals in refs.items()
            }
        for name in ("L0prime",):
            vals = per_seed_values(l0p_engage, lam)
            entries[name]["ci_vs"] = {
                ref_name: {
                    "mean_diff": float(np.mean(vals - ref_vals)),
                    **_seed_bootstrap_ci(vals - ref_vals, seeds_col),
                }
                for ref_name, ref_vals in refs.items() if ref_name != name
            }
        value_rows[f"lambda_{lam:g}"] = entries

    # ---- C3: onset prediction ----
    onset_arr = np.asarray([r["onset"] if r["onset"] is not None else np.nan
                            for r in rows])
    onset_auc: Dict[str, Any] = {}
    for horizon in ONSET_HORIZONS:
        label = np.asarray([
            1 if (np.isfinite(onset_arr[i]) and tds[i] < onset_arr[i] <= tds[i] + horizon)
            else 0 for i in range(len(rows))
        ])
        keys = _effective_keys(rows, sorted(rows[0]["features_v2"].keys()))
        X = _matrix_exact(rows, keys)
        uni = {}
        for j, key in enumerate(keys):
            uni[key] = _auc(X[:, j], label)
        top = sorted(((k, v) for k, v in uni.items() if np.isfinite(v)),
                     key=lambda kv: -kv[1])[:10]
        # Multivariate grouped-CV L2 logistic on the full vector: one OOF
        # pass per fold seed.
        per_seed_aucs = []
        for fold_seed in FOLD_SEEDS:
            score = np.full(len(rows), np.nan)
            for train, test in _cv_folds(rows, fold_seed):
                X_tr, X_te = X[train], X[test]
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(penalty="l2", C=1.0,
                                                       max_iter=2000))
                clf.fit(X_tr, label[train])
                score[test] = clf.decision_function(X_te)
            per_seed_aucs.append(_auc(score, label))
        onset_auc[f"within_{horizon}s"] = {
            "base_rate": float(label.mean()),
            "n_positive": int(label.sum()),
            "univariate_top10": [{"feature": k, "auc": float(v)} for k, v in top],
            "any_univariate_above_0.75": bool(any(v > 0.75 for _, v in top)),
            "univariate_all": {k: (float(v) if np.isfinite(v) else None)
                               for k, v in uni.items()},
            "multivariate_oof_auc_mean": float(np.mean(per_seed_aucs)),
            "multivariate_oof_auc_range": [float(min(per_seed_aucs)),
                                           float(max(per_seed_aucs))],
        }

    summary = {
        "schema": "stage1c2_partC/1",
        "n_rows": len(rows),
        "n_seeds": len(seed_ids),
        "C1_L0prime": l0p_out,
        "C2_models": {
            name: {"metrics": {k: v for k, v in block.items()
                               if k not in ("pred", "prob")}}
            for name, block in models.items()
        },
        "C2_policy_values": value_rows,
        "C3_onset_prediction": onset_auc,
        "conventions": {
            "cv": "outer 5-fold GroupKFold by seed, fold seeds (7, 17, 29)",
            "value": "engage if pred - lambda*P(benefit) < 0; value = -Delta + "
                     "lambda*prevented when engaged else 0 (s/veh)",
            "ci": "seed-level bootstrap, 2000 resamples, CI90",
            "l0_v1": "near_merge_count threshold (v1 rule), lambda-free training value",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output / "partC_summary.json", summary)

    print("== C1 L0' (chosen feature per fold, OOF value)")
    for lam, block in l0p_out.items():
        print(f"  lambda={lam}: value={block['oof_policy_value']:.3f} "
              f"engage_rate={block['oof_engagement_rate']:.2f}")
        for i, ch in enumerate(block["chosen_per_fold"]):
            print(f"    fold {i}: {ch['feature']} {ch['direction']} "
                  f"{ch['threshold']:.3f} (train value {ch['train_value']:.3f})")
    print("== C2 metrics")
    for name, block in summary["C2_models"].items():
        print(f"  {name}: {block['metrics']['metrics_mean']}")
    print("== C2 policy values")
    for lam, entries in value_rows.items():
        print(f"  {lam}:")
        for name, entry in entries.items():
            line = f"    {name}: value={entry['value']:.3f}"
            if "ci_vs" in entry:
                for ref, ci in entry["ci_vs"].items():
                    line += f" | vs {ref} {ci['mean_diff']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]"
            print(line)
    print("== C3 onset prediction")
    for horizon, block in onset_auc.items():
        print(f"  {horizon}: base rate {block['base_rate']:.3f}; "
              f"multivariate OOF AUC {block['multivariate_oof_auc_mean']:.3f} "
              f"any univariate > 0.75: {block['any_univariate_above_0.75']}")
        for entry in block["univariate_top10"]:
            print(f"    {entry['feature']}: {entry['auc']:.3f}")
    print(f"\nsummary written: {args.output / 'partC_summary.json'}")


if __name__ == "__main__":
    main()
