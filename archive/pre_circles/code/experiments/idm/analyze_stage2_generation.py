"""Stage 2 generation analysis: descriptive report over the final population.

Restricted to the registered stage2_generation seeds (8001-8240 p20,
8241-8360 p30). Descriptive only - no fitting. Covers:

1. inventory, integrity, repeat identity, no-onset count, t_on distribution
   by 100 veh/h hidden-demand bin;
2. per demand bin x stratum: fraction beneficial (Delta_delay < 0) and
   median Delta_delay for each hold (total s and s/veh), best hold per bin
   by mean Delta_delay with bootstrap CI90 (10k resamples, seed 7), and the
   value of choosing duration (best_k minus H450, as recorded per seed);
3. fraction of seeds where the best hold is beneficial and H450 is not,
   and vice versa;
4. p20 vs p30: boundary-bin and best-hold shift with penetration;
5. onset-prediction check: univariate AUC of the top five features at
   t_on - 60 for "onset within 60 s" (pool = pre-onset snapshots: the
   t_on-60 rows, label 1 by construction, plus t=150 rows with onset more
   than 60 s away as candidates; AUC over that pooled row set) and at
   t = 150 for "onset within 100 s" (t=150 rows with onset > 150 only).

Delta delay is arm minus D in TOTAL branch seconds and s/veh (total /
distinct vehicles seen in the branch). Completions = merge-edge exits
within the branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import rankdata

from experiments import REPO_ROOT
from experiments.io import atomic_write_json as _atomic_json

PROBE = REPO_ROOT / "runs" / "probe"
OUTPUT = PROBE / "stage2_generation"
STRATA = {"p20": tuple(range(8001, 8241)), "p30": tuple(range(8241, 8361))}
HOLD_ARMS = ("WON_H300", "WON_H450", "WON_H650")
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 7


def _load_stratum(strate: str) -> List[Mapping[str, Any]]:
    lo, hi = STRATA[strate][0], STRATA[strate][-1]
    dirs = sorted(PROBE.glob(f"stage2_generation_{strate}_s*/seeds"))
    if not dirs:
        return []
    rows = []
    for seed in STRATA[strate]:
        path = dirs[0] / f"seed{seed}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _bootstrap_ci(values: Sequence[float], *, n: int = BOOTSTRAP_N,
                  seed: int = BOOTSTRAP_SEED) -> Dict[str, Optional[float]]:
    arr = np.asarray([float(v) for v in values], dtype=float)
    if arr.size == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    stats = arr[idx].mean(axis=1)
    return {"mean": float(arr.mean()),
            "ci_lo": float(np.quantile(stats, 0.05)),
            "ci_hi": float(np.quantile(stats, 0.95))}


def _auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels)
    pos, neg = scores_arr[labels_arr == 1], scores_arr[labels_arr == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = rankdata(list(pos) + list(neg))
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _delay(row: Mapping[str, Any]) -> Dict[str, float]:
    delay = row["delay"]
    distinct = float(delay["distinct_vehicle_count"])
    total = float(delay["incremental_delay_s"])
    return {"total": total, "s_per_veh": total / distinct if distinct else 0.0,
            "completions": float(delay["completed_vehicles"])}


def _demand_bin(demand: int) -> str:
    return f"{(int(demand) // 100) * 100}"


def _bin_report(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-demand-bin descriptive block over seeds with a full arm set."""
    usable = [r for r in rows if r.get("onset_step") is not None
              and all(a in r.get("arms", {}) for a in HOLD_ARMS)]
    out: Dict[str, Any] = {"n_total": len(rows), "n_with_onset": len(usable)}
    for arm in HOLD_ARMS:
        d_tot = [(_delay(r["arms"][arm])["total"] - _delay(r["arms"]["D"])["total"])
                 for r in usable]
        d_sveh = [(_delay(r["arms"][arm])["s_per_veh"] - _delay(r["arms"]["D"])["s_per_veh"])
                  for r in usable]
        out[arm] = {
            "delta_delay_total_s": {**_bootstrap_ci(d_tot),
                                    "median": median(d_tot) if d_tot else None,
                                    "fraction_beneficial": (sum(1 for x in d_tot if x < 0)
                                                            / len(d_tot)) if d_tot else None},
            "delta_delay_s_per_veh": {**_bootstrap_ci(d_sveh),
                                      "median": median(d_sveh) if d_sveh else None},
        }
    best_by_mean, best_mean = None, None
    for arm in HOLD_ARMS:
        m = out[arm]["delta_delay_total_s"]["mean"]
        if m is None:
            continue
        if best_mean is None or m < best_mean:
            best_by_mean, best_mean = arm, m
    out["best_hold_by_mean"] = {"arm": best_by_mean, "mean_delta_delay_total_s": best_mean}
    # Value of choosing duration (recorded per seed; best_k minus H450).
    vals_total = [r["targets"]["duration_choice_value_total_s"] for r in usable
                  if r.get("targets")]
    vals_sveh = [r["targets"]["duration_choice_value_s_per_veh"] for r in usable
                 if r.get("targets")]
    out["duration_choice_value_total_s"] = {
        **_bootstrap_ci(vals_total), "median": median(vals_total) if vals_total else None}
    out["duration_choice_value_s_per_veh"] = {
        **_bootstrap_ci(vals_sveh), "median": median(vals_sveh) if vals_sveh else None}
    # Best-hold vs H450 discordance (per seed, by total s).
    best_beneficial_h450_not = 0
    h450_beneficial_best_not = 0
    for r in usable:
        deltas = {arm: _delay(r["arms"][arm])["total"] - _delay(r["arms"]["D"])["total"]
                  for arm in HOLD_ARMS}
        best = min(deltas, key=deltas.get)
        if deltas[best] < 0 <= deltas["WON_H450"]:
            best_beneficial_h450_not += 1
        if deltas["WON_H450"] < 0 <= deltas[best]:
            h450_beneficial_best_not += 1
    n = len(usable)
    out["discordance"] = {
        "n": n,
        "best_beneficial_h450_not_fraction": best_beneficial_h450_not / n if n else None,
        "h450_beneficial_best_not_fraction": h450_beneficial_best_not / n if n else None,
    }
    return out


def _t_on_distribution(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    by_bin: Dict[str, List[int]] = {}
    for r in rows:
        if r.get("onset_step") is None:
            continue
        by_bin.setdefault(_demand_bin(r["hidden_demand_vph"]), []).append(int(r["onset_step"]))
    return {
        bin_name: {
            "n": len(v),
            "min": min(v), "p25": float(np.quantile(v, 0.25)),
            "median": median(v), "p75": float(np.quantile(v, 0.75)), "max": max(v),
        }
        for bin_name, v in sorted(by_bin.items())
    }


def _onset_auc(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Univariate AUCs for imminent-onset labels over recorded snapshots."""
    at60_scores: Dict[str, List[float]] = {}
    at60_labels: List[int] = []
    at150_scores: Dict[str, List[float]] = {}
    at150_labels: List[int] = []
    for r in rows:
        onset = r.get("onset_step")
        if onset is None:
            continue
        feats = r.get("features", {})
        # t_on - 60 rows: onset exactly 60 s later (label 1 by construction),
        # pooled against t=150 rows whose onset is more than 60 s away.
        m60 = feats.get("t_on_minus_60")
        if m60 and onset - 60 >= 30:
            for k, v in m60.items():
                at60_scores.setdefault(k, []).append(float(v))
            at60_labels.append(1)
        f150 = feats.get("t150")
        if f150 and onset > 150:
            label60 = 1 if onset - 150 <= 60 else 0
            for k, v in f150.items():
                at60_scores.setdefault(k, []).append(float(v))
            at60_labels.append(label60)
            label100 = 1 if onset - 150 <= 100 else 0
            for k, v in f150.items():
                at150_scores.setdefault(k, []).append(float(v))
            at150_labels.append(label100)
    out: Dict[str, Any] = {}
    for name, scores, labels in (
        ("at_t_on_minus_60_onset_within_60s", at60_scores, at60_labels),
        ("at_t150_onset_within_100s", at150_scores, at150_labels),
    ):
        labels_arr = np.asarray(labels)
        aucs = {k: _auc(v, labels) for k, v in scores.items()}
        aucs = {k: v for k, v in aucs.items() if v is not None and np.isfinite(v)}
        # Direction-aware: a feature below-threshold before onset scores
        # AUC << 0.5 raw; discrimination is max(auc, 1 - auc).
        directed = {k: max(v, 1.0 - v) for k, v in aucs.items()}
        top = sorted(directed.items(), key=lambda kv: -kv[1])[:5]
        out[name] = {
            "n_rows": int(len(labels_arr)),
            "base_rate": float(labels_arr.mean()) if len(labels_arr) else None,
            "top5": [{"feature": k, "auc_direction_aware": float(v),
                      "auc_raw": float(aucs[k])} for k, v in top],
            "any_above_0.75": bool(any(v > 0.75 for _, v in top)),
        }
    return out


def analyze_stratum(strate: str) -> Dict[str, Any]:
    rows = _load_stratum(strate)
    expected = STRATA[strate]
    summary: Dict[str, Any] = {
        "strate": strate,
        "n_seeds": len(rows),
        "missing_seeds": [s for s in expected if s not in {int(r["seed"]) for r in rows}],
        "integrity_failures": [
            {"seed": int(r["seed"]), "arm": a, "check": chk}
            for r in rows for a, chk in r.get("integrity", {}).items()
            if not chk.get("prefix_identical_through_t_on")
        ],
        "repeat_identity_pass": sum(
            1 for r in rows if r.get("repeat_check") and r["repeat_check"]["exact_identity"]),
        "repeat_identity_fail": [
            int(r["seed"]) for r in rows
            if r.get("repeat_check") and not r["repeat_check"]["exact_identity"]],
        "no_onset_seeds": sorted(int(r["seed"]) for r in rows if r.get("no_onset_by_900")),
        "t_on_distribution_by_demand_bin": _t_on_distribution(rows),
        "by_demand_bin": {},
        "pooled": _bin_report(rows),
        "onset_prediction": _onset_auc(rows),
    }
    bins = sorted({_demand_bin(r["hidden_demand_vph"]) for r in rows})
    for bin_name in bins:
        subset = [r for r in rows if _demand_bin(r["hidden_demand_vph"]) == bin_name]
        summary["by_demand_bin"][bin_name] = _bin_report(subset)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--strata", nargs="+", default=["p20"],
                        choices=("p20", "p30"))
    args = parser.parse_args()
    summary: Dict[str, Any] = {"schema": "stage2_generation_summary/1", "strata": {}}
    for strate in args.strata:
        summary["strata"][strate] = analyze_stratum(strate)
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "summary.json"
    _atomic_json(out_path, summary)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    (args.output / "summary.json.sha256").write_text(digest + "\n", encoding="utf-8")
    summary["summary_json_sha256"] = digest
    _atomic_json(out_path, summary)

    for strate, block in summary["strata"].items():
        print(f"== {strate}: n={block['n_seeds']} missing={len(block['missing_seeds'])} "
              f"integrity fails={len(block['integrity_failures'])} "
              f"repeats {block['repeat_identity_pass']} pass / "
              f"{len(block['repeat_identity_fail'])} fail "
              f"no-onset={len(block['no_onset_seeds'])}")
        print("  t_on by demand bin:",
              {b: v["median"] for b, v in block["t_on_distribution_by_demand_bin"].items()})
        for bin_name, bin_block in block["by_demand_bin"].items():
            holds = "  ".join(
                f"{arm.replace('WON_', '')}: "
                f"{bin_block[arm]['delta_delay_total_s']['mean']:+.0f}"
                f"[{bin_block[arm]['delta_delay_total_s']['ci_lo']:+.0f},"
                f"{bin_block[arm]['delta_delay_total_s']['ci_hi']:+.0f}] "
                f"ben={bin_block[arm]['delta_delay_total_s']['fraction_beneficial']:.2f} "
                f"med={bin_block[arm]['delta_delay_total_s']['median']:+.0f}"
                for arm in HOLD_ARMS)
            best = bin_block["best_hold_by_mean"]
            dc = bin_block["duration_choice_value_total_s"]
            print(f"  bin {bin_name} (n={bin_block['n_with_onset']}): {holds}")
            print(f"      best={best['arm']} ({best['mean_delta_delay_total_s']:+.0f}); "
                  f"duration-choice value med={dc['median']:+.0f}s; "
                  f"discordance best-win={bin_block['discordance']['best_beneficial_h450_not_fraction']:.2f} "
                  f"h450-win={bin_block['discordance']['h450_beneficial_best_not_fraction']:.2f}")
        for name, blk in block["onset_prediction"].items():
            print(f"  onset AUC {name}: n={blk['n_rows']} base={blk['base_rate']:.3f} "
                  f">0.75: {blk['any_above_0.75']}")
            for entry in blk["top5"]:
                print(f"    {entry['feature']}: {entry['auc_direction_aware']:.3f} "
                      f"(raw {entry['auc_raw']:.3f})")
    print(f"\nsummary written: {out_path} (sha256 {digest[:16]}...)")


if __name__ == "__main__":
    main()
