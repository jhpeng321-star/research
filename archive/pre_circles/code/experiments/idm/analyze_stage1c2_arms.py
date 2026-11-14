"""Stage 1c-2 analysis: arms vs D, hold-duration curves, best arm, safety.

Per cell (2400/2600/2700; 2200 = WON_H300 false-engagement cost), paired
against the stored Stage 1c D branch (same seeds, same profile; digest
re-verified by the runner):

- Delta incremental delay per arm in TOTAL seconds (branch sum over all
  vehicles) and s/veh (total / distinct vehicles seen in the branch):
  median, mean, bootstrap CI90 (10k resamples over seeds), fraction of
  seeds with any delay loss;
- health-collapse incidence delta and duration delta (h < 0.15 for >= 60 s
  after t = 100; duration = total seconds below threshold);
- completions: vehicles exiting the merge edge within the branch
  (``merge_discharge_count``);
- merge flow 300-1200: merge-edge exit transitions summed over steps
  300-1200 (veh, and the mean veh/h over the 900 s window);
- safety: hard-brake (accel <= -4.5 m/s^2) vehicle-steps and low-TTC
  pair-steps, total and by location x phase for the new arms (the stored
  1c reference arms keep step counts only);
- R1b release-time distribution and plateau/decline/never counts; WON
  engagement-step distribution and the no-onset count;
- the hold-duration curve at t_d = 250: mean Delta delay at hold
  {300, 450, 650 (= 1c W_HOLD), 950 (= 1c W_ORACLE)} with CI90, and the
  onset-triggered curve at {300, 450};
- best arm per cell by mean Delta delay and whether each WON arm is within
  bootstrap CI90 of the best fixed-time arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from experiments import REPO_ROOT
from experiments.io import atomic_write_json as _atomic_json

PROBE = REPO_ROOT / "runs" / "probe"
OUTPUT = PROBE / "stage1c2_arms"
CELLS = {2400: tuple(range(7431, 7461)), 2600: tuple(range(7461, 7491)),
         2700: tuple(range(7491, 7521))}
WON_CELL = {2200: tuple(range(7401, 7431))}
NEW_ARMS = ("W250_H300", "W250_H450", "WON_H300", "WON_H450", "W250_R1b")
REF_ARMS = ("W_HOLD", "W_ORACLE")
PLATEAU_END = 1200
FLOW_WINDOW = (300, 1200)
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 7


def _load_cell_seeds(cell: int, won_only: bool = False):
    pattern = (f"stage1c2_won_m{cell}_p20_s*" if won_only
               else f"stage1c2_arms_m{cell}_p20_s*")
    dirs = sorted(PROBE.glob(pattern))
    if not dirs:
        return {}
    rows = {}
    for seed in (WON_CELL.get(cell) or CELLS[cell]):
        path = dirs[0] / "seeds" / f"seed{seed}.json"
        if path.exists():
            rows[seed] = json.loads(path.read_text(encoding="utf-8"))
    return rows


def _load_stage1c_cell(cell: int):
    dirs = sorted(PROBE.glob(f"stage1c_release_m{cell}_p20_s*"))
    if not dirs:
        return {}
    rows = {}
    for seed in (WON_CELL.get(cell) or CELLS[cell]):
        path = dirs[0] / "seeds" / f"seed{seed}.json"
        if path.exists():
            rows[seed] = json.loads(path.read_text(encoding="utf-8"))
    return rows


def _bootstrap_ci(values: Sequence[float], *, statistic=mean,
                  n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> Dict[str, Optional[float]]:
    arr = np.asarray([float(v) for v in values], dtype=float)
    if arr.size == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    stats = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_lo": float(np.quantile(stats, 0.05)),
        "ci_hi": float(np.quantile(stats, 0.95)),
    }


def _mean_diff_ci(a: Sequence[float], b: Sequence[float], *,
                  n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> Dict[str, Optional[float]]:
    """CI90 on mean(a) - mean(b), paired bootstrap over indices."""
    xa = np.asarray([float(v) for v in a], dtype=float)
    xb = np.asarray([float(v) for v in b], dtype=float)
    if xa.size == 0 or xa.size != xb.size:
        return {"diff": None, "ci_lo": None, "ci_hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, xa.size, size=(n, xa.size))
    diffs = xa[idx].mean(axis=1) - xb[idx].mean(axis=1)
    return {
        "diff": float(xa.mean() - xb.mean()),
        "ci_lo": float(np.quantile(diffs, 0.05)),
        "ci_hi": float(np.quantile(diffs, 0.95)),
    }


def _delay_metrics(row: Mapping[str, Any]) -> Dict[str, float]:
    delay = row["delay"]
    distinct = float(delay["distinct_vehicle_count"])
    total = float(delay["incremental_delay_s"])
    return {
        "total_delay_s": total,
        "delay_s_per_veh": (total / distinct) if distinct else 0.0,
        "completions": float(delay["completed_vehicles"]),
        "distinct_vehicles": distinct,
    }


def _merge_flow_300_1200(row: Mapping[str, Any]) -> Dict[str, float]:
    counts = row["series"]["discharge_count"]
    lo, hi = FLOW_WINDOW
    total = float(sum(counts[lo:hi + 1]))
    return {"veh": total, "veh_per_h_mean": total * 3600.0 / (hi - lo + 1)}


def _phase_safety_totals(row: Mapping[str, Any]) -> Dict[str, int]:
    series = row["series"]
    engage = int(row.get("engage_step") or 0)
    release = row.get("release_step")
    hb = series["hard_brake_count"]
    ltc = series["low_ttc_count"]
    out = {"hard_brake_pre_engagement": sum(hb[:engage + 1]),
           "hard_brake_treatment": 0, "hard_brake_post": 0,
           "low_ttc_pre_engagement": sum(ltc[:engage + 1]),
           "low_ttc_treatment": 0, "low_ttc_post": 0}
    for step in range(engage + 1, len(hb)):
        in_treatment = release is None or step <= (int(release) + 5)
        if in_treatment:
            out["hard_brake_treatment"] += hb[step]
            out["low_ttc_treatment"] += ltc[step]
        else:
            out["hard_brake_post"] += hb[step]
            out["low_ttc_post"] += ltc[step]
    return out


def _arm_report(cell_rows: Mapping[int, Mapping[str, Any]],
                ref_rows: Mapping[int, Mapping[str, Any]],
                arm: str, *, is_ref: bool) -> Dict[str, Any]:
    def get_row(seed: int) -> Optional[Mapping[str, Any]]:
        if is_ref:
            row = ref_rows.get(seed, {}).get("arms", {}).get(arm)
        else:
            row = cell_rows.get(seed, {}).get("arms", {}).get(arm)
        return row

    seeds = sorted(cell_rows.keys())
    d_total, d_sveh, a_total, a_sveh, a_comp, d_comp = [], [], [], [], [], []
    a_flow, d_flow = [], []
    incidence, duration_delta, onset_steps, release_steps = [], [], [], []
    hb_excess, ltc_excess = [], []
    n_missing = 0
    for seed in seeds:
        d_arm = ref_rows.get(seed, {}).get("arms", {}).get("D")
        a_arm = get_row(seed)
        if a_arm is None:
            n_missing += 1
            continue
        dm = _delay_metrics(d_arm)
        am = _delay_metrics(a_arm)
        d_total.append(dm["total_delay_s"])
        a_total.append(am["total_delay_s"])
        d_sveh.append(dm["delay_s_per_veh"])
        a_sveh.append(am["delay_s_per_veh"])
        d_comp.append(dm["completions"])
        a_comp.append(am["completions"])
        d_flow.append(_merge_flow_300_1200(d_arm)["veh"])
        a_flow.append(_merge_flow_300_1200(a_arm)["veh"])
        if not is_ref:
            onset_steps.append(cell_rows[seed].get("onset_step"))
        if not is_ref or arm in REF_ARMS:
            release_steps.append(a_arm.get("release_step"))
        inc_a = a_arm["health_collapse"]
        inc_d = d_arm["health_collapse"]
        incidence.append((bool(inc_a["incidence"]), bool(inc_d["incidence"])))
        duration_delta.append(float(inc_a["total_duration_s"]) - float(inc_d["total_duration_s"]))
        hb_excess.append(int(a_arm["outcomes"]["hard_brake_vehicle_steps"])
                         - int(d_arm["outcomes"]["hard_brake_vehicle_steps"]))
        ltc_excess.append(int(a_arm["outcomes"]["low_ttc_exposure_steps"])
                          - int(d_arm["outcomes"]["low_ttc_exposure_steps"]))
    delta_total = [a - d for a, d in zip(a_total, d_total)]
    delta_sveh = [a - d for a, d in zip(a_sveh, d_sveh)]
    delta_comp = [a - d for a, d in zip(a_comp, d_comp)]
    report: Dict[str, Any] = {
        "n": len(delta_total),
        "n_missing": n_missing,
        "delta_delay_total_s": {
            **_bootstrap_ci(delta_total),
            "median": median(delta_total) if delta_total else None,
            "fraction_beneficial": (sum(1 for x in delta_total if x < 0) / len(delta_total))
            if delta_total else None,
        },
        "delta_delay_s_per_veh": {
            **_bootstrap_ci(delta_sveh),
            "median": median(delta_sveh) if delta_sveh else None,
        },
        "delta_completions": {
            **_bootstrap_ci(delta_comp),
            "median": median(delta_comp) if delta_comp else None,
        },
        "merge_flow_300_1200_veh_delta": _bootstrap_ci(
            [a - d for a, d in zip(a_flow, d_flow)]),
        "delta_collapse_incidence": sum(
            1 for a, d in incidence if a and not d),
        "collapses_recovered": sum(1 for a, d in incidence if not a and d),
        "delta_duration_s": {
            **_bootstrap_ci(duration_delta),
            "median": median(duration_delta) if duration_delta else None,
        },
        "hard_brake_excess_vehicle_steps_median": median(hb_excess) if hb_excess else None,
        "low_ttc_excess_steps_median": median(ltc_excess) if ltc_excess else None,
        "onset_step_median": median([s for s in onset_steps if s is not None])
        if any(s is not None for s in onset_steps) else None,
        "release_step_median": median([s for s in release_steps if s is not None])
        if any(s is not None for s in release_steps) else None,
    }
    return report


def _location_safety(cell_rows: Mapping[int, Mapping[str, Any]], arm: str) -> Dict[str, Any]:
    totals: Dict[str, Dict[str, int]] = {}
    n = 0
    for row in cell_rows.values():
        events = row.get("arms", {}).get(arm, {}).get("safety_by_location")
        if not events:
            continue
        n += 1
        for kind in ("hard_brake", "low_ttc"):
            for location, phases in events.get(kind, {}).items():
                for phase, count in phases.items():
                    key = f"{kind}:{location}:{phase}"
                    totals.setdefault(kind, {}).setdefault(location, {}).setdefault(
                        phase, 0)
                    totals[kind][location][phase] += int(count)
    return {"n_seeds_aggregated": n, "totals": totals}


def analyze_cell(cell: int, *, won_only: bool = False) -> Dict[str, Any]:
    cell_rows = _load_cell_seeds(cell, won_only=won_only)
    ref_rows = _load_stage1c_cell(cell)
    arms = ["WON_H300"] if won_only else list(NEW_ARMS)
    result: Dict[str, Any] = {
        "cell": cell,
        "n_seeds": len(cell_rows),
        "missing_seeds": [s for s in (WON_CELL.get(cell) or CELLS[cell])
                          if s not in cell_rows],
        "digest_mismatches": [s for s, r in cell_rows.items()
                              if not r.get("stored_d_digest_match")],
        "repeat_failures": [s for s, r in cell_rows.items()
                            if r.get("repeat_check") is not None
                            and not r["repeat_check"]["exact_identity"]],
        "integrity_failures": [
            {"seed": s, "arm": a, "check": chk}
            for s, r in cell_rows.items() for a, chk in r.get("integrity", {}).items()
            if not chk.get("prefix_identical_through_t_d")
        ],
        "no_onset_seeds": sorted(s for s, r in cell_rows.items() if r.get("no_onset_by_900")),
        "arms": {},
    }
    onsets = [r.get("onset_step") for r in cell_rows.values() if r.get("onset_step") is not None]
    result["onset_distribution"] = {
        "n": len(onsets),
        "min": min(onsets) if onsets else None,
        "p25": float(np.quantile(onsets, 0.25)) if onsets else None,
        "median": median(onsets) if onsets else None,
        "p75": float(np.quantile(onsets, 0.75)) if onsets else None,
        "max": max(onsets) if onsets else None,
    }
    for arm in arms:
        is_ref = False
        result["arms"][arm] = _arm_report(cell_rows, ref_rows, arm, is_ref=is_ref)
        if arm in ("W250_R1b",):
            releases = []
            for row in cell_rows.values():
                step = row.get("arms", {}).get(arm, {}).get("release_step")
                if step is not None:
                    releases.append(int(step))
            result["arms"][arm]["release_counts"] = {
                "plateau_le1200": sum(1 for s in releases if s <= PLATEAU_END),
                "decline_1201_1500": sum(1 for s in releases if 1200 < s <= 1500),
                "tail_gt1500": sum(1 for s in releases if s > 1500),
                "by_rule": sum(1 for s in releases if s < 251 + 900),
                "hard_max": sum(1 for s in releases if s >= 251 + 900),
            }
    for arm in REF_ARMS:
        result["arms"][arm] = _arm_report(cell_rows, ref_rows, arm, is_ref=True)
    if not won_only:
        for arm in ("W250_H300", "W250_H450", "WON_H300", "WON_H450", "W250_R1b"):
            result["arms"][arm]["safety_by_location_totals"] = _location_safety(
                cell_rows, arm)

    # Hold-duration curves: mean Delta delay (total s and s/veh) by hold.
    def _curve(arms_and_holds):
        curve = []
        for arm, hold in arms_and_holds:
            rep = result["arms"].get(arm)
            if rep is None or rep["n"] == 0:
                continue
            curve.append({
                "arm": arm, "hold_s": hold,
                "mean_delta_delay_total_s": rep["delta_delay_total_s"]["mean"],
                "mean_delta_delay_s_per_veh": rep["delta_delay_s_per_veh"]["mean"],
                "ci_lo_total_s": rep["delta_delay_total_s"]["ci_lo"],
                "ci_hi_total_s": rep["delta_delay_total_s"]["ci_hi"],
                "n": rep["n"],
            })
        return curve

    if not won_only:
        result["hold_duration_curve_t250"] = _curve([
            ("W250_H300", 300), ("W250_H450", 450), ("W_HOLD", 650), ("W_ORACLE", 950)])
        result["hold_duration_curve_onset"] = _curve([
            ("WON_H300", 300), ("WON_H450", 450)])

        fixed_arms = [a for a in ("W250_H300", "W250_H450", "W_HOLD", "W_ORACLE")
                      if result["arms"].get(a, {}).get("n")]
        best_arm, best_mean = None, None
        for arm in fixed_arms:
            m = result["arms"][arm]["delta_delay_total_s"]["mean"]
            if best_mean is None or m < best_mean:
                best_arm, best_mean = arm, m
        result["best_fixed_arm"] = {"arm": best_arm, "mean_delta_delay_total_s": best_mean}
        result["won_vs_best_fixed"] = {}
        if best_arm is not None:
            source = ref_rows if best_arm in REF_ARMS else cell_rows
            for won in ("WON_H300", "WON_H450"):
                won_vals, best_vals = [], []
                for seed in sorted(cell_rows.keys()):
                    a = cell_rows.get(seed, {}).get("arms", {}).get(won)
                    b = source.get(seed, {}).get("arms", {}).get(best_arm)
                    d_arm = ref_rows.get(seed, {}).get("arms", {}).get("D")
                    if a is None or b is None or d_arm is None:
                        continue
                    d_total_s = _delay_metrics(d_arm)["total_delay_s"]
                    won_vals.append(_delay_metrics(a)["total_delay_s"] - d_total_s)
                    best_vals.append(_delay_metrics(b)["total_delay_s"] - d_total_s)
                result["won_vs_best_fixed"][won] = {
                    "best_fixed_arm": best_arm,
                    **_mean_diff_ci(won_vals, best_vals),
                }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    summary = {"schema": "stage1c2_arms_summary/1", "cells": {}}
    for cell in (2400, 2600, 2700):
        summary["cells"][str(cell)] = analyze_cell(cell)
    summary["cells"]["2200_won_only"] = analyze_cell(2200, won_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output / "summary.json", summary)

    for cell_key, cell_sum in summary["cells"].items():
        print(f"== cell {cell_key}: n={cell_sum['n_seeds']} "
              f"digest mismatches={len(cell_sum['digest_mismatches'])} "
              f"integrity failures={len(cell_sum['integrity_failures'])} "
              f"no-onset seeds={len(cell_sum['no_onset_seeds'])} "
              f"onset med={cell_sum['onset_distribution']['median']}")
        for arm, rep in cell_sum["arms"].items():
            dd = rep["delta_delay_total_s"]
            ds = rep["delta_delay_s_per_veh"]
            print(
                f"  {arm:10s} n={rep['n']} dDelay total {dd['mean']:.0f} "
                f"[{dd['ci_lo']:.0f}, {dd['ci_hi']:.0f}] med {dd['median']:.0f} "
                f"frac_ben={dd['fraction_beneficial']:.2f} | "
                f"s/veh {ds['mean']:+.2f} [{ds['ci_lo']:+.2f}, {ds['ci_hi']:+.2f}] | "
                f"dComp med {rep['delta_completions']['median']:+.0f} | "
                f"dIncidence +{rep['delta_collapse_incidence']}/-{rep['collapses_recovered']} | "
                f"dDuration {rep['delta_duration_s']['mean']:.0f}s | "
                f"HBexcess med {rep['hard_brake_excess_vehicle_steps_median']:+.0f} | "
                f"LTCexcess med {rep['low_ttc_excess_steps_median']:+.0f}"
            )
        for curve_name in ("hold_duration_curve_t250", "hold_duration_curve_onset"):
            if curve_name in cell_sum:
                line = "  ".join(
                    f"{pt['arm']}(h={pt['hold_s']}): {pt['mean_delta_delay_total_s']:.0f} "
                    f"[{pt['ci_lo_total_s']:.0f}, {pt['ci_hi_total_s']:.0f}]"
                    for pt in cell_sum[curve_name])
                print(f"  {curve_name}: {line}")
        if "best_fixed_arm" in cell_sum:
            print(f"  best fixed arm: {cell_sum['best_fixed_arm']['arm']} "
                  f"({cell_sum['best_fixed_arm']['mean_delta_delay_total_s']:.0f} s)")
            for won, cmp in cell_sum["won_vs_best_fixed"].items():
                print(f"  {won} vs best fixed: diff {cmp['diff']:.0f} "
                      f"[{cmp['ci_lo']:.0f}, {cmp['ci_hi']:.0f}]")
        r1b = cell_sum["arms"].get("W250_R1b")
        if r1b and "release_counts" in r1b:
            print(f"  R1b releases: {r1b['release_counts']} "
                  f"release step med={r1b['release_step_median']}")
    print(f"\nsummary written: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
