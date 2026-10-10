"""Stage 1b-prep-2 analysis: collapse premium, value decomposition,
severity, throttle law, and the wall-speed scan report.

Part A reads only artifacts (Stage 1a + Stage 1b-prep hold arms). Part B
reads the wall-speed scan artifacts produced by
``run_sustained_wall_v1_stage1b_prep2_wall_speed_scan.py`` alongside the
WHOLD_soft (10 m/s) arm, so the four-speed axis is computed by one method.

Reference Arm C per seed is the 1b-prep replayed C (rule-blind t_d = 200
window), which is digest-identical to the stored Stage 1a Arm C.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from experiments import REPO_ROOT
from experiments.idm.analyze_sustained_wall_v1_stage1a import _bootstrap_intervals
from experiments.idm.stage1b_common import series_in_window
from experiments.io import atomic_write_json as _atomic_json

SEEDS = tuple(range(6001, 6031))
STAGE1A_ROOT = REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1a_m2200_p20_s6001_6030"
HOLD_ROOT = REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_hold_m2200_p20_s6001_6030"
SCAN_ROOT = (
    REPO_ROOT
    / "runs"
    / "probe"
    / "sustained_wall_v1_stage1b_prep2_wallspeed_scan_m2200_p20_s6001_6030"
)
OUTPUT_DIR = REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep2"
APPROACH_EDGE = "1191554636"


def _load_seed(root: Path, seed: int) -> Mapping[str, Any]:
    return json.loads((root / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8"))


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray([float(v) for v in values], dtype=float)
    if array.size == 0:
        return {}
    return {
        "median": float(np.median(array)),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _ols_slope(x: Sequence[float], y: Sequence[float]) -> float:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    return float(np.cov(xv, yv)[0, 1] / np.var(xv))


def _bootstrap_ols_slope(
    x: Sequence[float], y: Sequence[float], resamples: int = 2000
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(20260902)
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    slopes = []
    indices = rng.integers(0, xv.size, size=(resamples, xv.size))
    for row in indices:
        xs, ys = xv[row], yv[row]
        if np.var(xs) <= 0.0:
            continue
        slopes.append(float(np.cov(xs, ys)[0, 1] / np.var(xs)))
    lo90, hi90 = np.percentile(slopes, 5), np.percentile(slopes, 95)
    lo95, hi95 = np.percentile(slopes, 2.5), np.percentile(slopes, 97.5)
    return {
        "ci90": {"lo": float(lo90), "hi": float(hi90)},
        "ci95": {"lo": float(lo95), "hi": float(hi95)},
    }


def _merge_edge_flow(branch: Mapping[str, Any], step_from: int, step_to: int) -> float:
    """Mean merge-edge discharge flow in veh/h over the window."""
    steps = _series_window_steps(branch["series"]["step"], step_from, step_to)
    if not steps:
        return float("nan")
    count = series_in_window(branch["series"], "discharge_count", step_from, step_to)
    return float(count / len(steps) * 3600.0)


def _series_window_steps(step_list: Sequence[int], step_from: int, step_to: int) -> list[int]:
    return [int(s) for s in step_list if step_from <= int(s) <= step_to]


# -- Part A ---------------------------------------------------------------
def part_a() -> Dict[str, Any]:
    rows = []
    for seed in SEEDS:
        hold = _load_seed(HOLD_ROOT, seed)
        c = hold["arms"]["C"]
        soft = hold["arms"]["WHOLD_SOFT"]
        c_inc = bool(c["health_collapse"]["incidence"])
        w_inc = bool(soft["health_collapse"]["incidence"])
        if c_inc and not w_inc:
            category = "prevented"
        elif c_inc and w_inc:
            category = "co_collapsing"
        elif not c_inc and w_inc:
            category = "induced"
        else:
            category = "never_collapsing"
        rows.append(
            {
                "seed": seed,
                "category": category,
                "C_incidence": c_inc,
                "W_incidence": w_inc,
                "C_delay_s_per_veh": float(c["delay"]["mean_per_vehicle_delay_s"]),
                "C_total_duration_s": float(c["health_collapse"]["total_duration_s"]),
                "soft_delay_s_per_veh": float(soft["delay"]["mean_per_vehicle_delay_s"]),
                "Delta_delay_s_per_veh": float(soft["delay"]["mean_per_vehicle_delay_s"])
                - float(c["delay"]["mean_per_vehicle_delay_s"]),
                "Delta_total_duration_s": float(soft["health_collapse"]["total_duration_s"])
                - float(c["health_collapse"]["total_duration_s"]),
                "C_onset": c["health_collapse"]["first_onset_step"],
                "h150": float(c["series"]["h"][150]),
                "h200": float(c["series"]["h"][200]),
                "C_merge_flow_300_900": _merge_edge_flow(c, 300, 900),
                "soft_merge_flow_300_900": _merge_edge_flow(soft, 300, 900),
            }
        )

    collapsing = [r for r in rows if bool(r["C_incidence"])]
    non_collapsing = [r for r in rows if not bool(r["C_incidence"])]
    collapse_delay = [r["C_delay_s_per_veh"] for r in collapsing]
    non_delay = [r["C_delay_s_per_veh"] for r in non_collapsing]
    premium = {
        "n_collapsing": len(collapsing),
        "n_non_collapsing": len(non_collapsing),
        "collapsing_mean": float(np.mean(collapse_delay)),
        "collapsing_median": float(np.median(collapse_delay)),
        "non_collapsing_mean": float(np.mean(non_delay)),
        "non_collapsing_median": float(np.median(non_delay)),
        "premium_mean": float(np.mean(collapse_delay) - np.mean(non_delay)),
        "premium_median": float(np.median(collapse_delay) - np.median(non_delay)),
    }
    slope_per_s = _ols_slope(
        [r["C_total_duration_s"] for r in rows], [r["C_delay_s_per_veh"] for r in rows]
    )
    slope_bootstrap = _bootstrap_ols_slope(
        [r["C_total_duration_s"] for r in rows], [r["C_delay_s_per_veh"] for r in rows]
    )
    regression = {
        "slope_s_per_veh_per_s_collapse": slope_per_s,
        "slope_s_per_veh_per_100s_collapse": slope_per_s * 100.0,
        "slope_bootstrap_per_100s": {
            key: {b: float(v * 100.0) for b, v in bounds.items()}
            for key, bounds in slope_bootstrap.items()
        },
    }
    by_category: Dict[str, Dict[str, Any]] = {}
    for category in ("prevented", "co_collapsing", "induced", "never_collapsing"):
        subset = [r for r in rows if r["category"] == category]
        by_category[category] = {
            "n": len(subset),
            "seeds": [r["seed"] for r in subset],
            "mean_Delta_delay": float(np.mean([r["Delta_delay_s_per_veh"] for r in subset])),
            "median_Delta_delay": float(np.median([r["Delta_delay_s_per_veh"] for r in subset])),
            "mean_Delta_duration": float(
                np.mean([r["Delta_total_duration_s"] for r in subset])
            ),
        }
    engagement_cost = by_category["never_collapsing"]["mean_Delta_delay"]
    prevented_mean = by_category["prevented"]["mean_Delta_delay"]

    # A3 listing
    a3 = {}
    for category in ("prevented", "co_collapsing"):
        subset = [r for r in rows if r["category"] == category]
        a3[category] = {
            "n": len(subset),
            "medians": {
                "h200": float(np.median([r["h200"] for r in subset])),
                "h150": float(np.median([r["h150"] for r in subset])),
                "C_onset": float(np.median([r["C_onset"] for r in subset])),
                "C_delay": float(np.median([r["C_delay_s_per_veh"] for r in subset])),
            },
            "listing": [
                {
                    "seed": r["seed"],
                    "h200": round(r["h200"], 4),
                    "h150": round(r["h150"], 4),
                    "C_onset": r["C_onset"],
                    "C_delay_s_per_veh": round(r["C_delay_s_per_veh"], 3),
                    "Delta_delay": round(r["Delta_delay_s_per_veh"], 3),
                }
                for r in subset
            ],
        }

    # A4 severity (WHOLD_soft approach-edge event starts)
    decels_all = []
    decels_w_only = []
    for seed in SEEDS:
        hold = _load_seed(HOLD_ROOT, seed)
        c_event_steps = {
            int(e["relative_step"])
            for e in hold["arms"]["C"]["outcomes"]["hard_brake_attribution"]["event_starts"]
        }
        for event in hold["arms"]["WHOLD_SOFT"]["outcomes"]["hard_brake_attribution"]["event_starts"]:
            if str(event.get("road_id")) != APPROACH_EDGE:
                continue
            try:
                decel = float(event["acceleration_mps2"])
            except (KeyError, TypeError, ValueError):
                continue
            decels_all.append(decel)
            if int(event["relative_step"]) not in c_event_steps:
                decels_w_only.append(decel)
    a4 = {
        "stored": bool(decels_all),
        "note": "event-start acceleration_mps2 field from hard_brake_attribution",
        "approach_edge_WHOLD_soft_event_starts": {
            "p10": float(np.percentile(decels_all, 10)) if decels_all else None,
            "p50": float(np.percentile(decels_all, 50)) if decels_all else None,
            "p90": float(np.percentile(decels_all, 90)) if decels_all else None,
            "n": len(decels_all),
            "fraction_below_minus_6": float(np.mean(np.asarray(decels_all) <= -6.0))
            if decels_all
            else None,
        },
        "approach_edge_WHOLD_soft_only_event_starts": {
            "p10": float(np.percentile(decels_w_only, 10)) if decels_w_only else None,
            "p50": float(np.percentile(decels_w_only, 50)) if decels_w_only else None,
            "p90": float(np.percentile(decels_w_only, 90)) if decels_w_only else None,
            "n": len(decels_w_only),
            "fraction_below_minus_6": float(np.mean(np.asarray(decels_w_only) <= -6.0))
            if decels_w_only
            else None,
        },
    }

    # A5 throttle law
    flow_c = [r["C_merge_flow_300_900"] for r in rows]
    flow_soft = [r["soft_merge_flow_300_900"] for r in rows]
    throttle = {
        "C_median_merge_flow_veh_h_300_900": float(np.median(flow_c)),
        "WHOLD_soft_median_merge_flow_veh_h_300_900": float(np.median(flow_soft)),
        "median_delta_veh_h": float(np.median(np.asarray(flow_soft) - np.asarray(flow_c))),
        "per_seed": [
            {"seed": r["seed"], "C": round(flow_c[i], 1), "soft": round(flow_soft[i], 1)}
            for i, r in enumerate(rows)
        ],
    }

    return {
        "schema": "sustained_wall_v1_stage1b_prep2_part_a/1",
        "premium": premium,
        "regression": regression,
        "decomposition": {
            "by_category": by_category,
            "engagement_cost_mean_Delta_delay_never_collapsing": engagement_cost,
            "prevented_mean_Delta_delay_negative": bool(prevented_mean < 0.0),
            "prevented_mean_Delta_delay": prevented_mean,
        },
        "a3": a3,
        "a4_severity": a4,
        "a5_throttle_law": throttle,
        "rows": rows,
    }


# -- Part B ---------------------------------------------------------------
def _arm_block(
    payloads: Sequence[Mapping[str, Any]],
    arm: str,
) -> Dict[str, Any]:
    durations = []
    longest = []
    onset_delays = []
    delay_deltas = []
    hb_total = []
    lttc_total = []
    phase_deltas = {"during_wall": [], "release_60": [], "after_60": []}
    location_deltas: Dict[str, list[float]] = {}
    member_seconds = []
    coverage_means = []
    coverage_full_fraction = []
    flow_c = []
    flow_arm = []
    incidence_c = []
    incidence_a = []
    speeds_approach: list[float] = []
    for payload in payloads:
        c = payload["arms"]["C"]
        branch = payload["arms"][arm]
        release_step = int(
            (payload.get("wall_dose") or {}).get(arm, {}).get("release_step") or 201
        )
        hb_c = float(c["outcomes"]["hard_brake_vehicle_steps"])
        hb_a = float(branch["outcomes"]["hard_brake_vehicle_steps"])
        hb_total.append(hb_a - hb_c)
        lttc_c = float(c["outcomes"]["low_ttc_exposure_steps"])
        lttc_a = float(branch["outcomes"]["low_ttc_exposure_steps"])
        lttc_total.append(lttc_a - lttc_c)
        durations.append(
            float(branch["health_collapse"]["total_duration_s"])
            - float(c["health_collapse"]["total_duration_s"])
        )
        longest.append(
            float(branch["health_collapse"]["longest_run_s"])
            - float(c["health_collapse"]["longest_run_s"])
        )
        delay_deltas.append(
            float(branch["delay"]["mean_per_vehicle_delay_s"])
            - float(c["delay"]["mean_per_vehicle_delay_s"])
        )
        phase_c = _phase(hb_c, c["series"], release_step)
        phase_a = _phase(hb_a, branch["series"], release_step)
        for key in ("during_wall", "release_60", "after_60"):
            phase_deltas[key].append(phase_a[key] - phase_c[key])
        loc_c = c["outcomes"]["hard_brake_attribution"]["exposure_counts"]["road_id"]
        loc_a = branch["outcomes"]["hard_brake_attribution"]["exposure_counts"]["road_id"]
        for road in sorted(set(loc_c) | set(loc_a)):
            location_deltas.setdefault(str(road), []).append(
                float(loc_a.get(road, 0)) - float(loc_c.get(road, 0))
            )
        lt_c = c["outcomes"]["low_ttc_attribution"]["exposure_counts"]["road_id"]
        lt_a = branch["outcomes"]["low_ttc_attribution"]["exposure_counts"]["road_id"]
        for road in sorted(set(lt_c) | set(lt_a)):
            location_deltas.setdefault(f"lttc::{road}", []).append(
                float(lt_a.get(road, 0)) - float(lt_c.get(road, 0))
            )
        for event in branch["outcomes"]["hard_brake_attribution"]["event_starts"]:
            if str(event.get("road_id")) == APPROACH_EDGE:
                try:
                    speeds_approach.append(float(event["speed_mps"]))
                except (TypeError, ValueError):
                    continue
        dose = payload["wall_dose"][arm]
        member_seconds.append(float(dose.get("member_seconds", 0.0)))
        coverage_means.append(float(dose["coverage_statistics"]["mean"]))
        coverage_full_fraction.append(float(dose.get("coverage_full_fraction", float("nan"))))
        inc_c = bool(c["health_collapse"]["incidence"])
        inc_a = bool(branch["health_collapse"]["incidence"])
        incidence_c.append(inc_c)
        incidence_a.append(inc_a)
        onset_c = c["health_collapse"]["first_onset_step"]
        onset_a = branch["health_collapse"]["first_onset_step"]
        if onset_c is not None and onset_a is not None:
            onset_delays.append(float(onset_a - onset_c))
        flow_c.append(_merge_edge_flow(c, 300, 900))
        flow_arm.append(_merge_edge_flow(branch, 300, 900))
    reductions = [
        float(incidence_c[i]) - float(incidence_a[i]) for i in range(len(incidence_c))
    ]
    return {
        "arm_key": arm,
        "incidence": {
            "C": int(sum(incidence_c)),
            arm: int(sum(incidence_a)),
            "paired_reduction_mean": float(np.mean(reductions)),
            "bootstrap": _bootstrap_intervals(reductions),
            "prevented": [
                int(payloads[i]["seed"])
                for i in range(len(payloads))
                if incidence_c[i] and not incidence_a[i]
            ],
            "induced": [
                int(payloads[i]["seed"])
                for i in range(len(payloads))
                if not incidence_c[i] and incidence_a[i]
            ],
        },
        "total_duration_delta_s": {
            **_stats(durations),
            "bootstrap_median_ci": _bootstrap_intervals(durations).get("median"),
        },
        "longest_run_delta_s": {
            **_stats(longest),
            "bootstrap_median_ci": _bootstrap_intervals(longest).get("median"),
        },
        "onset_delay_among_co_collapsing_s": {
            **_stats(onset_delays),
        },
        "delay_delta_s_per_veh": {
            **_stats(delay_deltas),
            "bootstrap_median_ci": _bootstrap_intervals(delay_deltas).get("median"),
            "fraction_beneficial": float(np.mean(np.asarray(delay_deltas) < 0.0)),
        },
        "hard_brake_delta_steps": {
            **_stats(hb_total),
            "phase_delta": {key: _stats(v) for key, v in phase_deltas.items()},
            "location_delta": {
                road: _stats(v) for road, v in sorted(location_deltas.items())
            },
        },
        "low_ttc_delta_steps": {**_stats(lttc_total)},
        "approach_edge_event_start_speed_mps": {
            "n": len(speeds_approach),
            **_stats(speeds_approach),
        },
        "merge_edge_flow_300_900_veh_h": {
            "C_median": float(np.median(flow_c)),
            f"{arm}_median": float(np.median(flow_arm)),
            "median_delta": float(np.median(np.asarray(flow_arm) - np.asarray(flow_c))),
        },
        "dose": {
            "member_seconds": _stats(member_seconds),
            "per_seed_mean_lane_coverage": _stats(coverage_means),
            "fraction_full_coverage_window": _stats(coverage_full_fraction),
        },
    }


def _phase(_baseline: float, series: Mapping[str, Any], release_step: int) -> Dict[str, float]:
    return {
        "during_wall": series_in_window(series, "hard_brake_count", 201, release_step - 1),
        "release_60": series_in_window(series, "hard_brake_count", release_step, release_step + 60),
        "after_60": series_in_window(series, "hard_brake_count", release_step + 61, 10 ** 6),
    }


def part_b() -> Dict[str, Any]:
    hold_payloads = [_load_seed(HOLD_ROOT, seed) for seed in SEEDS]
    scan_payloads = [_load_seed(SCAN_ROOT, seed) for seed in SEEDS]
    arms = {
        "W10 (WHOLD_soft)": _arm_block(hold_payloads, "WHOLD_SOFT"),
        "W15": _arm_block(scan_payloads, "W15"),
        "W18": _arm_block(scan_payloads, "W18"),
        "W21": _arm_block(scan_payloads, "W21"),
    }
    w10 = arms["W10 (WHOLD_soft)"]
    protection_line = {}
    for arm_name, block in arms.items():
        ci90 = block["incidence"]["bootstrap"]["mean"]["ci90"]
        protection_line[arm_name] = {
            "ci90_lower_bound_incidence_reduction": ci90["lo"],
            "protection_retained_relative_to_W10": bool(ci90["lo"] >= 0.15),
        }
    hb_w10_median = w10["hard_brake_delta_steps"]["location_delta"].get(APPROACH_EDGE, {}).get("median")
    ratio_line = {}
    for arm_name, block in arms.items():
        hb_median = block["hard_brake_delta_steps"]["location_delta"].get(APPROACH_EDGE, {}).get("median")
        ratio_line[arm_name] = {
            "approach_edge_hb_excess_median": hb_median,
            "ratio_to_W10": (float(hb_median) / float(hb_w10_median)) if hb_w10_median else None,
        }
    delay_line = {}
    for arm_name, block in arms.items():
        delay_line[arm_name] = {
            "delay_median": block["delay_delta_s_per_veh"]["median"],
            "ratio_to_W10": (
                float(block["delay_delta_s_per_veh"]["median"])
                / float(w10["delay_delta_s_per_veh"]["median"])
            )
            if w10["delay_delta_s_per_veh"]["median"]
            else None,
        }
    return {
        "schema": "sustained_wall_v1_stage1b_prep2_part_b/1",
        "arms": arms,
        "descriptive_paragraph": {
            "protection_retained": protection_line,
            "approach_edge_hb_excess": ratio_line,
            "delay": delay_line,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, default=SCAN_ROOT)
    parser.add_argument(
        "--output", type=Path, default=str(OUTPUT_DIR)
    )
    args = parser.parse_args()
    SCAN_ROOT_GLOBAL = args.scan_root.resolve()  # noqa: F841; part_b reads module constant
    a = part_a()
    a1 = a["premium"]
    print("=== A1 premium ===")
    print(json.dumps(a1, indent=1))
    print("slope per 100 s:", a["regression"]["slope_s_per_veh_per_100s_collapse"],
          a["regression"]["slope_bootstrap_per_100s"]["ci90"])
    print("=== A2 decomposition ===")
    print(json.dumps(a["decomposition"], indent=1))
    print("=== A3 medians ===")
    for cat, block in a["a3"].items():
        print(cat, block["medians"])
    print("=== A4 severity ===")
    print(json.dumps(a["a4_severity"], indent=1))
    print("=== A5 throttle ===")
    print(
        json.dumps(
            {
                "C_median": a["a5_throttle_law"]["C_median_merge_flow_veh_h_300_900"],
                "soft_median": a["a5_throttle_law"]["WHOLD_soft_median_merge_flow_veh_h_300_900"],
                "median_delta": a["a5_throttle_law"]["median_delta_veh_h"],
            }
        )
    )
    # Part B only when scan artifacts exist
    if (args.scan_root / "seeds").exists():
        b = part_b()
        for arm_name, block in b["arms"].items():
            arm_key = block["arm_key"]
            print("====", arm_name)
            print("incidence:", block["incidence"]["C"], "->", block["incidence"][arm_key],
                  "red", block["incidence"]["paired_reduction_mean"], "ci90:",
                  block["incidence"]["bootstrap"]["mean"]["ci90"])
            print("delay med:", block["delay_delta_s_per_veh"]["median"])
            print("hb approach med:", block["hard_brake_delta_steps"]["location_delta"].get(APPROACH_EDGE, {}).get("median"))
            print("flow:", block["merge_edge_flow_300_900_veh_h"])
        print(json.dumps(b["descriptive_paragraph"], indent=1))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_json(OUTPUT_DIR / "prep2_part_a.json", a)
    if (args.scan_root / "seeds").exists():
        b = part_b()
        _atomic_json(OUTPUT_DIR / "prep2_summary.json", {**a, **b})


if __name__ == "__main__":
    main()
