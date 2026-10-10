"""Stage 1b-prep-4 Part B + C analysis.

Part B: SE250_H120 / SE250_H300 vs Arm C, next to the prep-3 SE250-instant.
Part C: seed-level univariate descriptive AUC of every raw RSU observation
variable for the targets "collapse onset within the next 100/200 s", per
cell and per observation time; top-five tables and the >0.75 check where h
has saturated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from experiments import REPO_ROOT
from experiments.idm.analyze_sustained_wall_v1_stage1a import _bootstrap_intervals
from experiments.idm.stage1b_common import series_in_window
from experiments.io import atomic_write_json as _atomic_json

MINHOLD_ROOT = (
    REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep4_minhold_m2100_p20_s6091_6120"
)
PREP3_SE_ROOT = (
    REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep3_se_m2100_p20_s6091_6120"
)
PREP3_SUMMARY = REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep3" / "prep3_summary.json"
OBSMAP_ROOTS = {
    2000: REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep4_obsmap_m2000_p20_s6061_6090",
    2100: REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep4_obsmap_m2100_p20_s6091_6120",
    2200: REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep4_obsmap_m2200_p20_s6001_6030",
}
CELL_SEEDS = {
    2200: tuple(range(6001, 6031)),
    2100: tuple(range(6091, 6121)),
    2000: tuple(range(6061, 6091)),
}
STAGE1A_SEED_ONSET = None
OUTPUT_DIR = REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep4"
APPROACH_EDGE = "1191554636"
MERGE_EDGE = "1487606179"
DENSITY_GEOM = {
    "upstream": (744.95, 3),
    "approach": (200.0, 3),
    "near_merge": (200.0, 3),
    "merge_edge": (300.0, 3),
}


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray([float(v) for v in values], dtype=float)
    if array.size == 0:
        return {}
    return {
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def part_b() -> Dict[str, Any]:
    payloads = []
    for seed in range(6091, 6121):
        path = MINHOLD_ROOT / "seeds" / f"seed{seed}.json"
        if not path.exists():
            continue
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    if not payloads:
        raise ValueError(f"no minhold artifacts under {MINHOLD_ROOT}")
    arms = {}
    for arm in ("SE250_H120", "SE250_H300"):
        durations = []
        longest = []
        delay_deltas = []
        holds = []
        incidence_c = []
        incidence_a = []
        leads_prevented = []
        leads_co = []
        comfort_loc: Dict[str, list[float]] = {}
        lttc_loc: Dict[str, list[float]] = {}
        flow_c = []
        flow_a = []
        for payload in payloads:
            c = payload["arms"]["C"]
            a = payload["arms"][arm]
            dose = payload["wall_dose"][arm]
            durations.append(
                float(a["health_collapse"]["total_duration_s"])
                - float(c["health_collapse"]["total_duration_s"])
            )
            longest.append(
                float(a["health_collapse"]["longest_run_s"])
                - float(c["health_collapse"]["longest_run_s"])
            )
            delay_deltas.append(
                float(a["delay"]["mean_per_vehicle_delay_s"])
                - float(c["delay"]["mean_per_vehicle_delay_s"])
            )
            holds.append(float(dose["realized_wall_seconds"]))
            inc_c = bool(c["health_collapse"]["incidence"])
            inc_a = bool(a["health_collapse"]["incidence"])
            incidence_c.append(inc_c)
            incidence_a.append(inc_a)
            onset = c["health_collapse"]["first_onset_step"]
            lead = None if onset is None else float(onset - 250)
            if inc_c and not inc_a:
                leads_prevented.append(lead)
            elif inc_c and inc_a:
                leads_co.append(lead)
            loc_c = c["outcomes"]["hard_brake_attribution"]["exposure_counts"]["road_id"]
            loc_a = a["outcomes"]["hard_brake_attribution"]["exposure_counts"]["road_id"]
            for road in sorted(set(loc_c) | set(loc_a)):
                comfort_loc.setdefault(str(road), []).append(
                    float(loc_a.get(road, 0)) - float(loc_c.get(road, 0))
                )
            lt_c = c["outcomes"]["low_ttc_attribution"]["exposure_counts"]["road_id"]
            lt_a = a["outcomes"]["low_ttc_attribution"]["exposure_counts"]["road_id"]
            for road in sorted(set(lt_c) | set(lt_a)):
                lttc_loc.setdefault(str(road), []).append(
                    float(lt_a.get(road, 0)) - float(lt_c.get(road, 0))
                )
            flow_c.append(_flow(c))
            flow_a.append(_flow(a))
        reductions = [
            float(incidence_c[i]) - float(incidence_a[i]) for i in range(len(incidence_c))
        ]
        arms[arm] = {
            "min_hold_s": float(arm.split("_H")[1]),
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
            "duration_delta_s": {**_stats(durations)},
            "longest_run_delta_s": {**_stats(longest)},
            "delay_delta_s_per_veh": {
                **_stats(delay_deltas),
                "bootstrap_median_ci": _bootstrap_intervals(delay_deltas).get("median"),
                "fraction_beneficial": float(np.mean(np.asarray(delay_deltas) < 0.0)),
            },
            "hold_duration_s": {**_stats(holds)},
            "release_reason_counts": {"health_release": 0, "demand_end": 0, "taper_complete": 0},
            "lead_time_s": {
                "prevented": _stats(leads_prevented) if leads_prevented else {},
                "co_collapsing": _stats(leads_co) if leads_co else {},
            },
            "comfort_cost_excess_by_location": {
                road: _stats(values) for road, values in sorted(comfort_loc.items())
            },
            "low_ttc_excess_by_location": {
                road: _stats(values) for road, values in sorted(lttc_loc.items())
            },
            "merge_edge_flow_300_900_veh_h": {
                "C_median": float(np.median(flow_c)),
                f"{arm}_median": float(np.median(flow_a)),
                "median_delta": float(np.median(np.asarray(flow_a) - np.asarray(flow_c))),
            },
        }
    # Nonzero release reasons: labeled by the realized hold versus max hold
    # (900 - 250 = 650 s).
    for arm, block in arms.items():
        max_hold = 650.0
        holds = [float(payload["wall_dose"][arm]["realized_wall_seconds"]) for payload in payloads]
        block["release_reason_counts"] = {
            "health_release": sum(1 for h in holds if h < max_hold),
            "demand_end": sum(1 for h in holds if h >= max_hold),
        }
    prep3_se = {}
    if PREP3_SUMMARY.exists():
        prep3_se = json.loads(PREP3_SUMMARY.read_text(encoding="utf-8")).get("se", {}).get(
            "SE250", {}
        )
    return {
        "schema": "sustained_wall_v1_stage1b_prep4_part_b/1",
        "n": len(payloads),
        "arms": arms,
        "se250_instant_reference": {
            "incidence": prep3_se.get("incidence", {}),
            "delay": prep3_se.get("delay_delta_s_per_veh", {}),
            "hold": prep3_se.get("hold_duration_s", {}),
            "lead": prep3_se.get("lead_time_s", {}),
        },
    }


def _flow(branch: Mapping[str, Any]) -> float:
    steps = [int(s) for s in branch["series"]["step"] if 300 <= int(s) <= 900]
    if not steps:
        return float("nan")
    return float(series_in_window(branch["series"], "discharge_count", 300, 900) / len(steps) * 3600.0)


# -- Part C ---------------------------------------------------------------
def _feature_vector(features: Dict[str, Any], step: int, steps: Sequence[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    regions = features["regions"]
    for region in ("upstream", "approach", "near_merge", "merge_edge", "post_merge", "ramp"):
        stats = regions.get(region, {})
        out[f"{region}_count"] = float(stats.get("count", 0.0))
        out[f"{region}_mean_speed"] = float(stats.get("mean_speed", 0.0))
        out[f"{region}_speed_std"] = float(stats.get("speed_std", 0.0))
        out[f"{region}_harmonic_speed"] = float(stats.get("harmonic_speed", 0.0))
    for region, (length_m, lanes) in DENSITY_GEOM.items():
        count = float(regions.get(region, {}).get("count", 0.0))
        out[f"{region}_density"] = count / (length_m / 1000.0 * lanes) if length_m > 0 else 0.0
    out["mainline_lane_imbalance"] = float(features["mainline_lane_imbalance"])
    out["ramp_queue_length"] = float(features["ramp_queue_length"])
    out["ramp_30s_entries"] = _sum_feature(features, "ramp_entries", step - 30, step)
    out["upstream_30s_inflow"] = _sum_feature(features, "upstream_entries", step - 30, step)
    out["merge_30s_discharge"] = _sum_feature(features, "merge_discharge", step - 30, step)
    out["upstream_inflow_minus_discharge_30s"] = (
        out["upstream_30s_inflow"] - out["merge_30s_discharge"]
    )
    out["cav_count_band"] = float(features["cav_count_band"])
    out["cav_count_upstream"] = float(features["cav_count_upstream"])
    for horizon in (10, 30, 60):
        prev = features.get(str(step - horizon))
        if prev is not None:
            out[f"merge_density_diff_{horizon}s"] = (
                out["merge_edge_density"] - _density_at(prev, "merge_edge")
            )
            out[f"approach_std_diff_{horizon}s"] = (
                out["approach_speed_std"] - _speed_std_at(prev, "approach")
            )
    return out


def _density_at(features: Dict[str, Any], region: str) -> float:
    length_m, lanes = DENSITY_GEOM[region]
    return float(features["regions"].get(region, {}).get("count", 0.0)) / (
        length_m / 1000.0 * lanes
    )


def _speed_std_at(features: Dict[str, Any], region: str) -> float:
    return float(features["regions"].get(region, {}).get("speed_std", 0.0))


def _sum_feature(features: Dict[str, Any], key: str, step_from: int, step_to: int) -> float:
    total = 0.0
    for step in range(step_from, step_to + 1):
        entry = features.get(str(step))
        if entry is not None:
            total += float(entry.get(key, 0.0))
    return total


def _auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    combined = list(pos) + list(neg)
    ranks = rankdata(combined)
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def part_c() -> Dict[str, Any]:
    result: Dict[str, Any] = {"schema": "sustained_wall_v1_stage1b_prep4_part_c/1"}
    for cell, root in OBSMAP_ROOTS.items():
        seeds = CELL_SEEDS[cell]
        if not (root / "seeds").exists():
            continue
        cells_out: Dict[str, Any] = {}
        for horizon in (100, 200):
            per_t: Dict[str, Any] = {}
            for t in (100, 150, 200, 250, 300):
                rows: list[Dict[str, Any]] = []
                seed_rows = []
                for seed in seeds:
                    payload = json.loads(
                        (root / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8")
                    )
                    lookup = payload["step_features"]
                    features = lookup[str(t)]
                    onset = _onset_for(cell, seed)
                    target = onset is not None and t < onset <= t + horizon
                    vector = _feature_vector(features, t, range(40, 311))
                    vector["h"] = float(payload["h_series"][t])
                    rows.append({"seed": seed, "target": bool(target), **vector})
                    seed_rows.append({"seed": seed, "target": bool(target), "onset": onset})
                features = [k for k in rows[0] if k not in ("seed", "target")]
                aucs = {}
                for feature in features:
                    pos = [float(row[feature]) for row in rows if row["target"]]
                    neg = [float(row[feature]) for row in rows if not row["target"]]
                    aucs[feature] = _auc(pos, neg)
                top = sorted(aucs.items(), key=lambda kv: (-(kv[1] if kv[1] == kv[1] else -1), kv[0]))
                per_t[t] = {
                    "n_positive": sum(1 for r in rows if r["target"]),
                    "auc": aucs,
                    "top_five": [{"variable": k, "auc": v} for k, v in top[:5]],
                    "h_mean": float(np.mean([row["h"] for row in rows])),
                    "h_gt0_45_fraction": float(np.mean([row["h"] > 0.45 for row in rows])),
                    "auc_over_075": [{"variable": k, "auc": v} for k, v in top if v == v and v > 0.75],
                    "seed_targets": seed_rows,
                }
            cells_out[f"onset_{horizon}s"] = per_t
        result[str(cell)] = cells_out
    return result


def _onset_for(cell: int, seed: int) -> int:
    if cell == 2200:
        payload = json.loads(
            (
                REPO_ROOT
                / "runs"
                / "probe"
                / "sustained_wall_v1_stage1b_hold_m2200_p20_s6001_6030"
                / "seeds"
                / f"seed{seed}.json"
            ).read_text(encoding="utf-8")
        )
        return payload["arms"]["C"]["health_collapse"]["first_onset_step"]
    root = (
        REPO_ROOT / "runs" / "probe" / f"sustained_wall_v1_stage1b_band_locator_m{cell}"
    )
    if root.name.endswith("2000"):
        root = root / "s6061_6090"
    else:
        root = root / "s6091_6120"
    # glob the real seed dir name
    matches = list(root.parent.glob(f"{root.name}"))
    if not matches:
        matches = list(
            (REPO_ROOT / "runs" / "probe").glob(
                f"sustained_wall_v1_stage1b_band_locator_m{cell}_s*/seeds/seed{seed}.json"
            )
        )
        payload = json.loads(matches[0].read_text(encoding="utf-8"))
        return payload["branch"]["health_collapse"]["first_onset_step"]
    payload = json.loads((matches[0] / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8"))
    return payload["branch"]["health_collapse"]["first_onset_step"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=str(OUTPUT_DIR))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {}
    if (MINHOLD_ROOT / "seeds").exists():
        result["part_b"] = part_b()
    if (OBSMAP_ROOTS[2200] / "seeds").exists():
        result["part_c"] = part_c()
    _atomic_json(output / "prep4_summary.json", result)
    for arm, block in result.get("part_b", {}).get("arms", {}).items():
        print(arm, json.dumps(block["incidence"]), "delay med", block["delay_delta_s_per_veh"]["median"], "hold", block["hold_duration_s"]["median"])
    for cell, blocks in result.get("part_c", {}).items():
        if cell == "schema":
            continue
        for horizon, per_t in blocks.items():
            for t, block in per_t.items():
                top = ", ".join(f"{v['variable']}:{v['auc']:.3f}" for v in block["top_five"])
                print(f"{cell} {horizon} t={t}: h>0.45={block['h_gt0_45_fraction']:.2f} top: {top}")


if __name__ == "__main__":
    main()
