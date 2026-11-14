"""Stage 1b-prep-5 (amended) analysis: locator rows (6 cells), D-Arm damping,
W-D full-hold counterfactual, and the pooled observability AUC.

Reads the prep-5 artifact dirs (C at 2400/2600/2700, D at all six cells, W at
all six cells) plus the stored no-control C for the older cells, and writes
``runs/probe/sustained_wall_v1_stage1b_prep5/prep5_summary.json``.
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
from experiments.idm.run_sustained_wall_v1_stage1b_prep5 import CELLS
from experiments.io import atomic_write_json as _atomic_json

PROBE = REPO_ROOT / "runs" / "probe"
OUTPUT_DIR = PROBE / "sustained_wall_v1_stage1b_prep5"
APPROACH_EDGE = "1191554636"
MERGE_EDGE = "1487606179"


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


def _load_dirs(mode: str, vph: int) -> list[Mapping[str, Any]]:
    matches = sorted(PROBE.glob(f"sustained_wall_v1_stage1b_prep5_{mode}_m{vph}_p20_s*"))
    if not matches:
        return []
    seeds_dir = matches[0] / "seeds"
    payloads = []
    for path in sorted(seeds_dir.glob("seed*.json")):
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def _stored_locator_c(vph: int) -> Dict[int, Mapping[str, Any]]:
    if vph == 2200:
        root = PROBE / "sustained_wall_v1_stage1b_hold_m2200_p20_s6001_6030" / "seeds"
        return {
            int(seed): json.loads((root / f"seed{seed}.json").read_text(encoding="utf-8"))["arms"]["C"]
            for seed in CELLS[2200][1]
        }
    root_list = list((PROBE).glob(f"sustained_wall_v1_stage1b_band_locator_m{vph}_s*/seeds"))
    if not root_list:
        return {}
    root = root_list[0]
    return {
        int(seed): json.loads((root / f"seed{seed}.json").read_text(encoding="utf-8"))["branch"]
        for seed in CELLS[vph][1]
    }


def _locator_rows() -> Dict[str, Any]:
    cells: Dict[str, Any] = {}
    for vph in (2000, 2100, 2200, 2400, 2600, 2700):
        c_payloads = _load_dirs("C", vph) if vph >= 2400 else []
        rows = []
        if c_payloads:
            for payload in c_payloads:
                branch = payload["branch"]
                rows.append(
                    {
                        "seed": int(payload["seed"]),
                        "health_incidence": bool(branch["health_collapse"]["incidence"]),
                        "health_onset": branch["health_collapse"]["first_onset_step"],
                        "physical_incidence": bool(branch["physical_collapse"]["incidence"]),
                        "h300": float(branch["series"]["h"][300]),
                        "delay_s_per_veh": float(branch["delay"]["mean_per_vehicle_delay_s"]),
                    }
                )
                cells.setdefault(str(vph), {"location": "prep5_locator"})
        else:
            stored = _stored_locator_c(vph)
            for seed, branch in sorted(stored.items()):
                rows.append(
                    {
                        "seed": seed,
                        "health_incidence": bool(branch["health_collapse"]["incidence"]),
                        "health_onset": branch["health_collapse"]["first_onset_step"],
                        "physical_incidence": bool(branch["physical_collapse"]["incidence"]),
                        "h300": float(branch["series"]["h"][300]),
                        "delay_s_per_veh": float(branch["delay"]["mean_per_vehicle_delay_s"]),
                    }
                )
                cells.setdefault(str(vph), {"location": "stored"})
        onsets = [r["health_onset"] for r in rows if r["health_onset"] is not None]
        cells[str(vph)]["n"] = len(rows)
        cells[str(vph)]["health_incidence"] = sum(1 for r in rows if r["health_incidence"])
        cells[str(vph)]["physical_incidence"] = sum(1 for r in rows if r["physical_incidence"])
        cells[str(vph)]["onset_median"] = float(np.median(onsets)) if onsets else None
        cells[str(vph)]["h300_gt_0_45_fraction"] = float(
            np.mean([r["h300"] > 0.45 for r in rows])
        )
        cells[str(vph)]["delay_median"] = float(
            np.median([r["delay_s_per_veh"] for r in rows])
        )
        cells[str(vph)]["rows"] = rows
    return cells


def _paired_block(d_payloads: Sequence[Mapping[str, Any]], c_branches: Mapping[int, Any]) -> Dict[str, Any]:
    deltas = []
    durations = []
    incidence_c = []
    incidence_d = []
    comfort_loc: Dict[str, list[float]] = {}
    lttc_loc: Dict[str, list[float]] = {}
    for payload in d_payloads:
        seed = int(payload["seed"])
        d_branch = payload["branch"]
        c_branch = c_branches[seed]
        deltas.append(
            float(d_branch["delay"]["mean_per_vehicle_delay_s"])
            - float(c_branch["delay"]["mean_per_vehicle_delay_s"])
        )
        durations.append(
            float(d_branch["health_collapse"]["total_duration_s"])
            - float(c_branch["health_collapse"]["total_duration_s"])
        )
        incidence_c.append(bool(c_branch["health_collapse"]["incidence"]))
        incidence_d.append(bool(d_branch["health_collapse"]["incidence"]))
        loc_c = c_branch["outcomes"]["hard_brake_attribution"]["exposure_counts"]["road_id"]
        loc_d = d_branch["outcomes"]["hard_brake_attribution"]["exposure_counts"]["road_id"]
        for road in sorted(set(loc_c) | set(loc_d)):
            comfort_loc.setdefault(str(road), []).append(
                float(loc_d.get(road, 0)) - float(loc_c.get(road, 0))
            )
        lt_c = c_branch["outcomes"]["low_ttc_attribution"]["exposure_counts"]["road_id"]
        lt_d = d_branch["outcomes"]["low_ttc_attribution"]["exposure_counts"]["road_id"]
        for road in sorted(set(lt_c) | set(lt_d)):
            lttc_loc.setdefault(str(road), []).append(
                float(lt_d.get(road, 0)) - float(lt_c.get(road, 0))
            )
    return {
        "delay_delta_s_per_veh": {
            **_stats(deltas),
            "bootstrap_median_ci": _bootstrap_intervals(deltas).get("median"),
            "fraction_beneficial": float(np.mean(np.asarray(deltas) < 0.0)),
        },
        "duration_delta_s": {**_stats(durations)},
        "incidence": {"C": int(sum(incidence_c)), "D": int(sum(incidence_d))},
        "comfort_decel_cap_excess_by_location": {
            road: _stats(values) for road, values in sorted(comfort_loc.items())
        },
        "low_ttc_excess_by_location": {
            road: _stats(values) for road, values in sorted(lttc_loc.items())
        },
    }


def analyze() -> Dict[str, Any]:
    result: Dict[str, Any] = {"schema": "sustained_wall_v1_stage1b_prep5_summary/1"}
    result["locator"] = _locator_rows()
    result["damping"] = {}
    result["full_hold"] = {}
    d_payloads_by_cell: Dict[int, list[Mapping[str, Any]]] = {}
    for vph in (2000, 2100, 2200, 2400, 2600, 2700):
        d_payloads = _load_dirs("D", vph)
        c_payloads = _load_dirs("C", vph)
        if c_payloads:
            c_branches = {
                int(p["seed"]): p["branch"] for p in c_payloads
            }
        else:
            c_branches = _stored_locator_c(vph)
        d_payloads_by_cell[vph] = d_payloads
        block = _paired_block(d_payloads, c_branches)
        block["command_activity"] = {
            "mean_cav_seconds": float(np.mean([p["command_activity"]["cav_seconds"] for p in d_payloads])),
            "mean_distinct_cavs": float(np.mean([p["command_activity"]["distinct_cavs"] for p in d_payloads])),
            "mean_fraction_steps_with_command": float(np.mean([p["command_activity"]["fraction_steps_with_command"] for p in d_payloads])),
        }
        block["repeat_checks"] = [
            {"seed": int(p["seed"]), "exact": bool(p["repeat_check"]["exact_identity"])}
            for p in d_payloads
            if p.get("repeat_check")
        ]
        result["damping"][str(vph)] = block
    for vph in (2000, 2100, 2200, 2400, 2600, 2700):
        w_payloads = _load_dirs("W", vph)
        deltas = []
        durations = []
        incidence_w = []
        incidence_d = []
        comfort_loc: Dict[str, list[float]] = {}
        lttc_loc: Dict[str, list[float]] = {}
        flow_w = []
        for payload in w_payloads:
            d_branch = payload["d_branch"]
            w_branch = payload["w_branch"]
            deltas.append(
                float(w_branch["delay"]["mean_per_vehicle_delay_s"])
                - float(d_branch["delay"]["mean_per_vehicle_delay_s"])
            )
            durations.append(
                float(w_branch["health_collapse"]["total_duration_s"])
                - float(d_branch["health_collapse"]["total_duration_s"])
            )
            incidence_d.append(bool(d_branch["health_collapse"]["incidence"]))
            incidence_w.append(bool(w_branch["health_collapse"]["incidence"]))
            loc_d = d_branch["outcomes"]["hard_brake_attribution"]["exposure_counts"]["road_id"]
            loc_w = w_branch["outcomes"]["hard_brake_attribution"]["exposure_counts"]["road_id"]
            for road in sorted(set(loc_d) | set(loc_w)):
                comfort_loc.setdefault(str(road), []).append(
                    float(loc_w.get(road, 0)) - float(loc_d.get(road, 0))
                )
            lt_d = d_branch["outcomes"]["low_ttc_attribution"]["exposure_counts"]["road_id"]
            lt_w = w_branch["outcomes"]["low_ttc_attribution"]["exposure_counts"]["road_id"]
            for road in sorted(set(lt_d) | set(lt_w)):
                lttc_loc.setdefault(str(road), []).append(
                    float(lt_w.get(road, 0)) - float(lt_d.get(road, 0))
                )
            flow_w.append(_flow(w_branch))
        block: Dict[str, Any] = {
            "delay_delta_s_per_veh": {
                **_stats(deltas),
                "bootstrap_median_ci": _bootstrap_intervals(deltas).get("median"),
                "fraction_beneficial": float(np.mean(np.asarray(deltas) < 0.0)),
            },
            "duration_delta_s": {**_stats(durations)},
            "incidence": {"D": int(sum(incidence_d)), "W": int(sum(incidence_w))},
            "comfort_decel_cap_excess_by_location": {
                road: _stats(values) for road, values in sorted(comfort_loc.items())
            },
            "low_ttc_excess_by_location": {
                road: _stats(values) for road, values in sorted(lttc_loc.items())
            },
            "merge_flow_median": float(np.median(flow_w)),
        }
        result["full_hold"][str(vph)] = block
    # Collapse premium in D per cell (delay of collapsing vs not, D branch).
    premium = {}
    for vph, payloads in d_payloads_by_cell.items():
        collapsing = [p for p in payloads if p["branch"]["health_collapse"]["incidence"]]
        not_collapsing = [p for p in payloads if not p["branch"]["health_collapse"]["incidence"]]
        if collapsing and not_collapsing:
            premium[str(vph)] = {
                "collapsing_delay_mean": float(np.mean([p["branch"]["delay"]["mean_per_vehicle_delay_s"] for p in collapsing])),
                "not_collapsing_delay_mean": float(np.mean([p["branch"]["delay"]["mean_per_vehicle_delay_s"] for p in not_collapsing])),
                "premium": float(np.mean([p["branch"]["delay"]["mean_per_vehicle_delay_s"] for p in collapsing]))
                - float(np.mean([p["branch"]["delay"]["mean_per_vehicle_delay_s"] for p in not_collapsing])),
            }
    result["damping_premium"] = premium
    # C2 pooled observability at t = 250.
    result["c2_observability"] = _c2(d_payloads_by_cell, w_payloads_by_cell := {
        vph: _load_dirs("W", vph) for vph in (2000, 2100, 2200, 2400, 2600, 2700)
    })
    return result


def _flow(branch: Mapping[str, Any]) -> float:
    steps = [int(s) for s in branch["series"]["step"] if 300 <= int(s) <= 900]
    if not steps:
        return float("nan")
    return float(series_in_window(branch["series"], "discharge_count", 300, 900) / len(steps) * 3600.0)


def _c2(
    d_payloads: Mapping[int, Sequence[Mapping[str, Any]]],
    w_payloads: Mapping[int, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    rows = []
    for vph in (2000, 2100, 2200, 2400, 2600, 2700):
        w_seeds = {int(p["seed"]): p for p in w_payloads.get(vph, [])}
        for p in d_payloads.get(vph, []):
            seed = int(p["seed"])
            if seed not in w_seeds:
                continue
            delta = (
                float(w_seeds[seed]["w_branch"]["delay"]["mean_per_vehicle_delay_s"])
                - float(w_seeds[seed]["d_branch"]["delay"]["mean_per_vehicle_delay_s"])
            )
            features = p["step_features"].get("250")
            if features is None:
                continue
            flat = _flatten(features)
            flat["h"] = float(p["h_series"][250])
            rows.append({"seed": seed, "vph": vph, "target": bool(delta < 0.0), **flat})
    variables = [k for k in rows[0] if k not in ("seed", "vph", "target")]
    aucs = {}
    for variable in variables:
        pos = [float(r[variable]) for r in rows if r["target"]]
        neg = [float(r[variable]) for r in rows if not r["target"]]
        aucs[variable] = _auc(pos, neg)
    top = sorted(aucs.items(), key=lambda kv: (-(kv[1] if kv[1] == kv[1] else -1), kv[0]))
    return {
        "n_pooled": len(rows),
        "n_positive": sum(1 for r in rows if r["target"]),
        "auc": aucs,
        "top_ten": [{"variable": k, "auc": v} for k, v in top[:10]],
    }


def _flatten(features: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    regions = features.get("regions", {})
    for region in ("upstream", "approach", "near_merge", "merge_edge", "post_merge", "ramp"):
        stats = regions.get(region, {})
        out[f"{region}_count"] = float(stats.get("count", 0.0))
        out[f"{region}_mean_speed"] = float(stats.get("mean_speed", 0.0))
        out[f"{region}_speed_std"] = float(stats.get("speed_std", 0.0))
        out[f"{region}_harmonic_speed"] = float(stats.get("harmonic_speed", 0.0))
    out["mainline_lane_imbalance"] = float(features.get("mainline_lane_imbalance", 0.0))
    out["ramp_queue_length"] = float(features.get("ramp_queue_length", 0.0))
    out["cav_count_band"] = float(features.get("cav_count_band", 0.0))
    out["cav_count_upstream"] = float(features.get("cav_count_upstream", 0.0))
    return out


def _auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    combined = list(pos) + list(neg)
    ranks = rankdata(combined)
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=str(OUTPUT_DIR))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = analyze()
    _atomic_json(output / "prep5_summary.json", result)
    print("locator:", {k: {kk: v[kk] for kk in ("health_incidence", "onset_median", "delay_median")} for k, v in result["locator"].items()})
    print("damping:", {k: {kk: round(v[kk], 3) if isinstance(v[kk], float) else v[kk] for kk in ("incidence", "delay_delta_s_per_veh")} for k, v in result["damping"].items()})
    print("full_hold:", {k: {kk: round(v[kk], 3) if isinstance(v[kk], float) else v[kk] for kk in ("incidence", "delay_delta_s_per_veh")} for k, v in result["full_hold"].items()})
    print("c2 top:", [ (t["variable"], round(t["auc"], 3)) for t in result["c2_observability"]["top_ten"] ])


if __name__ == "__main__":
    main()
