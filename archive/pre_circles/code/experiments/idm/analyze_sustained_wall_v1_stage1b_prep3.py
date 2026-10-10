"""Stage 1b-prep-3 analysis: HRULE per-cell results, SE pilot, rule algebra.

Reads the HRULE (2200/2100/2000) and SE (2100) artifacts plus the Stage 1a
pulse and 1b-prep WHOLD_soft references and produces
``runs/probe/sustained_wall_v1_stage1b_prep3/prep3_summary.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from experiments import REPO_ROOT
from experiments.idm.analyze_sustained_wall_v1_stage1a import _bootstrap_intervals
from experiments.idm.stage1b_common import series_in_window
from experiments.io import atomic_write_json as _atomic_json

SEEDS_2200 = tuple(range(6001, 6031))
SEEDS_2100 = tuple(range(6091, 6121))
SEEDS_2000 = tuple(range(6061, 6091))
HRULE_ROOTS = {
    2200: REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep3_hrule_m2200_p20_s6001_6030",
    2100: REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep3_hrule_m2100_p20_s6091_6120",
    2000: REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep3_hrule_m2000_p20_s6061_6090",
}
SE_ROOT = REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep3_se_m2100_p20_s6091_6120"
STAGE1A_ROOT = REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1a_m2200_p20_s6001_6030"
HOLD_ROOT = REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_hold_m2200_p20_s6001_6030"
OUTPUT_DIR = REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep3"
APPROACH_EDGE = "1191554636"


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


def _bootstrap_slope(x: Sequence[float], y: Sequence[float], resamples: int = 2000) -> Dict[str, Any]:
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
    return {
        "ci90": {"lo": float(np.percentile(slopes, 5)), "hi": float(np.percentile(slopes, 95))},
        "ci95": {"lo": float(np.percentile(slopes, 2.5)), "hi": float(np.percentile(slopes, 97.5))},
    }


def _merge_flow(branch: Mapping[str, Any], step_from: int, step_to: int) -> float:
    steps = [int(s) for s in branch["series"]["step"] if step_from <= int(s) <= step_to]
    if not steps:
        return float("nan")
    return float(series_in_window(branch["series"], "discharge_count", step_from, step_to) / len(steps) * 3600.0)


def _load(root: Path, seed: int) -> Mapping[str, Any]:
    return json.loads((root / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8"))


def hrule_cell(mainline_vph: int, seeds: Sequence[int]) -> Dict[str, Any]:
    payloads = [_load(HRULE_ROOTS[mainline_vph], seed) for seed in seeds]
    deltas = []
    durations = []
    longest = []
    wall_on = []
    engagements = []
    first_engage = []
    before_onset = []
    after_onset = []
    delay_beneficial = []
    low_ttc_loc: Dict[str, list[float]] = {}
    comfort_loc: Dict[str, list[float]] = {}
    flow_c = []
    flow_a = []
    incidence_c = []
    incidence_a = []
    collision_a = []
    hrule_payloads = []
    for payload in payloads:
        c = payload["arms"]["C"]
        a = payload["arms"]["HRULE"]
        tele = payload["wall_telemetry"]["HRULE"]
        deltas.append(
            float(a["delay"]["mean_per_vehicle_delay_s"]) - float(c["delay"]["mean_per_vehicle_delay_s"])
        )
        durations.append(
            float(a["health_collapse"]["total_duration_s"])
            - float(c["health_collapse"]["total_duration_s"])
        )
        longest.append(
            float(a["health_collapse"]["longest_run_s"])
            - float(c["health_collapse"]["longest_run_s"])
        )
        wall_on.append(float(tele["wall_on_seconds"]))
        engagements.append(float(tele["engagements"]))
        first_engage.append(float(tele["first_engage_step"]) if tele["first_engage_step"] is not None else float("nan"))
        onset = c["health_collapse"]["first_onset_step"]
        if onset is not None:
            before = sum(1 for s in tele["wall_on_steps"] if s < onset)
            after = len(tele["wall_on_steps"]) - before
            before_onset.append(float(before))
            after_onset.append(float(after))
        delay_beneficial.append(deltas[-1] < 0.0)
        loc_c = c["outcomes"]["hard_brake_attribution"]["exposure_counts"]["road_id"]
        loc_a = a["outcomes"]["hard_brake_attribution"]["exposure_counts"]["road_id"]
        for road in sorted(set(loc_c) | set(loc_a)):
            comfort_loc.setdefault(str(road), []).append(
                float(loc_a.get(road, 0)) - float(loc_c.get(road, 0))
            )
        lt_c = c["outcomes"]["low_ttc_attribution"]["exposure_counts"]["road_id"]
        lt_a = a["outcomes"]["low_ttc_attribution"]["exposure_counts"]["road_id"]
        for road in sorted(set(lt_c) | set(lt_a)):
            low_ttc_loc.setdefault(str(road), []).append(
                float(lt_a.get(road, 0)) - float(lt_c.get(road, 0))
            )
        flow_c.append(_merge_flow(c, 300, 900))
        flow_a.append(_merge_flow(a, 300, 900))
        incidence_c.append(bool(c["health_collapse"]["incidence"]))
        incidence_a.append(bool(a["health_collapse"]["incidence"]))
        collision_a.append(a["events"])
        hrule_payloads.append(payload)

    reductions = [
        float(incidence_c[i]) - float(incidence_a[i]) for i in range(len(incidence_c))
    ]
    return {
        "cell_mainline_vph": mainline_vph,
        "n": len(payloads),
        "incidence": {
            "C": int(sum(incidence_c)),
            "HRULE": int(sum(incidence_a)),
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
        "delay_delta_s_per_veh": {
            **_stats(deltas),
            "bootstrap_median_ci": _bootstrap_intervals(deltas).get("median"),
            "fraction_beneficial": float(np.mean(delay_beneficial)),
        },
        "total_duration_delta_s": {**_stats(durations)},
        "longest_run_delta_s": {**_stats(longest)},
        "wall_on_time_s": {**_stats(wall_on)},
        "engagements_per_episode": {**_stats(engagements)},
        "first_engage_step": dict(_stats([v for v in first_engage if not np.isnan(v)])),
        "wall_on_split_around_C_onset_s": {
            "before_onset": {**_stats(before_onset)},
            "after_onset": {**_stats(after_onset)},
            "n_collapsing_with_onset": len(before_onset),
        },
        "low_ttc_excess_by_location": {
            road: _stats(values) for road, values in sorted(low_ttc_loc.items())
        },
        "comfort_cost_decel_cap_excess_by_location": {
            road: _stats(values) for road, values in sorted(comfort_loc.items())
        },
        "collisions_teleports_HRULE": {
            "collision_steps": sum(x["collision_vehicle_steps"] for x in collision_a),
            "teleport_steps": sum(x["teleport_vehicle_steps"] for x in collision_a),
        },
        "merge_edge_flow_300_900_veh_h": {
            "C_median": float(np.median(flow_c)),
            "HRULE_median": float(np.median(flow_a)),
            "median_delta": float(np.median(np.asarray(flow_a) - np.asarray(flow_c))),
        },
        "per_seed": hrule_payloads,
    }


def se_block(root: Path, seeds: Sequence[int], decision_time: int) -> Dict[str, Any]:
    arm = f"SE{decision_time}"
    payloads = [_load(root, seed) for seed in seeds]
    deltas = []
    hold = []
    reasons = []
    incidence_c = []
    incidence_a = []
    lead_prevented = []
    lead_co = []
    lead_induced = []
    for payload in payloads:
        c = payload["arms"]["C"]
        a = payload["arms"][arm]
        tele = payload["wall_telemetry"][arm]
        deltas.append(
            float(a["delay"]["mean_per_vehicle_delay_s"]) - float(c["delay"]["mean_per_vehicle_delay_s"])
        )
        hold.append(float(tele["wall_on_seconds"]))
        # Release reason is fully determined by the realized hold against the
        # demand-end maximum (750 s for t_d=150 over 151..900; 650 s for
        # t_d=250 over 251..900): shorter => health release.
        max_hold = float(900 - decision_time)
        reasons.append(
            "health_release" if hold[-1] < max_hold else "demand_end"
        )
        inc_c = bool(c["health_collapse"]["incidence"])
        inc_a = bool(a["health_collapse"]["incidence"])
        incidence_c.append(inc_c)
        incidence_a.append(inc_a)
        onset = c["health_collapse"]["first_onset_step"]
        lead = None if onset is None else float(onset - decision_time)
        if inc_c and not inc_a:
            lead_prevented.append(lead)
        elif inc_c and inc_a:
            lead_co.append(lead)
        elif not inc_c and inc_a:
            lead_induced.append(lead)
    reductions = [
        float(incidence_c[i]) - float(incidence_a[i]) for i in range(len(incidence_c))
    ]
    return {
        "decision_time_s": decision_time,
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
        "delay_delta_s_per_veh": {
            **_stats(deltas),
            "bootstrap_median_ci": _bootstrap_intervals(deltas).get("median"),
            "fraction_beneficial": float(np.mean(np.asarray(deltas) < 0.0)),
        },
        "hold_duration_s": {**_stats(hold)},
        "release_reason_counts": dict(sorted(_count(reasons).items())),
        "lead_time_s": {
            "prevented": _stats(lead_prevented) if lead_prevented else {},
            "co_collapsing": _stats(lead_co) if lead_co else {},
            "induced": _stats(lead_induced) if lead_induced else {},
        },
    }


def _count(values: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def part_c() -> Dict[str, Any]:
    slopes = {}
    for vph, seeds in ((2100, SEEDS_2100), (2000, SEEDS_2000), (2200, SEEDS_2200)):
        payloads = [_load(HRULE_ROOTS[vph], seed) for seed in seeds]
        wall = [float(p["wall_telemetry"]["HRULE"]["wall_on_seconds"]) for p in payloads]
        delta_delay = [
            float(p["arms"]["HRULE"]["delay"]["mean_per_vehicle_delay_s"])
            - float(p["arms"]["C"]["delay"]["mean_per_vehicle_delay_s"])
            for p in payloads
        ]
        slope = _ols_slope(wall, delta_delay)
        slopes[vph] = {
            "wall_on_vs_delta_delay_per_100s": slope * 100.0,
            "bootstrap_per_100s": {
                key: {b: float(v * 100.0) for b, v in bounds.items()}
                for key, bounds in _bootstrap_slope(wall, delta_delay).items()
            },
            "n": len(payloads),
        }
    hmap: Dict[str, Dict[str, Any]] = {}
    for vph, seeds in ((2200, SEEDS_2200), (2100, SEEDS_2100), (2000, SEEDS_2000)):
        rows = {}
        fractions = {}
        for t in (100, 150, 200, 250, 300):
            values = []
            for seed in seeds:
                payload = _load(HRULE_ROOTS[vph], seed)
                h = float(payload["arms"]["C"]["series"]["h"][t])
                values.append(h)
                rows.setdefault(seed, {})[f"h{t}"] = round(h, 4)
            fractions[f"h{t}_gt_0_45"] = float(np.mean(np.asarray(values) > 0.45))
        hmap[vph] = {"fractions": fractions, "per_seed": {str(k): v for k, v in sorted(rows.items())}}
    return {"c1_slopes": slopes, "c2_health_signal_map": hmap,
            "reference_engagement_cost_s_per_veh_per_100s": 0.5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=str(OUTPUT_DIR))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    result: Dict[str, Any] = {"schema": "sustained_wall_v1_stage1b_prep3_summary/1"}
    result["hrule"] = {}
    for vph, seeds in ((2200, SEEDS_2200), (2100, SEEDS_2100), (2000, SEEDS_2000)):
        if (HRULE_ROOTS[vph] / "seeds").exists():
            result["hrule"][str(vph)] = hrule_cell(vph, seeds)
    if (SE_ROOT / "seeds").exists():
        result["se"] = {
            "SE150": se_block(SE_ROOT, SEEDS_2100, 150),
            "SE250": se_block(SE_ROOT, SEEDS_2100, 250),
        }
    result["part_c"] = part_c()
    result["one_axis_2200"] = _one_axis_2200(result)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "prep3_summary.json", result)
    print(json.dumps(
        {
            "hrule": {
                k: {
                    "incidence": v["incidence"],
                    "delay": v["delay_delta_s_per_veh"]["median"],
                    "wall_on": v["wall_on_time_s"]["median"],
                    "engagements": v["engagements_per_episode"]["median"],
                }
                for k, v in result["hrule"].items()
            },
            "se": {
                k: {"incidence": v["incidence"], "delay": v["delay_delta_s_per_veh"]["median"],
                    "hold": v["hold_duration_s"]["median"]}
                for k, v in result.get("se", {}).items()
            },
        },
        indent=1,
    ))


def _one_axis_2200(result: Dict[str, Any]) -> Dict[str, Any]:
    stage1a = json.loads((STAGE1A_ROOT / "summary.json").read_text(encoding="utf-8"))
    hold_summary_path = REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep" / "hold_summary.json"
    hold_summary = (
        json.loads(hold_summary_path.read_text(encoding="utf-8"))
        if hold_summary_path.exists()
        else None
    )
    h = result["hrule"]["2200"]
    delay_deltas = h["delay_delta_s_per_veh"]["median"]
    incidence_never = stage1a["collapse"]["health"]["incidence_counts"]["C"]["collapse_incidences"]
    return {
        "never_C": {
            "n": 30,
            "health_collapse_incidence": incidence_never,
            "delay_median_s_per_veh": None,
            "wall_on_s": 0.0,
        },
        "pulse_180s": {
            "health_collapse_incidence": stage1a["collapse"]["health"]["incidence_counts"]["W"]["collapse_incidences"],
            "delay_delta_median_s_per_veh": stage1a["delay"]["W"]["delta_s_per_veh"]["median"],
            "wall_on_s": 180.0,
        },
        "whold_soft": {
            "health_collapse_incidence": (
                hold_summary["arms"]["WHOLD_SOFT"]["incidence"]["WHOLD_SOFT"] if hold_summary else None
            ),
            "delay_delta_median_s_per_veh": (
                hold_summary["arms"]["WHOLD_SOFT"]["delay_delta_s_per_veh"]["median"]
                if hold_summary
                else None
            ),
            "wall_on_s": 700.0,
        },
        "hrule": {
            "health_collapse_incidence": h["incidence"]["HRULE"],
            "delay_delta_median_s_per_veh": delay_deltas,
            "wall_on_s": h["wall_on_time_s"]["median"],
        },
    }


if __name__ == "__main__":
    main()
