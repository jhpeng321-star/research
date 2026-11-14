"""Stage 1a paired-effect analysis (no pass/fail gate: report intervals).

Consumes the per-seed JSON written by ``run_sustained_wall_v1_stage1a.py`` and
writes the pilot ``summary.json`` with: landmark statistics, collapse
incidence under both labels with seed-level bootstrap CIs of the paired
reduction, delay effects with intervals, the perturbation-noise floor
comparison for hard braking / low-TTC / delay / discharge (network-wide and
in the [t_d, t_release+60] window), the realized dose, and the per-seed table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from experiments import REPO_ROOT
from experiments.idm import sustained_wall_v1_contract as C
from experiments.io import atomic_write_json as _atomic_json

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260902


# -- statistics helpers ----------------------------------------------------
def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray([float(value) for value in values], dtype=float)
    if array.size == 0:
        return {"p10": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "p10": float(np.percentile(array, 10)),
        "median": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def _iqr(values: Sequence[float]) -> float:
    array = np.asarray([float(value) for value in values], dtype=float)
    if array.size == 0:
        return float("nan")
    q75, q25 = np.percentile(array, 75), np.percentile(array, 25)
    return float(q75 - q25)


def _bootstrap_intervals(
    values: Sequence[float], *, resamples: int = BOOTSTRAP_RESAMPLES
) -> Dict[str, Dict[str, float]]:
    """Percentile CIs of the bootstrap mean and median of ``values``."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    array = np.asarray([float(value) for value in values], dtype=float)
    if array.size == 0:
        return {}
    means: list[float] = []
    medians: list[float] = []
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    for row in indices:
        sample = array[row]
        means.append(float(sample.mean()))
        medians.append(float(np.median(sample)))
    return {
        "mean": {"ci90": _percentile_pair(means, 5, 95), "ci95": _percentile_pair(means, 2.5, 97.5)},
        "median": {
            "ci90": _percentile_pair(medians, 5, 95),
            "ci95": _percentile_pair(medians, 2.5, 97.5),
        },
    }


def _percentile_pair(values: Sequence[float], low: float, high: float) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "lo": float(np.percentile(array, low)),
        "hi": float(np.percentile(array, high)),
    }


def _paired_delta(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> float:
    return float(left[key]) - float(right[key])


def _series_slice(series: Mapping[str, Any], step_from: int, step_to: int) -> list[int]:
    """Index positions of steps in [step_from, step_to] inclusive."""
    positions = []
    for index, step in enumerate(series["step"]):
        if step_from <= int(step) <= step_to:
            positions.append(index)
    return positions


def _sum_metric(series: Mapping[str, Any], key: str, positions: Optional[Sequence[int]] = None) -> float:
    values = series.get(key)
    if values is None:
        return 0.0
    if positions is None:
        positions = range(len(values))
    total = 0.0
    for position in positions:
        if position >= len(values):
            continue
        item = values[position]
        if item is not None:
            total += float(item)
    return total


def _collapse_incidence(branch: Mapping[str, Any]) -> Dict[str, bool]:
    return {
        "health": bool(branch["health_collapse"]["incidence"]),
        "physical": bool(branch["physical_collapse"]["incidence"]),
    }


# -- per-seed rows ---------------------------------------------------------
def _per_seed_rows(seed_payloads: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows = []
    for payload in seed_payloads:
        seed = int(payload["seed"])
        arms = payload["arms"]
        landmark = payload["decision_landmark"]
        t_d = landmark["step"]
        row: Dict[str, Any] = {
            "seed": seed,
            "t_d_step": t_d,
            "landmark": bool(landmark["exists"]),
        }
        if t_d is None:
            row.update(
                {
                    "C_health_collapse": bool(arms["C"]["health_collapse"]["incidence"]),
                    "C_physical_collapse": bool(arms["C"]["physical_collapse"]["incidence"]),
                }
            )
        else:
            w = arms["W"]
            p = arms["P"] if "P" in arms else None
            c = arms["C"]
            row.update(
                {
                    "C_health_collapse": bool(c["health_collapse"]["incidence"]),
                    "W_health_collapse": bool(w["health_collapse"]["incidence"]),
                    "P_health_collapse": bool(p["health_collapse"]["incidence"]),
                    "C_physical_collapse": bool(c["physical_collapse"]["incidence"]),
                    "W_physical_collapse": bool(w["physical_collapse"]["incidence"]),
                    "P_physical_collapse": bool(p["physical_collapse"]["incidence"]),
                    "delay_delta_W_s_per_veh": (
                        float(w["delay"]["mean_per_vehicle_delay_s"])
                        - float(c["delay"]["mean_per_vehicle_delay_s"])
                    ),
                    "delay_delta_P_s_per_veh": (
                        float(p["delay"]["mean_per_vehicle_delay_s"])
                        - float(c["delay"]["mean_per_vehicle_delay_s"])
                    ),
                    "incremental_delay_delta_W_s": (
                        float(w["delay"]["incremental_delay_s"])
                        - float(c["delay"]["incremental_delay_s"])
                    ),
                    "hard_brake_delta_W_steps": (
                        int(w["outcomes"]["hard_brake_vehicle_steps"])
                        - int(c["outcomes"]["hard_brake_vehicle_steps"])
                    ),
                    "hard_brake_delta_P_steps": (
                        int(p["outcomes"]["hard_brake_vehicle_steps"])
                        - int(c["outcomes"]["hard_brake_vehicle_steps"])
                    ),
                    "low_ttc_delta_W_steps": (
                        int(w["outcomes"]["low_ttc_exposure_steps"])
                        - int(c["outcomes"]["low_ttc_exposure_steps"])
                    ),
                    "wall_realized_s": float((payload.get("wall_dose") or {}).get(
                        "realized_wall_seconds", float("nan")
                    )),
                    "release_reason": (payload.get("wall_dose") or {}).get("release_reason"),
                }
            )
        rows.append(row)
    return rows


# -- main analysis ---------------------------------------------------------
def analyze(output: Path) -> Dict[str, Any]:
    seeds_path = output / "seeds"
    seed_payloads: list[Mapping[str, Any]] = []
    missing: list[str] = []
    for seed in C.SEEDS:
        path = seeds_path / f"seed{seed}.json"
        if path.exists():
            seed_payloads.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(str(path))
    if not seed_payloads:
        raise ValueError(f"no per-seed artifacts found under {seeds_path}")

    landmark_rows = [p for p in seed_payloads if p["decision_landmark"]["exists"]]
    t_d_values = [int(p["decision_landmark"]["step"]) for p in landmark_rows]
    no_landmark = [int(p["seed"]) for p in seed_payloads if not p["decision_landmark"]["exists"]]

    summary: Dict[str, Any] = {
        "schema": "sustained_wall_v1_stage1a_summary/1",
        "seeds_requested": list(C.SEEDS),
        "seeds_present": [int(p["seed"]) for p in seed_payloads],
        "missing": missing,
        "landmark": {
            "seed_count_with_landmark": len(landmark_rows),
            "seed_count_no_landmark": len(no_landmark),
            "no_landmark_seeds": no_landmark,
            "t_d_values": sorted(t_d_values),
            "t_d_statistics": _quantiles(t_d_values) if t_d_values else {},
        },
    }

    collapse_summary: Dict[str, Any] = {}
    for label in ("health", "physical"):
        incidences: Dict[str, Dict[int, bool]] = {}
        for p in landmark_rows:
            incidences.setdefault("C", {})[int(p["seed"])] = bool(
                p["arms"]["C"][f"{label}_collapse"]["incidence"]
            )
            incidences.setdefault("W", {})[int(p["seed"])] = bool(
                p["arms"]["W"][f"{label}_collapse"]["incidence"]
            )
            if "P" in p["arms"]:
                incidences.setdefault("P", {})[int(p["seed"])] = bool(
                    p["arms"]["P"][f"{label}_collapse"]["incidence"]
                )
        all_incidence_c = {
            int(p["seed"]): bool(p["arms"]["C"][f"{label}_collapse"]["incidence"])
            for p in seed_payloads
        }
        label_summary: Dict[str, Any] = {
            "incidence_counts": {
                arm: {
                    "n": len(values),
                    "collapse_incidences": sum(bool(v) for v in values.values()),
                }
                for arm, values in incidences.items()
            },
            "incidence_counts_arm_C_all_seeds": {
                "n": len(all_incidence_c),
                "collapse_incidences": sum(bool(v) for v in all_incidence_c.values()),
            },
        }
        for arm in ("W", "P"):
            if arm not in incidences:
                continue
            seeds = sorted(incidences["C"])
            reductions = [
                float(incidences["C"][s]) - float(incidences[arm][s]) for s in seeds
            ]
            label_summary[f"paired_{arm}_minus_C"] = {
                "mean_reduction_incidence": float(np.mean(reductions)),
                "bootstrap": _bootstrap_intervals(reductions),
            }
        for arm in ("W", "P"):
            if arm not in incidences:
                continue
            seeds = sorted(incidences["C"])
            total_deltas = [
                float(p["arms"][arm][f"{label}_collapse"]["total_duration_s"])
                - float(p["arms"]["C"][f"{label}_collapse"]["total_duration_s"])
                for p in landmark_rows
                if int(p["seed"]) in seeds
            ]
            longest_deltas = [
                float(p["arms"][arm][f"{label}_collapse"]["longest_run_s"])
                - float(p["arms"]["C"][f"{label}_collapse"]["longest_run_s"])
                for p in landmark_rows
                if int(p["seed"]) in seeds
            ]
            label_summary[f"{arm}_vs_C_duration"] = {
                "total_duration_delta_s": {
                    "median": float(np.median(total_deltas)),
                    "iqr": float(np.percentile(total_deltas, 75) - np.percentile(total_deltas, 25)),
                    "p10": float(np.percentile(total_deltas, 10)),
                    "p90": float(np.percentile(total_deltas, 90)),
                    "bootstrap_median_ci": _bootstrap_intervals(total_deltas).get("median"),
                },
                "longest_run_delta_s": {
                    "median": float(np.median(longest_deltas)),
                    "iqr": float(
                        np.percentile(longest_deltas, 75) - np.percentile(longest_deltas, 25)
                    ),
                    "p10": float(np.percentile(longest_deltas, 10)),
                    "p90": float(np.percentile(longest_deltas, 90)),
                    "bootstrap_median_ci": _bootstrap_intervals(longest_deltas).get("median"),
                },
            }
        collapse_summary[label] = label_summary
    summary["collapse"] = collapse_summary

    delay_deltas: Dict[str, list[float]] = {"W": [], "P": []}
    for p in landmark_rows:
        c = p["arms"]["C"]
        for arm in ("W", "P"):
            delay_deltas[arm].append(
                float(p["arms"][arm]["delay"]["mean_per_vehicle_delay_s"])
                - float(c["delay"]["mean_per_vehicle_delay_s"])
            )
    delay_summary: Dict[str, Any] = {}
    for arm in ("W", "P"):
        values = delay_deltas[arm]
        array = np.asarray(values, dtype=float)
        delay_summary[arm] = {
            "delta_s_per_veh": {
                "median": float(np.median(array)),
                "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
                "p10": float(np.percentile(array, 10)),
                "p90": float(np.percentile(array, 90)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "bootstrap_median_ci": _bootstrap_intervals(values).get("median"),
            },
            "fraction_seeds_beneficial": float(np.mean(array < 0.0)),
        }
    summary["delay"] = delay_summary

    # -- noise floor ---------------------------------------------------------
    def _noise_arm_statistics(seeds: Sequence[Mapping[str, Any]], arm: str) -> Dict[str, Any]:
        metric_keys = {
            "hard_brake": "hard_brake_count",
            "low_ttc": "low_ttc_step_flag",
            "delay": "incremental_delay_s",
            "discharge": "discharge_count",
        }
        p_deltas_all: Dict[str, list[float]] = {key: [] for key in metric_keys}
        w_deltas_all: Dict[str, list[float]] = {key: [] for key in metric_keys}
        p_deltas_window: Dict[str, list[float]] = {key: [] for key in metric_keys}
        w_deltas_window: Dict[str, list[float]] = {key: [] for key in metric_keys}
        for p in seeds:
            if arm not in p["arms"]:
                continue
            c = p["arms"]["C"]
            w = p["arms"]["W"]
            pf = p["arms"][arm]
            w_series = w["series"]
            c_series = c["series"]
            p_series = pf["series"]
            t_d_step = int(p["decision_landmark"]["step"])
            release_step = int((p.get("wall_dose") or {}).get("release_step", t_d_step) or t_d_step)
            window_end = int(release_step + C.RELEASE_TAIL_WINDOW_S)
            c_pos_window = _series_slice(w_series, t_d_step, window_end)
            w_pos_window = _series_slice(w_series, t_d_step, window_end)
            p_pos_window = _series_slice(p_series, t_d_step, window_end)

            for metric, key in metric_keys.items():
                p_deltas_all[metric].append(
                    _sum_metric(p_series, key) - _sum_metric(c_series, key)
                )
                w_deltas_all[metric].append(
                    _sum_metric(w_series, key) - _sum_metric(c_series, key)
                )
                p_deltas_window[metric].append(
                    _sum_metric(p_series, key, p_pos_window)
                    - _sum_metric(c_series, key, c_pos_window)
                )
                w_deltas_window[metric].append(
                    _sum_metric(w_series, key, w_pos_window)
                    - _sum_metric(c_series, key, c_pos_window)
                )

        def _bounds_and_fraction(
            p_values: Sequence[float], w_values: Sequence[float]
        ) -> Dict[str, Any]:
            pv = np.asarray(p_values, dtype=float)
            wv = np.asarray(w_values, dtype=float)
            if pv.size == 0:
                return {"delta_p_percentiles": None, "fraction_delta_w_outside_p5_p95": None}
            p5, p50, p95 = np.percentile(pv, 5), np.percentile(pv, 50), np.percentile(pv, 95)
            outside = float(np.mean((wv < p5) | (wv > p95))) if wv.size else float("nan")
            return {
                "delta_p_percentiles": {
                    "p5": float(p5),
                    "p50": float(p50),
                    "p95": float(p95),
                },
                "fraction_delta_w_outside_p5_p95": outside,
            }

        result_all = {
            metric: _bounds_and_fraction(p_deltas_all[metric], w_deltas_all[metric])
            for metric in metric_keys
        }
        result_window = {
            metric: _bounds_and_fraction(p_deltas_window[metric], w_deltas_window[metric])
            for metric in metric_keys
        }
        return {"network_wide": result_all, "window": result_window}

    summary["noise_floor"] = _noise_arm_statistics(landmark_rows, "P")

    # -- dose ----------------------------------------------------------------
    dose_rows = [(p, p.get("wall_dose")) for p in landmark_rows if p.get("wall_dose")]
    durations = [float(dose["realized_wall_seconds"]) for _p, dose in dose_rows]
    release_reasons: Dict[str, int] = {}
    coverage_means: list[float] = []
    member_seconds: list[float] = []
    per_step_coverage: list[float] = []
    max_members: list[int] = []
    for p, dose in dose_rows:
        if dose.get("release_reason"):
            release_reasons[dose["release_reason"]] = release_reasons.get(
                dose["release_reason"], 0
            ) + 1
        coverage_means.append(float(dose["coverage_statistics"]["mean"]))
        member_seconds.append(float(dose["member_seconds"]))
        w_series = p["arms"]["W"]["series"]
        t_d_step = int(p["decision_landmark"]["step"])
        release_step = int(dose.get("release_step") or t_d_step)
        if "lane_coverage_fraction" in w_series:
            step_list = [int(value) for value in w_series["step"]]
            engaged = [
                index
                for index, step in enumerate(step_list)
                if t_d_step < step < release_step
            ]
            per_step_coverage.extend(
                float(w_series["lane_coverage_fraction"][index]) for index in engaged
            )
            engaged_members = [int(w_series["member_count"][index]) for index in engaged]
            max_members.append(max(engaged_members) if engaged_members else 0)
    summary["dose"] = {
        "seeds_with_wall": len(dose_rows),
        "realized_wall_duration_s": _quantiles(durations) if durations else {},
        "release_reason_counts": release_reasons,
        "per_lane_coverage_fraction": {
            "per_seed_mean": (
                {
                    "min": float(np.min(coverage_means)),
                    "median": float(np.median(coverage_means)),
                    "max": float(np.max(coverage_means)),
                }
                if coverage_means
                else {}
            ),
            "per_step_engaged_distribution": (
                _quantiles(per_step_coverage) if per_step_coverage else {}
            ),
        },
        "member_seconds": _quantiles(member_seconds) if member_seconds else {},
        "max_member_count_per_seed": _quantiles(max_members) if max_members else {},
    }

    summary["per_seed_table"] = _per_seed_rows(seed_payloads)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=str(
            REPO_ROOT
            / "runs"
            / "probe"
            / "sustained_wall_v1_stage1a_m2200_p20_s6001_6030"
        ),
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    summary = analyze(output)
    _atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
