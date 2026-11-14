"""Stage 1b-prep Part C analysis: WHOLD_soft / WHOLD_strong vs Arm C.

Reads the hold-arms seed directory and the Stage 1a summary (for the 180 s
pulse comparison on one axis) and reports per arm, paired n = 30: health
incidence with paired-reduction bootstrap CIs, duration effects, onset delay
among co-collapsing seeds, delay cost, hard-brake / low-TTC excess with the
same phase x location split as Part A, and dose / coverage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from experiments import REPO_ROOT
from experiments.idm.analyze_sustained_wall_v1_stage1a import _bootstrap_intervals
from experiments.idm.stage1b_common import (
    series_in_window,
)
from experiments.io import atomic_write_json as _atomic_json

HOLD_ROOT = REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_hold_m2200_p20_s6001_6030"
STAGE1A_SUMMARY = (
    REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1a_m2200_p20_s6001_6030" / "summary.json"
)
ARMS = ("WHOLD_SOFT", "WHOLD_STRONG")
FIXED_T_D = 200


def _stats(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray([float(v) for v in values], dtype=float)
    if array.size == 0:
        return {}
    return {
        "median": float(np.median(array)),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _phase_window_sums(branch: Mapping[str, Any], release_step: int) -> Dict[str, float]:
    series = branch["series"]
    return {
        "during_wall": series_in_window(series, "hard_brake_count", FIXED_T_D + 1, release_step - 1),
        "release_60": series_in_window(series, "hard_brake_count", release_step, release_step + 60),
        "after_60": series_in_window(series, "hard_brake_count", release_step + 61, 10 ** 6),
    }


def _location_deltas(c: Mapping[str, Any], arm: Mapping[str, Any]) -> Dict[str, float]:
    loc_c = c["outcomes"]["hard_brake_attribution"]["exposure_counts"]["road_id"]
    loc_a = arm["outcomes"]["hard_brake_attribution"]["exposure_counts"]["road_id"]
    roads = sorted(set(loc_c) | set(loc_a))
    return {str(road): float(loc_a.get(road, 0)) - float(loc_c.get(road, 0)) for road in roads}


def analyze(hold_root: Path, output: Path) -> Dict[str, Any]:
    seeds_dir = hold_root / "seeds"
    payloads = []
    for seed in range(6001, 6031):
        payloads.append(
            json.loads((seeds_dir / f"seed{seed}.json").read_text(encoding="utf-8"))
        )
    if not payloads:
        raise ValueError(f"no hold seed artifacts under {seeds_dir}")

    arm_results: Dict[str, Any] = {}
    for arm in ARMS:
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
        max_members = []
        incidence_c = []
        incidence_a = []
        co_collapsing = []
        for payload in payloads:
            c = payload["arms"]["C"]
            branch = payload["arms"][arm]
            release_step = int(
                (payload.get("wall_dose") or {}).get(arm, {}).get("release_step")
                or FIXED_T_D + 1
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
            phase_c = _phase_window_sums(c, release_step)
            phase_a = _phase_window_sums(branch, release_step)
            for phase, key in (("during_wall", "during_wall"), ("release_60", "release_60"), ("after_60", "after_60")):
                phase_deltas[phase].append(phase_a[key] - phase_c[key])
            loc_delta = _location_deltas(c, branch)
            for road, value in loc_delta.items():
                location_deltas.setdefault(road, []).append(value)
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
                co_collapsing.append(
                    {"seed": payload["seed"], "onset_C": onset_c, "onset_arm": onset_a}
                )
        reductions = [
            float(incidence_c[i]) - float(incidence_a[i]) for i in range(len(incidence_c))
        ]
        arm_results[arm] = {
            "incidence": {
                "C": int(sum(incidence_c)),
                arm: int(sum(incidence_a)),
                "paired_reduction_mean": float(np.mean(reductions)),
                "bootstrap": _bootstrap_intervals(reductions),
                "prevented_seeds": [
                    int(payloads[i]["seed"])
                    for i in range(len(payloads))
                    if incidence_c[i] and not incidence_a[i]
                ],
                "induced_seeds": [
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
                "n_co_collapsing": len(co_collapsing),
                "cases": co_collapsing,
            },
            "delay_delta_s_per_veh": {
                **_stats(delay_deltas),
                "bootstrap_median_ci": _bootstrap_intervals(delay_deltas).get("median"),
                "fraction_beneficial": float(np.mean(np.asarray(delay_deltas) < 0.0)),
            },
            "hard_brake_delta_steps": {
                **_stats(hb_total),
                "phase_delta": {phase: _stats(values) for phase, values in phase_deltas.items()},
                "location_delta": {
                    road: _stats(values) for road, values in sorted(location_deltas.items())
                },
            },
            "low_ttc_delta_steps": {
                **_stats(lttc_total),
            },
            "dose": {
                "member_seconds": _stats(member_seconds),
                "per_seed_mean_lane_coverage": _stats(coverage_means),
                "fraction_200_900_window_full_occupied_lane_coverage": _stats(coverage_full_fraction),
            },
        }

    stage1a = None
    if STAGE1A_SUMMARY.exists():
        stage1a = json.loads(STAGE1A_SUMMARY.read_text(encoding="utf-8"))
    result = {
        "schema": "sustained_wall_v1_stage1b_hold_summary/1",
        "n_seeds": len(payloads),
        "fixed_decision_step": FIXED_T_D,
        "arms": arm_results,
        "integrity_summary": {
            "all_prefix_identical_through_200": all(
                all(
                    payload["integrity"][arm]["prefix_identical_through_decision_step"]
                    for arm in ARMS
                )
                for payload in payloads
            ),
            "all_first_divergence_at_201": all(
                all(payload["integrity"][arm]["first_divergence_step"] == 201 for arm in ARMS)
                for payload in payloads
            ),
            "all_c_replay_matches_stage1a": all(
                payload["c_replay_digest_matches_stage1a"] for payload in payloads
            ),
            "all_tau_restored": all(
                payload["tau_integrity"][arm]["restored"] for payload in payloads for arm in ARMS
            ),
        },
        "stage1a_180s_pulse_reference": {
            "delay_median_s_per_veh": (stage1a or {}).get("delay", {}).get("W", {}).get(
                "delta_s_per_veh", {}
            ).get("median"),
            "delay_p10_p90": [
                (stage1a or {}).get("delay", {}).get("W", {}).get("delta_s_per_veh", {}).get("p10"),
                (stage1a or {}).get("delay", {}).get("W", {}).get("delta_s_per_veh", {}).get("p90"),
            ],
            "health_incidence": (stage1a or {}).get("collapse", {}).get("health", {}).get(
                "incidence_counts"
            ),
            "health_total_duration_delta_median": (stage1a or {}).get("collapse", {}).get(
                "health", {}
            ).get("W_vs_C_duration", {}).get("total_duration_delta_s", {}).get("median"),
            "health_total_duration_delta_ci90": (stage1a or {}).get("collapse", {}).get(
                "health", {}
            ).get("W_vs_C_duration", {}).get("total_duration_delta_s", {}).get(
                "bootstrap_median_ci", {}
            ).get("ci90"),
            "hard_brake_delta_median": _stage1a_hb_median(),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "hold_summary.json", result)
    return result


def _stage1a_hb_median() -> Optional[float]:
    import numpy as np

    seeds_dir = (
        REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1a_m2200_p20_s6001_6030" / "seeds"
    )
    deltas = []
    for seed in range(6001, 6031):
        payload = json.loads((seeds_dir / f"seed{seed}.json").read_text(encoding="utf-8"))
        deltas.append(
            float(payload["arms"]["W"]["outcomes"]["hard_brake_vehicle_steps"])
            - float(payload["arms"]["C"]["outcomes"]["hard_brake_vehicle_steps"])
        )
    return float(np.median(deltas)) if deltas else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold-root", type=Path, default=HOLD_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=str(REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep"),
    )
    args = parser.parse_args()
    result = analyze(Path(args.hold_root).resolve(), Path(args.output).resolve())
    print(json.dumps(result["integrity_summary"], indent=2))
    for arm in ARMS:
        block = result["arms"][arm]
        print(json.dumps({arm: block}, indent=1)[:1500])
        print()


if __name__ == "__main__":
    main()
