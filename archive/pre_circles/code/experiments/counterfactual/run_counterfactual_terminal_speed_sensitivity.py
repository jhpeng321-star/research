"""Run the bounded v5.2 terminal moving-wall speed sensitivity experiment.

This is deliberately a treatment-development diagnostic, not a headroom gate
or predictor dataset. It replays only decisions that have already been
inspected and excluded from later prospective evaluation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


from experiments import REPO_ROOT

PROJECT = REPO_ROOT
GENERATOR = PROJECT / "generate_counterfactual_escalation_headroom.py"
DEFAULT_OUTPUT = PROJECT / "runs" / "probe" / "counterfactual_terminal_speed_v5_2_development"
CASE_GROUPS: dict[int, tuple[int, ...]] = {
    120: (609, 612),
    180: (504, 606, 607),
}
GRID_SEEDS = tuple(sorted({seed for seeds in CASE_GROUPS.values() for seed in seeds}))
GRID_DECISION_STEPS = tuple(sorted(CASE_GROUPS))
DEFAULT_REDUCTION_FRACTIONS = (0.10, 0.20, 0.30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run the excluded-case terminal wall-speed sensitivity diagnostic.")
    )
    parser.add_argument(
        "--reduction-fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_REDUCTION_FRACTIONS),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Read completed subruns without launching SUMO.",
    )
    return parser.parse_args()


def _fraction_tag(value: float) -> str:
    return f"r{int(round(float(value) * 100)):02d}"


def _subrun_dir(output_dir: Path, fraction: float) -> Path:
    return output_dir / _fraction_tag(fraction)


def _generator_command(
    *,
    fraction: float,
    seeds: Sequence[int],
    decision_steps: Sequence[int],
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(GENERATOR),
        "--seeds",
        *(str(seed) for seed in seeds),
        "--decision-steps",
        *(str(step) for step in decision_steps),
        "--outcome-steps",
        "180",
        "--treatment-steps",
        "20",
        "--cav-penetration",
        "0.2",
        "--demand",
        "hyper",
        "--workers",
        "1",
        "--development",
        "--dataset-role",
        "treatment_development",
        "--wall-speed",
        "10.0",
        "--wall-d-lo",
        "200.0",
        "--wall-d-hi",
        "1000.0",
        "--wall-tau",
        "3.0",
        "--wall-a-max-dec",
        "0.3",
        "--tau-ramp-step",
        "0.05",
        "--geometry-mode",
        "compact_moving",
        "--max-longitudinal-span-m",
        "50.0",
        "--compact-release-boundary-m",
        "50.0",
        "--compact-selection-mode",
        "trajectory_coherent",
        "--compact-assembly-duration-s",
        "10.0",
        "--compact-assembly-speed-tolerance-mps",
        "2.0",
        "--compact-assembly-a-max-dec-mps2",
        "1.0",
        "--compact-control-mode",
        "adaptive_servo",
        "--compact-servo-a-max-dec-mps2",
        "1.0",
        "--compact-max-activation-speed-spread-mps",
        "4.0",
        "--compact-max-speed-reduction-fraction",
        str(float(fraction)),
        "--compact-shock-gap-lo-mps",
        "5.0",
        "--compact-shock-gap-hi-mps",
        "15.0",
        "--compact-position-kp-s-inv",
        "0.05",
        "--compact-position-kd",
        "0.5",
        "--compact-position-speed-offset-cap-mps",
        "1.0",
        "--compact-safety-min-front-ttc-s",
        "3.0",
        "--compact-safety-min-rear-ttc-s",
        "4.0",
        "--compact-influence-upstream-m",
        "100.0",
        "--compact-influence-max-followers",
        "3",
        "--compact-influence-min-ttc-s",
        "1.5",
        "--output-dir",
        str(output_dir),
    ]


def _run_subrun(job: Mapping[str, Any]) -> dict[str, Any]:
    fraction = float(job["fraction"])
    seeds = tuple(int(seed) for seed in job["seeds"])
    decision_steps = tuple(int(step) for step in job["decision_steps"])
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    command = _generator_command(
        fraction=fraction,
        seeds=seeds,
        decision_steps=decision_steps,
        output_dir=output_dir,
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = output_dir / "runner.log"
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"subrun {_fraction_tag(fraction)} failed; see {log_path}")
    return {
        "fraction": fraction,
        "seeds": list(seeds),
        "decision_steps": list(decision_steps),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
    }


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _strict_dose_geometry(row: Mapping[str, Any]) -> bool:
    dose = row.get("treatment_dose") or {}
    geometry = dose.get("wall_geometry") or {}
    return bool(
        dose.get("dose_at_least_80_percent")
        and float(geometry.get("fraction_within_configured_span") or 0.0) >= 0.8
    )


def _case_result(row: Mapping[str, Any], fraction: float) -> dict[str, Any]:
    dose = row.get("treatment_dose") or {}
    geometry = dose.get("wall_geometry") or {}
    oracle = row.get("safety_service_oracle") or {}
    effects = oracle.get("effects") or {}
    control = row["arms"]["0"]["outcomes"]
    treatment = row["arms"]["1"]["outcomes"]
    terminal = dose.get("terminal_tail_speed_delivery") or {}
    collapse_prevented = bool(
        control.get("sustained_congestion") and not treatment.get("sustained_congestion")
    )
    return {
        "case_id": row["case_id"],
        "seed": int(row["seed"]),
        "decision_step": int(row["decision_step"]),
        "reduction_fraction": float(fraction),
        "wall_engaged": bool(row["arms"]["1"].get("wall_engaged")),
        "strict_dose_geometry": _strict_dose_geometry(row),
        "realized_duration_fraction": dose.get("realized_duration_fraction"),
        "lane_coverage_fraction": dose.get("lane_coverage_fraction"),
        "geometry_fraction_within_span": geometry.get("fraction_within_configured_span"),
        "max_longitudinal_span_m": geometry.get("max_longitudinal_span_m"),
        "requested_common_target_mps": dose.get("mean_adaptive_common_target_mps"),
        "terminal_actual_speed_mps": terminal.get("mean_actual_speed_mps"),
        "terminal_projected_target_mps": terminal.get("mean_projected_target_mps"),
        "terminal_mean_absolute_error_mps": terminal.get("mean_absolute_target_error_mps"),
        "terminal_fraction_within_1_mps": terminal.get("fraction_within_1_mps"),
        "safety_projection_clip_count": dose.get("safety_projection_clip_count"),
        "safety_admissible": bool(oracle.get("safety_admissible")),
        "service_noninferior": bool(oracle.get("service_noninferior")),
        "meaningful_benefit": bool(oracle.get("meaningful_benefit")),
        "wall_selected": bool(oracle.get("wall_selected")),
        "collapse_prevented": collapse_prevented,
        "control_sustained_congestion": bool(control.get("sustained_congestion")),
        "treatment_sustained_congestion": bool(treatment.get("sustained_congestion")),
        "collision_excess_vehicle_count": effects.get("collision_excess_vehicle_count"),
        "teleport_excess_vehicle_count": effects.get("teleport_excess_vehicle_count"),
        "hard_brake_exposure_excess_vehicle_steps": effects.get(
            "hard_brake_exposure_excess_vehicle_steps"
        ),
        "hard_brake_onset_excess": effects.get("hard_brake_onset_excess"),
        "low_ttc_excess_steps": effects.get("low_ttc_excess_steps"),
        "merge_discharge_benefit": effects.get("merge_discharge_benefit"),
        "delay_benefit_s": effects.get("delay_benefit_s"),
        "refusal_reasons": list(oracle.get("refusal_reasons") or ()),
    }


def _mean(values: Iterable[Any]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def _summarize(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_fraction: dict[str, list[Mapping[str, Any]]] = {}
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in results:
        tag = _fraction_tag(float(row["reduction_fraction"]))
        by_fraction.setdefault(tag, []).append(row)
        by_case.setdefault(str(row["case_id"]), []).append(row)

    level_summary = {}
    for tag, rows in sorted(by_fraction.items()):
        strict_rows = [row for row in rows if row["strict_dose_geometry"]]
        level_summary[tag] = {
            "reduction_fraction": rows[0]["reduction_fraction"],
            "case_count": len(rows),
            "engaged_count": sum(bool(row["wall_engaged"]) for row in rows),
            "strict_dose_geometry_count": len(strict_rows),
            "safety_admissible_strict_count": sum(
                bool(row["safety_admissible"]) for row in strict_rows
            ),
            "wall_selected_strict_count": sum(bool(row["wall_selected"]) for row in strict_rows),
            "collapse_prevention_strict_count": sum(
                bool(row["collapse_prevented"]) for row in strict_rows
            ),
            "mean_requested_common_target_mps": _mean(
                row["requested_common_target_mps"] for row in strict_rows
            ),
            "mean_terminal_actual_speed_mps": _mean(
                row["terminal_actual_speed_mps"] for row in strict_rows
            ),
            "mean_terminal_fraction_within_1_mps": _mean(
                row["terminal_fraction_within_1_mps"] for row in strict_rows
            ),
            "mean_hard_brake_exposure_excess": _mean(
                row["hard_brake_exposure_excess_vehicle_steps"] for row in strict_rows
            ),
            "mean_low_ttc_excess_steps": _mean(row["low_ttc_excess_steps"] for row in strict_rows),
            "mean_merge_discharge_benefit": _mean(
                row["merge_discharge_benefit"] for row in strict_rows
            ),
            "mean_delay_benefit_s": _mean(row["delay_benefit_s"] for row in strict_rows),
        }

    case_comparison = {}
    selected_sets = []
    for case_id, rows in sorted(by_case.items()):
        rows = sorted(rows, key=lambda row: float(row["reduction_fraction"]))
        selected = [
            float(row["reduction_fraction"])
            for row in rows
            if row["strict_dose_geometry"] and row["wall_selected"]
        ]
        selected_sets.append(tuple(selected))
        case_comparison[case_id] = {
            "selected_reduction_fractions": selected,
            "safe_reduction_fractions": [
                float(row["reduction_fraction"])
                for row in rows
                if row["strict_dose_geometry"] and row["safety_admissible"]
            ],
            "collapse_preventing_reduction_fractions": [
                float(row["reduction_fraction"])
                for row in rows
                if row["strict_dose_geometry"] and row["collapse_prevented"]
            ],
            "results": rows,
        }

    return {
        "schema_version": "counterfactual_terminal_speed_sensitivity/1",
        "scientific_status": ("excluded-case treatment development; not prospective evidence"),
        "frozen_parameters": {
            "servo_a_max_dec_mps2": 1.0,
            "assembly_a_max_dec_mps2": 1.0,
            "treatment_duration_s": 20.0,
            "recruitment_band_m": [200.0, 1000.0],
            "compact_span_cap_m": 50.0,
            "release_boundary_m": 50.0,
            "wall_tau_s": 3.0,
            "control_arm": "identical frozen decentralized damping",
        },
        "level_summary": level_summary,
        "case_comparison": case_comparison,
        "state_dependent_selected_sets_observed": len(set(selected_sets)) > 1,
        "case_results": list(results),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    fractions = tuple(sorted(set(float(value) for value in args.reduction_fractions)))
    if not fractions or any(value <= 0.0 or value > 0.40 for value in fractions):
        raise ValueError("reduction fractions must be in (0, 0.40]")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "counterfactual_terminal_speed_sensitivity/1",
        "scientific_status": "excluded development cases only",
        "reduction_fractions": list(fractions),
        "case_groups": {str(step): list(seeds) for step, seeds in CASE_GROUPS.items()},
        "generator_grid_seeds": list(GRID_SEEDS),
        "generator_grid_decision_steps": list(GRID_DECISION_STEPS),
        "auxiliary_grid_rows_are_excluded_from_primary_summary": True,
        "fixed_servo_a_max_dec_mps2": 1.0,
        "command_preview": [
            _generator_command(
                fraction=fraction,
                seeds=GRID_SEEDS,
                decision_steps=GRID_DECISION_STEPS,
                output_dir=_subrun_dir(output_dir, fraction),
            )
            for fraction in fractions
        ],
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    jobs = [
        {
            "fraction": fraction,
            "seeds": GRID_SEEDS,
            "decision_steps": GRID_DECISION_STEPS,
            "output_dir": _subrun_dir(output_dir, fraction),
        }
        for fraction in fractions
    ]
    if not args.analyze_only:
        print(
            f"launching {len(jobs)} bounded subruns with {max(1, int(args.workers))} workers",
            flush=True,
        )
        completed_jobs = []
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = {executor.submit(_run_subrun, job): job for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                completed_jobs.append(result)
                print(
                    f"completed {_fraction_tag(result['fraction'])} "
                    f"in {result['elapsed_wall_seconds']:.1f}s",
                    flush=True,
                )
        (output_dir / "subrun_status.json").write_text(
            json.dumps(completed_jobs, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    expected = {
        (int(seed), int(step), float(fraction))
        for fraction in fractions
        for step, seeds in CASE_GROUPS.items()
        for seed in seeds
    }
    results = []
    observed = set()
    auxiliary_results = []
    for fraction in fractions:
        cases_path = _subrun_dir(output_dir, fraction) / "cases.jsonl"
        if not cases_path.exists():
            raise FileNotFoundError(f"missing completed subrun: {cases_path}")
        for row in _read_jsonl(cases_path):
            key = (int(row["seed"]), int(row["decision_step"]), fraction)
            result = _case_result(row, fraction)
            if key in expected:
                observed.add(key)
                results.append(result)
            else:
                auxiliary_results.append(result)
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"missing expected sensitivity cases: {missing}")
    summary = _summarize(results)
    summary["experiment_manifest"] = manifest
    summary["auxiliary_case_results"] = auxiliary_results
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
