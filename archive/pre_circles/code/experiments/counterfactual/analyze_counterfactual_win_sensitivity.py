"""Compare post-hoc wall-win definitions on frozen paired replay data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from counterfactual.win_sensitivity import WIN_RULES, analyze_split


DEFAULT_DEV_DATA = Path(
    "runs/probe/counterfactual_local_v2_dev_combined_s101_130_steps120_180_240.jsonl"
)
DEFAULT_DEV_READINESS = Path(
    "runs/probe/counterfactual_local_v2_dev_s101_130_steps120_180_240.readiness.json"
)
DEFAULT_LOCKED_DATA = Path(
    "runs/probe/counterfactual_local_v2_locked_s201_210_steps120_180_240.jsonl"
)
DEFAULT_LOCKED_READINESS = Path(
    "runs/probe/counterfactual_local_v2_locked_s201_210_steps120_180_240.readiness.json"
)
DEFAULT_JSON_OUTPUT = Path("runs/probe/counterfactual_win_sensitivity_local_v2.json")
DEFAULT_MARKDOWN_OUTPUT = Path("docs/COUNTERFACTUAL_WIN_SENSITIVITY_LOCAL_V2.md")
DEFAULT_MERGE_GATE_REPORT = Path(
    "runs/probe/counterfactual_merge_service_gate_local_v2.report.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-data", type=Path, default=DEFAULT_DEV_DATA)
    parser.add_argument("--dev-readiness", type=Path, default=DEFAULT_DEV_READINESS)
    parser.add_argument("--locked-data", type=Path, default=DEFAULT_LOCKED_DATA)
    parser.add_argument("--locked-readiness", type=Path, default=DEFAULT_LOCKED_READINESS)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument(
        "--merge-gate-report",
        type=Path,
        default=DEFAULT_MERGE_GATE_REPORT,
        help="Optional completed revised-gate diagnostic to summarize.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _markdown(report: Mapping[str, Any], sources: Mapping[str, str]) -> str:
    lines = [
        "# Counterfactual wall-win sensitivity (local-v2)",
        "",
        "## Bottom line",
        "",
        "The frozen label calls soft a win only when it is Pareto-nonworse on "
        "pre-merge sustained congestion, pre-merge queue area, aggregate "
        "incremental delay, and merge discharge, with at least one strict "
        "improvement. Safety is then imposed as a separate hard gate.",
        "",
        "That definition is too restrictive for an inflow-control wall: the "
        "wall deliberately creates upstream holding, and both queue area and "
        "SUMO timeLoss can therefore rise even when bottleneck service is "
        "preserved. More importantly, the frozen congestion label is measured "
        "on the same pre-merge edge where the wall operates, so it can mix an "
        "intentional control queue with an uncontrolled bottleneck collapse.",
        "",
        "The recommended candidate label is **merge service, exact throughput**: "
        "the soft arm must be safety-admissible, must not lower paired merge "
        "discharge or mean merge speed, and must strictly improve at least one "
        "of those two quantities. Queue and delay remain mandatory reported "
        "secondary costs, but they do not veto the label. The one-vehicle "
        "throughput-tolerance version is a sensitivity check, not the primary "
        "claim, until a longer recovery horizon demonstrates catch-up.",
        "",
        "## What each rule counts",
        "",
        "| Rule | Definition |",
        "|---|---|",
    ]
    for rule in WIN_RULES:
        lines.append(f"| {rule.title} | {rule.description} |")

    for split_name in ("development", "locked"):
        split = report["splits"][split_name]
        lines.extend(
            [
                "",
                f"## {split_name.title()} paired cases",
                "",
                f"Eligible paired cases: {split['eligible_case_count']}.",
                "",
                "| Rule | Traffic wins | Safe wins | Rescues | Induced pre-merge congestion | Policy discharge gain (veh/case) | Policy merge-speed gain (m/s) | Policy delay change (s/case) | Policy queue change (veh-s/case) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rule in WIN_RULES:
            result = split["rules"][rule.name]
            delta = result["policy_delta_vs_always_off"]
            lines.append(
                f"| {rule.title} | {result['traffic_win_count']} | "
                f"{result['selected_safe_soft_count']} | "
                f"{result['selected_rescue_count']} | "
                f"{result['selected_induced_pre_merge_congestion_count']} | "
                f"{_fmt(delta['mean_merge_discharge_gain'])} | "
                f"{_fmt(delta['mean_merge_speed_gain_mps'], 3)} | "
                f"{_fmt(delta['mean_incremental_delay_change_s'])} | "
                f"{_fmt(delta['mean_queue_area_change_vehicle_s'])} |"
            )

        candidate = split["rules"]["bottleneck_service_exact"]
        candidate_delta = candidate["policy_delta_vs_always_off"]
        delay_cost = candidate["selected_delay_cost_s"]
        queue_cost = candidate["selected_queue_cost_vehicle_s"]
        lines.extend(
            [
                "",
                "For the recommended candidate:",
                "",
                f"- {candidate['selected_safe_soft_count']} safe soft cases are "
                f"selected; {candidate['safety_rejected_traffic_win_count']} "
                "additional traffic-effective cases fail the frozen safety budget.",
                f"- The oracle-labelled policy changes mean merge discharge by "
                f"{_fmt(candidate_delta['mean_merge_discharge_gain'])} vehicles per "
                f"case and mean merge speed by "
                f"{_fmt(candidate_delta['mean_merge_speed_gain_mps'], 3)} m/s versus "
                "always off.",
                f"- Delay rises in {delay_cost['case_count']} selected cases "
                f"(mean positive cost {_fmt(delay_cost['mean'])} s; maximum "
                f"{_fmt(delay_cost['max'])} s).",
                f"- Pre-merge queue area rises in {queue_cost['case_count']} "
                f"selected cases (mean positive cost {_fmt(queue_cost['mean'])} "
                f"veh-s; maximum {_fmt(queue_cost['max'])} veh-s).",
            ]
        )

    lines.extend(
        [
            "",
            "## Can the current features predict the revised win?",
            "",
        ]
    )
    predictor = report.get("merge_service_predictor_diagnostic")
    if predictor:
        development = predictor["development"]
        retrospective = predictor.get("retrospective_test")
        dev_classification = development["classification"]
        dev_delta = development["predicted_gate_delta_vs_off"]
        speed_metrics = predictor["models"]["merge_speed_gain_mps"]["metrics"]
        discharge_metrics = predictor["models"]["merge_discharge_gain"]["metrics"]
        lines.extend(
            [
                "No. The new label is better aligned with the wall mechanism, "
                "but relabeling does not make the current pre-action features "
                "predictive.",
                "",
                f"- Nested grouped development: merge-speed gain "
                f"R2={_fmt(speed_metrics['r2'], 3)}, discharge gain "
                f"R2={_fmt(discharge_metrics['r2'], 3)}; both are worse than "
                "predicting the development mean.",
                f"- The cross-fitted gate activates "
                f"{dev_classification['predicted_soft_count']} of "
                f"{report['splits']['development']['eligible_case_count']} cases, "
                f"with precision {_fmt(100 * dev_classification['precision'], 1)}% "
                f"and recall {_fmt(100 * dev_classification['recall'], 1)}%. It "
                f"also makes {development['policies']['predicted_gate']['unsafe_soft_count']} "
                "unsafe activations under the observed frozen safety budget.",
                f"- Its development policy averages "
                f"{_fmt(dev_delta['mean_merge_discharge_gain'])} additional merge "
                f"discharges and {_fmt(dev_delta['mean_merge_speed_gain_mps'], 3)} "
                "m/s merge-speed gain per case versus always off, but this does "
                "not transfer.",
            ]
        )
        if retrospective:
            test_classification = retrospective["classification"]
            test_delta = retrospective["predicted_gate_delta_vs_off"]
            lines.extend(
                [
                    f"- On the already-consumed retrospective split, it activates "
                    f"{test_classification['predicted_soft_count']} of "
                    f"{retrospective['case_count']} cases with precision "
                    f"{_fmt(100 * test_classification['precision'], 1)}%, including "
                    f"{retrospective['policies']['predicted_gate']['unsafe_soft_count']} "
                    "unsafe activation.",
                    f"- That retrospective predicted policy loses "
                    f"{_fmt(abs(test_delta['mean_merge_discharge_gain']))} merge "
                    f"discharges per case and {_fmt(abs(test_delta['mean_merge_speed_gain_mps']), 3)} "
                    "m/s mean merge speed, while adding "
                    f"{_fmt(test_delta['mean_delay_change_s'])} s delay and "
                    f"{_fmt(test_delta['mean_queue_area_change_vehicle_s'])} veh-s "
                    "queue area per case versus always off.",
                    "",
                    "Therefore the revised rule is a label/design improvement, "
                    "not evidence of a working deployable gate. The next model "
                    "must use early post-pulse response features or a randomized "
                    "probe-and-commit design; more seeds alone will not fix a "
                    "feature set whose grouped signal is currently below baseline.",
                ]
            )
    else:
        lines.extend(
            [
                "The revised predictor diagnostic was not available when this "
                "report was generated. Run `train_counterfactual_merge_service_gate.py` "
                "and regenerate this report.",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation for the paper",
            "",
            "Use Krauss to test whether the wall can preserve bottleneck service "
            "under stochastic car-following heterogeneity. Do not claim that its "
            "timeLoss response reproduces the full economic cost of a real traffic "
            "breakdown. Present the available IDM collapse result as a model-form "
            "stress test showing that, under a car-following model with stronger "
            "collapse-delay amplification, the temporary wall cost can be offset "
            "by avoided breakdown loss. Insert the measured IDM effect size and "
            "uncertainty before making that numerical claim.",
            "",
            "This analysis is post-hoc and oracle-labelled: it changes what an "
            "observed paired case counts as, but it does not repair the failed "
            "pre-action predictor. If this label is adopted, retrain a two-head "
            "gate for merge-speed gain and discharge gain and evaluate it once on "
            "new untouched seeds. Also add a merge-edge sustained-collapse trace "
            "metric in the next data generation so the primary stability endpoint "
            "is not measured inside the wall's holding zone.",
            "",
            "## Sources",
            "",
            f"- Development data: `{sources['development_data']}`",
            f"- Development safety audit: `{sources['development_readiness']}`",
            f"- Locked data: `{sources['locked_data']}`",
            f"- Locked safety audit: `{sources['locked_readiness']}`",
            f"- Revised gate diagnostic: `{sources.get('merge_gate_report', 'not available')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    sources = {
        "development_data": str(args.dev_data.resolve()),
        "development_readiness": str(args.dev_readiness.resolve()),
        "locked_data": str(args.locked_data.resolve()),
        "locked_readiness": str(args.locked_readiness.resolve()),
    }
    merge_gate_report = None
    if args.merge_gate_report is not None and args.merge_gate_report.exists():
        merge_gate_report = _load_json(args.merge_gate_report)
        sources["merge_gate_report"] = str(args.merge_gate_report.resolve())
    report = {
        "schema_version": 1,
        "analysis_status": "post_hoc_label_sensitivity_not_predictor_evaluation",
        "frozen_experiment_modified": False,
        "sources": sources,
        "rules": [
            {
                "name": rule.name,
                "title": rule.title,
                "description": rule.description,
            }
            for rule in WIN_RULES
        ],
        "measurement_warning": (
            "The frozen sustained-congestion and queue-area endpoints use the "
            "PRE_MERGE_EDGE where the wall deliberately holds traffic."
        ),
        "splits": {
            "development": analyze_split(
                _load_jsonl(args.dev_data), _load_json(args.dev_readiness)
            ),
            "locked": analyze_split(
                _load_jsonl(args.locked_data), _load_json(args.locked_readiness)
            ),
        },
    }
    if merge_gate_report is not None:
        report["merge_service_predictor_diagnostic"] = merge_gate_report
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(_markdown(report, sources), encoding="utf-8")
    print(f"wrote {args.json_output}")
    print(f"wrote {args.markdown_output}")
    for split_name, split in report["splits"].items():
        current = split["rules"]["current_pareto"]
        candidate = split["rules"]["bottleneck_service_exact"]
        print(
            f"{split_name}: current safe wins="
            f"{current['selected_safe_soft_count']}; merge-service safe wins="
            f"{candidate['selected_safe_soft_count']}"
        )


if __name__ == "__main__":
    main()
