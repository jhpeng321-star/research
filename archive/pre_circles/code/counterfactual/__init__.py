"""Counterfactual data-generation, outcomes, and oracle utilities."""

from counterfactual.oracle import (
    analyze_oracle,
    choose_least_intensive_beneficial_action,
    choose_oracle_action,
    choose_traffic_oracle_action,
)
from counterfactual.calibration import calibrate_outcome_definition
from counterfactual.outcomes import OutcomeDefinition, summarize_branch
from counterfactual.pilot import features_from_trace, generate_paired_case
from counterfactual.readiness import evaluate_readiness
from counterfactual.twin_replay import FrozenPulseSchedule, compare_traces, run_replay

__all__ = [
    "OutcomeDefinition",
    "analyze_oracle",
    "calibrate_outcome_definition",
    "choose_oracle_action",
    "choose_least_intensive_beneficial_action",
    "choose_traffic_oracle_action",
    "compare_traces",
    "generate_paired_case",
    "features_from_trace",
    "FrozenPulseSchedule",
    "evaluate_readiness",
    "run_replay",
    "summarize_branch",
]
