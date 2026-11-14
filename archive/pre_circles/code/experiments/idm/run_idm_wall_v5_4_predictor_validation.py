"""Generate the frozen p30 paired predictor-validation counterfactual dataset."""

from pathlib import Path
from typing import Any

from experiments import REPO_ROOT
from experiments.idm import run_idm_wall_v5_4_support_pilot as runner
from experiments.idm.wall_v5_4_predictor_contract import (
    CAV_PENETRATION,
    VALIDATION_SEEDS,
    load_validation_feasibility_scan,
    predictor_manifest,
    require_support_authorization,
    select_partition_repeat_cases,
    validate_role_seeds,
)
from experiments.idm.wall_v5_4_predictor_validation_contract import (
    require_frozen_model_bundle,
)


_base_summary = runner.build_summary


def _validation_summary(output: Path) -> dict[str, Any]:
    summary = _base_summary(output)
    base_gates = dict(summary["gates"])
    required_names = (
        "complete_20_seed_4_window_grid",
        "unique_case_ids",
        "exact_replay",
        "causal_identity_and_damping_preservation",
        "no_partial_lane_hold",
        "compactness_not_silently_held",
        "effects_exceed_exact_replay_noise",
        "dose_success_at_least_80_percent",
    )
    integrity = {name: bool(base_gates[name]) for name in required_names}
    integrity["complete_60_seed_4_window_grid"] = integrity.pop(
        "complete_20_seed_4_window_grid"
    )
    passed = all(integrity.values())
    summary.update(
        {
            "schema": "idm_wall_v5_4_predictor_validation_summary/1",
            "scientific_role": "untouched_exploratory_gate_b_validation",
            "support_diversity_diagnostics": base_gates,
            "validation_integrity_gates": integrity,
            "passed": passed,
            "predictor_development_generation_authorized": False,
            "predictor_fitting_authorized": False,
            "validation_analysis_authorized": passed,
            "final_test_authorized": False,
        }
    )
    return summary


runner.CAV_PENETRATION = CAV_PENETRATION
runner.SUPPORT_SEEDS = VALIDATION_SEEDS
runner.validate_support_seeds = lambda seeds: validate_role_seeds("validation", seeds)
runner.load_feasibility_scan = load_validation_feasibility_scan
runner.select_repeat_cases = lambda case_root: select_partition_repeat_cases(
    case_root, seeds=VALIDATION_SEEDS, count=4
)
runner.support_manifest = lambda: predictor_manifest("validation", VALIDATION_SEEDS)
runner.build_summary = _validation_summary
runner.EXPECTED_REPEAT_COUNT = 4
runner.SCIENTIFIC_ROLE = "untouched_exploratory_gate_b_validation"
runner.MANIFEST_SCHEMA = "idm_wall_v5_4_predictor_validation_manifest/1"
runner.SEED_SCHEMA = "idm_wall_v5_4_predictor_validation_seed/1"
runner.CASE_SCHEMA = "idm_wall_v5_4_predictor_validation_case/1"
runner.SUMMARY_SCHEMA = "idm_wall_v5_4_predictor_validation_summary/1"
runner.DEFAULT_SCAN = (
    REPO_ROOT
    / "runs"
    / "probe"
    / "idm_wall_v5_4_predictor_validation_feasibility_p30_s4601_4660"
)
runner.DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs"
    / "eval"
    / "idm_wall_v5_4_predictor_validation_p30_s4601_4660"
)


if __name__ == "__main__":
    require_support_authorization(REPO_ROOT)
    frozen = require_frozen_model_bundle()
    original_manifest = runner.support_manifest
    runner.support_manifest = lambda: {
        **original_manifest(),
        "frozen_gate_b_bundle": frozen,
    }
    raise SystemExit(runner.main())
