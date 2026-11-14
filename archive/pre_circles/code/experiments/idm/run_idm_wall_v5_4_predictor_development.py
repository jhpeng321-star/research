"""Generate the frozen p30 paired predictor-development dataset."""

from pathlib import Path
from typing import Any

from experiments import REPO_ROOT
from experiments.idm import run_idm_wall_v5_4_support_pilot as runner
from experiments.idm.wall_v5_4_predictor_contract import (
    CAV_PENETRATION,
    DEVELOPMENT_SEEDS,
    fitting_gates,
    load_development_feasibility_scan,
    predictor_manifest,
    require_support_authorization,
    select_repeat_cases,
    validate_role_seeds,
)


_base_summary = runner.build_summary


def _development_summary(output: Path) -> dict[str, Any]:
    summary = _base_summary(output)
    support = summary["support"]
    thresholds = fitting_gates()
    gates = {
        "admitted_independent_seeds": (
            support["admitted_seed_count"] >= thresholds["minimum_admitted_independent_seeds"]
        ),
        "dose_qualified_independent_seeds": (
            support["dose_qualified_seed_count"]
            >= thresholds["minimum_dose_qualified_independent_seeds"]
        ),
        "safe_independent_seeds": (
            support["safe_seed_count"] >= thresholds["minimum_safe_independent_seeds"]
        ),
        "unsafe_independent_seeds": (
            support["unsafe_seed_count"] >= thresholds["minimum_unsafe_independent_seeds"]
        ),
        "joint_beneficial_independent_seeds": (
            support["joint_y1_seed_count"]
            >= thresholds["minimum_joint_beneficial_independent_seeds"]
        ),
        "joint_nonbeneficial_independent_seeds": (
            support["joint_y0_seed_count"]
            >= thresholds["minimum_joint_nonbeneficial_independent_seeds"]
        ),
    }
    summary.update(
        {
            "schema": "idm_wall_v5_4_predictor_development_summary/1",
            "scientific_role": "grouped_predictor_development_not_validation",
            "development_fitting_thresholds": thresholds,
            "development_fitting_gates": gates,
            "predictor_development_generation_authorized": False,
            "predictor_fitting_authorized": bool(summary["passed"] and all(gates.values())),
            "validation_generation_authorized": bool(summary["passed"] and all(gates.values())),
            "final_test_authorized": False,
        }
    )
    return summary


runner.CAV_PENETRATION = CAV_PENETRATION
runner.SUPPORT_SEEDS = DEVELOPMENT_SEEDS
runner.validate_support_seeds = lambda seeds: validate_role_seeds("development", seeds)
runner.load_feasibility_scan = load_development_feasibility_scan
runner.select_repeat_cases = select_repeat_cases
runner.support_manifest = lambda: predictor_manifest("development", DEVELOPMENT_SEEDS)
runner.build_summary = _development_summary
runner.EXPECTED_REPEAT_COUNT = 4
runner.SCIENTIFIC_ROLE = "grouped_predictor_development_not_validation"
runner.MANIFEST_SCHEMA = "idm_wall_v5_4_predictor_development_manifest/1"
runner.SEED_SCHEMA = "idm_wall_v5_4_predictor_development_seed/1"
runner.CASE_SCHEMA = "idm_wall_v5_4_predictor_development_case/1"
runner.SUMMARY_SCHEMA = "idm_wall_v5_4_predictor_development_summary/1"
runner.DEFAULT_SCAN = (
    REPO_ROOT
    / "runs"
    / "probe"
    / "idm_wall_v5_4_predictor_development_feasibility_p30_s4401_4520"
)
runner.DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs"
    / "eval"
    / "idm_wall_v5_4_predictor_development_p30_s4401_4520"
)


if __name__ == "__main__":
    require_support_authorization(REPO_ROOT)
    raise SystemExit(runner.main())
