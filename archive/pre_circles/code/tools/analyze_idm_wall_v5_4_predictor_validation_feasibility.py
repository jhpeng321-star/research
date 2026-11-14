"""Analyze the treatment-blind p30 predictor-validation feasibility partition."""

from experiments.idm.wall_v5_4_predictor_contract import (
    CAV_PENETRATION,
    VALIDATION_SEEDS,
    validation_availability_rule,
)
from tools import analyze_idm_wall_v5_4_feasibility as analyzer


analyzer.CAV_PENETRATION = CAV_PENETRATION
analyzer.FEASIBILITY_ANALYSIS_SCHEMA = "idm_wall_v5_4_predictor_validation_feasibility/1"
analyzer.SUPPORT_SEEDS = VALIDATION_SEEDS
analyzer.availability_rule = validation_availability_rule


if __name__ == "__main__":
    raise SystemExit(analyzer.main())
