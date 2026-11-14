"""Analyze the treatment-blind p30 predictor-development feasibility partition."""

from experiments.idm.wall_v5_4_predictor_contract import (
    CAV_PENETRATION,
    DEVELOPMENT_SEEDS,
    development_availability_rule,
)
from tools import analyze_idm_wall_v5_4_feasibility as analyzer


analyzer.CAV_PENETRATION = CAV_PENETRATION
analyzer.FEASIBILITY_ANALYSIS_SCHEMA = "idm_wall_v5_4_predictor_development_feasibility/1"
analyzer.SUPPORT_SEEDS = DEVELOPMENT_SEEDS
analyzer.availability_rule = development_availability_rule


if __name__ == "__main__":
    raise SystemExit(analyzer.main())
