"""Analyze the frozen treatment-blind fixed-p30 v5.4 feasibility amendment."""

from experiments.idm.wall_v5_4_p30_support_contract import (
    CAV_PENETRATION,
    FEASIBILITY_ANALYSIS_SCHEMA,
    SUPPORT_SEEDS,
    availability_rule,
)
from tools import analyze_idm_wall_v5_4_feasibility as analyzer


analyzer.CAV_PENETRATION = CAV_PENETRATION
analyzer.FEASIBILITY_ANALYSIS_SCHEMA = FEASIBILITY_ANALYSIS_SCHEMA
analyzer.SUPPORT_SEEDS = SUPPORT_SEEDS
analyzer.availability_rule = availability_rule


if __name__ == "__main__":
    raise SystemExit(analyzer.main())
