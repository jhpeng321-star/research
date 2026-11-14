"""Run the frozen treatment-blind fixed-p30 v5.4 feasibility amendment."""

from experiments import REPO_ROOT
from experiments.idm import scan_idm_wall_v5_4_feasibility as runner
from experiments.idm.wall_v5_4_p30_support_contract import (
    CAV_PENETRATION,
    FEASIBILITY_ANALYSIS_SCHEMA,
    SUPPORT_SEEDS,
    availability_rule,
    support_manifest,
)


runner.CAV_PENETRATION = CAV_PENETRATION
runner.SUPPORT_SEEDS = SUPPORT_SEEDS
runner.availability_rule = availability_rule
runner.support_manifest = support_manifest
runner.DEFAULT_OUTPUT = REPO_ROOT / "runs" / "probe" / "idm_wall_v5_4_feasibility_p30_s4333_4352"
runner.ADDITIONAL_ALLOWED_ROOTS = (
    REPO_ROOT / "runs" / "probe" / "idm_wall_v5_4_feasibility_p25_s4333_4352",
)


if __name__ == "__main__":
    runner.main()
