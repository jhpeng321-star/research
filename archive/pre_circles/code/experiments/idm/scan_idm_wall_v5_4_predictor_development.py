"""Generate the treatment-blind p30 predictor-development feasibility partition."""

from experiments import REPO_ROOT
from experiments.idm import scan_idm_wall_v5_4_feasibility as runner
from experiments.idm.wall_v5_4_predictor_contract import (
    CAV_PENETRATION,
    DEVELOPMENT_SEEDS,
    development_availability_rule,
    predictor_manifest,
    require_support_authorization,
    validate_role_seeds,
)


runner.CAV_PENETRATION = CAV_PENETRATION
runner.SUPPORT_SEEDS = DEVELOPMENT_SEEDS
runner.availability_rule = development_availability_rule
runner.support_manifest = lambda: predictor_manifest("development", DEVELOPMENT_SEEDS)
runner.validate_support_seeds = lambda seeds: validate_role_seeds("development", seeds)
runner.DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs"
    / "probe"
    / "idm_wall_v5_4_predictor_development_feasibility_p30_s4401_4520"
)


if __name__ == "__main__":
    require_support_authorization(REPO_ROOT)
    runner.main()
