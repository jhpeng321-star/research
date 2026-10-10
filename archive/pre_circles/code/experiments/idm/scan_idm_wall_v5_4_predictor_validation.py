"""Generate the treatment-blind p30 predictor-validation feasibility partition."""

from experiments import REPO_ROOT
from experiments.idm import scan_idm_wall_v5_4_feasibility as runner
from experiments.idm.wall_v5_4_predictor_contract import (
    CAV_PENETRATION,
    VALIDATION_SEEDS,
    predictor_manifest,
    require_support_authorization,
    validate_role_seeds,
    validation_availability_rule,
)


runner.CAV_PENETRATION = CAV_PENETRATION
runner.SUPPORT_SEEDS = VALIDATION_SEEDS
runner.availability_rule = validation_availability_rule
runner.support_manifest = lambda: predictor_manifest("validation", VALIDATION_SEEDS)
runner.validate_support_seeds = lambda seeds: validate_role_seeds("validation", seeds)
runner.DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs"
    / "probe"
    / "idm_wall_v5_4_predictor_validation_feasibility_p30_s4601_4660"
)


if __name__ == "__main__":
    require_support_authorization(REPO_ROOT)
    runner.main()
