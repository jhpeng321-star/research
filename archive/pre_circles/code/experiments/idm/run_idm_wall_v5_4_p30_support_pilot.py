"""Run the frozen prospective v5.4 p30 support amendment."""

from experiments import REPO_ROOT
from experiments.idm import run_idm_wall_v5_4_support_pilot as runner
from experiments.idm.wall_v5_4_p30_support_contract import (
    CAV_PENETRATION,
    SUPPORT_SEEDS,
    load_feasibility_scan,
    support_manifest,
)


runner.CAV_PENETRATION = CAV_PENETRATION
runner.SUPPORT_SEEDS = SUPPORT_SEEDS
runner.load_feasibility_scan = load_feasibility_scan
runner.support_manifest = support_manifest
runner.DEFAULT_SCAN = REPO_ROOT / "runs" / "probe" / "idm_wall_v5_4_feasibility_p30_s4333_4352"
runner.DEFAULT_OUTPUT = REPO_ROOT / "runs" / "eval" / "idm_wall_v5_4_p30_support_pilot_s4333_4352"
runner.ADDITIONAL_ALLOWED_ROOTS = (
    REPO_ROOT / "runs" / "probe" / "idm_wall_v5_4_feasibility_p25_s4333_4352",
)


if __name__ == "__main__":
    raise SystemExit(runner.main())
