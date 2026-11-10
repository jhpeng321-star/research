"""CPU-only tests for the amendment-57 driver (no simulator): names / budget / cap, the interaction estimator and the
three-way verdict rule (registered before any outcome)."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a42_bigp60 as D42  # noqa: E402
import a57_coopsyn as A57  # noqa: E402

D = A57.D


def test_names_budget_and_reused_code():
    assert (D.A42_PHASE, D.A42_CAP_SIM_S, D.STOP_A42, D.JOB_PREFIX, D.A42Runtime.phase) == \
        ('coopsyn_a57', 390_000, 'STOP_A57', 'a57', 'coopsyn_a57')
    assert (D42.A42_PHASE, D42.JOB_PREFIX) == ('bigp60_a42', 'a42')                 # the A42 module untouched
    assert D.BIG_BANK_FILE == 'A42_BIG_BANK_D2800_p60.json' and A57.SETTING == D.CONFIRM_SETTING
    b = A57.budget_plan()
    assert b['planned'] == 1_200+3*96*1_200 == 346_800 and b['margin'] == 43_200
    assert A57.PART_ARMS == ('PART_L0', 'PART_L1', 'PART_L2') and A57.BOOTSTRAP['seed'] == 20261104


def test_interaction_is_joint_minus_sum_of_parts():
    u0 = np.array([100., 200.])
    joint = np.array([70., 150.])
    parts = [np.array([90., 190.]), np.array([95., 200.]), np.array([100., 180.])]
    # (J - U0) - sum(P - U0) = (-30 - (-10-5-0)) = -15 ; (-50 - (-10+0-20)) = -20
    assert A57.interaction(joint, parts, u0).tolist() == [-15., -20.]


def test_verdict_rule():
    assert A57.verdict_of([-5., -1.]) == 'SUPPORTED (super-additive)'
    assert A57.verdict_of([1., 5.]) == 'SUB-ADDITIVE'
    assert A57.verdict_of([-1., 1.]) == 'NOT RESOLVED'
    assert A57.verdict_of([-1., 0.]) == 'NOT RESOLVED'
