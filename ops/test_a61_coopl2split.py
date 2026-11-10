"""CPU-only tests for the amendment-61 driver (no simulator): names / budget / cap, I2 and the verdict rule."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a57_coopsyn as A57  # noqa: E402
import a61_coopl2split as A61  # noqa: E402

D = A61.D


def test_names_budget():
    assert (D.A42_PHASE, D.A42_CAP_SIM_S, D.STOP_A42, D.JOB_PREFIX, D.A42Runtime.phase) == \
        ('coopl2split_a61', 245_000, 'STOP_A61', 'a61', 'coopl2split_a61')
    assert (A57.D.A42_PHASE, A57.D.JOB_PREFIX) == ('coopsyn_a57', 'a57')            # A57 view untouched
    b = A61.budget_plan()
    assert b['planned'] == 1_200+2*96*1_200 == 231_600 and b['margin'] == 13_400
    assert A61.PICK_ARMS == ('PART_L2A', 'PART_L2B') and A61.LANE == 2


def test_i2_and_verdict():
    u0 = np.array([100., 200.])
    l2 = np.array([60., 150.])
    a, b = np.array([80., 190.]), np.array([90., 180.])
    # (l2-u0) - (a-u0) - (b-u0) = -40 - (-20) - (-10) = -10 ; -50 - (-10) - (-20) = -20
    assert A61.i2(l2, a, b, u0).tolist() == [-10., -20.]
    assert A61.verdict_of([-3., -1.]) == 'WITHIN-LANE SYNERGY'
    assert A61.verdict_of([1., 3.]) == 'SUB-ADDITIVE'
    assert A61.verdict_of([-1., 1.]) == 'NOT RESOLVED'
