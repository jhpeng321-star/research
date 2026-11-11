"""CPU-only tests for the amendment-62 driver (no simulator): names / budget, Holm and the per-plan verdict rule."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a57_coopsyn as A57  # noqa: E402
import a62_coopperplan as A62  # noqa: E402

D = A62.D


def test_names_budget():
    assert (D.A42_PHASE, D.A42_CAP_SIM_S, D.STOP_A42, D.JOB_PREFIX, D.A42Runtime.phase) == \
        ('coopperplan_a62', 360_000, 'STOP_A62', 'a62', 'coopperplan_a62')
    assert (A57.D.A42_PHASE, A57.D42.A42_PHASE) == ('coopsyn_a57', 'bigp60_a42')
    b = A62.budget_plan()
    assert b['planned'] == 1_200+12*24*1_200 == 346_800 and b['margin'] == 13_200
    assert A62.SETTING == D.VERIFY_SETTING == 'a42v_D2800_p60'


def test_holm():
    adj = A62.holm([0.01, 0.04, 0.03, 0.5])
    assert adj == [0.04, 0.09, 0.09, 0.5]                # 4*.01, max(3*.03, .04), 2*.04 -> .09 monotone, .5
    assert A62.holm([0.2, 0.2]) == [0.4, 0.4]


def test_verdict():
    assert A62.verdict_of(-5.0, 0.01) == 'CROSS-LANE MEMBERS HELP'
    assert A62.verdict_of(5.0, 0.049) == 'CROSS-LANE MEMBERS HURT'
    assert A62.verdict_of(-5.0, 0.05) == 'NOT RESOLVED'
