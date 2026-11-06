"""CPU tests for the clean4 pre-launch potential gate (pure verdict, zero physics)."""
import pytest

from tools.coopmwm import clean4_potential as g

SRC = [dict(path='NEWMAP2_REPORT.md', sha256='a'*64)]
CELL = dict(demand=2800, penetration=0.4)


def _ex(n, reach):
    return [dict(id=f'e{i}', reachable=i < reach, robust=True) for i in range(n)]


def test_pass_at_thresholds():
    r = g.verdict(cell=CELL, screened_moments=20, robust_moments=3, b_examples=_ex(8, 4), sources=SRC, range_m=200)
    assert r['decision'] == 'PASS'
    assert g.require_pass(r) is r


@pytest.mark.parametrize('screened,robust,n,reach', [(20, 2, 8, 8), (20, 10, 7, 7), (20, 10, 8, 3), (0, 0, 8, 8)])
def test_fail_each_condition(screened, robust, n, reach):
    r = g.verdict(cell=CELL, screened_moments=screened, robust_moments=robust, b_examples=_ex(n, reach),
                  sources=SRC, range_m=200)
    assert r['decision'] == 'FAIL'
    with pytest.raises(ValueError):
        g.require_pass(r)
    text = g.fail_markdown(r)
    assert 'CLEAN4_GATE_FAIL' in text and r['receipt_sha256'] in text


def test_input_validation_and_tamper():
    with pytest.raises(ValueError):
        g.verdict(cell=CELL, screened_moments=2, robust_moments=3, b_examples=_ex(8, 8), sources=SRC, range_m=200)
    with pytest.raises(ValueError):
        g.verdict(cell=CELL, screened_moments=20, robust_moments=3, b_examples=_ex(8, 8), sources=[], range_m=200)
    with pytest.raises(ValueError):
        g.verdict(cell=CELL, screened_moments=20, robust_moments=3,
                  b_examples=[dict(id='a', reachable=True)]*2, sources=SRC, range_m=200)
    r = g.verdict(cell=CELL, screened_moments=20, robust_moments=3, b_examples=_ex(8, 8), sources=SRC, range_m=200)
    with pytest.raises(ValueError):
        g.require_pass(dict(r, decision='PASS', checks=dict(r['checks'], b_examples=dict(passed=True))))
