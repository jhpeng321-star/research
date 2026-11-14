"""AMENDMENT 50, diverse part (registered design; pre-outcome): exact replication of the AMENDMENT 45 chain on the A50
final root - the diverse candidates of the pinned A50 candidates document (a42_bigbank.select_diverse, cap 12, same
rule), verified with the same A37 UNC rule on the SAME 24 A50 verification worlds (U0 = the A50 wave A50V_U0, never
rerun), DIVERSE_BANK = every admitted plan of the A50 top 6 + the admitted diverse plans, one extra arm on the SAME 96
A50 confirmation worlds. Exploratory, never pooled.

This driver IS ops/a45_divp60.py: a private copy of that module (same file, same functions, same admission, contrasts
and bootstrap) bound to the A50 big-bank driver (ops/a50_fullp60.py) instead of the A42 one, with only the
amendment-specific names replaced: ledger phase ``divp60_a50`` cap 480,000 (separate from bigp60_a50; own ledger total
1,200,000), job ids a50dv_* / a50dc_*, waves A50DV_* / A50DC_DIVERSE_BANK, files ADMISSION_A50DIV_D2800_p60.json /
a50_banks/A50DIV_CANDS_D2800_p60.json / a50_banks/A50_DIVERSE_BANK_D2800_p60.json, stop file <root>/STOP_A50DIV.
As in A42 + A45: the big-bank runtime counts every row of the shared ledger against its 720,000 cap, so the diverse part
spends only after every A50 big-bank stage has finished (run_a50.sh order).

Stages (python ops/a50_divp60.py <cmd> --final-root <root> --setting <setting>): check-search, prepare, dry, verify,
admit, confirm, reports, analyse, status, budget (as ops/a45_divp60.py).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import a50_fullp60 as F  # noqa: E402

D = F.D
CELL = F.CELL
A50DIV_PHASE, A50DIV_CAP_SIM_S = 'divp60_a50', 480_000
OVERRIDES = dict(
    __doc__=__doc__,
    D=D, A45_PHASE=A50DIV_PHASE, A45_CAP_SIM_S=A50DIV_CAP_SIM_S,
    TOTAL_ALLOCATION_SIM_S=D.A42_CAP_SIM_S+A50DIV_CAP_SIM_S,
    STOP_A45='STOP_A50DIV',
    AMENDMENT=F.AMENDMENT, LABEL='A50DIV', JOB_PREFIX='a50d',
    VERIFY_TAG='a50dv', CONFIRM_TAG='a50dc', VERIFY_WAVE_PREFIX='A50DV_', CONFIRM_WAVE='A50DC_DIVERSE_BANK',
    ADMISSION_FILE=f'ADMISSION_A50DIV_{CELL}.json', CANDS_BANK=f'A50DIV_CANDS_{CELL}.json',
    DIVERSE_BANK_FILE=f'A50_DIVERSE_BANK_{CELL}.json', RESULTS_FILE=f'A50DIV_RESULTS_{CELL}.json',
    LOG_FILE='A50DIV_LOG.jsonl', SCHEMA_PREFIX='coopmwm.a50div',
    RESULT_LABEL='amendment 50 (diverse part); exploratory row at 60% CAVs; never pooled; one shot',
    AUTHORITY=('MAIN AMENDMENT 50 (registered): diverse structural-class candidates of the A50 candidates document '
               '(A45 rule), same 24 A50 verification seeds and UNC rule, DIVERSE_BANK = admitted A50 top 6 + diverse, '
               'one extra arm on the same 96 A50 confirmation worlds; phase divp60_a50 cap 480,000'),
)


def _build():
    m = sys.modules.get('a50_divp60_core')
    if m is not None:
        return m
    spec = importlib.util.spec_from_file_location('a50_divp60_core', HERE/'a45_divp60.py')
    m = importlib.util.module_from_spec(spec)
    sys.modules['a50_divp60_core'] = m
    spec.loader.exec_module(m)
    for k, v in OVERRIDES.items():
        if k != '__doc__' and not hasattr(m, k):
            raise AttributeError(f'a45_divp60.py has no {k} (the A50 override list is out of date)')
        setattr(m, k, v)
    m.A45Runtime.phase = m.A45_PHASE
    m.budget_plan()                                    # the arithmetic still holds under the A50 names
    return m


E = _build()


def main(argv=None):
    return E.main(argv)


if __name__ == '__main__':
    main()
