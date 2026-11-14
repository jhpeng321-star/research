"""AMENDMENT 50 (registered design; pre-outcome): exact replication of the AMENDMENT 42 chain (identity, verification,
UNC admission, confirmation, analysis) with a DIFFERENT candidate source - the full-size holds-only WM strategy search
(STRATEGY_SEARCH_D2800_p60.json, not the A42 file) - and FRESH audited seeds. Exploratory, never pooled with A42 / A45.

This driver IS ops/a42_bigp60.py: a private copy of that module (loaded from the same file, so every function, the
selection rule, the A37 executor arms, the UNC admission rule, the contrasts and the bootstrap are byte-identical) with
only the amendment-specific names replaced (settings, seed blocks via the settings, ledger phase, job ids / wave names,
bank / receipt file names, host base, stop file, candidates file + pin). The A42 module object and its outputs are not
touched. The A45 part (diverse candidates, DIVERSE_BANK) is ops/a50_divp60.py.

Seeds (ops/a50/SEED_AUDIT.json): verification 992800-992823 (992824-992899 reserved unused, setting a50v_D2800_p60),
confirmation 996800-996895 (996896-996899 reserved unused, setting a50c_D2800_p60); runner blocks A50_RANGES.

Budget: ONE ledger phase ``bigp60_a50`` capped at 720,000 sim-s (identity 4,800 + verification 7 x 24 x 1,200 =
201,600 + confirmation 4 x 96 x 1,200 = 460,800 + retry reserve 52,800), own ledger <final_root>/CHARGES.jsonl under
the host base /root/a50v (A50_HOST_BASE); never the A42 root /root/a42v. Job ids a50v_* / a50c_* / a50i_*, waves
A50V_* / A50C_* / A50_IDENT_U0. Stop file <root>/STOP_A50.

Stages / commands (python ops/a50_fullp60.py <cmd> --final-root <root> --setting <setting>): as ops/a42_bigp60.py
  select (--search <full-size holds-only STRATEGY_SEARCH_D2800_p60.json> --out ops/a50/CANDIDATES_D2800_p60.json),
  prepare, dry, identity, identity-compare, verify, admit, confirm, reports, analyse, status, budget;
  check-search (--search): the pinned A50 candidates document equals a fresh selection of that file (top 6 + diverse).
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import nm_finaleval as N  # noqa: E402

#: pinned when MAIN registers the A50 candidates (before any A50 SUMO); None = not registered -> verify refuses
CANDIDATES_DOC_SHA256 = '3b63698634a0ac7fc3217a1e73580c0f766211f001b0600fadd0fc48359d4ee5'

CELL = 'D2800_p60'
AMENDMENT = '50'
A50_DIR = HERE/'a50'
SOURCE = 'full-size holds-only WM strategy search STRATEGY_SEARCH_D2800_p60.json (not the A42 search file)'
AUTHORITY = ('MAIN AMENDMENT 50 (registered): exact replication of the A42 + A45 chain with the full-size holds-only '
             'WM search as the candidate source and fresh audited seeds (verification 992800-992823, confirmation '
             '996800-996895); phases bigp60_a50 cap 720,000 and divp60_a50 cap 480,000')
OVERRIDES = dict(
    __doc__=__doc__,
    A42_PHASE='bigp60_a50', A42_CAP_SIM_S=720_000, STOP_A42='STOP_A50',
    HOST_BASE=os.environ.get('A50_HOST_BASE', '/root/a50v'),
    VERIFY_SETTING='a50v_D2800_p60', CONFIRM_SETTING='a50c_D2800_p60',
    BANK_DIR='a50_banks', A42_DIR=A50_DIR, CANDS_FILE=A50_DIR/f'CANDIDATES_{CELL}.json',
    IDENT_WAVE='A50_IDENT_U0',
    AMENDMENT=AMENDMENT, LABEL='A50', JOB_PREFIX='a50',
    VERIFY_TAG='a50v', CONFIRM_TAG='a50c', IDENT_TAG='a50i',
    VERIFY_WAVE_PREFIX='A50V_', CONFIRM_WAVE_PREFIX='A50C_',
    CANDS_BANK_FILE=f'A50_CANDS_{CELL}.json', BIG_BANK_FILE=f'A50_BIG_BANK_{CELL}.json',
    ADMISSION_FILE=f'ADMISSION_A50_{CELL}.json', RESULTS_FILE=f'A50_RESULTS_{CELL}.json', LOG_FILE='A50_LOG.jsonl',
    SCHEMA_PREFIX='coopmwm.a50', RUNNER_RANGES_ATTR='A50_RANGES',
    RESULT_LABEL='amendment 50; exploratory row at 60% CAVs; never pooled; one shot',
    AUTHORITY=AUTHORITY,
    REPLAY_AUTHORITY=('MAIN AMENDMENT 50 registered rule (= A42): one charged retry per failed episode on the same '
                      'world; no outcome-based retries'),
)


def _clone(name, path, overrides):
    """A private module object executed from ``path`` (same code), then the amendment names replaced."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    for k, v in overrides.items():
        if k != '__doc__' and not hasattr(m, k):
            raise AttributeError(f'{path.name} has no {k} (the A50 override list is out of date)')
        setattr(m, k, v)
    return m


def _build():
    m = _clone('a50_bigp60_core', HERE/'a42_bigp60.py', OVERRIDES)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (m.VERIFY_SETTING, m.CONFIRM_SETTING)
    m.A42Runtime.phase = m.A42_PHASE
    m.CANDIDATES_DOC_SHA256 = CANDIDATES_DOC_SHA256

    def select(search_path, out):
        """The registered A42 selection (tools.coopmwm.a42_bigbank.select: top 6 + the A45 diverse list, cap 12),
        unchanged, on the A50 source; the document is stamped amendment 50 (+ the source note) before sealing."""
        from tools.coopmwm import a42_bigbank as B
        search_path = Path(search_path)
        doc = B.select(json.loads(search_path.read_text(encoding='utf-8')), N.file_sha(search_path))
        doc = B.seal(dict(doc, amendment=AMENDMENT,
                          a50=dict(source=SOURCE, search_name=search_path.name,
                                   rule='A42 top 6 + A45 diverse list (cap 12): a42_bigbank.select unchanged')))
        out = Path(out)
        if out.exists():
            raise N.Blocked(f'{out}: exists (pre-outcome file; never overwrite)')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
        return doc
    m.select = select
    return m


D = _build()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['check-search']:                  # the A45-part check, also offered here (same function)
        import a50_divp60 as E
        return E.E.main(argv)
    return D.main(argv)


if __name__ == '__main__':
    main()
