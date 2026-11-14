"""AMENDMENT 82 part A (A82-S3B20; POST-HOC, exploratory): SUMO admission of the A79 S3b reduced WM hold-form search
plans at 20 % CAVs (file TOP_OFFER_CONDITIONS_D2800_p20.json written by a75ss.main; <= 12 rows), the 20 % analogue of
A80 (60 %). DRAFT - not registered; nothing here has run.

Everything that decides is imported, unchanged:
  * MAPPING_RULE / FORM_RULE / form tokens / receipt invariants (plan_checks) / mech_verdict: ops/a80_s2b.py
  * plan specs on the comparator, the comparator binding, the admission computation and BANK_RULE:
    ops/a76_d2p20.py ``admit`` (A76 ADMIT_RULE / BANK_RULE; a75_d2forms.bank_choice / bank_plans) - the SAME function
    A76 used, with its module names re-pointed at this root (the A80 pattern for A75-D2)
  * worlds: the 24 A76 admission worlds 997200-997223 (A76_PANEL_V.json, byte copy, panel sha checked against the
    sealed A76_PINS.json of the A76 root)
  * reference: the STORED A76 comparator wave A76V_COMP (U0 d9042cc5, run by A76 on those 24 worlds) read from the A76
    root, NOT re-run; its wave receipt(s) and the 24 per-vehicle reports (DROPPED_RAW-sealed) are pinned here at ``pin``
    (zero SUMO) and re-checked at admission. Nothing is written into the A76 root (a reference episode whose raw vehicle
    log is still present refuses, because the report reader would cache next to it).
Deviations from A76 / A80 (fixed before any outcome): form families = the A80 FORM_RULE (seven, incl. "other": WM plans
can combine forms; A76 drew one form per plan); mechanics = the first 2 verification episodes of EVERY candidate
(997200 / 997201; same specs / job ids, counted once inside the verification cap) checked by the A80 receipt invariants
+ U0 executor + a75 digest + zone (+-1 m) + 0 collisions / teleports (A80 ran only plans with tokens outside the 60 %
fixture; at 20 % the A76 fixture left forms NOT EXERCISED, so every plan is checked; costless). No confirmation stage
here: the bank (if any) goes to the A82 four-arm confirmation (ops/a82_fourarm.py).

Stages (own root; ledger phase ``holdforms_a82s3b20``; cap in code: verification 12 x 24 x 1,200 = 345,600 sim-s):
  check-top   zero SUMO, no root: MAPPING_RULE on a file
  pin-top     zero SUMO, write-once: byte copy of the top file (A82_TOP_FILE + A82_TOP_SHA256) + A82_CANDIDATES.json
  pin         zero SUMO, write-once: A76_PANEL_V.json byte copy + A82_PINS.json (panel, comparator, stored reference)
  prepare     zero SUMO: candidate bank (a75_holdforms.bank_doc, cell D2800_p20) + A82_COMPARATOR.json
  dry         zero SUMO: waves / sim-s / reference binding check
  mech        first 2 verification episodes of every candidate -> mech-compare (A82_MECH.json PASS / FAIL)
  verify      the rest of the verification episodes (refuses unless mech PASS); --drop-raw (A75-D2 DROP_RULE)
  admit       zero physics: receipt invariants on every record, then a76_d2p20.admit (re-pointed) -> A82_ADMISSION +
              bank file a82_banks/A82_BANK_D2800_p20.json (none if nothing admitted)
Commands: python ops/a82_s3b20.py <cmd> --final-root R [--a76-root /root/a76p20/root]
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import a76_d2p20 as M76  # noqa: E402  (A76 driver, unchanged; module names re-pointed by _point)
import a80_s2b as M80  # noqa: E402  (A80 driver, unchanged; pure rules imported)
import a75_d2forms as D2  # noqa: E402  (A75-D2 driver, unchanged)

AMENDMENT = '82'
STAGE = 'S3B20'
CELL = M76.CELL                                          # D2800_p20
PHASE = 'holdforms_a82s3b20'
HORIZON = M76.HORIZON
A75_SHA256 = M76.A75_SHA256                              # a75_holdforms d1af7e0f
FROZEN_U0_SHA256 = M76.FROZEN_U0_SHA256                  # d9042cc5
N_MAX, VERIFY_N, MECH_WORLDS = M80.N_MAX, M76.VERIFY_N, 2
VERIFY_CAP = N_MAX*VERIFY_N*HORIZON
CAP_SIM_S = VERIFY_CAP
assert (N_MAX, VERIFY_N, VERIFY_CAP) == (12, 24, 345_600)
#: the A76 admission panel (registered A76 line: admission 997200-997223, panel sha 02e20925...)
A76_V_SEEDS = tuple(range(997200, 997224))
A76_V_PANEL_PREFIX = '02e20925'
TOP_ENV, TOP_SHA_ENV = 'A82_TOP_FILE', 'A82_TOP_SHA256'
A76_ROOT_ENV = 'A82_A76_ROOT'
HOST_BASE_ENV = 'A82_HOST_BASE'
TOP_NAME = f'TOP_OFFER_CONDITIONS_{CELL}.json'
TOP_COPY = f'A82_{TOP_NAME}'
CANDS_DOC = 'A82_CANDIDATES.json'
REFUSED_DOC = 'A82_MAPPING_REFUSED.json'
PINS_FILE = 'A82_PINS.json'
COMP_FILE = 'A82_COMPARATOR.json'
BANK_DIR = 'a82_banks'
CANDS_BANK = f'A82_CANDS_{CELL}.json'
BANK = f'A82_BANK_{CELL}.json'
MECH_FILE = 'A82_MECH.json'
ADMISSION_FILE = f'A82_ADMISSION_{CELL}.json'
EXTRA_FILE = 'A82_ADMISSION_EXTRA.json'
LOG_FILE = 'A82_LOG.jsonl'
STOP_FILE = 'STOP_A82'
DROP_MANIFEST = 'A82_DROPPED_RAW.jsonl'
PREFIX = dict(i='a82i_unused_', m='a82m_unused_', v='a82v_', c='a82c_unused_')
REF_WAVE = M76.WAVE['vref']                             # A76V_COMP (stored, read-only)
REF_TAG = M76.PREFIX['v']+'comp'                        # a76v_comp (the A76 job ids)
ARM = 'A82_BANK'
WRITER = 'a75ss.main TOP_OFFER_CONDITIONS (A79 S3b, D2800_p20)'
AUTHORITY = ('MAIN AMENDMENT 82 part A (A82-S3B20, DRAFT 2026-11-13; POST-HOC, exploratory): the A79 S3b WM search top '
             'plans (TOP_OFFER_CONDITIONS_D2800_p20, pinned file) each verified in SUMO on the 24 A76 admission worlds '
             'vs the stored A76 U0 wave; A76 admission / bank rules; phase holdforms_a82s3b20, cap 345,600')
DISCLOSURE = ('POST-HOC and exploratory. The 24 A76 admission worlds were already used by A76 (36 plans); the A76 '
              'admission and confirmation outcomes were known when this was drafted. The plans were chosen by the WM on '
              'WM start states (A79 S3b), not on these worlds. Multiplicity: up to 12 plans each tested at 95% on the '
              'same 24 worlds; admission is a screen and its bounds are selection-biased. Never pooled with other '
              'panels; the claim-bearing test is the A82 four-arm confirmation on fresh worlds')
ADMIT_RULE = M76.ADMIT_RULE.replace(
    'the comparator runs on the same worlds in this root (A76V_COMP)',
    'the comparator = the STORED A76V_COMP wave of the A76 root (U0 d9042cc5 on the same 24 worlds; not re-run; wave '
    'receipts and per-vehicle reports pinned)')
BANK_RULE = M76.BANK_RULE.replace('gap / short / zone / pulse / taper / rolling',
                                  '/'.join(M80.FORMS) + ' (A80 FORM_RULE)')
assert ADMIT_RULE != M76.ADMIT_RULE and BANK_RULE != M76.BANK_RULE
MECH_RULE = (
    'every candidate runs its first 2 verification episodes (997200 / 997201; same specs / job ids as its verification '
    'wave). PASS iff every such record loaded the U0 executor d9042cc5 and a75_sha256 d1af7e0f with a75 receipts, no '
    'executor refusal, collisions = 0 and teleports = 0, no receipt invariant violation (a80_s2b.plan_checks) and every '
    'activation x of every member (vehicle log s + S_OFF) inside the plan zone (+-1 m). Tokens never exercised on those '
    '2 worlds are reported NOT EXERCISED (20 %: groups may not form), not failed; the receipt invariants are re-checked '
    'on every verification record before admission')


# ================================================================================================ helpers

def _N():
    return D2._mods()[0]


def _H():
    return D2._H()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def a76_root(env=None):
    return Path((os.environ if env is None else env).get(A76_ROOT_ENV) or '/root/a76p20/root')


def host_base(env=None):
    return (os.environ if env is None else env).get(HOST_BASE_ENV, '/root/a82p20')


def top_env(env=None):
    env = os.environ if env is None else env
    p, s = env.get(TOP_ENV), env.get(TOP_SHA_ENV)
    if not p or not s:
        raise SystemExit(f'{TOP_ENV} and {TOP_SHA_ENV} are both required (the registered file and its sha256)')
    return Path(p), M80.check_sha(s)


def candidates_doc(top_bytes, top_sha):
    """Pure: MAPPING_RULE (a80_s2b.map_rows, cell D2800_p20) -> sealed candidate document; raises on any refusal."""
    _point_rules()
    rows = json.loads(top_bytes.decode('utf-8'))
    cands, bad = M80.map_rows(rows)
    if bad:
        raise SystemExit(json.dumps(dict(MAPPING_REFUSED=bad), default=str))
    cov = M80.covered_tokens()
    for c in cands:
        c['uncovered_by_a75d2_fixture'] = sorted(list(t) for t in M80.form_tokens(c['strategy']) - cov)
        c['mech'] = True
    return D2.seal_doc(dict(
        schema='coopmwm.a82.candidates.v1', amendment=AMENDMENT, stage=STAGE, cell=CELL, writer=WRITER,
        source=dict(file=TOP_NAME, sha256=top_sha, rows=len(rows)), mapping_rule=M80.MAPPING_RULE,
        form_rule=M80.FORM_RULE, mech_rule=MECH_RULE,
        executor=dict(module='tools/coopmwm/a75_holdforms.py', a75_sha256=A75_SHA256), n_candidates=len(cands),
        candidates_by_form=dict(sorted(Counter(c['form'] for c in cands).items())),
        n_extended=sum(c['extended'] for c in cands), mech_plans=[c['family'] for c in cands],
        world_model='plans and their order come from the WM search file; no WM number is used by any SUMO rule',
        STOP=not cands, candidates=cands))


def cap_check(used_stage, used_total, need):
    if used_stage+need > VERIFY_CAP or used_total+need > CAP_SIM_S:
        raise SystemExit(f'verify: {used_stage} + {need} > stage cap {VERIFY_CAP} (or phase cap {CAP_SIM_S})')
    return True


def ref_report_ok(job_dir_files, drop_receipt, report_sha, pinned_sha):
    """Pure: a stored reference episode is usable read-only iff its raw vehicle log is gone (dropped under the sealed
    DROPPED_RAW receipt), the report file is the one sealed in that receipt, and (once pinned) equals the pin."""
    errs = []
    if 'vehicles.jsonl.gz' in job_dir_files:
        errs.append('raw vehicle log present (the report reader would write a cache into the A76 root)')
    if drop_receipt is None:
        errs.append('no DROPPED_RAW.json receipt')
    elif drop_receipt.get('report', {}).get('sha256') != report_sha:
        errs.append('vehicles_report.json is not the report sealed in DROPPED_RAW.json')
    if pinned_sha is not None and pinned_sha != report_sha:
        errs.append('vehicles_report.json differs from the A82 pin')
    return errs


# ================================================================================================ re-pointing

def _point_rules():
    """Process-local: the A80 mapping round trip at the 20 % cell; the A75-D2 bank rule over the A80 form families."""
    M80.CELL = CELL
    D2.FORMS = M80.FORMS


class _AProxy:
    """a32_ablation with report_of routed: stored A76 reference episodes are read from the A76 root, read-only."""

    def __init__(self, A, root76, pins):
        self._A, self._root76, self._pins = A, Path(root76), pins

    def __getattr__(self, k):
        return getattr(self._A, k)

    def report_of(self, root, rec):
        if rec['job_id'].startswith(REF_TAG+'_'):
            ref_episode_check(self._root76, rec, (self._pins or {}).get('ref', {}).get('reports'))
            return self._A.report_of(self._root76, rec)
        return self._A.report_of(root, rec)


_STATE = dict(root76=None, pins=None)
_ORIG = dict(load=M76._load, mods=M76._mods)


def _mods_proxy():
    N, A, D = _ORIG['mods']()
    return N, _AProxy(A, _STATE['root76'], _STATE['pins']), D


def ref_episode_check(root76, rec, pinned):
    N = _N()
    d = Path(root76)/'evaluation'/rec.get('work_dir', rec['job_id'])
    if d.is_symlink() or not d.is_dir():
        raise N.Blocked(f'{d}: stored reference episode dir missing / symlink')
    files = {p.name for p in d.iterdir()}
    drop = N.verify(N.read(d/'DROPPED_RAW.json'), 'receipt_sha256', 'DROPPED_RAW.json') \
        if 'DROPPED_RAW.json' in files else None
    rep = N.file_sha(d/'vehicles_report.json') if 'vehicles_report.json' in files else None
    job = rec.get('work_dir', rec['job_id'])
    errs = ref_report_ok(files, drop, rep, None if pinned is None else pinned.get(job))
    if pinned is not None and job not in pinned:
        errs.append('not pinned')
    if errs:
        raise N.Blocked(f'{job}: {errs}')
    return rep


def _load(root, wave, specs):
    """a76_d2p20._load, except the reference wave: read from the A76 root with the pinned wave receipts."""
    N = _N()
    if wave != REF_WAVE:
        return _ORIG['load'](root, wave, specs)
    have, src = _ORIG['load'](_STATE['root76'], wave, specs)
    got = src['receipt_sha256'] if isinstance(src['receipt_sha256'], list) else [src['receipt_sha256']]
    want = ((_STATE['pins'] or {}).get('ref') or {}).get('wave_receipts')
    if want is not None and got != want:
        raise N.Blocked(f'{wave}: wave receipts {got} are not the pinned {want}')
    return have, src


def _point(root76, pins):
    _point_rules()
    _STATE.update(root76=Path(root76), pins=pins)
    M76.core = core
    M76.HOST_BASE = host_base()
    M76.AMENDMENT, M76.STAGE, M76.PHASE, M76.ARM = AMENDMENT, STAGE, PHASE, ARM
    M76.PINS_FILE, M76.COMP_FILE, M76.BANK_DIR = PINS_FILE, COMP_FILE, BANK_DIR
    M76.CANDS_BANK, M76.BANK, M76.MECH_FILE, M76.ADMISSION_FILE = CANDS_BANK, BANK, MECH_FILE, ADMISSION_FILE
    M76.LOG_FILE, M76.STOP_FILE = LOG_FILE, STOP_FILE
    M76.AUTHORITY, M76.DISCLOSURE, M76.ADMIT_RULE, M76.BANK_RULE = AUTHORITY, DISCLOSURE, ADMIT_RULE, BANK_RULE
    M76.verify_waves = admit_waves
    M76._load = _load
    M76._mods = _mods_proxy


#: every a76_d2p20 name _point replaces (tests check the list against the module)
M76_NAMES = ('core', 'HOST_BASE', 'AMENDMENT', 'STAGE', 'PHASE', 'ARM', 'PINS_FILE', 'COMP_FILE', 'BANK_DIR',
             'CANDS_BANK', 'BANK', 'MECH_FILE', 'ADMISSION_FILE', 'LOG_FILE', 'STOP_FILE', 'AUTHORITY', 'DISCLOSURE',
             'ADMIT_RULE', 'BANK_RULE', 'verify_waves', '_load', '_mods')


def core():
    """The a42 driver code (as A76 uses it) with the A82 names: own phase / cap / prefixes / bank dir / root."""
    N = _N()
    import a50_fullp60 as A50
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=CAP_SIM_S, STOP_A42=STOP_FILE, HOST_BASE=host_base(), CELL=CELL,
        VERIFY_SETTING=M76.SETTING['v'], CONFIRM_SETTING=M76.SETTING['v'], BANK_DIR=BANK_DIR,
        IDENT_WAVE='A82_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A82', JOB_PREFIX='a82', VERIFY_TAG='a82v',
        CONFIRM_TAG='a82c_unused', IDENT_TAG='a82i_unused', VERIFY_WAVE_PREFIX='A82V_', CONFIRM_WAVE_PREFIX='A82C_',
        RESULTS_FILE='A82_RESULTS_UNUSED.json', LOG_FILE=LOG_FILE, SCHEMA_PREFIX='coopmwm.a82',
        RUNNER_RANGES_ATTR='A76_RANGES',
        RESULT_LABEL='amendment 82 part A; POST-HOC exploratory SUMO admission of the A79 S3b WM top plans at 20 % '
                     'CAVs vs the stored A76 U0 wave; one shot', AUTHORITY=AUTHORITY)
    m = A50._clone('a82_s3b20_core', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (M76.SETTING['v'],)
    m.A42Runtime.phase = PHASE
    N.CHECKS['episode'] = D2.check_d2
    return m


# ================================================================================================ pins (zero SUMO)

def pin_top(root, env=None):
    N = _N()
    import a37_celltailor as C37
    root = Path(root)
    path, want = top_env(env)
    raw = path.read_bytes()
    root.mkdir(parents=True, exist_ok=True)
    if sha_bytes(raw) != want:
        raise N.Blocked(f'{path}: sha256 {sha_bytes(raw)} != registered {want}')
    if (root/CANDS_DOC).is_file():
        raise N.Blocked(f'{CANDS_DOC} exists (pin once)')
    try:
        doc = candidates_doc(raw, want)
    except SystemExit as e:
        N.publish(root/REFUSED_DOC, N.seal(dict(schema='coopmwm.a82.mapping_refused.v1', amendment=AMENDMENT,
                                                 source=dict(file=str(path), sha256=want), rule=M80.MAPPING_RULE,
                                                 detail=str(e), consequence='STOP: nothing runs'), 'receipt_sha256'))
        raise
    p = C37.write_once(root/TOP_COPY, raw.decode('utf-8'))
    if N.file_sha(p) != want:
        raise N.Blocked(f'{TOP_COPY}: copy is not byte-identical')
    C37.write_once(root/CANDS_DOC, json.dumps(doc, indent=1, sort_keys=True)+'\n')
    return dict(top_sha256=want, doc_sha256=doc['doc_sha256'], n=doc['n_candidates'],
                by_form=doc['candidates_by_form'], n_extended=doc['n_extended'])


def candidates(root):
    N = _N()
    root = Path(root)
    p, q = root/CANDS_DOC, root/TOP_COPY
    if not p.is_file() or not q.is_file():
        raise N.Blocked(f'{CANDS_DOC} / {TOP_COPY}: missing (run pin-top)')
    doc = json.loads(p.read_text(encoding='utf-8'))
    raw = q.read_bytes()
    if D2.seal_doc(doc)['doc_sha256'] != doc.get('doc_sha256') or doc['source']['sha256'] != sha_bytes(raw):
        raise N.Blocked(f'{CANDS_DOC}: seal / top-file sha mismatch')
    if doc != candidates_doc(raw, doc['source']['sha256']):
        raise N.Blocked(f'{CANDS_DOC}: not the mapping of the pinned file (edited / other code?)')
    if doc['STOP'] or doc['cell'] != CELL or not 1 <= doc['n_candidates'] <= N_MAX:
        raise N.Blocked(f'{CANDS_DOC}: STOP / wrong cell / size')
    return doc['candidates']


def read_pins(root):
    N = _N()
    p = Path(root)/PINS_FILE
    return N.verify(N.read(p), 'receipt_sha256', p.name) if p.is_file() else None


def a76_pins(root76):
    N = _N()
    p = Path(root76)/'A76_PINS.json'
    v = N.verify(N.read(p), 'receipt_sha256', p.name)
    if not v['panels']['v'].startswith(A76_V_PANEL_PREFIX) or v['comparator']['sha256'] != FROZEN_U0_SHA256:
        raise N.Blocked('A76_PINS.json: not the registered A76 admission panel / comparator')
    return v


def make_pins(root, root76):
    """Zero SUMO, write-once: panel byte copy + the stored reference (wave receipts, 24 sealed reports)."""
    N = _N()
    import a37_celltailor as C37
    root, root76 = Path(root), Path(root76)
    if read_pins(root) is not None:
        raise N.Blocked(f'{PINS_FILE} exists (write-once)')
    if root.resolve() == root76.resolve():
        raise N.Blocked('the A82 root is a new root')
    p76 = a76_pins(root76)
    src = root76/M76.PANEL['v']
    if N.file_sha(src) != p76['panel_files']['v']:
        raise N.Blocked(f'{src}: not the panel file sealed in A76_PINS.json')
    q = C37.write_once(root/M76.PANEL['v'], src.read_bytes().decode('utf-8'))
    if N.file_sha(q) != p76['panel_files']['v']:
        raise N.Blocked(f'{q}: not a byte copy')
    _point(root76, None)
    D, panel = M76.setup('v', root, need_panel=False)
    panel = N.load_panel(q, pinned=p76['panels']['v'])
    if tuple(w['world']['seed'] for w in panel['worlds']) != A76_V_SEEDS:
        raise N.Blocked('A76 admission panel: wrong seeds')
    comp = M76.comparator(root)
    if comp['sha256'] != FROZEN_U0_SHA256 or comp['rebind']:
        raise N.Blocked('the comparator of this root must be the frozen U0 (A76 registration)')
    specs = M76.u0_specs(panel, root76, comp, REF_TAG)
    have, src_ = _ORIG['load'](root76, REF_WAVE, specs)
    reports = {}
    for w in [r['world_id'] for r in panel['worlds']]:
        rec = have[w]
        reports[rec.get('work_dir', rec['job_id'])] = ref_episode_check(root76, rec, None)
    rec_sha = src_['receipt_sha256'] if isinstance(src_['receipt_sha256'], list) else [src_['receipt_sha256']]
    value = N.seal(dict(schema='coopmwm.a82.pins.v1', amendment=AMENDMENT, stage=STAGE, cell=CELL,
                        panels=dict(v=panel['panel_sha256']), panel_files=dict(v=N.file_sha(q)),
                        seeds=dict(v=[A76_V_SEEDS[0], A76_V_SEEDS[-1]]), a76_root=str(root76),
                        a76_pins_receipt_sha256=p76['receipt_sha256'], comparator=M76.comp_record(comp),
                        ref=dict(wave=REF_WAVE, wave_receipts=rec_sha, reports=reports, n=len(reports)),
                        a75_sha256=_H().a75_digest(), caps=dict(verify=VERIFY_CAP, total=CAP_SIM_S),
                        note='reference = stored A76V_COMP; outcomes are not read by this step (hashes only)'),
                   'receipt_sha256')
    if value['a75_sha256'] != A75_SHA256 or len(reports) != VERIFY_N:
        raise N.Blocked('a75 digest / reference size')
    N.publish(root/PINS_FILE, value)
    return value


def prepare(root):
    N = _N()
    import a37_celltailor as C37
    root = Path(root)
    comp = M76.comparator(root)
    out = dict(comparator=M76.seal_comparator(root, comp)['receipt_sha256'])
    p = root/BANK_DIR/CANDS_BANK
    C37.write_once(p, C37.bank_text(M76.cands_bank_doc(candidates(root))))
    out[CANDS_BANK] = N.file_sha(p)
    return out


# ================================================================================================ waves

def plan_waves(D, panel, root, comp, cands):
    N = _N()
    waves = [(f'A82V_{c["family"]}', 'episode',
              M76.plan_specs(D, panel, root, CANDS_BANK, 'A75_PLAN', c['family'], f'{PREFIX["v"]}{c["key"]}', comp))
             for c in cands]
    if any(len(s) != VERIFY_N for _, _, s in waves) or len({w for w, _, _ in waves}) != len(waves):
        raise N.Blocked('verification waves: wrong size / duplicate wave')
    return waves


def admit_waves(D, panel, root, comp, cands):
    """What a76_d2p20.admit loads: the stored reference wave (A76 job ids / specs, built on the A76 root) + the plans."""
    return [(REF_WAVE, 'episode', M76.u0_specs(panel, _STATE['root76'], comp, REF_TAG))] + \
        plan_waves(D, panel, root, comp, cands)


def mech_waves(D, panel, root, comp, cands):
    return [(n, k, s[:MECH_WORLDS]) for n, k, s in plan_waves(D, panel, root, comp, cands)]


def mech_compare(D, panel, root, comp, cands):
    N = _N()
    H = _H()
    root = Path(root)
    rows, safety, tokens = {}, [], {c['family']: sorted(list(t) for t in M80.form_tokens(c['strategy'])) for c in cands}
    for (wave, _, specs), c in zip(mech_waves(D, panel, root, comp, cands), cands):
        have, _ = _ORIG['load'](root, wave, specs)
        s = c['strategy']
        out = []
        for w, rec in have.items():
            r = rec['a75']
            errs, ex, st = M80.plan_checks(r, s)
            if rec.get('executor_sha256_loaded') != comp['sha256']:
                errs.append('executor is not the comparator U0')
            if rec.get('a75_sha256_loaded') != A75_SHA256:
                errs.append('a75 digest')
            p = root/'evaluation'/rec.get('work_dir', rec['job_id'])/N.VEHICLE_LOG
            if st['activations']:
                if not p.is_file():
                    errs.append('vehicle log missing (zone check impossible)')
                else:
                    errs += [x for x in D2.check_zone(r, H.zone_bounds(s), D2.vlog_pos(D2.read_vlog(p)))
                             if x != 'no activation']
            safety.append((c['family'], w, rec['safety']))
            out.append(dict(world_id=w, errors=errs, exercised=sorted(list(t) for t in ex), stats=st,
                            holdform_summary=r.get('holdform_summary'), hard_brakes=rec['safety'].get('hard_brakes')))
        rows[c['family']] = out
    verdict, fails, table = M80.mech_verdict(rows, safety, tokens)
    return N.seal(dict(schema='coopmwm.a82.mech.v1', amendment=AMENDMENT, stage=STAGE, rule=MECH_RULE,
                       a75_sha256=A75_SHA256, comparator=M76.comp_record(comp),
                       worlds=[r['world_id'] for r in panel['worlds'][:MECH_WORLDS]], mech_plans=sorted(tokens),
                       verdict=verdict, failures=fails, exercise=table, rows=rows,
                       consequence='PASS: verify / admit may run; FAIL: stop and report'), 'receipt_sha256')


def mech_passed(root):
    N = _N()
    p = Path(root)/MECH_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


def admit(D, panel, root, comp, cands):
    """Receipt invariants on every verification record (any violation: STOP), then a76_d2p20.admit unchanged."""
    N = _N()
    root = Path(root)
    viol, exercised = [], {}
    for (name, _, specs), c in zip(plan_waves(D, panel, root, comp, cands), cands):
        have, _ = _ORIG['load'](root, name, specs)
        ex = set()
        for w, rec in have.items():
            errs, e, _ = M80.plan_checks(rec['a75'], c['strategy'])
            ex |= e
            viol += [f'{c["family"]} {w}: {x}' for x in errs[:3]]
        exercised[c['family']] = sorted('/'.join(map(str, t)) for t in ex)
    if viol:
        raise N.Blocked(f'receipt invariants violated (STOP, nothing admitted): {viol[:10]}')
    v = M76.admit(D, panel, root, comp, cands)
    N.publish(root/EXTRA_FILE, N.seal(dict(
        schema='coopmwm.a82.admission_extra.v1', amendment=AMENDMENT, admission_receipt_sha256=v['receipt_sha256'],
        reference=dict(wave=REF_WAVE, root=str(_STATE['root76']), pins_receipt_sha256=_STATE['pins']['receipt_sha256']),
        receipt_invariants='PASS on every verification record', exercised_tokens=exercised,
        note=('a76_d2p20.admit ran unchanged with its names re-pointed; its verdict rows carry the literal source label '
              '"A76 (new run, D2800_p20)" and schema coopmwm.a76.admission.v1 from that code'),
        wm=[dict(family=c['family'], draw_rank=c['draw_rank'], form=c['form'], **c['wm']) for c in cands]),
        'receipt_sha256'))
    return v


def dropper():
    import a50_fullp60 as A50
    return A50._clone('a82_s3b20_drop', Path(D2.__file__), dict(
        PREFIX=PREFIX, AMENDMENT=AMENDMENT, STAGE=STAGE, DROP_MANIFEST=DROP_MANIFEST,
        DROP_RULE=D2.DROP_RULE.replace('a75d2v_* / a75d2c_*', 'a82v_*')))


# ================================================================================================ CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('check-top', 'pin-top', 'pin', 'prepare', 'dry', 'mech', 'mech-compare', 'verify',
                                       'admit', 'status', 'drop-raw'))
    p.add_argument('--final-root')
    p.add_argument('--a76-root')
    p.add_argument('--top')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    p.add_argument('--drop-raw', action='store_true')
    p.add_argument('--drop-procs', type=int, default=16)
    a = p.parse_args(argv)
    if a.command == 'check-top':
        raw = Path(a.top).read_bytes()
        doc = candidates_doc(raw, sha_bytes(raw))
        print(json.dumps(dict(sha256=sha_bytes(raw), n=doc['n_candidates'], by_form=doc['candidates_by_form'],
                              n_extended=doc['n_extended'],
                              families=[(c['family'], c['form'], c['n_roles']) for c in doc['candidates']]), indent=1))
        return
    N = _N()
    root = Path(a.final_root)
    root76 = Path(a.a76_root) if a.a76_root else a76_root()
    if root.resolve() == root76.resolve():
        raise N.Blocked('the A82 root is a new root')
    if a.command == 'pin-top':
        print(json.dumps(pin_top(root)))
        return
    if a.command == 'pin':
        v = make_pins(root, root76)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], panel=v['panels']['v'], ref=v['ref']['wave_receipts'],
                              n_reports=v['ref']['n'])))
        return
    pins = read_pins(root)
    if pins is None:
        raise N.Blocked(f'{PINS_FILE} missing (run pin)')
    _point(root76, pins)
    if a.command == 'prepare':
        M76.setup('v', root, need_panel=False)
        print(json.dumps(prepare(root)))
        return
    D, panel = M76.setup('v', root)
    if panel['panel_sha256'] != pins['panels']['v']:
        raise N.Blocked('panel is not the pinned A76 admission panel')
    comp = M76.comparator(root)
    if M76.comp_record(comp) != pins['comparator']:
        raise N.Blocked('comparator differs from the pinned one')

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=M76.SETTING['v']),
                          sort_keys=True, default=str)
        print(line, flush=True)
        with (root/LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    cands = candidates(root)
    if a.command == 'dry':
        ws = plan_waves(D, panel, root, comp, cands)
        mw = mech_waves(D, panel, root, comp, cands)
        (rn, _, rs), = admit_waves(D, panel, root, comp, cands)[:1]
        have, src = _load(root, rn, rs)
        reps = {w: _mods_proxy()[1].report_of(root, have[w]) for w in have}
        print(json.dumps(dict(
            stage='v', panel_sha256=panel['panel_sha256'], comparator=comp['sha256'][:12], n_waves=len(ws),
            waves=[(n, len(s), s[0]['arm'], s[0]['job_id'], s[0]['world']['seed'], s[-1]['world']['seed'],
                    s[0]['checkpoint']['executor_sha256'][:8]) for n, _, s in ws],
            sim_s=sum(len(s)*HORIZON for _, _, s in ws), mech=dict(n_waves=len(mw), sim_s=sum(len(s)*HORIZON
                                                                                        for _, _, s in mw)),
            reference=dict(wave=rn, n=len(have), receipts=src['receipt_sha256'],
                           reports_ok=all(x.get('ok') for x in reps.values())),
            charged=D.charged(root), cap=CAP_SIM_S, a75_sha256=_H().a75_digest()), indent=1, default=str))
    elif a.command in ('mech', 'verify'):
        import fcntl
        if _H().a75_digest() != A75_SHA256:
            raise N.Blocked('a75_holdforms in this tree is not the registered executor d1af7e0f')
        if a.command == 'verify' and not mech_passed(root):
            raise N.Blocked('mechanics is not PASS: nothing runs')
        if a.command == 'mech' and (root/MECH_FILE).is_file():
            raise N.Blocked(f'{MECH_FILE} exists: mechanics ran')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = (mech_waves if a.command == 'mech' else plan_waves)(D, panel, root, comp, cands)
            pending = sum(len(s)-N.published(root, w, s, D2.check_d2)[2] for w, _, s in waves)
            cap_check(D.charged(root, PREFIX['v']), D.charged(root), pending*HORIZON)
            drop = a.drop_raw and a.command == 'verify'
            log(dict(start=a.command, n_waves=len(waves), episodes=sum(len(w[2]) for w in waves), pending=pending,
                     workers=a.workers, chunk=a.chunk, pid=os.getpid(), charged_before=D.charged(root), drop_raw=drop))
            X = dropper() if drop else None

            def hook(x):
                log(x)
                if drop and 'round' in x:
                    X.drop_raw(root, waves, procs=a.drop_procs, log=log)
            if drop:
                X.drop_raw(root, waves, procs=a.drop_procs, log=log)
            N.CHECKS['episode'] = D2.check_d2
            res = D2.execute(D, root, waves, workers=a.workers, chunk=a.chunk, log=hook, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'mech-compare':
        if (root/MECH_FILE).is_file():
            raise N.Blocked(f'{MECH_FILE} exists (one shot)')
        v = mech_compare(D, panel, root, comp, cands)
        N.publish(root/MECH_FILE, v)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], verdict=v['verdict'], failures=v['failures'][:20],
                              exercise=v['exercise'])))
        if v['verdict'] != 'PASS':
            sys.exit(5)
    elif a.command == 'drop-raw':
        print(json.dumps(dropper().drop_raw(root, plan_waves(D, panel, root, comp, cands), procs=a.drop_procs, log=log)))
    elif a.command == 'admit':
        if not mech_passed(root):
            raise N.Blocked('mechanics is not PASS')
        v = admit(D, panel, root, comp, cands)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], n_tested=v['n_tested'], n_admitted=v['n_admitted'],
                              n_delay_pass=v['n_delay_pass'], per_form=v['per_form'], bank=v['bank'],
                              duplicates=v['duplicates'], consequence=v['consequence'])))
    else:
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, mech=mech_passed(root),
                   admission=(root/ADMISSION_FILE).is_file())
        ws = plan_waves(D, panel, root, comp, cands)
        done = [N.published(root, n, s, D2.check_d2)[2] for n, _, s in ws]
        out.update(waves=len(ws), episodes=f'{sum(done)}/{sum(len(s) for _, _, s in ws)}')
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
