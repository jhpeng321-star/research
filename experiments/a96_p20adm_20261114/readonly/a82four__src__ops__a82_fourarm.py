"""AMENDMENT 82 part B (A82-FOURARM20; POST-HOC, exploratory; DRAFT - not registered, nothing has run): ONE fresh
D2800_p20 panel (20 % CAVs, map wz3to2, 2,800 veh/h), four arms on identical worlds, all RUN in this root:

  A82_BANK  runner arm A75_BANK on a FROZEN bank file (parameter A82_BANK_FILE + A82_BANK_SHA256, fixed before
            launch: the A82 part-A bank if it admits anything, else the A76 bank f8b5237e), on the frozen U0 executor
            d9042cc5 - specs built exactly as a76_d2p20.confirm_waves builds A76C_BANK
  DUO_HOLD  the registered 20 % hand-written two-CAV hold: a32_ablation.a32_specs('DUO_HOLD') (A32 executor, bank
            UNION_UNC 39f384a2 = rule row duo12_v20, started whenever feasible) on the frozen U0 executor - exactly the
            A32 / A34 / A73 20 % arm (a73_lcside20.ref_specs)
  MAPPO20C  the A77 training run's iteration-84 export (bound by a77_mappo20c.bind: file sha = registered sha = the
            BOUND_00084.json receipt of the registered PLAN 57f51364; speed-only) through A52.with_executor on the U0
            specs (the A77 evaluation path)
  U0        the frozen speed-only transferred MAPPO d9042cc5 (a76_d2p20.u0_specs)

Mechanics / identity (setting a82c; 8 episodes, cap 9,600 sim-s), BEFORE any panel episode; each arm type reproduces a
stored record on 2 worlds:
  U0, A75_BANK (with the A76 bank), MAPPO20C path (u0 d9042cc5 bound as IDENT_U0 through a77 bind + A52.with_executor):
      A76 confirmation worlds 997300 / 997301 vs the stored A76C_COMP / A76C_BANK records of the A76 root (read-only)
  DUO_HOLD: p20c worlds 993100 / 993101 (the A35 / A50 / A77 identity worlds) vs the stored A32 WAVE_DUO_HOLD records
      (byte copy of the A32 chunk receipt 0000-0047, file sha pinned)
  MECH_RULE: J_veh_s exactly equal, safety counters equal, same world dict, executor loaded = d9042cc5, behavioural
  receipts equal (activations, activations_by_family, issued, acts of every a37 / a75 / a32 / oracle receipt), A76
  references: per-vehicle report equal (pinned, DROPPED_RAW-sealed); A32 reference: report not stored (descriptive).

Stages (own root; ledger phase ``fourarm_a82p20``; caps in code: mech 9,600 + confirm 4 x N x 1,200):
  stage-panels  zero SUMO: the 96 / 192 / 384-world panels of block 988200-988599 (write-once, staged in <base>/panels)
  panel         zero SUMO: copy the staged panel of the registered size (A82_N_WORLDS) into the root + pin file
  bind          zero SUMO: IDENT_U0 (setup) / MAPPO20C (A82_MAPPO20C_POLICY, _SHA256, A82_TRAIN_RECEIPT, A82_TRAIN_PLAN)
  prepare       zero SUMO: byte copies of the A76 bank (mechanics) and of the registered bank (A82_BANK_FILE / _SHA256)
  dry           zero SUMO (--stage i | c)
  mech          8 identity episodes -> mech-compare (A82F_MECH.json PASS / FAIL); confirm refuses unless PASS
  confirm       4 arms x N worlds (raw archives dropped per round: A75-D2 DROP_RULE)
  analyse       PRIMARY_RULE + secondaries -> A82F_RESULTS_D2800_p20.json
Commands: python ops/a82_fourarm.py <cmd> --final-root R [--a76-root /root/a76p20/root] [--a32-ref DIR]
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import a76_d2p20 as M76  # noqa: E402  (A76 driver, unchanged; never re-pointed in this process)
import a75_d2forms as D2  # noqa: E402  (A75-D2 driver, unchanged)

AMENDMENT = '82'
STAGE = 'FOURARM20'
CELL = 'D2800_p20'
SOURCE_SETTING = 'p20c'
SETTING = 'a82c_D2800_p20'
RUNNER_ATTR = 'A82_RANGES'
PHASE = 'fourarm_a82p20'
HORIZON = 1_200
BLOCK = (988200, 988599)
SIZES = (96, 192, 384)
FORBIDDEN = ((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699), (987200, 987599))
BOOTSTRAP = dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')
FROZEN_U0_SHA256 = M76.FROZEN_U0_SHA256
A75_SHA256 = M76.A75_SHA256
ARMS = ('A82_BANK', 'DUO_HOLD', 'MAPPO20C', 'U0')
TAG = dict(A82_BANK='bank', DUO_HOLD='duo_hold', MAPPO20C='mappo20c', U0='u0')
PRIMARY = ('A82_BANK', 'DUO_HOLD')
SECONDARY = (('A82_BANK', 'MAPPO20C'), ('DUO_HOLD', 'MAPPO20C'), ('MAPPO20C', 'U0'))
DESCRIPTIVE = (('A82_BANK', 'U0'), ('DUO_HOLD', 'U0'))
IDENT_ARMS = ('U0', 'A75_BANK_A76', 'IDENT_U0', 'DUO_HOLD')
IDENT_N = 2
MECH_EPISODES = len(IDENT_ARMS)*IDENT_N
MECH_CAP = MECH_EPISODES*HORIZON
assert MECH_CAP == 9_600
PREFIX = dict(i='a82fi_', c='a82fc_', v='a82fv_unused_', m='a82fm_unused_')
PANEL_FILE = 'A82F_PANEL_C.json'
PANEL_PIN = 'A82F_PANEL_C.sha256'
STAGED = 'panels'
BANK_DIR = 'a82f_banks'
BANK = f'A82F_BANK_{CELL}.json'
A76_BANK_COPY = f'A82F_IDENT_A76BANK_{CELL}.json'
A76_BANK_PREFIX = 'f8b5237e'
BANK_PIN = 'A82F_BANK.json'
MECH_FILE = 'A82F_MECH.json'
RESULTS_FILE = f'A82F_RESULTS_{CELL}.json'
LOG_FILE = 'A82F_LOG.jsonl'
STOP_FILE = 'STOP_A82F'
DROP_MANIFEST = 'A82F_DROPPED_RAW.jsonl'
#: the stored A32 DUO_HOLD chunk receipt used by the identity (byte copy; registered A32 run on the p20c panel)
A32_WAVE = 'WAVE_DUO_HOLD__0000-0047.json'
A32_WAVE_SHA256_ENV = 'A82_A32_WAVE_SHA256'
P20C_PANEL = 'nm_eval_panel_wz3to2_p20c_D2800.json'
P20C_PANEL_SHA256 = 'd2ed7783ae9ea071f89aec4397d3ee155161878b44e1717b02188b16e7ff7fa2'
IDENT_FIELDS = ('activations', 'activations_by_family', 'issued', 'acts')
RECEIPT_KEYS = ('a37', 'a75', 'oracle', 'a32', 'learned_trigger')
AUTHORITY = ('MAIN AMENDMENT 82 part B (A82-FOURARM20, DRAFT 2026-11-13; POST-HOC, exploratory): frozen 20 % bank vs '
             'the hand-written DUO_HOLD vs MAPPO20C vs U0, all four run on one fresh D2800_p20 panel (block '
             '988200-988599); phase fourarm_a82p20')
DISCLOSURE = ('POST-HOC and exploratory: designed after the A76 confirmation (A76 bank - U0, 96 worlds) and the A81 60 % '
              'result were known; the bank was chosen by a 24-world admission screen (selection-biased); DUO_HOLD and '
              'U0 are frozen registered arms; MAPPO20C is the fixed iteration-84 export (no selection). Fresh worlds; '
              'never pooled with any other panel')
PRIMARY_RULE = ('SUPPORTED iff A82_BANK - DUO_HOLD total delay CI95 upper < 0 (paired percentile bootstrap 10,000, seed '
                '20261104, n = the registered panel size) AND the five-group 10 s rule relative to DUO_HOLD '
                '(nm_finaleval._bank_compare groups.all_pass); diff = treatment - reference, negative = less delay')
MECH_RULE = ('PASS iff for each arm type (U0, A75_BANK with the A76 bank, the MAPPO20C path = u0 d9042cc5 bound as '
             'IDENT_U0 through a77_mappo20c.bind + A52.with_executor, DUO_HOLD) on its 2 identity worlds the record run '
             'from this tree equals the stored record (A76C_COMP / A76C_BANK of the A76 root on 997300 / 997301; A32 '
             'WAVE_DUO_HOLD chunk 0000-0047 on 993100 / 993101): J_veh_s exactly equal, safety counters equal, same '
             'world dict, executor loaded = d9042cc5 on both sides, behavioural receipts equal (activations, '
             'activations_by_family, issued, acts of every a37 / a75 / a32 / oracle receipt), bank records loaded '
             'a75 d1af7e0f; A76 references: per-vehicle report equal (pinned DROPPED_RAW-sealed report); A32 reference: '
             'no stored report (reported, not gating)')
DROP_RULE = D2.DROP_RULE.replace('a75d2v_* / a75d2c_*', 'a82fc_*')


def _N():
    return D2._mods()[0]


def _A():
    import a32_ablation as A
    return A


def _A77():
    import a77_mappo20c as A77
    return A77


def _A52():
    import a52_mappo60 as A52
    return A52


def host_base(env=None):
    return (os.environ if env is None else env).get('A82F_HOST_BASE', '/root/a82four')


# ================================================================================================ pure helpers

def seeds_of(n):
    if n not in SIZES:
        raise ValueError(f'panel size {n} is not one of {SIZES}')
    return tuple(range(BLOCK[0], BLOCK[0]+n))


def caps(n):
    conf = len(ARMS)*n*HORIZON
    return dict(mech=MECH_CAP, confirm=conf, total=MECH_CAP+conf)


def block_free(block, table):
    lo, hi = block
    if any(lo <= b1 and hi >= b0 for b0, b1 in FORBIDDEN):
        raise ValueError(f'{lo}-{hi}: forbidden range')
    own = [r for r in table if (r[0], r[1]) == (lo, hi)]
    other = [r for r in table if (r[0], r[1]) != (lo, hi) and lo <= r[1] and hi >= r[0]]
    if len(own) != 1 or own[0][2] != lo or other:
        raise ValueError(f'A82 block {lo}-{hi}: own rows {own}, overlapping {other}')
    return True


def spec_diff(a, b, path=''):
    """Pure (= a81_bigconfirm.spec_diff): the flattened keys whose values differ between two spec dicts."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += spec_diff(a.get(k, '<absent>'), b.get(k, '<absent>'), f'{path}.{k}' if path else str(k))
        return out
    return [] if a == b else [path]


def retag(specs, tag):
    return [dict(s, job_id=f'{tag}_{i:04}') for i, s in enumerate(specs)]


def receipts_of(rec):
    return {k: {f: json.dumps((rec.get(k) or {}).get(f), sort_keys=True) for f in IDENT_FIELDS}
            for k in RECEIPT_KEYS if isinstance(rec.get(k), dict)}


def ident_row(arm, new, ref, rep_new, rep_ref):
    rn, rr = receipts_of(new), receipts_of(ref)
    row = dict(arm=arm, world_id=new['world_id'], seed=new['world']['seed'], j_new=new['J_veh_s'],
               j_stored=ref['J_veh_s'], j_equal=new['J_veh_s'] == ref['J_veh_s'], safety_equal=new['safety'] == ref['safety'],
               same_world=new['world'] == ref['world'],
               executor_is_u0=new.get('executor_sha256_loaded') == ref.get('executor_sha256_loaded') == FROZEN_U0_SHA256,
               receipt_keys=[sorted(rn), sorted(rr)], receipts_equal=rn == rr and bool(rn) == bool(rr),
               report_compared=rep_ref is not None, safety=new['safety'])
    if rep_ref is not None:
        row['report_equal'] = json.dumps(rep_new, sort_keys=True) == json.dumps(rep_ref, sort_keys=True)
        row['report_ok'] = bool(rep_new.get('ok')) and bool(rep_ref.get('ok'))
    if arm == 'A75_BANK_A76':
        row['a75_sha256_loaded'] = new.get('a75_sha256_loaded')
    row['PASS'] = bool(row['j_equal'] and row['safety_equal'] and row['same_world'] and row['executor_is_u0']
                       and row['receipts_equal'] and (rep_ref is None or (row['report_equal'] and row['report_ok']))
                       and (arm != 'A75_BANK_A76' or row['a75_sha256_loaded'] == A75_SHA256))
    return row


def mech_verdict(rows):
    fails = [f'{r["arm"]} {r["seed"]}' for r in rows if not r['PASS']]
    if len(rows) != MECH_EPISODES or {r['arm'] for r in rows} != set(IDENT_ARMS):
        fails.append(f'{len(rows)} identity rows / arms {sorted({r["arm"] for r in rows})}')
    return ('PASS' if not fails else 'FAIL'), fails


def primary_verdict(row):
    ci = row['total_minus_ref']['ci95']
    groups = row.get('groups') or {}
    a_pass, b_pass = bool(ci[1] < 0), groups.get('all_pass') is True
    return dict(rule=PRIMARY_RULE, contrast='A82_BANK - DUO_HOLD', diff=row['total_minus_ref']['diff'], ci95=list(ci),
                p_two_sided=row['total_minus_ref'].get('p_two_sided'), delay_pass=a_pass, group_rule_pass=b_pass,
                supported=bool(a_pass and b_pass), verdict='SUPPORTED' if a_pass and b_pass else 'NOT SUPPORTED',
                worst_10pct_cvar90=row.get('tail_cvar90'), groups=groups,
                hard_brakes_minus_ref=row.get('hard_brakes_minus_ref'), ttc_events_minus_ref=row.get('ttc_events_minus_ref'))


def reading(rows):
    out = {}
    for c in rows:
        ci = c['total_minus_ref']['ci95']
        out[f'{c["treatment"]}-{c["reference"]}'] = dict(kind=c['kind'], ci95=list(ci), saves=bool(ci[1] < 0),
                                                         worse=bool(ci[0] > 0))
    return out


# ================================================================================================ setting / core

def core(n):
    """The a42 driver code with the A82 names; ledger phase cap = mech + 4 arms x n worlds (n = the pinned panel)."""
    N = _N()
    import a50_fullp60 as A50
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=caps(n)['total'], STOP_A42=STOP_FILE, HOST_BASE=host_base(), CELL=CELL,
        VERIFY_SETTING=SETTING, CONFIRM_SETTING=SETTING, BANK_DIR=BANK_DIR, IDENT_WAVE='A82F_IDENT_UNUSED',
        AMENDMENT=AMENDMENT, LABEL='A82F', JOB_PREFIX='a82f', VERIFY_TAG='a82fv_unused', CONFIRM_TAG='a82fc',
        IDENT_TAG='a82fi', VERIFY_WAVE_PREFIX='A82FV_', CONFIRM_WAVE_PREFIX='A82FC_', RESULTS_FILE=RESULTS_FILE,
        LOG_FILE=LOG_FILE, SCHEMA_PREFIX='coopmwm.a82f', RUNNER_RANGES_ATTR=RUNNER_ATTR,
        RESULT_LABEL='amendment 82 part B; POST-HOC exploratory four-arm confirmation at 20 % CAVs; one shot',
        AUTHORITY=AUTHORITY)
    m = A50._clone(f'a82_fourarm_core_{n}', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (SETTING,)
    m.A42Runtime.phase = PHASE
    N.CHECKS['episode'] = D2.check_d2
    return m


def _runner():
    N = _N()
    from tools.coopmwm import nm_eval_runner as R
    if tuple(getattr(R, RUNNER_ATTR, ())) != (BLOCK,):
        raise N.Blocked(f'runner {RUNNER_ATTR} is not {BLOCK} (patch_tables_a82.py)')
    block_free(BLOCK, R.EVAL_HELDOUT_RANGES)
    return R


def register_setting(panel_path, n, pin=None):
    """Process-local nm_finaleval setting (nm_finaleval unchanged): the p20c cell / plant / bindings with the A82 block
    (= a76_d2p20.register_setting / a77_mappo20c.register_setting pattern)."""
    N = _N()
    _runner()
    lo, hi = BLOCK
    seeds = seeds_of(n)
    N.SETTINGS[SETTING] = dict(
        N.SETTINGS[SOURCE_SETTING],
        N_WORLDS=n, EVAL_RANGE=(lo, hi), EVAL_EXTRA_RANGE=(lo, hi), SEEDS_RESERVED=(seeds[-1]+1, hi),
        PANEL_SIZE_REASON=f'MAIN amendment 82 part B: four-arm confirmation = {n} fresh worlds {lo}-{seeds[-1]} '
                          f'(D2800_p20); {seeds[-1]+1}-{hi} reserved unused',
        CELL_AUTHORITY='MAIN amendment 82 (A82-FOURARM20): bank vs DUO_HOLD vs MAPPO20C vs U0 at D2800_p20',
        PANEL_AUTHORITY=(f'A82 amendment 82: audited integers {lo}-{seeds[-1]} (runner block {lo}-{hi}) for the A82 '
                         'four-arm confirmation ONLY; executed through the eval runner (scope label only); never pooled'),
        EXPOSURE_EXTRA=dict(checked=('amendment-82 audit: SEED_AUDIT_A82.json',),
                            disclosed='; amendment 82: fresh integers, registered at runtime by the runner'),
        PANEL_FILE=Path(panel_path), PANEL_3C_FILE=None)
    if SETTING not in N.BANK_SETTINGS:
        N.BANK_SETTINGS = N.BANK_SETTINGS+(SETTING,)
    N.SETTINGS_PINS[SETTING] = dict(PANEL_SHA256=pin, PANEL_3C_SHA256=None)
    N.apply_setting(SETTING)
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    return SETTING


def stage_panels(base):
    """Zero SUMO, write-once: the three candidate panels (96 / 192 / 384 worlds from 988200), built exactly as the A76 /
    A81 panels (nm_finaleval.build_panel under this setting)."""
    N = _N()
    import a37_celltailor as C37
    from datetime import datetime, timezone
    out = {}
    for n in SIZES:
        d = Path(base)/STAGED
        p = d/f'A82F_PANEL_C_{n}.json'
        register_setting(p, n)
        if p.is_file():
            v = N.validate_panel(N.read(p))
        else:
            v = N.build_panel(N.default_spec_path(), datetime.now(timezone.utc).isoformat())
            N.validate_panel(v)
            C37.write_once(p, json.dumps(v, indent=1, sort_keys=True)+'\n')
        if tuple(w['world']['seed'] for w in v['worlds']) != seeds_of(n) or v['cell']['cell_key'] != CELL:
            raise N.Blocked(f'{p.name}: wrong seeds / cell')
        if dict(v['contract']['bootstrap']) != BOOTSTRAP:
            raise N.Blocked(f'{p.name}: bootstrap contract')
        out[n] = dict(file=str(p), file_sha256=N.file_sha(p), panel_sha256=v['panel_sha256'])
    return out


def panel_in_root(panels_dir, root, n, want_sha=None):
    """Zero SUMO, write-once: the staged panel of the registered size into the root (+ pin file)."""
    N = _N()
    import a37_celltailor as C37
    root = Path(root)
    seeds_of(n)
    src = Path(panels_dir)/f'A82F_PANEL_C_{n}.json'
    register_setting(src, n)
    v = N.validate_panel(N.read(src))
    if tuple(w['world']['seed'] for w in v['worlds']) != seeds_of(n):
        raise N.Blocked(f'{src.name}: wrong seeds')
    if want_sha and v['panel_sha256'] != want_sha:
        raise N.Blocked(f'staged {n}-world panel sha {v["panel_sha256"]} != registered {want_sha}')
    if (root/PANEL_FILE).exists() and N.file_sha(root/PANEL_FILE) != N.file_sha(src):
        raise N.Blocked(f'{PANEL_FILE} exists with another panel (write-once)')
    p = C37.write_once(root/PANEL_FILE, src.read_bytes().decode('utf-8'))
    C37.write_once(root/PANEL_PIN, f'{N.file_sha(p)}  {PANEL_FILE} panel_sha256 {v["panel_sha256"]} n {n}\n')
    return dict(file_sha256=N.file_sha(p), panel_sha256=v['panel_sha256'], n=n, seeds=[seeds_of(n)[0], seeds_of(n)[-1]])


def setup(root, *, need_panel=True):
    """-> (D, panel or None). The panel size is the pinned file's."""
    N = _N()
    root = Path(root)
    p = root/PANEL_FILE
    if not p.is_file():
        if need_panel:
            raise N.Blocked(f'{PANEL_FILE} missing (run panel)')
        register_setting(p, SIZES[0])
        return core(SIZES[0]), None
    pin = (root/PANEL_PIN).read_text(encoding='utf-8').split()
    if N.file_sha(p) != pin[0]:
        raise N.Blocked(f'{PANEL_FILE}: not the pinned file')
    n = int(pin[5])
    register_setting(p, n, pin=pin[3])
    D = core(n)
    panel = N.load_panel(p, pinned=pin[3])
    if tuple(w['world']['seed'] for w in panel['worlds']) != seeds_of(n) or panel['cell']['cell_key'] != CELL:
        raise N.Blocked(f'{PANEL_FILE}: not {CELL} {seeds_of(n)[0]}-{seeds_of(n)[-1]}')
    want = os.environ.get('A82F_PANEL_SHA256')
    if want and want != panel['panel_sha256']:
        raise N.Blocked(f'{PANEL_FILE}: panel sha {panel["panel_sha256"]} != registered A82F_PANEL_SHA256 {want}')
    if dict(panel['contract']['bootstrap']) != BOOTSTRAP:
        raise N.Blocked('panel bootstrap is not the registered one')
    return D, panel


# ================================================================================================ arms

def comparator(root):
    """The frozen U0 of this root (a76_d2p20.comparator with A76_COMP_SHA256 = d9042cc5; BOUND_U0.json, no rebind)."""
    N = _N()
    comp = M76.comparator(root, env={M76.COMP_SHA_ENV: FROZEN_U0_SHA256})
    if comp['rebind'] or comp['sha256'] != FROZEN_U0_SHA256:
        raise N.Blocked('U0 must be the frozen d9042cc5')
    return comp


def bank_specs(D, panel, root, bank_name, tag, comp):
    """= a76_d2p20.confirm_waves' A76C_BANK construction (A37 specs -> comparator executor -> A75_BANK)."""
    entry = D._bank_entry(root, bank_name)
    base = M76.with_exe(D._a37_specs('A37_PLAN', panel, root, entry, family='_', tag=tag), comp, tag)
    return D2.a75_from_a37(base, 'A75_BANK', entry, family=None, tag=tag)


def duo_specs(panel, root, tag):
    """= a73_lcside20.ref_specs('DUO_HOLD'): a32_ablation.a32_specs on the frozen U0 executor."""
    N = _N()
    specs = _A().a32_specs('DUO_HOLD', panel, root)
    if any(s['checkpoint'].get('executor_sha256') != FROZEN_U0_SHA256 for s in specs):
        raise N.Blocked('DUO_HOLD: executor is not the frozen U0')
    return retag(specs, tag)


_SYNTH = dict(on=False)


def policy_specs(panel, root, comp, which, tag):
    """= a77_mappo20c.confirm_specs: the U0 specs through A52.with_executor with the a77-bound executor.
    SYNTHETIC DRY ONLY (main sets _SYNTH for ``dry`` in a <base>/synth/root with A82F_SYNTH_SUBSTITUTE=1): the MAPPO20C
    wave is built with the IDENT_U0 binding because no MAPPO20C file exists yet; never for mech / confirm / analyse."""
    if which == 'MAPPO20C' and _SYNTH['on']:
        which = 'IDENT_U0'
    base = M76.u0_specs(panel, root, comp, tag)
    return _A52().with_executor(base, _A77().read_bound(root, which), tag)


def arm_specs(D, arm, panel, root, comp, tag):
    if arm == 'U0':
        return M76.u0_specs(panel, root, comp, tag)
    if arm == 'MAPPO20C':
        return policy_specs(panel, root, comp, 'MAPPO20C', tag)
    if arm == 'DUO_HOLD':
        return duo_specs(panel, root, tag)
    if arm == 'A82_BANK':
        return bank_specs(D, panel, root, BANK, tag, comp)
    raise _N().Blocked(f'no A82 arm {arm!r}')


def confirm_waves(D, panel, root):
    N = _N()
    comp = comparator(root)
    out = [(f'A82FC_{arm}', 'episode', arm_specs(D, arm, panel, root, comp, f'{PREFIX["c"]}{TAG[arm]}'))
           for arm in ARMS]
    n = len(panel['worlds'])
    if any(len(s) != n for _, _, s in out) or len({s['job_id'] for _, _, ss in out for s in ss}) != len(ARMS)*n:
        raise N.Blocked('confirmation waves: wrong size / duplicate job ids')
    return out


def ident_panels(root76):
    """(A76 confirmation panel, p20c panel), both pinned."""
    N = _N()
    p76 = N.verify(N.read(Path(root76)/'A76_PINS.json'), 'receipt_sha256', 'A76_PINS.json')
    name = M76.PANEL['c']
    c = N.verify(N.read(Path(root76)/name), 'panel_sha256', name)       # seal only: the active setting may be 192/384
    if c['panel_sha256'] != p76['panels']['c'] or [w['world']['seed'] for w in c['worlds']] != list(range(997300, 997396)):
        raise N.Blocked(f'{name}: not the pinned A76 confirmation panel 997300-997395')
    p20c = N.verify(N.read(HERE/P20C_PANEL), 'panel_sha256', P20C_PANEL)
    if p20c['panel_sha256'] != P20C_PANEL_SHA256:
        raise N.Blocked('p20c identity panel sha differs')
    return c, p20c


def ident_waves(D, root, root76):
    N = _N()
    comp = comparator(root)
    c76, p20c = ident_panels(root76)
    out = [('A82FI_U0', 'episode', M76.u0_specs(c76, root, comp, f'{PREFIX["i"]}u0')[:IDENT_N]),
           ('A82FI_A75_BANK_A76', 'episode', bank_specs(D, c76, root, A76_BANK_COPY, f'{PREFIX["i"]}bank_a76',
                                                        comp)[:IDENT_N]),
           ('A82FI_IDENT_U0', 'episode', policy_specs(c76, root, comp, 'IDENT_U0', f'{PREFIX["i"]}ident_u0')[:IDENT_N]),
           ('A82FI_DUO_HOLD', 'episode', duo_specs(p20c, root, f'{PREFIX["i"]}duo_hold')[:IDENT_N])]
    if any(len(s) != IDENT_N for _, _, s in out):
        raise N.Blocked('identity waves: wrong size')
    return out


# ================================================================================================ stored references

def _first2(have, seeds):
    by = {r['world']['seed']: r for r in have.values()}
    if any(s not in by for s in seeds):
        raise _N().Blocked(f'stored records lack seeds {seeds}')
    return [by[s] for s in seeds]


def stored_refs(root76, a32_ref):
    """{ident arm: [(stored record, stored report or None)] x 2}. Read-only: A76 root (its own driver / names / panel /
    comparator; A76C_COMP + A76C_BANK) and the A32 chunk copy (seal + file sha pin)."""
    N = _N()
    from a82_s3b20 import ref_episode_check
    A = _A()
    root76 = Path(root76)
    D76 = M76.core()                  # the A76 driver's own core / names (no second nm_finaleval setting is applied:
    c76, _ = ident_panels(root76)     # one setting per process; the A76 panel is checked by its seal + A76_PINS)
    comp76 = M76.comparator(root76, env={M76.COMP_SHA_ENV: FROZEN_U0_SHA256})
    (wc, _, sc), (wb, _, sb) = M76.confirm_waves(D76, c76, root76, comp76)
    seeds = [w['world']['seed'] for w in c76['worlds'][:IDENT_N]]
    out, src = {}, {}
    for arm, (wave, specs) in (('U0', (wc, sc)), ('A75_BANK_A76', (wb, sb))):
        have, s = M76._load(root76, wave, specs)
        src[arm] = s['receipt_sha256']
        rows = []
        for rec in _first2(have, seeds):
            ref_episode_check(root76, rec, None)
            rows.append((rec, A.report_of(root76, rec)))
        out[arm] = rows
    out['IDENT_U0'] = out['U0']
    src['IDENT_U0'] = src['U0']
    p = Path(a32_ref)/A32_WAVE
    want = os.environ.get(A32_WAVE_SHA256_ENV)
    if not want or N.file_sha(p) != want:
        raise N.Blocked(f'{p}: missing / not the pinned A32 chunk ({A32_WAVE_SHA256_ENV})')
    v = N.verify(N.read(p), 'receipt_sha256', p.name)
    recs = {r['job_id']: r for r in v['records']}
    _, p20c = ident_panels(root76)
    s20 = [w['world']['seed'] for w in p20c['worlds'][:IDENT_N]]
    out['DUO_HOLD'] = [(r, None) for r in _first2(recs, s20)]
    src['DUO_HOLD'] = v['receipt_sha256']
    return out, src


def stored_specs(root76):
    """The specs the stored records were run with, rebuilt by their own drivers (A76: a76_d2p20.confirm_waves on the A76
    root; A32: a32_ablation.a32_specs on the A32 bank root) - first 2 worlds; for the dry-run spec diff only."""
    A = _A()
    root76 = Path(root76)
    D76 = M76.core()
    c76, p20c = ident_panels(root76)
    comp76 = M76.comparator(root76, env={M76.COMP_SHA_ENV: FROZEN_U0_SHA256})
    (_, _, sc), (_, _, sb) = M76.confirm_waves(D76, c76, root76, comp76)
    return dict(U0=sc[:IDENT_N], A75_BANK_A76=sb[:IDENT_N], IDENT_U0=sc[:IDENT_N],
                DUO_HOLD=A.a32_specs('DUO_HOLD', p20c, A.RUN_ROOT)[:IDENT_N])


def mech_compare(root, root76, a32_ref):
    N = _N()
    A = _A()
    root = Path(root)
    refs, srcs = stored_refs(root76, a32_ref)
    D, _ = setup(root)
    rows = []
    for (wave, _, specs), arm in zip(ident_waves(D, root, root76), IDENT_ARMS):
        recs, _, cov = N.published(root, wave, specs, D2.check_d2)
        if cov != len(specs):
            raise N.Blocked(f'{wave}: {cov}/{len(specs)} published')
        for new, (ref, rep_ref) in zip(recs, refs[arm]):
            rows.append(ident_row(arm, new, ref, A.report_of(root, new), rep_ref))
    verdict, fails = mech_verdict(rows)
    return N.seal(dict(schema='coopmwm.a82f.mech.v1', amendment=AMENDMENT, rule=MECH_RULE, stored_receipts=srcs,
                       verdict=verdict, failures=fails, identity=rows,
                       consequence='PASS: confirm may run; FAIL: stop and report'), 'receipt_sha256')


def mech_passed(root):
    N = _N()
    p = Path(root)/MECH_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ================================================================================================ binding / banks (zero SUMO)

def bind(root, which, env=None):
    env = os.environ if env is None else env
    A77 = _A77()
    if which == 'IDENT_U0':
        u0, _ = _A().bindings(root)
        return A77.bind(root, 'IDENT_U0', env.get('A82_U0_POLICY') or u0['path'])
    return A77.bind(root, 'MAPPO20C', env['A82_MAPPO20C_POLICY'], sha=env['A82_MAPPO20C_SHA256'],
                    receipt=env['A82_TRAIN_RECEIPT'], plan=env['A82_TRAIN_PLAN'])


def prepare(D, root, root76, env=None):
    """Write-once byte copies: the A76 bank (identity only) and the REGISTERED confirmation bank (parameter)."""
    N = _N()
    import a37_celltailor as C37
    from tools.coopmwm import a75_holdforms as H
    env = os.environ if env is None else env
    root = Path(root)
    out = {}
    src76 = Path(root76)/M76.BANK_DIR/M76.BANK
    if not N.file_sha(src76).startswith(A76_BANK_PREFIX):
        raise N.Blocked(f'{src76}: not the A76 bank {A76_BANK_PREFIX}')
    jobs = [(A76_BANK_COPY, src76, N.file_sha(src76), 'A76 bank (identity episodes only)')]
    f, s = env.get('A82_BANK_FILE'), env.get('A82_BANK_SHA256')
    if f and s:
        jobs.append((BANK, Path(f), M76.check_sha(s, 'A82_BANK_SHA256'), env.get('A82_BANK_SOURCE', '?')))
    for name, src, want, what in jobs:
        if N.file_sha(src) != want:
            raise N.Blocked(f'{src}: sha256 {N.file_sha(src)} != {want}')
        q = C37.write_once(D.bank_path(root, name), src.read_bytes().decode('utf-8'))
        if N.file_sha(q) != want:
            raise N.Blocked(f'{q}: not a byte copy')
        got, refused = H.a75_bank_plans(json.loads(q.read_text(encoding='utf-8')), CELL)
        if not got or (refused and any(refused.values())):
            raise N.Blocked(f'{name}: the executor runs {len(got)} plans, refused {refused}')
        out[name] = dict(file_sha256=want, source=str(src), what=what, families=[p['family'] for p in got])
    if BANK in out:
        N.publish(root/BANK_PIN, N.seal(dict(schema='coopmwm.a82f.bank.v1', amendment=AMENDMENT, **out[BANK]),
                                       'receipt_sha256'))
    return out


def bank_pinned(root):
    N = _N()
    p = Path(root)/BANK_PIN
    if not p.is_file():
        raise N.Blocked(f'{BANK_PIN} missing (prepare with A82_BANK_FILE / A82_BANK_SHA256)')
    v = N.verify(N.read(p), 'receipt_sha256', p.name)
    want = os.environ.get('A82_BANK_SHA256')
    if want and want != v['file_sha256']:
        raise N.Blocked(f'{BANK_PIN}: bank {v["file_sha256"][:12]} != registered A82_BANK_SHA256 {want[:12]}')
    return v


# ================================================================================================ analysis

def analyse(D, panel, root):
    import numpy as np
    N = _N()
    root = Path(root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != BOOTSTRAP or len(worlds) not in SIZES:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    have, sources, reports = {}, {}, {}
    for (wave, _, specs), arm in zip(confirm_waves(D, panel, root), ARMS):
        recs, src, cov = N.published(root, wave, specs, D2.check_d2)
        if cov != len(specs):
            raise N.Blocked(f'{wave}: {cov}/{len(specs)} published')
        have[arm], sources[arm] = {r['world_id']: r for r in recs}, src
        reports[arm] = D2._own_reports(root, have[arm], worlds)
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    row = D._contrast(have, reports, worlds, PRIMARY[0], PRIMARY[1], bootstrap, 'primary')
    rows = [row] + [D._contrast(have, reports, worlds, a, r, bootstrap, 'secondary') for a, r in SECONDARY] + \
        [D._contrast(have, reports, worlds, a, r, bootstrap, 'descriptive') for a, r in DESCRIPTIVE]
    arms = {a: dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                    safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
            for a in have}
    by_fam = Counter()
    for w in worlds:
        for f, n in ((have['A82_BANK'][w]['a75'].get('activations_by_family')) or {}).items():
            by_fam[f] += int(n)
    arms['A82_BANK'].update(activation_counts=D2.form_activations(have['A82_BANK'], worlds),
                            activations_by_family=dict(by_fam))
    arms['DUO_HOLD']['activations_per_world'] = sum(
        int((have['DUO_HOLD'][w].get('a32') or have['DUO_HOLD'][w].get('oracle') or {}).get('activations') or 0)
        for w in worlds)/len(worlds)
    return N.seal(dict(schema='coopmwm.a82f.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       disclosure=DISCLOSURE, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                       seeds=[panel['worlds'][0]['world']['seed'], panel['worlds'][-1]['world']['seed']],
                       bootstrap=bootstrap, bank=bank_pinned(root),
                       executors=dict(U0=FROZEN_U0_SHA256, MAPPO20C=_A77().read_bound(root, 'MAPPO20C')['sha256']),
                       arms_run=list(ARMS), pooled_with=None, root=str(root), primary=primary_verdict(row),
                       contrasts=rows, reading=reading(rows), arms=arms,
                       multiplicity='one registered primary (A82_BANK - DUO_HOLD); three registered secondaries; '
                                    'two descriptive contrasts vs U0',
                       mech=N.read(root/MECH_FILE).get('receipt_sha256'),
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()}, label=D.RESULT_LABEL),
                  'analysis_sha256')


def dropper():
    import a50_fullp60 as A50
    return A50._clone('a82_fourarm_drop', Path(D2.__file__), dict(
        PREFIX=PREFIX, AMENDMENT=AMENDMENT, STAGE=STAGE, DROP_MANIFEST=DROP_MANIFEST, DROP_RULE=DROP_RULE))


# ================================================================================================ CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('stage-panels', 'panel', 'bind', 'prepare', 'dry', 'mech', 'mech-compare',
                                       'confirm', 'analyse', 'status', 'drop-raw'))
    p.add_argument('--final-root', required=True)
    p.add_argument('--a76-root', default='/root/a76p20/root')
    p.add_argument('--a32-ref')
    p.add_argument('--which', choices=('IDENT_U0', 'MAPPO20C'))
    p.add_argument('--n', type=int)
    p.add_argument('--panels-dir')
    p.add_argument('--stage', choices=('i', 'c'), default='c')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    p.add_argument('--drop-raw', action='store_true')
    p.add_argument('--drop-procs', type=int, default=16)
    a = p.parse_args(argv)
    N = _N()
    _A77()                           # imported before any setting / core: module-level side effects settle first
    _A52()
    root = Path(a.final_root)
    for other in (a.a76_root, a.a32_ref):
        if other and root.resolve() == Path(other).resolve():
            raise N.Blocked('the A82 root is a new root')
    base = root.parent
    if a.command == 'stage-panels':
        print(json.dumps(stage_panels(base), indent=1))
        return
    if a.command == 'panel':
        n = int(a.n or os.environ.get('A82_N_WORLDS') or 0)
        print(json.dumps(panel_in_root(a.panels_dir or base/STAGED, root, n, os.environ.get('A82F_PANEL_SHA256'))))
        return
    D, panel = setup(root, need_panel=a.command not in ('bind', 'prepare'))
    if a.command == 'bind':
        v = bind(root, a.which)
        print(json.dumps(dict(which=a.which, sha256=v['sha256'], binding_sha256=v['binding_sha256'],
                              training=v.get('training'))))
        return
    if a.command == 'prepare':
        print(json.dumps(prepare(D, root, a.a76_root), indent=1))
        return

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=SETTING), sort_keys=True,
                          default=str)
        print(line, flush=True)
        with (root/LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    n = len(panel['worlds'])
    cap = caps(n)
    if a.command == 'dry':
        _SYNTH['on'] = (os.environ.get('A82F_SYNTH_SUBSTITUTE') == '1' and root.as_posix().endswith('/synth/root'))
        ws = ident_waves(D, root, a.a76_root) if a.stage == 'i' else confirm_waves(D, panel, root)
        out = dict(stage=a.stage, setting=SETTING, panel_sha256=panel['panel_sha256'], n_worlds=n, n_waves=len(ws),
                   waves=[(w, len(s), s[0]['arm'], s[0]['job_id'], s[0]['world']['seed'], s[-1]['world']['seed'],
                           s[0]['checkpoint']['executor_sha256'][:8]) for w, _, s in ws],
                   episodes=sum(len(s) for _, _, s in ws), sim_s=sum(len(s)*HORIZON for _, _, s in ws), caps=cap,
                   synthetic_mappo20c_substitute=_SYNTH['on'], charged=D.charged(root), a75_sha256=__import__('tools.coopmwm.a75_holdforms',
                                                                   fromlist=['x']).a75_digest())
        if a.stage == 'c':
            out['bank'] = bank_pinned(root)
        else:
            theirs = stored_specs(a.a76_root)
            out['spec_diff_vs_stored_driver'] = {arm: sorted({k for x, y in zip(s, theirs[arm]) for k in spec_diff(x, y)})
                                                 for (_, _, s), arm in zip(ws, IDENT_ARMS)}
            refs, srcs = stored_refs(a.a76_root, a.a32_ref)
            out['stored'] = dict(receipts=srcs, n={k: len(v) for k, v in refs.items()},
                                 reports_ok={k: all((r or {}).get('ok', True) for _, r in v) for k, v in refs.items()})
        print(json.dumps(out, indent=1, default=str))
    elif a.command in ('mech', 'confirm'):
        import fcntl
        if __import__('tools.coopmwm.a75_holdforms', fromlist=['x']).a75_digest() != A75_SHA256:
            raise N.Blocked('a75_holdforms in this tree is not the registered executor d1af7e0f')
        if a.command == 'confirm' and not mech_passed(root):
            raise N.Blocked('mechanics is not PASS: nothing runs')
        if a.command == 'mech' and (root/MECH_FILE).is_file():
            raise N.Blocked(f'{MECH_FILE} exists: mechanics ran')
        if a.command == 'confirm':
            bank_pinned(root)
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            ws = ident_waves(D, root, a.a76_root) if a.command == 'mech' else confirm_waves(D, panel, root)
            pre = PREFIX['i'] if a.command == 'mech' else PREFIX['c']
            pending = sum(len(s)-N.published(root, w, s, D2.check_d2)[2] for w, _, s in ws)
            need, used = pending*HORIZON, D.charged(root, pre)
            if used+need > cap[a.command] or D.charged(root)+need > cap['total']:
                raise SystemExit(f'{a.command}: {used} + {need} > stage cap {cap[a.command]} (or total {cap["total"]})')
            drop = a.drop_raw and a.command == 'confirm'
            log(dict(start=a.command, n_waves=len(ws), episodes=sum(len(w[2]) for w in ws), pending=pending,
                     workers=a.workers, chunk=a.chunk, pid=os.getpid(), charged_before=D.charged(root), drop_raw=drop))
            X = dropper() if drop else None

            def hook(x):
                log(x)
                if drop and 'round' in x:
                    X.drop_raw(root, ws, procs=a.drop_procs, log=log)
            N.CHECKS['episode'] = D2.check_d2
            res = D2.execute(D, root, ws, workers=a.workers, chunk=a.chunk, log=hook, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'mech-compare':
        if (root/MECH_FILE).is_file():
            raise N.Blocked(f'{MECH_FILE} exists (one shot)')
        v = mech_compare(root, a.a76_root, a.a32_ref)
        N.publish(root/MECH_FILE, v)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], verdict=v['verdict'], failures=v['failures'],
                              rows=[(x['arm'], x['seed'], x['j_equal'], x['receipts_equal'], x.get('report_equal'),
                                     x['PASS']) for x in v['identity']])))
        if v['verdict'] != 'PASS':
            sys.exit(5)
    elif a.command == 'drop-raw':
        print(json.dumps(dropper().drop_raw(root, confirm_waves(D, panel, root), procs=a.drop_procs, log=log)))
    elif a.command == 'analyse':
        v = analyse(D, panel, root)
        N.publish(root/RESULTS_FILE, v)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], primary={k: v['primary'][k] for k in (
            'contrast', 'diff', 'ci95', 'delay_pass', 'group_rule_pass', 'verdict')})))
    else:
        out = dict(charged=D.charged(root), caps=cap, n_worlds=n, mech=mech_passed(root))
        try:
            ws = confirm_waves(D, panel, root)
            done = [N.published(root, w, s, D2.check_d2)[2] for w, _, s in ws]
            out.update(waves=len(ws), episodes=f'{sum(done)}/{sum(len(s) for _, _, s in ws)}')
        except N.Blocked as e:
            out['note'] = str(e)
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
