"""AMENDMENT 96 parts (c) / (d) (A96-P20; POST-HOC, exploratory): one D2800_p20 panel of 192 worlds per part, arms on
identical worlds, all RUN in this root. Derived from the A82 part-B driver a82_fourarm.py (5181bf4b; never launched):
same arm builders, setting, identity mechanics and analysis code path; changed: the BANK_HAND_P20 arm added, the U0 arm
dropped from the panel (kept in the identity mechanics), per-plan rule checks for the bank arms, A94-style contrasts and
descriptive block, part parameters.

  part (d) "earlier-search panel" (base /root/a96pd): worlds 988200-988391 (the staged A82 192-world panel, byte copy,
           panel sha 3c879f6f; block row A82_RANGES 988200-988599 already in the tree's runner); searched arm A82_BANK =
           the S3b bank 3609634d (parameter file + sha)
  part (c) "matched panel" (base /root/a96pc): 192 fresh worlds 984400-984591 (block 984400-984599, runner row
           A96C_RANGES added); searched arm BANK_SEARCHED_P20 = S4-p20 + S5-p20 admission bank rows (build_bank_a96.py)
  <searched>      runner arm A75_BANK on the frozen searched bank file (A96_BANK_FILE + A96_BANK_SHA256), on U0 d9042cc5
                  (specs exactly as a76_d2p20.confirm_waves builds A76C_BANK)
  BANK_HAND_P20   the same builder on the hand bank file (A96_HAND_FILE + A96_HAND_SHA256); NOT RUN when A96_HAND_SHA256
                  = none (no hand plan admitted): 3 arms, primary reverts to <searched> - DUO_HOLD
  DUO_HOLD        a32_ablation.a32_specs('DUO_HOLD') (duo12_v20 / UNION_UNC 39f384a2) on U0 (the A82 / A73 20 % arm)
  MAPPO20C        the A77 iteration-84 export bound by a77_mappo20c.bind, through A52.with_executor on the U0 specs

Mechanics (setting of this part; cap 14,400 sim-s): the A82 identity (8 episodes: U0, A75_BANK with the A76 bank,
the MAPPO20C path bound as IDENT_U0, DUO_HOLD vs their stored records) + per-plan RULE CHECKS of each bank arm that runs
(2 episodes each on the A76 confirmation worlds 997300 / 997301: a80_s2b.plan_checks per family, U0 executor, a75
d1af7e0f, activation inside that plan's zone +-1 m, no CAV in two sessions, no refusal, 0 collisions / teleports).
Required env: A96_PART (c | d), A96_HAND_SHA256 (sha | none).
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

AMENDMENT = '96'
PART = os.environ.get('A96_PART', '')
if PART not in ('c', 'd'):
    raise SystemExit('A96_PART must be c or d')
P = PART.upper()
STAGE = f'P20{P}'
CELL = 'D2800_p20'
SOURCE_SETTING = 'p20c'
SETTING = f'a96{PART}_D2800_p20'
RUNNER_ATTR = dict(c='A96C_RANGES', d='A82_RANGES')[PART]
PHASE = f'a96p{PART}_p20'
HORIZON = 1_200
BLOCK = dict(c=(984400, 984599), d=(988200, 988599))[PART]
SIZES = (192,)
FORBIDDEN = ((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699), (987200, 987599))
if PART == 'c':
    FORBIDDEN = FORBIDDEN+((988200, 988999), (989000, 989699), (992400, 992523))
HAND_SHA_ENV = os.environ.get('A96_HAND_SHA256', '')
if not (HAND_SHA_ENV == 'none' or len(HAND_SHA_ENV) == 64):
    raise SystemExit('A96_HAND_SHA256 must be the registered BANK_HAND_P20 sha256 or "none" (no hand plan admitted)')
HAND_RUN = HAND_SHA_ENV != 'none'
SEARCHED = dict(c='BANK_SEARCHED_P20', d='A82_BANK')[PART]
HAND = 'BANK_HAND_P20'
BOOTSTRAP = dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')
FROZEN_U0_SHA256 = M76.FROZEN_U0_SHA256
A75_SHA256 = M76.A75_SHA256
BANK_ARMS = (SEARCHED, HAND) if HAND_RUN else (SEARCHED,)
ARMS = BANK_ARMS+('DUO_HOLD', 'MAPPO20C')
TAG = {SEARCHED: 'bank_searched', HAND: 'bank_hand', 'DUO_HOLD': 'duo_hold', 'MAPPO20C': 'mappo20c'}
PRIMARY = (SEARCHED, HAND) if HAND_RUN else (SEARCHED, 'DUO_HOLD')
SECONDARY = tuple((b, r) for r in ('DUO_HOLD', 'MAPPO20C') for b in BANK_ARMS)
DESCRIPTIVE = (('DUO_HOLD', 'MAPPO20C'),)
IDENT_ARMS = ('U0', 'A75_BANK_A76', 'IDENT_U0', 'DUO_HOLD')
IDENT_N = 2
RULE_ARMS = BANK_ARMS                       # per-plan rule checks (2 episodes each, A76 confirmation worlds)
MECH_EPISODES = (len(IDENT_ARMS)+len(RULE_ARMS))*IDENT_N
MECH_CAP = 14_400                           # registered: 936,000 = 14,400 + 4 x 192 x 1,200
assert MECH_EPISODES*HORIZON <= MECH_CAP
PREFIX = dict(i=f'a96{PART}i_', c=f'a96{PART}c_', v=f'a96{PART}v_unused_', m=f'a96{PART}m_unused_')
PANEL_FILE = f'A96{P}_PANEL_C.json'
PANEL_PIN = f'A96{P}_PANEL_C.sha256'
STAGED = 'panels'
A82_STAGED_192 = '/root/a82four/panels/A82F_PANEL_C_192.json'      # part (d) source panel (read-only byte copy)
A82_STAGED_192_SHA256 = '1648eb4b6d578a0256896120f49531d99edc4badf1e9e87e21762eb432462a89'
A82_PANEL_192 = '3c879f6ff47fa75ebbf5f5055f4ed35ad7e54be96a704dffa743287b86aa6485'
BANK_DIR = f'a96{PART}_banks'
BANK_FILE_OF = {SEARCHED: f'A96{P}_BANK_SEARCHED_{CELL}.json', HAND: f'A96{P}_BANK_HAND_{CELL}.json'}
BANK_ENV = {SEARCHED: ('A96_BANK_FILE', 'A96_BANK_SHA256'), HAND: ('A96_HAND_FILE', 'A96_HAND_SHA256')}
A76_BANK_COPY = f'A96{P}_IDENT_A76BANK_{CELL}.json'
A76_BANK_PREFIX = 'f8b5237e'
BANK_PIN = f'A96{P}_BANKS.json'
MECH_FILE = f'A96{P}_MECH.json'
RESULTS_FILE = f'A96{P}_RESULTS_{CELL}.json'
LOG_FILE = f'A96{P}_LOG.jsonl'
STOP_FILE = f'STOP_A96P{P}'
DROP_MANIFEST = f'A96{P}_DROPPED_RAW.jsonl'
#: the stored A32 DUO_HOLD chunk receipt used by the identity (byte copy; registered A32 run on the p20c panel)
A32_WAVE = 'WAVE_DUO_HOLD__0000-0047.json'
A32_WAVE_SHA256_ENV = 'A82_A32_WAVE_SHA256'
P20C_PANEL = 'nm_eval_panel_wz3to2_p20c_D2800.json'
P20C_PANEL_SHA256 = 'd2ed7783ae9ea071f89aec4397d3ee155161878b44e1717b02188b16e7ff7fa2'
IDENT_FIELDS = ('activations', 'activations_by_family', 'issued', 'acts')
RECEIPT_KEYS = ('a37', 'a75', 'oracle', 'a32', 'learned_trigger')
AUTHORITY = (f'MAIN AMENDMENT 96 part ({PART}) (A96-P20, 2026-11-14; POST-HOC, exploratory): {SEARCHED} vs BANK_HAND_P20 '
             f'vs DUO_HOLD vs MAPPO20C on one 192-world D2800_p20 panel (block {BLOCK[0]}-{BLOCK[1]}); phase {PHASE}')
DISCLOSURE = ('POST-HOC and exploratory, registered after the 60 % searched-bank screening and random-control outcomes were '
              'known; banks from uncorrected 24-world admission screens (best admitted plan per form); hand-written plans '
              'are the search seeds; bank-size asymmetry possible; MAPPO20C is a single training run; coordination arms '
              'sit on U0, not on MAPPO20C; never pooled'
              + ('; part (d): earlier-search S3b bank admitted vs U0 on reused A76 worlds; NOT a second attempt at the '
                 'part (c) test' if PART == 'd' else ''))
PRIMARY_RULE = (f'"searched better" SUPPORTED iff {PRIMARY[0]} - {PRIMARY[1]} total delay CI95 upper < 0 (paired '
                'percentile bootstrap 10,000, seed 20261104, n = 192) AND the five-group 10 s rule relative to '
                f'{PRIMARY[1]} (nm_finaleval._bank_compare groups.all_pass); otherwise NOT_SEPARATED (CI95 contains 0, '
                'or the group rule fails) or REFERENCE_BETTER (CI95 lower > 0); diff = treatment - reference, negative = '
                'less delay')
MECH_RULE = ('PASS iff (a) A82 identity: for each arm type (U0, A75_BANK with the A76 bank, the MAPPO20C path = u0 '
             'd9042cc5 bound as IDENT_U0 through a77_mappo20c.bind + A52.with_executor, DUO_HOLD) on its 2 identity '
             'worlds the record equals the stored record (A76C_COMP / A76C_BANK on 997300 / 997301; A32 WAVE_DUO_HOLD '
             'chunk 0000-0047 on 993100 / 993101): J_veh_s, safety, world, executor d9042cc5, behavioural receipts, A76 '
             'reports equal; (b) each bank arm that runs, on 997300 / 997301: executor U0 d9042cc5, a75 d1af7e0f, per '
             'plan no receipt invariant violation (a80_s2b.plan_checks on the acts / events of that plan), every '
             'activation inside the zone of that plan +-1 m, no CAV in two sessions at once, no refusal, collisions = '
             'teleports = 0 (a plan that never activates is reported, not a failure)')
DROP_RULE = D2.DROP_RULE.replace('a75d2v_* / a75d2c_*', PREFIX['c']+'*')


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
    return (os.environ if env is None else env).get('A96_HOST_BASE', f'/root/a96p{PART}')


# ================================================================================================ pure helpers

def seeds_of(n):
    if n not in SIZES:
        raise ValueError(f'panel size {n} is not one of {SIZES}')
    return tuple(range(BLOCK[0], BLOCK[0]+n))


def caps(n):
    conf = len(ARMS)*n*HORIZON
    assert MECH_CAP+conf <= 936_000
    return dict(mech=MECH_CAP, confirm=conf, total=MECH_CAP+conf)


def block_free(block, table):
    lo, hi = block
    if any(lo <= b1 and hi >= b0 for b0, b1 in FORBIDDEN):
        raise ValueError(f'{lo}-{hi}: forbidden range')
    own = [r for r in table if (r[0], r[1]) == (lo, hi)]
    other = [r for r in table if (r[0], r[1]) != (lo, hi) and lo <= r[1] and hi >= r[0]]
    if len(own) != 1 or own[0][2] != lo or other:
        raise ValueError(f'A96 block {lo}-{hi}: own rows {own}, overlapping {other}')
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


def mech_verdict(rows, rule_fails=()):
    fails = [f'{r["arm"]} {r["seed"]}' for r in rows if not r['PASS']] + list(rule_fails)
    if len(rows) != len(IDENT_ARMS)*IDENT_N or {r['arm'] for r in rows} != set(IDENT_ARMS):
        fails.append(f'{len(rows)} identity rows / arms {sorted({r["arm"] for r in rows})}')
    return ('PASS' if not fails else 'FAIL'), fails


def primary_verdict(row, primary=None):
    """The registered primary (Amendment 96 (c)/(d)): SUPPORTED iff CI95 upper < 0 AND the five-group rule relative to
    the primary reference; direction SEARCHED_BETTER / NOT_SEPARATED / REFERENCE_BETTER (CI95 lower > 0)."""
    t, r = primary or PRIMARY
    ci = row['total_minus_ref']['ci95']
    groups = row.get('groups') or {}
    a_pass, b_pass = bool(ci[1] < 0), groups.get('all_pass') is True
    ok = bool(a_pass and b_pass)
    return dict(rule=PRIMARY_RULE, contrast=f'{t} - {r}', group_rule_reference=r, diff=row['total_minus_ref']['diff'],
                ci95=list(ci), p_two_sided=row['total_minus_ref'].get('p_two_sided'), delay_pass=a_pass,
                group_rule_pass=b_pass, supported=ok, verdict='SUPPORTED' if ok else 'NOT SUPPORTED',
                direction='SEARCHED_BETTER' if ok else ('REFERENCE_BETTER' if ci[0] > 0 else 'NOT_SEPARATED'),
                worst_10pct_cvar90=row.get('tail_cvar90'), groups=groups,
                hard_brakes_minus_ref=row.get('hard_brakes_minus_ref'), ttc_events_minus_ref=row.get('ttc_events_minus_ref'))


def reading(rows):
    out = {}
    for c in rows:
        ci = c['total_minus_ref']['ci95']
        out[f'{c["treatment"]}-{c["reference"]}'] = dict(kind=c['kind'], ci95=list(ci), saves=bool(ci[1] < 0),
                                                         worse=bool(ci[0] > 0))
    return out


def split_family(r, fam):
    """(= a94_bank.split_family) the a75 receipts of a bank episode restricted to one plan family."""
    return dict(r, acts=[a for a in r.get('acts') or [] if a.get('family', a.get('strategy')) == fam],
                holdforms=[e for e in r.get('holdforms') or [] if e.get('family') == fam])


def sessions(r, horizon=HORIZON):
    out = []
    for act in r.get('acts') or []:
        fam, t0 = act.get('family', act.get('strategy')), int(act['t'])
        for vid in act['members']:
            rel = (act.get('release') or {}).get(vid)
            t1 = max(horizon, t0+1) if rel is None else t0+int(rel[0])
            out.append((fam, vid, t0, t1))
    return out


def overlap_stats(r, horizon=HORIZON):
    """(= a94_bank.overlap_stats, A91 definition) per-family active seconds, seconds with >= 2 plans active,
    activations started while another plan was active, CAVs in two sessions at overlapping times (must be 0), seconds by
    number of plans active."""
    ss = sessions(r, horizon)
    span = {}
    for fam, vid, t0, t1 in ss:
        span[(fam, t0)] = max(span.get((fam, t0), t0), t1)
    secs = {}
    for (fam, t0), t1 in span.items():
        secs.setdefault(fam, set()).update(range(t0, max(t1, t0+1)))
    fams = sorted(secs)
    cover = Counter(t for f in fams for t in secs[f])
    both = sum(1 for n in cover.values() if n >= 2)
    started = sum(1 for (fam, t0) in span if any(t0 in secs[g] for g in fams if g != fam))
    clash = 0
    by_vid = {}
    for fam, vid, t0, t1 in ss:
        by_vid.setdefault(vid, []).append((t0, t1, fam))
    for xs in by_vid.values():
        xs.sort()
        clash += sum(1 for a, b in zip(xs, xs[1:]) if b[0] < a[1] and a[2] != b[2])
    return dict(active_s_by_family={f: len(secs[f]) for f in fams}, concurrent_s=both,
                acts_by_family=dict(Counter(f for f, _ in span)), acts_started_while_other_active=started,
                cav_in_two_sessions=clash, seconds_by_n_plans_active=dict(sorted(Counter(cover.values()).items())))


def engaged_hist(r, horizon=HORIZON):
    """{k: seconds with exactly k distinct CAVs in a plan session (any plan), k >= 1}."""
    at = {}
    for _, vid, t0, t1 in sessions(r, horizon):
        for t in range(t0, max(t1, t0+1)):
            at.setdefault(t, set()).add(vid)
    return dict(sorted(Counter(len(v) for v in at.values()).items()))


def offers_by_plan(offers):
    out = {}
    for k, n in (offers or {}).items():
        fam, _, what = str(k).partition('|')
        out.setdefault(fam, Counter())[what or '?'] += int(n)
    return {f: dict(c) for f, c in sorted(out.items())}


def contrasts():
    """[(treatment, reference, kind)] of the registered analysis."""
    return ([PRIMARY+('primary',)] + [(a, r, 'secondary') for a, r in SECONDARY]
            + [(a, r, 'descriptive') for a, r in DESCRIPTIVE])


# ================================================================================================ setting / core

def core(n):
    """The a42 driver code with the A82 names; ledger phase cap = mech + 4 arms x n worlds (n = the pinned panel)."""
    N = _N()
    import a50_fullp60 as A50
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=caps(n)['total'], STOP_A42=STOP_FILE, HOST_BASE=host_base(), CELL=CELL,
        VERIFY_SETTING=SETTING, CONFIRM_SETTING=SETTING, BANK_DIR=BANK_DIR, IDENT_WAVE=f'A96{P}_IDENT_UNUSED',
        AMENDMENT=AMENDMENT, LABEL=f'A96{P}', JOB_PREFIX=f'a96{PART}', VERIFY_TAG=PREFIX['v'][:-1],
        CONFIRM_TAG=PREFIX['c'][:-1], IDENT_TAG=PREFIX['i'][:-1], VERIFY_WAVE_PREFIX=f'A96{P}V_',
        CONFIRM_WAVE_PREFIX=f'A96{P}C_', RESULTS_FILE=RESULTS_FILE,
        LOG_FILE=LOG_FILE, SCHEMA_PREFIX=f'coopmwm.a96{PART}', RUNNER_RANGES_ATTR=RUNNER_ATTR,
        RESULT_LABEL=f'amendment 96 part ({PART}); POST-HOC exploratory 20 % searched bank vs hand bank; one shot',
        AUTHORITY=AUTHORITY)
    m = A50._clone(f'a96_p20_core_{PART}_{n}', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (SETTING,)
    m.A42Runtime.phase = PHASE
    N.CHECKS['episode'] = D2.check_d2
    return m


def _runner():
    N = _N()
    from tools.coopmwm import nm_eval_runner as R
    if tuple(getattr(R, RUNNER_ATTR, ())) != (BLOCK,):
        raise N.Blocked(f'runner {RUNNER_ATTR} is not {BLOCK} (patch_tables_a96c.py / A82 tree)')
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
        PANEL_SIZE_REASON=f'MAIN amendment 96 part ({PART}): {n} worlds {lo}-{seeds[-1]} (D2800_p20); '
                          f'{seeds[-1]+1}-{hi} reserved unused',
        CELL_AUTHORITY=f'MAIN amendment 96 part ({PART}): searched bank vs hand bank vs DUO_HOLD vs MAPPO20C at D2800_p20',
        PANEL_AUTHORITY=(f'A96 amendment 96 part ({PART}): audited integers {lo}-{seeds[-1]} (runner block {lo}-{hi}) for '
                         'this panel ONLY; executed through the eval runner (scope label only); never pooled'),
        EXPOSURE_EXTRA=dict(checked=(('amendment-96 (c) audit: SEED_AUDIT_A96C.json',) if PART == 'c' else
                                     ('amendment-82 audit: SEED_AUDIT_A82.json (A82 staged panel)',)),
                            disclosed=f'; amendment 96 part ({PART}): registered at runtime by the runner'),
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
    if PART != 'c':
        raise N.Blocked('stage-panels is part (c) only (part (d) copies the staged A82 192-world panel)')
    for n in SIZES:
        d = Path(base)/STAGED
        p = d/f'A96C_PANEL_C_{n}.json'
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
    src = Path(A82_STAGED_192) if PART == 'd' else Path(panels_dir)/f'A96C_PANEL_C_{n}.json'
    if PART == 'd' and N.file_sha(src) != A82_STAGED_192_SHA256:
        raise N.Blocked(f'{src}: not the staged A82 192-world panel {A82_STAGED_192_SHA256[:8]}')
    register_setting(src, n)
    v = N.validate_panel(N.read(src))
    if PART == 'd' and v['panel_sha256'] != A82_PANEL_192:
        raise N.Blocked('part (d) panel sha is not 3c879f6f')
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
    want = os.environ.get('A96_PANEL_SHA256')
    if want and want != panel['panel_sha256']:
        raise N.Blocked(f'{PANEL_FILE}: panel sha {panel["panel_sha256"]} != registered A96_PANEL_SHA256 {want}')
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
    SYNTHETIC DRY ONLY (main sets _SYNTH for ``dry`` in a /dev/shm/a96* root with A96_SYNTH_SUBSTITUTE=1): the MAPPO20C
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
    if arm in BANK_ARMS:
        return bank_specs(D, panel, root, BANK_FILE_OF[arm], tag, comp)
    raise _N().Blocked(f'no A96 arm {arm!r}')


def confirm_waves(D, panel, root):
    N = _N()
    comp = comparator(root)
    out = [(f'A96{P}C_{arm}', 'episode', arm_specs(D, arm, panel, root, comp, f'{PREFIX["c"]}{TAG[arm]}'))
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
    W = f'A96{P}I_'
    out = [(W+'U0', 'episode', M76.u0_specs(c76, root, comp, f'{PREFIX["i"]}u0')[:IDENT_N]),
           (W+'A75_BANK_A76', 'episode', bank_specs(D, c76, root, A76_BANK_COPY, f'{PREFIX["i"]}bank_a76',
                                                    comp)[:IDENT_N]),
           (W+'IDENT_U0', 'episode', policy_specs(c76, root, comp, 'IDENT_U0', f'{PREFIX["i"]}ident_u0')[:IDENT_N]),
           (W+'DUO_HOLD', 'episode', duo_specs(p20c, root, f'{PREFIX["i"]}duo_hold')[:IDENT_N])]
    out += [(W+'RULE_'+arm, 'episode', bank_specs(D, c76, root, BANK_FILE_OF[arm], f'{PREFIX["i"]}rule_{TAG[arm]}',
                                                  comp)[:IDENT_N]) for arm in RULE_ARMS]
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


def _a80():
    import a80_s2b as A80            # registered A80 driver 48c5c448 (plan_checks only)
    return A80


def rule_rows(root, have, arm, fams, plans, zones, worlds, fails, exe=FROZEN_U0_SHA256):
    """Per-plan rule checks of one bank arm's mechanics records (pure apart from reading the vehicle logs)."""
    N = _N()
    out = []
    for w in worlds:
        rec = have.get(w)
        if rec is None:
            fails.append(f'{arm} {w}: missing')
            continue
        r = rec.get('a75') or {}
        errs, per = [], {}
        log = Path(root)/'evaluation'/rec.get('work_dir', rec['job_id'])/N.VEHICLE_LOG
        pos = None
        for fam in fams:
            rf = split_family(r, fam)
            e, ex, st = _a80().plan_checks(rf, plans[fam]['spec'])
            errs += [f'{fam}: {x}' for x in e]
            per[fam] = dict(activations=st['activations'], events=st.get('events'),
                            exercised=sorted('/'.join(map(str, t)) for t in ex))
            if rf['acts']:
                if pos is None:
                    if not log.is_file():
                        errs.append('vehicle log missing (zone check impossible)')
                        continue
                    pos = D2.vlog_pos(D2.read_vlog(log))
                errs += [f'{fam}: {x}' for x in D2.check_zone(dict(rf, activations=len(rf['acts'])), zones[fam], pos)
                         if x != 'no activation']
        if r.get('refused') not in (None, {}, []):
            errs.append(f'refused {r.get("refused")}')
        if rec.get('executor_sha256_loaded') != exe:
            errs.append('executor is not the expected one')
        if rec.get('a75_sha256_loaded') != A75_SHA256:
            errs.append('a75 digest is not d1af7e0f')
        if any(int(rec['safety'].get(k) or 0) for k in D2.SAFETY_ZERO):
            errs.append('collisions / teleports')
        ov = overlap_stats(r)
        if ov['cav_in_two_sessions']:
            errs.append('a CAV in two sessions at once')
        unknown = sorted({a.get('family', a.get('strategy')) for a in r.get('acts') or []} - set(fams))
        if unknown:
            errs.append(f'acts of unknown families {unknown}')
        out.append(dict(world_id=w, errors=errs[:8], per_family=per, overlap=ov, engaged=engaged_hist(r)))
        if errs:
            fails.append(f'{arm} {w}: {errs[:3]}')
    return dict(families=list(fams), rows=out,
                activated={f: any(x['per_family'].get(f, {}).get('activations') for x in out)
                           or 'NOT ACTIVATED on 997300/997301' for f in fams})


def bank_plans_of(D, root, arm):
    from tools.coopmwm import a75_holdforms as H
    doc = json.loads(D.bank_path(root, BANK_FILE_OF[arm]).read_text(encoding='utf-8'))
    plans = {p['family']: p for p in H.a75_bank_plans(doc, CELL)[0]}
    return [r['family'] for r in doc['rows']], plans, {f: H.zone_bounds(p['spec']) for f, p in plans.items()}


def mech_compare(root, root76, a32_ref):
    N = _N()
    A = _A()
    root = Path(root)
    refs, srcs = stored_refs(root76, a32_ref)
    D, _ = setup(root)
    rows, checks, rule_fails = [], {}, []
    ws = ident_waves(D, root, root76)
    for wave, _, specs in ws:
        recs, _, cov = N.published(root, wave, specs, D2.check_d2)
        if cov != len(specs):
            raise N.Blocked(f'{wave}: {cov}/{len(specs)} published')
        arm = wave[len(f'A96{P}I_'):]
        if arm.startswith('RULE_'):
            arm = arm[len('RULE_'):]
            fams, plans, zones = bank_plans_of(D, root, arm)
            have = {r['world_id']: r for r in recs}
            checks[arm] = rule_rows(root, have, arm, fams, plans, zones, sorted(have), rule_fails)
            checks[arm]['zones'] = {f: list(z) for f, z in zones.items()}
            continue
        for new, (ref, rep_ref) in zip(recs, refs[arm]):
            rows.append(ident_row(arm, new, ref, A.report_of(root, new), rep_ref))
    if sorted(checks) != sorted(RULE_ARMS):
        rule_fails.append(f'rule checks for {sorted(checks)} (want {sorted(RULE_ARMS)})')
    verdict, fails = mech_verdict(rows, rule_fails)
    return N.seal(dict(schema=f'coopmwm.a96{PART}.mech.v1', amendment=AMENDMENT, part=PART, rule=MECH_RULE,
                       stored_receipts=srcs, verdict=verdict, failures=fails, identity=rows, rule_checks=checks,
                       hand_run=HAND_RUN, consequence='PASS: confirm may run; FAIL: stop and report'), 'receipt_sha256')


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
        return A77.bind(root, 'IDENT_U0', env.get('A96_U0_POLICY') or u0['path'])
    return A77.bind(root, 'MAPPO20C', env['A96_MAPPO20C_POLICY'], sha=env['A96_MAPPO20C_SHA256'],
                    receipt=env['A96_TRAIN_RECEIPT'], plan=env['A96_TRAIN_PLAN'])


def prepare(D, root, root76, env=None):
    """Write-once byte copies: the A76 bank (identity only) and the REGISTERED bank of every bank arm that runs
    (searched: A96_BANK_FILE / A96_BANK_SHA256; hand: A96_HAND_FILE / A96_HAND_SHA256, skipped when hand = none)."""
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
    for arm in BANK_ARMS:
        fv, sv = BANK_ENV[arm]
        f, s = env.get(fv), env.get(sv)
        if not (f and s):
            raise N.Blocked(f'{arm}: {fv} and {sv} required')
        jobs.append((BANK_FILE_OF[arm], Path(f), M76.check_sha(s, sv), arm))
    for name, src, want, what in jobs:
        if N.file_sha(src) != want:
            raise N.Blocked(f'{src}: sha256 {N.file_sha(src)} != {want}')
        q = C37.write_once(D.bank_path(root, name), src.read_bytes().decode('utf-8'))
        if N.file_sha(q) != want:
            raise N.Blocked(f'{q}: not a byte copy')
        doc = json.loads(q.read_text(encoding='utf-8'))
        got, refused = H.a75_bank_plans(doc, CELL)
        if not got or (refused and any(refused.values())) or sorted(p['family'] for p in got) != \
                sorted(r['family'] for r in doc['rows'] if r.get('regime') == CELL):
            raise N.Blocked(f'{name}: the executor runs {len(got)} plans, refused {refused}')
        core_ = H.make_core('A75_BANK', CELL, doc)
        out[name] = dict(file_sha256=want, source=str(src), what=what, families=[p['family'] for p in got],
                         executor_priority=[(p['family'], p['lb95'], type(c).__name__)
                                            for p, c in zip(core_.plans, core_.cores)])
    N.publish(root/BANK_PIN, N.seal(dict(schema=f'coopmwm.a96{PART}.banks.v1', amendment=AMENDMENT, part=PART,
                                         hand_run=HAND_RUN, banks={a: out[BANK_FILE_OF[a]] for a in BANK_ARMS}),
                                    'receipt_sha256'), replace=True)
    return out


def bank_pinned(root):
    N = _N()
    p = Path(root)/BANK_PIN
    if not p.is_file():
        raise N.Blocked(f'{BANK_PIN} missing (prepare with the bank files / shas)')
    v = N.verify(N.read(p), 'receipt_sha256', p.name)
    if v['hand_run'] != HAND_RUN or sorted(v['banks']) != sorted(BANK_ARMS):
        raise N.Blocked(f'{BANK_PIN}: pinned arms {sorted(v["banks"])} != this run {sorted(BANK_ARMS)}')
    for arm in BANK_ARMS:
        want = os.environ.get(BANK_ENV[arm][1])
        if want and want != v['banks'][arm]['file_sha256']:
            raise N.Blocked(f'{BANK_PIN}: {arm} {v["banks"][arm]["file_sha256"][:12]} != registered {want[:12]}')
    return v


# ================================================================================================ analysis

def analyse_core(D, have, reports, worlds, bootstrap, *, fams, meta):
    """Every registered / descriptive number from loaded records (pure apart from D._contrast's bootstrap)."""
    import numpy as np
    N = _N()
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    rows = [D._contrast(have, reports, worlds, a, r, bootstrap, k) for a, r, k in contrasts()]
    arms = {a: dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                    safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
            for a in have}
    for a in BANK_ARMS:
        by_fam, offers, tot, eng, nact, per_world = Counter(), Counter(), Counter(), Counter(), Counter(), {}
        for w in worlds:
            r = have[a][w].get('a75') or {}
            for f, n in (r.get('activations_by_family') or {}).items():
                by_fam[f] += int(n)
            for k, n in (r.get('offers') or {}).items():
                offers[k] += int(n)
            ov = overlap_stats(r)
            per_world[w] = ov['concurrent_s']
            for k in ('concurrent_s', 'acts_started_while_other_active', 'cav_in_two_sessions'):
                tot[k] += int(ov[k])
            for f, n in ov['active_s_by_family'].items():
                tot[f'active_s|{f}'] += int(n)
            for k, n in ov['seconds_by_n_plans_active'].items():
                nact[int(k)] += int(n)
            for k, n in engaged_hist(r).items():
                eng[int(k)] += int(n)
        arms[a].update(families=list(fams[a]), activation_counts=D2.form_activations(have[a], worlds),
                       activations_by_family=dict(by_fam), offers_by_family_outcome=dict(offers),
                       offers_by_plan=offers_by_plan(offers),
                       overlap=dict(tot, worlds_with_overlap=sum(1 for v in per_world.values() if v > 0),
                                    seconds_by_n_plans_active=dict(sorted(nact.items())),
                                    definition=overlap_stats.__doc__.strip()),
                       cavs_engaged_at_once=dict(seconds_by_k=dict(sorted(eng.items())), max_k=max(eng) if eng else 0,
                                                 definition=engaged_hist.__doc__.strip()))
    arms['DUO_HOLD']['activations_per_world'] = sum(
        int((have['DUO_HOLD'][w].get('a32') or have['DUO_HOLD'][w].get('oracle') or {}).get('activations') or 0)
        for w in worlds)/len(worlds)
    return N.seal(dict(schema=f'coopmwm.a96{PART}.analysis.v1', amendment=AMENDMENT, part=PART, authority=AUTHORITY,
                       cell=CELL, disclosure=DISCLOSURE, n_worlds=len(worlds), bootstrap=bootstrap, hand_run=HAND_RUN,
                       hand_note=None if HAND_RUN else f'no hand plan admitted: {HAND} not run; primary = '
                                                       f'{SEARCHED} - DUO_HOLD',
                       arms_run=list(ARMS), pooled_with=None, primary=primary_verdict(rows[0]), contrasts=rows,
                       reading=reading(rows), arms=arms, families={k: list(v) for k, v in fams.items()},
                       multiplicity='one registered primary; secondaries uncorrected; DUO_HOLD - MAPPO20C descriptive',
                       **meta), 'analysis_sha256')


def summary(v):
    p = v['primary']
    return dict(analysis_sha256=v['analysis_sha256'],
                primary={k: p[k] for k in ('contrast', 'diff', 'ci95', 'delay_pass', 'group_rule_pass', 'verdict',
                                           'direction')},
                secondary=[dict(contrast=f"{r['treatment']} - {r['reference']}", diff=r['total_minus_ref']['diff'],
                                ci95=r['total_minus_ref']['ci95']) for r in v['contrasts'] if r['kind'] == 'secondary'],
                banks={a: dict(concurrent_s=v['arms'][a]['overlap']['concurrent_s'],
                               activations_by_family=v['arms'][a].get('activations_by_family'))
                       for a in v['arms'] if 'overlap' in v['arms'][a]})


def analyse(D, panel, root):
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
    banks = bank_pinned(root)
    fams = {a: tuple(banks['banks'][a]['families']) for a in BANK_ARMS}
    meta = dict(panel_sha256=panel['panel_sha256'],
                seeds=[panel['worlds'][0]['world']['seed'], panel['worlds'][-1]['world']['seed']], banks=banks,
                executors=dict(U0=FROZEN_U0_SHA256, MAPPO20C=_A77().read_bound(root, 'MAPPO20C')['sha256']),
                root=str(root), mech=N.read(root/MECH_FILE).get('receipt_sha256'),
                sources={k: v.get('receipt_sha256') for k, v in sources.items()}, label=D.RESULT_LABEL)
    return analyse_core(D, have, reports, worlds, bootstrap, fams=fams, meta=meta)


def dropper():
    import a50_fullp60 as A50
    return A50._clone(f'a96_p20_drop_{PART}', Path(D2.__file__), dict(
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
            raise N.Blocked('the A96 root is a new root')
    base = root.parent
    if a.command == 'stage-panels':
        print(json.dumps(stage_panels(base), indent=1))
        return
    if a.command == 'panel':
        n = int(a.n or 192)
        print(json.dumps(panel_in_root(a.panels_dir or base/STAGED, root, n, os.environ.get('A96_PANEL_SHA256'))))
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
        _SYNTH['on'] = (os.environ.get('A96_SYNTH_SUBSTITUTE') == '1' and root.as_posix().startswith('/dev/shm/a96'))
        ws = ident_waves(D, root, a.a76_root) if a.stage == 'i' else confirm_waves(D, panel, root)
        out = dict(stage=a.stage, part=PART, setting=SETTING, panel_sha256=panel['panel_sha256'], n_worlds=n,
                   arms=list(ARMS), hand_run=HAND_RUN, contrasts=contrasts(), n_waves=len(ws),
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
            out['rule_check_waves'] = [w for w, _, _ in ws[len(IDENT_ARMS):]]
            out['bank'] = bank_pinned(root)
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
        print(json.dumps(summary(v), default=str))           # own summary (the A91 fix: never an inherited layout)
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
