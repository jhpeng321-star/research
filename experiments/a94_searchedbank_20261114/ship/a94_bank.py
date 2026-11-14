"""A94 / A95 (AMENDMENTS 94 + 95; POST-HOC, exploratory; D2800_p60, 60% CAVs, map wz3to2): searched-plan bank vs
hand-written-plan bank on ONE fresh 192-world panel 988700-988891 (block 988700-988899; 988892-988899 reserved unused).
Amendment 95 (02:09) overrides Amendment 94 where they differ: BEST_SINGLE dropped, BANK_HAND added, primary changed.

Arms (all on the MAPPO60C executor 2abed317 / U0 base, same plant / setting as A81 / A91; diff = treatment - reference):
  BANK_SEARCHED  A75_BANK on A94_BANK_SEARCHED (rows of the S5 bank 577a0564 + the S4 bank a428cd1d, unchanged,
                 identical keys deduplicated; executor priority from the carried lb95 fields; build_bank_a94 searched)
  BANK_HAND      A75_BANK on A94_BANK_HAND (rows of the hand-written admission bank file(s), unchanged; two roots: the
                 admission code's own bank rule across both; build_bank_a94 hand). NOT RUN when no hand plan was
                 admitted (A94_HAND_SHA256=none): then 3 arms and the primary reverts to BANK_SEARCHED - DUO_HOLD_60C.
  DUO_HOLD_60C   A81 arm unchanged;  MAPPO60C  A81 arm unchanged
The arm set is fixed at import from A94_HAND_SHA256 (a sha256 = 4 arms; "none" = 3 arms; unset = refuse).

THIN WRAPPER over the registered A81 driver ops/a81_bigconfirm.py (c755a551, imported UNCHANGED), the A89 / A91 pattern
(copied from the A91 driver a91_combine.py aed9bd9d): A81 module names are re-pointed at A94 names. The analyse CLI
prints its own summary line (the A81 summary reads an A81-only key; A89's chain crashed there; A91 fix kept).

Mechanics (A70 worlds 996300 / 996301, setting a70c; 10 episodes = 12,000 sim-s with BANK_HAND, 8 = 9,600 without;
registered cap 14,400):
  identity A81: MAPPO60C, DUO_HOLD_60C and the A75-D2 bank (A75_BANK on c325ec1d; same executor path as the bank arms)
            equal the stored A70 / A75-D2 records (A81 ident_row unchanged);
  each bank arm: per plan (acts / events split by family) a80_s2b.plan_checks, executor MAPPO60C, a75 digest d1af7e0f,
            every activation inside that plan's zone +-1 m, no CAV in two sessions at once, no executor refusal,
            collisions = teleports = 0. A plan that never activates on those worlds is reported (not a failure).
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import a81_bigconfirm as B  # noqa: E402  (registered A81 driver, unchanged; names re-pointed below)

AMENDMENT = '94'
CELL = B.CELL
PHASE = 'searchedbank_a94'
HORIZON = B.HORIZON
IDENT_N, CONFIRM_N = 2, 192
BLOCK = (988700, 988899)                      # 988700-988891 used; 988892-988899 reserved unused
SEEDS_C = tuple(range(BLOCK[0], BLOCK[0]+CONFIRM_N))
RESERVED_C = (BLOCK[0]+CONFIRM_N, BLOCK[1])
assert SEEDS_C[-1] == 988891
HAND_SHA_ENV = os.environ.get('A94_HAND_SHA256', '')
if not (HAND_SHA_ENV == 'none' or len(HAND_SHA_ENV) == 64):
    raise SystemExit('A94_HAND_SHA256 must be the registered BANK_HAND sha256 or "none" (no hand plan admitted)')
HAND_RUN = HAND_SHA_ENV != 'none'
BANK_ARMS = ('BANK_SEARCHED', 'BANK_HAND') if HAND_RUN else ('BANK_SEARCHED',)
REF_ARMS = ('DUO_HOLD_60C', 'MAPPO60C')
ARMS = BANK_ARMS+REF_ARMS
IDENT_ARMS = (B.BANK_ARM, 'DUO_HOLD_60C', 'MAPPO60C')     # the A81 identity arms (stored A70 / A75-D2 records)
MECH_ARMS = IDENT_ARMS+BANK_ARMS
MECH_EPISODES = len(MECH_ARMS)*IDENT_N                    # 10 (8 without BANK_HAND) episodes
MECH_CAP = 14_400                                         # registered mechanics cap (Amendment 94)
CONFIRM_CAP = len(ARMS)*CONFIRM_N*HORIZON
CAP_SIM_S = MECH_CAP+CONFIRM_CAP
assert MECH_EPISODES*HORIZON <= MECH_CAP and CONFIRM_CAP == len(ARMS)*230_400 and CAP_SIM_S <= 936_000
SETTINGS = dict(i=B.SETTINGS['i'], c='a94c_D2800_p60')
RANGES = dict(i=B.RANGES['i'], c='A94_RANGES')
BANK_DIR = 'a94_banks'
SEARCHED_BANK = f'A94_BANK_SEARCHED_{CELL}.json'
HAND_BANK = f'A94_BANK_HAND_{CELL}.json'
BANK_FILE_OF = dict(BANK_SEARCHED=SEARCHED_BANK, BANK_HAND=HAND_BANK)
PREFIX = dict(i='a94i_', c='a94c_')
TAG = {B.BANK_ARM: 'bank', 'DUO_HOLD_60C': 'duo_hold_60c', 'MAPPO60C': 'mappo60c', 'BANK_HAND': 'bank_hand',
       'BANK_SEARCHED': 'bank_searched'}
AUTHORITY = ('MAIN AMENDMENTS 94 + 95 (A94, 2026-11-14; POST-HOC, exploratory): searched-plan bank (S5 + S4 admission '
             'bank rows, unchanged) vs hand-written-plan bank (hand admission bank rows, unchanged), DUO_HOLD_60C and '
             'MAPPO60C on one fresh 192-world D2800_p60 panel 988700-988891; phase searchedbank_a94, caps mech 14,400 + '
             'confirm %d' % CONFIRM_CAP)
DISCLOSURE = ('POST-HOC and exploratory. Both banks come from uncorrected 24-world admission screens on reused worlds; the '
              'bank rule (best admitted plan per form) replaced Amendment 94 rule "every admitted plan" after the S4 '
              'admission outcome was seen (Amendment 95); hand-written seeds were written by the same team that built '
              'the search; the banks may differ in number of plans; executor priority from the carried lb95 fields; '
              'bank arms run on the U0 base; no hand edits; never pooled with other panels; seed audit box-side only '
              '(laptop working tree not scanned)')
PRIMARY_CONTRAST = ('BANK_SEARCHED', 'BANK_HAND') if HAND_RUN else ('BANK_SEARCHED', 'DUO_HOLD_60C')
PRIMARY_RULE = ('"searched better" SUPPORTED iff %s - %s total delay CI95 upper < 0 (paired percentile bootstrap '
                '10,000, seed 20261104, n = 192) AND the five-group 10 s rule relative to %s (nm_finaleval._bank_compare '
                'groups.all_pass, reference = %s, as the A81 / A91 primary); otherwise reported as NOT SEPARATED (CI95 '
                'contains 0) or REFERENCE BETTER (CI95 lower > 0); diff = treatment - reference, negative = less delay'
                % (PRIMARY_CONTRAST+(PRIMARY_CONTRAST[1],)*2))
MECH_RULE = ('PASS iff (a) MAPPO60C, DUO_HOLD_60C and A75D2_BANK_60C on 996300 / 996301 equal the stored A70 / A75-D2 '
             'records (A81 ident_row, unchanged); (b) each bank arm on 996300 / 996301: executor '
             'MAPPO60C, a75 digest d1af7e0f, per plan no receipt invariant violation (a80_s2b.plan_checks on that '
             'plan\'s acts / events), every activation inside that plan\'s zone +-1 m, no CAV in two sessions at once, '
             'no executor refusal, collisions = teleports = 0. A plan that never activates on those worlds is reported '
             '(not a failure)')

# ------------------------------------------------------------------------------------------------ re-pointing
_ORIG = {}


def point():
    """Re-point the a81_bigconfirm module names (its functions read them at call time). Idempotent."""
    if _ORIG:
        return
    names = dict(AMENDMENT=AMENDMENT, PHASE=PHASE, IDENT_N=IDENT_N, CONFIRM_N=CONFIRM_N, ARMS=ARMS,
                 MECH_EPISODES=MECH_EPISODES, MECH_CAP=MECH_CAP, CONFIRM_CAP=CONFIRM_CAP, CAP_SIM_S=CAP_SIM_S,
                 BLOCK=BLOCK, SEEDS_C=SEEDS_C, RESERVED_C=RESERVED_C, SETTINGS=SETTINGS, RANGES=RANGES,
                 PANEL_C='A94_PANEL_C.json', PANEL_C_PIN='A94_PANEL_C.sha256', BANK_DIR=BANK_DIR, PREFIX=PREFIX,
                 TAG=TAG, STAGE_CAP=dict(mech=MECH_CAP, confirm=CONFIRM_CAP), MECH_FILE='A94_MECH.json',
                 RESULTS_FILE=f'A94_RESULTS_{CELL}.json', LOG_FILE='A94_LOG.jsonl',
                 DROP_MANIFEST='A94_DROPPED_RAW.jsonl', AUTHORITY=AUTHORITY, DISCLOSURE=DISCLOSURE,
                 MECH_RULE=MECH_RULE, PRIMARY_RULE=PRIMARY_RULE,
                 DROP_RULE=B.DROP_RULE.replace('a81c_*', 'a94c_*'),
                 core=core, register_c_setting=register_c_setting, _runner_check=_runner_check, prepare=prepare,
                 waves=waves, mech_compare=mech_compare, analyse=analyse)
    for k, v in names.items():
        _ORIG[k] = getattr(B, k)
        setattr(B, k, v)


def core(which):
    """B.core with the A94 labels / phase / caps / HOST_BASE (A94_HOST_BASE)."""
    N, A50 = B._mods()[:2]
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A94',
        HOST_BASE=os.environ.get('A94_HOST_BASE', '/root/a94'),
        VERIFY_SETTING=SETTINGS[which], CONFIRM_SETTING=SETTINGS[which],
        BANK_DIR=BANK_DIR, IDENT_WAVE='A94_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A94', JOB_PREFIX='a94',
        VERIFY_TAG='a94v', CONFIRM_TAG=f'a94{which}', IDENT_TAG='a94i',
        VERIFY_WAVE_PREFIX='A94V_', CONFIRM_WAVE_PREFIX='A94C_',
        RESULTS_FILE=B.RESULTS_FILE, LOG_FILE=B.LOG_FILE, SCHEMA_PREFIX='coopmwm.a94',
        RUNNER_RANGES_ATTR=RANGES[which],
        RESULT_LABEL='amendment 94; POST-HOC exploratory searched-plan bank on a fresh 192-world panel',
        AUTHORITY=AUTHORITY)
    m = A50._clone(f'a94_core_{which}', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (SETTINGS[which],)
    m.A42Runtime.phase = PHASE
    N.CHECKS['episode'] = B._D2().check_d2
    return m


def register_c_setting(root):
    """B.register_c_setting with A94 block / size / labels (the a70c setting otherwise)."""
    N = B._mods()[0]
    name = SETTINGS['c']
    if name not in N.SETTINGS:
        lo, hi = BLOCK
        N.SETTINGS[name] = dict(
            N.SETTINGS[SETTINGS['i']], N_WORLDS=CONFIRM_N, EVAL_RANGE=BLOCK, EVAL_EXTRA_RANGE=BLOCK,
            SEEDS_RESERVED=RESERVED_C,
            PANEL_SIZE_REASON=(f'MAIN amendment 94: {CONFIRM_N} fresh worlds {lo}-{lo+CONFIRM_N-1} (D2800_p60); '
                               f'{RESERVED_C[0]}-{hi} reserved unused'),
            CELL_AUTHORITY='MAIN amendment 94 (A94, post-hoc): D2800_p60 searched-plan bank fresh panel',
            PANEL_AUTHORITY=(f'A94 amendment 94: fresh audited integers {lo}-{lo+CONFIRM_N-1} (D2800_p60) for A94 ONLY; '
                             'executed through the eval runner (scope label only); never pooled'),
            EXPOSURE_EXTRA=dict(
                checked=(f'amendment-94 audit: {lo}-{hi} absent from REGISTRATION.md (except the A94 paragraph), '
                         'protocols/experiment_history.json, every panel file / runner range table / code under /root '
                         'and /home/ubuntu on the run host (SEED_AUDIT_A94.json)',),
                disclosed='; amendment 94: fresh integers, registered at runtime by the runner'),
            PANEL_FILE=Path(root)/B.PANEL_C, PANEL_3C_FILE=None)
        N.BANK_SETTINGS = N.BANK_SETTINGS+(name,)
        N.SETTINGS_PINS[name] = dict(PANEL_SHA256=None, PANEL_3C_SHA256=None)
    return name


def _runner_check():
    N = B._mods()[0]
    from tools.coopmwm import nm_eval_runner as R
    if tuple(getattr(R, 'A94_RANGES', ())) != (BLOCK,):
        raise N.Blocked(f'runner A94_RANGES is not {BLOCK} (patch_tables_a94.py)')
    B.block_free(BLOCK, R.EVAL_HELDOUT_RANGES)
    return R


# ------------------------------------------------------------------------------------------------ banks

def _sha_env(var):
    s = os.environ.get(var, '')
    if len(s) != 64:
        raise SystemExit(f'{var} (a registered bank sha256) is required')
    return s


def bank_shas():
    """{bank file: registered sha256} for the bank arms that run (BANK_HAND only when HAND_RUN)."""
    out = {SEARCHED_BANK: _sha_env('A94_BANK_SHA256')}
    if HAND_RUN:
        out[HAND_BANK] = _sha_env('A94_HAND_SHA256')
    return out


def bank_src_dir():
    """Source of the built bank files: <tree>/ops/a94 (A94_BANK_SRC_DIR only for the zero-SUMO dry test)."""
    return Path(os.environ.get('A94_BANK_SRC_DIR') or HERE/'a94')


def families(docs):
    """Pure: {arm: plan families in row order} from {arm: bank doc}; every row needs exactly one template."""
    out = {}
    for arm, doc in docs.items():
        fams = tuple(r['family'] for r in doc['rows'])
        if not fams or len(set(fams)) != len(fams) or                 sorted(t['family'] for t in doc['clean5c']['templates']) != sorted(fams):
            raise ValueError(f'{arm}: rows / templates malformed')
        out[arm] = fams
    return out


def _bank_docs(root=None):
    """{arm: bank doc} (from the root copies when root is given, else from the source dir), sha-checked."""
    N = B._mods()[0]
    pins = bank_shas()
    out = {}
    for arm in BANK_ARMS:
        name = BANK_FILE_OF[arm]
        p = Path(root)/BANK_DIR/name if root is not None else bank_src_dir()/name      # = D.bank_path(root, name)
        if N.file_sha(p) != pins[name]:
            raise N.Blocked(f'{p}: not {pins[name][:8]}')
        out[arm] = json.loads(p.read_text(encoding='utf-8'))
    return out


def arm_families(root=None):
    return families(_bank_docs(root))


def prepare(D, root, a75d2_root):
    """Write-once byte copies: the frozen A75-D2 bank (identity arm) and the bank file of every bank arm that runs."""
    N = B._mods()[0]
    from tools.coopmwm import a75_holdforms as H
    out = {}
    pins = bank_shas()
    todo = [(Path(a75d2_root)/'a75d2_banks'/B.BANK_FILE, B.BANK_FILE, B.BANK_SHA256)]
    todo += [(bank_src_dir()/BANK_FILE_OF[a], BANK_FILE_OF[a], pins[BANK_FILE_OF[a]]) for a in BANK_ARMS]
    for src, name, pin in todo:
        if N.file_sha(src) != pin:
            raise N.Blocked(f'{src}: not {pin[:8]}')
        q = D.C37.write_once(D.bank_path(root, name), src.read_bytes().decode('utf-8'))
        if N.file_sha(q) != pin or q.read_bytes() != src.read_bytes():
            raise N.Blocked(f'{q}: not the pinned bytes')
        out[name] = N.file_sha(q)
    fams = arm_families(root)
    prio = {}
    for arm in BANK_ARMS:
        doc = json.loads(D.bank_path(root, BANK_FILE_OF[arm]).read_text(encoding='utf-8'))
        c = H.make_core('A75_BANK', CELL, doc)
        if sorted(p['family'] for p in c.plans) != sorted(fams[arm]):
            raise N.Blocked(f'{BANK_FILE_OF[arm]}: executor plans {[p["family"] for p in c.plans]}')
        prio[arm] = [(p['family'], p['lb95'], type(k).__name__) for p, k in zip(c.plans, c.cores)]
    out['families'] = fams
    out['executor_priority'] = prio
    out['hand_run'] = HAND_RUN
    return out


def arm_specs(D, arm, panel, root, a66_root, tag):
    _, _, A67, A68 = B._mods()
    D2 = B._D2()
    exe = A68.executor(a66_root)
    if arm in BANK_ARMS:            # = the A81 / A91 bank arm (A75_BANK) on this arm's bank file
        name = BANK_FILE_OF[arm]
        entry = A68.bank_entry(D, root, name, bank_shas()[name])
        base = A67.with_exec(D._a37_specs(D.RUNNER_ARM['PLAN'], panel, root, entry, family='_', tag=tag), exe, tag)
        return D2.a75_from_a37(base, 'A75_BANK', entry, family=None, tag=tag)
    # MAPPO60C, DUO_HOLD_60C and (identity only) A75D2_BANK_60C: the A81 arm_specs unchanged
    return B.arm_specs(D, arm, panel, root, a66_root, tag)


def waves(D, panel, root, a66_root, which):
    N = B._mods()[0]
    n = IDENT_N if which == 'i' else CONFIRM_N
    arms = MECH_ARMS if which == 'i' else ARMS
    out = []
    for arm in arms:
        specs = arm_specs(D, arm, panel, root, a66_root, f'{PREFIX[which]}{TAG[arm]}')[:n]
        if len(specs) != n:
            raise N.Blocked(f'{arm}: {len(specs)} specs (want {n})')
        out.append((f'A94{which.upper()}_{arm}', 'episode', specs))
    if len({s['job_id'] for _, _, ss in out for s in ss}) != len(arms)*n:
        raise N.Blocked('duplicate job ids')
    return out


def _a80():
    import a80_s2b as A80            # registered A80 driver 48c5c448 (plan_checks only; its _point_d2 is NOT called)
    return A80


# ------------------------------------------------------------------------------------------------ pure helpers

def split_family(r, fam):
    """The a75 receipts of a bank episode restricted to one plan family (acts + hold-form events)."""
    return dict(r, acts=[a for a in r.get('acts') or [] if a.get('family', a.get('strategy')) == fam],
                holdforms=[e for e in r.get('holdforms') or [] if e.get('family') == fam])


def sessions(r, horizon=HORIZON):
    """[(family, vid, t0, t1)] member sessions from the executor acts (open session: t1 = max(horizon, t0 + 1))."""
    out = []
    for act in r.get('acts') or []:
        fam, t0 = act.get('family', act.get('strategy')), int(act['t'])
        for vid in act['members']:
            rel = (act.get('release') or {}).get(vid)
            t1 = max(horizon, t0+1) if rel is None else t0+int(rel[0])
            out.append((fam, vid, t0, t1))
    return out


def overlap_stats(r, horizon=HORIZON):
    """Pure, descriptive (A91 definition): per-family active seconds (a plan is active while any member of one of its
    activations is in session), seconds with >= 2 families active, activations started while another family was
    active, and CAVs found in two sessions at overlapping times (must be 0: the executor forbids it). A94 adds
    seconds_by_n_plans_active {n: seconds with exactly n plans active, n >= 1}."""
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
    """Pure, descriptive: {k: seconds with exactly k distinct CAVs in a plan session (any plan), k >= 1}."""
    at = {}
    for _, vid, t0, t1 in sessions(r, horizon):
        for t in range(t0, max(t1, t0+1)):
            at.setdefault(t, set()).add(vid)
    return dict(sorted(Counter(len(v) for v in at.values()).items()))


def offers_by_plan(offers):
    """{'family|outcome': n} -> {family: {outcome: n}} (outcomes: fire, member_busy, or the executor's refusal key)."""
    out = {}
    for k, n in (offers or {}).items():
        fam, _, what = str(k).partition('|')
        out.setdefault(fam, Counter())[what or '?'] += int(n)
    return {f: dict(c) for f, c in sorted(out.items())}


def primary_verdict(rows, contrast=None):
    """Pure: the registered primary from the _contrast rows (keyed 'treatment - reference'). Five-group rule relative
    to the primary's reference. direction: SEARCHED_BETTER (= SUPPORTED) / NOT_SEPARATED (CI95 contains 0, or upper < 0
    with the group rule failing) / REFERENCE_BETTER (CI95 lower > 0)."""
    t, ref = contrast or PRIMARY_CONTRAST
    by = {f"{r['treatment']} - {r['reference']}": r for r in rows}
    name = f'{t} - {ref}'
    p = by[name]
    lo, u = p['total_minus_ref']['ci95']
    g = (p.get('groups') or {}).get('all_pass') is True
    ok = bool(u < 0 and g)
    return dict(rule=PRIMARY_RULE, contrast=name, group_rule_reference=ref,
                diff=p['total_minus_ref']['diff'], ci95=p['total_minus_ref']['ci95'], delay_pass=bool(u < 0),
                group_rule_pass=g, supported=ok, verdict='SUPPORTED' if ok else 'NOT SUPPORTED',
                direction='SEARCHED_BETTER' if ok else ('REFERENCE_BETTER' if lo > 0 else 'NOT_SEPARATED'),
                groups=p.get('groups'), worst_10pct_cvar90=p.get('tail_cvar90'))


def contrasts(hand_run):
    """(treatment, reference, kind) rows of the registered analysis (Amendment 95; fallback of Amendment 94)."""
    if hand_run:
        return (('BANK_SEARCHED', 'BANK_HAND', 'primary'),
                ('BANK_SEARCHED', 'DUO_HOLD_60C', 'secondary'),
                ('BANK_HAND', 'DUO_HOLD_60C', 'secondary'),
                ('BANK_SEARCHED', 'MAPPO60C', 'secondary'),
                ('BANK_HAND', 'MAPPO60C', 'secondary'),
                ('DUO_HOLD_60C', 'MAPPO60C', 'descriptive'))
    return (('BANK_SEARCHED', 'DUO_HOLD_60C', 'primary'),
            ('BANK_SEARCHED', 'MAPPO60C', 'secondary'),
            ('DUO_HOLD_60C', 'MAPPO60C', 'descriptive'))


CONTRASTS = contrasts(HAND_RUN)


def summary(v):
    """The one-line console summary (A94 keys; never the A81 'primary' layout - the A91 fix)."""
    p = v['primary']
    return dict(analysis_sha256=v['analysis_sha256'],
                primary={k: p[k] for k in ('contrast', 'diff', 'ci95', 'delay_pass', 'group_rule_pass', 'verdict',
                                           'direction')},
                secondary=[dict(contrast=f"{r['treatment']} - {r['reference']}", diff=r['total_minus_ref']['diff'],
                                ci95=r['total_minus_ref']['ci95']) for r in v['contrasts']
                           if r['kind'].startswith('secondary')],
                banks={a: dict(concurrent_s=v['arms'][a]['overlap']['concurrent_s'],
                               activations_by_family=v['arms'][a].get('activations_by_family'))
                       for a in v['arms'] if a.startswith('BANK_')})


# ------------------------------------------------------------------------------------------------ mechanics

def _rule_rows(D, root, ws, arm, fams, plans, zones, worlds, fails):
    N, _, A67, _ = B._mods()
    D2 = B._D2()
    wave = f'A94I_{arm}'
    have, _ = B._load(root, wave, ws[wave])
    out = []
    for w in worlds:
        rec = have.get(w)
        if rec is None:
            fails.append(f'{arm} {w}: missing')
            continue
        r = rec.get('a75') or {}
        errs, per = [], {}
        log = root/'evaluation'/rec.get('work_dir', rec['job_id'])/N.VEHICLE_LOG
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
        if rec.get('executor_sha256_loaded') != A67.MAPPO60C_SHA256:
            errs.append('executor is not MAPPO60C')
        if rec.get('a75_sha256_loaded') != D2.A75_SHA256:
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
                           or 'NOT ACTIVATED on 996300/996301' for f in fams})


def mech_compare(root, a70_root, ref70_reports, a75d2_root, a66_root):
    N, _, A67, _ = B._mods()
    from a32_ablation import report_of
    from tools.coopmwm import a75_holdforms as H
    root = Path(root)
    D, panel = B.setup('i', root)
    ws = dict((w, s) for w, _, s in waves(D, panel, root, a66_root, 'i'))
    worlds = [r['world_id'] for r in panel['worlds']][:IDENT_N]
    ref, ref_rep, srcs = B.stored_refs(panel, a70_root, ref70_reports, a75d2_root, a66_root, worlds)
    rows, fails = [], []
    for arm in IDENT_ARMS:
        wave = f'A94I_{arm}'
        have, _ = B._load(root, wave, ws[wave])
        if sorted(have) != sorted(worlds):
            fails.append(f'{wave}: worlds {sorted(have)}')
            continue
        for w in worlds:
            row = B.ident_row(arm, have[w], ref[arm][w], report_of(root, have[w]), ref_rep[arm][w],
                              A67.MAPPO60C_SHA256)
            rows.append(row)
            if not row['PASS']:
                fails.append(f'{arm} {w}')
    fams = arm_families(root)
    checks, zones_out = {}, {}
    for arm in BANK_ARMS:
        doc = json.loads(D.bank_path(root, BANK_FILE_OF[arm]).read_text(encoding='utf-8'))
        plans = {p['family']: p for p in H.a75_bank_plans(doc, CELL)[0]}
        zones = {f: H.zone_bounds(p['spec']) for f, p in plans.items()}
        zones_out[arm] = {f: list(z) for f, z in zones.items()}
        checks[arm] = _rule_rows(D, root, ws, arm, fams[arm], plans, zones, worlds, fails)
    if len(rows) != len(IDENT_ARMS)*IDENT_N:
        fails.append(f'{len(rows)} identity rows (want {len(IDENT_ARMS)*IDENT_N})')
    return N.seal(dict(schema='coopmwm.a94.mech.v1', amendment=AMENDMENT, rule=MECH_RULE,
                       panel_sha256=panel['panel_sha256'], worlds=worlds, stored_receipts=srcs,
                       verdict='PASS' if not fails else 'FAIL', failures=fails, identity=rows, rule_checks=checks,
                       zones=zones_out, families=fams, hand_run=HAND_RUN,
                       consequence='PASS: confirm may run; FAIL: stop and report'), 'receipt_sha256')


# ------------------------------------------------------------------------------------------------ analysis

def analyse_core(D, have, reports, worlds, bootstrap, *, sources, panel_sha256, root, seeds, fams, bank_meta=None):
    """Every registered / descriptive number from loaded records (pure apart from D._contrast's bootstrap)."""
    import numpy as np
    N = B._mods()[0]
    D2 = B._D2()
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    rows = [D._contrast(have, reports, worlds, a, r, bootstrap, k) for a, r, k in CONTRASTS]
    prim = primary_verdict(rows)
    arms = {a: dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                    safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
            for a in ARMS}
    for a in BANK_ARMS:
        by_fam, offers, tot, eng = Counter(), Counter(), Counter(), Counter()
        nact = Counter()
        per_world = {}
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
                       cavs_engaged_at_once=dict(seconds_by_k=dict(sorted(eng.items())),
                                                 max_k=max(eng) if eng else 0,
                                                 definition=engaged_hist.__doc__.strip()))
    arms['DUO_HOLD_60C']['activations_per_world'] = sum(
        int((have['DUO_HOLD_60C'][w].get('a37') or have['DUO_HOLD_60C'][w].get('oracle') or {}).get('activations') or 0)
        for w in worlds)/len(worlds)
    return N.seal(dict(schema='coopmwm.a94.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       disclosure=DISCLOSURE, panel_sha256=panel_sha256, n_worlds=len(worlds), seeds=seeds,
                       bootstrap=bootstrap, banks=bank_meta, families={k: list(v) for k, v in fams.items()},
                       hand_run=HAND_RUN, bank_hand_note=None if HAND_RUN else
                       'no hand plan admitted: BANK_HAND not run; primary = BANK_SEARCHED - DUO_HOLD_60C (Amendment 94)',
                       arms_run=list(ARMS), arms_reused=None, pooled_with=None, root=str(root),
                       primary=prim, contrasts=rows, arms=arms,
                       multiplicity='one registered primary; two registered secondaries (uncorrected); every other '
                                    'contrast descriptive; no correction',
                       sources=sources, label=D.RESULT_LABEL), 'analysis_sha256')


def analyse(D, panel, root, a66_root):
    N = B._mods()[0]
    D2 = B._D2()
    root = Path(root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile') or len(worlds) != CONFIRM_N:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    if [r['world']['seed'] for r in panel['worlds']] != list(SEEDS_C):
        raise N.Blocked('not the A94 panel')
    have, sources, reports = {}, {}, {}
    for (wave, _, specs), arm in zip(waves(D, panel, root, a66_root, 'c'), ARMS):
        have[arm], sources[arm] = B._load(root, wave, specs)
        reports[arm] = D2._own_reports(root, have[arm], worlds)
    meta = {name: dict(file=name, file_sha256=sha) for name, sha in bank_shas().items()}
    return analyse_core(D, have, reports, worlds, bootstrap, panel_sha256=panel['panel_sha256'], root=root,
                        seeds=[SEEDS_C[0], SEEDS_C[-1]], fams=arm_families(root), bank_meta=meta,
                        sources={k: v.get('receipt_sha256') for k, v in sources.items()})


# ------------------------------------------------------------------------------------------------ CLI

def dry(root, a, which):
    D, panel = B.setup(which, root)
    if panel is None:
        raise SystemExit('panel missing (run panel-c)')
    ws = waves(D, panel, root, a.a66_root, which)
    out = dict(stage=which, setting=SETTINGS[which], panel_sha256=panel['panel_sha256'], hand_run=HAND_RUN,
               arms=list(ARMS), primary=list(PRIMARY_CONTRAST), contrasts=[list(c) for c in CONTRASTS], n_waves=len(ws),
               waves=[(n, len(s), s[0]['arm'], s[0]['job_id'], s[-1]['job_id'], s[0]['world']['seed'],
                       s[-1]['world']['seed']) for n, _, s in ws],
               episodes=sum(len(s) for _, _, s in ws), sim_s=sum(len(s)*HORIZON for _, _, s in ws),
               stage_cap=MECH_CAP if which == 'i' else CONFIRM_CAP, charged=D.charged(root), cap=CAP_SIM_S,
               a75_sha256=__import__('tools.coopmwm.a75_holdforms', fromlist=['x']).a75_digest(),
               families=arm_families(root),
               checkpoints={n: {k: v for k, v in s[0]['checkpoint'].items() if k != 'label'} for n, _, s in ws},
               horizon=[s[0].get('sim_s') for _, _, s in ws], windows=[s[0].get('window') for _, _, s in ws])
    by = {n: s for n, _, s in ws}
    W = which.upper()
    base = by[f'A94{W}_BANK_SEARCHED']
    fields = {f'{arm}_vs_BANK_SEARCHED': sorted({k for x, y in zip(by[f'A94{W}_{arm}'], base) for k in B.spec_diff(x, y)})
              for arm in BANK_ARMS[1:]}
    fields['MAPPO60C_vs_DUO_HOLD_60C'] = sorted({k for x, y in zip(by[f'A94{W}_MAPPO60C'], by[f'A94{W}_DUO_HOLD_60C'])
                                                for k in B.spec_diff(x, y)})
    out['spec_diff'] = fields
    if which == 'i':
        worlds = [r['world_id'] for r in panel['worlds']][:IDENT_N]
        _, reps, srcs = B.stored_refs(panel, a.a70_root, a.ref70_reports, a.a75d2_root, a.a66_root, worlds)
        out['stored_refs'] = dict(receipts=srcs, reports_ok={k: all(x['ok'] for x in v.values()) for k, v in reps.items()})
    print(json.dumps(out, indent=1, default=str))


def _args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('command')
    for k in ('--final-root', '--a66-root', '--a70-root', '--ref70-reports', '--a75d2-root'):
        p.add_argument(k)
    p.add_argument('--stage', choices=('i', 'c'), default='c')
    return p.parse_args(argv)


def main(argv=None):
    point()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'dry':
        a = _args(argv)
        root = Path(a.final_root)
        if a.stage == 'c':
            register_c_setting(root)
        return dry(root, a, a.stage)
    if argv and argv[0] == 'analyse':          # own CLI: the A81 summary line would KeyError on 'primary' keys
        N = B._mods()[0]
        a = _args(argv)
        root = Path(a.final_root)
        for other in (a.a66_root, a.a70_root, a.ref70_reports, a.a75d2_root):
            if other and root.resolve() == Path(other).resolve():
                raise N.Blocked('the A94 final root is a new root')
        D, panel = B.setup('c', root)
        if panel is None:
            raise N.Blocked(f'{B.PANEL_C} missing (run panel-c)')
        v = analyse(D, panel, root, a.a66_root)
        N.publish(root/B.RESULTS_FILE, v)
        print(json.dumps(summary(v), default=str))
        return
    return B.main(argv)


if __name__ == '__main__':
    main()
