"""A89 (POST-HOC, exploratory; D2800_p60, 60% CAVs, map wz3to2): six arms on ONE fresh 96-world panel 988400-988495.

THIN WRAPPER over the registered A81 driver ops/a81_bigconfirm.py (c755a551, imported UNCHANGED): its module names are
re-pointed at A89 names (the A80 -> A75-D2 pattern) and only these functions are A89's own: core / register_c_setting
/ _runner_check (labels, block, HOST_BASE), prepare (two pinned bank copies), waves (six arms + mechanics), mech_compare
(A81 identity rows + A80 plan_checks), analyse (A89 contrasts, every statistic from the a42 ``_contrast`` /
nm_finaleval ``fe.paired_stats`` code A81 used), dry. Everything else (panel-c, mech, confirm, drop-raw, status, the
cap / OWNER.lock / STOP logic, D2.execute) is the A81 code path, called through a81_bigconfirm.main.

Arms (all on the MAPPO60C executor 2abed317, same plant / setting as A81; diff = treatment - reference):
  ROLL_BOTH     A75_PLAN on family d2_d86b1d6ac653e8ad of the arms bank (template/row byte-copied from the A75-D2 bank
                c325ec1d): lane-0 rank-1 CAV 24 m/s 30 s + lane-2 rank-0 CAV 16 m/s 20 s onset 20 s, lanes / far
  ROLL_L0_ONLY  A75_PLAN, the same plan with only the lane-2 role deleted (build_a89_arms.py)
  ROLL_L2_ONLY  A75_PLAN, the same plan with only the lane-0 role deleted (onset 20 s kept)
  TAPER         A75_PLAN on mut_1fca373046a3c66e (template/row byte-copied from the E1R candidate bank)
  DUO_HOLD_60C  A81 arm (a32 DUO_HOLD duo12_v20 / UNION_UNC 39f384a2, executor fields bound to MAPPO60C)
  MAPPO60C      A81 arm (A66 MAPPO60C policy_00084 2abed317 alone)

Mechanics (A70 worlds 996300 / 996301, setting a70c, 14 episodes = 16,800 sim-s):
  identity: MAPPO60C, DUO_HOLD_60C and the A75-D2 bank (A75_BANK on c325ec1d) must reproduce the stored A70 / A75-D2
            records exactly (the A81 ident_row, unchanged);
  plan arms: each of the four plans x 2 worlds checked by a80_s2b.plan_checks (receipt invariants), executor MAPPO60C,
            a75 digest d1af7e0f, every activation inside the zone (+-1 m), collisions = teleports = 0.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import a81_bigconfirm as B  # noqa: E402  (registered A81 driver, unchanged; names re-pointed below)

AMENDMENT = '89'
CELL = B.CELL
PHASE = 'sixarm_a89'
HORIZON = B.HORIZON
IDENT_N, CONFIRM_N = 2, 96
BLOCK = (988400, 988499)                      # 988400-988495 used; 988496-988499 reserved unused
SEEDS_C = tuple(range(BLOCK[0], BLOCK[0]+CONFIRM_N))
RESERVED_C = (BLOCK[0]+CONFIRM_N, BLOCK[1])
assert SEEDS_C[-1] == 988495
PLAN_ARMS = ('ROLL_BOTH', 'ROLL_L0_ONLY', 'ROLL_L2_ONLY', 'TAPER')
REF_ARMS = ('DUO_HOLD_60C', 'MAPPO60C')
ARMS = PLAN_ARMS+REF_ARMS
IDENT_ARMS = (B.BANK_ARM, 'DUO_HOLD_60C', 'MAPPO60C')     # the A81 identity arms (stored records exist)
MECH_EPISODES = (len(IDENT_ARMS)+len(PLAN_ARMS))*IDENT_N
MECH_CAP, CONFIRM_CAP = MECH_EPISODES*HORIZON, len(ARMS)*CONFIRM_N*HORIZON
CAP_SIM_S = MECH_CAP+CONFIRM_CAP
assert (MECH_CAP, CONFIRM_CAP, CAP_SIM_S) == (16_800, 691_200, 708_000)
SETTINGS = dict(i=B.SETTINGS['i'], c='a89c_D2800_p60')
RANGES = dict(i=B.RANGES['i'], c='A89_RANGES')
BANK_DIR = 'a89_banks'
ARMS_BANK = f'A89_ARMS_{CELL}.json'
ARMS_BANK_SHA256 = os.environ.get('A89_ARMS_SHA256_PIN', 'SET_AT_SHIP')
ARM_FAMILY = dict(ROLL_BOTH='d2_d86b1d6ac653e8ad', ROLL_L0_ONLY='a89l0_d86b1d6ac653e8ad',
                  ROLL_L2_ONLY='a89l2_d86b1d6ac653e8ad', TAPER='mut_1fca373046a3c66e')
PREFIX = dict(i='a89i_', c='a89c_')
TAG = {B.BANK_ARM: 'bank', 'DUO_HOLD_60C': 'duo_hold_60c', 'MAPPO60C': 'mappo60c', 'ROLL_BOTH': 'roll_both',
       'ROLL_L0_ONLY': 'roll_l0', 'ROLL_L2_ONLY': 'roll_l2', 'TAPER': 'taper'}
AUTHORITY = ('MAIN AMENDMENT 89 (A89, 2026-11-13; POST-HOC, exploratory): role ablation of the A75-D2 rolling plan '
             '(ROLL_BOTH / ROLL_L0_ONLY / ROLL_L2_ONLY) + fresh-panel check of E1R plan mut_1fca373046a3c66e (TAPER) '
             'vs MAPPO60C and DUO_HOLD_60C on one fresh 96-world D2800_p60 panel 988400-988495; phase sixarm_a89, '
             'caps mech 16,800 + confirm 691,200')
DISCLOSURE = ('POST-HOC and exploratory. The rolling plan was admitted from 36 A75-D2 plans on 24 worlds and confirmed in '
              'a bank (A75-D2 / A81); the TAPER plan was admitted in the exploratory E1R random draw on 24 worlds; the '
              'single-role arms are deterministic deletions, never run before. Fresh worlds; never pooled with any '
              'other panel; arms and plans frozen (no retuning)')
ABLATION_RULE = ('COOPERATION SUPPORTED iff ROLL_BOTH - ROLL_L0_ONLY AND ROLL_BOTH - ROLL_L2_ONLY total delay CI95 '
                 'uppers are both < 0 (paired percentile bootstrap 10,000, seed 20261104, n = 96); diff = treatment - '
                 'reference, negative = less delay')
TAPER_RULE = ('CONFIRMED iff TAPER - MAPPO60C total delay CI95 upper < 0 AND the five-group 10 s rule relative to '
              'MAPPO60C (nm_finaleval._bank_compare groups.all_pass)')
INTERACTION = ('(ROLL_BOTH - MAPPO60C) - [(ROLL_L0_ONLY - MAPPO60C) + (ROLL_L2_ONLY - MAPPO60C)] per world = '
               'fe.paired_stats(BOTH + MAPPO60C, L0 + L2) (same bootstrap); descriptive')
MECH_RULE = ('PASS iff (a) MAPPO60C, DUO_HOLD_60C and A75D2_BANK_60C on 996300 / 996301 equal the stored A70 / A75-D2 '
             'records (the A81 MECH_RULE / ident_row, unchanged) and (b) each of ROLL_BOTH, ROLL_L0_ONLY, ROLL_L2_ONLY, '
             'TAPER on 996300 / 996301: executor MAPPO60C 2abed317, a75_sha256 d1af7e0f, no receipt invariant '
             'violation (a80_s2b.plan_checks), every activation inside the zone +-1 m (a75_d2forms.check_zone), '
             'collisions = teleports = 0. A plan that never activates on those worlds is reported (not a failure)')

# ------------------------------------------------------------------------------------------------ re-pointing
_ORIG = {}


def point():
    """Re-point the a81_bigconfirm module names (its functions read them at call time). Idempotent."""
    if _ORIG:
        return
    names = dict(AMENDMENT=AMENDMENT, PHASE=PHASE, IDENT_N=IDENT_N, CONFIRM_N=CONFIRM_N, ARMS=ARMS,
                 MECH_EPISODES=MECH_EPISODES, MECH_CAP=MECH_CAP, CONFIRM_CAP=CONFIRM_CAP, CAP_SIM_S=CAP_SIM_S,
                 BLOCK=BLOCK, SEEDS_C=SEEDS_C, RESERVED_C=RESERVED_C, SETTINGS=SETTINGS, RANGES=RANGES,
                 PANEL_C='A89_PANEL_C.json', PANEL_C_PIN='A89_PANEL_C.sha256', BANK_DIR=BANK_DIR, PREFIX=PREFIX,
                 TAG=TAG, STAGE_CAP=dict(mech=MECH_CAP, confirm=CONFIRM_CAP), MECH_FILE='A89_MECH.json',
                 RESULTS_FILE=f'A89_RESULTS_{CELL}.json', LOG_FILE='A89_LOG.jsonl',
                 DROP_MANIFEST='A89_DROPPED_RAW.jsonl', AUTHORITY=AUTHORITY, DISCLOSURE=DISCLOSURE,
                 MECH_RULE=MECH_RULE, PRIMARY_RULE=ABLATION_RULE+' ; '+TAPER_RULE,
                 DROP_RULE=B.DROP_RULE.replace('a81c_*', 'a89c_*'),
                 core=core, register_c_setting=register_c_setting, _runner_check=_runner_check, prepare=prepare,
                 waves=waves, mech_compare=mech_compare, analyse=analyse)
    for k, v in names.items():
        _ORIG[k] = getattr(B, k)
        setattr(B, k, v)


def core(which):
    """B.core with the A89 labels / phase / caps / HOST_BASE (A89_HOST_BASE)."""
    N, A50 = B._mods()[:2]
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A89',
        HOST_BASE=os.environ.get('A89_HOST_BASE', '/root/a89'),
        VERIFY_SETTING=SETTINGS[which], CONFIRM_SETTING=SETTINGS[which],
        BANK_DIR=BANK_DIR, IDENT_WAVE='A89_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A89', JOB_PREFIX='a89',
        VERIFY_TAG='a89v', CONFIRM_TAG=f'a89{which}', IDENT_TAG='a89i',
        VERIFY_WAVE_PREFIX='A89V_', CONFIRM_WAVE_PREFIX='A89C_',
        RESULTS_FILE=B.RESULTS_FILE, LOG_FILE=B.LOG_FILE, SCHEMA_PREFIX='coopmwm.a89',
        RUNNER_RANGES_ATTR=RANGES[which],
        RESULT_LABEL='amendment 89; POST-HOC exploratory six-arm fresh panel (rolling role ablation + E1R taper plan)',
        AUTHORITY=AUTHORITY)
    m = A50._clone(f'a89_core_{which}', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (SETTINGS[which],)
    m.A42Runtime.phase = PHASE
    N.CHECKS['episode'] = B._D2().check_d2
    return m


def register_c_setting(root):
    """B.register_c_setting with A89 block / size / labels (the a70c setting otherwise)."""
    N = B._mods()[0]
    name = SETTINGS['c']
    if name not in N.SETTINGS:
        lo, hi = BLOCK
        N.SETTINGS[name] = dict(
            N.SETTINGS[SETTINGS['i']], N_WORLDS=CONFIRM_N, EVAL_RANGE=BLOCK, EVAL_EXTRA_RANGE=BLOCK,
            SEEDS_RESERVED=RESERVED_C,
            PANEL_SIZE_REASON=(f'MAIN amendment 89: {CONFIRM_N} fresh worlds {lo}-{lo+CONFIRM_N-1} (D2800_p60); '
                               f'{RESERVED_C[0]}-{hi} reserved unused'),
            CELL_AUTHORITY='MAIN amendment 89 (A89, post-hoc): D2800_p60 six-arm fresh panel',
            PANEL_AUTHORITY=(f'A89 amendment 89: fresh audited integers {lo}-{lo+CONFIRM_N-1} (D2800_p60) for A89 ONLY; '
                             'executed through the eval runner (scope label only); never pooled'),
            EXPOSURE_EXTRA=dict(
                checked=(f'amendment-89 audit: {lo}-{hi} absent from REGISTRATION.md, protocols/experiment_history.json, '
                         'every panel file / runner range table / code under /root and /home/ubuntu on the run host '
                         '(SEED_AUDIT_A89.json; inside the A82-audited block 988200-988599)',),
                disclosed='; amendment 89: fresh integers, registered at runtime by the runner'),
            PANEL_FILE=Path(root)/B.PANEL_C, PANEL_3C_FILE=None)
        N.BANK_SETTINGS = N.BANK_SETTINGS+(name,)
        N.SETTINGS_PINS[name] = dict(PANEL_SHA256=None, PANEL_3C_SHA256=None)
    return name


def _runner_check():
    N = B._mods()[0]
    from tools.coopmwm import nm_eval_runner as R
    if tuple(getattr(R, 'A89_RANGES', ())) != (BLOCK,):
        raise N.Blocked(f'runner A89_RANGES is not {BLOCK} (patch_tables_a89.py)')
    B.block_free(BLOCK, R.EVAL_HELDOUT_RANGES)
    return R


def _arms_sha():
    s = os.environ.get('A89_ARMS_SHA256') or ARMS_BANK_SHA256
    if len(s) != 64:
        raise SystemExit('A89_ARMS_SHA256 (the registered arms-bank sha256) is required')
    return s


def prepare(D, root, a75d2_root):
    """Write-once byte copies: the frozen A75-D2 bank (identity arm) and the pinned A89 arms bank (from the tree)."""
    N = B._mods()[0]
    from tools.coopmwm import a75_holdforms as H
    out = {}
    for src, name, pin in ((Path(a75d2_root)/'a75d2_banks'/B.BANK_FILE, B.BANK_FILE, B.BANK_SHA256),
                           (HERE/'a89'/ARMS_BANK, ARMS_BANK, _arms_sha())):
        if N.file_sha(src) != pin:
            raise N.Blocked(f'{src}: not {pin[:8]}')
        q = D.C37.write_once(D.bank_path(root, name), src.read_bytes().decode('utf-8'))
        if N.file_sha(q) != pin or q.read_bytes() != src.read_bytes():
            raise N.Blocked(f'{q}: not the pinned bytes')
        out[name] = N.file_sha(q)
    doc = json.loads(D.bank_path(root, ARMS_BANK).read_text(encoding='utf-8'))
    for arm, fam in ARM_FAMILY.items():
        H.make_core('A75_PLAN', CELL, doc, family=fam)          # raises on any refusal
    out['families'] = ARM_FAMILY
    return out


def plan_specs(D, arm, panel, root, a66_root, tag):
    """= a75_d2forms._plan_specs (A75_PLAN on the pinned arms bank, family of ``arm``), with the bank sha pinned."""
    _, _, A67, A68 = B._mods()
    exe = A68.executor(a66_root)
    entry = A68.bank_entry(D, root, ARMS_BANK, _arms_sha())
    fam = ARM_FAMILY[arm]
    base = A67.with_exec(D._a37_specs(D.RUNNER_ARM['PLAN'], panel, root, entry, family=fam, tag=tag), exe, tag)
    return B._D2().a75_from_a37(base, 'A75_PLAN', entry, family=fam, tag=tag)


def arm_specs(D, arm, panel, root, a66_root, tag):
    if arm in PLAN_ARMS:
        return plan_specs(D, arm, panel, root, a66_root, tag)
    # MAPPO60C, DUO_HOLD_60C and (identity only) A75D2_BANK_60C: the A81 arm_specs unchanged (bank = byte copy in
    # BANK_DIR, sha c325ec1d checked by A68.bank_entry)
    return B.arm_specs(D, arm, panel, root, a66_root, tag)


def waves(D, panel, root, a66_root, which):
    N = B._mods()[0]
    n = IDENT_N if which == 'i' else CONFIRM_N
    arms = IDENT_ARMS+PLAN_ARMS if which == 'i' else ARMS
    out = []
    for arm in arms:
        specs = arm_specs(D, arm, panel, root, a66_root, f'{PREFIX[which]}{TAG[arm]}')[:n]
        if len(specs) != n:
            raise N.Blocked(f'{arm}: {len(specs)} specs (want {n})')
        out.append((f'A89{which.upper()}_{arm}', 'episode', specs))
    if len({s['job_id'] for _, _, ss in out for s in ss}) != len(arms)*n:
        raise N.Blocked('duplicate job ids')
    return out


def _a80():
    import a80_s2b as A80            # registered A80 driver 48c5c448 (plan_checks only; its _point_d2 is NOT called)
    return A80


def mech_compare(root, a70_root, ref70_reports, a75d2_root, a66_root):
    N, _, A67, _ = B._mods()
    D2 = B._D2()
    from a32_ablation import report_of
    from tools.coopmwm import a75_holdforms as H
    root = Path(root)
    D, panel = B.setup('i', root)
    ws = dict((w, s) for w, _, s in waves(D, panel, root, a66_root, 'i'))
    worlds = [r['world_id'] for r in panel['worlds']][:IDENT_N]
    ref, ref_rep, srcs = B.stored_refs(panel, a70_root, ref70_reports, a75d2_root, a66_root, worlds)
    rows, fails = [], []
    for arm in IDENT_ARMS:
        wave = f'A89I_{arm}'
        have, _ = B._load(root, wave, ws[wave])
        if sorted(have) != sorted(worlds):
            fails.append(f'{wave}: worlds {sorted(have)}')
            continue
        for w in worlds:
            row = B.ident_row(arm, have[w], ref[arm][w], report_of(root, have[w]), ref_rep[arm][w], A67.MAPPO60C_SHA256)
            rows.append(row)
            if not row['PASS']:
                fails.append(f'{arm} {w}')
    doc = json.loads(D.bank_path(root, ARMS_BANK).read_text(encoding='utf-8'))
    plans = {}
    for arm in PLAN_ARMS:
        wave = f'A89I_{arm}'
        have, _ = B._load(root, wave, ws[wave])
        spec = H.make_core('A75_PLAN', CELL, doc, family=ARM_FAMILY[arm]).plans[0]['spec']
        out = []
        for w in worlds:
            rec = have.get(w)
            if rec is None:
                fails.append(f'{arm} {w}: missing')
                continue
            r = rec.get('a75') or {}
            errs, ex, st = _a80().plan_checks(r, spec)
            if rec.get('executor_sha256_loaded') != A67.MAPPO60C_SHA256:
                errs.append('executor is not MAPPO60C')
            if rec.get('a75_sha256_loaded') != D2.A75_SHA256:
                errs.append('a75 digest is not d1af7e0f')
            if any(int(rec['safety'].get(k) or 0) for k in D2.SAFETY_ZERO):
                errs.append('collisions / teleports')
            p = root/'evaluation'/rec.get('work_dir', rec['job_id'])/N.VEHICLE_LOG
            if st['activations']:
                if not p.is_file():
                    errs.append('vehicle log missing (zone check impossible)')
                else:
                    errs += [x for x in D2.check_zone(r, H.zone_bounds(spec), D2.vlog_pos(D2.read_vlog(p)))
                             if x != 'no activation']
            out.append(dict(world_id=w, errors=errs[:5], activations=st['activations'],
                            exercised=sorted('/'.join(map(str, t)) for t in ex)))
            if errs:
                fails.append(f'{arm} {w}: {errs[:3]}')
        plans[arm] = dict(family=ARM_FAMILY[arm], rows=out,
                          activated=any(x['activations'] for x in out) or 'NOT ACTIVATED on 996300/996301')
    if len(rows) != len(IDENT_ARMS)*IDENT_N:
        fails.append(f'{len(rows)} identity rows (want {len(IDENT_ARMS)*IDENT_N})')
    return N.seal(dict(schema='coopmwm.a89.mech.v1', amendment=AMENDMENT, rule=MECH_RULE,
                       panel_sha256=panel['panel_sha256'], worlds=worlds, stored_receipts=srcs,
                       verdict='PASS' if not fails else 'FAIL', failures=fails, identity=rows, plan_checks=plans,
                       consequence='PASS: confirm may run; FAIL: stop and report'), 'receipt_sha256')


# ------------------------------------------------------------------------------------------------ analysis

CONTRASTS = (  # (treatment, reference, kind)
    ('ROLL_BOTH', 'ROLL_L0_ONLY', 'primary_ablation'),
    ('ROLL_BOTH', 'ROLL_L2_ONLY', 'primary_ablation'),
    ('TAPER', 'MAPPO60C', 'primary_taper'),
    ('TAPER', 'DUO_HOLD_60C', 'secondary'),
    ('ROLL_BOTH', 'MAPPO60C', 'descriptive'),
    ('ROLL_L0_ONLY', 'MAPPO60C', 'descriptive'),
    ('ROLL_L2_ONLY', 'MAPPO60C', 'descriptive'),
    ('DUO_HOLD_60C', 'MAPPO60C', 'descriptive_sanity_replicate'),
)


def verdicts(rows):
    """Pure: the two registered verdicts from the _contrast rows (keyed treatment-reference)."""
    by = {f"{r['treatment']} - {r['reference']}": r for r in rows}
    a, b = by['ROLL_BOTH - ROLL_L0_ONLY'], by['ROLL_BOTH - ROLL_L2_ONLY']
    ua, ub = a['total_minus_ref']['ci95'][1], b['total_minus_ref']['ci95'][1]
    abl = dict(rule=ABLATION_RULE, both_minus_l0=dict(diff=a['total_minus_ref']['diff'], ci95=a['total_minus_ref']['ci95']),
               both_minus_l2=dict(diff=b['total_minus_ref']['diff'], ci95=b['total_minus_ref']['ci95']),
               pass_vs_l0=bool(ua < 0), pass_vs_l2=bool(ub < 0), supported=bool(ua < 0 and ub < 0),
               verdict='COOPERATION SUPPORTED' if ua < 0 and ub < 0 else 'NOT SUPPORTED')
    t = by['TAPER - MAPPO60C']
    ut = t['total_minus_ref']['ci95'][1]
    g = (t.get('groups') or {}).get('all_pass') is True
    tap = dict(rule=TAPER_RULE, diff=t['total_minus_ref']['diff'], ci95=t['total_minus_ref']['ci95'],
               delay_pass=bool(ut < 0), group_rule_pass=g, confirmed=bool(ut < 0 and g),
               verdict='CONFIRMED' if ut < 0 and g else 'NOT CONFIRMED', groups=t.get('groups'),
               worst_10pct_cvar90=t.get('tail_cvar90'))
    return abl, tap


def analyse(D, panel, root, a66_root):
    import numpy as np
    N = B._mods()[0]
    D2 = B._D2()
    root = Path(root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile') or len(worlds) != CONFIRM_N:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    if [r['world']['seed'] for r in panel['worlds']] != list(SEEDS_C):
        raise N.Blocked('not the A89 panel')
    have, sources, reports = {}, {}, {}
    for (wave, _, specs), arm in zip(waves(D, panel, root, a66_root, 'c'), ARMS):
        have[arm], sources[arm] = B._load(root, wave, specs)
        reports[arm] = D2._own_reports(root, have[arm], worlds)
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    rows = [D._contrast(have, reports, worlds, a, r, bootstrap, k) for a, r, k in CONTRASTS]
    abl, tap = verdicts(rows)
    J = {a: np.array([have[a][w]['J_veh_s'] for w in worlds], float) for a in ARMS}
    inter = N.fe.paired_stats(J['ROLL_BOTH']+J['MAPPO60C'], J['ROLL_L0_ONLY']+J['ROLL_L2_ONLY'], bootstrap, pct=False)
    arms = {a: dict(mean_J_veh_s=float(np.mean(J[a])),
                    safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
            for a in ARMS}
    for a in PLAN_ARMS:
        arms[a].update(family=ARM_FAMILY[a], activation_counts=D2.form_activations(have[a], worlds))
    arms['DUO_HOLD_60C']['activations_per_world'] = sum(
        int((have['DUO_HOLD_60C'][w].get('a37') or have['DUO_HOLD_60C'][w].get('oracle') or {}).get('activations') or 0)
        for w in worlds)/len(worlds)
    return N.seal(dict(schema='coopmwm.a89.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       disclosure=DISCLOSURE, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                       seeds=[SEEDS_C[0], SEEDS_C[-1]], bootstrap=bootstrap,
                       arms_bank=dict(file=ARMS_BANK, file_sha256=_arms_sha(), families=ARM_FAMILY),
                       arms_run=list(ARMS), arms_reused=None, pooled_with=None, root=str(root),
                       primary_ablation=abl, primary_taper=tap,
                       interaction=dict(definition=INTERACTION, **inter), contrasts=rows, arms=arms,
                       multiplicity='two registered primaries (ablation conjunction; taper); every other contrast '
                                    'descriptive; no correction',
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()},
                       label=D.RESULT_LABEL), 'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

def dry(root, a, which):
    D, panel = B.setup(which, root)
    if panel is None:
        raise SystemExit('panel missing (run panel-c)')
    ws = waves(D, panel, root, a.a66_root, which)
    out = dict(stage=which, setting=SETTINGS[which], panel_sha256=panel['panel_sha256'], n_waves=len(ws),
               waves=[(n, len(s), s[0]['arm'], s[0]['job_id'], s[-1]['job_id'], s[0]['world']['seed'],
                       s[-1]['world']['seed']) for n, _, s in ws],
               episodes=sum(len(s) for _, _, s in ws), sim_s=sum(len(s)*HORIZON for _, _, s in ws),
               stage_cap=MECH_CAP if which == 'i' else CONFIRM_CAP, charged=D.charged(root), cap=CAP_SIM_S,
               a75_sha256=__import__('tools.coopmwm.a75_holdforms', fromlist=['x']).a75_digest(),
               checkpoints={n: {k: v for k, v in s[0]['checkpoint'].items() if k != 'label'} for n, _, s in ws},
               horizon=[s[0].get('sim_s') for _, _, s in ws], windows=[s[0].get('window') for _, _, s in ws])
    fields = {}
    by = {n: s for n, _, s in ws}
    # plan arms vs ROLL_BOTH: only checkpoint a75 family / labels / job ids may differ
    base = by[f'A89{which.upper()}_ROLL_BOTH']
    for arm in PLAN_ARMS[1:]:
        fields[arm] = sorted({k for x, y in zip(by[f'A89{which.upper()}_{arm}'], base) for k in B.spec_diff(x, y)})
    fields['MAPPO60C_vs_DUO_HOLD_60C'] = sorted({k for x, y in zip(by[f'A89{which.upper()}_MAPPO60C'],
                                                                   by[f'A89{which.upper()}_DUO_HOLD_60C'])
                                                for k in B.spec_diff(x, y)})
    out['spec_diff'] = fields
    if which == 'i':
        worlds = [r['world_id'] for r in panel['worlds']][:IDENT_N]
        _, reps, srcs = B.stored_refs(panel, a.a70_root, a.ref70_reports, a.a75d2_root, a.a66_root, worlds)
        out['stored_refs'] = dict(receipts=srcs, reports_ok={k: all(x['ok'] for x in v.values()) for k, v in reps.items()})
    print(json.dumps(out, indent=1, default=str))


def main(argv=None):
    point()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'dry':
        p = argparse.ArgumentParser()
        p.add_argument('command')
        for k in ('--final-root', '--a66-root', '--a70-root', '--ref70-reports', '--a75d2-root'):
            p.add_argument(k)
        p.add_argument('--stage', choices=('i', 'c'), default='c')
        a = p.parse_args(argv)
        root = Path(a.final_root)
        if a.stage == 'c':
            register_c_setting(root)
        return dry(root, a, a.stage)
    return B.main(argv)


if __name__ == '__main__':
    main()
