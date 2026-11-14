"""A91 (POST-HOC, exploratory; D2800_p60, 60% CAVs, map wz3to2): five arms on ONE fresh 96-world panel 988600-988695.

Question (owner, 2026-11-14): can two different plans execute at the same moment? The frozen a75 bank executor
(HoldFormCellBankCore -> oracle_plan.OracleCore.step) already keeps ONE session ledger PER PLAN (own slots /
cooldown) and only forbids a CAV from being in two plans' sessions, so two plans of one bank may be active together on
disjoint CAVs. BANK_TURNS runs the L2 hold and the TAPER plan as such a two-plan bank, both rows unchanged; whether and
how often they overlap is reported (receipt-derived concurrency), not assumed. A one-plan "union" of the two plans is
NOT executable without a change (the union has two lane-2 rank-0 roles -> select_members never fills it), so there is
no CONCURRENT arm (PREP_NOTES.md).

THIN WRAPPER over the registered A81 driver ops/a81_bigconfirm.py (c755a551, imported UNCHANGED), the A89 pattern:
its module names are re-pointed at A91 names. A91's own functions: core / register_c_setting / _runner_check (labels,
block, HOST_BASE), prepare (three pinned bank copies), waves, mech_compare (A81 identity rows + identity of the two
A89 plan arms vs the stored A89 mechanics records + per-plan rule checks of the bank arm), analyse (A91 contrasts,
every statistic from the a42 ``_contrast`` / nm_finaleval code A81 used), dry, and the analyse CLI with its own
summary line (the A81 summary line reads a 'primary' key the A89 results did not have; A89's chain stopped there).

Arms (all on the MAPPO60C executor 2abed317, same plant / setting as A81 / A89; diff = treatment - reference):
  L2_ONLY      A75_PLAN, family a89l2_d86b1d6ac653e8ad of the A89 arms bank de9a30a9 (byte copy; = A89 ROLL_L2_ONLY)
  BANK_TURNS   A75_BANK on the two-row bank A91_TURNS (rows a89l2_d86b1d6ac653e8ad + mut_1fca373046a3c66e copied
               unchanged from de9a30a9); executor priority: L2 row first (carried lb95 2917.05 > 0.0)
  ROLL_BOTH    A75_PLAN, family d2_d86b1d6ac653e8ad of de9a30a9 (= A89 ROLL_BOTH)
  DUO_HOLD_60C A81 arm unchanged;  MAPPO60C  A81 arm unchanged

Mechanics (A70 worlds 996300 / 996301, setting a70c, 12 episodes = 14,400 sim-s):
  identity A81: MAPPO60C, DUO_HOLD_60C and the A75-D2 bank (A75_BANK on c325ec1d; same executor path as BANK_TURNS)
            equal the stored A70 / A75-D2 records (A81 ident_row unchanged);
  identity A89: L2_ONLY and ROLL_BOTH equal the stored A89 mechanics records (WAVE_A89I_ROLL_L2_ONLY / _ROLL_BOTH,
            receipts pinned) - same fields as ident_row; the stored side's report is recomputed read-only from its log;
  BANK_TURNS: per plan (acts / events split by family) a80_s2b.plan_checks, executor MAPPO60C, a75 digest d1af7e0f,
            every activation inside the zone +-1 m, no CAV in two sessions at once, collisions = teleports = 0.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import a81_bigconfirm as B  # noqa: E402  (registered A81 driver, unchanged; names re-pointed below)

AMENDMENT = '91'
CELL = B.CELL
PHASE = 'combine_a91'
HORIZON = B.HORIZON
IDENT_N, CONFIRM_N = 2, 96
BLOCK = (988600, 988699)                      # 988600-988695 used; 988696-988699 reserved unused
SEEDS_C = tuple(range(BLOCK[0], BLOCK[0]+CONFIRM_N))
RESERVED_C = (BLOCK[0]+CONFIRM_N, BLOCK[1])
assert SEEDS_C[-1] == 988695
PLAN_ARMS = ('L2_ONLY', 'ROLL_BOTH')
BANK_ARMS = ('BANK_TURNS',)
REF_ARMS = ('DUO_HOLD_60C', 'MAPPO60C')
ARMS = ('L2_ONLY', 'BANK_TURNS', 'ROLL_BOTH')+REF_ARMS
IDENT_ARMS = (B.BANK_ARM, 'DUO_HOLD_60C', 'MAPPO60C')     # the A81 identity arms (stored A70 / A75-D2 records)
MECH_ARMS = IDENT_ARMS+PLAN_ARMS+BANK_ARMS
MECH_EPISODES = len(MECH_ARMS)*IDENT_N
MECH_CAP, CONFIRM_CAP = MECH_EPISODES*HORIZON, len(ARMS)*CONFIRM_N*HORIZON
CAP_SIM_S = MECH_CAP+CONFIRM_CAP
assert (MECH_CAP, CONFIRM_CAP, CAP_SIM_S) == (14_400, 576_000, 590_400)
SETTINGS = dict(i=B.SETTINGS['i'], c='a91c_D2800_p60')
RANGES = dict(i=B.RANGES['i'], c='A91_RANGES')
BANK_DIR = 'a91_banks'
ARMS_BANK = f'A89_ARMS_{CELL}.json'           # byte copy of the pinned A89 arms bank
ARMS_BANK_SHA256 = 'de9a30a944eda470777cbd3247c6ca81db2745547549cde3e2bff154e23824c2'
TURNS_BANK = f'A91_TURNS_{CELL}.json'
TURNS_BANK_SHA256 = os.environ.get('A91_TURNS_SHA256_PIN', 'SET_AT_SHIP')
L2_FAMILY, TAPER_FAMILY, ROLL_FAMILY = 'a89l2_d86b1d6ac653e8ad', 'mut_1fca373046a3c66e', 'd2_d86b1d6ac653e8ad'
ARM_FAMILY = dict(L2_ONLY=L2_FAMILY, ROLL_BOTH=ROLL_FAMILY)
BANK_FAMILIES = dict(BANK_TURNS=(L2_FAMILY, TAPER_FAMILY))
#: A89 mechanics records the two plan arms must reproduce (wave receipt sha256 pinned at prep, 2026-11-14)
A89_ROOT = os.environ.get('A91_A89_ROOT', '/root/a89/root')
A89_REF = dict(L2_ONLY=('A89I_ROLL_L2_ONLY__0000-0001',
                        '30269ad414e2610f2ef39796d4ddca2a32dd75a20966b7a5deed05e3831e2e95'),
               ROLL_BOTH=('A89I_ROLL_BOTH__0000-0001',
                          '1699e04801601c73ae5ec5ed46930f3a760291cd97d2bc7318f5080c019ec6b5'))
PREFIX = dict(i='a91i_', c='a91c_')
TAG = {B.BANK_ARM: 'bank', 'DUO_HOLD_60C': 'duo_hold_60c', 'MAPPO60C': 'mappo60c', 'L2_ONLY': 'l2_only',
       'ROLL_BOTH': 'roll_both', 'BANK_TURNS': 'bank_turns'}
AUTHORITY = ('MAIN AMENDMENT 91 (A91, 2026-11-14; POST-HOC, exploratory): can two plans run at the same moment - '
             'L2_ONLY (A89 ROLL_L2_ONLY) vs the two-plan bank {L2, TAPER} under the unchanged executor arbitration '
             '(BANK_TURNS), ROLL_BOTH, DUO_HOLD_60C and MAPPO60C on one fresh 96-world D2800_p60 panel 988600-988695; '
             'phase combine_a91, caps mech 14,400 + confirm 576,000')
DISCLOSURE = ('POST-HOC and exploratory. L2_ONLY was chosen as the best A89 arm on the A89 panel (988400-988495); '
              'BANK_TURNS combines it with the A89 TAPER plan by bank file only (rows unchanged; executor priority '
              'from the carried lb95 fields). No one-plan union arm: not executable without a code change. Fresh '
              'worlds; never pooled with any other panel; arms and plans frozen (no retuning)')
PRIMARY_RULE = ('SUPPORTED iff L2_ONLY - DUO_HOLD_60C total delay CI95 upper < 0 (paired percentile bootstrap 10,000, '
                'seed 20261104, n = 96) AND the five-group 10 s rule relative to DUO_HOLD_60C '
                '(nm_finaleval._bank_compare groups.all_pass, reference = DUO_HOLD_60C, as the A81 primary); '
                'diff = treatment - reference, negative = less delay')
MECH_RULE = ('PASS iff (a) MAPPO60C, DUO_HOLD_60C and A75D2_BANK_60C on 996300 / 996301 equal the stored A70 / A75-D2 '
             'records (A81 ident_row, unchanged); (b) L2_ONLY and ROLL_BOTH on 996300 / 996301 equal the stored A89 '
             'mechanics records (A89I_ROLL_L2_ONLY / A89I_ROLL_BOTH receipts 30269ad4 / 1699e048): J_veh_s, safety, '
             'world, executor MAPPO60C, a75 digest d1af7e0f, executor receipts (activations, activations_by_family, '
             'issued, acts) and the per-vehicle report equal; (c) BANK_TURNS on 996300 / 996301: executor MAPPO60C, '
             'a75 digest d1af7e0f, per plan no receipt invariant violation (a80_s2b.plan_checks on that plan\'s acts '
             '/ events), every activation inside the zone +-1 m, no CAV in two sessions at once, collisions = '
             'teleports = 0. A plan that never activates on those worlds is reported (not a failure)')

# ------------------------------------------------------------------------------------------------ re-pointing
_ORIG = {}


def point():
    """Re-point the a81_bigconfirm module names (its functions read them at call time). Idempotent."""
    if _ORIG:
        return
    names = dict(AMENDMENT=AMENDMENT, PHASE=PHASE, IDENT_N=IDENT_N, CONFIRM_N=CONFIRM_N, ARMS=ARMS,
                 MECH_EPISODES=MECH_EPISODES, MECH_CAP=MECH_CAP, CONFIRM_CAP=CONFIRM_CAP, CAP_SIM_S=CAP_SIM_S,
                 BLOCK=BLOCK, SEEDS_C=SEEDS_C, RESERVED_C=RESERVED_C, SETTINGS=SETTINGS, RANGES=RANGES,
                 PANEL_C='A91_PANEL_C.json', PANEL_C_PIN='A91_PANEL_C.sha256', BANK_DIR=BANK_DIR, PREFIX=PREFIX,
                 TAG=TAG, STAGE_CAP=dict(mech=MECH_CAP, confirm=CONFIRM_CAP), MECH_FILE='A91_MECH.json',
                 RESULTS_FILE=f'A91_RESULTS_{CELL}.json', LOG_FILE='A91_LOG.jsonl',
                 DROP_MANIFEST='A91_DROPPED_RAW.jsonl', AUTHORITY=AUTHORITY, DISCLOSURE=DISCLOSURE,
                 MECH_RULE=MECH_RULE, PRIMARY_RULE=PRIMARY_RULE,
                 DROP_RULE=B.DROP_RULE.replace('a81c_*', 'a91c_*'),
                 core=core, register_c_setting=register_c_setting, _runner_check=_runner_check, prepare=prepare,
                 waves=waves, mech_compare=mech_compare, analyse=analyse)
    for k, v in names.items():
        _ORIG[k] = getattr(B, k)
        setattr(B, k, v)


def core(which):
    """B.core with the A91 labels / phase / caps / HOST_BASE (A91_HOST_BASE)."""
    N, A50 = B._mods()[:2]
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A91',
        HOST_BASE=os.environ.get('A91_HOST_BASE', '/root/a91'),
        VERIFY_SETTING=SETTINGS[which], CONFIRM_SETTING=SETTINGS[which],
        BANK_DIR=BANK_DIR, IDENT_WAVE='A91_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A91', JOB_PREFIX='a91',
        VERIFY_TAG='a91v', CONFIRM_TAG=f'a91{which}', IDENT_TAG='a91i',
        VERIFY_WAVE_PREFIX='A91V_', CONFIRM_WAVE_PREFIX='A91C_',
        RESULTS_FILE=B.RESULTS_FILE, LOG_FILE=B.LOG_FILE, SCHEMA_PREFIX='coopmwm.a91',
        RUNNER_RANGES_ATTR=RANGES[which],
        RESULT_LABEL='amendment 91; POST-HOC exploratory five-arm fresh panel (L2 hold alone vs two-plan bank with '
                     'the taper plan)',
        AUTHORITY=AUTHORITY)
    m = A50._clone(f'a91_core_{which}', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (SETTINGS[which],)
    m.A42Runtime.phase = PHASE
    N.CHECKS['episode'] = B._D2().check_d2
    return m


def register_c_setting(root):
    """B.register_c_setting with A91 block / size / labels (the a70c setting otherwise)."""
    N = B._mods()[0]
    name = SETTINGS['c']
    if name not in N.SETTINGS:
        lo, hi = BLOCK
        N.SETTINGS[name] = dict(
            N.SETTINGS[SETTINGS['i']], N_WORLDS=CONFIRM_N, EVAL_RANGE=BLOCK, EVAL_EXTRA_RANGE=BLOCK,
            SEEDS_RESERVED=RESERVED_C,
            PANEL_SIZE_REASON=(f'MAIN amendment 91: {CONFIRM_N} fresh worlds {lo}-{lo+CONFIRM_N-1} (D2800_p60); '
                               f'{RESERVED_C[0]}-{hi} reserved unused'),
            CELL_AUTHORITY='MAIN amendment 91 (A91, post-hoc): D2800_p60 five-arm fresh panel',
            PANEL_AUTHORITY=(f'A91 amendment 91: fresh audited integers {lo}-{lo+CONFIRM_N-1} (D2800_p60) for A91 ONLY; '
                             'executed through the eval runner (scope label only); never pooled'),
            EXPOSURE_EXTRA=dict(
                checked=(f'amendment-91 audit: {lo}-{hi} absent from REGISTRATION.md, protocols/experiment_history.json, '
                         'every panel file / runner range table / code under /root and /home/ubuntu on the run host '
                         '(SEED_AUDIT_A91.json)',),
                disclosed='; amendment 91: fresh integers, registered at runtime by the runner'),
            PANEL_FILE=Path(root)/B.PANEL_C, PANEL_3C_FILE=None)
        N.BANK_SETTINGS = N.BANK_SETTINGS+(name,)
        N.SETTINGS_PINS[name] = dict(PANEL_SHA256=None, PANEL_3C_SHA256=None)
    return name


def _runner_check():
    N = B._mods()[0]
    from tools.coopmwm import nm_eval_runner as R
    if tuple(getattr(R, 'A91_RANGES', ())) != (BLOCK,):
        raise N.Blocked(f'runner A91_RANGES is not {BLOCK} (patch_tables_a91.py)')
    B.block_free(BLOCK, R.EVAL_HELDOUT_RANGES)
    return R


def _turns_sha():
    s = os.environ.get('A91_TURNS_SHA256') or TURNS_BANK_SHA256
    if len(s) != 64:
        raise SystemExit('A91_TURNS_SHA256 (the registered BANK_TURNS bank sha256) is required')
    return s


def prepare(D, root, a75d2_root):
    """Write-once byte copies: the frozen A75-D2 bank (identity arm), the A89 arms bank and the A91 turns bank."""
    N = B._mods()[0]
    from tools.coopmwm import a75_holdforms as H
    out = {}
    for src, name, pin in ((Path(a75d2_root)/'a75d2_banks'/B.BANK_FILE, B.BANK_FILE, B.BANK_SHA256),
                           (HERE/'a91'/ARMS_BANK, ARMS_BANK, ARMS_BANK_SHA256),
                           (HERE/'a91'/TURNS_BANK, TURNS_BANK, _turns_sha())):
        if N.file_sha(src) != pin:
            raise N.Blocked(f'{src}: not {pin[:8]}')
        q = D.C37.write_once(D.bank_path(root, name), src.read_bytes().decode('utf-8'))
        if N.file_sha(q) != pin or q.read_bytes() != src.read_bytes():
            raise N.Blocked(f'{q}: not the pinned bytes')
        out[name] = N.file_sha(q)
    doc = json.loads(D.bank_path(root, ARMS_BANK).read_text(encoding='utf-8'))
    for arm, fam in ARM_FAMILY.items():
        H.make_core('A75_PLAN', CELL, doc, family=fam)          # raises on any refusal
    turns = json.loads(D.bank_path(root, TURNS_BANK).read_text(encoding='utf-8'))
    c = H.make_core('A75_BANK', CELL, turns)
    if sorted(p['family'] for p in c.plans) != sorted(BANK_FAMILIES['BANK_TURNS']):
        raise N.Blocked(f'{TURNS_BANK}: plans {[p["family"] for p in c.plans]}')
    out['families'] = ARM_FAMILY
    out['bank_turns_priority'] = [p['family'] for p in c.plans]
    return out


def arm_specs(D, arm, panel, root, a66_root, tag):
    _, _, A67, A68 = B._mods()
    D2 = B._D2()
    exe = A68.executor(a66_root)
    if arm in PLAN_ARMS:            # = a89_sixarm.plan_specs (A75_PLAN on the pinned arms bank)
        entry = A68.bank_entry(D, root, ARMS_BANK, ARMS_BANK_SHA256)
        fam = ARM_FAMILY[arm]
        base = A67.with_exec(D._a37_specs(D.RUNNER_ARM['PLAN'], panel, root, entry, family=fam, tag=tag), exe, tag)
        return D2.a75_from_a37(base, 'A75_PLAN', entry, family=fam, tag=tag)
    if arm in BANK_ARMS:            # = the A81 bank arm (A75_BANK) on the turns bank
        entry = A68.bank_entry(D, root, TURNS_BANK, _turns_sha())
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
        out.append((f'A91{which.upper()}_{arm}', 'episode', specs))
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
    """Pure, descriptive: per-family active seconds (a plan is active while any member of one of its activations is
    in session), seconds with >= 2 families active, activations started while another family was active, and CAVs
    found in two sessions at overlapping times (must be 0: the executor forbids it)."""
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
                cav_in_two_sessions=clash)


def primary_verdict(rows):
    """Pure: the registered primary from the _contrast rows (keyed 'treatment - reference')."""
    by = {f"{r['treatment']} - {r['reference']}": r for r in rows}
    p = by['L2_ONLY - DUO_HOLD_60C']
    u = p['total_minus_ref']['ci95'][1]
    g = (p.get('groups') or {}).get('all_pass') is True
    return dict(rule=PRIMARY_RULE, contrast='L2_ONLY - DUO_HOLD_60C', group_rule_reference='DUO_HOLD_60C',
                diff=p['total_minus_ref']['diff'], ci95=p['total_minus_ref']['ci95'], delay_pass=bool(u < 0),
                group_rule_pass=g, supported=bool(u < 0 and g),
                verdict='SUPPORTED' if u < 0 and g else 'NOT SUPPORTED', groups=p.get('groups'),
                worst_10pct_cvar90=p.get('tail_cvar90'))


CONTRASTS = (  # (treatment, reference, kind)
    ('L2_ONLY', 'DUO_HOLD_60C', 'primary'),
    ('BANK_TURNS', 'L2_ONLY', 'secondary'),
    ('L2_ONLY', 'ROLL_BOTH', 'secondary_replication_a89'),
    ('L2_ONLY', 'MAPPO60C', 'descriptive'),
    ('BANK_TURNS', 'MAPPO60C', 'descriptive'),
    ('ROLL_BOTH', 'MAPPO60C', 'descriptive'),
    ('DUO_HOLD_60C', 'MAPPO60C', 'descriptive'),
)


def summary(v):
    """The one-line console summary (A91 keys; never the A81 'primary' layout)."""
    p = v['primary']
    return dict(analysis_sha256=v['analysis_sha256'],
                primary={k: p[k] for k in ('contrast', 'diff', 'ci95', 'delay_pass', 'group_rule_pass', 'verdict')},
                secondary=[dict(contrast=f"{r['treatment']} - {r['reference']}", diff=r['total_minus_ref']['diff'],
                                ci95=r['total_minus_ref']['ci95']) for r in v['contrasts']
                           if r['kind'].startswith('secondary')],
                bank_turns_concurrent_s=v['arms']['BANK_TURNS']['overlap']['concurrent_s'])


# ------------------------------------------------------------------------------------------------ mechanics

def a89_refs(worlds, a89_root=None):
    """Stored A89 mechanics records of L2_ONLY / ROLL_BOTH + their per-vehicle reports, read-only (the report is
    recomputed from the sealed log with nm_finaleval.vehicle_log_report; nothing is written under the A89 root)."""
    N = B._mods()[0]
    root = Path(a89_root or A89_ROOT)
    have, reps, srcs = {}, {}, {}
    for arm, (wave, pin) in A89_REF.items():
        p = root/f'WAVE_{wave}.json'
        v = N.verify(N.read(p), 'receipt_sha256', p.name)
        if v['receipt_sha256'] != pin:
            raise N.Blocked(f'{p}: not the pinned A89 receipt {pin[:8]}')
        by = {r['world_id']: r for r in v['records']}
        have[arm], reps[arm] = {}, {}
        for w in worlds:
            rec = by[w]
            log = root/'evaluation'/rec.get('work_dir', rec['job_id'])/N.VEHICLE_LOG
            if N.file_sha(log) != rec['vehicle_log']['sha256']:
                raise N.Blocked(f'{log}: not the sealed A89 vehicle log')
            have[arm][w] = rec
            reps[arm][w] = N.vehicle_log_report(log, t0=N.WARMUP_S, t1=N.HORIZON_S)
        srcs[arm] = v['receipt_sha256']
    return have, reps, srcs


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
    r89, rep89, src89 = a89_refs(worlds)
    rows, fails = [], []
    for arm in IDENT_ARMS+PLAN_ARMS:
        wave = f'A91I_{arm}'
        have, _ = B._load(root, wave, ws[wave])
        if sorted(have) != sorted(worlds):
            fails.append(f'{wave}: worlds {sorted(have)}')
            continue
        for w in worlds:
            if arm in PLAN_ARMS:
                row = B.ident_row(arm, have[w], r89[arm][w], report_of(root, have[w]), rep89[arm][w],
                                  A67.MAPPO60C_SHA256)
                row['a75_sha256_loaded'] = have[w].get('a75_sha256_loaded')
                row['stored'] = 'A89 mechanics'
                row['PASS'] = bool(row['PASS'] and have[w].get('a75_sha256_loaded') == D2.A75_SHA256
                                   == r89[arm][w].get('a75_sha256_loaded'))
            else:
                row = B.ident_row(arm, have[w], ref[arm][w], report_of(root, have[w]), ref_rep[arm][w],
                                  A67.MAPPO60C_SHA256)
            rows.append(row)
            if not row['PASS']:
                fails.append(f'{arm} {w}')
    turns = json.loads(D.bank_path(root, TURNS_BANK).read_text(encoding='utf-8'))
    plans = {p['family']: p for p in H.a75_bank_plans(turns, CELL)[0]}
    zones = {H.zone_bounds(p['spec']) for p in plans.values()}
    if len(zones) != 1:
        fails.append(f'BANK_TURNS plans have different zones {zones}')
    bank = {}
    for arm in BANK_ARMS:
        wave = f'A91I_{arm}'
        have, _ = B._load(root, wave, ws[wave])
        out = []
        for w in worlds:
            rec = have.get(w)
            if rec is None:
                fails.append(f'{arm} {w}: missing')
                continue
            r = rec.get('a75') or {}
            errs, per = [], {}
            for fam in BANK_FAMILIES[arm]:
                e, ex, st = _a80().plan_checks(split_family(r, fam), plans[fam]['spec'])
                errs += [f'{fam}: {x}' for x in e]
                per[fam] = dict(activations=st['activations'], events=st.get('events'),
                                exercised=sorted('/'.join(map(str, t)) for t in ex))
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
            p = root/'evaluation'/rec.get('work_dir', rec['job_id'])/N.VEHICLE_LOG
            if int(r.get('activations') or 0):
                if not p.is_file():
                    errs.append('vehicle log missing (zone check impossible)')
                else:
                    errs += [x for x in D2.check_zone(r, next(iter(zones)), D2.vlog_pos(D2.read_vlog(p)))
                             if x != 'no activation']
            out.append(dict(world_id=w, errors=errs[:5], per_family=per, overlap=ov))
            if errs:
                fails.append(f'{arm} {w}: {errs[:3]}')
        bank[arm] = dict(families=list(BANK_FAMILIES[arm]), rows=out,
                         activated={f: any(x['per_family'].get(f, {}).get('activations') for x in out)
                                    or 'NOT ACTIVATED on 996300/996301' for f in BANK_FAMILIES[arm]})
    if len(rows) != len(IDENT_ARMS+PLAN_ARMS)*IDENT_N:
        fails.append(f'{len(rows)} identity rows (want {len(IDENT_ARMS+PLAN_ARMS)*IDENT_N})')
    return N.seal(dict(schema='coopmwm.a91.mech.v1', amendment=AMENDMENT, rule=MECH_RULE,
                       panel_sha256=panel['panel_sha256'], worlds=worlds, stored_receipts=dict(srcs, **src89),
                       verdict='PASS' if not fails else 'FAIL', failures=fails, identity=rows, bank_checks=bank,
                       consequence='PASS: confirm may run; FAIL: stop and report'), 'receipt_sha256')


# ------------------------------------------------------------------------------------------------ analysis

def analyse_core(D, have, reports, worlds, bootstrap, *, sources, panel_sha256, root, seeds):
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
    for a in PLAN_ARMS:
        arms[a].update(family=ARM_FAMILY[a], activation_counts=D2.form_activations(have[a], worlds))
    for a in BANK_ARMS:
        by_fam, offers, tot = Counter(), Counter(), Counter()
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
        arms[a].update(families=list(BANK_FAMILIES[a]), activation_counts=D2.form_activations(have[a], worlds),
                       activations_by_family=dict(by_fam), offers_by_family_outcome=dict(offers),
                       overlap=dict(tot, worlds_with_overlap=sum(1 for v in per_world.values() if v > 0),
                                    definition=overlap_stats.__doc__.strip()))
    arms['DUO_HOLD_60C']['activations_per_world'] = sum(
        int((have['DUO_HOLD_60C'][w].get('a37') or have['DUO_HOLD_60C'][w].get('oracle') or {}).get('activations') or 0)
        for w in worlds)/len(worlds)
    return N.seal(dict(schema='coopmwm.a91.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       disclosure=DISCLOSURE, panel_sha256=panel_sha256, n_worlds=len(worlds), seeds=seeds,
                       bootstrap=bootstrap,
                       banks=dict(arms=dict(file=ARMS_BANK, file_sha256=ARMS_BANK_SHA256, families=ARM_FAMILY),
                                  turns=dict(file=TURNS_BANK, file_sha256=_turns_sha(), families=BANK_FAMILIES)),
                       arms_run=list(ARMS), arms_reused=None, pooled_with=None, root=str(root),
                       primary=prim, contrasts=rows, arms=arms, concurrent_arm=None,
                       concurrent_arm_note='no one-plan union arm: not executable without a code change',
                       multiplicity='one registered primary; secondaries and every other contrast descriptive; '
                                    'no correction',
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
        raise N.Blocked('not the A91 panel')
    have, sources, reports = {}, {}, {}
    for (wave, _, specs), arm in zip(waves(D, panel, root, a66_root, 'c'), ARMS):
        have[arm], sources[arm] = B._load(root, wave, specs)
        reports[arm] = D2._own_reports(root, have[arm], worlds)
    return analyse_core(D, have, reports, worlds, bootstrap, panel_sha256=panel['panel_sha256'], root=root,
                        seeds=[SEEDS_C[0], SEEDS_C[-1]],
                        sources={k: v.get('receipt_sha256') for k, v in sources.items()})


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
    by = {n: s for n, _, s in ws}
    W = which.upper()
    base = by[f'A91{W}_L2_ONLY']
    fields = {arm: sorted({k for x, y in zip(by[f'A91{W}_{arm}'], base) for k in B.spec_diff(x, y)})
              for arm in ('ROLL_BOTH', 'BANK_TURNS')}
    fields['MAPPO60C_vs_DUO_HOLD_60C'] = sorted({k for x, y in zip(by[f'A91{W}_MAPPO60C'], by[f'A91{W}_DUO_HOLD_60C'])
                                                for k in B.spec_diff(x, y)})
    out['spec_diff_vs_L2_ONLY'] = fields
    if which == 'i':
        worlds = [r['world_id'] for r in panel['worlds']][:IDENT_N]
        _, reps, srcs = B.stored_refs(panel, a.a70_root, a.ref70_reports, a.a75d2_root, a.a66_root, worlds)
        r89, rep89, src89 = a89_refs(worlds)
        out['stored_refs'] = dict(receipts=dict(srcs, **src89),
                                  reports_ok={k: all(x['ok'] for x in v.values()) for k, v in {**reps, **rep89}.items()},
                                  a89_worlds={k: sorted(v) for k, v in r89.items()})
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
                raise N.Blocked('the A91 final root is a new root')
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
