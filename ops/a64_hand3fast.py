"""AMENDMENT 64 (S-HAND3FAST60; registered 2026-11-11 before any outcome of this test): HAND3 with a faster lane-2 hold.

Plans (fixed in the registration): HAND3_F23 / HAND3_F26 = the A59 HAND3 plan file (a59_banks/A59_HAND3_D2800_p60.json,
sha256 498a5206...) with ONLY the SUMO-lane-2 role's v_w changed to 23.0 / 26.0 (lanes 0 and 1 stay at 20.0). The
template's strategy_key (and the row's copy of it) is a content hash of the spec that the executor re-derives and
requires equal (oracle_plan refusal 'strategy_key_mismatch'); it is recomputed and is the only other change (derived,
disclosed in the build note). Family name, row, trigger, zone, spread, cooldown, slots, dmax, hold, onset unchanged.

Arms (the 96 A42 confirmation worlds 998600-998695, setting a42c_D2800_p60; each plan as a ONE-ROW CELL_BANK, the A62 /
A63 path, so that the A57/A63 lane-set filter applies; the runner allows the filter only with CELL_BANK):
  HAND3_F23     F23 bank, no filter          F23_PART_L2   F23 bank, only the planned lane-2 member commanded
  HAND3_F26     F26 bank, no filter          F26_PART_L2   F26 bank, only the planned lane-2 member commanded
Identity: the UNCHANGED A59 HAND3 bank (byte copy) run as a one-row CELL_BANK on panel world index 0 must reproduce the
stored A59 confirmation HAND3 record (A59 ran A37_PLAN; a one-row CELL_BANK is CellBankCore over the same rows):
J difference 0, identical safety counters, identical activation receipts, same world, same a37 digest. Stored U0 /
BIG_BANK / DUO_HOLD (A42 root) and HAND3 (A59 root) are then reused read-only.

This driver IS ops/a42_bigp60.py (private module copy, as A50 / A56 / A57 / A63) with the A64 names: final root
/root/a64/root, ledger phase hand3fast_a64 cap 480,000 sim-s, job ids a64*, stop file STOP_A64. No automatic replay.

Commands (python ops/a64_hand3fast.py <cmd> --final-root <root>):
  prepare           zero physics: copy the pinned plan files (ops/a64/plans + the A59 source bytes) into a64_banks,
                    check the shas and that the executor accepts each bank with the registered role speeds
  dry               spec summary + budget
  identity          1 episode: A59 HAND3 bytes as one-row CELL_BANK, filter OFF, panel world index 0
  identity-compare  -> IDENTITY_A64.json
  smoke             world index 0 of each arm (4 episodes; first chunk of the waves)
  smoke-check       -> SMOKE_A64.json: receipts + observable log: every plan command implies the planned role's v_w
                    (requested = clip(v_w - v, -dmax, 1)); the lane-2 member at 23 / 26, lanes 0/1 at 20; PART_L2 arms
                    command only planned lane-2 members
  run               the 4 x 96 waves (refused unless identity PASS and smoke PASS)
  analyse           P1 / P2 / P3 per variant + secondaries -> HAND3FAST.json (+ per-world CSV) in --out
  status / budget
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import nm_finaleval as N  # noqa: E402
import a50_fullp60 as A50  # noqa: E402   (only for its _clone helper)

AMENDMENT = '64'
CELL = 'D2800_p60'
SETTING = 'a42c_D2800_p60'
N_WORLDS = 96
CAP_SIM_S = 480_000
A42_ROOT = '/root/a42v/root'
A59_ROOT = '/root/a59/root'
FAMILY = 'HAND3'
SRC_PLAN = f'A59_HAND3_{CELL}.json'
SRC_PLAN_SHA256 = '498a520629fc5305d63f4e9696a4119afab04ab345fba8fa015e33797fa4e90c'
#: variant -> (plan file, lane-2 v_w); plan shas pinned after `build` (recorded in REGISTRATION.md before any SUMO)
VARIANTS = dict(F23=(f'A64_HAND3_F23_{CELL}.json', 23.0), F26=(f'A64_HAND3_F26_{CELL}.json', 26.0))
PLAN_SHA256 = dict(F23='69d9f9cce304d472ebedc9ec60999501a3c91e03ee9c8bd4a00b5e586de8842b',
                   F26='f17311747fdc3f81db410a7e6bd756b7d0e797ddf3b8a17c583abb1f2b4695bb')
FAST_LANE = 2
BASE_VW = 20.0
#: arm -> (variant, member lane set or None = no filter)
ARMS = dict(HAND3_F23=('F23', None), HAND3_F26=('F26', None), F23_PART_L2=('F23', [2]), F26_PART_L2=('F26', [2]))
NEW_ARMS = tuple(ARMS)
FULL_OF = dict(F23='HAND3_F23', F26='HAND3_F26')
PART_OF = dict(F23='F23_PART_L2', F26='F26_PART_L2')
STORED_A42 = dict(U0='U0', BIG_BANK='BIG_BANK', DUO_HOLD='DUO_HOLD')   # A64 name -> A42 confirm arm
STORED_A59 = ('HAND3',)
BOOTSTRAP = dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')
AUTHORITY = ('MAIN AMENDMENT 64 S-HAND3FAST60 (owner request, post hoc, disclosed): HAND3_F23, HAND3_F26, F23_PART_L2, '
             'F26_PART_L2 on the 96 A42 confirmation worlds, paired with the stored A42 U0 / BIG_BANK / DUO_HOLD and '
             'the stored A59 HAND3 runs; phase hand3fast_a64 cap 480,000')
RULES = dict(
    P1='v - U0 total delay: SAVES if CI95 upper < 0, else NOT SHOWN',
    P2=('v - HAND3 total delay (CI95) and the lane-2 group (origin:L2) mean increase vs U0 (CI95); whether every '
        'registered group is within 10 s and within 5 s of U0 (point estimate and CI95 upper)'),
    P3=('v - v_PART_L2 total delay, Holm over the two variants (smaller bootstrap p tested with the 97.5 % CI, the '
        'other with the 95 % CI only if the first rejects): OTHER LANES HELP if CI upper < 0; HURT if CI lower > 0; '
        'else NOT RESOLVED'),
    bootstrap='paired percentile bootstrap 10,000, seed 20261104; diff = treatment - reference; negative = less delay')
OVERRIDES = dict(
    A42_PHASE='hand3fast_a64', A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A64',
    HOST_BASE=os.environ.get('A64_HOST_BASE', '/root/a64'),
    BANK_DIR='a64_banks', IDENT_WAVE='A64_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A64', JOB_PREFIX='a64',
    VERIFY_TAG='a64v', CONFIRM_TAG='a64c', IDENT_TAG='a64i', VERIFY_WAVE_PREFIX='A64V_', CONFIRM_WAVE_PREFIX='A64C_',
    CANDS_BANK_FILE=f'A64_UNUSED_CANDS_{CELL}.json', BIG_BANK_FILE=f'A64_UNUSED_BIG_{CELL}.json',
    ADMISSION_FILE=f'ADMISSION_A64_UNUSED_{CELL}.json', RESULTS_FILE='HAND3FAST.json', LOG_FILE='A64_LOG.jsonl',
    SCHEMA_PREFIX='coopmwm.a64', RUNNER_RANGES_ATTR='A42_RANGES',
    RESULT_LABEL='amendment 64; HAND3 with a faster lane-2 hold at 60% CAVs; one shot', AUTHORITY=AUTHORITY)


def _build():
    m = A50._clone('a64_hand3fast_core', HERE/'a42_bigp60.py', OVERRIDES)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (m.VERIFY_SETTING, m.CONFIRM_SETTING)
    m.A42Runtime.phase = m.A42_PHASE
    return m


D = _build()                       # A42 driver code, A64 names (own root / phase / cap / ids / stop file)
D42 = A50._clone('a64_a42_view', HERE/'a42_bigp60.py', {})   # the A42 driver itself (read-only use)
IDENT_WAVE = 'A64_IDENT_HAND3'
IDENTITY_FILE = 'IDENTITY_A64.json'
SMOKE_FILE = 'SMOKE_A64.json'
PLANS_DIR = HERE/'a64'/'plans'


def wave(arm):
    return f'{D.CONFIRM_WAVE_PREFIX}{arm}'


def _a59():
    """The A59 driver (ops/a59_hand.py, byte copy of the A59 code 332c35b5) for the stored HAND3 specs (read-only)."""
    import a59_hand
    return a59_hand


# ------------------------------------------------------------------------------------------------ plan files

def variant_doc(src_doc, v_w):
    """Pure (unit-tested): the A59 HAND3 bank doc with the SUMO-lane-2 role's v_w = ``v_w`` and the derived
    strategy_key (template.source + row) recomputed; nothing else changes."""
    from tools.coopmwm import oracle_plan as O
    doc = deepcopy(src_doc)
    (tpl,), (row,) = doc['clean5c']['templates'], doc['rows']
    roles = [r for r in tpl['offer_condition']['roles'] if int(r['sumo_lane']) == FAST_LANE]
    if len(roles) != 1 or tpl['family'] != FAMILY or row['family'] != FAMILY:
        raise N.Blocked('source is not the one-row HAND3 plan with one lane-2 role')
    roles[0]['v_w'] = float(v_w)
    key = O.strategy_key(O.spec_of(FAMILY, tpl['offer_condition']))
    tpl['template']['source']['strategy_key'] = key
    row['strategy_key'] = key
    return doc


def json_diff(a, b, path=''):
    """[(path, a, b)] of differing leaves (pure)."""
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return [d for k in sorted(a) for d in json_diff(a[k], b[k], f'{path}/{k}')]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in json_diff(x, y, f'{path}/{i}')]
    return [] if a == b else [(path, a, b)]


def role_speeds(doc):
    """{sumo_lane: v_w} of the executor's accepted plan (CELL_BANK one row)."""
    from tools.coopmwm import a37_cellbank as C
    core = C.make_core('CELL_BANK', CELL, doc)
    if [p['family'] for p in core.plans] != [FAMILY]:
        raise N.Blocked(f'executor refused the plan {core.refused}')
    return {int(r['lane']): float(r['v_w']) for r in core.plans[0]['spec']['roles']}


def build(src_path, out_dir):
    """Zero physics (laptop): write ops/a64/plans/A64_HAND3_F2x from the A59 source plan file."""
    src_path, out_dir = Path(src_path), Path(out_dir)
    if N.file_sha(src_path) != SRC_PLAN_SHA256:
        raise N.Blocked('source plan file is not the A59 HAND3 plan 498a5206')
    text = src_path.read_text(encoding='utf-8')
    src = json.loads(text)
    if D.C37.bank_text(src) != text:
        raise N.Blocked('source is not in the bank_text form')
    out = dict(source=dict(path=str(src_path), file_sha256=SRC_PLAN_SHA256, role_speeds=role_speeds(src)))
    for v, (name, vw) in VARIANTS.items():
        doc = variant_doc(src, vw)
        p = D.C37.write_once(out_dir/name, D.C37.bank_text(doc))
        sp = role_speeds(json.loads(p.read_text(encoding='utf-8')))
        if sp != {0: BASE_VW, 1: BASE_VW, FAST_LANE: vw}:
            raise N.Blocked(f'{v}: executor role speeds {sp}')
        out[v] = dict(path=str(p), file_sha256=N.file_sha(p), role_speeds=sp, diff=json_diff(src, doc))
    return out


def bank_path(final_root, variant):
    return D.bank_path(final_root, VARIANTS[variant][0])


def prepare(final_root):
    """Copy the pinned plan files into <root>/a64_banks (write once) and check shas + executor role speeds."""
    final_root = Path(final_root)
    out = {}
    src = Path(A59_ROOT)/'a59_banks'/SRC_PLAN
    if N.file_sha(src) != SRC_PLAN_SHA256:
        raise N.Blocked(f'{src}: not the A59 HAND3 plan')
    p = D.C37.write_once(D.bank_path(final_root, SRC_PLAN), src.read_text(encoding='utf-8'))
    out['HAND3'] = dict(path=str(p), file_sha256=N.file_sha(p))
    for v, (name, vw) in VARIANTS.items():
        if PLAN_SHA256[v] is None:
            raise N.Blocked(f'{v}: plan sha not pinned (run build, pin, record the build note first)')
        q = PLANS_DIR/name
        if N.file_sha(q) != PLAN_SHA256[v]:
            raise N.Blocked(f'{q}: not the pinned plan file')
        p = D.C37.write_once(bank_path(final_root, v), q.read_text(encoding='utf-8'))
        if N.file_sha(p) != PLAN_SHA256[v]:
            raise N.Blocked(f'{p}: not the pinned plan file')
        sp = role_speeds(json.loads(p.read_text(encoding='utf-8')))
        if sp != {0: BASE_VW, 1: BASE_VW, FAST_LANE: vw}:
            raise N.Blocked(f'{v}: executor role speeds {sp}')
        out[v] = dict(path=str(p), file_sha256=PLAN_SHA256[v], role_speeds=sp)
    return out


def _check(final_root):
    for v in VARIANTS:
        p = bank_path(final_root, v)
        if PLAN_SHA256[v] is None or not p.is_file() or N.file_sha(p) != PLAN_SHA256[v]:
            raise N.Blocked(f'{p}: missing or not the pinned plan')
    p = D.bank_path(final_root, SRC_PLAN)
    if not p.is_file() or N.file_sha(p) != SRC_PLAN_SHA256:
        raise N.Blocked(f'{p}: missing or not the A59 HAND3 bytes')


# ------------------------------------------------------------------------------------------------ specs

def _entry(final_root, name, check_files=True):
    if check_files:
        _check(final_root)
    return D._bank_entry(final_root, name, check_files=check_files) or dict(path='', file_sha256='')


def arm_specs(arm, panel, final_root, *, check_files=True):
    from tools.coopmwm import a57_partlane as P
    v, lanes = ARMS[arm]
    specs = D._a37_specs(D.RUNNER_ARM['BIG_BANK'], panel, final_root, _entry(final_root, VARIANTS[v][0], check_files),
                         tag=f'{D.CONFIRM_TAG}_{arm.lower()}')
    if lanes is None:
        return specs
    out = []
    for s in specs:
        ck = deepcopy(s['checkpoint'])
        ck.update(a57_member_lane=list(lanes), a57_sha256=P.a57_digest())
        out.append(dict(s, checkpoint=ck))
    return out


def arm_waves(panel, final_root, *, check_files=True, first_only=False):
    out = []
    for arm in NEW_ARMS:
        specs = arm_specs(arm, panel, final_root, check_files=check_files)
        out.append((wave(arm), 'episode', specs[:1] if first_only else specs))
    return out


IDENT_INDEX = 0


def ident_waves(panel, final_root, *, check_files=True):
    s = D._a37_specs(D.RUNNER_ARM['BIG_BANK'], panel, final_root, _entry(final_root, SRC_PLAN, check_files),
                     tag=f'{D.IDENT_TAG}_hand3')[IDENT_INDEX]
    return [(IDENT_WAVE, 'episode', [dict(s, job_id=f'{D.IDENT_TAG}_hand3_0000')])]


def stored_records(panel):
    """{'U0' | 'BIG_BANK' | 'DUO_HOLD' | 'HAND3': {world: rec}} (A42 / A59 roots, read-only), sources, roots."""
    have, src, roots = {}, {}, {}
    for arm, name in STORED_A42.items():
        specs = D42.confirm_specs(name, panel, A42_ROOT, check_files=True)
        have[arm], src[arm] = D42.C37._load(A42_ROOT, D42.confirm_wave(name), specs)
        roots[arm] = A42_ROOT
    a59 = _a59()
    name, _, specs = a59.confirm_wave('HAND3', panel, A59_ROOT)
    have['HAND3'], src['HAND3'] = a59.D.C37._load(A59_ROOT, name, specs)
    roots['HAND3'] = A59_ROOT
    return have, src, roots


def _acts(rec):
    a = rec.get('a37') or {}
    return [(x['t'], x['family'], list(x['members']), x.get('release')) for x in a.get('acts') or []]


def identity_compare(final_root, panel):
    final_root = Path(final_root)
    have, _, _ = stored_records(panel)
    (name, _, specs), = ident_waves(panel, final_root)
    got, _, cov = N.published(final_root, name, specs, D.C37.check_a37)
    if cov != 1:
        raise N.Blocked(f'{name}: {cov}/1 published')
    r = got[0]
    a = have['HAND3'][r['world_id']]
    row = dict(world_id=r['world_id'], seed=r['world']['seed'], j_a64=r['J_veh_s'], j_a59=a['J_veh_s'],
               abs_diff=abs(r['J_veh_s']-a['J_veh_s']), safety_a64=r['safety'], safety_a59=a['safety'],
               safety_equal=r['safety'] == a['safety'], same_world=r['world'] == a['world'],
               activations=[r['a37'].get('activations'), a['a37'].get('activations')],
               acts_equal=_acts(r) == _acts(a), a37_sha256_equal=r['a37_sha256_loaded'] == a['a37_sha256_loaded'],
               executor_sha256_equal=r['executor_sha256_loaded'] == a['executor_sha256_loaded'],
               arms=[r['a37'].get('a37_arm'), a['a37'].get('a37_arm')], no_filter=('a57_part' not in r['a37']))
    ok = (row['abs_diff'] == 0.0 and row['safety_equal'] and row['same_world'] and row['acts_equal']
          and row['a37_sha256_equal'] and row['executor_sha256_equal'] and row['no_filter'])
    return N.seal(dict(schema='coopmwm.a64.identity.v1', amendment=AMENDMENT,
                       rule=('A64 path (one-row CELL_BANK of the unchanged A59 HAND3 bytes, lane-set filter code '
                             'present, filter off) reproduces the stored A59 confirmation HAND3 record (A37_PLAN) of '
                             'panel world index 0 exactly: J_veh_s difference 0, identical safety counters, identical '
                             'activation receipts, same world, same a37 and executor digests'),
                       verdict='PASS' if ok else 'FAILED', world=row,
                       consequence='PASS: stored A42 U0 / BIG_BANK / DUO_HOLD and A59 HAND3 reused; FAILED: stop'),
                  'receipt_sha256')


def _passed(final_root, name):
    p = Path(final_root)/name
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


def part_receipt_ok(part, lanes, digest):
    """Pure (unit-tested; the A63 rule with the single family HAND3): only members planned on ``lanes`` commanded."""
    lanes = sorted(lanes)
    on = {v for a in part.get('acts', ()) for v in a['on_lane']}
    cmd = part.get('commanded_lanes') or {}
    return bool(part.get('member_lanes') == lanes and part.get('a57_sha256') == digest
                and all(set(a['on_lane']) <= set(a['members']) for a in part.get('acts', ()))
                and all(a['family'] == FAMILY for a in part.get('acts', ()))
                and all(int(x) in lanes for x in cmd.values()) and set(cmd) <= on
                and len(cmd) == part.get('vehicles_with_accel_command'))


def plan_roles(final_root, variant):
    from tools.coopmwm import oracle_plan as O
    doc = json.loads(bank_path(final_root, variant).read_text(encoding='utf-8'))
    tpl, = doc['clean5c']['templates']
    return O.active_roles(O.spec_of(FAMILY, tpl['offer_condition']), 0)


def implied_speeds(acts, roles, actions_by_t, speed_by_t, tol=1e-3):
    """Pure (unit-tested). acts = receipt acts (t, members in role order); actions_by_t = {t: [action dicts with vid,
    requested_raw, oracle_plan]}; speed_by_t = {t: {vid: v_mps}}. Every plan action is mapped to the latest activation
    (t_act <= t) containing its vehicle -> planned role. An interior action (-dmax < requested < 1) implies
    v_w = requested + v. Returns dict(per_lane={lane: dict(n_interior, n_saturated, implied_min, implied_max)},
    mismatches, unmapped, commanded_lanes)."""
    per = {}
    bad, unmapped, cmd = [], 0, Counter()
    starts = sorted((int(a['t']), i) for i, a in enumerate(acts))
    for t in sorted(actions_by_t):
        for x in actions_by_t[t]:
            if x.get('oracle_plan') != FAMILY:
                continue
            vid = x['vid']
            cand = [i for (ta, i) in starts if ta <= t and vid in acts[i]['members']]
            if not cand:
                unmapped += 1
                continue
            act = acts[cand[-1]]
            role = roles[act['members'].index(vid)]
            lane = int(role['lane'])
            cmd[lane] += 1
            d = per.setdefault(lane, dict(n_interior=0, n_saturated=0, implied_min=None, implied_max=None,
                                          registered_v_w=float(role['v_w'])))
            r, v = float(x['requested_raw']), (speed_by_t.get(t) or {}).get(vid)
            if v is None or not (-float(role['dmax'])+1e-6 < r < 1.0-1e-6):
                d['n_saturated'] += 1
                continue
            imp = r+float(v)
            d['n_interior'] += 1
            d['implied_min'] = imp if d['implied_min'] is None else min(d['implied_min'], imp)
            d['implied_max'] = imp if d['implied_max'] is None else max(d['implied_max'], imp)
            if abs(imp-float(role['v_w'])) > tol:
                bad.append((t, vid, lane, imp))
    return dict(per_lane={str(k): v for k, v in sorted(per.items())}, mismatches=bad[:20], n_mismatch=len(bad),
                unmapped=unmapped, commanded_lanes={str(k): v for k, v in sorted(cmd.items())})


def _observable(root, rec):
    d = Path(root)/'evaluation'/rec.get('work_dir', rec['job_id'])
    acts, speeds = {}, {}
    with gzip.open(d/'observable.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            x = json.loads(line)
            t = int(round(float(x['time_s'])))
            if x.get('actions'):
                acts[t] = x['actions']
                speeds[t] = {v['vid']: v['v_mps'] for v in x['vehicles']}
    return acts, speeds


def speed_check_ok(arm, chk):
    """Pure: every interior action implies its role's v_w; the lane-2 role has >= 1 interior action at the registered
    speed; full arms command all three planned lanes, PART_L2 arms only lane 2; nothing unmapped."""
    v, lanes = ARMS[arm]
    want = VARIANTS[v][1]
    pl = chk['per_lane']
    l2 = pl.get(str(FAST_LANE)) or {}
    lanes_cmd = set(chk['commanded_lanes'])
    ok = (chk['n_mismatch'] == 0 and chk['unmapped'] == 0 and l2.get('n_interior', 0) > 0
          and l2.get('registered_v_w') == want and abs(l2['implied_min']-want) <= 1e-3
          and abs(l2['implied_max']-want) <= 1e-3)
    if lanes is None:
        ok = ok and lanes_cmd == {'0', '1', '2'} and all(pl[k]['registered_v_w'] == BASE_VW for k in ('0', '1'))
    else:
        ok = ok and lanes_cmd == {str(x) for x in lanes}
    return bool(ok)


def smoke_check(final_root, panel):
    """Zero physics, outcome-blind (no delay read): per smoke record: valid (check_a37); family HAND3 only; full arms no
    filter receipt; PART_L2 arms the a57_part receipt of lane set {2} with the registered digest; observable log:
    every plan command implies the planned role's registered v_w (lane 2 = 23 / 26, lanes 0/1 = 20) and PART_L2 arms
    command only planned lane-2 members."""
    from tools.coopmwm import a57_partlane as P
    rows, ok = [], True
    for (name, _, specs), arm in zip(arm_waves(panel, final_root, first_only=True), NEW_ARMS):
        got, _, cov = N.published(final_root, name, specs, D.C37.check_a37)
        if cov != 1:
            raise N.Blocked(f'{name}: smoke episode not published')
        v, lanes = ARMS[arm]
        rec = got[0]
        a37 = rec.get('a37') or {}
        acts = a37.get('acts') or []
        fams = sorted({x['family'] for x in acts})
        acts_t, speeds = _observable(final_root, rec)
        chk = implied_speeds(acts, plan_roles(final_root, v), acts_t, speeds)
        row = dict(wave=name, world_id=rec['world_id'], variant=v, lane2_v_w=VARIANTS[v][1],
                   activations=a37.get('activations'), families_activated=fams,
                   members_planned=sum(len(x['members']) for x in acts), speed_check=chk)
        good = fams == [FAMILY] and speed_check_ok(arm, chk)
        if lanes is None:
            good = good and 'a57_part' not in a37
        else:
            part = a37.get('a57_part') or {}
            good = good and part_receipt_ok(part, lanes, P.a57_digest())
            row.update(member_lanes=part.get('member_lanes'), members_on_lanes=part.get('members_on_lane_total'),
                       no_member_on_lanes=part.get('activations_no_member_on_lane'),
                       commanded_vehicles=part.get('vehicles_with_accel_command'),
                       commanded_by_planned_lane=dict(Counter(str(x) for x in
                                                              (part.get('commanded_lanes') or {}).values())),
                       kept_accel=part.get('kept_accel'), dropped_accel=part.get('dropped_accel'))
        row['receipt_ok'] = bool(good)
        ok = ok and good
        rows.append(row)
    return N.seal(dict(schema='coopmwm.a64.smoke.v1', verdict='PASS' if ok else 'FAILED', rows=rows,
                       rule=('record valid; family HAND3 only; observable log: every interior plan command '
                             '(-dmax < requested < 1) implies requested + v = the planned role v_w (|err| <= 1e-3), the '
                             'lane-2 role at the registered 23 / 26 with >= 1 interior command, lanes 0/1 at 20 in the '
                             'full arms; PART_L2: a57_part receipt of lane set {2} + registered digest and only planned '
                             'lane-2 members commanded; full arms: no filter receipt (no outcome read)')),
                  'receipt_sha256')


# ------------------------------------------------------------------------------------------------ analysis

def p1_verdict(ci):
    return 'SAVES' if ci[1] < 0 else 'NOT SHOWN'


def p3_verdict(ci):
    return 'OTHER LANES HELP' if ci[1] < 0 else 'HURT' if ci[0] > 0 else 'NOT RESOLVED'


def holm_p3(deltas, bootstrap):
    """Pure (unit-tested). deltas = {variant: v - v_PART_L2 per world}. Holm over the two variants with CI inversion:
    the smaller bootstrap p is tested with the 1 - alpha/2 CI; the other with the 1 - alpha CI only if the first
    rejected (CI excludes 0), else NOT RESOLVED."""
    from tools.coopmwm import planner_eval as pe
    alpha = round(1-bootstrap['level'], 10)
    st = {v: pe._paired_ci(d, bootstrap['reps'], bootstrap['seed'], alpha) for v, d in deltas.items()}
    order = sorted(st, key=lambda v: (st[v]['p_two_sided'], v))
    out, rejected_prev = {}, True
    for i, v in enumerate(order):
        a = alpha/(len(order)-i)
        ci = pe._paired_ci(deltas[v], bootstrap['reps'], bootstrap['seed'], a)
        verdict = p3_verdict(ci['ci95']) if rejected_prev else 'NOT RESOLVED'
        rejected_prev = rejected_prev and verdict != 'NOT RESOLVED'
        out[v] = dict(diff=st[v]['mean_delta_vs'], ci95=st[v]['ci95'], p_two_sided=st[v]['p_two_sided'],
                      holm_step=i+1, holm_alpha=a, holm_ci=ci['ci95'], verdict=verdict,
                      worlds_negative=st[v]['worlds_improved'])
    return out


def group_rows(groups):
    """Registered groups of a co_primary_groups result with the 10 s / 5 s checks (point estimate and CI upper)."""
    out = {}
    for g in N.CO_PRIMARY_GROUPS:
        v = groups['groups'].get(g) or {}
        if 'increase_mean_s' not in v:
            raise N.Blocked(f'registered group {g} unavailable')
        inc, ci = float(v['increase_mean_s']), list(v['ci95'])
        out[g] = dict(increase_mean_s=inc, ci95=ci, within_10s_point=inc <= 10.0, within_5s_point=inc <= 5.0,
                      within_10s_ci_upper=ci[1] <= 10.0, within_5s_ci_upper=ci[1] <= 5.0)
    return out


def _all(rows, k):
    return all(r[k] for r in rows.values())


def _mean_ci(x, bootstrap):
    import numpy as np
    x = np.asarray(x, float)
    s = N.fe.paired_stats(x, np.zeros_like(x), bootstrap, pct=False)
    return dict(mean=s['diff'], ci95=s['ci95'], p_two_sided=s['p_two_sided'], worlds_negative=s['worlds_lower'],
                worlds_positive=s['worlds_higher'])


def _reports_readonly(roots, have, worlds):
    from a32_ablation import report_of
    out = {}
    for a in have:
        out[a] = {}
        for w in worlds:
            rec = have[a][w]
            d = Path(roots[a])/'evaluation'/rec.get('work_dir', rec['job_id'])
            if str(roots[a]) in (A42_ROOT, A59_ROOT) and not (d/'vehicles_report.json').is_file():
                raise N.Blocked(f'{d}: no cached report (stored roots are read-only here)')
            out[a][w] = report_of(roots[a], rec)
            if not out[a][w]['ok']:
                raise N.Blocked(f'{rec["job_id"]}: invalid vehicle-log report')
    return out


def _arm_counts(arm, recs, worlds):
    rows = [(recs[w].get('a37') or {}) for w in worlds]
    planned = sum(len(x['members']) for r in rows for x in (r.get('acts') or []))
    sizes = Counter(len(x['members']) for r in rows for x in (r.get('acts') or []))
    out = dict(activations=sum(int(r.get('activations') or 0) for r in rows),
               worlds_with_activation=sum(1 for r in rows if r.get('activations')), members_planned_total=planned,
               group_sizes={str(k): v for k, v in sorted(sizes.items())},
               activations_by_family=dict(Counter(x['family'] for r in rows for x in (r.get('acts') or []))))
    if arm in ARMS and ARMS[arm][1] is not None:
        parts = [r['a57_part'] for r in rows]
        out.update(members_commanded_total=sum(p['members_on_lane_total'] for p in parts),
                   activations_no_member_on_lanes=sum(p['activations_no_member_on_lane'] for p in parts),
                   vehicles_with_accel_command=sum(p['vehicles_with_accel_command'] for p in parts),
                   commanded_by_planned_lane=dict(sum((Counter(str(x) for x in p['commanded_lanes'].values())
                                                       for p in parts), Counter())),
                   kept_accel_steps=sum(p['kept_accel'] for p in parts),
                   dropped_accel_steps=sum(p['dropped_accel'] for p in parts))
    elif arm != 'U0':
        out.update(members_commanded_total=planned)
    return out


def analyse(final_root, panel):
    import numpy as np
    from tools.coopmwm import a57_partlane as P
    final_root = Path(final_root)
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != BOOTSTRAP or len(worlds) != N_WORLDS:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    if not _passed(final_root, IDENTITY_FILE) or not _passed(final_root, SMOKE_FILE):
        raise N.Blocked('identity / smoke not PASS')
    _check(final_root)
    have, sources, roots = stored_records(panel)
    for (name, _, specs), arm in zip(arm_waves(panel, final_root), NEW_ARMS):
        have[arm], sources[arm] = D.C37._load(final_root, name, specs)
        roots[arm] = str(final_root)
        lanes = ARMS[arm][1]
        for w in worlds:
            a37 = have[arm][w].get('a37') or {}
            if not {x['family'] for x in a37.get('acts') or []} <= {FAMILY}:
                raise N.Blocked(f'{arm} {w}: activation of a family other than HAND3')
            if lanes is None:
                if 'a57_part' in a37:
                    raise N.Blocked(f'{arm} {w}: unexpected filter receipt')
            elif not part_receipt_ok(a37.get('a57_part') or {}, lanes, P.a57_digest()):
                raise N.Blocked(f'{arm} {w}: a57_part receipt missing / wrong lanes / wrong digest')
    for a in tuple(STORED_A42)+STORED_A59:
        if any('a57_part' in (have[a][w].get('a37') or {}) for w in worlds):
            raise N.Blocked(f'{a}: stored record carries a filter receipt')
    reports = _reports_readonly(roots, have, worlds)
    arms = tuple(STORED_A42)+STORED_A59+NEW_ARMS
    J = {a: np.array([have[a][w]['J_veh_s'] for w in worlds], float) for a in arms}
    S = {k: {a: np.array([have[a][w]['safety'][k] for w in worlds], float) for a in arms} for k in N.SAFETY}
    vs_u0 = {a: N._bank_compare(have, reports, worlds, a, 'U0', bootstrap) for a in arms if a != 'U0'}
    variants = {}
    for v in VARIANTS:
        full = FULL_OF[v]
        p1 = N.fe.paired_stats(J[full], J['U0'], bootstrap)
        p1.update(rule=RULES['P1'], verdict=p1_verdict(p1['ci95']))
        g = group_rows(vs_u0[full]['groups'])
        vs_h3 = N._bank_compare(have, reports, worlds, full, 'HAND3', bootstrap)
        p2 = dict(rule=RULES['P2'], total_minus_hand3=vs_h3['total_minus_ref'],
                  lane2_increase_vs_u0=g['origin:L2'],
                  hand3_lane2_increase_vs_u0=group_rows(vs_u0['HAND3']['groups'])['origin:L2'],
                  lane2_minus_hand3_s=dict(increase_mean_s=vs_h3['groups']['groups']['origin:L2']['increase_mean_s'],
                                           ci95=vs_h3['groups']['groups']['origin:L2']['ci95'],
                                           note='same vehicles labelled by HAND3 origin; descriptive'),
                  groups_vs_u0=g, all_within_10s_point=_all(g, 'within_10s_point'),
                  all_within_5s_point=_all(g, 'within_5s_point'), all_within_10s_ci_upper=_all(g, 'within_10s_ci_upper'),
                  all_within_5s_ci_upper=_all(g, 'within_5s_ci_upper'))
        variants[v] = dict(arm=full, part_arm=PART_OF[v], lane2_v_w=VARIANTS[v][1], P1=p1, P2=p2, vs_hand3=vs_h3)
    p3 = holm_p3({v: J[FULL_OF[v]]-J[PART_OF[v]] for v in VARIANTS}, bootstrap)
    for v in VARIANTS:
        variants[v]['P3'] = dict(rule=RULES['P3'], **p3[v])
    group_table = {a: group_rows(vs_u0[a]['groups']) for a in vs_u0}
    sec = dict(
        vs_duo_hold={a: N._bank_compare(have, reports, worlds, a, 'DUO_HOLD', bootstrap) for a in NEW_ARMS},
        vs_big_bank={a: N._bank_compare(have, reports, worlds, a, 'BIG_BANK', bootstrap) for a in NEW_ARMS},
        part_vs_hand3={a: N.fe.paired_stats(J[a], J['HAND3'], bootstrap) for a in ('F23_PART_L2', 'F26_PART_L2')},
        f26_minus_f23=N.fe.paired_stats(J['HAND3_F26'], J['HAND3_F23'], bootstrap),
        arms_vs_u0={a: dict(total=vs_u0[a]['total_minus_ref'], tail_cvar90=vs_u0[a]['tail_cvar90'],
                            groups_all_within_10s=vs_u0[a]['groups'].get('all_pass')) for a in vs_u0},
        tail_cvar90_vs_hand3={v: variants[v]['vs_hand3']['tail_cvar90'] for v in VARIANTS},
        safety_minus_u0={k: {a: _mean_ci(S[k][a]-S[k]['U0'], bootstrap) for a in arms if a != 'U0'}
                         for k in ('hard_brakes', 'ttc_events') if k in S},
        safety_minus_hand3={k: {a: _mean_ci(S[k][a]-S[k]['HAND3'], bootstrap) for a in NEW_ARMS}
                            for k in ('hard_brakes', 'ttc_events') if k in S},
        group_table_vs_u0=group_table)
    arm_rows = {a: dict(mean_J_veh_s=float(J[a].mean()),
                        safety_totals={k: int(S[k][a].sum()) for k in N.SAFETY},
                        **_arm_counts(a, have[a], worlds)) for a in arms}
    per_world = [dict(world_id=w, seed=seed[w], **{f'J_{a}': float(have[a][w]['J_veh_s']) for a in arms},
                      **{f'hard_brakes_{a}': int(S['hard_brakes'][a][i]) for a in arms},
                      **{f'ttc_{a}': int(S['ttc_events'][a][i]) for a in arms},
                      **{f'acts_{a}': int((have[a][w].get('a37') or {}).get('activations') or 0)
                         for a in arms if a != 'U0'},
                      **{f'commanded_{a}': int(have[a][w]['a37']['a57_part']['members_on_lane_total'])
                         for a in NEW_ARMS if ARMS[a][1] is not None},
                      **{f'collisions_{a}': int(have[a][w]['safety']['collisions']) for a in arms},
                      **{f'teleports_{a}': int(have[a][w]['safety']['teleports']) for a in arms})
                 for i, w in enumerate(worlds)]
    return N.seal(dict(schema='coopmwm.a64.hand3fast.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       setting=SETTING, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), bootstrap=bootstrap,
                       convention='diff = treatment - reference; negative = less delay', rules=RULES,
                       variants=variants, secondary=sec, arms=arm_rows, per_world=per_world,
                       a57_sha256=P.a57_digest(), plan_sha256=dict(PLAN_SHA256, HAND3=SRC_PLAN_SHA256),
                       reused=dict(**{a: f'{A42_ROOT} WAVE_A42C_{n}' for a, n in STORED_A42.items()},
                                   HAND3=f'{A59_ROOT} WAVE_A59C_HAND3'),
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()},
                       charged_sim_s=D.charged(final_root), root=str(final_root)), 'analysis_sha256')


def write_table(v, path):
    rows = v['per_world']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------------------------ CLI

def budget_plan():
    ident = N.HORIZON_S
    run = len(NEW_ARMS)*N_WORLDS*N.HORIZON_S
    return dict(phase=D.A42_PHASE, cap_sim_s=CAP_SIM_S, identity=ident, smoke_included_in_run=len(NEW_ARMS)*N.HORIZON_S,
                run=run, planned=ident+run, margin=CAP_SIM_S-ident-run)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('build', 'prepare', 'dry', 'identity', 'identity-compare', 'smoke',
                                       'smoke-check', 'run', 'analyse', 'status', 'budget'))
    p.add_argument('--final-root')
    p.add_argument('--src')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--chunk', type=int, default=16)
    a = p.parse_args(argv)
    if a.command == 'budget':
        print(json.dumps(budget_plan(), indent=1))
        return
    if a.command == 'build':
        print(json.dumps(build(a.src, a.out or PLANS_DIR), indent=1, default=str))
        return
    if not a.final_root:
        p.error('--final-root required')
    root = Path(a.final_root)
    if a.command == 'prepare':
        print(json.dumps(prepare(root), indent=1))
        return
    panel = D.setup(SETTING)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=SETTING), sort_keys=True,
                          default=str)
        print(line, flush=True)
        with (root/D.LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    if a.command == 'dry':
        iw = ident_waves(panel, root)
        aw = arm_waves(panel, root)
        a59 = _a59()
        _, _, ref = a59.confirm_wave('HAND3', panel, A59_ROOT)
        mine = iw[0][2][0]

        def norm(s):
            s = dict(s)
            for k in ('job_id', 'arm'):
                s.pop(k, None)
            ck = {k: v for k, v in s['checkpoint'].items()
                  if k not in ('a57_member_lane', 'a57_sha256', 'a37_bank', 'a37_arm', 'a37_family')}
            s['checkpoint'] = ck
            t = json.dumps(s, sort_keys=True, default=str)
            return t.replace(str(root), '<root>').replace(D.HOST_ROOT, '<root>').replace(A59_ROOT, '<root>')
        print(json.dumps(dict(
            budget=budget_plan(), charged=D.charged(root), identity_world=mine['world_id'],
            identity_ref_job=ref[IDENT_INDEX]['job_id'], identity_ref_arm=ref[IDENT_INDEX]['arm'],
            identity_bank=mine['checkpoint']['a37_bank'], identity_arm=mine['arm'],
            identity_spec_equals_a59_except_job_arm_bank=norm(mine) == norm(ref[IDENT_INDEX]),
            arms=[(n, len(sp), sp[0]['job_id'], sp[0]['arm'], sp[0]['checkpoint']['a37_bank'],
                   sp[0]['checkpoint'].get('a57_member_lane'), (sp[0]['checkpoint'].get('a57_sha256') or '')[:12],
                   [sp[0]['world']['seed'], sp[-1]['world']['seed']]) for n, _, sp in aw],
            arm_specs_equal_a59_hand3_except_filter_bank_arm=all(
                norm(s) == norm(r) for _, _, sp in aw for s, r in zip(sp, ref)),
            a37_sha256=mine['checkpoint']['a37_sha256'], worlds=len(panel['worlds'])), indent=1, default=str))
    elif a.command in ('identity', 'smoke', 'run'):
        import fcntl
        if a.command != 'identity' and not _passed(root, IDENTITY_FILE):
            raise N.Blocked(f'{IDENTITY_FILE} is not PASS: the A64 arms are not run')
        if a.command == 'run' and not _passed(root, SMOKE_FILE):
            raise N.Blocked(f'{SMOKE_FILE} is not PASS')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = (ident_waves(panel, root) if a.command == 'identity' else
                     arm_waves(panel, root, first_only=a.command == 'smoke'))
            done = sum(N.published(root, n, s, D.C37.check_a37)[2] for n, _, s in waves)
            need = (sum(len(s) for _, _, s in waves)-done)*N.HORIZON_S
            if D.charged(root)+need > CAP_SIM_S:
                raise N.Blocked(f'{D.A42_PHASE}: {D.charged(root)} + {need} > cap {CAP_SIM_S}')
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=D.charged(root)))
            # no automatic replay (A64 registration): a failed dispatch raises and is reported
            res = D.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(root, panel)
        N.publish(root/IDENTITY_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], world=v['world']['world_id'], abs_diff=v['world']['abs_diff'],
                              safety_equal=v['world']['safety_equal'], acts_equal=v['world']['acts_equal'],
                              activations=v['world']['activations'], arms=v['world']['arms'])))
    elif a.command == 'smoke-check':
        v = smoke_check(root, panel)
        N.publish(root/SMOKE_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], rows=[{k: x[k] for k in ('wave', 'activations', 'receipt_ok')}
                                                          | dict(lanes=x['speed_check']['per_lane'])
                                                          for x in v['rows']]), default=str))
    elif a.command == 'analyse':
        v = analyse(root, panel)
        out = Path(a.out or root)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/'HAND3FAST.json', v, replace=True)
        write_table(v, out/'HAND3FAST_PER_WORLD.csv')
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], **{
            x: dict(P1=[r['P1']['verdict'], r['P1']['diff'], r['P1']['ci95']],
                    P2=[r['P2']['total_minus_hand3']['diff'], r['P2']['total_minus_hand3']['ci95'],
                        r['P2']['lane2_increase_vs_u0']['increase_mean_s'], r['P2']['lane2_increase_vs_u0']['ci95']],
                    P3=[r['P3']['verdict'], r['P3']['diff'], r['P3']['holm_ci']])
            for x, r in v['variants'].items()})))
    else:   # status
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, identity_pass=_passed(root, IDENTITY_FILE),
                   smoke_pass=_passed(root, SMOKE_FILE))
        for name, _, specs in ident_waves(panel, root)+arm_waves(panel, root):
            out[name] = f'{N.published(root, name, specs, D.C37.check_a37)[2]}/{len(specs)}'
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
