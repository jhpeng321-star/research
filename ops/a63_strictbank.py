"""AMENDMENT 63 (S-STRICTBANK60; registered 2026-11-11 before any outcome of this test): STRICT_BANK and lane-set parts.

Arms (96 A42 confirmation worlds 998600-998695, setting a42c_D2800_p60, D2800 p60; the A37 CELL_BANK executor, trigger
and settings exactly as the A42 BIG_BANK):
  BIG_PART_L01     frozen A42 BIG_BANK, only the planned members on SUMO lanes {0, 1} commanded
  STRICT_BANK      the four admitted plans with the lowest verification-world worst-group increase (fixed in the
                   registration): rand_2c800424, rand_bb1cb39d, rand_cfa9b70a, mut_2922c89d; each keeps its stored offer
                   condition (template) unchanged; rows = the verified lb95 of its own admission, bank order = the A42
                   admitted_plans rule (lb95 descending, then strategy_key)
  STRICT_PART_L2   STRICT_BANK, only the planned members on SUMO lane 2 commanded
  STRICT_PART_L01  STRICT_BANK, only the planned members on SUMO lanes {0, 1} commanded
The lane filter is tools/coopmwm/a57_partlane.py (A57, eb704616) extended to a lane set (checkpoint a57_member_lane
may be a list); activation eligibility on the FULL planned group. U0, BIG_BANK and DUO_HOLD are the stored A42 per-world
records (A42 root, read-only) after an identity check.

This driver IS ops/a42_bigp60.py (private module copy, as A50 / A56 / A57) with the A63 names: final root /root/a63/root,
ledger phase strictbank_a63 cap 480,000 sim-s, job ids a63*, waves A63_IDENT_BIG_BANK / A63C_<arm>, stop file STOP_A63.
No automatic replay (a failed dispatch raises and is reported).

Commands (python ops/a63_strictbank.py <cmd> --final-root <root>):
  build-strict      zero physics: build a63_banks/A63_STRICT_BANK_D2800_p60.json from the pinned sources (write once)
                    + BUILD_A63.json (sealed); copies the A42 BIG_BANK bytes into a63_banks
  prepare / dry     zero physics checks + spec summary + budget
  identity          1 episode: BIG_BANK, filter OFF, identity world (first panel world with a stored activation)
  identity-compare  -> IDENTITY_A63.json
  smoke             4 episodes: world index 0 of each arm (refused unless identity PASS); first chunk of the waves
  smoke-check       -> SMOKE_A63.json: only members on the intended lanes are commanded (activation receipts)
  run               the 4 x 96 waves (refused unless identity PASS and smoke PASS)
  analyse           P1 / P2 / P3 + secondaries -> STRICTBANK.json (+ per-world CSV) in --out
  status / budget
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from copy import deepcopy
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

AMENDMENT = '63'
CELL = 'D2800_p60'
SETTING = 'a42c_D2800_p60'
N_WORLDS = 96
CAP_SIM_S = 480_000
A42_ROOT = '/root/a42v/root'
BIG_BANK = f'A42_BIG_BANK_{CELL}.json'
STRICT_BANK = f'A63_STRICT_BANK_{CELL}.json'
A42_ADMISSION = 'ADMISSION_D2800_p60.json'
A42_ADMISSION_RECEIPT = '5fa2a7015fb4ae9a95e1d17456e5caacb6114405cc197831d2c3d674f1b332d5'
BIG_BANK_SHA256 = '7e7dc76b8a0cd5c158205a0e3f6ef96d73a36a483c46d03ac422428dc8c573ff'
#: pinned after build-strict (recorded in REGISTRATION.md before any SUMO); None -> identity / smoke / run refuse
STRICT_BANK_SHA256 = 'e2e341afbfbee87334500743c1cf9e456cb1966418c6fce87199c9bfd91e90ae'
STRICT_FAMILIES = ('rand_2c80042490ec698f', 'rand_bb1cb39d47f0a524', 'rand_cfa9b70a3782d48e', 'mut_2922c89d0b31d4c8')
#: the plan sources (host paths, read-only) and their sha256, pinned before the build
SOURCES = dict(
    A42_BIG_BANK=(f'{A42_ROOT}/a42_banks/A42_BIG_BANK_D2800_p60.json', BIG_BANK_SHA256),
    A42_CANDS=(f'{A42_ROOT}/a42_banks/A42_CANDS_D2800_p60.json',
               '3f9f02cd492ab181d26e51b6730b5c4ab743da030ada8b17b2ace8adddd7d2aa'),
    A45_DIV_CANDS=(f'{A42_ROOT}/a42_banks/A45_DIV_CANDS_D2800_p60.json',
                   '880473846fe094bdbe7bd774734b19767c9bc671fbd7be399ce4247e35d8fb1f'),
    A45_DIVERSE_BANK=(f'{A42_ROOT}/a42_banks/A45_DIVERSE_BANK_D2800_p60.json',
                      '236eb04de7ae89a6928b441cc6596df77b8163dff71d9426aec0b8e2ccb85c0e'),
    A56_RAND_CANDS=('/root/a56/root/a56_banks/A56_RAND_CANDS_D2800_p60.json',
                    '597cc2da279156e503f55373f82266339a48f1f730f5a21da949e1ea26d3c188'),
    A56_RANDPLANS=('/root/a56/src/ops/a56/RANDPLANS_D2800_p60.json',
                   '687663b078c768643d56758b42dc8113d297be1d606ecaf7994217964ec3dbaa'),
    A42_ADMISSION=(f'{A42_ROOT}/ADMISSION_D2800_p60.json',
                   '5ed0317ce39f6c6bb7a6756695fb50a64bf96513b17768a49ab58489a18f0cf9'),
    A45_ADMISSION=(f'{A42_ROOT}/ADMISSION_A45_D2800_p60.json',
                   '4714805cf757b5361af8d59941e6f90c0128d082bdcbfcc59bce4b1c35ff6eb0'),
    A56_REDUCE=('/root/a56/root/A56_REDUCE_D2800_p60.json',
                '2aea2de482050bebfdddb0046bb2c3529025aca37b7f551eeb7dde067b4a4dcc'))
BANK_SOURCES = ('A42_BIG_BANK', 'A42_CANDS', 'A45_DIV_CANDS', 'A45_DIVERSE_BANK', 'A56_RAND_CANDS')
#: arm -> (bank file, member lane set or None = no filter)
ARMS = dict(BIG_PART_L01=(BIG_BANK, [0, 1]), STRICT_BANK=(STRICT_BANK, None),
            STRICT_PART_L2=(STRICT_BANK, [2]), STRICT_PART_L01=(STRICT_BANK, [0, 1]))
NEW_ARMS = tuple(ARMS)
STORED_ARMS = dict(U0='U0', BIG_BANK='BIG_BANK', DUO_HOLD='DUO_HOLD')   # A63 name -> A42 confirm arm
LIMIT_S = 5.0
P3_ARMS = ('STRICT_BANK', 'STRICT_PART_L2', 'STRICT_PART_L01', 'BIG_BANK')
BOOTSTRAP = dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')
AUTHORITY = ('MAIN AMENDMENT 63 S-STRICTBANK60 (owner-requested, post hoc, disclosed): BIG_PART_L01, STRICT_BANK, '
             'STRICT_PART_L2, STRICT_PART_L01 on the 96 A42 confirmation worlds, paired with the stored A42 U0 / BIG_BANK '
             '/ DUO_HOLD runs; phase strictbank_a63 cap 480,000')
RULES = dict(
    P1='BIG_PART_L01 - U0 total delay: L0+L1 PAIR SAVES if CI95 upper < 0, else NOT SHOWN',
    P2='STRICT_BANK - STRICT_PART_L2 total delay: OTHER LANES HELP if CI95 upper < 0; HURT if CI95 lower > 0; else '
       'NOT RESOLVED',
    P3=('for STRICT_BANK, STRICT_PART_L2, STRICT_PART_L01 and BIG_BANK: is every registered group mean increase vs U0 '
        '<= +5 s (panel mean, CI95 reported); JOINT-ONLY FEASIBLE if STRICT_BANK meets it and saves delay vs U0 '
        '(CI95 upper < 0) while neither part arm does both; otherwise NOT SHOWN'),
    bootstrap='paired percentile bootstrap 10,000, seed 20261104; diff = treatment - reference')
OVERRIDES = dict(
    A42_PHASE='strictbank_a63', A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A63',
    HOST_BASE=os.environ.get('A63_HOST_BASE', '/root/a63'),
    BANK_DIR='a63_banks', IDENT_WAVE='A63_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A63', JOB_PREFIX='a63',
    VERIFY_TAG='a63v', CONFIRM_TAG='a63c', IDENT_TAG='a63i', VERIFY_WAVE_PREFIX='A63V_', CONFIRM_WAVE_PREFIX='A63C_',
    CANDS_BANK_FILE=f'A63_UNUSED_CANDS_{CELL}.json', BIG_BANK_FILE=BIG_BANK,
    ADMISSION_FILE=f'ADMISSION_A63_UNUSED_{CELL}.json', RESULTS_FILE='STRICTBANK.json', LOG_FILE='A63_LOG.jsonl',
    SCHEMA_PREFIX='coopmwm.a63', RUNNER_RANGES_ATTR='A42_RANGES',
    RESULT_LABEL='amendment 63; strict bank and lane-set parts at 60% CAVs; one shot', AUTHORITY=AUTHORITY)


def _build():
    m = A50._clone('a63_strictbank_core', HERE/'a42_bigp60.py', OVERRIDES)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (m.VERIFY_SETTING, m.CONFIRM_SETTING)
    m.A42Runtime.phase = m.A42_PHASE
    return m


D = _build()                       # A42 driver code, A63 names (own root / phase / cap / ids / stop file)
D42 = A50._clone('a63_a42_view', HERE/'a42_bigp60.py', {})   # the A42 driver itself (A42 names; read-only use)
IDENT_WAVE = 'A63_IDENT_BIG_BANK'
IDENTITY_FILE = 'IDENTITY_A63.json'
SMOKE_FILE = 'SMOKE_A63.json'
BUILD_FILE = 'BUILD_A63.json'


def wave(arm):
    return f'{D.CONFIRM_WAVE_PREFIX}{arm}'


# ------------------------------------------------------------------------------------------------ STRICT_BANK build

def _canon(x):
    return json.dumps(x, sort_keys=True)


def strict_doc(banks, lb95, randplans):
    """Pure (unit-tested): the STRICT_BANK document from the source bank docs ``banks`` {name: doc}, the verified lb95
    {family: value} and the A56 RANDPLANS doc. Templates = the stored template, required identical in every source
    bank that carries the family; rows in the bank_doc row form with the verified lb95; order = (-lb95, strategy_key)."""
    from tools.coopmwm import a37_cellbank as C
    rows, tpls, origin = [], [], {}
    for fam in STRICT_FAMILIES:
        seen = [(n, t) for n, d in banks.items() for t in d['clean5c']['templates'] if t['family'] == fam]
        if not seen or len({_canon(t) for _, t in seen}) != 1:
            raise N.Blocked(f'{fam}: template missing or not identical across sources {[n for n, _ in seen]}')
        tpl = seen[0][1]
        if tpl['cell'] != CELL:
            raise N.Blocked(f'{fam}: template cell {tpl["cell"]}')
        src_rows = [r for d in banks.values() for r in d['rows'] if r['family'] == fam]
        key = tpl['template']['source']['strategy_key']
        row = dict(regime=CELL, family=fam, lb95_veh_s=float(lb95[fam]), plan_source=C.PLAN_SOURCE,
                   row_class='unconditional', situational=dict(unconditional=True), template_sha256=None,
                   strategy_key=key)
        drop = ('lb95_veh_s',)
        if not src_rows or any({k: v for k, v in r.items() if k not in drop} != {k: v for k, v in row.items()
                                                                                    if k not in drop} for r in src_rows):
            raise N.Blocked(f'{fam}: source rows differ from the bank_doc row form (other than lb95)')
        plan = [p for p in randplans['plans'] if p['family'] == fam]
        if plan:   # an A56 random plan: its stored template must be bank_doc of the stored strategy
            try:
                ref = C.bank_doc([dict(family=fam, strategy=plan[0]['strategy'], lb95_veh_s=0.0)], CELL)
            except ValueError as e:
                raise N.Blocked(f'{fam}: RANDPLANS strategy not executable: {e}')
            if _canon(ref['clean5c']['templates'][0]) != _canon(tpl):
                raise N.Blocked(f'{fam}: A56 template is not bank_doc of the RANDPLANS strategy')
        rows.append(row)
        tpls.append(tpl)
        origin[fam] = sorted(n for n, _ in seen)
    order = sorted(range(len(rows)), key=lambda i: (-rows[i]['lb95_veh_s'], rows[i]['strategy_key']))
    doc = dict(schema=C.SCHEMA+'.bank', rows=[rows[i] for i in order],
               clean5c=dict(templates=[tpls[i] for i in order]))
    return doc, origin


def verified_lb95(adm42, adm45, reduce56):
    """{family: verified lb95(gain)} of each STRICT family from its own admission (equal where admitted twice)."""
    out = {}
    for fam in STRICT_FAMILIES:
        vals = [float(v) for adm in (adm42, adm45) for f, v in adm['priority'] if f == fam]
        v56 = (reduce56.get('verdicts') or {}).get(fam)
        if v56 is not None:
            if v56.get('PASS') is not True:
                raise N.Blocked(f'{fam}: not admitted in A56')
            vals.append(float(v56['lb95_gain_veh_s']))
        if not vals or len(set(vals)) != 1:
            raise N.Blocked(f'{fam}: verified lb95 missing or inconsistent {vals}')
        out[fam] = vals[0]
    return out


def build_strict(final_root):
    from tools.coopmwm import a37_cellbank as C
    final_root = Path(final_root)
    shas = {}
    for name, (path, sha) in SOURCES.items():
        got = N.file_sha(Path(path))
        if got != sha:
            raise N.Blocked(f'{name}: {path} sha {got} is not the pinned {sha} (STOP: source not byte-exact)')
        shas[name] = dict(path=path, file_sha256=got)
    adm42 = N.verify(N.read(Path(SOURCES['A42_ADMISSION'][0])), 'receipt_sha256', 'A42 admission')
    if adm42['receipt_sha256'] != A42_ADMISSION_RECEIPT:
        raise N.Blocked('A42 admission receipt is not the registered one')
    adm45 = N.verify(N.read(Path(SOURCES['A45_ADMISSION'][0])), 'receipt_sha256', 'A45 admission')
    red56 = N.verify(N.read(Path(SOURCES['A56_REDUCE'][0])), 'analysis_sha256', 'A56 reduce')
    banks = {n: json.loads(Path(SOURCES[n][0]).read_text(encoding='utf-8')) for n in BANK_SOURCES}
    big_text = Path(SOURCES['A42_BIG_BANK'][0]).read_text(encoding='utf-8')
    if D.C37.bank_text(banks['A42_BIG_BANK']) != big_text:
        raise N.Blocked('BIG_BANK is not in the bank_text form')
    lb95 = verified_lb95(adm42, adm45, red56)
    doc, origin = strict_doc(banks, lb95, json.loads(Path(SOURCES['A56_RANDPLANS'][0]).read_text(encoding='utf-8')))
    core = C.make_core('CELL_BANK', CELL, doc)
    if [p['family'] for p in core.plans] != [r['family'] for r in doc['rows']]:
        raise N.Blocked('executor priority order differs from the bank row order')
    big_copy = D.C37.write_once(D.bank_path(final_root, BIG_BANK), big_text)
    p = D.C37.write_once(D.bank_path(final_root, STRICT_BANK), D.C37.bank_text(doc))
    out = N.seal(dict(schema='coopmwm.a63.build.v1', amendment=AMENDMENT, strict_bank=dict(
        path=str(p), file_sha256=N.file_sha(p), families=[r['family'] for r in doc['rows']],
        priority=[(r['family'], r['lb95_veh_s']) for r in doc['rows']], template_sources=origin),
        big_bank_copy=dict(path=str(big_copy), file_sha256=N.file_sha(big_copy)), sources=shas,
        rule=('templates (offer conditions) copied unchanged and required identical in every source bank; rows in the '
              'a37_cellbank.bank_doc form with the verified lb95 of each plan\'s own admission (A42 / A45 receipts, '
              'A56 reduce); order = admitted_plans rule (lb95 desc, strategy_key); executor accepts all four')),
        'receipt_sha256')
    N.publish(final_root/BUILD_FILE, out, replace=True)
    return out


def prepare(final_root):
    out = {}
    for name, sha in ((BIG_BANK, BIG_BANK_SHA256), (STRICT_BANK, STRICT_BANK_SHA256)):
        p = D.bank_path(final_root, name)
        if sha is None or not p.is_file() or N.file_sha(p) != sha:
            raise N.Blocked(f'{p}: missing or not the registered bank (sha {sha})')
        out[name] = N.file_sha(p)
    if N.file_sha(Path(A42_ROOT)/D42.BANK_DIR/BIG_BANK) != BIG_BANK_SHA256:
        raise N.Blocked('A42 BIG_BANK changed')
    return out


# ------------------------------------------------------------------------------------------------ specs

def _entry(final_root, bank, check_files=True):
    if check_files:
        prepare(final_root)
    return D._bank_entry(final_root, bank, check_files=check_files) or dict(path='', file_sha256='')


def arm_specs(arm, panel, final_root, *, check_files=True):
    from tools.coopmwm import a57_partlane as P
    bank, lanes = ARMS[arm]
    specs = D._a37_specs(D.RUNNER_ARM['BIG_BANK'], panel, final_root, _entry(final_root, bank, check_files),
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


def a42_records(panel):
    """{'U0' | 'BIG_BANK' | 'DUO_HOLD': {world: rec}} from the stored A42 confirmation waves (A42 root, read-only)."""
    have, src = {}, {}
    for arm, name in STORED_ARMS.items():
        specs = D42.confirm_specs(name, panel, A42_ROOT, check_files=True)
        have[arm], src[arm] = D42.C37._load(A42_ROOT, D42.confirm_wave(name), specs)
    return have, src


def identity_index(panel):
    have, _ = a42_records(panel)
    for i, r in enumerate(panel['worlds']):
        if int((have['BIG_BANK'][r['world_id']].get('a37') or {}).get('activations') or 0) > 0:
            return i
    raise N.Blocked('no stored A42 BIG_BANK world with an activation')


def ident_waves(panel, final_root, *, check_files=True):
    i = identity_index(panel)
    s = D._a37_specs(D.RUNNER_ARM['BIG_BANK'], panel, final_root, _entry(final_root, BIG_BANK, check_files),
                     tag=f'{D.IDENT_TAG}_big_bank')[i]
    return [(IDENT_WAVE, 'episode', [dict(s, job_id=f'{D.IDENT_TAG}_big_bank_0000')])]


def _acts(rec):
    a = rec.get('a37') or {}
    return [(x['t'], x['family'], list(x['members']), x.get('release')) for x in a.get('acts') or []]


def identity_compare(final_root, panel):
    final_root = Path(final_root)
    have, _ = a42_records(panel)
    (name, _, specs), = ident_waves(panel, final_root)
    got, _, cov = N.published(final_root, name, specs, D.C37.check_a37)
    if cov != 1:
        raise N.Blocked(f'{name}: {cov}/1 published')
    r = got[0]
    a = have['BIG_BANK'][r['world_id']]
    row = dict(world_id=r['world_id'], seed=r['world']['seed'], j_a63=r['J_veh_s'], j_a42=a['J_veh_s'],
               abs_diff=abs(r['J_veh_s']-a['J_veh_s']), safety_a63=r['safety'], safety_a42=a['safety'],
               safety_equal=r['safety'] == a['safety'], same_world=r['world'] == a['world'],
               activations=[r['a37'].get('activations'), a['a37'].get('activations')],
               acts_equal=_acts(r) == _acts(a), a37_sha256_equal=r['a37_sha256_loaded'] == a['a37_sha256_loaded'],
               no_filter=('a57_part' not in r['a37']))
    ok = (row['abs_diff'] == 0.0 and row['safety_equal'] and row['same_world'] and row['acts_equal']
          and row['a37_sha256_equal'] and row['no_filter'])
    return N.seal(dict(schema='coopmwm.a63.identity.v1', amendment=AMENDMENT,
                       rule=('A63 code path (lane-set filter code present, filter off) reproduces the stored A42 '
                             'BIG_BANK record of the identity world exactly: J_veh_s difference 0, identical safety '
                             'counters, identical activation receipts, same a37 digest'),
                       verdict='PASS' if ok else 'FAILED', world=row,
                       consequence='PASS: stored A42 U0 / BIG_BANK / DUO_HOLD reused; FAILED: stop and report'),
                  'receipt_sha256')


def _passed(final_root, name):
    p = Path(final_root)/name
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


def bank_families(final_root, bank):
    return {r['family'] for r in json.loads(D.bank_path(final_root, bank).read_text(encoding='utf-8'))['rows']}


def part_receipt_ok(part, lanes, digest, families):
    """Pure (unit-tested): the a57_part receipt shows that only members planned on ``lanes`` were commanded."""
    lanes = sorted(lanes)
    on = {v for a in part.get('acts', ()) for v in a['on_lane']}
    cmd = part.get('commanded_lanes') or {}
    return bool(part.get('member_lanes') == lanes and part.get('a57_sha256') == digest
                and all(set(a['on_lane']) <= set(a['members']) for a in part.get('acts', ()))
                and all(a['family'] in families for a in part.get('acts', ()))
                and all(int(x) in lanes for x in cmd.values()) and set(cmd) <= on
                and len(cmd) == part.get('vehicles_with_accel_command'))


def smoke_check(final_root, panel):
    """Zero physics, outcome-blind: each smoke record is valid (check_a37); part arms carry the a57_part receipt of
    their own lane set with the registered digest, every commanded vehicle was planned on an intended lane and was an
    on-lane member of an activation; STRICT_BANK carries no filter receipt; activations come from the arm's bank."""
    from tools.coopmwm import a57_partlane as P
    rows, ok = [], True
    for (name, _, specs), arm in zip(arm_waves(panel, final_root, first_only=True), NEW_ARMS):
        got, _, cov = N.published(final_root, name, specs, D.C37.check_a37)
        if cov != 1:
            raise N.Blocked(f'{name}: smoke episode not published')
        bank, lanes = ARMS[arm]
        fams = bank_families(final_root, bank)
        a37 = got[0].get('a37') or {}
        acts_fams = sorted({x['family'] for x in a37.get('acts') or []})
        row = dict(wave=name, world_id=got[0]['world_id'], bank=bank, activations=a37.get('activations'),
                   families_activated=acts_fams, members_planned=sum(len(x['members']) for x in a37.get('acts') or []))
        if lanes is None:
            good = 'a57_part' not in a37 and set(acts_fams) <= fams
        else:
            part = a37.get('a57_part') or {}
            good = part_receipt_ok(part, lanes, P.a57_digest(), fams)
            row.update(member_lanes=part.get('member_lanes'), members_on_lanes=part.get('members_on_lane_total'),
                       no_member_on_lanes=part.get('activations_no_member_on_lane'),
                       commanded_vehicles=part.get('vehicles_with_accel_command'),
                       commanded_by_planned_lane=dict(Counter(str(x) for x in
                                                              (part.get('commanded_lanes') or {}).values())),
                       kept_accel=part.get('kept_accel'), dropped_accel=part.get('dropped_accel'))
        row['receipt_ok'] = bool(good)
        ok = ok and good
        rows.append(row)
    return N.seal(dict(schema='coopmwm.a63.smoke.v1', verdict='PASS' if ok else 'FAILED', rows=rows,
                       rule=('record valid; part arms: a57_part receipt of the intended lane set + registered digest, '
                             'every commanded vehicle planned on an intended lane; STRICT_BANK: no filter receipt; '
                             'activated families from the arm\'s bank (no outcome read)')), 'receipt_sha256')


# ------------------------------------------------------------------------------------------------ analysis

def p1_verdict(ci):
    return 'L0+L1 PAIR SAVES' if ci[1] < 0 else 'NOT SHOWN'


def p2_verdict(ci):
    return 'OTHER LANES HELP' if ci[1] < 0 else 'HURT' if ci[0] > 0 else 'NOT RESOLVED'


def worst_group(groups):
    """(group, increase_mean_s, ci95) of the largest registered-group mean increase; groups = co_primary_groups."""
    reg = {g: v for g, v in groups['groups'].items() if g in N.CO_PRIMARY_GROUPS}
    if len(reg) != len(N.CO_PRIMARY_GROUPS) or any('increase_mean_s' not in v for v in reg.values()):
        raise N.Blocked('a registered group is unavailable')
    g = max(reg, key=lambda k: reg[k]['increase_mean_s'])
    return g, float(reg[g]['increase_mean_s']), list(reg[g]['ci95'])


def p3_verdict(rows):
    """rows = {arm: dict(meets_limit, saves)}: JOINT-ONLY FEASIBLE iff STRICT_BANK meets and saves while neither
    STRICT part arm does both."""
    s = rows['STRICT_BANK']
    both = [rows[a]['meets_limit'] and rows[a]['saves'] for a in ('STRICT_PART_L2', 'STRICT_PART_L01')]
    return 'JOINT-ONLY FEASIBLE' if (s['meets_limit'] and s['saves'] and not any(both)) else 'NOT SHOWN'


def _mean_ci(x, bootstrap):
    import numpy as np
    x = np.asarray(x, float)
    s = N.fe.paired_stats(x, np.zeros_like(x), bootstrap, pct=False)
    return dict(mean=s['diff'], ci95=s['ci95'], p_two_sided=s['p_two_sided'], worlds_negative=s['worlds_lower'],
                worlds_positive=s['worlds_higher'])


def _reports_readonly(root, have, worlds):
    from a32_ablation import report_of
    out = {}
    for a in have:
        out[a] = {}
        for w in worlds:
            rec = have[a][w]
            d = Path(root[a])/'evaluation'/rec.get('work_dir', rec['job_id'])
            if str(root[a]) == A42_ROOT and not (d/'vehicles_report.json').is_file():
                raise N.Blocked(f'{d}: no cached report (A42 root is read-only here)')
            out[a][w] = report_of(root[a], rec)
            if not out[a][w]['ok']:
                raise N.Blocked(f'{rec["job_id"]}: invalid vehicle-log report')
    return out


def _arm_counts(arm, recs, worlds):
    rows = [(recs[w].get('a37') or {}) for w in worlds]
    planned = sum(len(x['members']) for r in rows for x in (r.get('acts') or []))
    out = dict(activations=sum(int(r.get('activations') or 0) for r in rows),
               worlds_with_activation=sum(1 for r in rows if r.get('activations')), members_planned_total=planned,
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
    prepare(final_root)
    have, sources = a42_records(panel)
    roots = {a: A42_ROOT for a in have}
    for (name, _, specs), arm in zip(arm_waves(panel, final_root), NEW_ARMS):
        have[arm], sources[arm] = D.C37._load(final_root, name, specs)
        roots[arm] = str(final_root)
        bank, lanes = ARMS[arm]
        fams = bank_families(final_root, bank)
        for w in worlds:
            a37 = have[arm][w].get('a37') or {}
            if not {x['family'] for x in a37.get('acts') or []} <= fams:
                raise N.Blocked(f'{arm} {w}: activation of a family outside its bank')
            if lanes is None:
                if 'a57_part' in a37:
                    raise N.Blocked(f'{arm} {w}: unexpected filter receipt')
            elif not part_receipt_ok(a37.get('a57_part') or {}, lanes, P.a57_digest(), fams):
                raise N.Blocked(f'{arm} {w}: a57_part receipt missing / wrong lanes / wrong digest')
    for a in STORED_ARMS:
        if any('a57_part' in (have[a][w].get('a37') or {}) for w in worlds):
            raise N.Blocked(f'{a}: stored record carries a filter receipt')
    reports = _reports_readonly(roots, have, worlds)
    arms = tuple(STORED_ARMS)+NEW_ARMS
    J = {a: np.array([have[a][w]['J_veh_s'] for w in worlds], float) for a in arms}
    HB = {a: np.array([have[a][w]['safety']['hard_brakes'] for w in worlds], float) for a in arms}
    vs_u0 = {a: N._bank_compare(have, reports, worlds, a, 'U0', bootstrap) for a in arms if a != 'U0'}
    p1 = N.fe.paired_stats(J['BIG_PART_L01'], J['U0'], bootstrap)
    p1.update(rule=RULES['P1'], verdict=p1_verdict(p1['ci95']))
    p2 = N.fe.paired_stats(J['STRICT_BANK'], J['STRICT_PART_L2'], bootstrap)
    p2.update(rule=RULES['P2'], verdict=p2_verdict(p2['ci95']))
    p3_rows = {}
    for a in P3_ARMS:
        g, inc, ci = worst_group(vs_u0[a]['groups'])
        regs = {k: dict(increase_mean_s=v['increase_mean_s'], ci95=v['ci95'],
                        within_limit=bool(v['increase_mean_s'] <= LIMIT_S))
                for k, v in vs_u0[a]['groups']['groups'].items() if k in N.CO_PRIMARY_GROUPS}
        p3_rows[a] = dict(worst_group=g, worst_increase_s=inc, worst_ci95=ci,
                          meets_limit=all(v['within_limit'] for v in regs.values()),
                          saves=bool(vs_u0[a]['total_lower']), total_minus_u0=vs_u0[a]['total_minus_ref']['diff'],
                          total_ci95=vs_u0[a]['total_minus_ref']['ci95'], groups=regs)
    p3 = dict(rule=RULES['P3'], limit_s=LIMIT_S, arms=p3_rows, verdict=p3_verdict(p3_rows))
    group_table = {a: dict(groups={k: dict(increase_mean_s=v.get('increase_mean_s'), ci95=v.get('ci95'))
                                   for k, v in vs_u0[a]['groups']['groups'].items()},
                           worst=worst_group(vs_u0[a]['groups'])) for a in vs_u0}
    sec = dict(
        strict_vs_u0=vs_u0['STRICT_BANK'],
        strict_vs_big_bank=N._bank_compare(have, reports, worlds, 'STRICT_BANK', 'BIG_BANK', bootstrap),
        strict_vs_duo_hold=N._bank_compare(have, reports, worlds, 'STRICT_BANK', 'DUO_HOLD', bootstrap),
        arms_vs_u0={a: dict(total=vs_u0[a]['total_minus_ref'], tail_cvar90=vs_u0[a]['tail_cvar90'],
                            groups_all_within_10s=vs_u0[a]['groups'].get('all_pass')) for a in vs_u0},
        hard_brake_minus_u0_veh_s={a: _mean_ci(HB[a]-HB['U0'], bootstrap) for a in arms if a != 'U0'},
        group_table_vs_u0=group_table,
        strict_part_l01_minus_u0_total=vs_u0['STRICT_PART_L01']['total_minus_ref'],
        strict_minus_part_l01_total=N.fe.paired_stats(J['STRICT_BANK'], J['STRICT_PART_L01'], bootstrap))
    arm_rows = {a: dict(mean_J_veh_s=float(J[a].mean()),
                        safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY},
                        **_arm_counts(a, have[a], worlds)) for a in arms}
    per_world = [dict(world_id=w, seed=seed[w], **{f'J_{a}': float(have[a][w]['J_veh_s']) for a in arms},
                      **{f'hard_brakes_{a}': int(HB[a][i]) for a in arms},
                      **{f'acts_{a}': int((have[a][w].get('a37') or {}).get('activations') or 0)
                         for a in arms if a != 'U0'},
                      **{f'commanded_{a}': int(have[a][w]['a37']['a57_part']['members_on_lane_total'])
                         for a in NEW_ARMS if ARMS[a][1] is not None},
                      **{f'collisions_{a}': int(have[a][w]['safety']['collisions']) for a in arms},
                      **{f'teleports_{a}': int(have[a][w]['safety']['teleports']) for a in arms})
                 for i, w in enumerate(worlds)]
    build = N.verify(N.read(final_root/BUILD_FILE), 'receipt_sha256', BUILD_FILE)
    return N.seal(dict(schema='coopmwm.a63.strictbank.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       setting=SETTING, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), bootstrap=bootstrap,
                       convention='diff = treatment - reference; negative = less delay', rules=RULES,
                       primary=dict(P1=p1, P2=p2, P3=p3), secondary=sec, arms=arm_rows, per_world=per_world,
                       a57_sha256=P.a57_digest(), strict_bank_sha256=STRICT_BANK_SHA256,
                       big_bank_sha256=BIG_BANK_SHA256, build_receipt=build['receipt_sha256'],
                       a42_admission_receipt=A42_ADMISSION_RECEIPT,
                       reused={a: f'{A42_ROOT} WAVE_A42C_{n}' for a, n in STORED_ARMS.items()},
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
    p.add_argument('command', choices=('build-strict', 'prepare', 'dry', 'identity', 'identity-compare', 'smoke',
                                       'smoke-check', 'run', 'analyse', 'status', 'budget'))
    p.add_argument('--final-root')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--chunk', type=int, default=16)
    a = p.parse_args(argv)
    if a.command == 'budget':
        print(json.dumps(budget_plan(), indent=1))
        return
    if not a.final_root:
        p.error('--final-root required')
    root = Path(a.final_root)
    if a.command == 'build-strict':
        print(json.dumps(build_strict(root), indent=1))
        return
    panel = D.setup(SETTING)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=SETTING), sort_keys=True,
                          default=str)
        print(line, flush=True)
        with (root/D.LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    if a.command == 'prepare':
        print(json.dumps(prepare(root)))
    elif a.command == 'dry':
        iw = ident_waves(panel, root)
        aw = arm_waves(panel, root)
        ref = D42.confirm_specs('BIG_BANK', panel, A42_ROOT, check_files=True)
        i = identity_index(panel)
        mine = iw[0][2][0]

        def norm(s, drop_bank=False):
            s = dict(s)
            s.pop('job_id', None)
            ck = {k: v for k, v in s['checkpoint'].items() if k not in ('a57_member_lane', 'a57_sha256')}
            if drop_bank:
                ck.pop('a37_bank')
            s['checkpoint'] = ck
            t = json.dumps(s, sort_keys=True, default=str)
            t = t.replace(f'{D.HOST_ROOT}/{D.BANK_DIR}', '<bank>').replace(f'{A42_ROOT}/{D42.BANK_DIR}', '<bank>')
            return t.replace(str(root), '<root>').replace(D.HOST_ROOT, '<root>').replace(A42_ROOT, '<root>')
        print(json.dumps(dict(
            budget=budget_plan(), charged=D.charged(root), identity_index=i, identity_world=mine['world_id'],
            identity_spec_equals_a42_except_job_id_and_bank_path=norm(mine) == norm(ref[i]),
            arms=[(n, len(sp), sp[0]['job_id'], sp[0]['checkpoint']['a37_bank'],
                   sp[0]['checkpoint'].get('a57_member_lane'), (sp[0]['checkpoint'].get('a57_sha256') or '')[:12],
                   [sp[0]['world']['seed'], sp[-1]['world']['seed']]) for n, _, sp in aw],
            arm_specs_equal_a42_big_bank_except_filter_and_bank=all(
                norm(s, True) == norm(r, True) for _, _, sp in aw for s, r in zip(sp, ref)),
            big_part_bank_equals_a42_bank_bytes=aw[0][2][0]['checkpoint']['a37_bank']['file_sha256'] == BIG_BANK_SHA256,
            a37_sha256=mine['checkpoint']['a37_sha256'], worlds=len(panel['worlds'])), indent=1, default=str))
    elif a.command in ('identity', 'smoke', 'run'):
        import fcntl
        if a.command != 'identity' and not _passed(root, IDENTITY_FILE):
            raise N.Blocked(f'{IDENTITY_FILE} is not PASS: the A63 arms are not run')
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
            # no automatic replay (A63 registration): a failed dispatch raises and is reported
            res = D.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(root, panel)
        N.publish(root/IDENTITY_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], world=v['world']['world_id'], abs_diff=v['world']['abs_diff'],
                              safety_equal=v['world']['safety_equal'], acts_equal=v['world']['acts_equal'],
                              activations=v['world']['activations'])))
    elif a.command == 'smoke-check':
        v = smoke_check(root, panel)
        N.publish(root/SMOKE_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], rows=v['rows'])))
    elif a.command == 'analyse':
        v = analyse(root, panel)
        out = Path(a.out or root)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/'STRICTBANK.json', v, replace=True)
        write_table(v, out/'STRICTBANK_PER_WORLD.csv')
        pr = v['primary']
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'],
                              P1=[pr['P1']['verdict'], pr['P1']['diff'], pr['P1']['ci95']],
                              P2=[pr['P2']['verdict'], pr['P2']['diff'], pr['P2']['ci95']], P3=pr['P3']['verdict'])))
    else:   # status
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, identity_pass=_passed(root, IDENTITY_FILE),
                   smoke_pass=_passed(root, SMOKE_FILE))
        for name, _, specs in ident_waves(panel, root)+arm_waves(panel, root):
            out[name] = f'{N.published(root, name, specs, D.C37.check_a37)[2]}/{len(specs)}'
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
