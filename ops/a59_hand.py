"""AMENDMENTS 59 / 59a / 59b (registered before any HAND outcome): hand-written comparators at D2800_p60.

HAND5 (A59/59a): trigger 'lanes', zone full, spread 200 m, cooldown 120 s, slots 1, no gate/start_gate/phases; roles
SUMO L1 rank 0 + rank 1, L2 rank 0 + rank 1, L0 rank 0; every member 20 m/s, dmax 1.0, hold 120 s, onset 0.
HAND3 (A59b): duo12_v20 unchanged (trigger 'wall', zone full, spread 200, cooldown 120, slots 1, no gate) + one role in
the closing lane L0: one member per lane L1 / L2 / L0, each 20 m/s, dmax 1.0, hold 120 s, onset 0.

Both run through the A37_PLAN executor (one unconditional plan, started whenever feasible) on frozen MAPPO U0, exactly as
the searched A42 plans were verified. Comparator runs are the stored A42 records (A42 root /root/a42v/root, read-only).

This driver IS ops/a42_bigp60.py (private module copy, as A50 / A56 / A57) with the A59 names: own final root
/root/a59/root, ledger phase hand_a59 (phase cap 525,000 = HAND5 265,000 + HAND3 260,000; per-arm caps checked by job
id), job ids a59*, stop file STOP_A59. No automatic replay.

Commands (python ops/a59_hand.py <cmd> --final-root <root> [--arm HAND3|HAND5]):
  prepare          zero physics: a59_banks/A59_<ARM>_D2800_p60.json + executor acceptance check
  dry              spec summary (identity spec vs the A42 spec, bank entries, seeds)
  identity         1 episode: U0 on verification world index 0 (charged to HAND5)
  identity-compare -> IDENTITY_A59.json (J diff 0, identical safety, same world)
  smoke --n K      first K worlds of the arm's verification wave (published as part of that wave, not run twice)
  smoke-check      zero physics, outcome-blind: per activation, distinct members commanded ('setpoint') on the
                   registered lanes (frame calibrated on a stored A42 plan record) -> SMOKE_A59_<ARM>.json
  verify           the arm on the 24 verification worlds (refused unless identity + smoke PASS)
  admit            zero physics: UNC rule vs the stored A42 U0 -> ADMISSION_A59_<ARM>.json
  confirm          the arm on the 96 panel-1 worlds (setting a42c; refused unless smoke PASS)
  analyse          zero physics (setting a42c): primary BIG_BANK - ARM + secondaries -> --out
  status
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
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

AMENDMENT = '59'
CELL = 'D2800_p60'
VSET, CSET = 'a42v_D2800_p60', 'a42c_D2800_p60'
A42_ROOT = '/root/a42v/root'
A42_ADMISSION_RECEIPT = '5fa2a7015fb4ae9a95e1d17456e5caacb6114405cc197831d2c3d674f1b332d5'
CAL_FAMILY, CAL_KEY = 'rand_3a452b1348b01083', '3a452b1348b01083'     # stored A42 plan used for the lane-frame check
BOOTSTRAP = dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')
CAPS = dict(HAND5=265_000, HAND3=260_000)
PHASE_CAP = sum(CAPS.values())
ARM_TAG = dict(HAND5='h5', HAND3='h3')


def _role(lane, rank):
    return dict(lane=lane, rank=rank, v_w=20.0, dmax=1.0, hold_s=120, onset_s=0, lc_target=None)


SPECS = dict(
    HAND5=dict(name='HAND5', trigger='lanes', zone='full', spread_m=200.0, cooldown_s=120, slots=1, gate=None,
               start_gate=None, phases=None, roles=[_role(1, 0), _role(1, 1), _role(2, 0), _role(2, 1), _role(0, 0)]),
    HAND3=dict(name='HAND3', trigger='wall', zone='full', spread_m=200.0, cooldown_s=120, slots=1, gate=None,
               start_gate=None, phases=None, roles=[_role(1, 0), _role(2, 0), _role(0, 0)]))
EXPECT_LANES = {a: dict(Counter(r['lane'] for r in s['roles'])) for a, s in SPECS.items()}
AUTHORITY = ('MAIN AMENDMENTS 59 / 59a / 59b (owner 2026-11-11): hand-written HAND5 (lanes trigger) and HAND3 (duo12_v20 '
             '+ L0 role) at D2800_p60 through the A37_PLAN executor; admission on the 24 A42 verification worlds vs '
             'stored U0; 96 panel-1 worlds; PRIMARY BIG_BANK - HAND; caps HAND5 265,000, HAND3 260,000')
OVERRIDES = dict(
    A42_PHASE='hand_a59', A42_CAP_SIM_S=PHASE_CAP, STOP_A42='STOP_A59',
    HOST_BASE=os.environ.get('A59_HOST_BASE', '/root/a59'),
    BANK_DIR='a59_banks', IDENT_WAVE='A59_IDENT_U0', AMENDMENT=AMENDMENT, LABEL='A59', JOB_PREFIX='a59',
    VERIFY_TAG='a59v', CONFIRM_TAG='a59c', IDENT_TAG='a59i', VERIFY_WAVE_PREFIX='A59V_', CONFIRM_WAVE_PREFIX='A59C_',
    CANDS_BANK_FILE=f'A59_UNUSED_{CELL}.json', BIG_BANK_FILE=f'A59_UNUSED_BIG_{CELL}.json',
    ADMISSION_FILE=f'ADMISSION_A59_UNUSED_{CELL}.json', RESULTS_FILE='HAND.json', LOG_FILE='A59_LOG.jsonl',
    SCHEMA_PREFIX='coopmwm.a59', RUNNER_RANGES_ATTR='A42_RANGES',
    RESULT_LABEL='amendment 59; hand-written comparators at 60% CAVs; one shot', AUTHORITY=AUTHORITY)


def _build():
    m = A50._clone('a59_hand_core', HERE/'a42_bigp60.py', OVERRIDES)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (m.VERIFY_SETTING, m.CONFIRM_SETTING)
    m.A42Runtime.phase = m.A42_PHASE
    return m


D = _build()
D42 = A50._clone('a59_a42_view', HERE/'a42_bigp60.py', {})
IDENTITY_FILE = 'IDENTITY_A59.json'


def bank_name(arm):
    return f'A59_{arm}_{CELL}.json'


def arm_charged(root, arm):
    rows = N.fe.ledger_rows(root)
    tag = ARM_TAG[arm]
    return sum(r['sim_s'] for r in rows
               if f'_{tag}_' in str(r['id']) or (arm == 'HAND5' and str(r['id']).startswith('a59i')))


# ------------------------------------------------------------------------------------------------ bank / specs

def prepare(final_root, arm):
    from tools.coopmwm import a37_cellbank as C
    s = SPECS[arm]
    doc = C.bank_doc([dict(family=arm, strategy=s, lb95_veh_s=0.0)], CELL)
    p = D.C37.write_once(D.bank_path(final_root, bank_name(arm)), D.C37.bank_text(doc))
    core = C.make_core('A37_PLAN', CELL, bank=json.loads(p.read_text(encoding='utf-8')), family=arm)
    if [x['family'] for x in core.plans] != [arm] or core.refused:
        raise N.Blocked(f'{arm}: executor refused {core.refused}')
    if core.plans[0]['spec'] != s:
        raise N.Blocked(f'{arm}: executor spec differs from the registered spec')
    return dict(path=str(p), file_sha256=N.file_sha(p), strategy_key=core.plans[0]['key'],
                spec_sha256=C.spec_sha(s), executor_spec=core.plans[0]['spec'], a37_sha256=C.a37_digest())


def _entry(final_root, arm, check_files=True):
    return D._bank_entry(final_root, bank_name(arm), check_files=check_files) or dict(path='', file_sha256='')


def verify_wave(arm, panel, final_root, *, check_files=True):
    return (f'{D.VERIFY_WAVE_PREFIX}{arm}', 'episode',
            D._a37_specs(D.RUNNER_ARM['PLAN'], panel, final_root, _entry(final_root, arm, check_files), family=arm,
                         tag=f'{D.VERIFY_TAG}_{ARM_TAG[arm]}'))


def confirm_wave(arm, panel, final_root, *, check_files=True):
    return (f'{D.CONFIRM_WAVE_PREFIX}{arm}', 'episode',
            D._a37_specs(D.RUNNER_ARM['PLAN'], panel, final_root, _entry(final_root, arm, check_files), family=arm,
                         tag=f'{D.CONFIRM_TAG}_{ARM_TAG[arm]}'))


def ident_wave(panel, final_root):
    u0 = D.verify_waves(panel, final_root, check_files=False, cands=[])[0][2]
    return (D.IDENT_WAVE, 'episode', [dict(u0[0], job_id=f'{D.IDENT_TAG}_u0_0000')])


def a42_u0_verify(panel):
    name, _, specs = D42.verify_waves(panel, A42_ROOT, check_files=True, cands=[])[0]
    return D42.C37._load(A42_ROOT, name, specs), specs


def _norm(s, root):
    s = dict(s)
    s.pop('job_id', None)
    t = json.dumps(s, sort_keys=True, default=str)
    return t.replace(str(root), '<root>').replace(D.HOST_ROOT, '<root>').replace(A42_ROOT, '<root>')


# ------------------------------------------------------------------------------------------------ identity

def identity_compare(final_root, panel):
    final_root = Path(final_root)
    (have, _), a42specs = a42_u0_verify(panel)
    name, _, specs = ident_wave(panel, final_root)
    got, _, cov = N.published(final_root, name, specs, D.C37.check_a37)
    if cov != 1:
        raise N.Blocked(f'{name}: {cov}/1 published')
    r = got[0]
    a = have[r['world_id']]
    row = dict(world_id=r['world_id'], seed=r['world']['seed'], j_a59=r['J_veh_s'], j_a42=a['J_veh_s'],
               abs_diff=abs(r['J_veh_s']-a['J_veh_s']), safety_a59=r['safety'], safety_a42=a['safety'],
               safety_equal=r['safety'] == a['safety'], same_world=r['world'] == a['world'],
               spec_equal_except_job_and_root=_norm(specs[0], final_root) == _norm(a42specs[0], final_root))
    ok = row['abs_diff'] == 0.0 and row['safety_equal'] and row['same_world'] and row['spec_equal_except_job_and_root']
    return N.seal(dict(schema='coopmwm.a59.identity.v1', amendment=AMENDMENT,
                       rule='A59 path reproduces the stored A42 verification U0 record of world index 0 exactly '
                            '(J_veh_s diff 0, identical safety counters, same world, same spec)',
                       verdict='PASS' if ok else 'FAILED', world=row,
                       consequence='PASS: stored A42 records reused; FAILED: stop and report'), 'receipt_sha256')


def passed(final_root, name):
    p = Path(final_root)/name
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ------------------------------------------------------------------------------------------------ smoke check

def _obs(root, rec):
    d = Path(root)/'evaluation'/rec.get('work_dir', rec['job_id'])
    out = {}
    with gzip.open(d/'observable.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            x = json.loads(line)
            out[int(round(float(x['time_s'])))] = {v['vid']: v for v in x['vehicles']}
    return out


def _act_rows(root, rec, roles=None, window=2):
    obs = _obs(root, rec)
    rows = []
    for a in (rec.get('a37') or {}).get('acts') or []:
        t = int(a['t'])
        mem = list(a['members'])
        lane_at, setpoint = {}, {}
        for vid in mem:
            for k in range(t, t+window+1):
                v = (obs.get(k) or {}).get(vid)
                if v is None:
                    continue
                lane_at.setdefault(vid, int(v['lane']))
                if v.get('controller') == 'setpoint':
                    setpoint[vid] = k
                    break
        rows.append(dict(t=t, members=mem, distinct=len(set(mem)), obs_lane=[lane_at.get(v) for v in mem],
                         setpoint_at=[setpoint.get(v) for v in mem],
                         role_lanes=None if roles is None else [r['lane'] for r in roles][:len(mem)]))
    return rows


def lane_calibration(panel):
    """Stored A42 verification records of CAL_FAMILY (first 4 worlds): observed lane vs the role's SUMO lane."""
    from tools.coopmwm import a37_cellbank as C  # noqa: F401
    from tools.coopmwm import oracle_plan as O
    bank = json.loads((Path(A42_ROOT)/D42.BANK_DIR/D42.CANDS_BANK_FILE).read_text(encoding='utf-8'))
    tpl = next(t for t in bank['clean5c']['templates'] if t['family'] == CAL_FAMILY)
    roles = O.spec_of(CAL_FAMILY, tpl['offer_condition'])['roles']
    name, _, specs = D42.verify_waves(panel, A42_ROOT, check_files=True,
                                      cands=[dict(family=CAL_FAMILY, key=CAL_KEY)])[1]
    have, _ = D42.C37._load(A42_ROOT, name, specs)
    pairs = Counter()
    for w in [s['world_id'] for s in specs[:4]]:
        rec = have[w]
        for row in _act_rows(A42_ROOT, rec, roles):
            for ol, rl in zip(row['obs_lane'], row['role_lanes']):
                pairs[(rl, ol)] += 1
    ok = bool(pairs) and all(ol == 2-rl for (rl, ol) in pairs)
    return dict(pairs={f'sumo{rl}->obs{ol}': n for (rl, ol), n in sorted(pairs.items(), key=str)},
                frame_sumo_eq_2_minus_obs=ok)


def smoke_check(final_root, panel, arm, n):
    final_root = Path(final_root)
    cal = lane_calibration(panel)
    name, _, specs = verify_wave(arm, panel, final_root)
    got, _, cov = N.published(final_root, name, specs[:n], D.C37.check_a37)
    if cov != n:
        raise N.Blocked(f'{name}: {cov}/{n} smoke episodes published')
    want = {str(k): v for k, v in EXPECT_LANES[arm].items()}
    n_mem = len(SPECS[arm]['roles'])
    acts = []
    for rec in got:
        for row in _act_rows(final_root, rec):
            sumo = [None if x is None else 2-x for x in row['obs_lane']]
            lanes = {str(k): v for k, v in Counter(sumo).items()}
            good = (row['distinct'] == n_mem and all(x is not None for x in row['setpoint_at'])
                    and lanes == want)
            acts.append(dict(row, world_id=rec['world_id'], sumo_lane=sumo, lane_counts=lanes, ok=good))
    ok = cal['frame_sumo_eq_2_minus_obs'] and bool(acts) and all(a['ok'] for a in acts)
    return N.seal(dict(schema='coopmwm.a59.smoke.v1', arm=arm, verdict='PASS' if ok else 'FAILED',
                       rule=(f'every activation in the smoke episodes has {n_mem} distinct members, each commanded '
                             f'(controller setpoint within 2 s of activation), SUMO lane counts {want}; at least one '
                             'activation; lane frame checked on stored A42 records (no outcome read)'),
                       lane_calibration=cal, n_episodes=n, activations=len(acts), acts=acts), 'receipt_sha256')


# ------------------------------------------------------------------------------------------------ admission / analysis

def _reports_readonly(roots, have, worlds):
    from a32_ablation import report_of
    out = {}
    for a in have:
        out[a] = {}
        for w in worlds:
            rec = have[a][w]
            d = Path(roots[a])/'evaluation'/rec.get('work_dir', rec['job_id'])
            if str(roots[a]) == A42_ROOT and not (d/'vehicles_report.json').is_file():
                raise N.Blocked(f'{d}: no cached report (A42 root is read-only here)')
            out[a][w] = report_of(roots[a], rec)
            if not out[a][w]['ok']:
                raise N.Blocked(f'{rec["job_id"]}: invalid vehicle-log report')
    return out


def _acts_summary(arm, recs, worlds):
    rows = [(recs[w].get('a37') or {}) for w in worlds]
    return dict(activations=sum(int(r.get('activations') or 0) for r in rows),
                worlds_with_activation=sum(1 for r in rows if r.get('activations')),
                group_size=D._group_sizes(recs, worlds, {arm: SPECS[arm]}),
                full_group_n=len(SPECS[arm]['roles']))


def admit(final_root, panel, arm):
    from tools.coopmwm import a37_cellbank as C
    final_root = Path(final_root)
    out = final_root/f'ADMISSION_A59_{arm}.json'
    if out.exists():
        raise N.Blocked(f'{out.name} exists (one shot)')
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    (u0, src_u0), _ = a42_u0_verify(panel)
    name, _, specs = verify_wave(arm, panel, final_root)
    mine, src = D.C37._load(final_root, name, specs)
    have = {'U0': u0, arm: mine}
    reports = _reports_readonly({'U0': A42_ROOT, arm: str(final_root)}, have, worlds)
    cmp_ = N._bank_compare(have, reports, worlds, arm, 'U0', bootstrap)
    v = C.unc_verdict({seed[w]: mine[w]['J_veh_s'] for w in worlds}, {seed[w]: u0[w]['J_veh_s'] for w in worlds},
                      cmp_['groups'])
    v.update(family=arm, groups=cmp_['groups'], cvar90_vs_u0=cmp_['tail_cvar90'],
             total_minus_u0_secondary=cmp_['total_minus_ref'],
             safety_totals={k: int(sum(mine[w]['safety'][k] for w in worlds)) for k in N.SAFETY},
             u0_safety_totals={k: int(sum(u0[w]['safety'][k] for w in worlds)) for k in N.SAFETY},
             **_acts_summary(arm, mine, worlds))
    per_world = [dict(world_id=w, seed=seed[w], J_U0=u0[w]['J_veh_s'], **{f'J_{arm}': mine[w]['J_veh_s']},
                      acts=int((mine[w].get('a37') or {}).get('activations') or 0)) for w in worlds]
    value = N.seal(dict(schema='coopmwm.a59.admission.v1', amendment=AMENDMENT, arm=arm, cell=CELL, setting=VSET,
                        rule=C.UNC_RULE, panel_sha256=panel['panel_sha256'], verdict=v,
                        admitted=bool(v['PASS']), per_world=per_world,
                        sources=dict(U0=src_u0.get('receipt_sha256'), **{arm: src.get('receipt_sha256')}),
                        consequence='reported either way; the 96-world confirmation runs regardless (comparator)'),
                   'receipt_sha256')
    N.publish(out, value)
    return value


def verdict_of(ci, arm):
    return 'SEARCH AHEAD' if ci[1] < 0 else f'{arm} AHEAD' if ci[0] > 0 else 'NOT RESOLVED'


def analyse(final_root, panel, arm):
    import numpy as np
    final_root = Path(final_root)
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != BOOTSTRAP or len(worlds) != 96:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    have, sources, roots = {}, {}, {}
    for a in ('U0', 'DUO_HOLD', 'BIG_BANK'):
        specs = D42.confirm_specs(a, panel, A42_ROOT, check_files=True)
        have[a], sources[a] = D42.C37._load(A42_ROOT, D42.confirm_wave(a), specs)
        roots[a] = A42_ROOT
    arms_mine = [arm]
    other = 'HAND5' if arm == 'HAND3' else 'HAND3'
    oname, _, ospecs = confirm_wave(other, panel, final_root, check_files=False)
    if N.published(final_root, oname, ospecs, D.C37.check_a37)[2] == len(ospecs):
        arms_mine.append(other)
    for a in arms_mine:
        name, _, specs = confirm_wave(a, panel, final_root)
        have[a], sources[a] = D.C37._load(final_root, name, specs)
        roots[a] = str(final_root)
    reports = _reports_readonly(roots, have, worlds)
    con = [D._contrast(have, reports, worlds, 'BIG_BANK', arm, bootstrap, 'primary'),
           D._contrast(have, reports, worlds, arm, 'U0', bootstrap, 'secondary'),
           D._contrast(have, reports, worlds, arm, 'DUO_HOLD', bootstrap, 'secondary')]
    if arm == 'HAND3' and 'HAND5' in arms_mine:
        con.append(D._contrast(have, reports, worlds, 'HAND3', 'HAND5', bootstrap, 'secondary'))
    ci = con[0]['total_minus_ref']['ci95']
    g = con[1]['groups'] or {}
    prim = dict(rule=(f'BIG_BANK - {arm} total delay per world, paired percentile bootstrap 10,000 seed 20261104: '
                      f'SEARCH AHEAD if CI upper < 0, {arm} AHEAD if CI lower > 0, else NOT RESOLVED'),
                mean=con[0]['total_minus_ref']['diff'], ci95=ci, verdict=verdict_of(ci, arm))
    group_rule = dict(all_pass=g.get('all_pass'),
                      worst=max(((k, v.get('increase_mean_s')) for k, v in (g.get('groups') or {}).items()
                                 if v.get('registered') and 'increase_mean_s' in v), key=lambda x: x[1], default=None),
                      rule='no registered group-mean per-vehicle delay increase > 10 s vs U0 (U0 labels)')
    arm_rows = {}
    for a in have:
        arm_rows[a] = dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                           safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        if a in SPECS:
            arm_rows[a].update(_acts_summary(a, have[a], worlds))
    per_world = [dict(world_id=w, seed=seed[w], **{f'J_{a}': float(have[a][w]['J_veh_s']) for a in have},
                      **{f'hard_brakes_{a}': int(have[a][w]['safety']['hard_brakes']) for a in have},
                      **{f'ttc_{a}': int(have[a][w]['safety']['ttc_events']) for a in have},
                      **{f'acts_{a}': int((have[a][w].get('a37') or {}).get('activations') or 0) for a in arms_mine})
                 for w in worlds]
    return N.seal(dict(schema='coopmwm.a59.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, arm=arm, cell=CELL,
                       setting=CSET, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), bootstrap=bootstrap,
                       convention='diff = treatment - reference; negative = treatment has less delay', primary=prim,
                       group_rule_vs_u0=group_rule, contrasts=con, arms=arm_rows, spec=SPECS[arm],
                       per_world=per_world, a42_admission_receipt=A42_ADMISSION_RECEIPT,
                       reused={a: f'{A42_ROOT} {D42.confirm_wave(a)}' for a in ('U0', 'DUO_HOLD', 'BIG_BANK')},
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()},
                       charged_sim_s=dict(total=D.charged(final_root),
                                          **{a: arm_charged(final_root, a) for a in SPECS}),
                       root=str(final_root)), 'analysis_sha256')


def write_table(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------------------------ CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('prepare', 'dry', 'identity', 'identity-compare', 'smoke', 'smoke-check',
                                       'verify', 'admit', 'confirm', 'analyse', 'status'))
    p.add_argument('--final-root', required=True)
    p.add_argument('--arm', choices=tuple(SPECS))
    p.add_argument('--n', type=int, default=1)
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--chunk', type=int, default=8)
    a = p.parse_args(argv)
    root = Path(a.final_root)
    setting = CSET if a.command in ('confirm', 'analyse') else VSET
    panel = D.setup(setting)
    arm = a.arm

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=setting, arm=arm),
                          sort_keys=True, default=str)
        print(line, flush=True)
        with (root/D.LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    if a.command == 'prepare':
        print(json.dumps(prepare(root, arm), indent=1))
    elif a.command == 'dry':
        (have, _), a42specs = a42_u0_verify(panel)
        iw = ident_wave(panel, root)
        out = dict(identity=dict(job=iw[2][0]['job_id'], seed=iw[2][0]['world']['seed'],
                                 spec_equal=_norm(iw[2][0], root) == _norm(a42specs[0], root),
                                 stored_J=have[iw[2][0]['world_id']]['J_veh_s']),
                   charged=D.charged(root), arms={})
        for x in SPECS:
            n, _, sp = verify_wave(x, panel, root, check_files=False)
            out['arms'][x] = dict(wave=n, n=len(sp), first=sp[0]['job_id'], bank=sp[0]['checkpoint']['a37_bank'],
                                  family=sp[0]['checkpoint'].get('a37_family'),
                                  seeds=[sp[0]['world']['seed'], sp[-1]['world']['seed']],
                                  charged=arm_charged(root, x), cap=CAPS[x])
        print(json.dumps(out, indent=1, default=str))
    elif a.command in ('identity', 'smoke', 'verify', 'confirm'):
        import fcntl
        if a.command != 'identity':
            if not passed(root, IDENTITY_FILE):
                raise N.Blocked('identity not PASS')
            if a.command != 'smoke' and not passed(root, f'SMOKE_A59_{arm}.json'):
                raise N.Blocked(f'smoke for {arm} not PASS')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            if a.command == 'identity':
                waves, bucket = [ident_wave(panel, root)], 'HAND5'
            elif a.command == 'smoke':
                n, k, sp = verify_wave(arm, panel, root)
                waves, bucket = [(n, k, sp[:a.n])], arm
            elif a.command == 'verify':
                waves, bucket = [verify_wave(arm, panel, root)], arm
            else:
                waves, bucket = [confirm_wave(arm, panel, root)], arm
            done = sum(N.published(root, n, s, D.C37.check_a37)[2] for n, _, s in waves)
            need = (sum(len(s) for _, _, s in waves)-done)*N.HORIZON_S
            if arm_charged(root, bucket)+need > CAPS[bucket]:
                raise N.Blocked(f'{bucket}: {arm_charged(root, bucket)} + {need} > cap {CAPS[bucket]}')
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=D.charged(root), need_sim_s=need, bucket=bucket))
            res = D.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(root, panel)
        N.publish(root/IDENTITY_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], world=v['world'])))
    elif a.command == 'smoke-check':
        v = smoke_check(root, panel, arm, a.n)
        N.publish(root/f'SMOKE_A59_{arm}.json', v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], activations=v['activations'], cal=v['lane_calibration'],
                              acts=[(x['world_id'], x['t'], x['distinct'], x['lane_counts'], x['setpoint_at'])
                                    for x in v['acts']])))
    elif a.command == 'admit':
        v = admit(root, panel, arm)
        x = v['verdict']
        print(json.dumps(dict(arm=arm, admitted=v['admitted'], ci95=x['ci95'], mean=x['mean'], allowed=x['allowed'],
                              worst=[x['worst_group'], x['worst_group_inc']], acts=x['activations'],
                              group_size=x['group_size'])))
    elif a.command == 'analyse':
        v = analyse(root, panel, arm)
        out = Path(a.out or root/'analysis')
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/f'{arm}.json', v, replace=True)
        write_table(v['per_world'], out/f'{arm}_PER_WORLD.csv')
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], primary=v['primary'],
                              group_rule=v['group_rule_vs_u0'])))
    else:
        out = dict(charged=D.charged(root), identity=passed(root, IDENTITY_FILE),
                   **{x: arm_charged(root, x) for x in SPECS})
        for x in SPECS:
            for n, _, s in (verify_wave(x, panel, root, check_files=False),):
                out[n] = f'{N.published(root, n, s, D.C37.check_a37)[2]}/{len(s)}'
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
