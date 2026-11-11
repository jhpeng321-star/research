"""AMENDMENT 62 (S-COOPPERPLAN60; registered before any outcome of this test): per-plan cross-lane test at D2800_p60.

62a (zero physics): per-plan activations / commanded members in the stored A42 BIG_BANK panel-1 records (A42 root,
read-only; the sealed WAVE receipts, never the per-episode RESULT files) -> PLANSHARE.json.
62b: for each of the four A42-admitted plans p, PART_L0_p / PART_L1_p / PART_L2_p on the 24 A42 verification worlds
(setting a42v_D2800_p60, 992000-992023) with the A57 lane filter (tools/coopmwm/a57_partlane.py at eb704616,
UNCHANGED). The runner at eb704616 applies the a57 filter to the CELL_BANK arm only, so the single-plan arm runs as
CELL_BANK on the one-row bank C.restrict(A42_CANDS bank, p) (exactly the core the A37_PLAN arm builds); an identity
check (no filter) against the stored A42 admission run FULL_p proves the equivalence before any part arm runs.
Reused, read-only: A42 admission FULL_p (WAVE_A42V_<p>) and U0 (WAVE_A42V_U0).

Driver = ops/a42_bigp60.py module copy with A62 names: own root /root/a62/root, phase coopperplan_a62 cap 360,000,
job ids a62*, stop file STOP_A62. No automatic replay.
"""
from __future__ import annotations

import argparse
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
import a50_fullp60 as A50  # noqa: E402
import a57_coopsyn as A57  # noqa: E402   (helpers; the A42 read-only view A57.D42)

AMENDMENT = '62'
CELL = 'D2800_p60'
SETTING = 'a42v_D2800_p60'
LANES = (0, 1, 2)
N_WORLDS = 24
CAP_SIM_S = 360_000
A42_ROOT = A57.A42_ROOT
CANDS = f'A42_CANDS_{CELL}.json'
BOOTSTRAP = A57.BOOTSTRAP
AUTHORITY = ('MAIN AMENDMENT 62 (2026-11-11): S-COOPPERPLAN60, per-plan PART_L0/L1/L2 of the four A42-admitted plans on '
             'the 24 verification worlds with the unchanged A57 filter; reuse A42 FULL_p and U0; cap 360,000')
PRIMARY_RULE = ('D_p = FULL_p - PART_L2_p per world (total delay); paired percentile bootstrap 10,000, seed 20261104; '
                'Holm over the four plans (two-sided p): CROSS-LANE MEMBERS HELP if Holm p < 0.05 and D_p < 0; HURT if '
                'Holm p < 0.05 and D_p > 0; else NOT RESOLVED')
OVERRIDES = dict(
    A42_PHASE='coopperplan_a62', A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A62',
    HOST_BASE=os.environ.get('A62_HOST_BASE', '/root/a62'),
    BANK_DIR='a62_banks', IDENT_WAVE='A62_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A62', JOB_PREFIX='a62',
    VERIFY_TAG='a62v', CONFIRM_TAG='a62c', IDENT_TAG='a62i', VERIFY_WAVE_PREFIX='A62V_', CONFIRM_WAVE_PREFIX='A62C_',
    CANDS_BANK_FILE=CANDS, BIG_BANK_FILE=f'A62_UNUSED_{CELL}.json',
    ADMISSION_FILE=f'ADMISSION_A62_UNUSED_{CELL}.json', RESULTS_FILE='COOPPERPLAN.json', LOG_FILE='A62_LOG.jsonl',
    SCHEMA_PREFIX='coopmwm.a62', RUNNER_RANGES_ATTR='A42_RANGES',
    RESULT_LABEL='amendment 62; per-plan cross-lane test at 60% CAVs; one shot', AUTHORITY=AUTHORITY)


def _build():
    m = A50._clone('a62_coopperplan_core', HERE/'a42_bigp60.py', OVERRIDES)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (m.VERIFY_SETTING, m.CONFIRM_SETTING)
    m.A42Runtime.phase = m.A42_PHASE
    return m


D = _build()
D42 = A57.D42
IDENT_WAVE = 'A62_IDENT_FULL'
IDENTITY_FILE = 'IDENTITY_A62.json'
SMOKE_FILE = 'SMOKE_A62.json'


# ------------------------------------------------------------------------------------------------ plans / banks

def admitted():
    """[dict(family, key)] of the four A42-admitted plans, in the A42 admission (bank priority) order."""
    adm = A57.a42_admission()
    keys = {v['family']: v['key'] for v in adm['verdicts']}
    return [dict(family=f, key=keys[f]) for f in adm['admitted']]


def plan_bank_name(fam):
    return f'A62_PLAN_{fam}.json'


def prepare(final_root):
    """Zero physics: a62_banks/A42_CANDS (byte copy check) + one one-row bank per admitted plan (write once)."""
    from tools.coopmwm import a37_cellbank as C
    final_root = Path(final_root)
    copy = D.bank_path(final_root, CANDS)
    orig = Path(A42_ROOT)/D42.BANK_DIR/CANDS
    if not copy.is_file() or N.file_sha(copy) != N.file_sha(orig):
        raise N.Blocked(f'{copy}: missing or not a byte copy of {orig}')
    doc = json.loads(copy.read_text(encoding='utf-8'))
    out = {}
    for p in admitted():
        path = D.C37.write_once(D.bank_path(final_root, plan_bank_name(p['family'])),
                                D.C37.bank_text(C.restrict(doc, p['family'])))
        out[p['family']] = N.file_sha(path)
    return out


def _entry(final_root, fam, check_files=True):
    return D._bank_entry(final_root, plan_bank_name(fam), check_files=check_files) or dict(path='', file_sha256='')


def plan_specs(panel, final_root, p, *, lane=None, tag, check_files=True):
    from tools.coopmwm import a57_partlane as P
    out = []
    for s in D._a37_specs(D.RUNNER_ARM['BIG_BANK'], panel, final_root, _entry(final_root, p['family'], check_files),
                          tag=tag):
        if lane is not None:
            ck = deepcopy(s['checkpoint'])
            ck.update(a57_member_lane=int(lane), a57_sha256=P.a57_digest())
            s = dict(s, checkpoint=ck)
        out.append(s)
    return out


def arm_name(p, lane):
    return f'PART_L{lane}_{p["family"]}'


def part_waves(panel, final_root, *, check_files=True, first_only=False):
    out = []
    for p in admitted():
        for k in LANES:
            sp = plan_specs(panel, final_root, p, lane=k, tag=f'{D.VERIFY_TAG}_{p["key"][:8]}_l{k}',
                            check_files=check_files)
            out.append((f'{D.VERIFY_WAVE_PREFIX}{arm_name(p, k)}', 'episode', sp[:1] if first_only else sp))
    return out


def a42_records(panel):
    """{'U0': {w: rec}, family: {w: rec}} from the stored A42 verification waves (A42 root, read-only)."""
    have, src = {}, {}
    for name, _, specs in D42.verify_waves(panel, A42_ROOT, check_files=True, cands=admitted()):
        arm = name[len(D42.VERIFY_WAVE_PREFIX):]
        have[arm], src[arm] = D42.C37._load(A42_ROOT, name, specs)
    return have, src


def identity_target(panel):
    """(plan, world index): the first admitted plan (admission order), first world (panel order) with >= 1 stored
    activation."""
    have, _ = a42_records(panel)
    for p in admitted():
        for i, r in enumerate(panel['worlds']):
            if int((have[p['family']][r['world_id']].get('a37') or {}).get('activations') or 0) > 0:
                return p, i
    raise N.Blocked('no stored FULL_p world with an activation')


def ident_waves(panel, final_root, *, check_files=True):
    p, i = identity_target(panel)
    s = plan_specs(panel, final_root, p, tag=f'{D.IDENT_TAG}_full', check_files=check_files)[i]
    return [(IDENT_WAVE, 'episode', [dict(s, job_id=f'{D.IDENT_TAG}_full_0000')])]


def identity_compare(final_root, panel):
    final_root = Path(final_root)
    have, _ = a42_records(panel)
    p, _ = identity_target(panel)
    (name, _, specs), = ident_waves(panel, final_root)
    got, _, cov = N.published(final_root, name, specs, D.C37.check_a37)
    if cov != 1:
        raise N.Blocked(f'{name}: {cov}/1 published')
    r = got[0]
    a = have[p['family']][r['world_id']]
    row = dict(family=p['family'], world_id=r['world_id'], j_a62=r['J_veh_s'], j_a42=a['J_veh_s'],
               abs_diff=abs(r['J_veh_s']-a['J_veh_s']), safety_equal=r['safety'] == a['safety'],
               same_world=r['world'] == a['world'], activations=[r['a37'].get('activations'), a['a37'].get('activations')],
               acts_equal=A57._acts(r) == A57._acts(a), arms=[r['a37'].get('a37_arm'), a['a37'].get('a37_arm')],
               a37_sha256_equal=r['a37_sha256_loaded'] == a['a37_sha256_loaded'], no_filter='a57_part' not in r['a37'])
    ok = (row['abs_diff'] == 0.0 and row['safety_equal'] and row['same_world'] and row['acts_equal']
          and row['a37_sha256_equal'] and row['no_filter'])
    return N.seal(dict(schema='coopmwm.a62.identity.v1', amendment=AMENDMENT,
                       rule=('single-plan CELL_BANK (one-row bank) without filter reproduces the stored A42 admission '
                             'A37_PLAN record exactly: J diff 0, safety, activation receipts, a37 digest'),
                       verdict='PASS' if ok else 'FAILED', world=row,
                       consequence='PASS: stored FULL_p / U0 reused; FAILED: stop'), 'receipt_sha256')


def _passed(final_root, name):
    p = Path(final_root)/name
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


def smoke_check(final_root, panel):
    from tools.coopmwm import a57_partlane as P
    rows, ok = [], True
    waves = part_waves(panel, final_root, first_only=True)
    for (name, _, specs), (p, k) in zip(waves, [(p, k) for p in admitted() for k in LANES]):
        got, _, cov = N.published(final_root, name, specs, D.C37.check_a37)
        if cov != 1:
            raise N.Blocked(f'{name}: smoke episode not published')
        part = (got[0].get('a37') or {}).get('a57_part') or {}
        fams = {x['family'] for x in (got[0].get('a37') or {}).get('acts') or []}
        good = (part.get('member_lane') == k and part.get('a57_sha256') == P.a57_digest() and fams <= {p['family']})
        ok = ok and good
        rows.append(dict(wave=name, receipt_ok=good, activations=part.get('activations'),
                         members_on_lane=part.get('members_on_lane_total'),
                         no_member_on_lane=part.get('activations_no_member_on_lane')))
    return N.seal(dict(schema='coopmwm.a62.smoke.v1', verdict='PASS' if ok else 'FAILED', rows=rows,
                       rule='record valid + own-lane a57 receipt + registered digest + only plan p acts (no outcome)'),
                  'receipt_sha256')


# ------------------------------------------------------------------------------------------------ 62a

def planshare():
    """62a, zero physics: per-plan activations / members (by planned lane) in the stored A42 BIG_BANK panel-1 records."""
    from tools.coopmwm import oracle_plan as O
    panel = D.setup(D.CONFIRM_SETTING)
    worlds = [r['world_id'] for r in panel['worlds']]
    specs = D42.confirm_specs('BIG_BANK', panel, A42_ROOT, check_files=True)
    have, src = D42.C37._load(A42_ROOT, D42.confirm_wave('BIG_BANK'), specs)
    bank = json.loads((Path(A42_ROOT)/D42.BANK_DIR/D42.BIG_BANK_FILE).read_text(encoding='utf-8'))
    roles = {t['family']: O.spec_of(t['family'], t['offer_condition'])['roles'] for t in bank['clean5c']['templates']}
    fams = [p['family'] for p in admitted()]
    tot = {f: dict(activations=0, members=0, members_by_lane={str(k): 0 for k in LANES}, worlds_with_activation=0)
           for f in fams}
    per_world = []
    for w in worlds:
        acts = (have[w].get('a37') or {}).get('acts') or []
        row = dict(world_id=w, seed=have[w]['world']['seed'], total_activations=len(acts))
        for f in fams:
            mine = [a for a in acts if a['family'] == f]
            row[f'acts_{f}'] = len(mine)
            row[f'members_{f}'] = sum(len(a['members']) for a in mine)
            tot[f]['activations'] += len(mine)
            tot[f]['members'] += row[f'members_{f}']
            tot[f]['worlds_with_activation'] += bool(mine)
            for a in mine:
                if len(a['members']) != len(roles[f]):
                    raise N.Blocked(f'{w} {f}: members do not match the plan roles')
                for r in roles[f]:
                    tot[f]['members_by_lane'][str(int(r['lane']))] += 1
        if sum(row[f'acts_{f}'] for f in fams) != len(acts):
            raise N.Blocked(f'{w}: an activation of a family outside the bank')
        per_world.append(row)
    n_act = sum(t['activations'] for t in tot.values())
    for f in fams:
        tot[f]['share_of_activations'] = tot[f]['activations']/n_act if n_act else None
        tot[f]['roles'] = [(int(r['lane']), int(r['rank']), float(r['v_w']), int(r['onset_s']), int(r['hold_s']))
                           for r in roles[f]]
    return N.seal(dict(schema='coopmwm.a62.planshare.v1', amendment=AMENDMENT, part='62a (zero simulation)',
                       source=f'{A42_ROOT} WAVE_A42C_BIG_BANK (sealed receipts; panel 1, 96 worlds)',
                       source_receipts=src.get('receipt_sha256'), panel_sha256=panel['panel_sha256'],
                       note='commanded members in BIG_BANK = all planned members of each activation (no filter)',
                       plans=fams, totals=tot, total_activations=n_act, per_world=per_world), 'receipt_sha256')


# ------------------------------------------------------------------------------------------------ 62b analysis

def holm(ps):
    """Holm-adjusted p-values (step-down, monotone), same order as ``ps``."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0]*m, 0.0
    for j, i in enumerate(order):
        run = max(run, min(1.0, (m-j)*ps[i]))
        adj[i] = run
    return adj


def verdict_of(d, p_holm):
    if p_holm < 0.05:
        return 'CROSS-LANE MEMBERS HELP' if d < 0 else 'CROSS-LANE MEMBERS HURT'
    return 'NOT RESOLVED'


def analyse(final_root, panel):
    import numpy as np
    final_root = Path(final_root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != BOOTSTRAP or len(worlds) != N_WORLDS:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    if not _passed(final_root, IDENTITY_FILE):
        raise N.Blocked('identity not PASS')
    have, sources = a42_records(panel)
    roots = {a: A42_ROOT for a in have}
    for name, _, specs in part_waves(panel, final_root):
        arm = name[len(D.VERIFY_WAVE_PREFIX):]
        have[arm], sources[arm] = D.C37._load(final_root, name, specs)
        roots[arm] = str(final_root)
    reports = {}
    from a32_ablation import report_of
    for a in have:
        reports[a] = {}
        for w in worlds:
            rec = have[a][w]
            d = Path(roots[a])/'evaluation'/rec.get('work_dir', rec['job_id'])
            if roots[a] == A42_ROOT and not (d/'vehicles_report.json').is_file():
                raise N.Blocked(f'{d}: no cached report (A42 root is read-only here)')
            reports[a][w] = report_of(roots[a], rec)
    J = {a: np.array([have[a][w]['J_veh_s'] for w in worlds], float) for a in have}
    HB = {a: np.array([have[a][w]['safety']['hard_brakes'] for w in worlds], float) for a in have}
    plans, ps = {}, []
    for p in admitted():
        f = p['family']
        parts = [arm_name(p, k) for k in LANES]
        dp = N.fe.paired_stats(J[f], J[parts[2]], bootstrap, pct=True)
        ps.append(dp['p_two_sided'])
        arms = ['U0', f]+parts
        CV, n_paired = A57._five_way_cvar90(reports, worlds, arms)
        CV = {a: np.array(v, float) for a, v in CV.items()}
        plans[f] = dict(
            D_p=dp,
            I_p=A57._mean_ci(A57.interaction(J[f], [J[a] for a in parts], J['U0']), bootstrap),
            I_p_cvar90_s=dict(A57._mean_ci(A57.interaction(CV[f], [CV[a] for a in parts], CV['U0']), bootstrap),
                              paired_vehicles_per_world=dict(min=min(n_paired), max=max(n_paired))),
            I_p_hard_brake_veh_s=A57._mean_ci(A57.interaction(HB[f], [HB[a] for a in parts], HB['U0']), bootstrap),
            full_minus_u0=N._bank_compare(have, reports, worlds, f, 'U0', bootstrap),
            part_minus_u0={a: N._bank_compare(have, reports, worlds, a, 'U0', bootstrap) for a in parts},
            arms={a: dict(mean_J_veh_s=float(J[a].mean()),
                          safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY},
                          activations=sum(int((have[a][w].get('a37') or {}).get('activations') or 0) for w in worlds),
                          members_planned=sum(len(x['members']) for w in worlds
                                              for x in (have[a][w].get('a37') or {}).get('acts') or []),
                          members_commanded=(sum(int(have[a][w]['a37']['a57_part']['members_on_lane_total'])
                                                 for w in worlds) if a in parts else None),
                          activations_no_member_on_lane=(sum(int(have[a][w]['a37']['a57_part'][
                              'activations_no_member_on_lane']) for w in worlds) if a in parts else None))
                  for a in [f]+parts})
    adj = holm(ps)
    for (f, v), pa in zip(plans.items(), adj):
        v['D_p'].update(p_holm=pa, verdict=verdict_of(v['D_p']['diff'], pa))
    per_world = [dict(world_id=w, seed=have['U0'][w]['world']['seed'],
                      **{f'J_{a}': float(J[a][i]) for a in sorted(have)},
                      **{f'hard_brakes_{a}': int(HB[a][i]) for a in sorted(have)},
                      **{f'acts_{a}': int((have[a][w].get('a37') or {}).get('activations') or 0)
                         for a in sorted(have) if a != 'U0'},
                      **{f'collisions_{a}': int(have[a][w]['safety']['collisions']) for a in sorted(have)},
                      **{f'teleports_{a}': int(have[a][w]['safety']['teleports']) for a in sorted(have)})
                 for i, w in enumerate(worlds)]
    return N.seal(dict(schema='coopmwm.a62.coopperplan.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       setting=SETTING, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), bootstrap=bootstrap,
                       rule=PRIMARY_RULE, convention='diff = treatment - reference; negative = less delay',
                       plans=plans, u0=dict(mean_J_veh_s=float(J['U0'].mean()),
                                            safety_totals={k: int(sum(have['U0'][w]['safety'][k] for w in worlds))
                                                           for k in N.SAFETY}),
                       per_world=per_world, reused=dict(U0=f'{A42_ROOT} WAVE_A42V_U0',
                                                        FULL_p=f'{A42_ROOT} WAVE_A42V_<plan>'),
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()},
                       charged_sim_s=D.charged(final_root), root=str(final_root)), 'analysis_sha256')


def budget_plan():
    ident = N.HORIZON_S
    run = 4*len(LANES)*N_WORLDS*N.HORIZON_S
    return dict(phase=D.A42_PHASE, cap_sim_s=CAP_SIM_S, identity=ident, run=run, smoke_included_in_run=12*N.HORIZON_S,
                planned=ident+run, margin=CAP_SIM_S-ident-run)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('planshare', 'prepare', 'dry', 'identity', 'identity-compare', 'smoke',
                                       'smoke-check', 'run', 'analyse', 'status', 'budget'))
    p.add_argument('--final-root')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--chunk', type=int, default=8)
    a = p.parse_args(argv)
    if a.command == 'budget':
        print(json.dumps(budget_plan(), indent=1))
        return
    if a.command == 'planshare':
        v = planshare()
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/'PLANSHARE.json', v, replace=True)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], total=v['total_activations'],
                              totals={f: {k: x[k] for k in ('activations', 'members', 'share_of_activations',
                                                            'worlds_with_activation', 'members_by_lane')}
                                      for f, x in v['totals'].items()})))
        return
    if not a.final_root:
        p.error('--final-root required')
    root = Path(a.final_root)
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
        pw = part_waves(panel, root)
        ref = {name[len(D42.VERIFY_WAVE_PREFIX):]: specs
               for name, _, specs in D42.verify_waves(panel, A42_ROOT, check_files=True, cands=admitted())}
        tp, ti = identity_target(panel)

        def norm(s):
            ck = {k: v for k, v in s['checkpoint'].items()
                  if k not in ('a37_bank', 'a37_arm', 'a37_family', 'a57_member_lane', 'a57_sha256')}
            s = dict(s, checkpoint=ck, arm=None)
            s.pop('job_id', None)
            t = json.dumps(s, sort_keys=True, default=str)
            return t.replace(D.HOST_ROOT, '<root>').replace(A42_ROOT, '<root>')
        print(json.dumps(dict(
            budget=budget_plan(), charged=D.charged(root), plans=[x['family'] for x in admitted()],
            identity=[tp['family'], ti, iw[0][2][0]['world_id']],
            identity_spec_equal_except_arm_and_bank=norm(iw[0][2][0]) == norm(ref[tp['family']][ti]),
            parts_equal_full_except_arm_bank_filter=all(norm(s) == norm(r) for name, _, sp in pw
                                                        for s, r in zip(sp, ref[name.split('_', 3)[-1]])),
            waves=[(n, len(sp), sp[0]['job_id'], sp[0]['checkpoint']['a57_member_lane'],
                    sp[0]['checkpoint']['a37_bank']['path'].rsplit('/', 1)[-1], [sp[0]['world']['seed'],
                                                                                 sp[-1]['world']['seed']])
                   for n, _, sp in pw],
            a37_sha256=iw[0][2][0]['checkpoint']['a37_sha256'], worlds=len(panel['worlds'])), indent=1, default=str))
    elif a.command in ('identity', 'smoke', 'run'):
        import fcntl
        if a.command != 'identity' and not _passed(root, IDENTITY_FILE):
            raise N.Blocked(f'{IDENTITY_FILE} is not PASS')
        if a.command == 'run' and not _passed(root, SMOKE_FILE):
            raise N.Blocked(f'{SMOKE_FILE} is not PASS')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = (ident_waves(panel, root) if a.command == 'identity' else
                     part_waves(panel, root, first_only=a.command == 'smoke'))
            done = sum(N.published(root, n, s, D.C37.check_a37)[2] for n, _, s in waves)
            need = (sum(len(s) for _, _, s in waves)-done)*N.HORIZON_S
            if D.charged(root)+need > CAP_SIM_S:
                raise N.Blocked(f'{D.A42_PHASE}: {D.charged(root)} + {need} > cap {CAP_SIM_S}')
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=D.charged(root)))
            res = D.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(root, panel)
        N.publish(root/IDENTITY_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], **{k: v['world'][k] for k in (
            'family', 'world_id', 'abs_diff', 'safety_equal', 'acts_equal', 'activations', 'arms')})))
    elif a.command == 'smoke-check':
        v = smoke_check(root, panel)
        N.publish(root/SMOKE_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], rows=v['rows'])))
    elif a.command == 'analyse':
        v = analyse(root, panel)
        out = Path(a.out or root)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/'COOPPERPLAN.json', v, replace=True)
        rows = v['per_world']
        with open(out/'COOPPERPLAN_PER_WORLD.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'],
                              D_p={f: (x['D_p']['diff'], x['D_p']['ci95'], x['D_p']['p_holm'], x['D_p']['verdict'])
                                   for f, x in v['plans'].items()})))
    else:
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, identity_pass=_passed(root, IDENTITY_FILE))
        for name, _, specs in part_waves(panel, root):
            out[name] = f'{N.published(root, name, specs, D.C37.check_a37)[2]}/{len(specs)}'
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
