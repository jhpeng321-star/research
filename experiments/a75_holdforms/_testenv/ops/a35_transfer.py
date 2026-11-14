"""AMENDMENT 35 (MAIN 2026-11-10 ~04:30, owner-relayed reviewer request; registered before any amendment-35 data):
FROZEN-controller transfer. U0 (MAPPO), LEARNED_TRIGGER (bank fd385a2b + classifier 8e0ddd3e, threshold 0.378785) and
DUO_HOLD (duo12_v20 of UNION_UNC 39f384a2), exactly as evaluated (no retuning), on four FRESH 96-world panels, one per
cell (exploratory; never pooled across cells): D2400_p20 (998000-998095), D3200_p20 (998100-998195), D2800_p60
(998200-998295; substitute for D2800_p10) and D2800_p40 (998300-998395; substitute for D2800_p30). The plans and the
classifier were built for D2800_p20 and run as-is: the bank rows are admitted under that source cell (runner
``bank_cell_key``); the plan executor's feasibility / session / cooldown / exclusivity rules are unchanged.

Thin driver over ops/nm_finaleval.py and ops/a32_ablation.py (same mirrored bank-root bindings as amendments 32/33:
BOUND_U0, ORACLE_BINDING, PRERUN_PLAN, prerun_plant); waves through the unchanged nm_finaleval.run_waves (canary
first, per-episode ledger rows precharged in <final_root>/CHARGES.jsonl, phase transfer_a35, cap 1,450,000 sim-s).

Commands (python ops/a35_transfer.py <cmd> --final-root <root> --setting a35_<cell>):
  dry       zero physics: build every spec, check bindings/digests, print counts
  identity  plant identity gate (p20c sealed worlds 993100/993101): U0 x 2 and LEARNED_TRIGGER x 2 (the latter with
            bank_cell_key = D2800_p20, i.e. the a35 code path) must reproduce the sealed amendment-28 J and safety
  run       --arms U0,LEARNED_TRIGGER,DUO_HOLD --workers 24 --chunk 48 (STOP_A35 stops after the chunk in flight)
  reports   zero physics: vehicles_report.json next to every published episode log of this cell
  identity-compare --bank-root <sealed p20c bank copy>   J/safety of the identity waves vs the sealed records
  analyse   --out DIR: the registered per-cell contrasts -> A35_RESULTS_<cell>.json
"""
from __future__ import annotations

import argparse
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
import a32_ablation as A  # noqa: E402  (registers check_a32 for the DUO_HOLD arm)

A35_PHASE = 'transfer_a35'
A35_CAP_SIM_S = 1_450_000
STOP_A35 = 'STOP_A35'
BANK_CELL = 'D2800_p20'
ARMS = ('U0', 'LEARNED_TRIGGER', 'DUO_HOLD')
IDENT_U0, IDENT_LT = 'A35_IDENT_U0', 'A35_IDENT_LT'
SETTINGS = tuple(c[0] for c in N.A35_CELLS)
AUTHORITY = ('MAIN 2026-11-10 AMENDMENT 35 (frozen-controller transfer, exploratory): U0, LEARNED_TRIGGER (fd385a2b + '
             '8e0ddd3e, threshold 0.378785), DUO_HOLD (duo12_v20) frozen; fresh 96-world panels per cell; phase '
             'transfer_a35 cap 1,450,000 sim-s')
CONTRASTS = (('LEARNED_TRIGGER', 'U0'), ('DUO_HOLD', 'U0'), ('LEARNED_TRIGGER', 'DUO_HOLD'))


def setup(setting):
    if setting not in SETTINGS:
        raise N.Blocked(f'setting is one of {SETTINGS}')
    N.apply_setting(setting)
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    return N.load_panel(N.PANEL_FILE)


def cell_tag(panel):
    return panel['cell']['cell_key']


class A35Runtime(N.PrerunRuntime):
    """The pre-run plant of the mirror root, own ledger phase transfer_a35, cap = allocation = 1,450,000."""
    phase = A35_PHASE

    def check_capacity(self, amount):
        rows = N.fe.ledger_rows(self.final_root)
        if sum(r['sim_s'] for r in rows)+amount > A35_CAP_SIM_S:
            raise N.Blocked(f'A35: {A35_PHASE} cap {A35_CAP_SIM_S} would be exceeded')

    def reserve(self, cfg, jobs):
        N.own_reserve(self.final_root, jobs, A35_PHASE, cap=A35_CAP_SIM_S, calls=self.calls, allocation=A35_CAP_SIM_S)


def transfer(spec):
    """Bind the bank rows to their source cell (the frozen plans/classifier run as-is in the transfer cell)."""
    ck = deepcopy(spec['checkpoint'])
    ck['bank_cell_key'] = BANK_CELL
    return dict(spec, checkpoint=ck)


def _arm_specs(arm, panel, final_root, *, check_files=True):
    final_root = Path(final_root)
    u0, oracle = A.bindings(final_root)
    plant = final_root/N.PRERUN_PLANT
    if arm == 'U0':
        return N.episode_specs('U0', panel, None, plant, 'mappo_frozen', None, u0)
    if arm == 'LEARNED_TRIGGER':
        return [transfer(s) for s in N.episode_specs('LEARNED_TRIGGER', panel, None, plant, 'mappo_frozen', oracle, u0)]
    if arm == 'DUO_HOLD':
        return [transfer(s) for s in A.a32_specs('DUO_HOLD', panel, final_root, check_files=check_files)]
    raise N.Blocked(f'no A35 arm {arm!r}')


def arm_specs(arm, panel, final_root, *, check_files=True):
    """The arm's specs with job ids prefixed 'a35_<cell>_' (one root for the four cells, no clash)."""
    tag = 'a35_'+cell_tag(panel).lower()+'_'
    return [dict(s, job_id=tag+s['job_id']) for s in _arm_specs(arm, panel, final_root, check_files=check_files)]


def wave_name(arm, panel):
    return f'A35_{cell_tag(panel)}_{arm}'


def ident_specs(final_root):
    """U0 and LEARNED_TRIGGER (a35 path: bank_cell_key) on the first 2 sealed p20c worlds."""
    p20c = N.read(HERE/'nm_eval_panel_wz3to2_p20c_D2800.json')
    N.verify(p20c, 'panel_sha256', 'p20c panel')
    if p20c['panel_sha256'] != 'd2ed7783ae9ea071f89aec4397d3ee155161878b44e1717b02188b16e7ff7fa2':
        raise N.Blocked('identity worlds: p20c panel sha differs')
    u0, oracle = A.bindings(final_root)
    plant = Path(final_root)/N.PRERUN_PLANT
    k = N.IDENTITY_WORLDS
    us = N.episode_specs('U0', p20c, None, plant, 'mappo_frozen', None, u0)[:k]
    ls = N.episode_specs('LEARNED_TRIGGER', p20c, None, plant, 'mappo_frozen', oracle, u0)[:k]
    return ([dict(s, job_id=s['job_id'].replace('nmfinal_u0_', 'a35ident_u0_')) for s in us],
            [transfer(dict(s, job_id=s['job_id'].replace('nmfinal_learned_trigger_', 'a35ident_lt_'))) for s in ls])


def check_a35(record, spec):
    return A.check_a32(record, spec)


N.CHECKS['episode'] = check_a35


def execute(final_root, panel, waves, *, workers, chunk, log, map_panel=True):
    final_root = Path(final_root)
    runtime = A35Runtime(N.Runtime(N.RUNNER), final_root)
    plant = final_root/N.PRERUN_PLANT
    cfg = runtime.config(plant)
    if not os.environ.get('COOPMWM_MAP_SPEC'):
        raise N.Blocked('set COOPMWM_MAP_SPEC before physics')
    status = runtime.runner_status()
    if not status['ok'] or not status.get('learned'):
        raise N.Blocked(f'runner cannot run the A35 arms: {status}')
    from tools.coopmwm import nm_eval_runner as R
    if getattr(R, 'A35_BANK_CELL', None) != BANK_CELL:
        raise N.Blocked('runner lacks the amendment-35 bank_cell_key path')
    if map_panel:
        N.map_check(N.bind_map_api(None), panel)
    plan = N.prerun_plan(final_root)
    u0, _ = A.bindings(final_root)
    if (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
        raise N.Blocked('the U0 binding residual is not the pre-run u0 copy')

    def coverage():
        return {name: (N.published(final_root, name, specs, check_a35)[2], len(specs)) for name, _, specs in waves}
    rounds = 0
    while True:
        cov = coverage()
        pending = [w for w in waves if cov[w[0]][0] < cov[w[0]][1]]
        if not pending:
            break
        if (final_root/STOP_A35).exists():
            log(dict(stopped=True, coverage=cov))
            return dict(status='STOPPED', coverage=cov)
        budget, limits, selection = chunk, {}, []
        for name, kind, specs in pending:
            take = min(cov[name][1]-cov[name][0], budget)
            selection.append((name, kind, specs))
            limits[name] = take
            budget -= take
            if budget == 0:
                break
        N.run_waves(selection, root=plant, final_root=final_root, cfg=cfg, runtime=runtime, workers=workers,
                    base='mappo_frozen', chunked=True, limit=limits)
        rounds += 1
        log(dict(round=rounds, published={k: v[0] for k, v in coverage().items()},
                 charged_sim_s=sum(r['sim_s'] for r in N.fe.ledger_rows(final_root))))
    cov = coverage()
    return dict(status='PASS', coverage=cov, charged_sim_s=sum(r['sim_s'] for r in N.fe.ledger_rows(final_root)))


def identity_compare(final_root, bank_root):
    """J / safety / vehicle-log sha of the identity waves vs the sealed amendment-28 records (first 2 worlds only)."""
    bank_root = Path(bank_root)
    u_specs, l_specs = ident_specs(final_root)
    rows = []
    for wave, sealed_prefix, specs in ((IDENT_U0, 'WAVE_U0__', u_specs), (IDENT_LT, 'WAVE_LEARNED_TRIGGER__', l_specs)):
        first = sorted(bank_root.glob(sealed_prefix+'0000-*.json'))
        if len(first) != 1:
            raise N.Blocked(f'{bank_root}: no unique {sealed_prefix}0000-* wave file')
        value = N.verify(N.read(first[0]), 'receipt_sha256', first[0].name)
        ref = {r['world_id']: r for r in value['records'][:N.IDENTITY_WORLDS]}
        got, _, cov = N.published(final_root, wave, specs, check_a35)
        if cov != len(specs):
            raise N.Blocked(f'{wave}: {cov}/{len(specs)} published')
        for r in got:
            s = ref[r['world_id']]
            rows.append(dict(wave=wave, world_id=r['world_id'], j_gpuB=r['J_veh_s'], j_sealed=s['J_veh_s'],
                             abs_diff=abs(r['J_veh_s']-s['J_veh_s']), safety_equal=r['safety'] == s['safety'],
                             vehicle_log_equal=r['vehicle_log']['sha256'] == s['vehicle_log']['sha256']))
    ok = len(rows) == 2*N.IDENTITY_WORLDS and all(x['abs_diff'] <= 1e-6 and x['safety_equal'] for x in rows)
    return dict(verdict='PASS' if ok else 'FAILED', worlds=rows, rule='|J diff| <= 1e-6 veh-s and identical safety')


# ------------------------------------------------------------------------------------------------ analysis

def analyse(panel, root):
    """Per-cell registered amendment-35 contrasts: paired percentile bootstrap (10,000, seed 20261104), diff =
    treatment - reference (negative = less delay); the five fairness groups (origin L0 closing / L1 / L2 leftmost,
    hdv / cav; 10 s rule, U0 labels), CVaR90, safety. Exploratory; never pooled across cells."""
    import numpy as np
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile'):
        raise N.Blocked('panel bootstrap is not the registered one')
    have, sources = {}, {}
    for arm in ARMS:
        specs = arm_specs(arm, panel, root, check_files=False)
        recs, src, cov = N.published(root, wave_name(arm, panel), specs, check_a35)
        if cov != len(specs):
            raise N.Blocked(f'{arm}: {cov}/{len(specs)} published in {root}')
        have[arm], sources[arm] = {r['world_id']: r for r in recs}, src
    reports = {a: {w: A.report_of(root, have[a][w]) for w in worlds} for a in have}
    if any(not r['ok'] for a in reports for r in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    contrasts = []
    for arm, ref in CONTRASTS:
        row = dict(treatment=arm, reference=ref,
                   convention='diff = treatment - reference; negative = treatment has less delay')
        row.update(N._bank_compare(have, reports, worlds, arm, ref, bootstrap))
        for k in ('hard_brakes', 'ttc_events'):
            row[f'{k}_minus_ref'] = N.fe.paired_stats(
                np.array([have[arm][w]['safety'][k] for w in worlds], float),
                np.array([have[ref][w]['safety'][k] for w in worlds], float), bootstrap)
        contrasts.append(row)
    arms = {}
    for a in have:
        arms[a] = dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                       safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        rec = [have[a][w].get('a32') or have[a][w].get('learned_trigger') or {} for w in worlds]
        if any(rec):
            fam = {}
            for r in rec:
                for f, n in (r.get('activations_by_family') or {}).items():
                    fam[f] = fam.get(f, 0)+int(n)
            arms[a]['activations'] = dict(worlds_with_any=sum(1 for r in rec if r.get('activations')),
                                          total=sum(int(r.get('activations') or 0) for r in rec), by_family=fam,
                                          scores=sum(int(r.get('scores') or 0) for r in rec),
                                          fires=sum(int(r.get('fires') or 0) for r in rec if r.get('fires') is not None))
    return N.seal(dict(schema='coopmwm.a35_transfer.analysis.v1', amendment='35', authority=AUTHORITY,
                       cell=panel['cell']['cell_key'], cell_authority=panel['cell']['authority'],
                       bank_cell_key=BANK_CELL, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                       bootstrap=bootstrap, contrasts=contrasts, arms=arms,
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()}, root=str(root),
                       label='exploratory generalization (amendment 35); never pooled across cells; one shot'),
                  'analysis_sha256')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('dry', 'identity', 'run', 'reports', 'identity-compare', 'analyse'))
    p.add_argument('--final-root', required=True)
    p.add_argument('--setting', required=True, choices=SETTINGS)
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--chunk', type=int, default=48)
    p.add_argument('--bank-root')
    p.add_argument('--out')
    a = p.parse_args(argv)
    panel = setup(a.setting)
    root = Path(a.final_root)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T')), sort_keys=True, default=str)
        print(line, flush=True)
        with (root/'A35_LOG.jsonl').open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    arms = tuple(x for x in a.arms.split(',') if x)
    if not set(arms) <= set(ARMS):
        p.error(f'arms are {ARMS}')
    if a.command == 'dry':
        out = {}
        for arm in arms:
            specs = arm_specs(arm, panel, root)
            out[arm] = dict(n=len(specs), first=specs[0]['job_id'], cell=specs[0]['world']['cell_key'],
                            bank_cell_key=specs[0]['checkpoint'].get('bank_cell_key'),
                            seeds=[specs[0]['world']['seed'], specs[-1]['world']['seed']])
        u, l_ = ident_specs(root)
        out['ident'] = [s['job_id'] for s in u+l_]
        out['panel_sha256'] = panel['panel_sha256']
        out['cell'] = panel['cell']['cell_key']
        from tools.coopmwm import a32_arms, learned_trigger
        out['a32_sha256'] = a32_arms.a32_digest()
        out['learned_trigger_sha256'] = learned_trigger.learned_trigger_digest()
        print(json.dumps(out, indent=1))
    elif a.command in ('identity', 'run'):
        import fcntl
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            if a.command == 'identity':
                u, l_ = ident_specs(root)
                waves = [(IDENT_U0, 'episode', u), (IDENT_LT, 'episode', l_)]
            else:
                waves = [(wave_name(arm, panel), 'episode', arm_specs(arm, panel, root)) for arm in arms]
            log(dict(start=a.command, setting=a.setting, waves=[w[0] for w in waves], workers=a.workers,
                     chunk=a.chunk, pid=os.getpid()))
            res = execute(root, panel, waves, workers=a.workers, chunk=a.chunk, log=log,
                          map_panel=a.command != 'identity')
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        print(json.dumps(identity_compare(root, a.bank_root), indent=1))
    elif a.command == 'analyse':
        value = analyse(panel, root)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/f'A35_RESULTS_{cell_tag(panel)}.json', value)
        print(json.dumps(dict(analysis_sha256=value['analysis_sha256'], cell=value['cell'])))
    else:
        n = 0
        for arm in arms:
            recs, _, cov = N.published(root, wave_name(arm, panel), arm_specs(arm, panel, root, check_files=False),
                                       check_a35)
            for r in recs:
                rep = N.episode_log_report(root, r)
                if rep is None or not rep['ok']:
                    raise N.Blocked(f'{r["job_id"]}: report failed')
                n += 1
        print(json.dumps(dict(reports=n)))


if __name__ == '__main__':
    main()
