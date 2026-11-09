"""AMENDMENT 32 ablation (MAIN 2026-11-10, owner-approved "yes, start it on gpub"): SOLO_HOLD, DUO_HOLD, BANK_ALWAYS on
the sealed 192-world p20c panel (993100-993291), paired against the sealed amendment-28 U0 and LEARNED_TRIGGER.

Thin driver over ops/nm_finaleval.py (setting p20c): the specs are the registered LEARNED_TRIGGER episode specs of
the bank root with only the arm and its start rule / plan set changed (checkpoint kind 'a32'); waves run through the
unchanged nm_finaleval.run_waves (canary first, per-episode ledger rows precharged in <final_root>/CHARGES.jsonl,
phase ablation_a32, cap 750,000 sim-s). The final root on the run host is a mirror of the bank root's BINDING files
at the same absolute path (BOUND_U0, ORACLE_BINDING, PRERUN_PLAN, prerun_plant, oracle_banks) with a fresh ledger;
no sealed wave of the bank root is copied there.

Commands (python -m ops.a32_ablation <cmd> --final-root <root>):
  dry       zero physics: build every spec, check bindings/digests, print counts
  identity  plant identity gate: U0 on the first 2 panel worlds (wave A32_IDENT_U0)
  run       --arms SOLO_HOLD,DUO_HOLD,BANK_ALWAYS --workers 24 --chunk 48 (STOP_A32 stops after the chunk in flight)
  reports   zero physics: write vehicles_report.json next to every published A32 episode log
  identity-compare --bank-root <sealed copy>  (laptop) J/safety of A32_IDENT_U0 vs the sealed U0 records
  analyse   --bank-root <sealed copy> --out <dir>  (laptop) the registered amendment-32 contrasts
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

A32_ARMS = ('SOLO_HOLD', 'DUO_HOLD', 'BANK_ALWAYS')
A32_PHASE = 'ablation_a32'
A32_CAP_SIM_S = 750_000
STOP_A32 = 'STOP_A32'
IDENT_WAVE = 'A32_IDENT_U0'
DUO_BANK_REL = 'a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json'
#: the run host's final root (gpuB mirror of the gpuA bank root path); specs name the DUO bank there on every host
RUN_ROOT = '/root/autodl-fs/nm_finaleval_p20c_bank'
AUTHORITY = ('MAIN 2026-11-10 AMENDMENT 32 (owner approval 2026-11-10 "yes, start it on gpub"): SOLO_HOLD, DUO_HOLD, '
             'BANK_ALWAYS on the sealed p20c panel 993100-993291; phase ablation_a32 cap 750,000 sim-s')
CONTRASTS = (('LEARNED_TRIGGER', 'DUO_HOLD'), ('LEARNED_TRIGGER', 'BANK_ALWAYS'), ('LEARNED_TRIGGER', 'SOLO_HOLD'),
             ('SOLO_HOLD', 'U0'), ('DUO_HOLD', 'U0'), ('BANK_ALWAYS', 'U0'), ('LEARNED_TRIGGER', 'U0'))


def setup():
    N.apply_setting('p20c')
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    return N.load_panel(N.PANEL_FILE)


class A32Runtime(N.PrerunRuntime):
    """The pre-run plant of the mirror root, own ledger phase ablation_a32, cap = allocation = 750,000."""
    phase = A32_PHASE

    def check_capacity(self, amount):
        rows = N.fe.ledger_rows(self.final_root)
        if sum(r['sim_s'] for r in rows)+amount > A32_CAP_SIM_S:
            raise N.Blocked(f'A32: {A32_PHASE} cap {A32_CAP_SIM_S} would be exceeded')

    def reserve(self, cfg, jobs):
        N.own_reserve(self.final_root, jobs, A32_PHASE, cap=A32_CAP_SIM_S, calls=self.calls, allocation=A32_CAP_SIM_S)


def bindings(final_root):
    final_root = Path(final_root)
    u0, oracle = N.mappo_binding(final_root, 'U0'), N.oracle_binding(final_root)
    if u0 is None or oracle is None:
        raise N.Blocked('A32 needs the bank-root U0 binding and the sealed ORACLE_BINDING (mirror)')
    return u0, oracle


def a32_specs(arm, panel, final_root, *, check_files=True):
    """The registered LEARNED_TRIGGER specs with arm -> ``arm`` and checkpoint kind 'a32' (+ start-rule fields)."""
    from tools.coopmwm import a32_arms
    final_root = Path(final_root)
    u0, oracle = bindings(final_root)
    plant = final_root/N.PRERUN_PLANT
    base = N.episode_specs('LEARNED_TRIGGER', panel, None, plant, 'mappo_frozen', oracle, u0)
    duo = final_root/DUO_BANK_REL
    if check_files and N.file_sha(duo) != a32_arms.DUO_BANK_SHA256:
        raise N.Blocked(f'{duo}: not UNION_UNC 39f384a2')
    out = []
    for s in base:
        ck = deepcopy(s['checkpoint'])
        ck.pop('learned_trigger_sha256')
        ck.update(kind='a32', a32_arm=arm, a32_sha256=a32_arms.a32_digest(),
                  duo_bank=dict(path=f'{RUN_ROOT}/{DUO_BANK_REL}', file_sha256=a32_arms.DUO_BANK_SHA256))
        if arm != 'BANK_ALWAYS':
            ck.pop('learned_trigger_model')
        out.append(dict(s, arm=arm, checkpoint=ck, job_id=s['job_id'].replace('learned_trigger', arm.lower())))
    return out


def ident_specs(panel, final_root):
    u0, _ = bindings(final_root)
    specs = N.episode_specs('U0', panel, None, Path(final_root)/N.PRERUN_PLANT, 'mappo_frozen', None, u0)
    return [dict(s, job_id=s['job_id'].replace('nmfinal_u0_', 'a32ident_u0_')) for s in specs[:N.IDENTITY_WORLDS]]


def check_a32(record, spec):
    """nm_finaleval.check_episode + the A32 pins (executor actually loaded, a32 digest loaded, receipts present)."""
    if spec['arm'] not in A32_ARMS:
        return _ORIG_CHECK(record, spec)
    what = f'{spec["arm"]} {spec["world_id"]}'
    out = _ORIG_CHECK(record, spec)
    if record.get('executor_sha256_loaded') != spec['checkpoint']['executor_sha256']:
        raise N.Blocked(f'{what}: executor actually loaded differs from its binding (never silent)')
    if record.get('a32_sha256_loaded') != spec['checkpoint']['a32_sha256']:
        raise N.Blocked(f'{what}: a32_arms actually loaded differs from its registration')
    if not isinstance(record.get('a32'), dict) or record['a32'].get('a32_arm') != spec['arm']:
        raise N.Blocked(f'{what}: a32 receipts missing')
    return dict(out, executor_sha256_loaded=record['executor_sha256_loaded'],
                a32_sha256_loaded=record['a32_sha256_loaded'], a32=deepcopy(record['a32']))


_ORIG_CHECK = N.check_episode
N.CHECKS['episode'] = check_a32


def execute(final_root, panel, waves, *, workers, chunk, log):
    final_root = Path(final_root)
    runtime = A32Runtime(N.Runtime(N.RUNNER), final_root)
    plant = final_root/N.PRERUN_PLANT
    cfg = runtime.config(plant)
    if not os.environ.get('COOPMWM_MAP_SPEC'):
        raise N.Blocked('set COOPMWM_MAP_SPEC before physics')
    status = runtime.runner_status()
    if not status['ok'] or not status.get('learned'):
        raise N.Blocked(f'runner cannot run the A32 arms: {status}')
    N.map_check(N.bind_map_api(None), panel)
    plan = N.prerun_plan(final_root)
    u0, _ = bindings(final_root)
    if (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
        raise N.Blocked('the U0 binding residual is not the pre-run u0 copy')

    def coverage():
        return {name: (N.published(final_root, name, specs, check_a32)[2], len(specs)) for name, _, specs in waves}
    rounds = 0
    while True:
        cov = coverage()
        pending = [w for w in waves if cov[w[0]][0] < cov[w[0]][1]]
        if not pending:
            break
        if (final_root/STOP_A32).exists():
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


# ------------------------------------------------------------------------------------------------ analysis (laptop)

def report_of(root, record):
    """The episode's vehicle-log report: the cached vehicles_report.json (log sha + version checked) or, when the log
    itself is present, nm_finaleval.episode_log_report."""
    d = Path(root)/'evaluation'/record.get('work_dir', record['job_id'])
    if (d/N.VEHICLE_LOG).is_file():
        return N.episode_log_report(root, record)
    cache = d/'vehicles_report.json'
    if not cache.is_file():
        raise N.Blocked(f'{record["job_id"]}: no vehicle log / report in {d}')
    c = N.read(cache)
    if c.get('log_sha256') != record['vehicle_log']['sha256'] or c.get('version') != N.REPORT_VERSION:
        raise N.Blocked(f'{record["job_id"]}: cached report does not match the sealed record')
    return c['report']


def load_arms(panel, bank_root, a32_root):
    have, roots, sources = {}, {}, {}
    u0, oracle = bindings(bank_root)
    for arm in ('U0', 'LEARNED_TRIGGER'):
        specs = N.episode_specs(arm, panel, None, Path(bank_root)/N.PRERUN_PLANT, 'mappo_frozen',
                                oracle if arm == 'LEARNED_TRIGGER' else None, u0)
        recs, src, cov = N.published(bank_root, arm, specs, _ORIG_CHECK)
        if cov != len(specs):
            raise N.Blocked(f'{arm}: {cov}/{len(specs)} published in {bank_root}')
        have[arm], roots[arm], sources[arm] = {r['world_id']: r for r in recs}, Path(bank_root), src
    for arm in A32_ARMS:
        specs = a32_specs(arm, panel, a32_root, check_files=False)
        recs, src, cov = N.published(a32_root, arm, specs, check_a32)
        if cov != len(specs):
            raise N.Blocked(f'{arm}: {cov}/{len(specs)} published in {a32_root}')
        have[arm], roots[arm], sources[arm] = {r['world_id']: r for r in recs}, Path(a32_root), src
    return have, roots, sources


def analyse(panel, bank_root, a32_root):
    import numpy as np
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    have, roots, sources = load_arms(panel, bank_root, a32_root)
    reports = {a: {w: report_of(roots[a], have[a][w]) for w in worlds} for a in have}
    if any(not r['ok'] for a in reports for r in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    contrasts = []
    for arm, ref in CONTRASTS:
        row = dict(treatment=arm, reference=ref, convention='diff = treatment - reference; negative = treatment has less delay')
        row.update(N._bank_compare(have, reports, worlds, arm, ref, bootstrap))
        hb = N.fe.paired_stats(np.array([have[arm][w]['safety']['hard_brakes'] for w in worlds], float),
                               np.array([have[ref][w]['safety']['hard_brakes'] for w in worlds], float), bootstrap)
        row['hard_brakes_minus_ref'] = hb
        contrasts.append(row)
    arms = {}
    for a in have:
        arms[a] = dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                       safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        rec = [have[a][w].get('a32') or have[a][w].get('learned_trigger') or {} for w in worlds]
        if any(rec):
            arms[a]['activations'] = dict(worlds_with_any=sum(1 for r in rec if r.get('activations')),
                                          total=sum(int(r.get('activations') or 0) for r in rec))
    def verdict(arm, ref):
        c = next(x for x in contrasts if (x['treatment'], x['reference']) == (arm, ref))
        return bool(c['total_minus_ref']['ci95'][1] < 0)
    ci = {f'{a}-{b}': next(x for x in contrasts if (x['treatment'], x['reference']) == (a, b))['total_minus_ref']['ci95']
          for a, b in CONTRASTS}
    lt_duo_sig = ci['LEARNED_TRIGGER-DUO_HOLD'][1] < 0 or ci['LEARNED_TRIGGER-DUO_HOLD'][0] > 0
    interp = dict(learning_when_to_start_adds_value=verdict('LEARNED_TRIGGER', 'BANK_ALWAYS'),
                  wm_bank_policy_beats_hand_written_hold=verdict('LEARNED_TRIGGER', 'DUO_HOLD'),
                  reframe_around_duo_hold=bool(verdict('DUO_HOLD', 'U0') and not lt_duo_sig),
                  rule=('pre-registered (amendment 32): learning adds value iff LT - BANK_ALWAYS CI upper < 0; WM bank + '
                        'policy beats the hand-written hold iff LT - DUO_HOLD CI upper < 0; reframe iff DUO_HOLD - U0 CI '
                        'upper < 0 and LT - DUO_HOLD not significant (either sign)'))
    return N.seal(dict(schema='coopmwm.a32_ablation.analysis.v1', amendment='32', authority=AUTHORITY,
                       panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), bootstrap=bootstrap,
                       contrasts=contrasts, arms=arms, interpretation=interp,
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()},
                       roots={k: str(v) for k, v in roots.items()},
                       pooling='never pooled with other amendments; U0 / LEARNED_TRIGGER = sealed amendment-28 records'),
                  'analysis_sha256')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('dry', 'identity', 'run', 'reports', 'identity-compare', 'analyse'))
    p.add_argument('--final-root', required=True)
    p.add_argument('--bank-root')
    p.add_argument('--arms', default=','.join(A32_ARMS))
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--chunk', type=int, default=48)
    p.add_argument('--out')
    a = p.parse_args(argv)
    panel = setup()
    root = Path(a.final_root)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T')), sort_keys=True, default=str)
        print(line, flush=True)
        with (root/'A32_LOG.jsonl').open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    arms = tuple(x for x in a.arms.split(',') if x)
    if not set(arms) <= set(A32_ARMS):
        p.error(f'arms are {A32_ARMS}')
    if a.command == 'dry':
        out = {arm: len(a32_specs(arm, panel, root)) for arm in arms}
        out['ident'] = [s['job_id'] for s in ident_specs(panel, root)]
        from tools.coopmwm import a32_arms
        out['a32_sha256'] = a32_arms.a32_digest()
        print(json.dumps(out, indent=1))
    elif a.command in ('identity', 'run'):
        import fcntl
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = ([(IDENT_WAVE, 'episode', ident_specs(panel, root))] if a.command == 'identity' else
                     [(arm, 'episode', a32_specs(arm, panel, root)) for arm in arms])
            log(dict(start=a.command, waves=[w[0] for w in waves], workers=a.workers, chunk=a.chunk, pid=os.getpid()))
            res = execute(root, panel, waves, workers=a.workers, chunk=a.chunk, log=log)
            log(dict(done=a.command, **res))
    elif a.command == 'reports':
        n = 0
        for arm in arms:
            recs, _, cov = N.published(root, arm, a32_specs(arm, panel, root), check_a32)
            for r in recs:
                rep = N.episode_log_report(root, r)
                if rep is None or not rep['ok']:
                    raise N.Blocked(f'{r["job_id"]}: report failed')
                n += 1
        print(json.dumps(dict(reports=n)))
    elif a.command == 'identity-compare':
        u0, _ = bindings(a.bank_root)
        specs = N.episode_specs('U0', panel, None, Path(a.bank_root)/N.PRERUN_PLANT, 'mappo_frozen', None, u0)
        first = Path(a.bank_root)/'WAVE_U0__0000-0035.json'       # only the 2 identity worlds are read
        value = N.verify(N.read(first), 'receipt_sha256', first.name)
        k = N.IDENTITY_WORLDS
        recs = [_ORIG_CHECK(r, s) for r, s in zip(value['records'][:k], specs[:k])]
        ref = {r['world_id']: r for r in recs}
        got = N._wave_records(root, IDENT_WAVE)
        rows = [dict(world_id=r['world_id'], j_gpuB=r['J_veh_s'], j_sealed=ref[r['world_id']]['J_veh_s'],
                     safety_equal=r['safety'] == ref[r['world_id']]['safety'],
                     vehicle_log_equal=r['vehicle_log']['sha256'] == ref[r['world_id']]['vehicle_log']['sha256'])
                for r in got]
        ok = len(rows) == N.IDENTITY_WORLDS and all(abs(x['j_gpuB']-x['j_sealed']) <= 1e-6 and x['safety_equal']
                                                   for x in rows)
        print(json.dumps(dict(verdict='PASS' if ok else 'FAILED', worlds=rows,
                              rule='|J diff| <= 1e-6 veh-s and identical safety counters'), indent=1))
    else:
        value = analyse(panel, Path(a.bank_root), root)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/'A32_RESULTS.json', value)
        print(json.dumps(dict(analysis_sha256=value['analysis_sha256'], interpretation=value['interpretation'])))


if __name__ == '__main__':
    main()
