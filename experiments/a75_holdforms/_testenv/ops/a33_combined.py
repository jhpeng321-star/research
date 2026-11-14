"""AMENDMENT 33 (MAIN 2026-11-10, owner "register option 1 and start it"; registered before any amendment-33 data):
COMBINED bank (5 WM plans fd385a2b + hand-written duo12_v20, retrained classifier gate) on a FRESH 192-world
D2800_p20 panel 999500-999691; arms U0, DUO_HOLD, LEARNED_TRIGGER (unchanged, replication), COMBINED.

Thin driver over ops/nm_finaleval.py and ops/a32_ablation.py (reused: the A32 DUO_HOLD / SOLO_HOLD specs and checks):
the specs are the registered LEARNED_TRIGGER / U0 episode specs of the mirrored bank-root bindings (BOUND_U0,
ORACLE_BINDING, PRERUN_PLAN, prerun_plant) built on the a33 panel; waves run through the unchanged
nm_finaleval.run_waves (canary first, per-episode ledger rows precharged in <final_root>/CHARGES.jsonl, phase
combined_a33, cap 1,100,000 sim-s shared by the classifier-data block, identity, smoke and the panel).

Settings: --setting a33 (the FRESH panel) | a33t (the 40-world TRAIN-type classifier-data block 991600-991639).

Commands (python -m ops.a33_combined <cmd> --final-root <root> [--setting a33|a33t]):
  dry       zero physics: build every spec, check bindings/digests, print counts
  identity  (a33 setting, p20c panel) plant identity gate: U0 on the first 2 sealed p20c worlds (wave A33_IDENT_U0)
  train     (a33t) U0, DUO_HOLD, SOLO_HOLD on the classifier-data block; observable tapes kept
  run       (a33) --arms U0,DUO_HOLD,LEARNED_TRIGGER,COMBINED --workers 24 --chunk 48 (STOP_A33 stops after the chunk)
  acts      (a33t, zero physics) ACTS.json of the duo12_v20 executions (J(duo) - J(u0) per seed) + tape index
  reports   zero physics: vehicles_report.json next to every published episode log
  analyse   (laptop, a33) --out DIR: the registered amendment-33 contrasts -> A33_RESULTS.json
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
import a32_ablation as A  # noqa: E402  (registers check_a32 for the A32 arms)

A33_PHASE = 'combined_a33'
A33_CAP_SIM_S = 1_100_000
STOP_A33 = 'STOP_A33'
IDENT_WAVE = 'A33_IDENT_U0'
PANEL_ARMS = ('U0', 'DUO_HOLD', 'LEARNED_TRIGGER', 'COMBINED')
TRAIN_ARMS = ('U0', 'DUO_HOLD', 'SOLO_HOLD')
TRAIN_PREFIX = 'A33T_'
MODEL_REL = 'a33_model/COMBINED_MODEL.json'
AUTHORITY = ('MAIN 2026-11-10 AMENDMENT 33 (owner "register option 1 and start it"): COMBINED (WM plans fd385a2b + '
             'duo12_v20, retrained classifier) on the FRESH panel 999500-999691; U0, DUO_HOLD, LEARNED_TRIGGER, '
             'COMBINED; phase combined_a33 cap 1,100,000 sim-s')
CONTRASTS = (('COMBINED', 'U0'), ('COMBINED', 'DUO_HOLD'), ('LEARNED_TRIGGER', 'U0'), ('DUO_HOLD', 'U0'),
             ('COMBINED', 'LEARNED_TRIGGER'))


def setup(setting):
    N.apply_setting(setting)
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    return N.load_panel(N.PANEL_FILE)


class A33Runtime(N.PrerunRuntime):
    """The pre-run plant of the mirror root, own ledger phase combined_a33, cap = allocation = 1,100,000."""
    phase = A33_PHASE

    def check_capacity(self, amount):
        rows = N.fe.ledger_rows(self.final_root)
        if sum(r['sim_s'] for r in rows)+amount > A33_CAP_SIM_S:
            raise N.Blocked(f'A33: {A33_PHASE} cap {A33_CAP_SIM_S} would be exceeded')

    def reserve(self, cfg, jobs):
        N.own_reserve(self.final_root, jobs, A33_PHASE, cap=A33_CAP_SIM_S, calls=self.calls, allocation=A33_CAP_SIM_S)


def model_entry(final_root, check_files=True):
    path = Path(final_root)/MODEL_REL
    pin = Path(str(path)+'.sha256')
    sha = pin.read_text(encoding='utf-8').split()[0] if pin.is_file() else None
    if check_files and (sha is None or N.file_sha(path) != sha):
        raise N.Blocked(f'{path}: COMBINED model doc missing or its sha256 differs from its pin')
    return dict(path=str(path), file_sha256=sha)


def combined_specs(panel, final_root, *, check_files=True):
    """The registered LEARNED_TRIGGER specs with arm -> COMBINED, checkpoint kind 'a33' (+ duo bank, model, digest)."""
    from tools.coopmwm import a33_combined, a32_arms
    final_root = Path(final_root)
    u0, oracle = A.bindings(final_root)
    if N.arm_bank(oracle, 'LEARNED_TRIGGER')['file_sha256'] != a33_combined.WM_BANK_SHA256:
        raise N.Blocked('A33: the LEARNED_TRIGGER bank of the binding is not fd385a2b')
    base = N.episode_specs('LEARNED_TRIGGER', panel, None, final_root/N.PRERUN_PLANT, 'mappo_frozen', oracle, u0)
    duo = final_root/A.DUO_BANK_REL
    if check_files and N.file_sha(duo) != a32_arms.DUO_BANK_SHA256:
        raise N.Blocked(f'{duo}: not UNION_UNC 39f384a2')
    model = model_entry(final_root, check_files)
    out = []
    for s in base:
        ck = deepcopy(s['checkpoint'])
        ck.pop('learned_trigger_sha256')
        ck.pop('learned_trigger_model')
        ck.update(kind='a33', a33_sha256=a33_combined.a33_digest(), combined_model=model,
                  duo_bank=dict(path=str(duo), file_sha256=a32_arms.DUO_BANK_SHA256))
        out.append(dict(s, arm='COMBINED', checkpoint=ck, job_id=s['job_id'].replace('learned_trigger', 'combined')))
    return out


def arm_specs(arm, panel, final_root, *, check_files=True):
    """The arm's specs with job ids prefixed 'a33_' (panel) / 'a33t_' (classifier-data block): one root, no clash."""
    tag = dict(a33='a33_', a33t='a33t_')[N.SETTING]
    return [dict(s, job_id=tag+s['job_id']) for s in _arm_specs(arm, panel, final_root, check_files=check_files)]


def _arm_specs(arm, panel, final_root, *, check_files=True):
    final_root = Path(final_root)
    u0, oracle = A.bindings(final_root)
    plant = final_root/N.PRERUN_PLANT
    if arm == 'U0':
        return N.episode_specs('U0', panel, None, plant, 'mappo_frozen', None, u0)
    if arm == 'LEARNED_TRIGGER':
        return N.episode_specs('LEARNED_TRIGGER', panel, None, plant, 'mappo_frozen', oracle, u0)
    if arm in A.A32_ARMS:
        return A.a32_specs(arm, panel, final_root, check_files=check_files)
    if arm == 'COMBINED':
        return combined_specs(panel, final_root, check_files=check_files)
    raise N.Blocked(f'no A33 arm {arm!r}')


def wave_name(arm, setting):
    return (TRAIN_PREFIX if setting == 'a33t' else 'A33_')+arm


def ident_specs(final_root):
    """U0 on the first 2 sealed p20c worlds (the a28 bank-root U0 identity worlds), job ids a33ident_u0_*."""
    p20c = N.read(HERE/'nm_eval_panel_wz3to2_p20c_D2800.json')
    N.verify(p20c, 'panel_sha256', 'p20c panel')
    if p20c['panel_sha256'] != 'd2ed7783ae9ea071f89aec4397d3ee155161878b44e1717b02188b16e7ff7fa2':
        raise N.Blocked('identity worlds: p20c panel sha differs')
    u0, _ = A.bindings(final_root)
    specs = N.episode_specs('U0', p20c, None, Path(final_root)/N.PRERUN_PLANT, 'mappo_frozen', None, u0)
    return [dict(s, job_id=s['job_id'].replace('nmfinal_u0_', 'a33ident_u0_')) for s in specs[:N.IDENTITY_WORLDS]]


def check_a33(record, spec):
    if spec['arm'] != 'COMBINED':
        return A.check_a32(record, spec)
    what = f'{spec["arm"]} {spec["world_id"]}'
    out = A._ORIG_CHECK(record, spec)
    if record.get('executor_sha256_loaded') != spec['checkpoint']['executor_sha256']:
        raise N.Blocked(f'{what}: executor actually loaded differs from its binding (never silent)')
    if record.get('a33_sha256_loaded') != spec['checkpoint']['a33_sha256']:
        raise N.Blocked(f'{what}: a33_combined actually loaded differs from its registration')
    if not isinstance(record.get('a33'), dict) or record['a33'].get('executor') != 'combined':
        raise N.Blocked(f'{what}: a33 receipts missing')
    return dict(out, executor_sha256_loaded=record['executor_sha256_loaded'],
                a33_sha256_loaded=record['a33_sha256_loaded'], a33=deepcopy(record['a33']))


N.CHECKS['episode'] = check_a33


def execute(final_root, panel, waves, *, workers, chunk, log, map_panel=True):
    final_root = Path(final_root)
    runtime = A33Runtime(N.Runtime(N.RUNNER), final_root)
    plant = final_root/N.PRERUN_PLANT
    cfg = runtime.config(plant)
    if not os.environ.get('COOPMWM_MAP_SPEC'):
        raise N.Blocked('set COOPMWM_MAP_SPEC before physics')
    status = runtime.runner_status()
    if not status['ok'] or not status.get('learned'):
        raise N.Blocked(f'runner cannot run the A33 arms: {status}')
    if map_panel:
        N.map_check(N.bind_map_api(None), panel)
    plan = N.prerun_plan(final_root)
    u0, _ = A.bindings(final_root)
    if (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
        raise N.Blocked('the U0 binding residual is not the pre-run u0 copy')

    def coverage():
        return {name: (N.published(final_root, name, specs, check_a33)[2], len(specs)) for name, _, specs in waves}
    rounds = 0
    while True:
        cov = coverage()
        pending = [w for w in waves if cov[w[0]][0] < cov[w[0]][1]]
        if not pending:
            break
        if (final_root/STOP_A33).exists():
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


def train_acts(panel, final_root):
    """(ACTS list, tape index): every duo12_v20 activation of the A33T DUO_HOLD episodes with the episode-level
    diff J(DUO_HOLD) - J(U0) on that seed; tapes = observable.jsonl.gz of the U0 / DUO_HOLD / SOLO_HOLD work dirs."""
    from tools.coopmwm import a33_classifier as K
    recs, tapes = {}, {}
    arm_tag = dict(U0='u0', DUO_HOLD=K.C.DUO_FAMILY, SOLO_HOLD=K.T.INCUMBENT)
    for arm in TRAIN_ARMS:
        specs = arm_specs(arm, panel, final_root, check_files=False)
        got, _, cov = N.published(final_root, wave_name(arm, 'a33t'), specs, check_a33)
        if cov != len(specs):
            raise N.Blocked(f'{arm}: {cov}/{len(specs)} published')
        for r in got:
            seed = int(r['world']['seed'])
            recs[(seed, arm)] = r
            work = Path(final_root)/'evaluation'/r.get('work_dir', r['job_id'])
            tapes[K.tape_key(seed, arm_tag[arm])] = str(work/'observable.jsonl.gz')
    return K.acts_doc(recs), tapes


# ------------------------------------------------------------------------------------------------ analysis (laptop)

def analyse(panel, root):
    """The registered amendment-33 analysis on the 4 panel arms of ``root`` (records = the sealed wave receipts;
    groups = vehicle-log reports): paired percentile bootstrap (10,000, seed 20261104), diff = treatment - reference
    (negative = less delay); 5 fairness groups (10 s rule, U0 labels), CVaR90, safety. PRIMARY: COMBINED - U0 total
    delay CI95 upper < 0 AND every group mean increase <= 10 s/veh AND 0 collisions."""
    import numpy as np
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile'):
        raise N.Blocked('panel bootstrap is not the registered one')
    have, sources = {}, {}
    for arm in PANEL_ARMS:
        specs = arm_specs(arm, panel, root, check_files=False)
        recs, src, cov = N.published(root, wave_name(arm, 'a33'), specs, check_a33)
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
        rec = [have[a][w].get('a33') or have[a][w].get('a32') or have[a][w].get('learned_trigger') or {}
               for w in worlds]
        if any(rec):
            fam = {}
            for r in rec:
                for f, n in (r.get('activations_by_family') or {}).items():
                    fam[f] = fam.get(f, 0)+int(n)
            arms[a]['activations'] = dict(worlds_with_any=sum(1 for r in rec if r.get('activations')),
                                          total=sum(int(r.get('activations') or 0) for r in rec), by_family=fam,
                                          scores=sum(int(r.get('scores') or 0) for r in rec),
                                          fires=sum(int(r.get('fires') or 0) for r in rec if r.get('fires') is not None))
    prim = next(x for x in contrasts if (x['treatment'], x['reference']) == ('COMBINED', 'U0'))
    groups_ok = (prim.get('groups') or {}).get('all_pass') is True
    coll = arms['COMBINED']['safety_totals'].get('collisions', 0)
    primary = dict(total_ci95_upper_below_0=bool(prim['total_minus_ref']['ci95'][1] < 0),
                   every_group_increase_le_10s=bool(groups_ok), zero_collisions=bool(coll == 0),
                   passes=bool(prim['total_minus_ref']['ci95'][1] < 0 and groups_ok and coll == 0),
                   rule=('PRIMARY (amendment 33): COMBINED - U0 total delay 95% CI upper < 0 (paired percentile '
                         'bootstrap, 10,000, seed 20261104) AND every group mean increase <= 10 s/veh AND 0 collisions'))
    return N.seal(dict(schema='coopmwm.a33_combined.analysis.v1', amendment='33', authority=AUTHORITY,
                       panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), bootstrap=bootstrap,
                       primary=primary, contrasts=contrasts, arms=arms,
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()}, root=str(root),
                       pooling='never pooled with other panels; one shot, no outcome-based retries'),
                  'analysis_sha256')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('dry', 'identity', 'train', 'run', 'acts', 'reports', 'analyse'))
    p.add_argument('--final-root', required=True)
    p.add_argument('--setting', default='a33', choices=('a33', 'a33t'))
    p.add_argument('--arms', default='')
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--chunk', type=int, default=48)
    p.add_argument('--out')
    a = p.parse_args(argv)
    if a.command == 'train' and a.setting != 'a33t' or a.command == 'run' and a.setting != 'a33':
        p.error('train runs with --setting a33t; run with --setting a33')
    panel = setup(a.setting)
    root = Path(a.final_root)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T')), sort_keys=True, default=str)
        print(line, flush=True)
        with (root/'A33_LOG.jsonl').open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    allowed = TRAIN_ARMS if a.setting == 'a33t' else PANEL_ARMS
    arms = tuple(x for x in a.arms.split(',') if x) or allowed
    if not set(arms) <= set(allowed):
        p.error(f'arms are {allowed}')
    if a.command == 'dry':
        out = {arm: len(arm_specs(arm, panel, root, check_files=arm != 'COMBINED')) for arm in arms}
        out['ident'] = [s['job_id'] for s in ident_specs(root)]
        from tools.coopmwm import a33_combined
        out['a33_sha256'] = a33_combined.a33_digest()
        out['panel_sha256'] = panel['panel_sha256']
        print(json.dumps(out, indent=1))
    elif a.command in ('identity', 'train', 'run'):
        import fcntl
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            if a.command == 'identity':
                waves = [(IDENT_WAVE, 'episode', ident_specs(root))]
            else:
                waves = [(wave_name(arm, a.setting), 'episode', arm_specs(arm, panel, root)) for arm in arms]
            log(dict(start=a.command, setting=a.setting, waves=[w[0] for w in waves], workers=a.workers,
                     chunk=a.chunk, pid=os.getpid()))
            res = execute(root, panel, waves, workers=a.workers, chunk=a.chunk, log=log,
                          map_panel=a.command != 'identity')
            log(dict(done=a.command, **res))
    elif a.command == 'acts':
        acts, tapes = train_acts(panel, root)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out/'ACTS.json').write_text(json.dumps(acts, indent=1, sort_keys=True)+'\n', encoding='utf-8')
        tdir = out/'tapes'
        tdir.mkdir(exist_ok=True)
        for key, src in sorted(tapes.items()):
            dst = tdir/f'{key}.jsonl.gz'
            if not dst.exists():
                os.symlink(src, dst)
        print(json.dumps(dict(acts=len(acts), helped=sum(x['diff'] < 0 for x in acts),
                              hurt=sum(x['diff'] > 0 for x in acts), tapes=len(tapes))))
    elif a.command == 'analyse':
        value = analyse(panel, root)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/'A33_RESULTS.json', value)
        print(json.dumps(dict(analysis_sha256=value['analysis_sha256'], primary=value['primary'])))
    else:
        n = 0
        for arm in arms:
            recs, _, cov = N.published(root, wave_name(arm, a.setting), arm_specs(arm, panel, root), check_a33)
            for r in recs:
                rep = N.episode_log_report(root, r)
                if rep is None or not rep['ok']:
                    raise N.Blocked(f'{r["job_id"]}: report failed')
                n += 1
        print(json.dumps(dict(reports=n)))


if __name__ == '__main__':
    main()
