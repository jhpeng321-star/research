"""AMENDMENT 38 (MAIN 2026-11-10; build, registration pending): the second road wz2to1 (2->1 lane drop). U0 (frozen
MAPPO + bound u0) and the hand-written hold ports HOLD_R2 (PRIMARY, degenerate single-CAV SUMO lane-1 hold of
duo12_v20) [and HOLD_R2_DUO01, the documented two-CAV alternative, only if MAIN registers it] on a fresh 96-world
wz2to1 panel, after a U0-only demand calibration on three TRAIN-type blocks (D1000 / D1200 / D1400, p20, 8 worlds
996000-996007 each). The WM-S model cannot load wz2to1 (road2.wm_feasibility): every WM stage fails closed.

Thin driver over ops/nm_finaleval.py (settings a38_cal_D<d> / a38_D<d>_p20) and the a32/a35 bindings (BOUND_U0,
PRERUN_PLAN, prerun_plant of the mirror root); waves through the unchanged nm_finaleval.run_waves (canary first,
per-episode ledger rows precharged in <final_root>/CHARGES.jsonl), one ledger phase per stage with its own cap.

ONE MAP PER PROCESS (N1). Env per command (paths relative to the source tree; absolute on the box):
  wz2to1 commands (dry, calibrate, run, reports, analyse, calsummary, wm-search):
      COOPMWM_MAP_SPEC=<src>/assets/research/newmap_candidates/wz2to1/MAP_SPEC.json
  wz3to2 command (identity, identity-compare):
      COOPMWM_MAP_SPEC=<src>/assets/research/newmap_candidates/wz3to2/MAP_SPEC.json

Commands (python ops/a38_road2.py <cmd> --final-root <root> [...]):
  dry        --setting S             zero physics: build every spec of the setting (+ the calibration blocks), print
  identity                           plant identity gate on sealed p20c worlds 993100/993101, exactly the a35 gate
                                     (U0 x 2 + LEARNED_TRIGGER x 2 with bank_cell_key), charged to road2_a38_eval
  identity-compare --bank-root R     J / safety of the identity waves vs the sealed amendment-28 records
  calibrate  --workers N --chunk M   U0 on the three calibration blocks (phase road2_a38_cal)
  calsummary --out DIR               zero physics: per demand U0 mean J, approach speed, gridlock flags
  run        --setting a38_D<d>_p20 --arms U0,HOLD_R2[,HOLD_R2_DUO01] --workers N --chunk M  (STOP_A38 stops
                                     after the chunk in flight; phase road2_a38_eval)
  reports    --setting S             zero physics: vehicles_report.json next to every published episode log
  analyse    --setting a38_D<d>_p20 --out DIR   per-arm J table + registered contrasts -> A38_RESULTS_<cell>.json
  wm-search                          always Blocked (WM-S cannot load wz2to1)
--cap overrides the phase cap of the command (MAIN decision only).
"""
from __future__ import annotations

import argparse
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
import a32_ablation as A  # noqa: E402  (report_of; registers check_a32)
import a35_transfer as T  # noqa: E402  (the identity gate; registers check_a35)

MAP = N.A38_MAP
A38_CAL_PHASE = 'road2_a38_cal'
A38_EVAL_PHASE = 'road2_a38_eval'
CAL_SETTINGS = N.A38_CAL_SETTINGS
EVAL_SETTINGS = N.A38_EVAL_SETTINGS
SETTINGS = CAL_SETTINGS+EVAL_SETTINGS
HOLD_ARMS = ('HOLD_R2', 'HOLD_R2_DUO01')
ARMS = ('U0',)+HOLD_ARMS
DEFAULT_ARMS = ('U0', 'HOLD_R2')
IDENT_SETTING = 'a35_D2400_p20'        # identity runs in a wz3to2 process (sealed p20c worlds; a35 code path)
IDENT_U0, IDENT_LT = 'A38_IDENT_U0', 'A38_IDENT_LT'
#: planned sim-s and caps (= planned + 10%, MAIN may override with --cap)
CAL_PLANNED_SIM_S = len(CAL_SETTINGS)*(N.A38_CAL_SEEDS[1]-N.A38_CAL_SEEDS[0]+1)*N.HORIZON_S  # 3 x 8 x 1200 = 28,800
CAL_CAP_SIM_S = 31_680
EVAL_PLANNED_SIM_S = 2*96*N.HORIZON_S + 2*N.IDENTITY_WORLDS*N.HORIZON_S   # 230,400 + identity 4,800 = 235,200
EVAL_CAP_SIM_S = 258_720
STOP_A38 = 'STOP_A38'
CONTRASTS = (('HOLD_R2', 'U0'), ('HOLD_R2_DUO01', 'U0'), ('HOLD_R2_DUO01', 'HOLD_R2'))
WM_BLOCK = 'WM-S cannot load wz2to1 (WM lane 0 has no road lane; road2.wm_feasibility)'
AUTHORITY = ('MAIN 2026-11-10 AMENDMENT 38 (second road wz2to1, 2->1 lane drop): U0 calibration on 3 TRAIN-type '
             'blocks (phase road2_a38_cal), then U0 vs HOLD_R2 on one fresh 96-world panel (phase road2_a38_eval); '
             'WM stages refused (road2.wm_feasibility)')


def env_hint(map_name):
    return (f'COOPMWM_MAP_SPEC=<src>/assets/research/newmap_candidates/{map_name}/MAP_SPEC.json '
            '(one map per process)')


def process_map():
    from tools.coopmwm import map_scenarios as ms
    return ms.MAP


def require_map(name):
    """Fail closed unless this process is bound to ``name`` (registry, descriptor and runner agree)."""
    from tools.coopmwm import map_descriptor as md
    from tools.coopmwm import nm_eval_runner as R
    got = process_map()
    if got != name or R.MAP != name:
        raise N.Blocked(f'this command needs a {name} process (process map {got}, runner {R.MAP}): set {env_hint(name)}')
    if os.environ.get(md.ENV) and md.current().name != name:
        raise N.Blocked(f'COOPMWM_MAP_SPEC names {md.current().name}, not {name}')
    return name


def setup(setting):
    """Apply a wz2to1 setting in a wz2to1 process and load its (registered) panel."""
    if setting not in SETTINGS:
        raise N.Blocked(f'setting is one of {SETTINGS}')
    require_map(MAP)
    N.apply_setting(setting)
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    return load_setting_panel(setting)


def load_setting_panel(name):
    """The panel file of setting ``name`` validated under the APPLIED setting (the three calibration settings share
    every validated key; only CELL / PANEL_FILE / texts differ); pinned by SETTINGS_PINS when MAIN pinned it."""
    s = N.SETTINGS[name]
    for k in ('MAP', 'NET_SHA256', 'EVAL_RANGE', 'TRAIN_RANGE', 'N_WORLDS', 'PANEL_TRAIN_TYPE'):
        if getattr(N, k) != s[k]:
            raise N.Blocked(f'{name}: {k} differs from the applied setting {N.SETTING}')
    path = Path(s['PANEL_FILE'])
    if not path.is_file():
        raise N.Blocked(f'{name}: panel {path.name} not registered (python ops/nm_finaleval.py register-panel '
                        f'--setting {name} --root <root> --final-root <final_root>, in a {MAP} process)')
    panel = N.load_panel(path, pinned=N.SETTINGS_PINS.get(name, {}).get('PANEL_SHA256'))
    want = s['CELL']
    if (panel['cell']['demand_vph']['main'], panel['cell']['penetration']) != (want['demand'], want['penetration']):
        raise N.Blocked(f'{name}: panel cell {panel["cell"]["cell_key"]} is not the setting cell')
    return panel


def cell_tag(panel):
    return panel['cell']['cell_key']


class A38Runtime(N.PrerunRuntime):
    """The pre-run plant of the mirror root, own ledger phase + cap (per stage); other phases are not debited."""

    def __init__(self, inner, final_root, phase, cap):
        super().__init__(inner, final_root)
        self.phase, self.cap = phase, int(cap)

    def _spent(self):
        rows = N.fe.ledger_rows(self.final_root)
        mine = sum(r['sim_s'] for r in rows if r.get('phase') == self.phase)
        return mine, sum(r['sim_s'] for r in rows)-mine

    def check_capacity(self, amount):
        mine, _ = self._spent()
        if mine+amount > self.cap:
            raise N.Blocked(f'A38: {self.phase} cap {self.cap} would be exceeded')

    def reserve(self, cfg, jobs):
        _, other = self._spent()
        N.own_reserve(self.final_root, jobs, self.phase, cap=self.cap, calls=self.calls, allocation=self.cap+other)


def u0_binding(final_root):
    u0 = N.mappo_binding(final_root, 'U0')
    if u0 is None:
        raise N.Blocked('A38 needs the bank-root U0 binding (BOUND_U0.json mirror)')
    return u0


def hold_spec(spec, arm, cell):
    """A U0 episode spec -> the hold arm: same executor + u0, checkpoint kind 'a38' + the registered code digest."""
    from tools.coopmwm import road2
    ck = deepcopy(spec['checkpoint'])
    ck.update(kind='a38', a38_arm=arm, a38_sha256=road2.a38_digest(), cell_key=cell,
              hold=road2.HOLDS[arm]['name'], hold_spec_sha256=road2.spec_digest(road2.HOLDS[arm]))
    return dict(spec, arm=arm, checkpoint=ck, job_id=spec['job_id'].replace('_u0_', f'_{arm.lower()}_', 1))


def arm_specs(arm, panel, final_root, *, prefix='a38'):
    """The arm's specs with job ids prefixed '<prefix>_<cell>_' (one root for every cell, no clash)."""
    if arm not in ARMS:
        raise N.Blocked(f'no A38 arm {arm!r}')
    final_root = Path(final_root)
    u0 = u0_binding(final_root)
    base = N.episode_specs('U0', panel, None, final_root/N.PRERUN_PLANT, 'mappo_frozen', None, u0)
    specs = base if arm == 'U0' else [hold_spec(s, arm, cell_tag(panel)) for s in base]
    tag = f'{prefix}_{cell_tag(panel).lower()}_'
    return [dict(s, job_id=tag+s['job_id']) for s in specs]


def wave_name(arm, panel, prefix='A38'):
    return f'{prefix}_{cell_tag(panel)}_{arm}'


def cal_waves(final_root):
    """(name, kind, specs) of the three calibration blocks (U0 only); needs an applied calibration setting."""
    out = []
    for name in CAL_SETTINGS:
        panel = load_setting_panel(name)
        out.append((wave_name('U0', panel, 'A38CAL'), 'episode', arm_specs('U0', panel, final_root, prefix='a38cal')))
    return out


def ident_specs(final_root):
    """Exactly the amendment-35 identity specs (U0 + LEARNED_TRIGGER with bank_cell_key on 993100/993101)."""
    u, l_ = T.ident_specs(final_root)
    return ([dict(s, job_id=s['job_id'].replace('a35ident_', 'a38ident_', 1)) for s in u],
            [dict(s, job_id=s['job_id'].replace('a35ident_', 'a38ident_', 1)) for s in l_])


_PREV_CHECK = N.CHECKS['episode']


def check_a38(record, spec):
    """The previous episode check (U0 / a32 / a35 unchanged) + the A38 pins for the hold arms."""
    if spec['arm'] not in HOLD_ARMS:
        return _PREV_CHECK(record, spec)
    what = f'{spec["arm"]} {spec["world_id"]}'
    out = N.check_episode(record, spec)
    if record.get('executor_sha256_loaded') != spec['checkpoint']['executor_sha256']:
        raise N.Blocked(f'{what}: executor actually loaded differs from its binding (never silent)')
    if record.get('a38_sha256_loaded') != spec['checkpoint']['a38_sha256']:
        raise N.Blocked(f'{what}: road2 actually loaded differs from its registration')
    if not isinstance(record.get('a38'), dict) or record['a38'].get('a38_arm') != spec['arm']:
        raise N.Blocked(f'{what}: a38 receipts missing')
    return dict(out, executor_sha256_loaded=record['executor_sha256_loaded'],
                a38_sha256_loaded=record['a38_sha256_loaded'], a38=deepcopy(record['a38']))


N.CHECKS['episode'] = check_a38


def execute(final_root, waves, *, phase, cap, workers, chunk, log, panels=(), holds=False, identity=False):
    final_root = Path(final_root)
    runtime = A38Runtime(N.Runtime(N.RUNNER), final_root, phase, cap)
    plant = final_root/N.PRERUN_PLANT
    cfg = runtime.config(plant)
    if not os.environ.get('COOPMWM_MAP_SPEC'):
        raise N.Blocked('set COOPMWM_MAP_SPEC before physics')
    status = runtime.runner_status()
    if not status['ok']:
        raise N.Blocked(f'runner cannot run on {N.MAP}: {status}')
    from tools.coopmwm import nm_eval_runner as R
    if holds and tuple(getattr(R, 'A38_ARMS', ())) != HOLD_ARMS:
        raise N.Blocked('runner lacks the amendment-38 hold arms')
    if identity and (not status.get('learned') or getattr(R, 'A35_BANK_CELL', None) != T.BANK_CELL):
        raise N.Blocked('runner lacks the amendment-35 identity path')
    for panel in panels:
        log(dict(map_check={k: v for k, v in N.map_check(N.bind_map_api(None), panel).items() if k != 'module'},
                 cell=cell_tag(panel)))
    plan = N.prerun_plan(final_root)
    u0 = u0_binding(final_root)
    if (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
        raise N.Blocked('the U0 binding residual is not the pre-run u0 copy')

    def coverage():
        return {name: (N.published(final_root, name, specs, check_a38)[2], len(specs)) for name, _, specs in waves}
    rounds = 0
    while True:
        cov = coverage()
        pending = [w for w in waves if cov[w[0]][0] < cov[w[0]][1]]
        if not pending:
            break
        if (final_root/STOP_A38).exists():
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
                 charged_sim_s=sum(r['sim_s'] for r in N.fe.ledger_rows(final_root) if r.get('phase') == phase)))
    cov = coverage()
    return dict(status='PASS', coverage=cov,
                charged_sim_s=sum(r['sim_s'] for r in N.fe.ledger_rows(final_root) if r.get('phase') == phase))


def identity_compare(final_root, bank_root):
    """J / safety / vehicle-log sha of the A38 identity waves vs the sealed amendment-28 records (first 2 worlds)."""
    bank_root = Path(bank_root)
    u_specs, l_specs = ident_specs(final_root)
    rows = []
    for wave, sealed_prefix, specs in ((IDENT_U0, 'WAVE_U0__', u_specs), (IDENT_LT, 'WAVE_LEARNED_TRIGGER__', l_specs)):
        first = sorted(bank_root.glob(sealed_prefix+'0000-*.json'))
        if len(first) != 1:
            raise N.Blocked(f'{bank_root}: no unique {sealed_prefix}0000-* wave file')
        value = N.verify(N.read(first[0]), 'receipt_sha256', first[0].name)
        ref = {r['world_id']: r for r in value['records'][:N.IDENTITY_WORLDS]}
        got, _, cov = N.published(final_root, wave, specs, check_a38)
        if cov != len(specs):
            raise N.Blocked(f'{wave}: {cov}/{len(specs)} published')
        for r in got:
            s = ref[r['world_id']]
            rows.append(dict(wave=wave, world_id=r['world_id'], j=r['J_veh_s'], j_sealed=s['J_veh_s'],
                             abs_diff=abs(r['J_veh_s']-s['J_veh_s']), safety_equal=r['safety'] == s['safety'],
                             vehicle_log_equal=r['vehicle_log']['sha256'] == s['vehicle_log']['sha256']))
    ok = len(rows) == 2*N.IDENTITY_WORLDS and all(x['abs_diff'] <= 1e-6 and x['safety_equal'] for x in rows)
    return dict(verdict='PASS' if ok else 'FAILED', worlds=rows, rule='|J diff| <= 1e-6 veh-s and identical safety')


def wm_search(*_a, **_k):
    """The WM-S search / verification stage of the chain: refused on wz2to1 (fail closed)."""
    raise N.Blocked(WM_BLOCK)


# ------------------------------------------------------------------------------------------------ analysis

def _records(root, panel, arm, prefix='a38', wave_prefix='A38'):
    specs = arm_specs(arm, panel, root, prefix=prefix)
    recs, src, cov = N.published(root, wave_name(arm, panel, wave_prefix), specs, check_a38)
    return recs, src, cov, len(specs)


def analyse(panel, root, arms):
    """Per-arm J table + the registered contrasts among ``arms``: paired percentile bootstrap (10,000, seed 20261104),
    diff = treatment - reference (negative = less delay); nm_finaleval._bank_compare (total, group co-criteria with
    the U0-labelled origins - origin:L2 (WM lane 0) is empty on wz2to1 - and CVaR90), hard brakes / TTC events."""
    import numpy as np
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile'):
        raise N.Blocked('panel bootstrap is not the registered one')
    have, sources = {}, {}
    for arm in arms:
        recs, src, cov, n = _records(root, panel, arm)
        if cov != n:
            raise N.Blocked(f'{arm}: {cov}/{n} published in {root}')
        have[arm], sources[arm] = {r['world_id']: r for r in recs}, src
    reports = {a: {w: A.report_of(root, have[a][w]) for w in worlds} for a in have}
    if any(not r['ok'] for a in reports for r in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    contrasts = []
    for arm, ref in CONTRASTS:
        if arm not in have or ref not in have:
            continue
        row = dict(treatment=arm, reference=ref, convention='diff = treatment - reference; negative = treatment has less delay')
        row.update(N._bank_compare(have, reports, worlds, arm, ref, bootstrap))
        for k in ('hard_brakes', 'ttc_events'):
            row[f'{k}_minus_ref'] = N.fe.paired_stats(
                np.array([have[arm][w]['safety'][k] for w in worlds], float),
                np.array([have[ref][w]['safety'][k] for w in worlds], float), bootstrap)
        contrasts.append(row)
    table = {}
    for a in have:
        js = [have[a][w]['J_veh_s'] for w in worlds]
        table[a] = dict(mean_J_veh_s=float(np.mean(js)), median_J_veh_s=float(np.median(js)),
                        sd_J_veh_s=float(np.std(js, ddof=1)) if len(js) > 1 else None,
                        safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        rec = [have[a][w].get('a38') or {} for w in worlds]
        if any(rec):
            fam = {}
            for r in rec:
                for f, n in (r.get('activations_by_family') or {}).items():
                    fam[f] = fam.get(f, 0)+int(n)
            table[a]['activations'] = dict(worlds_with_any=sum(1 for r in rec if r.get('activations')),
                                           total=sum(int(r.get('activations') or 0) for r in rec), by_family=fam)
    return N.seal(dict(schema='coopmwm.a38_road2.analysis.v1', amendment='38', authority=AUTHORITY, map=N.MAP,
                       cell=panel['cell']['cell_key'], panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                       bootstrap=bootstrap, arms=table, contrasts=contrasts,
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()}, root=str(root),
                       label='amendment 38, second road wz2to1; one panel, one shot'), 'analysis_sha256')


def approach_speed(log_path, drop_s=N.DROP_S_M):
    """Zero physics, from the ground-truth vehicle log: mean speed of every logged row upstream of the drop
    (s < drop_s) over the logged control window, and the number of such rows."""
    total, n = 0.0, 0
    with gzip.open(log_path, 'rt', encoding='utf-8') as f:
        for line in f:
            x = json.loads(line)
            if 'veh' in x:
                continue
            for s, v in zip(x.get('s', ()), x.get('v', ())):
                if s < drop_s:
                    total += float(v)
                    n += 1
    return (total/n if n else None), n


def calsummary(root):
    """Per calibration demand: U0 mean J, mean approach speed (vehicle logs, if present), gridlock flags."""
    import numpy as np
    out = {}
    for name in CAL_SETTINGS:
        panel = load_setting_panel(name)
        recs, _, cov, n = _records(root, panel, 'U0', prefix='a38cal', wave_prefix='A38CAL')
        row = dict(setting=name, cell=cell_tag(panel), worlds_published=cov, worlds=n,
                   seeds=[panel['worlds'][0]['world']['seed'], panel['worlds'][-1]['world']['seed']])
        if recs:
            row['mean_J_veh_s'] = float(np.mean([r['J_veh_s'] for r in recs]))
            row['J_veh_s'] = [r['J_veh_s'] for r in recs]
            row['safety_totals'] = {k: int(sum(r['safety'][k] for r in recs)) for k in N.SAFETY}
            speeds, logs, vehicles, dmax, cross = [], 0, [], [], []
            for r in recs:
                d = Path(root)/'evaluation'/r.get('work_dir', r['job_id'])
                vehicles.append(r['vehicle_log']['vehicles'])
                if (d/N.VEHICLE_LOG).is_file():
                    logs += 1
                    v, _ = approach_speed(d/N.VEHICLE_LOG)
                    speeds.append(v)
                    rep = N.episode_log_report(root, r)
                    dmax.append((rep.get('delay') or {}).get('max_s'))
                    cross.append(rep.get('crossings_drop'))
            row['vehicle_logs'] = logs
            row['vehicles_logged_mean'] = float(np.mean(vehicles))
            row['mean_approach_speed_mps'] = (float(np.mean([s for s in speeds if s is not None]))
                                              if any(s is not None for s in speeds) else None)
            row['drop_crossings_mean'] = (float(np.mean([c for c in cross if c is not None]))
                                          if any(c is not None for c in cross) else None)
            row['max_vehicle_delay_s'] = max((x for x in dmax if x is not None), default=None)
            row['gridlock_flags'] = dict(
                collisions=row['safety_totals']['collisions'] > 0, teleports=row['safety_totals']['teleports'] > 0,
                approach_below_2mps=row['mean_approach_speed_mps'] is not None and row['mean_approach_speed_mps'] < 2.0,
                note='vehicles never inserted are not in the log; their source waiting is inside J (plant team delay)')
        out[name] = row
    return N.seal(dict(schema='coopmwm.a38_road2.calsummary.v1', amendment='38', map=N.MAP, root=str(root),
                       cells=out, physics='zero (published records + vehicle logs only)'), 'summary_sha256')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('dry', 'identity', 'identity-compare', 'calibrate', 'calsummary', 'run',
                                       'reports', 'analyse', 'wm-search'))
    p.add_argument('--final-root', required=True)
    p.add_argument('--setting', choices=SETTINGS)
    p.add_argument('--arms', default=','.join(DEFAULT_ARMS))
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--chunk', type=int, default=48)
    p.add_argument('--cap', type=int, help='override the phase cap of this command (MAIN decision)')
    p.add_argument('--bank-root')
    p.add_argument('--out')
    a = p.parse_args(argv)
    if a.command == 'wm-search':
        wm_search()
    root = Path(a.final_root)
    arms = tuple(x for x in a.arms.split(',') if x)
    if not set(arms) <= set(ARMS) or 'U0' not in arms:
        p.error(f'arms are U0 + a subset of {HOLD_ARMS}')

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T')), sort_keys=True, default=str)
        print(line, flush=True)
        with (root/'A38_LOG.jsonl').open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    def locked(fn):
        import fcntl
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            return fn()
    if a.command in ('identity', 'identity-compare'):
        require_map('wz3to2')
        T.setup(IDENT_SETTING)
        if a.command == 'identity-compare':
            print(json.dumps(identity_compare(root, a.bank_root), indent=1))
            return
        u, l_ = ident_specs(root)
        waves = [(IDENT_U0, 'episode', u), (IDENT_LT, 'episode', l_)]

        def go():
            log(dict(start='identity', waves=[w[0] for w in waves], workers=a.workers, pid=os.getpid()))
            res = execute(root, waves, phase=A38_EVAL_PHASE, cap=a.cap or EVAL_CAP_SIM_S, workers=a.workers,
                          chunk=a.chunk, log=log, identity=True)
            log(dict(done='identity', **res))
        return locked(go)
    if a.command in ('calibrate', 'calsummary'):
        setup(a.setting if a.setting in CAL_SETTINGS else CAL_SETTINGS[0])
        if a.command == 'calsummary':
            value = calsummary(root)
            out = Path(a.out)
            out.mkdir(parents=True, exist_ok=True)
            N.publish(out/'A38_CALSUMMARY.json', value)
            print(json.dumps({k: dict(mean_J_veh_s=v.get('mean_J_veh_s'), approach=v.get('mean_approach_speed_mps'),
                                      flags=v.get('gridlock_flags')) for k, v in value['cells'].items()}, indent=1))
            return
        waves = cal_waves(root)

        def go():
            log(dict(start='calibrate', waves=[w[0] for w in waves], workers=a.workers, chunk=a.chunk, pid=os.getpid()))
            res = execute(root, waves, phase=A38_CAL_PHASE, cap=a.cap or CAL_CAP_SIM_S, workers=a.workers,
                          chunk=a.chunk, log=log, panels=[load_setting_panel(n) for n in CAL_SETTINGS])
            log(dict(done='calibrate', **res))
        return locked(go)
    if a.setting is None:
        p.error(f'{a.command}: --setting required')
    panel = setup(a.setting)
    is_cal = a.setting in CAL_SETTINGS
    if a.command == 'dry':
        out = dict(setting=a.setting, map=N.MAP, panel_sha256=panel['panel_sha256'], cell=cell_tag(panel),
                   pinned=N.SETTINGS_PINS.get(a.setting, {}).get('PANEL_SHA256'))
        from tools.coopmwm import road2
        out['a38_sha256'] = road2.a38_digest()
        for arm in (('U0',) if is_cal else arms):
            specs = arm_specs(arm, panel, root, prefix='a38cal' if is_cal else 'a38')
            out[arm] = dict(n=len(specs), first=specs[0]['job_id'], cell=specs[0]['world']['cell_key'],
                            seeds=[specs[0]['world']['seed'], specs[-1]['world']['seed']],
                            sim_s=sum(s['sim_s'] for s in specs))
        out['caps'] = {A38_CAL_PHASE: CAL_CAP_SIM_S, A38_EVAL_PHASE: EVAL_CAP_SIM_S}
        print(json.dumps(out, indent=1))
        return out
    if is_cal and a.command in ('run', 'analyse'):
        p.error('calibration settings run through `calibrate` / `calsummary`')
    if a.command == 'run':
        waves = [(wave_name(arm, panel), 'episode', arm_specs(arm, panel, root)) for arm in arms]

        def go():
            log(dict(start='run', setting=a.setting, waves=[w[0] for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid()))
            res = execute(root, waves, phase=A38_EVAL_PHASE, cap=a.cap or EVAL_CAP_SIM_S, workers=a.workers,
                          chunk=a.chunk, log=log, panels=[panel], holds=any(x in HOLD_ARMS for x in arms))
            log(dict(done='run', **res))
        return locked(go)
    if a.command == 'analyse':
        value = analyse(panel, root, arms)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/f'A38_RESULTS_{cell_tag(panel)}.json', value)
        print(json.dumps(dict(analysis_sha256=value['analysis_sha256'], cell=value['cell'])))
        return value
    n = 0                                                    # reports
    for arm in (('U0',) if is_cal else arms):
        prefix, wprefix = ('a38cal', 'A38CAL') if is_cal else ('a38', 'A38')
        recs, _, _, _ = _records(root, panel, arm, prefix=prefix, wave_prefix=wprefix)
        for r in recs:
            rep = N.episode_log_report(root, r)
            if rep is None or not rep['ok']:
                raise N.Blocked(f'{r["job_id"]}: report failed')
            n += 1
    print(json.dumps(dict(reports=n)))


if __name__ == '__main__':
    main()
