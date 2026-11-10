"""AMENDMENT 40 (MAIN 2026-11-10; build, full-run registration pending): frozen-controller transfer to the wz3to2
GEOMETRY VARIANT wz3to2v80 (identical 3-lane approach and lane drop; the 2-lane work zone posted 22.22 m/s instead of
31.29; E:/agent-work/a40/VARIANT_CHOICE.md sha256 14c64b08, fixed before any controller arm). Arms exactly as in
amendment 35: U0 (frozen MAPPO d9042cc5 + bound u0), LEARNED_TRIGGER (bank fd385a2b + classifier 8e0ddd3e, threshold
0.378785) and DUO_HOLD (duo12_v20), the bank rows admitted under their source cell D2800_p20 (runner bank_cell_key).
No retuning; plans are applied mechanically (zones relative to the drop, which is unchanged).

Thin driver over ops/nm_finaleval.py (settings a40_cal_D<d> / a40_D<d>_p20) and the a32/a35 bindings of the mirror
root (BOUND_U0, ORACLE_BINDING, PRERUN_PLAN, prerun_plant); waves through the unchanged nm_finaleval.run_waves (canary
first, per-episode ledger rows precharged in <final_root>/CHARGES.jsonl). One process = one map: wz3to2v80 commands
need COOPMWM_MAP_SPEC = the wz3to2v80 MAP_SPEC; the identity gate needs the wz3to2 MAP_SPEC.
  dry        --setting S                zero physics: specs, digests, counts
  identity                              plant identity gate on sealed p20c worlds 993100/993101 (U0 x2 + LT x2 via
                                        the a35 path), wz3to2 process, charged to geomvar_a40_eval
  identity-compare --bank-root DIR      J/safety vs the sealed amendment-28 first-wave records
  calibrate  --workers N --chunk M      U0 on the three calibration blocks (phase geomvar_a40_cal)
  calsummary --out DIR                  zero physics: per demand U0 mean J, approach speed, safety
  run        --setting a40_D<d>_p20 --arms U0,LEARNED_TRIGGER,DUO_HOLD   (STOP_A40 stops after the chunk in flight)
  reports    --setting S                zero physics: vehicles_report.json next to every published episode log
  analyse    --setting a40_D<d>_p20 --out DIR    registered contrasts -> A40_RESULTS_<cell>.json
--cap overrides the phase cap of the command (MAIN decision only).
"""
from __future__ import annotations

import argparse
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
import a32_ablation as A  # noqa: E402  (report_of, bindings, a32_specs; registers check_a32)
import a35_transfer as T  # noqa: E402  (transfer(), BANK_CELL, identity specs; registers check_a35)

MAP = N.A40_MAP
BASE_MAP = 'wz3to2'
CAL_PHASE = 'geomvar_a40_cal'
EVAL_PHASE = 'geomvar_a40_eval'
SMOKE_PHASE = 'geomvar_a40_smoke'
CAL_SETTINGS = N.A40_CAL_SETTINGS
EVAL_SETTINGS = N.A40_EVAL_SETTINGS
SETTINGS = CAL_SETTINGS+EVAL_SETTINGS
ARMS = ('U0', 'LEARNED_TRIGGER', 'DUO_HOLD')
BANK_CELL = T.BANK_CELL
IDENT_SETTING = 'a35_D2400_p20'          # identity runs in a wz3to2 process (sealed p20c worlds; a35 code path)
IDENT_U0, IDENT_LT = 'A40_IDENT_U0', 'A40_IDENT_LT'
#: planned sim-s; caps = planned + 10% (MAIN may override with --cap)
CAL_PLANNED_SIM_S = len(CAL_SETTINGS)*(N.A40_CAL_SEEDS[1]-N.A40_CAL_SEEDS[0]+1)*N.HORIZON_S   # 3 x 8 x 1200 = 28,800
CAL_CAP_SIM_S = 31_680
EVAL_PLANNED_SIM_S = len(ARMS)*96*N.HORIZON_S + 2*N.IDENTITY_WORLDS*N.HORIZON_S         # 345,600 + 4,800 = 350,400
EVAL_CAP_SIM_S = 385_440
STOP_A40 = 'STOP_A40'
CONTRASTS = (('LEARNED_TRIGGER', 'U0'), ('DUO_HOLD', 'U0'), ('LEARNED_TRIGGER', 'DUO_HOLD'))
AUTHORITY = ('MAIN 2026-11-10 AMENDMENT 40 (geometry variant wz3to2v80: work zone 22.22 m/s): frozen U0, '
             'LEARNED_TRIGGER (fd385a2b + 8e0ddd3e, 0.378785), DUO_HOLD (duo12_v20), bank_cell_key D2800_p20; U0 '
             'demand calibration (phase geomvar_a40_cal) then one fresh 96-world panel (phase geomvar_a40_eval)')


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
    """Apply a wz3to2v80 setting in a wz3to2v80 process and load its (registered) panel."""
    if setting not in SETTINGS:
        raise N.Blocked(f'setting is one of {SETTINGS}')
    require_map(MAP)
    N.apply_setting(setting)
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    return load_setting_panel(setting)


def load_setting_panel(name):
    """The panel of setting ``name`` validated under the APPLIED setting (calibration settings share every validated
    key; only CELL / PANEL_FILE / texts differ); pinned by SETTINGS_PINS when MAIN pinned it."""
    s = N.SETTINGS[name]
    for k in ('MAP', 'NET_SHA256', 'EVAL_RANGE', 'TRAIN_RANGE', 'N_WORLDS', 'PANEL_TRAIN_TYPE'):
        if getattr(N, k) != s[k]:
            raise N.Blocked(f'{name}: {k} differs from the applied setting {N.SETTING}')
    path = Path(s['PANEL_FILE'])
    if not path.is_file():
        raise N.Blocked(f'{name}: panel {path.name} not registered (python ops/nm_finaleval.py register-panel '
                        f'--setting {name} --panel {path.name}, in a {MAP} process)')
    panel = N.load_panel(path, pinned=N.SETTINGS_PINS.get(name, {}).get('PANEL_SHA256'))
    want = s['CELL']
    if (panel['cell']['demand_vph']['main'], panel['cell']['penetration']) != (want['demand'], want['penetration']):
        raise N.Blocked(f'{name}: panel cell {panel["cell"]["cell_key"]} is not the setting cell')
    return panel


def cell_tag(panel):
    return panel['cell']['cell_key']


class A40Runtime(N.PrerunRuntime):
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
            raise N.Blocked(f'A40: {self.phase} cap {self.cap} would be exceeded')

    def reserve(self, cfg, jobs):
        _, other = self._spent()
        N.own_reserve(self.final_root, jobs, self.phase, cap=self.cap, calls=self.calls, allocation=self.cap+other)


def _arm_specs(arm, panel, final_root, *, check_files=True):
    """Exactly the amendment-35 arm specs (T._arm_specs): U0; LEARNED_TRIGGER / DUO_HOLD with bank_cell_key."""
    if arm not in ARMS:
        raise N.Blocked(f'no A40 arm {arm!r}')
    return T._arm_specs(arm, panel, final_root, check_files=check_files)


def arm_specs(arm, panel, final_root, *, prefix='a40', check_files=True):
    """The arm's specs with job ids prefixed '<prefix>_<cell>_' (one root for every cell, no clash)."""
    tag = f'{prefix}_{cell_tag(panel).lower()}_'
    return [dict(s, job_id=tag+s['job_id']) for s in _arm_specs(arm, panel, final_root, check_files=check_files)]


def wave_name(arm, panel, prefix='A40'):
    return f'{prefix}_{cell_tag(panel)}_{arm}'


def cal_waves(final_root):
    """(name, kind, specs) of the three calibration blocks (U0 only); needs an applied calibration setting."""
    return [(wave_name('U0', load_setting_panel(n), 'A40CAL'), 'episode',
             arm_specs('U0', load_setting_panel(n), final_root, prefix='a40cal')) for n in CAL_SETTINGS]


def ident_specs(final_root):
    """Exactly the amendment-35 identity specs (U0 + LEARNED_TRIGGER with bank_cell_key on 993100/993101)."""
    u, l_ = T.ident_specs(final_root)
    return ([dict(s, job_id=s['job_id'].replace('a35ident_', 'a40ident_', 1)) for s in u],
            [dict(s, job_id=s['job_id'].replace('a35ident_', 'a40ident_', 1)) for s in l_])


def check_a40(record, spec):
    return T.check_a35(record, spec)


N.CHECKS['episode'] = check_a40


def execute(final_root, waves, *, phase, cap, workers, chunk, log, panels=(), identity=False):
    final_root = Path(final_root)
    runtime = A40Runtime(N.Runtime(N.RUNNER), final_root, phase, cap)
    plant = final_root/N.PRERUN_PLANT
    cfg = runtime.config(plant)
    if not os.environ.get('COOPMWM_MAP_SPEC'):
        raise N.Blocked('set COOPMWM_MAP_SPEC before physics')
    status = runtime.runner_status()
    if not status['ok'] or not status.get('learned'):
        raise N.Blocked(f'runner cannot run the A40 arms on {N.MAP}: {status}')
    from tools.coopmwm import nm_eval_runner as R
    if getattr(R, 'A35_BANK_CELL', None) != BANK_CELL:
        raise N.Blocked('runner lacks the amendment-35 bank_cell_key path')
    if not identity and getattr(R, 'RANGES_BY_MAP', {}).get(MAP) != R.A40_EVAL_RANGES:
        raise N.Blocked('runner lacks the amendment-40 calibration range')
    for panel in panels:
        log(dict(map_check={k: v for k, v in N.map_check(N.bind_map_api(None), panel).items() if k != 'module'},
                 cell=cell_tag(panel)))
    plan = N.prerun_plan(final_root)
    u0, _ = A.bindings(final_root)
    if (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
        raise N.Blocked('the U0 binding residual is not the pre-run u0 copy')

    def coverage():
        return {name: (N.published(final_root, name, specs, check_a40)[2], len(specs)) for name, _, specs in waves}
    rounds = 0
    while True:
        cov = coverage()
        pending = [w for w in waves if cov[w[0]][0] < cov[w[0]][1]]
        if not pending:
            break
        if (final_root/STOP_A40).exists():
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
    """J / safety / vehicle-log sha of the A40 identity waves vs the sealed amendment-28 records (first 2 worlds)."""
    bank_root = Path(bank_root)
    u_specs, l_specs = ident_specs(final_root)
    rows = []
    for wave, sealed_prefix, specs in ((IDENT_U0, 'WAVE_U0__', u_specs), (IDENT_LT, 'WAVE_LEARNED_TRIGGER__', l_specs)):
        first = sorted(bank_root.glob(sealed_prefix+'0000-*.json'))
        if len(first) != 1:
            raise N.Blocked(f'{bank_root}: no unique {sealed_prefix}0000-* wave file')
        value = N.verify(N.read(first[0]), 'receipt_sha256', first[0].name)
        ref = {r['world_id']: r for r in value['records'][:N.IDENTITY_WORLDS]}
        got, _, cov = N.published(final_root, wave, specs, check_a40)
        if cov != len(specs):
            raise N.Blocked(f'{wave}: {cov}/{len(specs)} published')
        for r in got:
            s = ref[r['world_id']]
            rows.append(dict(wave=wave, world_id=r['world_id'], j=r['J_veh_s'], j_sealed=s['J_veh_s'],
                             abs_diff=abs(r['J_veh_s']-s['J_veh_s']), safety_equal=r['safety'] == s['safety'],
                             vehicle_log_equal=r['vehicle_log']['sha256'] == s['vehicle_log']['sha256']))
    ok = len(rows) == 2*N.IDENTITY_WORLDS and all(x['abs_diff'] <= 1e-6 and x['safety_equal'] for x in rows)
    return dict(verdict='PASS' if ok else 'FAILED', worlds=rows, rule='|J diff| <= 1e-6 veh-s and identical safety')


# ------------------------------------------------------------------------------------------------ analysis

def _records(root, panel, arm, prefix='a40', wave_prefix='A40'):
    specs = arm_specs(arm, panel, root, prefix=prefix, check_files=False)
    recs, src, cov = N.published(root, wave_name(arm, panel, wave_prefix), specs, check_a40)
    return recs, src, cov, len(specs)


def activations(recs):
    rec = [r.get('a32') or r.get('learned_trigger') or {} for r in recs]
    if not any(rec):
        return None
    fam = {}
    for r in rec:
        for f, n in (r.get('activations_by_family') or {}).items():
            fam[f] = fam.get(f, 0)+int(n)
    return dict(worlds_with_any=sum(1 for r in rec if r.get('activations')),
                total=sum(int(r.get('activations') or 0) for r in rec), by_family=fam,
                scores=sum(int(r.get('scores') or 0) for r in rec),
                fires=sum(int(r.get('fires') or 0) for r in rec if r.get('fires') is not None))


def analyse(panel, root, arms=ARMS):
    """Registered contrasts: paired percentile bootstrap (10,000, seed 20261104), diff = treatment - reference
    (negative = less delay); nm_finaleval._bank_compare = total delay, the five fairness groups (10 s rule, U0
    labels), CVaR90 on common vehicles; hard brakes / TTC events; safety totals; activation receipts."""
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
        for k in ('hard_brakes', 'ttc_events', 'collisions', 'teleports'):
            row[f'{k}_minus_ref'] = N.fe.paired_stats(
                np.array([have[arm][w]['safety'][k] for w in worlds], float),
                np.array([have[ref][w]['safety'][k] for w in worlds], float), bootstrap)
        contrasts.append(row)
    primary = next((c for c in contrasts if (c['treatment'], c['reference']) == ('LEARNED_TRIGGER', 'U0')), None)
    table = {}
    for a in have:
        js = [have[a][w]['J_veh_s'] for w in worlds]
        table[a] = dict(mean_J_veh_s=float(np.mean(js)), median_J_veh_s=float(np.median(js)),
                        sd_J_veh_s=float(np.std(js, ddof=1)) if len(js) > 1 else None,
                        safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY},
                        activations=activations([have[a][w] for w in worlds]))
    verdict = None
    if primary is not None and isinstance(primary.get('total_minus_ref'), dict):
        verdict = 'SUPPORTED' if primary['total_minus_ref']['ci95'][1] < 0 else 'NOT_SUPPORTED'
    return N.seal(dict(schema='coopmwm.a40_geomvar.analysis.v1', amendment='40', authority=AUTHORITY, map=N.MAP,
                       cell=panel['cell']['cell_key'], bank_cell_key=BANK_CELL, panel_sha256=panel['panel_sha256'],
                       n_worlds=len(worlds), bootstrap=bootstrap, arms=table, contrasts=contrasts,
                       primary=dict(contrast='LEARNED_TRIGGER - U0 total delay',
                                    rule='SUPPORTED iff the 95% CI upper bound < 0', verdict=verdict),
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()}, root=str(root),
                       label='amendment 40, geometry variant wz3to2v80; exploratory; one panel, one shot, never pooled'),
                  'analysis_sha256')


def approach_speed(log_path, drop_s=N.DROP_S_M):
    """Zero physics, from the ground-truth vehicle log: mean speed of every logged row upstream of the drop
    (s < drop_s) over the logged control window, and the number of such rows (identical to the A38 reducer)."""
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
    """Per calibration demand: U0 mean J, mean approach speed (vehicle logs), safety totals."""
    import numpy as np
    out = {}
    for name in CAL_SETTINGS:
        panel = load_setting_panel(name)
        recs, _, cov, n = _records(root, panel, 'U0', prefix='a40cal', wave_prefix='A40CAL')
        row = dict(setting=name, cell=cell_tag(panel), worlds_published=cov, worlds=n,
                   seeds=[panel['worlds'][0]['world']['seed'], panel['worlds'][-1]['world']['seed']])
        if recs:
            row['mean_J_veh_s'] = float(np.mean([r['J_veh_s'] for r in recs]))
            row['safety_totals'] = {k: int(sum(r['safety'][k] for r in recs)) for k in N.SAFETY}
            speeds = []
            for r in recs:
                d = Path(root)/'evaluation'/r.get('work_dir', r['job_id'])
                if (d/N.VEHICLE_LOG).is_file():
                    speeds.append(approach_speed(d/N.VEHICLE_LOG)[0])
            row['vehicle_logs'] = len(speeds)
            ok = [s for s in speeds if s is not None]
            row['mean_approach_speed_mps'] = float(np.mean(ok)) if ok and len(speeds) == n else None
        out[name] = row
    return N.seal(dict(schema='coopmwm.a40_geomvar.calsummary.v1', amendment='40', map=N.MAP, root=str(root),
                       cells=out, physics='zero (published records + vehicle logs only)'), 'summary_sha256')


#: registered demand pick rule (proposed; binding once MAIN registers it, BEFORE calibration runs)
V_GRID, V_CONG = 5.0, 25.0
PICK_ORDER = (2800, 2400, 3200)


def pick_demand(doc):
    """D2800 (the bank's source demand) if U0 there is congested and not gridlocked; else D2400, else D3200 under the
    same test; none -> None (chain stops, MAIN decides). gridlocked := mean approach speed < 5 m/s or any collision or
    teleport; congested := mean approach speed <= 25 m/s. Every block must be complete with vehicle logs."""
    rows = {int(r['cell'].split('_')[0][1:]): r for r in doc['cells'].values()}
    for d in PICK_ORDER:
        r = rows[d]
        v, st = r.get('mean_approach_speed_mps'), r.get('safety_totals') or {}
        if v is None or r.get('worlds_published') != r.get('worlds'):
            raise N.Blocked(f'D{d}: calibration incomplete or vehicle logs missing')
        grid = v < V_GRID or st.get('collisions', 0) > 0 or st.get('teleports', 0) > 0
        if v <= V_CONG and not grid:
            return f'a40_D{d}_p20'
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('dry', 'identity', 'identity-compare', 'calibrate', 'calsummary', 'pick', 'run',
                                       'reports', 'analyse'))
    p.add_argument('--final-root', required=True)
    p.add_argument('--setting', choices=SETTINGS)
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--chunk', type=int, default=48)
    p.add_argument('--cap', type=int, help='override the phase cap of this command (MAIN decision)')
    p.add_argument('--bank-root')
    p.add_argument('--summary')
    p.add_argument('--out')
    a = p.parse_args(argv)
    root = Path(a.final_root)
    arms = tuple(x for x in a.arms.split(',') if x)
    if not set(arms) <= set(ARMS) or 'U0' not in arms:
        p.error(f'arms are U0 + a subset of {ARMS[1:]}')
    if a.command == 'pick':
        s = pick_demand(json.loads(Path(a.summary).read_text(encoding='utf-8')))
        print(s or 'NO_CELL')
        sys.exit(0 if s else 3)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T')), sort_keys=True, default=str)
        print(line, flush=True)
        with (root/'A40_LOG.jsonl').open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    def locked(fn):
        import fcntl
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            return fn()
    if a.command in ('identity', 'identity-compare'):
        require_map(BASE_MAP)
        T.setup(IDENT_SETTING)
        if a.command == 'identity-compare':
            print(json.dumps(identity_compare(root, a.bank_root), indent=1))
            return
        u, l_ = ident_specs(root)
        waves = [(IDENT_U0, 'episode', u), (IDENT_LT, 'episode', l_)]

        def go():
            log(dict(start='identity', waves=[w[0] for w in waves], workers=a.workers, pid=os.getpid()))
            res = execute(root, waves, phase=EVAL_PHASE, cap=a.cap or EVAL_CAP_SIM_S, workers=a.workers,
                          chunk=a.chunk, log=log, identity=True)
            log(dict(done='identity', **res))
        return locked(go)
    if a.command in ('calibrate', 'calsummary'):
        setup(a.setting if a.setting in CAL_SETTINGS else CAL_SETTINGS[0])
        if a.command == 'calsummary':
            value = calsummary(root)
            out = Path(a.out)
            out.mkdir(parents=True, exist_ok=True)
            N.publish(out/'A40_CALSUMMARY.json', value, replace=True)
            print(json.dumps(value, indent=1))
            return
        waves = cal_waves(root)

        def go():
            log(dict(start='calibrate', waves=[w[0] for w in waves], workers=a.workers, chunk=a.chunk, pid=os.getpid()))
            res = execute(root, waves, phase=CAL_PHASE, cap=a.cap or CAL_CAP_SIM_S, workers=a.workers,
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
        from tools.coopmwm import a32_arms, learned_trigger
        out['a32_sha256'] = a32_arms.a32_digest()
        out['learned_trigger_sha256'] = learned_trigger.learned_trigger_digest()
        for arm in (('U0',) if is_cal else arms):
            specs = arm_specs(arm, panel, root, prefix='a40cal' if is_cal else 'a40')
            out[arm] = dict(n=len(specs), first=specs[0]['job_id'], cell=specs[0]['world']['cell_key'],
                            bank_cell_key=specs[0]['checkpoint'].get('bank_cell_key'),
                            seeds=[specs[0]['world']['seed'], specs[-1]['world']['seed']],
                            sim_s=sum(s['sim_s'] for s in specs))
        out['caps'] = {CAL_PHASE: CAL_CAP_SIM_S, EVAL_PHASE: EVAL_CAP_SIM_S}
        print(json.dumps(out, indent=1))
        return out
    if is_cal and a.command in ('run', 'analyse'):
        p.error('calibration settings run through `calibrate` / `calsummary`')
    if a.command == 'run':
        waves = [(wave_name(arm, panel), 'episode', arm_specs(arm, panel, root)) for arm in arms]

        def go():
            log(dict(start='run', setting=a.setting, waves=[w[0] for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid()))
            res = execute(root, waves, phase=EVAL_PHASE, cap=a.cap or EVAL_CAP_SIM_S, workers=a.workers,
                          chunk=a.chunk, log=log, panels=[panel])
            log(dict(done='run', **res))
        return locked(go)
    if a.command == 'analyse':
        value = analyse(panel, root, arms)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/f'A40_RESULTS_{cell_tag(panel)}.json', value, replace=True)
        print(json.dumps(dict(analysis_sha256=value['analysis_sha256'], cell=value['cell'],
                              primary=value['primary']['verdict'])))
        return value
    n = 0                                                    # reports
    for arm in (('U0',) if is_cal else arms):
        prefix, wprefix = ('a40cal', 'A40CAL') if is_cal else ('a40', 'A40')
        recs, _, _, _ = _records(root, panel, arm, prefix=prefix, wave_prefix=wprefix)
        for r in recs:
            rep = N.episode_log_report(root, r)
            if rep is None or not rep['ok']:
                raise N.Blocked(f'{r["job_id"]}: report failed')
            n += 1
    print(json.dumps(dict(reports=n)))


if __name__ == '__main__':
    main()
