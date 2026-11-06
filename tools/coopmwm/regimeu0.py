"""S-REGIMEU0 (phase regime_u0): S-REGIME rule strategies ON TOP OF the frozen MAPPO u0 base.

Question (MAIN 2026-11-06 ~19:20): S-REGIME (native base) found that one CAV holding in SUMO lane 2
(wall_h_solo2: 15 m/s, 90 s, dmax 1.5, whenever feasible, cooldown = hold) cuts average team delay
22.5% vs native and beats the 3-CAV wall. MAPPO u0 (our base and paper comparator) is ~21% better than
native already. Do these rules still help on top of u0?

Runtime = the coopmwm clean4 plant (accel_plant.episode, wz3to2, MAPPO u0 deterministic = the N1 sanity
'mappo' / clean3_u0 path) with the V2V controller wrapper (u0 widened by v2v_runtime.load_checkpoint with
zero columns: action-identical). Native warmup 0-300 s in every arm; rules act on 300-1200 s.

Rule port (S-REGIME rg_core.py / pg2_plan.py / pg2_core.Arm 'hold', verbatim parameters):
* every whole second (observe -> act -> step): active members first (release at hold end, at the drop
  x >= X_B = 2696 m (s >= 75.44), or when gone); then, if t >= next_ok, pick_wall on the CAVs not busy:
  one CAV per lane (SUMO lanes 0/1/2 = WM lanes 2/1/0) in the zone x in [X_B-1800, X_B-400]
  (s in [-1724.56, -324.56]), spread <= 200 m, key (spread, -max x); start at once; next_ok = t + 90;
* wall_h: all three members hold; wall_h_solo{L}: only the member in SUMO lane L holds (others native);
* hold law: a = clip(15 - v, -1.5, +1.0) requested as the absolute pre-safety acceleration (the plant's
  single writer; MAPPO safety downstream) and lane kept with laneChangeMode 513 through the G5
  single-writer lane path (target = own current WM lane, i.e. no changeLane is issued) = LC_MODE_POLICY.
HDVs are never commanded (members are is_cav rows only).

Arms (paired CRN: same scenario bytes, SUMO seed, noise seed): u0, u0+wall_h_solo2, u0+wall_h,
u0+wall_h_solo0. Plus one identity episode (u0 on the N1 sanity world 989599 D3200_p60) that must
reproduce the registered sanity MAPPO J 53,269 veh-s.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from . import imitreplay as ir

SCHEMA = 'coopmwm.regimeu0.v1'
STABLE_ID = 'S-REGIMEU0'
PHASE = 'regime_u0'
CAP = 100_000
X_B = 2696.0
S_OFF = 2620.56                       # s = x - S_OFF (N1 descriptor: drop x 2696 <-> s 75.44)
ZONE_S = (X_B-1800.0-S_OFF, X_B-400.0-S_OFF)
WALL_SPREAD = 200.0
RELEASE_S = X_B-S_OFF
WALL_H = dict(v_w=15.0, hold_s=90, dmax=1.5, amax=1.0)
STRATEGIES = {'wall_h': None, 'wall_h_solo2': 2, 'wall_h_solo0': 0, 'wall_h_solo1': 1}
ARMS = ('u0', 'wall_h_solo2', 'wall_h', 'wall_h_solo0')
SEEDS = tuple(range(989850, 989866))
CELL = (3200, 0.6)
IDENTITY = dict(seed=989599, demand=3200, penetration=0.6, expected_team_delay_vs=53269.32889342696)
EPISODE_S = 1200
WARMUP_S = 300
BOOT = 10_000
BOOT_SEED = 20261106


def sumo_lane(wm_lane):
    return 2-int(wm_lane)


def pick_wall(cavs):
    """pg2_plan.pick_wall on WM-frame rows: one CAV per lane in the zone, spread <= 200 m, key (spread, -max x)."""
    z = [c for c in cavs if ZONE_S[0] <= c['s_m'] <= ZONE_S[1] and int(c['lane']) in (0, 1, 2)]
    lanes = [[c for c in z if sumo_lane(c['lane']) == ln] for ln in (0, 1, 2)]
    if any(not x for x in lanes):
        return None
    best = None
    import itertools
    for combo in itertools.product(*lanes):
        xs = [c['s_m'] for c in combo]
        sp = max(xs)-min(xs)
        if sp <= WALL_SPREAD:
            key = (sp, -max(xs))
            if best is None or key < best[0]:
                best = (key, list(combo))
    return None if best is None else best[1]


class RuleHook(ir.Recorder):
    """Controller hook: the S-REGIME rule executed through the plant's single writer."""

    def __init__(self, strategy):
        super().__init__()
        if strategy is not None and strategy not in STRATEGIES:
            raise ValueError(strategy)
        self.strategy = strategy
        self.active = {}            # vid -> dict(start, act)
        self.acts, self.issued = [], []
        self.next_ok = -1
        self.lane_held = 0

    def __call__(self, controller, job, prefixes):
        self.controller = controller
        if self.strategy is None:
            return
        previous = controller.apply

        def apply(rows, commands, now):
            desired = previous(rows, commands, now)
            self.step(rows, commands, int(now), desired, controller)
            return desired

        controller.apply = apply

    def _law(self, row):
        v = float(row['v_mps'])
        return max(-WALL_H['dmax'], min(WALL_H['amax'], WALL_H['v_w']-v))

    def step(self, rows, commands, t, desired, controller):
        by = {r['vid']: r for r in rows if r.get('is_cav')}
        lanes = {}
        # 1) active members (release at hold end / drop / gone), then 2) a new group if allowed (pg2 order)
        for vid, m in list(self.active.items()):
            el = t-m['start']
            row = by.get(vid)
            why = ('left' if row is None or vid not in commands else 'hold_end' if el >= WALL_H['hold_s']
                   else 'at_drop' if float(row['s_m']) >= RELEASE_S else None)
            if why is not None:
                m['act']['release'][vid] = [el, why]
                self.active.pop(vid)
                continue
            a = self._law(row)
            desired[vid] = dict(requested_raw=a, acceleration_residual=a-commands[vid], regime_rule=self.strategy)
            lanes[vid] = int(row['lane'])
            self.issued.append([t, vid, round(a, 4)])
        if t >= self.next_ok:
            free = [r for r in by.values() if r['vid'] not in self.active and r['vid'] in commands]
            wall = pick_wall(free)
            if wall:
                solo = STRATEGIES[self.strategy]
                members = [c for c in wall if solo is None or sumo_lane(c['lane']) == solo]
                xs = [c['s_m'] for c in wall]
                act = dict(t=t, strategy=self.strategy, members=[c['vid'] for c in members],
                           wall=[c['vid'] for c in wall], sumo_lanes=[sumo_lane(c['lane']) for c in members],
                           s=[round(c['s_m'], 2) for c in members], v=[round(c['v_mps'], 3) for c in members],
                           span=round(max(xs)-min(xs), 2), release={})
                self.acts.append(act)
                self.next_ok = t+WALL_H['hold_s']
                for c in members:
                    self.active[c['vid']] = dict(start=t, act=act)
                    a = self._law(c)
                    desired[c['vid']] = dict(requested_raw=a, acceleration_residual=a-commands[c['vid']],
                                             regime_rule=self.strategy)
                    lanes[c['vid']] = int(c['lane'])
                    self.issued.append([t, c['vid'], round(a, 4)])
        if lanes:
            # G5 single-writer lane path: stay in the current WM lane under laneChangeMode 513.
            targets = dict(getattr(controller, 'g5_lane_targets', {}) or {})
            for vid, lane in lanes.items():
                targets.setdefault(vid, lane)
            controller.g5_lane_targets = targets
            self.lane_held += len(lanes)


# ============================================================================ plan / ledger

def world_of(seed, demand=CELL[0], penetration=CELL[1]):
    from . import map_scenarios as ms
    return ms.world(int(seed), demand, penetration)


def init(root, cap=CAP):
    root = ir.guard_path(root)
    root.mkdir(parents=True, exist_ok=True)
    reg = root/'REGISTRATION.json'
    if reg.exists():
        return ir._read(reg)
    value = dict(schema=SCHEMA, stable_id=STABLE_ID, phase=PHASE, cap_sim_s=int(cap), module_sha256=ir.sha_file(__file__),
                 seeds=list(SEEDS), arms=list(ARMS), identity=IDENTITY, unix=time.time())
    ir._write_new(reg, value)
    return value


def plan(root, *, config, u0, u0_sha256=None, seeds=SEEDS, arms=ARMS):
    import shutil
    root = ir.guard_path(root)
    reg = ir.registration(root)
    if (root/'PLAN.json').exists():
        raise ValueError('PLAN.json already frozen')
    (root/'inputs').mkdir(exist_ok=True)
    files = {}
    for src, name in ((config, 'CONFIG.json'), (u0, 'u0.pt')):
        shutil.copyfile(ir.guard_path(src), root/'inputs'/name)
        files[name] = dict(source=str(src), sha256=ir.sha_file(root/'inputs'/name))
    if u0_sha256 and files['u0.pt']['sha256'] != u0_sha256:
        raise ValueError('MAPPO u0 identity differs')
    groups = [dict(stage='identity', jobs=[dict(id=f'identity_s{IDENTITY["seed"]}_u0', arm='u0', strategy=None,
                                              world=world_of(IDENTITY['seed']), sim_s=EPISODE_S)])]
    for s in seeds:
        w = world_of(int(s))
        groups.append(dict(stage='paired', jobs=[dict(id=f's{s}_{a}', arm=a, strategy=None if a == 'u0' else a,
                                                      world=w, sim_s=EPISODE_S) for a in arms]))
    body = dict(schema=SCHEMA+'.plan', stable_id=STABLE_ID, phase=PHASE, cap_sim_s=reg['cap_sim_s'], inputs=files,
                groups=groups, planned_sim_s=sum(j['sim_s'] for g in groups for j in g['jobs']),
                module_sha256=ir.sha_file(__file__), unix=time.time())
    body['plan_sha256'] = ir.digest({k: v for k, v in body.items() if k != 'unix'})
    ir._write_new(root/'PLAN.json', body)
    return body


def _job(root, ident):
    body = ir._read(Path(root)/'PLAN.json')
    for g in body['groups']:
        for j in g['jobs']:
            if j['id'] == ident:
                return j
    raise KeyError(ident)


def worker(root, ident):
    root = Path(root)
    job = _job(root, ident)
    if not any(c['id'] == ident for c in ir.charges(root)):
        raise ValueError('job not precharged')
    work = root/'jobs'/ident
    work.mkdir(parents=True, exist_ok=False)
    started = time.time()
    ir._write_new(work/'CLAIM.json', dict(pid=os.getpid(), id=ident, sim_s=job['sim_s'], unix=started))
    from . import map_descriptor as md, clean4_verify as c4v
    if os.environ.get(md.ENV) is None or os.environ.get('COOPMWM_G5_RUNTIME') != '1':
        raise RuntimeError('COOPMWM_MAP_SPEC and COOPMWM_G5_RUNTIME=1 required')
    from .coop_warmstart import state_sha256
    from .v2v_runtime import load_checkpoint
    cfg = ir._read(root/'inputs'/'CONFIG.json')
    base_model = load_checkpoint(root/'inputs'/'u0.pt', config=cfg)[0]
    weights = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
    world = job['world']
    accel = c4v.job_fields(ident=ident, path=work/'plant', world=world, arm='A', weights=weights,
                           noise_seed=0, runtime=c4v.RUNTIME_DIR, checkpoint=c4v.CHECKPOINT, sim_s=job['sim_s'],
                           v2v=True, deterministic=True)
    accel.update(stop=EPISODE_S, queue_root=str(root/'jobs'))
    hook = RuleHook(job['strategy'])
    from tools.coopfree import backend as cfb
    from . import accel_plant
    original_tick = cfb.tick
    cfb.tick = hook.tick_wrapper(original_tick)
    result, error = None, None
    try:
        result = accel_plant.episode(accel, controller_hook=hook)
    except Exception as exc:
        import traceback
        error = dict(error=repr(exc), traceback=traceback.format_exc())
    finally:
        cfb.tick = original_tick
    lane_out = {}
    steps = (result or {}).get('steps') or (hook.controller.events if hook.controller is not None else [])
    for e in steps:
        o = e.get('g5_lane_outcome')
        if o:
            for k, v in (o.get('counts') or o).items() if isinstance(o, dict) else ():
                if isinstance(v, (int, float)):
                    lane_out[k] = lane_out.get(k, 0)+v
    summary = dict(schema=SCHEMA+'.summary', id=ident, arm=job['arm'], strategy=job['strategy'], world=world,
                   totals=(result or {}).get('totals'), actual_sim_s=(result or {}).get('actual_sim_s'),
                   series_keys=list(ir.SERIES_KEYS), series=hook.series, activations=hook.acts,
                   issued=len(hook.issued), lane_held_steps=hook.lane_held, lane_outcome=lane_out,
                   base_sha256=state_sha256(base_model), error=error, wall_s=round(time.time()-started, 2))
    with gzip.open(work/'SUMMARY.json.gz', 'xt', encoding='utf-8') as f:
        json.dump(summary, f, default=str)
    if error is not None:
        ir._write_new(work/'FAILED.json', dict(id=ident, **error))
    else:
        ir._write_new(work/'DONE.json', dict(id=ident, wall_s=summary['wall_s'], unix=time.time()))
    return summary


def run(root, *, workers=16, load_high=44.0, low_workers=8, poll_s=5.0, python=None):
    root = Path(root)
    body = ir._read(root/'PLAN.json')
    if os.name != 'nt' and (os.getpriority(os.PRIO_PROCESS, 0) != 19 or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise ValueError('run under nice 19 with CUDA_VISIBLE_DEVICES=')
    python = python or sys.executable
    log = (root/'DISPATCH.jsonl').open('a', encoding='utf-8')

    def note(**kw):
        log.write(json.dumps(dict(kw, unix=time.time()), sort_keys=True, default=str)+'\n')
        log.flush()

    running = {}
    charged = {c['id'] for c in ir.charges(root)}
    note(event='start', pid=os.getpid(), spent=ir.spent(root))
    for g in body['groups']:
        if (root/'STOP').exists():
            note(event='stop_file')
            break
        jobs = [j for j in g['jobs'] if j['id'] not in charged]
        if not jobs:
            continue
        if not ir.charge(root, [(j['id'], j['sim_s'], g['stage']) for j in jobs]):
            note(event='cap_reached', group=[j['id'] for j in jobs])
            break
        charged.update(j['id'] for j in jobs)
        for j in jobs:
            while True:
                ir._reap(running, note)
                try:
                    load = os.getloadavg()[0]
                except (AttributeError, OSError):
                    load = 0.
                limit = workers if load <= load_high else min(workers, low_workers)
                if len(running) < limit:
                    break
                time.sleep(poll_s)
            (root/'jobs'/'_logs').mkdir(parents=True, exist_ok=True)
            stream = (root/'jobs'/'_logs'/f'{j["id"]}.log').open('x')
            p = subprocess.Popen([python, '-B', '-m', 'tools.coopmwm.regimeu0', 'worker', '--root', str(root),
                                  '--job', j['id']], stdout=stream, stderr=subprocess.STDOUT)
            running[p.pid] = (p, j['id'], stream)
            note(event='launch', id=j['id'], pid=p.pid, load=load, limit=limit)
            with (root/'PIDS.jsonl').open('a') as f:
                f.write(json.dumps(dict(pid=p.pid, id=j['id'], unix=time.time()))+'\n')
    while running:
        ir._reap(running, note)
        time.sleep(poll_s)
    note(event='end', spent=ir.spent(root))
    return ir.spent(root)


# ============================================================================ analysis (zero physics)

def control_j(summary, a=WARMUP_S, b=EPISODE_S):
    r = ir.window_j(summary, a, b)
    return None if r is None else r[0]


def boot_ci(diffs, *, n=BOOT, seed=BOOT_SEED):
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k))/k for _ in range(n))
    return means[int(.025*n)], means[int(.975*n)-1]


def analyze(root):
    root = Path(root)
    body = ir._read(root/'PLAN.json')

    def load(ident):
        p = root/'jobs'/ident/'SUMMARY.json.gz'
        return ir._read(p) if p.exists() else None

    rep = dict(schema=SCHEMA+'.analysis', plan_sha256=body['plan_sha256'], spent=ir.spent(root), arms={}, seeds=[])
    ident = load(body['groups'][0]['jobs'][0]['id'])
    if ident is not None:
        tot = (ident.get('totals') or {}).get('team_delay_vs')
        rep['identity'] = dict(team_delay_vs=tot, expected=IDENTITY['expected_team_delay_vs'],
                               equal=tot is not None and abs(tot-IDENTITY['expected_team_delay_vs']) < 1e-6,
                               error=(ident.get('error') or {}).get('error'))
    per = {}
    for g in body['groups']:
        if g['stage'] != 'paired':
            continue
        sums = {j['arm']: load(j['id']) for j in g['jobs']}
        seed = g['jobs'][0]['world']['seed']
        base = sums.get('u0')
        row = dict(seed=seed, arms={})
        for arm, s in sums.items():
            if s is None:
                row['arms'][arm] = dict(status='not_run')
                continue
            if s.get('error'):
                row['arms'][arm] = dict(status='failed', error=s['error']['error'][:200])
                continue
            j = control_j(s)
            safety = ir.safety(s)
            rec = dict(status='ok', J=j, totals=s.get('totals'), safety=safety,
                       activations=len(s.get('activations') or ()), lane_held_steps=s.get('lane_held_steps'),
                       lane_outcome=s.get('lane_outcome'))
            if arm != 'u0' and base is not None and not base.get('error'):
                rec['diff'] = j-control_j(base)
                rec['first_divergence_s'] = ir.first_divergence(base, s)
                first_act = min((a['t'] for a in s.get('activations') or ()), default=None)
                rec['first_activation_s'] = first_act
                rec['crn_ok'] = rec['first_divergence_s'] is None or (first_act is not None
                                                                       and rec['first_divergence_s'] >= first_act)
            row['arms'][arm] = rec
            per.setdefault(arm, []).append(rec)
        rep['seeds'].append(row)
    for arm in ARMS:
        recs = [r for r in per.get(arm, ()) if r.get('status') == 'ok']
        out = dict(n=len(recs), mean_J=(sum(r['J'] for r in recs)/len(recs)) if recs else None)
        if arm != 'u0':
            diffs = [r['diff'] for r in recs if r.get('diff') is not None]
            if diffs:
                lo, hi = boot_ci(diffs)
                out.update(n_paired=len(diffs), mean_diff=sum(diffs)/len(diffs), ci95=[lo, hi],
                           harm_rate=sum(d > 0 for d in diffs)/len(diffs),
                           rel_diff=(sum(diffs)/len(diffs))/out['mean_J'] if False else None,
                           crn_ok=sum(bool(r.get('crn_ok')) for r in recs), mean_activations=sum(
                               r['activations'] for r in recs)/len(recs))
        for k in ('collisions', 'teleports', 'ttc_below_3_vs', 'hard_brake'):
            vals = [r['safety'].get(k) for r in recs if r['safety'].get(k) is not None]
            out[k] = sum(vals) if vals else None
        rep['arms'][arm] = out
    base_mean = rep['arms'].get('u0', {}).get('mean_J')
    for arm, out in rep['arms'].items():
        if arm != 'u0' and out.get('mean_diff') is not None and base_mean:
            out['rel_diff'] = out['mean_diff']/base_mean
    # group vs best solo strategy (best solo re-chosen inside every resample, S-REGIME rule)
    ok = [r for r in rep['seeds'] if all(r['arms'].get(a, {}).get('status') == 'ok' for a in ARMS)]
    if ok:
        rng = random.Random(BOOT_SEED)
        solos = [a for a in ARMS if 'solo' in a]

        def stat(rows):
            mean = {a: sum(r['arms'][a]['J'] for r in rows)/len(rows) for a in ('wall_h', *solos)}
            return mean['wall_h']-min(mean[a] for a in solos)
        point = stat(ok)
        draws = sorted(stat([ok[rng.randrange(len(ok))] for _ in ok]) for _ in range(BOOT))
        rep['group_minus_best_solo'] = dict(point=point, ci95=[draws[int(.025*BOOT)], draws[int(.975*BOOT)-1]],
                                            n=len(ok), solos=solos)
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('init')
    a.add_argument('--root', required=True)
    a.add_argument('--cap', type=int, default=CAP)
    a = sub.add_parser('plan')
    a.add_argument('--root', required=True)
    a.add_argument('--config', required=True)
    a.add_argument('--u0', required=True)
    a.add_argument('--u0-sha256')
    a = sub.add_parser('run')
    a.add_argument('--root', required=True)
    a.add_argument('--workers', type=int, default=16)
    a.add_argument('--low-workers', type=int, default=8)
    a.add_argument('--load-high', type=float, default=44.0)
    a = sub.add_parser('worker')
    a.add_argument('--root', required=True)
    a.add_argument('--job', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--root', required=True)
    a.add_argument('--out')
    args = p.parse_args(argv)
    if args.cmd == 'init':
        print(json.dumps(init(args.root, args.cap), indent=1))
    elif args.cmd == 'plan':
        body = plan(args.root, config=args.config, u0=args.u0, u0_sha256=args.u0_sha256)
        print(json.dumps(dict(plan_sha256=body['plan_sha256'], planned_sim_s=body['planned_sim_s']), indent=1))
    elif args.cmd == 'run':
        print(run(args.root, workers=args.workers, load_high=args.load_high, low_workers=args.low_workers))
    elif args.cmd == 'worker':
        s = worker(args.root, args.job)
        print(json.dumps(dict(id=s['id'], error=bool(s['error']), totals=s['totals'], acts=len(s['activations']))))
    elif args.cmd == 'analyze':
        rep = analyze(args.root)
        text = json.dumps(rep, indent=1, sort_keys=True, default=str)
        if args.out:
            Path(args.out).write_text(text+'\n', encoding='utf-8')
        print(json.dumps(dict(identity=rep.get('identity'), arms=rep['arms'],
                              group_minus_best_solo=rep.get('group_minus_best_solo')), indent=1, default=str))


if __name__ == '__main__':
    main()
