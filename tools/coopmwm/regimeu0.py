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
WALL_G = dict(v_w=22.0, hold_s=120, dmax=1.0, amax=1.0)          # pg2_plan VARIANTS['wall_g'] (S-LOWSHARE-U0)
WALL_M = dict(v_w=18.0, hold_s=90, dmax=1.0, amax=1.0)           # pg2_plan R_GRID point (v_w 18, hold 90, dmax 1.0) (S-MULTICRIT)
VARIANT_PARAMS = dict(wall_h=WALL_H, wall_g=WALL_G, wall_m=WALL_M)
# strategy -> (variant, solo SUMO lane or None, or a tuple of SUMO lanes = partial wall)
STRATEGIES = {f'{w}{suffix}': (w, solo) for w in ('wall_h', 'wall_g')
              for suffix, solo in (('', None), ('_solo0', 0), ('_solo1', 1), ('_solo2', 2))}
# S-MULTICRIT (MAIN 2026-11-06 ~23:35, fair/light-traffic candidates; registered parameters only)
STRATEGIES.update({'wall_m_solo2': ('wall_m', 2),            # lane-2 member holds 18 m/s / 90 s / dmax 1.0
                   'wall_g_duo12': ('wall_g', (1, 2)),       # lane-1 + lane-2 members hold 22 m/s / 120 s (shared partial wall)
                   'wall_g_solo2_gate0': ('wall_g', 2)})     # wall_g_solo2, started only if its lane is empty 100 m behind
#: start gate: a new group starts only if, for every holding member, at most max_behind observed vehicles
#: (CAV reports + RSU detections, i.e. the rows the controller sees) are in its WM lane within behind_m behind it.
GATES = {'wall_g_solo2_gate0': dict(behind_m=100.0, max_behind=0)}
# S-MULTICRIT-R2 (MAIN 2026-11-07 23:40; thresholds fixed before the run): situation trigger, no wall pick.
STRATEGIES['r2_solo2'] = ('wall_g', 2)
R2 = dict(drop_window_m=600.0, drop_speed_max=15.0, ahead_m=500.0, lane_gap_min=5.0)
S_DROP = RELEASE_S


def _mean(x):
    return sum(x)/len(x) if x else None


def r2_situation(rows, cand):
    """(a) mean observed speed over the last 600 m before the drop < 15 m/s; (b) observed WM-lane-0 (SUMO 2) mean
    speed over [s, s+500] exceeds WM-lane-1 (SUMO 1) by > 5 m/s. Rows = the controller's rows (CAV reports + RSU)."""
    up = [r for r in rows if float(r['s_m']) < S_DROP]
    ds = _mean([float(r['v_mps']) for r in up if float(r['s_m']) >= S_DROP-R2['drop_window_m']])
    s = float(cand['s_m'])
    ahead = [r for r in up if s <= float(r['s_m']) <= s+R2['ahead_m']]
    v0 = _mean([float(r['v_mps']) for r in ahead if int(r['lane']) == 0])
    v1 = _mean([float(r['v_mps']) for r in ahead if int(r['lane']) == 1])
    ok = ds is not None and ds < R2['drop_speed_max'] and v0 is not None and v1 is not None and v0-v1 > R2['lane_gap_min']
    return ok, dict(drop_speed=ds, v_l2_ahead=v0, v_l1_ahead=v1)


def pick_r2(cavs, rows):
    """Free CAVs in SUMO lane 2 (WM lane 0) inside the zone that satisfy R2; the most downstream one holds."""
    c = [r for r in cavs if int(r['lane']) == 0 and ZONE_S[0] <= float(r['s_m']) <= ZONE_S[1]]
    ok = []
    for r in c:
        good, info = r2_situation(rows, r)
        if good:
            ok.append((float(r['s_m']), r, info))
    if not ok:
        return None, None
    ok.sort(key=lambda x: -x[0])
    return ok[0][1], ok[0][2]
GROUP_ARMS = ('wall_h', 'wall_g')
ARMS = ('u0', 'wall_h_solo2', 'wall_h', 'wall_h_solo0')
ALL_ARMS = ('u0', 'wall_h_solo2', 'wall_h', 'wall_h_solo0', 'wall_g', 'wall_g_solo2', 'wall_g_solo0',
            'wall_m_solo2', 'wall_g_duo12', 'wall_g_solo2_gate0', 'r2_solo2')


def solo_lanes(solo):
    """None (all wall members hold) or the tuple of SUMO lanes whose wall member holds."""
    return None if solo is None else ((solo,) if isinstance(solo, int) else tuple(solo))


def behind_count(rows, member, behind_m):
    """Observed vehicles in the member's WM lane within behind_m behind it (excluding itself)."""
    s, lane = float(member['s_m']), int(member['lane'])
    return sum(1 for r in rows if r['vid'] != member['vid'] and int(r['lane']) == lane
               and s-behind_m <= float(r['s_m']) < s)
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
        self.params = VARIANT_PARAMS[STRATEGIES[strategy][0]] if strategy is not None else WALL_H
        self.active = {}            # vid -> dict(start, act)
        self.acts, self.issued = [], []
        self.next_ok = -1
        self.lane_held = 0
        self.gate_refusals = 0

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
        p = self.params
        return max(-p['dmax'], min(p['amax'], p['v_w']-v))

    def step(self, rows, commands, t, desired, controller):
        by = {r['vid']: r for r in rows if r.get('is_cav')}
        lanes = {}
        # 1) active members (release at hold end / drop / gone), then 2) a new group if allowed (pg2 order)
        for vid, m in list(self.active.items()):
            el = t-m['start']
            row = by.get(vid)
            why = ('left' if row is None or vid not in commands else 'hold_end' if el >= self.params['hold_s']
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
            r2info = None
            if self.strategy == 'r2_solo2':
                cand, r2info = pick_r2(free, rows)
                wall = [cand] if cand is not None else None
                members = list(wall) if wall else []
            else:
                wall = pick_wall(free)
            if wall and self.strategy != 'r2_solo2':
                lanes_sel = solo_lanes(STRATEGIES[self.strategy][1])
                members = [c for c in wall if lanes_sel is None or sumo_lane(c['lane']) in lanes_sel]
                gate = GATES.get(self.strategy)
                if gate is not None and any(behind_count(rows, c, gate['behind_m']) > gate['max_behind'] for c in members):
                    self.gate_refusals += 1
                    wall = None
            if wall:
                xs = [c['s_m'] for c in wall]
                act = dict(t=t, strategy=self.strategy, members=[c['vid'] for c in members],
                           wall=[c['vid'] for c in wall], sumo_lanes=[sumo_lane(c['lane']) for c in members],
                           s=[round(c['s_m'], 2) for c in members], v=[round(c['v_mps'], 3) for c in members],
                           span=round(max(xs)-min(xs), 2), release={})
                if r2info is not None:
                    act['r2'] = {k: (None if v is None else round(v, 3)) for k, v in r2info.items()}
                self.acts.append(act)
                self.next_ok = t+self.params['hold_s']
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


def plan(root, *, config, u0, u0_sha256=None, cells=((CELL[0], CELL[1], SEEDS),), arms=ARMS, identity=True):
    """cells: [(demand, penetration, seeds)] in run order (seed-major inside a cell)."""
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
    groups = []
    if identity:
        groups.append(dict(stage='identity', jobs=[dict(id=f'identity_s{IDENTITY["seed"]}_u0', arm='u0', strategy=None,
                                                        world=world_of(IDENTITY['seed']), sim_s=EPISODE_S)]))
    for demand, pen, seeds in cells:
        for s in seeds:
            w = world_of(int(s), int(demand), float(pen))
            prefix = '' if (int(demand), float(pen)) == CELL else f'{w["cell_key"]}_'
            groups.append(dict(stage='paired', cell=w['cell_key'], jobs=[
                dict(id=f'{prefix}s{s}_{a}', arm=a, strategy=None if a == 'u0' else a, world=w, sim_s=EPISODE_S)
                for a in arms]))
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
    truth = None
    if os.environ.get('MULTICRIT_TRUTH_LOG') == '1':
        # S-MULTICRIT-R2: read-only TraCI getters after each step (evaluation-only per-vehicle truth: id, lane id,
        # lane position, speed; never vType / driver parameters).
        truth = gzip.open(work/'truth.jsonl.gz', 'xt', encoding='utf-8')
        inner = cfb.tick

        def tick(b, controller=None, trace=None):
            audit = inner(b, controller, trace)
            v = b.traci.vehicle
            truth.write(json.dumps(dict(t=round(float(b.env.now), 3), v=[[i, v.getLaneID(i), round(v.getLanePosition(i), 2),
                                                                          round(v.getSpeed(i), 4)] for i in v.getIDList()]),
                                   separators=(',', ':'))+'\n')
            return audit
        cfb.tick = tick
    result, error = None, None
    try:
        result = accel_plant.episode(accel, controller_hook=hook)
    except Exception as exc:
        import traceback
        error = dict(error=repr(exc), traceback=traceback.format_exc())
    finally:
        cfb.tick = original_tick
        if truth is not None:
            truth.close()
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
                   gate_refusals=hook.gate_refusals,
                   base_sha256=state_sha256(base_model), error=error, wall_s=round(time.time()-started, 2))
    with gzip.open(work/'SUMMARY.json.gz', 'xt', encoding='utf-8') as f:
        json.dump(summary, f, default=str)
    if error is not None:
        ir._write_new(work/'FAILED.json', dict(id=ident, **error))
    else:
        ir._write_new(work/'DONE.json', dict(id=ident, wall_s=summary['wall_s'], unix=time.time()))
    return summary


def run(root, *, workers=16, load_high=44.0, low_workers=8, poll_s=5.0, python=None, normal_priority=False):
    root = Path(root)
    body = ir._read(root/'PLAN.json')
    if os.name != 'nt' and ((not normal_priority and os.getpriority(os.PRIO_PROCESS, 0) != 19)
                            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise ValueError('run under nice 19 (unless MAIN allowed normal priority) with CUDA_VISIBLE_DEVICES=')
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


def analyze(roots):
    """Merge the paired groups of one or more roots (any host copy); statistics per cell."""
    roots = [Path(r) for r in ([roots] if isinstance(roots, (str, Path)) else roots)]
    rep = dict(schema=SCHEMA+'.analysis', roots=[str(r) for r in roots], spent=sum(ir.spent(r) for r in roots),
               plan_sha256=[ir._read(r/'PLAN.json')['plan_sha256'] for r in roots], cells={}, seeds=[])
    for root in roots:
        body = ir._read(root/'PLAN.json')

        def load(ident):
            p = root/'jobs'/ident/'SUMMARY.json.gz'
            return ir._read(p) if p.exists() else None

        for g in body['groups']:
            if g['stage'] == 'identity':
                s = load(g['jobs'][0]['id'])
                if s is not None:
                    tot = (s.get('totals') or {}).get('team_delay_vs')
                    rep['identity'] = dict(team_delay_vs=tot, expected=IDENTITY['expected_team_delay_vs'],
                                           equal=tot is not None and abs(tot-IDENTITY['expected_team_delay_vs']) < 1e-6,
                                           error=(s.get('error') or {}).get('error'))
                continue
            sums = {j['arm']: load(j['id']) for j in g['jobs']}
            w = g['jobs'][0]['world']
            base = sums.get('u0')
            row = dict(seed=w['seed'], cell=w['cell_key'], root=str(root), arms={})
            for arm, s in sums.items():
                if s is None:
                    row['arms'][arm] = dict(status='not_run')
                    continue
                if s.get('error'):
                    row['arms'][arm] = dict(status='failed', error=s['error']['error'][:200])
                    continue
                rec = dict(status='ok', J=control_j(s), totals=s.get('totals'), safety=ir.safety(s),
                           activations=len(s.get('activations') or ()), lane_held_steps=s.get('lane_held_steps'))
                if arm != 'u0' and base is not None and not base.get('error'):
                    rec['diff'] = rec['J']-control_j(base)
                    rec['first_divergence_s'] = ir.first_divergence(base, s)
                    first_act = min((a['t'] for a in s.get('activations') or ()), default=None)
                    rec['first_activation_s'] = first_act
                    rec['crn_ok'] = rec['first_divergence_s'] is None or (
                        first_act is not None and rec['first_divergence_s'] >= first_act)
                row['arms'][arm] = rec
            rep['seeds'].append(row)
    for cell in sorted({r['cell'] for r in rep['seeds']}):
        rows = [r for r in rep['seeds'] if r['cell'] == cell]
        out = {}
        for arm in [x for x in ALL_ARMS if any(x in r['arms'] for r in rows)]:
            recs = [r['arms'][arm] for r in rows if r['arms'].get(arm, {}).get('status') == 'ok']
            a = dict(n=len(recs), mean_J=(sum(x['J'] for x in recs)/len(recs)) if recs else None,
                     failed=sum(r['arms'].get(arm, {}).get('status') == 'failed' for r in rows))
            diffs = [x['diff'] for x in recs if x.get('diff') is not None]
            if arm != 'u0' and diffs:
                lo, hi = boot_ci(diffs)
                a.update(n_paired=len(diffs), mean_diff=sum(diffs)/len(diffs), ci95=[lo, hi],
                         benefit_rate=sum(d < 0 for d in diffs)/len(diffs), harm_rate=sum(d > 0 for d in diffs)/len(diffs),
                         crn_ok=sum(bool(x.get('crn_ok')) for x in recs),
                         mean_activations=sum(x['activations'] for x in recs)/len(recs))
            for k in ('collisions', 'teleports', 'ttc_below_3_vs', 'hard_brake'):
                vals = [x['safety'].get(k) for x in recs if x['safety'].get(k) is not None]
                a[k] = sum(vals) if vals else None
                a[k+'_mean'] = (sum(vals)/len(vals)) if vals else None
            out[arm] = a
        base_mean = out.get('u0', {}).get('mean_J')
        for arm, a in out.items():
            if arm != 'u0' and a.get('mean_diff') is not None and base_mean:
                a['rel_diff'] = a['mean_diff']/base_mean
                a['rel_ci95'] = [a['ci95'][0]/base_mean, a['ci95'][1]/base_mean]
        present = [x for x in ALL_ARMS if any(x in r['arms'] for r in rows)]
        ok = [r for r in rows if all(r['arms'].get(x, {}).get('status') == 'ok' for x in present)]
        solos = [x for x in present if 'solo' in x]
        for group in [g for g in GROUP_ARMS if g in present]:
            if not ok or not solos:
                continue
            rng = random.Random(BOOT_SEED)

            def stat(rs, group=group):
                mean = {x: sum(r['arms'][x]['J'] for r in rs)/len(rs) for x in (group, *solos)}
                return mean[group]-min(mean[x] for x in solos)
            draws = sorted(stat([ok[rng.randrange(len(ok))] for _ in ok]) for _ in range(BOOT))
            beats = [r['arms'][group]['J'] < min(r['arms'][x]['J'] for x in solos) for r in ok]
            key = 'group_minus_best_solo' if group == 'wall_h' else f'{group}_minus_best_solo'
            out[key] = dict(point=stat(ok), ci95=[draws[int(.025*BOOT)], draws[int(.975*BOOT)-1]], n=len(ok),
                            solos=solos, group=group, beats_all_solos_fraction=sum(beats)/len(beats))
        rep['cells'][cell] = out
    return rep


WINDOW_STARTS = tuple(range(WARMUP_S, EPISODE_S, 100))


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    pos = q*(len(sorted_vals)-1)
    lo = int(pos)
    hi = min(lo+1, len(sorted_vals)-1)
    return sorted_vals[lo]+(sorted_vals[hi]-sorted_vals[lo])*(pos-lo)


def _seed_boot(values_by_seed, *, n=BOOT, seed=BOOT_SEED):
    """values_by_seed: {seed: [window values]} -> (mean, lo, hi) of the pooled window mean, seeds resampled."""
    seeds = [s for s, v in values_by_seed.items() if v]
    if not seeds:
        return None, None, None
    flat = [x for s in seeds for x in values_by_seed[s]]
    point = sum(flat)/len(flat)
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        pick = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        vals = [x for s in pick for x in values_by_seed[s]]
        draws.append(sum(vals)/len(vals))
    draws.sort()
    return point, draws[int(.025*n)], draws[int(.975*n)-1]


def amendment4(roots):
    """Registered S-REGIMEU0 amendment 4 (zero SUMO): per-seed distributions, window-level conditional effect
    on the observable closing-lane congestion proxy (u0 arm, window start), split-seed gated estimate."""
    roots = [Path(r) for r in ([roots] if isinstance(roots, (str, Path)) else roots)]
    windows, proxies = {}, {}
    for root in roots:
        body = ir._read(root/'PLAN.json')
        prox = ir._read(root/'PROXIES.json')['proxies'] if (root/'PROXIES.json').exists() else {}
        for g in body['groups']:
            if g['stage'] != 'paired':
                continue
            w = g['jobs'][0]['world']
            key = (w['cell_key'], w['seed'])
            sums = {}
            for j in g['jobs']:
                p = root/'jobs'/j['id']/'SUMMARY.json.gz'
                s = ir._read(p) if p.exists() else None
                if s is not None and not s.get('error'):
                    sums[j['arm']] = s
                if j['arm'] == 'u0' and j['id'] in prox:
                    proxies[key] = prox[j['id']]['windows']
            if 'u0' not in sums:
                continue
            for arm, s in sums.items():
                if arm == 'u0':
                    continue
                d = {}
                for t in WINDOW_STARTS:
                    b = ir.window_j(sums['u0'], t, t+100)
                    c = ir.window_j(s, t, t+100)
                    if b and c and b[1] == c[1]:
                        d[t] = c[0]-b[0]
                windows.setdefault(key, {})[arm] = d
    out = {}
    for cell in sorted({k[0] for k in windows}):
        seeds = sorted(k[1] for k in windows if k[0] == cell)
        cell_out = dict(seeds=seeds, strategies={})
        cs = [proxies[(cell, s)][str(t)]['blk_c'] for s in seeds if (cell, s) in proxies
              for t in WINDOW_STARTS if str(t) in proxies[(cell, s)]]
        srt = sorted(cs)
        cuts = [_quantile(srt, 1/3), _quantile(srt, 2/3)]
        slows = sorted(proxies[(cell, s)][str(t)]['slow_field'] for s in seeds if (cell, s) in proxies
                       for t in WINDOW_STARTS if str(t) in proxies[(cell, s)])
        scuts = [_quantile(slows, 1/3), _quantile(slows, 2/3)]
        cell_out['primary_cuts'] = cuts
        cell_out['secondary_cuts'] = scuts
        cell_out['primary_zero_share'] = (sum(c == 0 for c in cs)/len(cs)) if cs else None
        half = len(seeds)//2
        train, test = seeds[:half], seeds[half:]
        for arm in [a for a in ARMS if a != 'u0' and any(a in windows[(cell, s)] for s in seeds)]:
            per_seed = {s: sum(windows[(cell, s)].get(arm, {}).values()) for s in seeds if arm in windows[(cell, s)]}
            vals = sorted(per_seed.values())
            res = dict(per_seed=dict(sorted(per_seed.items(), key=lambda kv: kv[1])),
                       benefit=sum(v < 0 for v in vals)/len(vals) if vals else None,
                       harm=sum(v > 0 for v in vals)/len(vals) if vals else None,
                       quartiles=[_quantile(vals, q) for q in (.25, .5, .75)] if vals else None)

            def binned(field, cut):
                bins = {}
                for b, (lo, hi) in enumerate(((None, cut[0]), (cut[0], cut[1]), (cut[1], None))):
                    by = {}
                    for s in seeds:
                        if (cell, s) not in proxies:
                            continue
                        for t, dv in windows[(cell, s)].get(arm, {}).items():
                            pv = proxies[(cell, s)].get(str(t))
                            if pv is None:
                                continue
                            x = pv[field]
                            inside = ((lo is None or x > lo) and (hi is None or x <= hi)) if b else (x <= hi)
                            if inside:
                                by.setdefault(s, []).append(dv)
                    m, lo_, hi_ = _seed_boot(by)
                    bins[f't{b+1}'] = dict(range=[lo, hi], n_windows=sum(len(v) for v in by.values()),
                                           n_seeds=len(by), mean_dw=m, ci95=[lo_, hi_])
                return bins
            res['by_primary_tercile'] = binned('blk_c', cuts)
            res['by_secondary_tercile'] = binned('slow_field', scuts)
            # split-seed gate on the primary index
            cand = sorted({proxies[(cell, s)][str(t)]['blk_c'] for s in train if (cell, s) in proxies
                           for t in WINDOW_STARTS if str(t) in proxies[(cell, s)]})
            best = None
            for th in cand:
                gated = [(dv if proxies[(cell, s)][str(t)]['blk_c'] >= th else 0.)
                         for s in train if (cell, s) in proxies for t, dv in windows[(cell, s)].get(arm, {}).items()
                         if str(t) in proxies[(cell, s)]]
                if gated:
                    benefit = -sum(gated)/len(gated)
                    if best is None or benefit > best[0]:
                        best = (benefit, th)
            if best is not None and test:
                th = best[1]
                g_seed = {s: sum(dv for t, dv in windows[(cell, s)].get(arm, {}).items()
                                 if str(t) in proxies.get((cell, s), {}) and proxies[(cell, s)][str(t)]['blk_c'] >= th)
                          for s in test if (cell, s) in proxies}
                a_seed = {s: sum(windows[(cell, s)].get(arm, {}).values()) for s in test}
                gv, av = list(g_seed.values()), list(a_seed.values())
                glo, ghi = boot_ci(gv) if gv else (None, None)
                alo, ahi = boot_ci(av) if av else (None, None)
                res['split_gate'] = dict(train_seeds=train, test_seeds=test, theta=th, train_mean_window_benefit=best[0],
                                         gated_fraction_test=(sum(proxies[(cell, s)][str(t)]['blk_c'] >= th
                                                                  for s in test if (cell, s) in proxies
                                                                  for t in WINDOW_STARTS if str(t) in proxies[(cell, s)])
                                                              / max(1, sum(1 for s in test if (cell, s) in proxies
                                                                           for t in WINDOW_STARTS
                                                                           if str(t) in proxies[(cell, s)]))),
                                         gated_test_mean=sum(gv)/len(gv) if gv else None, gated_test_ci95=[glo, ghi],
                                         gated_benefit=sum(v < 0 for v in gv)/len(gv) if gv else None,
                                         gated_harm=sum(v > 0 for v in gv)/len(gv) if gv else None,
                                         always_test_mean=sum(av)/len(av) if av else None, always_test_ci95=[alo, ahi])
            cell_out['strategies'][arm] = res
        out[cell] = cell_out
    return out


def _collect(roots):
    """(cell, seed) -> {arm: {t_w: d_w}} window differences and (cell, seed) -> u0 window proxies."""
    roots = [Path(r) for r in ([roots] if isinstance(roots, (str, Path)) else roots)]
    windows, proxies = {}, {}
    for root in roots:
        body = ir._read(root/'PLAN.json')
        prox = ir._read(root/'PROXIES.json')['proxies'] if (root/'PROXIES.json').exists() else {}
        for g in body['groups']:
            if g['stage'] != 'paired':
                continue
            w = g['jobs'][0]['world']
            key = (w['cell_key'], w['seed'])
            sums = {}
            for j in g['jobs']:
                p = root/'jobs'/j['id']/'SUMMARY.json.gz'
                s = ir._read(p) if p.exists() else None
                if s is not None and not s.get('error'):
                    sums[j['arm']] = s
                if j['arm'] == 'u0' and j['id'] in prox:
                    proxies[key] = prox[j['id']]['windows']
            if 'u0' not in sums:
                continue
            for arm, s in sums.items():
                if arm == 'u0':
                    continue
                d = {}
                for t in WINDOW_STARTS:
                    b, c = ir.window_j(sums['u0'], t, t+100), ir.window_j(s, t, t+100)
                    if b and c and b[1] == c[1]:
                        d[t] = c[0]-b[0]
                windows.setdefault(key, {})[arm] = d
    return windows, proxies


def pooled(roots, arms=('wall_h_solo2', 'wall_h')):
    """Registered S-REGIMEU0 amendment 6: cross-cell pooled conditional analysis + pooled split-seed gate."""
    windows, proxies = _collect(roots)
    keys = sorted(k for k in windows if k in proxies)
    cells = sorted({k[0] for k in keys})
    out = dict(cells=cells, n_episodes=len(keys))
    cs = sorted(proxies[k][str(t)]['blk_c'] for k in keys for t in WINDOW_STARTS if str(t) in proxies[k])
    ss = sorted(proxies[k][str(t)]['slow_field'] for k in keys for t in WINDOW_STARTS if str(t) in proxies[k])
    cuts, scuts = [_quantile(cs, 1/3), _quantile(cs, 2/3)], [_quantile(ss, 1/3), _quantile(ss, 2/3)]
    out.update(primary_cuts=cuts, secondary_cuts=scuts, primary_zero_share=sum(c == 0 for c in cs)/len(cs))
    split = {}
    for cell in cells:
        seeds = sorted(k[1] for k in keys if k[0] == cell)
        half = len(seeds)//2
        for i, s in enumerate(seeds):
            split[(cell, s)] = 'train' if i < half else 'test'
    out['strategies'] = {}
    for arm in arms:
        res = {}

        def binned(field, cut):
            bins = {}
            for b, (lo, hi) in enumerate(((None, cut[0]), (cut[0], cut[1]), (cut[1], None))):
                by, by_cell = {}, {}
                for k in keys:
                    for t, dv in windows[k].get(arm, {}).items():
                        x = proxies[k][str(t)][field]
                        inside = ((lo is None or x > lo) and (hi is None or x <= hi)) if b else (x <= hi)
                        if inside:
                            by.setdefault(k, []).append(dv)
                            by_cell.setdefault(k[0], []).append(dv)
                m, l_, h_ = _seed_boot(by)
                bins[f't{b+1}'] = dict(range=[lo, hi], n_windows=sum(len(v) for v in by.values()), mean_dw=m,
                                       ci95=[l_, h_], by_cell={c: dict(n=len(v), mean_dw=sum(v)/len(v))
                                                              for c, v in sorted(by_cell.items())})
            return bins
        res['by_primary_tercile'] = binned('blk_c', cuts)
        res['by_secondary_tercile'] = binned('slow_field', scuts)
        train = [k for k in keys if split[k] == 'train' and arm in windows[k]]
        test = [k for k in keys if split[k] == 'test' and arm in windows[k]]
        cand = [('always', None), ('never', None)] + [('ge', th) for th in sorted(
            {proxies[k][str(t)]['blk_c'] for k in train for t in WINDOW_STARTS if str(t) in proxies[k]})]

        def gate(k, t, rule):
            kind, th = rule
            return kind == 'always' or (kind == 'ge' and proxies[k][str(t)]['blk_c'] >= th)

        best = None
        for rule in cand:
            vals = [(dv if gate(k, t, rule) else 0.) for k in train for t, dv in windows[k][arm].items()]
            if vals:
                benefit = -sum(vals)/len(vals)
                if best is None or benefit > best[0] + 1e-9:
                    best = (benefit, rule)
        rule = best[1]
        g_ep = {k: sum(dv for t, dv in windows[k][arm].items() if gate(k, t, rule)) for k in test}
        a_ep = {k: sum(windows[k][arm].values()) for k in test}

        def boot(d):
            return _seed_boot({k: [v] for k, v in d.items()})
        gm, gl, gh = boot(g_ep)
        am, al, ah = boot(a_ep)
        per_cell = {}
        for cell in cells:
            gk = [v for k, v in g_ep.items() if k[0] == cell]
            ak = [v for k, v in a_ep.items() if k[0] == cell]
            if gk:
                per_cell[cell] = dict(n=len(gk), gated_mean=sum(gk)/len(gk), always_mean=sum(ak)/len(ak),
                                      gated_benefit=sum(v < 0 for v in gk)/len(gk), gated_harm=sum(v > 0 for v in gk)/len(gk))
        n_win = sum(1 for k in test for t in windows[k][arm])
        res['split_gate'] = dict(rule=list(rule), train_mean_window_benefit=best[0], n_train=len(train), n_test=len(test),
                                 gated_share_test_windows=sum(1 for k in test for t in windows[k][arm] if gate(k, t, rule))
                                 / max(1, n_win), gated_test_mean=gm, gated_test_ci95=[gl, gh],
                                 gated_benefit=sum(v < 0 for v in g_ep.values())/len(g_ep),
                                 gated_harm=sum(v > 0 for v in g_ep.values())/len(g_ep),
                                 always_test_mean=am, always_test_ci95=[al, ah], per_cell=per_cell)
        out['strategies'][arm] = res
    return out


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
    a.add_argument('--cells', nargs='*', help='demand:share:first-last, e.g. 3200:0.6:989866-989873')
    a.add_argument('--no-identity', action='store_true')
    a.add_argument('--arms', nargs='*', default=list(ARMS))
    a = sub.add_parser('run')
    a.add_argument('--normal-priority', action='store_true', help='MAIN 19:4x: cpuV free, normal priority allowed')
    a.add_argument('--root', required=True)
    a.add_argument('--workers', type=int, default=16)
    a.add_argument('--low-workers', type=int, default=8)
    a.add_argument('--load-high', type=float, default=44.0)
    a = sub.add_parser('worker')
    a.add_argument('--root', required=True)
    a.add_argument('--job', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--root', required=True, nargs='+')
    a.add_argument('--out')
    a = sub.add_parser('pooled')
    a.add_argument('--root', required=True, nargs='+')
    a.add_argument('--out', required=True)
    a = sub.add_parser('amend4')
    a.add_argument('--root', required=True, nargs='+')
    a.add_argument('--out', required=True)
    args = p.parse_args(argv)
    if args.cmd == 'init':
        print(json.dumps(init(args.root, args.cap), indent=1))
    elif args.cmd == 'plan':
        cells = ((CELL[0], CELL[1], SEEDS),)
        if args.cells:
            cells = []
            for c in args.cells:
                d, sh, span = c.split(':')
                lo, hi = (int(x) for x in span.split('-'))
                cells.append((int(d), float(sh), tuple(range(lo, hi+1))))
        body = plan(args.root, config=args.config, u0=args.u0, u0_sha256=args.u0_sha256, cells=cells,
                    identity=not args.no_identity, arms=tuple(args.arms))
        print(json.dumps(dict(plan_sha256=body['plan_sha256'], planned_sim_s=body['planned_sim_s']), indent=1))
    elif args.cmd == 'run':
        print(run(args.root, workers=args.workers, load_high=args.load_high, low_workers=args.low_workers,
                  normal_priority=args.normal_priority))
    elif args.cmd == 'worker':
        s = worker(args.root, args.job)
        print(json.dumps(dict(id=s['id'], error=bool(s['error']), totals=s['totals'], acts=len(s['activations']))))
    elif args.cmd == 'pooled':
        Path(args.out).write_text(json.dumps(pooled(args.root), indent=1, sort_keys=True, default=str), encoding='utf-8')
    elif args.cmd == 'amend4':
        Path(args.out).write_text(json.dumps(amendment4(args.root), indent=1, sort_keys=True, default=str)+chr(10),
                                  encoding='utf-8')
    elif args.cmd == 'analyze':
        rep = analyze(args.root)
        text = json.dumps(rep, indent=1, sort_keys=True, default=str)
        if args.out:
            Path(args.out).write_text(text+'\n', encoding='utf-8')
        keys = ('n', 'mean_J', 'mean_diff', 'ci95', 'rel_diff', 'harm_rate', 'benefit_rate', 'collisions',
                'teleports', 'crn_ok', 'failed')
        brief = {c: {arm: ({k: v[k] for k in keys if k in v} if arm != 'group_minus_best_solo' else v)
                     for arm, v in arms.items()} for c, arms in rep['cells'].items()}
        print(json.dumps(dict(identity=rep.get('identity'), cells=brief), indent=1, default=str))


if __name__ == '__main__':
    main()
