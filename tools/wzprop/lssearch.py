"""S-LOWSHARE-SEARCH (phase lowshare_search): SUMO structured search for GROUP cooperation strategies that beat the
single-CAV incumbent wall_h_solo2 ("solo2") at LOW CAV share on the frozen MAPPO u0 base (owner 2026-11-06 23:28).

Harness = S-REGIMEU0 (tools.coopmwm.regimeu0 worker / run / ledger, plant accel_plant.episode on wz3to2, MAPPO u0
5ea22964 deterministic, native warmup 0-300 s, rules act on 300-1200 s, J = team delay over [300, 1200), paired CRN)
with the WM-C strategy executor (strategy_search / strategy_hook grammar). Every candidate is applied whenever its
trigger is feasible, cooldown 90 s (= the incumbent's hold), one activation at a time (slots 1: a new activation
only after every member of the previous one is released).

Hold-only (no lane-change roles: the S-LCEXEC-SMOKE status is not confirmed). Parameter values are the registered
ones: hold programs wall_h (15 m/s, dmax 1.5, 90 s) and wall_g (22 m/s, dmax 1.0, 120 s) of pg2_plan VARIANTS
(S-REGIME / S-LOWSHARE-U0), the NEWMAP2 R grid speed 18 m/s (dmax 1.0, 90 s; WM-C l2hold_l1gap role), WMDATA onsets
0/20/40/60 s, wall zone [X_B-1800, X_B-400], wall spread 200 m (NEWMAP2), helper spread 300 m (pg_plan R_SPREAD).

Triggers: 'wall' = pg_plan.pick_wall exactly as the incumbent (the group fires at the same geometry; extra
members are the wall's lane-0/1 members); 'wall_anchor' = the same wall pick, plus OPTIONAL rank-1 helpers: the most
downstream free zone CAV in that lane upstream of the lane's wall member, within 300 m; when no helper exists the
activation runs without it (the group falls back to its wall members, e.g. relay -> solo2).

Eval-only per-vehicle logging (owner fairness request 23:3x; no physics change, no driver parameters): the env's
_after_physics is wrapped on the instance (installed at the first warmup hook, t = 1 s); per vehicle: class
(request is_cav fleet label), origin lane (SUMO lane index at first sighting), window road delay (0.5 s x
max(0, 1 - v / 31.29), the env's own term) and window source wait (ledger convention). Sums reconcile with the
episode totals.

    python -m tools.wzprop.lssearch plan    --root R --inputs DIR --stage screen --seeds 989900-989905 [--arms ...]
    python -m tools.wzprop.lssearch run     --root R --workers 12      (nice 19, CUDA_VISIBLE_DEVICES=)
    python -m tools.wzprop.lssearch worker  --root R --job ID
    python -m tools.wzprop.lssearch analyze --root R1 R2 ... --out DIR
Convention (as S-REGIMEU0): diff = J(strategy) - J(reference), veh-s; negative = the strategy is better.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

from . import geometry as G
from . import strategy_search as SS

SCHEMA = 'c4.lssearch.v1'
STABLE_ID = 'S-LOWSHARE-SEARCH'
PHASE = 'lowshare_search'
PHASE_CAP = 700_000
EPISODE_S = 1200
WARMUP_S = 300
BOOT, BOOT_SEED = 10_000, 20261106
CELLS = (('D3200_p20', 3200, 0.2), ('D3200_p30', 3200, 0.3))
SEED_RANGE = (989900, 989949)
SCREEN_SEEDS = tuple(range(989900, 989912))
CONFIRM_SEEDS = tuple(range(989912, 989936))
U0_SHA256 = '5ea2296470c2af2855f9e292b03eeaa62bafdfec08146712d7347d7ce117d734'
CONFIG_SHA256 = '214f5106f1569b913c24ee189312eacc0b6a8c47ca51bdc6ad52bcb78db7db75'
# Parity episodes (S-LOWSHARE-U0 world 989820 D3200_p20, J registered in LOWSHARE_U0_RESULTS.jsonl 3b90b6b9):
PARITY = {'wall_h_solo2': 94680.48012256423, 'wall_h': 84482.35808152931}
PARITY_WORLD = (989820, 3200, 0.2)
SPEED_LIMIT = G.SPEED_LIMIT

H = dict(v_w=15.0, dmax=1.5, hold_s=90)        # pg2 VARIANTS wall_h (the incumbent's program)
GN = dict(v_w=22.0, dmax=1.0, hold_s=120)      # pg2 VARIANTS wall_g (S-LOWSHARE-U0)
M = dict(v_w=18.0, dmax=1.0, hold_s=90)        # NEWMAP2 R grid middle speed (WM-C l2hold_l1gap lane-1 role)


def _role(lane, prog, rank=0, onset=0):
    return dict(lane=lane, rank=rank, onset_s=onset, lc_target=None, **prog)


def _spec(name, trigger, roles, spread=G.WALL_SPREAD):
    return dict(name=name, trigger=trigger, zone='full', spread_m=float(spread), cooldown_s=90, slots=1, gate=None,
                roles=roles)


INCUMBENT = _spec('wall_h_solo2', 'wall', [_role(2, H)])
WALL_H = _spec('wall_h', 'wall', [_role(0, H), _role(1, H), _role(2, H)])
GROUPS = (
    # (b) two-lane partial walls (the wall's lane-1 member also holds; lane-0 member native)
    _spec('pw2_h12', 'wall', [_role(2, H), _role(1, H)]),
    _spec('pw2_h2g1', 'wall', [_role(2, H), _role(1, GN)]),
    # (c) lane-2 hold + lane-1 gap-giver (speed only, 18 m/s), immediate or 20 s later
    _spec('gap1_m', 'wall', [_role(2, H), _role(1, M)]),
    _spec('gap1_m_o20', 'wall', [_role(2, H), _role(1, M, onset=20)]),
    # (d) lane-2 hold + closing-lane (SUMO lane 0) early-merge assist (speed only)
    _spec('assist0_m', 'wall', [_role(2, H), _role(0, M)]),
    _spec('assist0_g', 'wall', [_role(2, H), _role(0, GN)]),
    # (a) relay metering: a second lane-2 CAV upstream takes over (onset 40 overlap / 60 at the latest onset)
    _spec('relay2_o40', 'wall_anchor', [_role(2, H), _role(2, H, rank=1, onset=40)], spread=G.R_SPREAD),
    _spec('relay2_o60', 'wall_anchor', [_role(2, H), _role(2, H, rank=1, onset=60)], spread=G.R_SPREAD),
    _spec('relay2_o60_g', 'wall_anchor', [_role(2, H), _role(2, dict(H, hold_s=120), rank=1, onset=60)],
          spread=G.R_SPREAD),
    # (e) 3-CAV staggered metering across lanes 2/1
    _spec('stagger3', 'wall_anchor', [_role(2, H), _role(1, M, onset=20), _role(2, H, rank=1, onset=60)],
          spread=G.R_SPREAD),
)
SOLOS = ('wall_h_solo2',)
GROUP_NAMES = ('wall_h',) + tuple(s['name'] for s in GROUPS)
SCREEN_ARMS = ('u0', 'wall_h_solo2', 'wall_h') + tuple(s['name'] for s in GROUPS)
CONFIRM_TOP = 3


def validate(s):
    """Registered values (strategy_search.validate on the same roles), hold-only."""
    s = dict(s)
    if any(r['lc_target'] is not None for r in s['roles']):
        raise ValueError('hold-only: lane-change roles need S-LCEXEC-SMOKE confirmation')
    if s['trigger'] == 'wall':
        SS.validate(s)
    elif s['trigger'] == 'wall_anchor':
        SS.validate(dict(s, trigger='lanes'))
        lanes0 = {r['lane'] for r in s['roles'] if r['rank'] == 0}
        if any(r['rank'] == 1 and r['lane'] not in lanes0 for r in s['roles']):
            raise ValueError('a rank-1 helper needs the rank-0 role of its lane')
    else:
        raise ValueError('trigger')
    if int(s['cooldown_s']) != 90 or int(s.get('slots', 1)) != 1:
        raise ValueError('cooldown 90 s / slots 1 (the incumbent rule)')
    return s


SPECS = {s['name']: validate(s) for s in (INCUMBENT, WALL_H) + GROUPS}


# ------------------------------------------------------------------------------------------ selection / core
def select(s, cavs, observed=None):
    """Members aligned with s['roles'] (SUMO-frame rows id/edge/lane/x/v); None entries = missing optional helpers;
    None if the trigger is infeasible."""
    if s['trigger'] == 'wall':
        return SS.select_members(s, cavs, observed)
    lo, hi = SS.ZONES[s['zone']]
    z = [c for c in cavs if c['edge'] == 'up' and lo <= float(c['x']) <= hi]
    wall = SS._pick_wall(z)
    if wall is None:
        return None
    used = {wall[r['lane']]['id'] for r in s['roles'] if r['rank'] == 0}
    out = []
    for r in s['roles']:
        a = wall[r['lane']]
        if r['rank'] == 0:
            out.append(a)
            continue
        xa = float(a['x'])
        cand = sorted((c for c in z if int(c['lane']) == r['lane'] and c['id'] not in used
                       and float(c['x']) < xa and xa - float(c['x']) <= float(s['spread_m'])),
                      key=lambda c: (-float(c['x']), c['id']))
        pick = cand[0] if cand else None
        if pick is not None:
            used.add(pick['id'])
        out.append(pick)
    return out


class LSCore:
    """strategy_hook.StrategyCore with optional rank-1 helpers (identical for 'wall' specs)."""

    def __init__(self, strategy, warmup_s=WARMUP_S):
        from . import strategy_hook as SH
        self.SH = SH
        self.s = validate(strategy)
        self.warmup = int(warmup_s)
        self.active = {}
        self.acts, self.issued = [], []
        self.next_ok = -1
        self.lane_held = 0
        self.lane_requests = 0

    def law(self, role, v):
        return max(-float(role['dmax']), min(1.0, float(role['v_w']) - float(v)))

    def step(self, rows, commands, t):
        SH = self.SH
        by = {r['vid']: r for r in rows if r.get('is_cav')}
        accel, lanes = {}, {}
        if t < self.warmup:
            return accel, lanes
        for vid, m in list(self.active.items()):
            el = t - m['start']
            role = m['role']
            row = by.get(vid)
            end = int(role['onset_s']) + int(role['hold_s'])
            why = ('left' if row is None or vid not in commands else 'hold_end' if el >= end
                   else 'at_drop' if float(row['s_m']) >= SH.RELEASE_S else None)
            if why is not None:
                m['act']['release'][vid] = [el, why]
                self.active.pop(vid)
                continue
            if el >= int(role['onset_s']):
                accel[vid] = self.law(role, row['v_mps'])
                self.issued.append([t, vid, round(accel[vid], 4)])
            lanes[vid] = int(row['lane'])
        busy = len({id(m['act']) for m in self.active.values()})
        if t >= self.next_ok and busy < int(self.s.get('slots', 1)):
            free = [SH.to_sumo_row(r) for vid, r in by.items() if vid not in self.active and vid in commands]
            mem = select(self.s, free, SH.observed_rows(rows))
            if mem:
                picked = [(i, role, c) for i, (role, c) in enumerate(zip(self.s['roles'], mem)) if c is not None]
                act = dict(t=t, strategy=self.s['name'], members=[c['id'] for _i, _r, c in picked],
                           roles=[i for i, _r, _c in picked], missing=[i for i, c in enumerate(mem) if c is None],
                           sumo_lanes=[c['lane'] for _i, _r, c in picked],
                           x=[round(c['x'], 2) for _i, _r, c in picked], release={})
                self.acts.append(act)
                self.next_ok = t + int(self.s['cooldown_s'])
                for _i, role, c in picked:
                    self.active[c['id']] = dict(start=t, role=role, act=act)
                    row = by[c['id']]
                    lanes[c['id']] = int(row['lane'])
                    if int(role['onset_s']) == 0:
                        accel[c['id']] = self.law(role, row['v_mps'])
                        self.issued.append([t, c['id'], round(accel[c['id']], 4)])
        self.lane_held += len(lanes)
        return accel, lanes


# ------------------------------------------------------------------------------------------ per-vehicle logging
class PerVehicle:
    """Eval-only per-vehicle window delay with origin lane and class (see module docstring)."""

    def __init__(self, a=WARMUP_S, b=EPISODE_S):
        self.a, self.b = float(a), float(b)
        self.origin, self.road, self.ins = {}, {}, {}
        self.n_ins = 0
        self.arrived_a = None
        self.requests = None
        self.limit = None
        self.installed_at = None
        self.substeps = 0
        self.window_substeps = 0
        self.error = None

    def install(self, env):
        if getattr(env, '_lss_pv', None) is self:
            return
        try:
            self.limit = float(sys.modules[type(env).__module__].SPEED_LIMIT_MPS)
            orig = env._after_physics
            pv = self

            def wrapped(after, *args, **kwargs):
                out = orig(after, *args, **kwargs)
                pv.substep(env, float(after))
                return out
            env._after_physics = wrapped
            env._lss_pv = self
            self.installed_at = float(env.now)
            self.requests = {vid: (float(r.depart_s), bool(r.is_cav)) for vid, r in env.ledger.requests.items()}
            self._sync(env, float(env.now))
        except Exception as exc:          # logging must never alter the episode
            self.error = repr(exc)

    def _sync(self, env, after):
        led = env.ledger
        if len(led.inserted) != self.n_ins:
            for v in led.inserted:
                if v not in self.ins:
                    self.ins[v] = float(led.time)
            self.n_ins = len(led.inserted)
        for vid, t in env._table.items():
            if vid not in self.origin:
                self.origin[vid] = (str(t.get('road_id')), int(t.get('lane_index')), after)
        if self.arrived_a is None and after >= self.a - 1e-9:
            self.arrived_a = set(led.arrived)

    def substep(self, env, after):
        if self.error is not None:
            return
        try:
            self.substeps += 1
            self._sync(env, after)
            if self.a + 1e-9 < after <= self.b + 1e-9:
                self.window_substeps += 1
                lim, road = self.limit, self.road
                for vid, t in env._table.items():
                    road[vid] = road.get(vid, 0.) + 0.5 * max(0.0, 1.0 - float(t['speed_mps']) / lim)
        except Exception as exc:
            self.error = repr(exc)

    def finish(self):
        rows, sum_road, sum_src = [], 0., 0.
        edges = {}
        for vid, (dep, cav) in sorted((self.requests or {}).items()):
            if dep >= self.b or (self.arrived_a is not None and vid in self.arrived_a):
                continue
            ins = self.ins.get(vid)
            src = max(0., min(self.b if ins is None else ins, self.b) - max(dep, self.a))
            road = self.road.get(vid, 0.)
            o = self.origin.get(vid)
            sum_road += road
            sum_src += src
            if o is not None:
                edges[o[0]] = edges.get(o[0], 0) + 1
            rows.append([vid, int(cav), -1 if o is None else o[1], round(road, 6), round(src, 6)])
        pop = {r[0] for r in rows}
        missing_road = sum(v for k, v in self.road.items() if k not in pop)
        return dict(schema=SCHEMA + '.perveh', window=[self.a, self.b], columns=['vid', 'cav', 'origin_lane',
                                                                                'road_vs', 'source_vs'],
                    rows=rows, sum_road_vs=sum_road, sum_source_vs=sum_src, road_outside_population_vs=missing_road,
                    origin_edges=edges, installed_at=self.installed_at, substeps=self.substeps,
                    window_substeps=self.window_substeps, error=self.error)


def make_hook(spec, warmup_s=WARMUP_S):
    """S-REGIMEU0 RuleHook drop-in (tools.coopmwm.imitreplay.Recorder base) + per-vehicle logger."""
    from tools.coopmwm import imitreplay as ir

    class LSHook(ir.Recorder):
        def __init__(self):
            super().__init__()
            self.core = LSCore(spec, warmup_s) if spec is not None else None
            self.strategy = None if spec is None else spec['name']
            self.pv = PerVehicle()

        @property
        def acts(self):
            return [] if self.core is None else self.core.acts

        @property
        def issued(self):
            return [] if self.core is None else self.core.issued

        @property
        def lane_held(self):
            return 0 if self.core is None else self.core.lane_held

        def _install_pv(self):
            mod = sys.modules.get('tools.leverprobe.backend')
            if mod is None:
                self.pv.error = 'tools.leverprobe.backend not imported before the hook'
                return
            original = mod.Backend.observe
            pv = self.pv

            def observe(b, env):
                pv.install(env)
                return original(b, env)
            mod.Backend.observe = observe

        def __call__(self, controller, job, prefixes):
            self.controller = controller
            self._install_pv()
            if self.core is None:
                return
            previous = controller.apply

            def apply(rows, commands, now):
                desired = previous(rows, commands, now)
                accel, lanes = self.core.step(rows, commands, int(now))
                for vid, a in accel.items():
                    desired[vid] = dict(requested_raw=a, acceleration_residual=a - commands[vid],
                                        lssearch_rule=self.strategy)
                if lanes:
                    targets = dict(getattr(controller, 'g5_lane_targets', {}) or {})
                    targets.update(lanes)
                    controller.g5_lane_targets = targets
                return desired

            controller.apply = apply

    return LSHook()


# ------------------------------------------------------------------------------------------ plan / run / worker
def _harness():
    from tools.coopmwm import imitreplay as ir
    from tools.coopmwm import regimeu0 as R
    return ir, R


def plan_groups(stage, seeds, arms_by_cell, world_of, parity=False):
    groups = []
    if parity:
        s, d, p = PARITY_WORLD
        w = world_of(s, d, p)
        for arm in parity:
            groups.append(dict(stage='parity', cell=w['cell_key'], jobs=[
                dict(id=f'parity_{w["cell_key"]}_s{s}_{arm}', arm=arm, strategy=arm, strategy_spec=SPECS[arm], world=w,
                     sim_s=EPISODE_S, expected_J=PARITY[arm])]))
    for s in seeds:
        if not SEED_RANGE[0] <= int(s) <= SEED_RANGE[1]:
            raise ValueError(f'seed {s} outside the registered S-LOWSHARE-SEARCH range')
        for cell, d, p in CELLS:
            if cell not in arms_by_cell:
                continue
            w = world_of(int(s), d, p)
            groups.append(dict(stage=stage, cell=cell, jobs=[
                dict(id=f'{stage}_{cell}_s{s}_{arm}', arm=arm, strategy=None if arm == 'u0' else arm,
                     strategy_spec=None if arm == 'u0' else SPECS[arm], world=w, sim_s=EPISODE_S)
                for arm in arms_by_cell[cell]]))
    return groups


def plan(root, *, inputs, stage, seeds=(), arms_by_cell=None, parity=(), selection=None):
    import shutil
    ir, R = _harness()
    root = ir.guard_path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'PLAN.json').exists():
        raise ValueError('PLAN.json already frozen')
    (root / 'inputs').mkdir(exist_ok=True)
    files = {}
    for name, want in (('CONFIG.json', CONFIG_SHA256), ('u0.pt', U0_SHA256)):
        src = ir.guard_path(Path(inputs) / name)
        shutil.copyfile(src, root / 'inputs' / name)
        files[name] = dict(source=str(src), sha256=ir.sha_file(root / 'inputs' / name))
        if files[name]['sha256'] != want:
            raise ValueError(f'{name} differs from the S-REGIMEU0 / S-LOWSHARE-U0 input')
    arms_by_cell = arms_by_cell or {c: list(SCREEN_ARMS) for c, _d, _p in CELLS}
    groups = plan_groups(stage, seeds, arms_by_cell, R.world_of, parity=list(parity))
    planned = sum(j['sim_s'] for g in groups for j in g['jobs'])
    ir._write_new(root / 'REGISTRATION.json', dict(schema=SCHEMA, stable_id=STABLE_ID, phase=PHASE, cap_sim_s=planned,
                                                   phase_cap_sim_s=PHASE_CAP, stage=stage,
                                                   module_sha256=ir.sha_file(__file__), seeds=list(seeds),
                                                   arms_by_cell=arms_by_cell, parity=list(parity),
                                                   selection=selection, unix=time.time()))
    body = dict(schema=SCHEMA + '.plan', stable_id=STABLE_ID, phase=PHASE, stage=stage, cap_sim_s=planned,
                inputs=files, groups=groups, specs=SPECS, planned_sim_s=planned, selection=selection,
                module_sha256=ir.sha_file(__file__), unix=time.time())
    body['plan_sha256'] = ir.digest({k: v for k, v in body.items() if k != 'unix'})
    ir._write_new(root / 'PLAN.json', body)
    return body


def worker(root, ident):
    ir, R = _harness()
    root = Path(root)
    body = ir._read(root / 'PLAN.json')
    job = next(j for g in body['groups'] for j in g['jobs'] if j['id'] == ident)
    holder = {}

    def factory(_name):
        holder['hook'] = make_hook(job['strategy_spec'], warmup_s=R.WARMUP_S)
        return holder['hook']
    R.RuleHook = factory
    R._job = lambda _root, _ident: dict(job)
    try:
        return R.worker(root, ident)
    finally:
        hook = holder.get('hook')
        work = root / 'jobs' / ident
        if hook is not None and work.exists() and not (work / 'PERVEH.json.gz').exists():
            with gzip.open(work / 'PERVEH.json.gz', 'xt', encoding='utf-8') as f:
                json.dump(hook.pv.finish(), f)


def run(root, **kw):
    """regimeu0.run with every worker = `python -m tools.wzprop.lssearch worker` (own PIDs, ledger, cap)."""
    ir, R = _harness()
    import subprocess
    real = subprocess.Popen

    def popen(cmd, *a, **k2):
        if isinstance(cmd, list) and 'tools.coopmwm.regimeu0' in cmd:
            cmd = [c if c != 'tools.coopmwm.regimeu0' else 'tools.wzprop.lssearch' for c in cmd]
        return real(cmd, *a, **k2)
    R.subprocess.Popen = popen
    try:
        return R.run(root, **kw)
    finally:
        R.subprocess.Popen = real


# ------------------------------------------------------------------------------------------ analysis (zero physics)
def _boot_np(values, n=BOOT, seed=BOOT_SEED, alpha=0.05):
    import numpy as np
    v = np.asarray(values, float)
    if len(v) == 0:
        return None
    idx = np.random.default_rng(seed).integers(0, len(v), (n, len(v)))
    m = v[idx].mean(1)
    return [float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))]


N_CONFIRM_TESTS = 6          # 3 confirmed group candidates x 2 cells (Bonferroni robustness CI)


def ineq(x):
    """Gini, Jain and worst-off-10% mean of per-vehicle delays."""
    x = sorted(float(v) for v in x)
    n = len(x)
    if n == 0 or sum(x) <= 0:
        return dict(gini=None, jain=None, worst10=None, n=n)
    s = sum(x)
    gini = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(x)) / (n * s)
    jain = s * s / (n * sum(v * v for v in x))
    k = max(1, math.ceil(0.1 * n))
    return dict(gini=gini, jain=jain, worst10=sum(x[-k:]) / k, n=n)


FAIR_GROUPS = ('lane0', 'lane1', 'lane2', 'hdv', 'cav')


def fairness_of(pv):
    if pv is None or pv.get('error'):
        return None
    groups = {g: [0, 0.] for g in FAIR_GROUPS + ('uninserted',)}
    per = []
    for _vid, cav, lane, road, src in pv['rows']:
        d = road + src
        per.append(d)
        groups['cav' if cav else 'hdv'][0] += 1
        groups['cav' if cav else 'hdv'][1] += d
        key = f'lane{lane}' if lane in (0, 1, 2) else 'uninserted'
        groups[key][0] += 1
        groups[key][1] += d
    out = dict(groups={g: dict(n=n, total_vs=t, mean_vs=(t / n if n else None)) for g, (n, t) in groups.items()},
               **ineq(per), sum_vs=sum(per))
    return out


def _episode_row(R, ir, root, job, s, pv, base, solo):
    rec = dict(seed=job['world']['seed'], cell=job['world']['cell_key'], arm=job['arm'], stage=None, root=str(root),
               status='not_run')
    if s is None:
        return rec
    if s.get('error'):
        rec.update(status='failed', error=s['error']['error'][:200])
        return rec
    tot = s.get('totals') or {}
    saf = ir.safety(s)
    J = R.control_j(s)
    acts = s.get('activations') or ()
    rec.update(status='ok', J_300_1200_vs=J, team_delay_vs=tot.get('team_delay_vs'), road_delay_vs=tot.get('road_delay_vs'),
               source_wait_vs=tot.get('source_wait_vs'), collisions=saf.get('collisions'), teleports=saf.get('teleports'),
               ttc_below_3_vs=saf.get('ttc_below_3_vs'), hard_brake_vs=saf.get('hard_brake'), activations=len(acts),
               members_per_activation=(sum(len(a.get('members') or ()) for a in acts) / len(acts)) if acts else None,
               helpers_missing=sum(len(a.get('missing') or ()) for a in acts), diff_vs_u0=None, crn_ok=None,
               first_divergence_s=None, first_activation_s=None)
    if job['arm'] != 'u0' and base is not None and not base.get('error'):
        rec['diff_vs_u0'] = J - R.control_j(base)
        rec['first_divergence_s'] = ir.first_divergence(base, s)
        rec['first_activation_s'] = min((a['t'] for a in acts), default=None)
        rec['crn_ok'] = rec['first_divergence_s'] is None or (
            rec['first_activation_s'] is not None and rec['first_divergence_s'] >= rec['first_activation_s'])
    if job['arm'] not in ('u0', 'wall_h_solo2') and solo is not None and not solo.get('error'):
        rec['diff_vs_solo2'] = J - R.control_j(solo)
    f = fairness_of(pv)
    if f is not None:
        rec['fair'] = f
        rec['pv_reconcile'] = dict(road=pv['sum_road_vs'] - (tot.get('road_delay_vs') or 0.),
                                   source=pv['sum_source_vs'] - (tot.get('source_wait_vs') or 0.),
                                   outside=pv.get('road_outside_population_vs'))
    elif pv is not None:
        rec['pv_error'] = pv.get('error')
    return rec


def collect(roots):
    ir, R = _harness()
    rows, parity, plans = [], [], []
    for root in [Path(r) for r in roots]:
        body = ir._read(root / 'PLAN.json')
        plans.append(dict(root=str(root), plan_sha256=body['plan_sha256'], stage=body['stage'], spent=ir.spent(root)))

        def load(ident, name='SUMMARY.json.gz'):
            p = root / 'jobs' / ident / name
            return ir._read(p) if p.exists() else None
        for g in body['groups']:
            sums = {j['arm']: load(j['id']) for j in g['jobs']}
            base, solo = sums.get('u0'), sums.get('wall_h_solo2')
            for j in g['jobs']:
                rec = _episode_row(R, ir, root, j, sums[j['arm']], load(j['id'], 'PERVEH.json.gz'), base, solo)
                rec['stage'] = g['stage']
                if g['stage'] == 'parity':
                    rec['expected_J'] = j['expected_J']
                    rec['parity_equal'] = rec.get('J_300_1200_vs') is not None and abs(
                        rec['J_300_1200_vs'] - j['expected_J']) < 1e-6
                    parity.append(rec)
                else:
                    rows.append(rec)
    return rows, parity, plans


def _paired(rows, cell, arm, ref, key='J_300_1200_vs'):
    E = {(r['seed'], r['arm']): r for r in rows if r['cell'] == cell and r['status'] == 'ok'}
    seeds = sorted({s for (s, a) in E if a == arm} & {s for (s, a) in E if a == ref})
    return seeds, [E[(s, arm)][key] - E[(s, ref)][key] for s in seeds], E


def arm_stats(rows, cell, arm, R):
    """S-REGIMEU0 ANALYSIS per-arm entry (+ vs solo2 and fairness)."""
    recs = [r for r in rows if r['cell'] == cell and r['arm'] == arm and r['status'] == 'ok']
    a = dict(n=len(recs), mean_J=(sum(x['J_300_1200_vs'] for x in recs) / len(recs)) if recs else None,
             failed=sum(r['status'] == 'failed' for r in rows if r['cell'] == cell and r['arm'] == arm))
    for k in ('collisions', 'teleports', 'ttc_below_3_vs', 'hard_brake_vs'):
        vals = [x[k] for x in recs if x.get(k) is not None]
        kk = 'hard_brake' if k == 'hard_brake_vs' else k
        a[kk] = sum(vals) if vals else None
        a[kk + '_mean'] = (sum(vals) / len(vals)) if vals else None
    if arm == 'u0' or not recs:
        return a
    seeds, d, E = _paired(rows, cell, arm, 'u0')
    if d:
        lo, hi = R.boot_ci(d)
        a.update(n_paired=len(d), mean_diff=sum(d) / len(d), ci95=[lo, hi], benefit_rate=sum(x < 0 for x in d) / len(d),
                 harm_rate=sum(x > 0 for x in d) / len(d), crn_ok=sum(bool(E[(s, arm)].get('crn_ok')) for s in seeds),
                 mean_activations=sum(x['activations'] for x in recs) / len(recs),
                 mean_members=_mean([x.get('members_per_activation') for x in recs]))
    if arm != 'wall_h_solo2':
        s2, d2, _ = _paired(rows, cell, arm, 'wall_h_solo2')
        if d2:
            lo, hi = R.boot_ci(d2)
            a['vs_solo2'] = dict(n=len(d2), mean=sum(d2) / len(d2), ci95=[lo, hi],
                                 ci_bonferroni6=_boot_np(d2, alpha=0.05 / N_CONFIRM_TESTS),
                                 beats_fraction=sum(x < 0 for x in d2) / len(d2), seeds=s2, diffs=d2)
    a['fairness'] = fairness_stats(rows, cell, arm, 'u0')
    return a


def _mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def fairness_stats(rows, cell, arm, ref):
    """Paired (arm - ref) per-seed diffs: per-group mean per-vehicle delay, group totals, Gini/Jain/worst-10%.
    FAIR (registered MAIN 23:33): total delay lower than ref (paired CI upper < 0) AND no group (origin lane 0/1/2,
    HDV, CAV) worse off (paired CI lower > 0 = significantly worse); point-estimate variant reported too."""
    E = {(r['seed'], r['arm']): r for r in rows if r['cell'] == cell and r['status'] == 'ok' and r.get('fair')}
    seeds = sorted({s for (s, a) in E if a == arm} & {s for (s, a) in E if a == ref})
    if not seeds:
        return None
    out = dict(n=len(seeds), ref=ref, groups={})
    for g in FAIR_GROUPS:
        d_mean, d_tot = [], []
        for s in seeds:
            ga, gr = E[(s, arm)]['fair']['groups'][g], E[(s, ref)]['fair']['groups'][g]
            if ga['n'] and gr['n']:
                d_mean.append(ga['mean_vs'] - gr['mean_vs'])
                d_tot.append(ga['total_vs'] - gr['total_vs'])
        if d_mean:
            ci = _boot_np(d_mean)
            out['groups'][g] = dict(n=len(d_mean), mean_per_vehicle_diff=sum(d_mean) / len(d_mean), ci95=ci,
                                    total_diff=sum(d_tot) / len(d_tot), total_ci95=_boot_np(d_tot),
                                    worse_sig=ci[0] > 0, worse_point=sum(d_mean) > 0,
                                    ref_mean_per_vehicle=_mean([E[(s, ref)]['fair']['groups'][g]['mean_vs']
                                                                for s in seeds]))
    for k in ('gini', 'jain', 'worst10'):
        d = [E[(s, arm)]['fair'][k] - E[(s, ref)]['fair'][k] for s in seeds
             if E[(s, arm)]['fair'][k] is not None and E[(s, ref)]['fair'][k] is not None]
        if d:
            out[k] = dict(mean_diff=sum(d) / len(d), ci95=_boot_np(d),
                          ref_mean=_mean([E[(s, ref)]['fair'][k] for s in seeds]))
    dj = [E[(s, arm)]['J_300_1200_vs'] - E[(s, ref)]['J_300_1200_vs'] for s in seeds]
    ci = _boot_np(dj)
    lowers = ci[1] < 0
    out['total'] = dict(mean_diff=sum(dj) / len(dj), ci95=ci, lowers_sig=lowers)
    out['FAIR'] = bool(lowers and out['groups'] and not any(v['worse_sig'] for v in out['groups'].values()))
    out['FAIR_point'] = bool(lowers and out['groups'] and not any(v['worse_point'] for v in out['groups'].values()))
    out['worse_groups_sig'] = [g for g, v in out['groups'].items() if v['worse_sig']]
    out['worse_groups_point'] = [g for g, v in out['groups'].items() if v['worse_point']]
    return out


def select_top(rows, cell, k=CONFIRM_TOP):
    """Pre-registered screen -> confirm rule: the k group candidates (GROUP_NAMES order breaks ties) with the most
    negative screen mean of J(group) - J(solo2) over paired seeds."""
    scored = []
    for i, arm in enumerate(GROUP_NAMES):
        _s, d, _E = _paired(rows, cell, arm, 'wall_h_solo2')
        if d:
            scored.append((sum(d) / len(d), i, arm))
    return [a for _m, _i, a in sorted(scored)[:k]]


def analyze(roots, stage):
    ir, R = _harness()
    rows, parity, plans = collect(roots)
    rows = [r for r in rows if r['stage'] == stage]
    rep = dict(schema='coopmwm.regimeu0.v1.analysis', lssearch_schema=SCHEMA + '.analysis', stable_id=STABLE_ID,
               stage=stage, roots=[p['root'] for p in plans], plan_sha256=[p['plan_sha256'] for p in plans],
               spent=sum(p['spent'] for p in plans), plans=plans, specs=SPECS, cells={}, parity=parity,
               convention='diff = J(strategy) - J(reference) over [300, 1200) s, veh-s; negative = better',
               fairness_rule=fairness_stats.__doc__.strip())
    for cell in sorted({r['cell'] for r in rows}):
        arms = [a for a in SCREEN_ARMS if any(r['cell'] == cell and r['arm'] == a for r in rows)]
        out = {arm: arm_stats(rows, cell, arm, R) for arm in arms}
        base = out.get('u0', {}).get('mean_J')
        for arm, a in out.items():
            if arm != 'u0' and a.get('mean_diff') is not None and base:
                a['rel_diff'] = a['mean_diff'] / base
                a['rel_ci95'] = [a['ci95'][0] / base, a['ci95'][1] / base]
        for arm in arms:
            if arm in GROUP_NAMES and out[arm].get('vs_solo2'):
                v = out[arm]['vs_solo2']
                out[f'{arm}_minus_best_solo'] = dict(point=v['mean'], ci95=v['ci95'], n=v['n'], solos=list(SOLOS),
                                                     group=arm, beats_all_solos_fraction=v['beats_fraction'])
        rep['cells'][cell] = out
        if stage == 'screen':
            rep.setdefault('selection', {})[cell] = select_top(rows, cell)
    return rep, rows


def write_outputs(roots, out_dir, stage):
    rep, rows = analyze(roots, stage)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = 'SCREEN' if stage == 'screen' else 'CONFIRM'
    (out / f'{name}_ANALYSIS.json').write_text(json.dumps(rep, indent=1, default=str))
    with (out / f'{name}_RESULTS.jsonl').open('w', encoding='utf-8') as f:
        for r in sorted(rows, key=lambda r: (r['cell'], r['seed'], r['arm'])):
            f.write(json.dumps({k: v for k, v in r.items()}, sort_keys=True, default=str) + '\n')
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop.lssearch')
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('plan')
    a.add_argument('--root', required=True)
    a.add_argument('--inputs', required=True)
    a.add_argument('--stage', required=True, choices=('parity', 'screen', 'confirm'))
    a.add_argument('--seeds', default='', help='first-last')
    a.add_argument('--parity', nargs='*', default=[])
    a.add_argument('--arms-json', help='{cell: [arms]} (confirm: u0, wall_h_solo2 + the selected top 3)')
    a.add_argument('--selection')
    a = sub.add_parser('run')
    a.add_argument('--root', required=True)
    a.add_argument('--workers', type=int, default=12)
    a.add_argument('--low-workers', type=int, default=12)
    a.add_argument('--load-high', type=float, default=1e9)
    a = sub.add_parser('worker')
    a.add_argument('--root', required=True)
    a.add_argument('--job', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--root', required=True, nargs='+')
    a.add_argument('--stage', required=True, choices=('screen', 'confirm'))
    a.add_argument('--out', required=True)
    x = ap.parse_args(argv)
    if x.cmd == 'plan':
        seeds = []
        if x.seeds:
            lo, hi = (int(v) for v in x.seeds.split('-'))
            seeds = list(range(lo, hi + 1))
        arms = json.loads(Path(x.arms_json).read_text()) if x.arms_json else None
        body = plan(x.root, inputs=x.inputs, stage=x.stage, seeds=seeds, arms_by_cell=arms, parity=x.parity,
                    selection=x.selection)
        print(json.dumps(dict(planned_sim_s=body['planned_sim_s'], plan_sha256=body['plan_sha256'],
                              groups=len(body['groups']))))
    elif x.cmd == 'run':
        print(run(x.root, workers=x.workers, low_workers=x.low_workers, load_high=x.load_high))
    elif x.cmd == 'worker':
        worker(x.root, x.job)
    else:
        rep = write_outputs(x.root, x.out, x.stage)
        print(json.dumps(dict(stage=x.stage, spent=rep['spent'], selection=rep.get('selection'),
                              parity=[(p['arm'], p.get('parity_equal')) for p in rep['parity']]), default=str))


if __name__ == '__main__':
    main()
