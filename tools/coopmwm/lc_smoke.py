"""S-LCEXEC-SMOKE (phase lcexec_smoke, MAIN 2026-11-06 ~23:3x): SUMO mechanics smoke of the LCEXEC
action interface on wz3to2 on top of the frozen speed-only MAPPO u0 runtime.

NEVER run ``run`` locally: it starts SUMO. ``plan``/``summarize`` are zero-physics.

    source env.sh   # SUMO 1.27.1 interpreter, PYTHONPATH=<bundle>, COOPMWM_MAP_SPEC=<wz3to2 MAP_SPEC>,
                    # COOPMWM_G5_RUNTIME=1 COOPMWM_LC_RUNTIME=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
    nice -n 19 python -m tools.coopmwm.lc_smoke plan --root <root>/run --runtime <e3final> --checkpoint <u0.pt>
    nice -n 19 python -m tools.coopmwm.lc_smoke run --root <root>/run --workers 8 [--only canary|main]
    python -m tools.coopmwm.lc_smoke summarize --root <root>/run

Episode (one per world): the frozen e3final env (G5 overlay) bound to wz3to2 (map_runtime) with the
LCEXEC env subclass (lc_runtime), native warm-up 0-300 s, then the frozen MAPPO u0 (d9042cc5,
deterministic, native lanes) commands every CAV each second. At the registered moments the
probe's OWN family builders (verbatim S-LCMOVES lcm_families / lcm_plan, sha-pinned fixture) pick
one move on the live snapshot (family rotation per world/moment, members disjoint from running
moves). The move runs through the PRODUCTION executor path: lc_roles.member_program ->
lc_families.template_from_program -> Protocol.start (initiator + joins at the move second) ->
each second Protocol.receive (own rows/sensing = cav_report_rows, teammates' own t-1 packets) ->
Protocol.commands (lc_program executors + own-sensing lane gate) -> env.lc_tick -> env.step (the
single writers: CommandOwnership for speed, the LCEXEC lane path, lcMode/tau knobs). Program
members bypass the MAPPO safety filter as the probes did (MAIN 23:3x); speedMode 27, the env
edge-end guard and the collision/teleport stop rule stay.

TraCI command parity on the live states: a SHADOW copy of the verbatim probe ``Executor``
(lcm_families.Executor = lcm_core.Arm) gets the same members and reads the SAME live state each
second BEFORE the env writes; its writes are recorded, not executed. Our actual TraCI writes on the
members are recorded by a pass-through spy. Per (move second, member) the multisets are compared
(slowDown target within 1e-3 m/s, changeLane index/duration, setLaneChangeMode, setTau, setSpeed)
after the documented base normalisations (the probes' base was native, ours is MAPPO u0):
N1 setSpeed(-1) without a live speed command is a SUMO no-op (dropped both sides); N2 our
setSpeed(-1) right after a program speed command == the probe's 1 s slowDown expiry; N3 at the
probe's release, its setSpeed(-1) hand-back == our hand-back to the base controller (MAPPO
slowDown); base-controller (MAPPO) slowDowns of members are not move commands. Remaining
differences are classified: gate (own-sensing lane gate withheld a probe request), guard (env
edge-end deferral), other.

Ledger: <root>/../ledger/CHARGES.jsonl (one phase ledger shared by every attempt root), ALLOC row per
job (full episode sim-s) under flock BEFORE SUMO;
a job directory that exists is never re-run; failures stay charged with a receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

SCHEMA = 'lcexec-smoke-v1'
STUDY = 'S-LCEXEC-SMOKE'
PHASE = 'lcexec_smoke'
PHASE_CAP_SIM_S = 20_000
MAP = 'wz3to2'
WARMUP_S = 300
END_S = 1050
MOMENTS = (450, 600, 750, 900)
FAMILIES = ('wall_fill', 'lane_redistribute', 'rolling_wall', 'gom_1', 'gom_multi', 'free_program')
WORLDS = tuple([dict(seed=989060+i, demand=3200, penetration=.6) for i in range(8)]
               + [dict(seed=989068+i, demand=3200, penetration=.3) for i in range(4)])
CANARY = dict(seed=989072, demand=3200, penetration=.6)
FORBIDDEN_SEEDS = (989600, 989699)
MAX_WORKERS = 8
TARGET_TOL = 1e-3
WRITES = ('slowDown', 'setSpeed', 'changeLane', 'setLaneChangeMode', 'setTau')
FORBIDDEN = ('holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'sealed')
DEFAULT_RUNTIME = '/root/mining20261106/cpuv_home/e3final'
DEFAULT_CHECKPOINT = '/root/wzwmdata20261106/inputs/policy_00169.pt'


def _write(path, value, mode='x'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding='utf-8') as f:
        f.write(json.dumps(value, sort_keys=True, default=_json) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _json(x):
    if isinstance(x, (set, frozenset, tuple)):
        return list(x)
    if hasattr(x, 'item'):
        return x.item()
    raise TypeError(type(x))


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _guard_path(p):
    if any(x in str(p).lower() for x in FORBIDDEN):
        raise ValueError(f'restricted path refused: {p}')


def cell_key(w):
    return f"D{w['demand']}_p{int(round(w['penetration']*100)):02d}"


# --------------------------------------------------------------------------- plan (zero physics)
def make_jobs(root, *, runtime, checkpoint, attempt=None):
    jobs = []
    tag = '' if attempt is None else f'_{attempt}'
    for wave, worlds in (('canary', (CANARY,)), ('main', WORLDS)):
        for w in worlds:
            if FORBIDDEN_SEEDS[0] <= w['seed'] <= FORBIDDEN_SEEDS[1]:
                raise ValueError('held-out world refused')
            world = dict(w, map=MAP, cell_key=cell_key(w))
            jid = f"{world['cell_key']}_s{w['seed']}{tag}"
            jobs.append(dict(id=jid, wave=wave, world=world, warmup_s=WARMUP_S, end_s=END_S, sim_s=END_S,
                             moments=list(MOMENTS), runtime=str(runtime), checkpoint=str(checkpoint),
                             path=str(Path(root)/'jobs'/jid)))
    return jobs


def family_order(world_index, moment_index):
    k = (world_index+moment_index) % len(FAMILIES)
    return FAMILIES[k:]+FAMILIES[:k]


def plan(root, *, runtime=DEFAULT_RUNTIME, checkpoint=DEFAULT_CHECKPOINT, commit=None, host='gpuA', attempt=None):
    from . import map_descriptor as md, map_scenarios
    root = Path(root)
    _guard_path(root)
    if root.exists():
        raise ValueError('new smoke root required; never overwrite')
    desc = md.current()
    if desc.name != MAP:
        raise ValueError(f'set {md.ENV} to the wz3to2 MAP_SPEC.json (process map is {desc.name})')
    jobs = make_jobs(root, runtime=runtime, checkpoint=checkpoint, attempt=attempt)
    for j in jobs:
        map_scenarios.check_case(j['world']['seed'], j['world']['demand'])   # registered TRAIN world
    total = sum(j['sim_s'] for j in jobs)
    if total > PHASE_CAP_SIM_S:
        raise ValueError(f'plan {total} sim-s exceeds the phase cap {PHASE_CAP_SIM_S}')
    from .lc_parity import manifest
    body = dict(schema=SCHEMA, study=STUDY, phase=PHASE, created_unix=time.time(), host=host,
                map=desc.name, map_descriptor_sha256=desc.digest, jobs=jobs, moments=list(MOMENTS),
                families=list(FAMILIES), budget=dict(jobs=len(jobs), total_sim_s=total, cap_sim_s=PHASE_CAP_SIM_S),
                probe_fixtures={k: v for k, v in manifest()['probes'].items() if k.startswith('lcmoves/')},
                source_commit=commit, runtime=str(runtime), checkpoint=str(checkpoint),
                rules='CPU only nice 19 <= 8 workers own PIDs; precharged per job, never retried; stop on any '
                      'collision/teleport (job FAILED receipt, no further jobs); HDVs never commanded; '
                      'program members bypass the MAPPO safety filter (as the probes), speedMode 27 + env guards')
    body['plan_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True, default=_json).encode()).hexdigest()
    root.mkdir(parents=True)
    (root.parent/'ledger').mkdir(exist_ok=True)       # one phase ledger for every attempt root
    _write(root/'PLAN.json', body)
    return body


def charge(root, job):
    import fcntl
    path = Path(root).parent/'ledger'/'CHARGES.jsonl'
    lock = Path(root).parent/'ledger'/'CHARGES.lock'
    with lock.open('a') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
        if any(r['id'] == job['id'] for r in rows):
            raise ValueError(f"{job['id']} already charged; never retried")
        total = sum(float(r['sim_s']) for r in rows)
        if total+job['sim_s'] > PHASE_CAP_SIM_S:
            raise ValueError(f'phase cap: {total}+{job["sim_s"]}')
        row = dict(phase=PHASE, study=STUDY, kind='ALLOC', id=job['id'], sim_s=job['sim_s'], unix=time.time(),
                   pid=os.getpid(), cumulative_sim_s=total+job['sim_s'])
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, sort_keys=True)+'\n')
            f.flush()
            os.fsync(f.fileno())
    return row


# --------------------------------------------------------------------------- TraCI shadow + spy
class _ShadowVehicle:
    """Reads pass through to the live TraCI vehicle domain; writes are recorded, not executed."""

    def __init__(self, real, log):
        self._real, self._log = real, log

    def __getattr__(self, name):
        return getattr(self._real, name)

    def slowDown(self, vid, speed, duration):
        self._log.append(('slowDown', vid, (float(speed), float(duration))))

    def setSpeed(self, vid, speed):
        self._log.append(('setSpeed', vid, float(speed)))

    def changeLane(self, vid, index, duration):
        self._log.append(('changeLane', vid, (int(index), float(duration))))

    def setLaneChangeMode(self, vid, mode):
        self._log.append(('setLaneChangeMode', vid, int(mode)))

    def setTau(self, vid, tau):
        self._log.append(('setTau', vid, float(tau)))


class ShadowApi:
    def __init__(self, real):
        self.log = []
        self.vehicle = _ShadowVehicle(real.vehicle, self.log)
        self.lane = real.lane
        self.simulation = real.simulation

    def take(self):
        out = list(self.log)
        self.log.clear()
        return out


class WriteSpy:
    """Pass-through recorder of the env's real TraCI writes (installed for the episode only)."""

    def __init__(self, real):
        self.real, self.log, self.saved = real, [], {}

    def install(self):
        vehicle = self.real.vehicle
        for name in WRITES:
            original = getattr(vehicle, name)
            self.saved[name] = original

            def wrapped(*args, _name=name, _orig=original):
                vid = args[0]
                value = args[1:]
                if _name in ('slowDown', 'changeLane'):
                    value = (float(value[0]), float(value[1])) if _name == 'slowDown' else (int(value[0]), float(value[1]))
                elif _name == 'setLaneChangeMode':
                    value = int(value[0])
                else:
                    value = float(value[0])
                self.log.append((_name, vid, value))
                return _orig(*args)
            setattr(vehicle, name, wrapped)
        return self

    def uninstall(self):
        for name, original in self.saved.items():
            setattr(self.real.vehicle, name, original)
        self.saved.clear()

    def take(self):
        out = list(self.log)
        self.log.clear()
        return out


# --------------------------------------------------------------------------- parity (pure)
def _same(kind, a, b):
    if kind == 'slowDown':
        return abs(a[0]-b[0]) <= TARGET_TOL and abs(a[1]-b[1]) <= 1e-9
    if kind == 'setTau':
        return abs(a-b) <= 1e-9
    return a == b


def compare_member(probe_by_el, ours_by_el, program_cmd_els, released_el=None):
    """Per-second comparison of one member's probe vs our calls ({el: [(kind, value)]}).

    ``program_cmd_els`` = seconds at which OUR program issued a speed command (else our slowDowns
    are the base controller). Returns dict(match, mismatches[], normalised Counter, max_target_dev)."""
    norm, mismatches, matched, dev = Counter(), [], 0, 0.
    els = sorted(set(probe_by_el) | set(ours_by_el))
    probe_speed = {el for el, calls in probe_by_el.items() if any(k == 'slowDown' for k, _ in calls)}
    for el in els:
        p = list(probe_by_el.get(el, ()))
        o = list(ours_by_el.get(el, ()))
        base = [c for c in o if c[0] == 'slowDown' and el not in program_cmd_els]
        if base:
            o = [c for c in o if c not in base]
            norm['base_slowdown_dropped'] += len(base)
            if any(k == 'setSpeed' and v == -1. for k, v in p) and el-1 in probe_speed:
                p = [c for c in p if not (c[0] == 'setSpeed' and c[1] == -1.)]
                norm['N3_release_handback'] += 1
        if any(k == 'setSpeed' and v == -1. for k, v in p) and el-1 not in probe_speed:
            p = [c for c in p if not (c[0] == 'setSpeed' and c[1] == -1.)]
            norm['N1_probe_noop_release'] += 1
        if any(k == 'setSpeed' and v == -1. for k, v in o) and not any(k == 'setSpeed' for k, _ in p):
            o = [c for c in o if not (c[0] == 'setSpeed' and c[1] == -1.)]
            norm['N2_ours_release_equivalent'] += 1
        left = list(o)
        for kind, value in p:
            hit = next((c for c in left if c[0] == kind and _same(kind, value, c[1])), None)
            if hit is None:
                mismatches.append(dict(el=el, side='probe_only', kind=kind, value=value))
                continue
            if kind == 'slowDown':
                dev = max(dev, abs(value[0]-hit[1][0]))
            left.remove(hit)
            matched += 1
        for kind, value in left:
            mismatches.append(dict(el=el, side='ours_only', kind=kind, value=value))
    return dict(matched=matched, mismatches=mismatches, normalised=dict(norm), max_target_dev=dev)


def classify(mismatch, gate_els, guard_els):
    """gate / guard / other for one mismatch (receipts of our lane gate and env writer)."""
    if mismatch['kind'] == 'changeLane' and mismatch['side'] == 'probe_only':
        if mismatch['el'] in gate_els:
            return 'gate'
        if mismatch['el'] in guard_els:
            return 'guard'
    return 'other'


# --------------------------------------------------------------------------- one episode (SUMO)
def _mask_executor(exe, env, previous):
    import numpy as np
    import torch
    ids = np.asarray(env.frame.vid_index)
    keep = np.zeros(len(ids), bool) if previous is None else (ids == previous) & (ids >= 0)
    for obj in (exe.memory, exe.dwell):
        if obj.present is not None:
            obj.present = obj.present & torch.as_tensor(keep, device=obj.present.device)[None]
    return ids.copy()


class _Commit:
    """MoveExecutor stand-in: the smoke starts every member of a move at the move second."""

    def start(self, move, *, ego, observable_cav_ids, now, until):
        return True


def _rows(env):
    from tools.leverprobe.backend import observations
    return [dict(r, length_m=float(env.api.vehicle.getLength(r['vid']))) for r in observations(env)]


def _reports(table, cavs, now):
    from .merge_features import ObservedVehicle, SensorReport, SENSOR_RANGE_M
    out = {}
    vehicle = lambda r: ObservedVehicle(r['vid'], r['s_m'], int(r['lane']), r['v_mps'], r['length_m'])  # noqa: E731
    for vid in cavs:
        ego = table[vid]
        if int(ego['lane']) not in (0, 1, 2):
            continue
        sensed = [r for r in table.values() if r['vid'] != vid and abs(r['s_m']-ego['s_m']) <= SENSOR_RANGE_M
                  and int(r['lane']) in (0, 1, 2)]
        out[vid] = SensorReport(float(now), vehicle(ego), (0, 1, 2), tuple(vehicle(r) for r in sensed))
    return out


def run_episode(job, root):
    import importlib
    import random
    import numpy as np
    import torch
    from tools.leverprobe import backend as exact
    exact.setup(job['runtime'], job['checkpoint'])
    from . import g5_runtime, lc_runtime, map_runtime, lc_program as lp, lc_roles, lc_families as lf
    from .merge_features import SENSOR_RANGE_M
    if not g5_runtime.enabled() or not lc_runtime.enabled():
        raise ValueError('COOPMWM_G5_RUNTIME=1 and COOPMWM_LC_RUNTIME=1 required')
    g5_runtime.install(job['runtime'])
    desc = map_runtime.install(job)
    if desc is None or desc.name != MAP:
        raise ValueError('wz3to2 map binding required')
    lc_runtime.install(job)
    import traci
    env_mod = importlib.import_module('traffic_shepherd.c4.env')
    if getattr(env_mod.CavSumoEnv, 'LC_RUNTIME', None) is None:
        raise ValueError('LCEXEC env subclass not bound')
    from .lc_parity import probe
    P = probe('lcmoves')
    F = P.families
    frame = lp.frame_from_descriptor(desc)
    world = job['world']
    random.seed(world['seed'])
    np.random.seed(world['seed'])
    torch.manual_seed(world['seed'])
    original, used = traci.simulationStep, [0.]

    def counted(*args, **kwargs):
        if used[0]+.5 > job['sim_s']+1e-9:
            raise ValueError('step exceeds precharge')
        before = float(traci.simulation.getTime())
        out = original(*args, **kwargs)
        used[0] += float(traci.simulation.getTime())-before
        return out
    traci.simulationStep = counted
    path = Path(job['path'])
    env = exe = spy = None
    cfg = dict(group_lane_moves=True, **lf.config_keys(provenance=f'{STUDY} smoke (MAIN 23:3x)', cap=6))
    proto = lf.Protocol(lf.settings(cfg))
    moves, active, reports_prev = [], {}, {}
    totals = Counter()
    window = defaultdict(Counter)
    try:
        from wms.control.policy import load_policy_fn
        exe = load_policy_fn(str(job['checkpoint']), device='cpu')
        if not exe.native_lanes or exe.residual_on_native:
            raise ValueError('speed-only direct MAPPO required')
        exe.reset()
        env = env_mod.CavSumoEnv(k_slots=192, controller='direct', ttc=True, scenario_root=path/'tmp_scenarios',
                                 physics_log=False, population='textbook')
        env.reset(seed=world['seed'], demand_vph=world['demand'], penetration=world['penetration'],
                  warmup_s=job['warmup_s'], horizon_s=job['end_s']-job['warmup_s'], controller='direct',
                  episode_dir=None, meta=dict(study=STUDY, job=job['id']))
        spy = WriteSpy(traci).install()
        k = env.k
        previous = None
        world_index = next((i for i, w in enumerate(WORLDS) if w['seed'] == world['seed']), 0)
        for t in range(int(job['warmup_s']), int(job['end_s'])):
            previous = _mask_executor(exe, env, previous)
            out, lane_out = exe(env.state(), with_lanes=True)
            if lane_out is not None:
                raise ValueError('native lanes required')
            cmd = out.detach().cpu().numpy().astype(np.float64).reshape(1, k) if torch.is_tensor(out) \
                else np.asarray(out, np.float64).reshape(1, k)
            rows = _rows(env)
            table = {r['vid']: r for r in rows}
            cavs = sorted(v for v, r in table.items() if r['is_cav'])
            live = set(traci.vehicle.getIDList())
            # ---- new move at a registered moment (probe's own family builders on the live snapshot)
            if t in job['moments']:
                busy = {v for m in moves if not m['done'] for v in m['ids']}
                snap = F.snapshot(traci, sorted(env.cav_ids & live))
                chosen = None
                for family in family_order(world_index, job['moments'].index(t)):
                    members = F.propose(snap, family, sid=f"lcexec_smoke:{world['seed']}:{t}", i=0)
                    if not members:
                        continue
                    ids = [m['id'] for m in members]
                    if busy & set(ids) or any(v not in table or not table[v]['is_cav'] for v in ids):
                        continue
                    chosen = (family, members)
                    break
                refusal = None
                if chosen is not None:
                    family, members = chosen
                    ids = [m['id'] for m in members]
                    sid = f'free_program:{ids[0]}@{t}'
                    try:
                        programs = tuple(lc_roles.member_program(m['role'], m['params'], frame) for m in members)
                        hold = max(int(p.hold_s or (job['end_s']-t)) for p in programs)
                        fp = lp.FreeProgram(tuple(ids), programs, MAP, (0,)*len(ids),
                                            max(1, min(hold, job['end_s']-t)), dict(study=STUDY, family=family, t0=t))
                        template = lf.template_from_program(fp, frame, table)
                        proto.receive(t, {v: table[v] for v in cavs},
                                      {v: [r for r in rows if abs(r['s_m']-table[v]['s_m']) <= SENSOR_RANGE_M]
                                       for v in cavs}, reports_prev)
                        options = []
                        for slot, vid in enumerate(ids):
                            kind = 'initiate' if slot == 0 else 'join'
                            body = (template.to_dict() if slot == 0 else proto.sessions[sid]['template_dict'])
                            option = lf.ProgramOption(
                                f'free_program_{kind}', (vid,), sid, ids[0], slot, body, 0,
                                int(template.duration_s), 0., max(1, slot), proto.cap, 0., proto.range_m,
                                int(table[vid]['lane']))
                            if not proto.start(option, ego=vid, executor=_Commit(), observable_cav_ids=[vid], now=t):
                                raise ValueError(f'protocol refused slot {slot} ({vid})')
                            options.append((vid, option))
                    except ValueError as exc:
                        refusal = repr(exc)
                        for v in ids:                     # never leave a half-started session behind
                            if proto.membership.get(v) == sid:
                                del proto.membership[v]
                        proto.sessions.pop(sid, None)
                if chosen is not None and refusal is not None:
                    moves.append(dict(family=chosen[0], t0=t, ids=[], done=True, refused=refusal,
                                      members=chosen[1]))
                elif chosen is not None:
                    shadow = ShadowApi(traci)
                    ref_exec = F.Executor(shadow, [dict(m) for m in members], frozenset(env.cav_ids))
                    moves.append(dict(family=family, t0=t, ids=ids, members=members, sid=sid,
                                      options=dict(options), shadow=shadow, executor=ref_exec, done=False,
                                      probe=defaultdict(list), ours=defaultdict(list), prog_els=defaultdict(set),
                                      gate=defaultdict(set), guard=defaultdict(set), lanes={}, end=None,
                                      ours_release={}, lanes_t0={v: int(table[v]['lane']) for v in ids}))
                    for v, o in options:
                        active[v] = o
                else:
                    moves.append(dict(family=None, t0=t, ids=[], done=True, infeasible=True))
            # ---- shadow probe (reads the live state BEFORE any write of this second)
            for m in moves:
                if m['done']:
                    continue
                el = t-m['t0']
                m['executor'].step(el)
                for kind, vid, value in m['shadow'].take():
                    m['probe'][(vid, el)].append((kind, value))
                if not m['executor'].active():
                    m['end'] = el
            # ---- our production path
            tick = {}
            if active:
                sensing = {v: [r for r in rows if abs(r['s_m']-table[v]['s_m']) <= SENSOR_RANGE_M] for v in cavs}
                proto.receive(t, {v: table[v] for v in cavs}, sensing, reports_prev)
                live_active = [(v, o) for v, o in active.items() if v in table]
                tick = proto.commands(t, live_active)
                for v in proto.just_released:
                    active.pop(v, None)
                for v in [v for v in active if v not in table]:
                    active.pop(v, None)
                if tick:
                    env.lc_tick(tick)
            reports_prev = _reports(table, cavs, t)
            spy.take()
            _, info = env.step(cmd)
            writes = spy.take()
            iv = info['interval']
            step_totals = dict(collisions=float(iv.get('collision_involvements', 0)),
                               teleports=float(iv.get('starting_teleports', 0)),
                               hard_brake_vs=float(iv.get('braking_vehicle_seconds', 0.)),
                               ttc_below_3_vs=float(iv.get('ttc_below_3_vehicle_seconds', 0.)),
                               dropped_vehicle_steps=float(iv.get('dropped_vehicle_steps', 0)))
            totals.update(step_totals)
            receipts = info.get('lc_lane_vehicles') or {}
            for m in moves:
                if m['done']:
                    continue
                el = t-m['t0']
                window[m['sid']].update(step_totals)
                for kind, vid, value in writes:
                    if vid in m['ids']:
                        m['ours'][(vid, el)].append((kind, value))
                for vid in m['ids']:
                    if vid in (tick.get('accel') or {}):
                        m['prog_els'][vid].add(el)
                    gate = (proto.last_gate or {}).get(vid)
                    if gate is not None:
                        m['gate'][vid].add(el)
                    rec = receipts.get(vid)
                    if rec is not None and rec.get('status') == 'deferred_edge_end':
                        m['guard'][vid].add(el)
                    if rec is not None and rec.get('status') in ('issued', 'held'):
                        m['lanes'].setdefault(vid, []).append((el, rec['target'], rec['status'], rec['completed']))
                for lab in proto.last_labels or ():
                    if lab['vid'] in m['ids'] and lab['session_id'] == m['sid'] and lab['event'] in ('release', 'left'):
                        m['ours_release'].setdefault(lab['vid'], (el, lab['release_reason'] or lab['event']))
                if m['end'] is not None:
                    m['done'] = True
            if step_totals['collisions'] or step_totals['teleports']:
                raise ValueError(f'collision/teleport at {t}: {step_totals} (stop rule)')
            if info['done']:
                break
        summaries = []
        for m in moves:
            if m.get('infeasible'):
                summaries.append(dict(t0=m['t0'], family=None, infeasible=True))
                continue
            if m.get('refused'):
                summaries.append(dict(t0=m['t0'], family=m['family'], refused=m['refused'],
                                      roles=[x['role'] for x in m['members']]))
                continue
            per_member = {}
            for vid in m['ids']:
                probe_by_el = defaultdict(list)
                ours_by_el = defaultdict(list)
                for (v, el), calls in m['probe'].items():
                    if v == vid:
                        probe_by_el[el] += calls
                for (v, el), calls in m['ours'].items():
                    if v == vid:
                        ours_by_el[el] += calls
                last = m['end'] if m['end'] is not None else max([el for (v, el) in m['probe']] or [0])
                ours_by_el = {el: c for el, c in ours_by_el.items() if el <= last}
                res = compare_member(probe_by_el, ours_by_el, m['prog_els'][vid])
                res['classes'] = dict(Counter(classify(x, m['gate'][vid], m['guard'][vid])
                                              for x in res['mismatches']))
                res['mismatches'] = res['mismatches'][:20]
                spec = next(x for x in m['members'] if x['id'] == vid)
                p = spec['params']
                target = p.get('lc_target', p.get('target_lane'))
                res.update(role=spec['role'], lane_t0_wm=m['lanes_t0'][vid],
                           lane_target_wm=None if target is None else frame.wm(int(target)),
                           lane_requests=len(m['lanes'].get(vid, ())),
                           lane_reached=any(c for (_, tg, st, c) in m['lanes'].get(vid, ()) if st == 'issued'),
                           held_seconds=sum(1 for (_, tg, st, c) in m['lanes'].get(vid, ()) if st == 'held' and c),
                           gate_blocked_seconds=len(m['gate'][vid]), guard_seconds=len(m['guard'][vid]),
                           probe_lc_done=m['executor'].log['lc_done'].get(vid),
                           probe_release_el=m['executor'].log['release_el'].get(vid),
                           ours_release=m['ours_release'].get(vid))
                per_member[vid] = res
            matched = sum(r['matched'] for r in per_member.values())
            classes = Counter()
            for r in per_member.values():
                classes.update(r['classes'])
            closing = [v for v, r in per_member.items() if r['lane_target_wm'] == frame.closing_lane]
            summaries.append(dict(
                t0=m['t0'], family=m['family'], members=m['ids'], roles=[x['role'] for x in m['members']],
                end_el=m['end'], matched=matched, mismatch_classes=dict(classes),
                parity_exact=sum(classes.values()) == 0,
                max_target_dev=max([r['max_target_dev'] for r in per_member.values()] or [0.]),
                closing_lane_members=closing,
                closing_lane_entered=[v for v in closing if per_member[v]['lane_reached'] or per_member[v]['held_seconds']],
                window=dict(window[m['sid']]), per_member=per_member,
                params={x['id']: x['params'] for x in m['members']}))
        return dict(totals=dict(totals), moves=summaries, actual_sim_s=used[0],
                    lc_runtime=lc_runtime.installed_record(), map_descriptor_sha256=desc.digest,
                    design=dict(population=len(env.requests), designated_cavs=len(env.cav_ids)))
    finally:
        try:
            if spy is not None:
                spy.uninstall()
            if env is not None:
                env.close()
        finally:
            traci.simulationStep = original


def run_job(arg):
    job, root = arg
    root = Path(root)
    path = Path(job['path'])
    if path.exists():
        return dict(id=job['id'], status='skipped_exists')
    if (root/'STOP').exists():
        return dict(id=job['id'], status='stopped')
    try:
        row = charge(root, job)
    except ValueError as exc:
        return dict(id=job['id'], status='refused', error=repr(exc))
    path.mkdir(parents=True, exist_ok=False)
    _write(path/'CLAIM.json', dict(pid=os.getpid(), job=job['id'], charged_sim_s=job['sim_s'], ledger_row=row,
                                   unix=time.time()))
    started = time.time()
    try:
        out = run_episode(job, root)
        out.update(id=job['id'], world=job['world'], status='ok', wall_s=round(time.time()-started, 2),
                   charged_sim_s=job['sim_s'])
        _write(path/'RESULT.json', out)
        return dict(id=job['id'], status='ok', wall_s=out['wall_s'])
    except Exception as exc:                                   # failure receipt, never retried
        import traceback
        _write(path/'FAILED.json', dict(id=job['id'], world=job['world'], error=repr(exc),
                                        traceback=traceback.format_exc(), charged_sim_s=job['sim_s'],
                                        wall_s=round(time.time()-started, 2)))
        if 'collision/teleport' in repr(exc):
            _write(root/'STOP', dict(reason=repr(exc), job=job['id'], unix=time.time()))
        return dict(id=job['id'], status='failed', error=repr(exc))


def run(root, *, workers=MAX_WORKERS, only=None):
    import multiprocessing
    root = Path(root)
    _guard_path(root)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise ValueError('run under nice 19 with CUDA_VISIBLE_DEVICES= (CPU only)')
    if not 1 <= int(workers) <= MAX_WORKERS:
        raise ValueError(f'workers must be 1..{MAX_WORKERS}')
    plan_body = _read(root/'PLAN.json')
    jobs = [j for j in plan_body['jobs'] if only is None or j['wave'] == only]
    ctx = multiprocessing.get_context('spawn')
    results = []
    with ctx.Pool(int(workers), maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(run_job, [(j, str(root)) for j in jobs]):
            results.append(res)
            print(json.dumps(res), flush=True)
    return results


def summarize(root):
    root = Path(root)
    plan_body = _read(root/'PLAN.json')
    out = dict(schema=SCHEMA+'.summary', study=STUDY, jobs=[], totals=Counter(), families=defaultdict(Counter))
    ledger = root.parent/'ledger'/'CHARGES.jsonl'
    charged = sum(float(json.loads(s)['sim_s']) for s in ledger.read_text().splitlines() if s.strip()) \
        if ledger.exists() else 0.
    for job in plan_body['jobs']:
        path = Path(job['path'])
        if (path/'RESULT.json').exists():
            r = _read(path/'RESULT.json')
            out['totals'].update(r['totals'])
            for m in r['moves']:
                fam = m['family'] or 'infeasible'
                c = out['families'][fam]
                c['moves'] += 1
                if m.get('refused'):
                    c['refused'] += 1
                    continue
                if m['family']:
                    c['parity_exact'] += int(m['parity_exact'])
                    c['matched_calls'] += m['matched']
                    for k, v in m['mismatch_classes'].items():
                        c['mismatch_'+k] += v
                    c['closing_lane_members'] += len(m['closing_lane_members'])
                    c['closing_lane_entered'] += len(m['closing_lane_entered'])
                    for r_ in m['per_member'].values():
                        if r_['lane_target_wm'] is not None and r_['lane_target_wm'] != r_['lane_t0_wm']:
                            c['lane_members'] += 1
                            c['lane_reached'] += int(bool(r_['lane_reached'] or r_['held_seconds']))
                            c['probe_lc_done'] += int(r_['probe_lc_done'] is not None)
            out['jobs'].append(dict(id=job['id'], status='ok', totals=r['totals'], actual_sim_s=r['actual_sim_s'],
                                    moves=[dict(t0=m['t0'], family=m['family'],
                                                parity_exact=m.get('parity_exact'),
                                                classes=m.get('mismatch_classes'),
                                                closing=m.get('closing_lane_entered'),
                                                max_target_dev=m.get('max_target_dev')) for m in r['moves']]))
        elif (path/'FAILED.json').exists():
            out['jobs'].append(dict(id=job['id'], status='failed', error=_read(path/'FAILED.json')['error']))
        else:
            out['jobs'].append(dict(id=job['id'], status='not_run'))
    out['charged_sim_s'] = charged
    out['totals'] = dict(out['totals'])
    out['families'] = {k: dict(v) for k, v in out['families'].items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--root', required=True)
    p.add_argument('--runtime', default=DEFAULT_RUNTIME)
    p.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    p.add_argument('--commit')
    p.add_argument('--attempt')
    r = sub.add_parser('run')
    r.add_argument('--root', required=True)
    r.add_argument('--workers', type=int, default=MAX_WORKERS)
    r.add_argument('--only', choices=('canary', 'main'))
    s = sub.add_parser('summarize')
    s.add_argument('--root', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'plan':
        body = plan(a.root, runtime=a.runtime, checkpoint=a.checkpoint, commit=a.commit, attempt=a.attempt)
        print(json.dumps(dict(plan_sha256=body['plan_sha256'], budget=body['budget'])))
    elif a.cmd == 'run':
        run(a.root, workers=a.workers, only=a.only)
    else:
        print(json.dumps(summarize(a.root), indent=1, default=_json))


if __name__ == '__main__':
    main()
