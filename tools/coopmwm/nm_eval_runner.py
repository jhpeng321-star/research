"""wz3to2 final-evaluation runner (N6) for ``ops/nm_finaleval.py`` on the clean4 deployed source.

Contract: ``ops/nm_finaleval.RUNNER_CONTRACT``. One process = one map: every worker needs
``COOPMWM_MAP_SPEC`` = the staged wz3to2 MAP_SPEC.json (N1 ``map_descriptor``) and
``COOPMWM_G5_RUNTIME=1`` like the clean4 owners; the launcher sets nothing else.

Training parity (FAIRAUDIT 2026-11-06): A/B/R are evaluated with the SAME policy objects and the
SAME controller wrapper as the clean4 training episodes (``track2_backend.episode_components``):
* A: ``v2v_runtime.from_weights(weights, enabled=True)`` exactly as ``accel_plant`` builds it;
* B/R: ``EpisodePolicy(base, backend_config, arm)`` + ``restore_worker`` of the FULL saved worker
  state (layer, trigger and, when N2's RSU option is on, the RSU head - ``launch_production.
  load_policy`` would drop it), then layer/candidate/RSU streams reseeded from the job noise seed;
* every control second ``policy.observe(rows, now, command_commitments=controller.active_plans,
  native_commands=commands)`` when ``policy.lane_drop`` is set (N2 speed law needs each CAV's own
  native command), else without ``native_commands`` - byte-for-byte the training wrapper;
* the plant, v2v controller class (incl. N9 ``observe_native``), G5 lane path, RSU block and map
  binding are the same ``accel_plant.episode``/``bind_runtime`` code; deterministic like the
  training waves (``accel_runner.wave`` default). Differences kept on purpose: role final_test
  (held-out worlds), noise seed = world seed, no training-only telemetry/health/abort-on-hard-event
  flags (events are counted, not aborted - the registered clean3 final-eval convention).

RULE (MAIN registration 2026-11-06, pre-results): the U0 episode with the EXACT S-REGIMEU0 rule
(``tools.coopmwm.regimeu0.RuleHook('wall_h_solo2')``, byte-identical to c4/imitreplay) wrapped on the
controller after the U0 hook: one CAV in SUMO lane 2 holds at 15 m/s for 90 s (cooldown = hold),
through the plant's single writer and the G5 lane path; the rule source digest (``rule_digest``) is
checked against the spec and recorded.

Comparators (MAIN F6 + clean5a registration): U0 = the transferred MAPPO u0: the bound frozen
d9042cc5 executor (path + sha256, re-verified) with the run's common u0 (zero residual). MAPPO_SPEED = the bound speed-only
executor (``bind-mappo``: path + sha256, re-verified here) replacing the plant's frozen executor;
``tools.leverprobe.backend.setup`` pins d9042cc5, so for a bound non-d9042cc5 comparator ONLY that
pin is swapped, inside this worker, for the bound sha (``comparator_setup``: same function code,
every other frozen-runtime check unchanged; the training pin is never touched). MAPPO_LANE = the
bound lane-capable executor driving every CAV's acceleration AND lane target through the env's
existing lane path (the reviewed clean3 lane-on loop, c4/laneon-eval-claude e4596e74, on the same
N1-bound plant; the plant's own frozen executor stays d9042cc5). Records carry the sha loaded.

NATIVE: the qualified true-native plant (``tools.coopnative.plant``) with the N1 map install +
CAV-report rows; J = its ``metrics['900']['team_delay_vs']`` (identical reducer).

Decisions (B/R): descriptive capture only (every new 2+-member cooperative move); no
``decision_branch`` (a faithful branch needs the MoveExecutor state restored inside the CRN
prefix replay; not built, never faked).

F11 parity (FAIRAUDIT, registered): ``parity(spec)`` traces one bound MAPPO comparator on a TRAIN
world (role 'check', native 0-300 s warmup + ``rollout_s`` controlled seconds) either through the
evaluation plant (``path='plant'``: exactly the MAPPO_SPEED / MAPPO_LANE episode code) or through its
TRAINING env path (``path='train'``: N7 ``tools.nm_mappo.sanity_eval`` resolve/build_env/load_actor in a
fresh subprocess, as the trainer's rollout workers). Each tick records every CAV's (s, v, lane, command,
lane target); the launcher compares the two traces (zero physics). Descriptive; never a gate.

Ground-truth vehicle log (MAIN 2026-11-06, pre-results secondaries): every episode of every arm writes
``vehicles.jsonl.gz`` - EVALUATION-ONLY, read-only (never fed to any policy, observation or controller): per
control second t in [300, 1200) every present vehicle's s, v, a (frame kinematics) and lane; per vehicle once
its route origin edge, SUMO depart delay and the CAV/HDV fleet label. No driver parameters, no vType truth
(lengths are not logged; analyses use the nominal 5 m). Logged before the step, like the observable trace.

HDVs are never commanded. Never run locally except ``setup-check`` (zero SUMO steps).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import types

MAP = 'wz3to2'
NATIVE = True
ROLE = 'final_test'
HORIZON_S = 1200
WARMUP_S = 300
IDENTITY = ('arm', 'world_id', 'world', 'checkpoint', 'window', 'panel_sha256')
LOG_LIMIT = 200
MAPPO_LANE = True
#: amendment 17 (N7 constraint): MAPPO_LANE runs ONLY on the S-MAPPOWZL training execution path, never a native-lane
#: or G5-tensor executor: tools.nm_mappo.sanity_eval resolve(config) -> wms build_env (n1_adapter.make_env with
#: lane_writer='lcexec') -> load_actor(lane_actions=LCEXEC_LANE_ACTIONS) = LaneActor (ActorStep, deterministic)
LANE_TRAIN_PATH = True
LCEXEC_LANE_ACTIONS = 'tools.nm_mappo.lane_actions:lcexec'
LANE_TRAIN_FILES = ('tools/nm_mappo/sanity_eval.py', 'tools/nm_mappo/lane_actions.py', 'tools/nm_mappo/n1_adapter.py',
                    'src/wms/control/mappo/agent.py', 'src/wms/control/mappo/rollout.py', 'tools/coopmwm/lc_runtime.py',
                    'tools/coopmwm/map_observations.py', 'tools/coopmwm/replay_bank.py')
RULE = True
RULE_STRATEGIES = ('wall_h_solo2', 'wall_g_solo2', 'wall_g_duo12')
#: strategy -> the module holding its EXACT rule code (RULE: S-REGIMEU0; RULE_FAIR: S-MULTICRIT byte copy)
RULE_MODULES = dict(wall_h_solo2='regimeu0', wall_g_solo2='regimeu0_multicrit',
                    wall_g_duo12='regimeu0_multicrit')   # amendment 22: the p20 RULE (S-MULTICRIT-LOWCAV rule)
PARITY = True
PARITY_ROLE = 'check'
LANE_GRID = (-1, 0, 1, 2)
LANE_CHANGE_MODE = 513
FROZEN_MAPPO_SHA = 'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'


#: amendment 23: evaluation-only held-out range (fresh integers for the p20x panels). Registered in this process's
#: tools.coopmwm.map_scenarios at import: seed_scope -> 'heldout', repeat = seed - 993000. Physics-neutral: the
#: N1 world dict only carries seed/cell/repeat labels; TRAIN roles still refuse these seeds (guard).
EVAL_HELDOUT_RANGE = (993000, 993099)


def install_eval_ranges(ms=None):
    """Idempotent: add EVAL_HELDOUT_RANGE to map_scenarios (seed_scope, repeat_index if present, world)."""
    if ms is None:
        try:
            from . import map_scenarios as ms
        except ImportError:
            return False
    if getattr(ms, '_N6_EVAL_RANGES', None) == EVAL_HELDOUT_RANGE:
        return True
    lo, hi = EVAL_HELDOUT_RANGE
    scope0, world0 = ms.seed_scope, ms.world

    def seed_scope(seed):
        if type(seed) is int and lo <= seed <= hi:
            return 'heldout'
        return scope0(seed)

    def world(seed, demand, penetration):
        if not (type(seed) is int and lo <= seed <= hi):
            return world0(seed, demand, penetration)
        cell = next((c for c in ms.cells() if c['demand'] == demand and c['penetration'] == penetration), None)
        if cell is None or type(demand) is not int:
            raise ValueError(f'({demand!r}, {penetration!r}) is not a registered {ms.MAP} cell')
        return dict(seed=seed, cell=cell['cell'], repeat=seed-lo, penetration=penetration, demand=demand,
                    map=ms.MAP, cell_key=cell['cell_key'])
    ms.seed_scope, ms.world = seed_scope, world
    if hasattr(ms, 'repeat_index'):
        rep0 = ms.repeat_index
        ms.repeat_index = lambda seed: seed-lo if type(seed) is int and lo <= seed <= hi else rep0(seed)
    ms._N6_EVAL_RANGES = EVAL_HELDOUT_RANGE
    return True


install_eval_ranges()


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _check_spec(spec):
    """Registered held-out map world in a process bound to the same map (fail closed)."""
    from . import map_descriptor as md, map_scenarios as ms
    world = spec['world']
    ms.guard(world, ROLE)
    if md.for_world(world).name != MAP:
        raise ValueError(f'process map is not {MAP}')
    if spec['window'] != dict(start_s=0, end_s=HORIZON_S) or int(spec['sim_s']) != HORIZON_S:
        raise ValueError('full 1200 s evaluation window required')
    if spec['noise_seed'] != world['seed'] or spec.get('deterministic') is not True:
        raise ValueError('deterministic evaluation with noise seed = world seed required')
    return world


def _record(spec, J, safety, **extra):
    return dict({k: deepcopy(spec[k]) for k in IDENTITY}, status='graded', J_veh_s=float(J),
                safety={k: int(v) for k, v in safety.items()}, **extra)


# ---------------------------------------------------------------- ground-truth vehicle log (evaluation only)

VEHICLE_LOG = 'vehicles.jsonl.gz'


class VehicleLog:
    """EVALUATION-ONLY ground truth, read-only: never returned to any policy/controller/observation path."""

    def __init__(self, work):
        import gzip
        self.path = Path(work)/VEHICLE_LOG
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = gzip.open(self.path, 'xt', encoding='utf-8')
        self.index, self.last, self.ticks = {}, None, 0

    def tick(self, env):
        import numpy as np
        now = int(round(float(env.now)))
        if now == self.last or now < WARMUP_S or now >= HORIZON_S:
            return
        self.last = now
        f, ids = env.frame, env.builder.vehicle_ids
        api = getattr(env, 'api', None)
        k, s, v, a, lane = [], [], [], [], []
        for i in np.flatnonzero(f.present):
            vid = ids[int(f.vid_index[i])]
            n = self.index.get(vid)
            if n is None:
                n = self.index[vid] = len(self.index)
                origin = delay = None
                if api is not None:
                    try:
                        route = api.vehicle.getRoute(vid)
                        origin = str(route[0]) if route else None
                    except Exception:      # read-only getter; absence is logged as None
                        origin = None
                    try:
                        delay = float(api.vehicle.getDepartDelay(vid))
                    except Exception:
                        delay = None
                self.stream.write(json.dumps(dict(veh=n, id=str(vid), cav=bool(f.is_cav[i]), origin=origin,
                                                  depart_delay_s=delay, first_t=now))+'\n')
            k.append(n)
            s.append(round(float(f.s[i]), 2))
            v.append(round(float(f.v[i]), 3))
            a.append(round(float(f.a[i]), 3))
            lane.append(int(f.lane_carry[i] if f.junction[i] else f.lane[i]))
        self.stream.write(json.dumps(dict(t=now, k=k, s=s, v=v, a=a, lane=lane))+'\n')
        self.ticks += 1

    def close(self):
        if not self.stream.closed:
            self.stream.close()
        return dict(file=VEHICLE_LOG, sha256=_sha(self.path), ticks=self.ticks, vehicles=len(self.index))


@contextmanager
def logged_tick(log):
    """Wrap tools.coopfree.backend.tick (the plant's 1 s step) to log the pre-step ground truth (read-only)."""
    from tools.coopfree import backend as cf
    original = cf.tick

    def tick(b, controller=None, trace=None):
        log.tick(b.env)
        return original(b, controller, trace)
    cf.tick = tick
    try:
        yield log
    finally:
        cf.tick = original


# ---------------------------------------------------------------- MAPPO comparator binding

def read_binding(path):
    """The sealed bind-mappo record (MAPPO_COMPARATOR.json)."""
    from .exact_grade import verify_sealed
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    verify_sealed(value, 'binding_sha256')
    return value


def bound_executor(ck):
    """MAPPO arm: binding seal, spec, binding and file sha must all agree; returns (path, sha)."""
    binding = read_binding(ck['binding_path'])
    if (binding['binding_sha256'] != ck['binding_sha256'] or binding['sha256'] != ck['executor_sha256']
            or binding['path'] != ck['executor_path']):
        raise ValueError('MAPPO comparator spec differs from its bound record')
    loaded = _sha(ck['executor_path'])
    if loaded != ck['executor_sha256']:
        raise ValueError('MAPPO comparator checkpoint differs from its bound sha256')
    return ck['executor_path'], loaded


@contextmanager
def comparator_setup(expected_sha):
    """``tools.leverprobe.backend.setup`` accepting exactly ``expected_sha`` as the executor.

    d9042cc5 -> the original function (unchanged path). Any other sha -> the SAME function code
    with only its ``MAPPO_SHA`` global replaced, installed for this context in this process and
    restored afterwards; the runtime manifest, file, nice/CPU, SUMO-version and deterministic
    checks are untouched, and any other checkpoint still raises."""
    from tools.leverprobe import backend as lp
    original = lp.setup
    if expected_sha == lp.MAPPO_SHA:
        yield original
        return
    shim = types.FunctionType(original.__code__, dict(original.__globals__, MAPPO_SHA=expected_sha),
                              original.__name__, original.__defaults__, original.__closure__)
    lp.setup = shim
    try:
        yield shim
    finally:
        lp.setup = original


# ---------------------------------------------------------------- RULE (exact S-REGIMEU0)

RULE_CONSTANTS = ('X_B', 'S_OFF', 'ZONE_S', 'WALL_SPREAD', 'RELEASE_S', 'WALL_H', 'STRATEGIES')
#: S-MULTICRIT rule: everything RuleHook reads (variant parameters, solo-lane and start-gate helpers)
MULTICRIT_FUNCTIONS = ('sumo_lane', 'pick_wall', 'solo_lanes', 'behind_count', 'RuleHook')
MULTICRIT_CONSTANTS = ('X_B', 'S_OFF', 'ZONE_S', 'WALL_SPREAD', 'RELEASE_S', 'WALL_H', 'WALL_G', 'WALL_M',
                       'VARIANT_PARAMS', 'STRATEGIES', 'GATES')


def rule_digest(module=None):
    """sha256 of the S-REGIMEU0 rule as executed: source of sumo_lane/pick_wall/RuleHook plus the
    Recorder base and every rule constant (a file edit elsewhere in regimeu0.py does not move it)."""
    import inspect
    if module is None:
        from . import regimeu0 as module
    parts = [inspect.getsource(getattr(module, n)) for n in ('sumo_lane', 'pick_wall', 'RuleHook')]
    parts.append(inspect.getsource(module.ir.Recorder))
    parts.append(json.dumps({n: getattr(module, n) for n in RULE_CONSTANTS}, sort_keys=True, default=list))
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()


def multicrit_digest(module=None):
    """sha256 of the S-MULTICRIT rule as executed (RULE_FAIR): every function and constant RuleHook reads."""
    import importlib
    import inspect
    module = module or importlib.import_module(f'{__package__}.regimeu0_multicrit')
    parts = [inspect.getsource(getattr(module, n)) for n in MULTICRIT_FUNCTIONS]
    parts.append(inspect.getsource(module.ir.Recorder))
    parts.append(json.dumps({n: getattr(module, n) for n in MULTICRIT_CONSTANTS}, sort_keys=True, default=list))
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()


ORACLE_ARMS = ('ORACLE', 'ORACLE_WM')          # amendment 25 / 25b B1 (N6 runner_status contract)
ORACLE_PLAN_SOURCE = dict(ORACLE='all', ORACLE_WM='wm')


def oracle_hook(ck, world, arm='ORACLE'):
    """Amendment 25/25b ORACLE (plan_source None: every admitted row of the union bank) / ORACLE_WM (plan_source
    'wm': the WM-source rows of the same bank): oracle_plan.make_hook on the FROZEN union bank (file sha re-verified) for this world's
    cell_key (world['cell_key'] only; must equal the bound ck['cell_key']); the executed ORACLE code must equal its
    registration (oracle_digest == ck['oracle_sha256']). Returns (hook, loaded digest)."""
    from . import oracle_plan
    if arm not in ORACLE_ARMS:
        raise ValueError(f'unknown ORACLE arm {arm!r}')
    source = ck.get('oracle_plan_source')
    if source != ORACLE_PLAN_SOURCE[arm]:
        raise ValueError(f'ORACLE: ck oracle_plan_source {source!r} is not {ORACLE_PLAN_SOURCE[arm]!r} for arm {arm}')
    cell = world.get('cell_key')
    if not cell or cell != ck.get('cell_key'):
        raise ValueError('ORACLE: world cell_key missing or not the bound cell_key')
    bank = ck.get('oracle_bank') or {}
    path = Path(bank.get('path', ''))
    if not path.is_file() or _sha(path) != bank.get('file_sha256'):
        raise ValueError('ORACLE: bank changed after registration')
    loaded = oracle_plan.oracle_digest()
    if loaded != ck.get('oracle_sha256'):
        raise ValueError('ORACLE: oracle_plan differs from its registration (oracle_sha256)')
    doc = json.loads(path.read_text(encoding='utf-8'))
    return oracle_plan.make_hook(doc, cell, plan_source=None if source == 'all' else 'wm'), loaded


LEARNED_ARMS = ('LEARNED_TRIGGER',)            # TRIGCLF 2026-11-09 (N6 amendment 26): ORACLE_WM executor, learned admission


def learned_trigger_hook(ck, world):
    """LEARNED_TRIGGER: learned_trigger.make_hook on the FROZEN union bank (file sha re-verified; its WM rows, as
    ORACLE_WM) with the frozen classifier model doc (file sha re-verified) for this world's cell_key; the executed code
    must equal its registration (learned_trigger_digest == ck['learned_trigger_sha256']). Returns (hook, digest)."""
    from . import learned_trigger
    cell = world.get('cell_key')
    if not cell or cell != ck.get('cell_key'):
        raise ValueError('LEARNED_TRIGGER: world cell_key missing or not the bound cell_key')
    bank = ck.get('oracle_bank') or {}
    path = Path(bank.get('path', ''))
    if not path.is_file() or _sha(path) != bank.get('file_sha256'):
        raise ValueError('LEARNED_TRIGGER: bank changed after registration')
    mdoc = ck.get('learned_trigger_model') or {}
    mpath = Path(mdoc.get('path', ''))
    if not mpath.is_file() or _sha(mpath) != mdoc.get('file_sha256'):
        raise ValueError('LEARNED_TRIGGER: classifier model doc changed after registration')
    loaded = learned_trigger.learned_trigger_digest()
    if loaded != ck.get('learned_trigger_sha256'):
        raise ValueError('LEARNED_TRIGGER: learned_trigger differs from its registration (learned_trigger_sha256)')
    model = learned_trigger.load_model(mpath, mdoc['file_sha256'])
    doc = json.loads(path.read_text(encoding='utf-8'))
    return learned_trigger.make_hook(doc, cell, model), loaded


def rule_hook(strategy, expected_sha256):
    import importlib
    if strategy not in RULE_STRATEGIES:
        raise ValueError(f'unregistered RULE strategy {strategy!r}')
    module = importlib.import_module(f'{__package__}.{RULE_MODULES[strategy]}')
    loaded = rule_digest(module) if RULE_MODULES[strategy] == 'regimeu0' else multicrit_digest(module)
    if loaded != expected_sha256:
        raise ValueError(f'{RULE_MODULES[strategy]} rule differs from its registration (rule_sha256)')
    return module.RuleHook(strategy), loaded


# ---------------------------------------------------------------- policies (training parity)

def eval_policy(native, arm, update, noise_seed):
    """The evaluated policy object, built exactly like the clean4 training episode builds it."""
    import numpy as np
    import torch
    from . import launch_production as production, track2_backend as p1
    from .v2v_runtime import from_weights
    weights = production.checkpoint_weights(native, arm, update)
    base = from_weights({k: v for k, v in weights.items() if not k.startswith('p1.')}, enabled=True)
    if arm not in ('B', 'R'):
        return base.eval()
    side = Path(native['run_root'])/f'{arm}_track2_{update:03}.pt'
    state = torch.load(side, map_location='cpu', weights_only=False)
    if state['arm'] != arm or state['update'] != update:
        raise ValueError('evaluation layer arm/update mismatch')
    policy = p1.EpisodePolicy(base, native, arm)
    policy.restore_worker(state['worker'])                      # full worker state (incl. RSU head)
    policy.layer_rng.manual_seed(int(noise_seed))               # as track2_backend.episode_components
    policy.candidate_rng = np.random.default_rng(int(noise_seed))
    if getattr(policy, 'rsu', None) is not None:
        policy.rsu.reseed(int(noise_seed))
    return policy.eval()


def wrap_controller(controller, policy):
    """The training controller wrapper (track2_backend.episode_components.apply), verbatim logic."""
    from . import track2_backend as p1
    if not isinstance(policy, p1.EpisodePolicy):
        return controller
    original = controller.apply

    def apply(rows, commands, now):
        if getattr(policy, 'lane_drop', None) is not None:
            # N2 speed law: each CAV's own native command this second (single writer).
            policy.observe(rows, now, command_commitments=controller.active_plans, native_commands=commands)
        else:
            policy.observe(rows, now, command_commitments=controller.active_plans)
        return original(rows, commands, now)
    controller.apply = apply
    return controller


class DecisionLog:
    """Controller hook part: logs new cooperative moves from the live samples (no physics change)."""

    def __init__(self, capture):
        self.capture, self.controller = capture, None
        self.seen, self.log, self.first = set(), [], None

    def after_step(self, t, rows=None):
        if self.controller is None or not self.controller.events:
            return
        sample = self.controller.events[-1].get('sample') or {}
        for entry in (sample.get('track2') or {}).get('active_moves') or ():
            move = entry.get('move') or {}
            members = list(move.get('members') or ())
            key = (move.get('family'), tuple(members), entry.get('start'))
            if len(members) < 2 or key in self.seen:
                continue
            self.seen.add(key)
            start = entry.get('start')
            row = dict(t_s=start, move=move.get('family'), members=members, plan=deepcopy(move))
            if len(self.log) < LOG_LIMIT:
                self.log.append(row)
            c = self.capture
            if (c is not None and self.first is None and isinstance(start, (int, float))
                    and c['min_members'] <= len(members) <= c['max_members'] and c['min_t_s'] <= start <= c['max_t_s']):
                self.first = dict(row, state_sha256=None, state_ref=None)


def eval_components(native, spec):
    """(job weights, policy_loader, controller_hook, decision log) for one policy-arm episode."""
    from . import launch_production as production
    ck = spec['checkpoint']
    arm, update = ck['arm'], int(ck['update'])
    weights = production.checkpoint_weights(native, arm, update)
    holder = {}
    log = DecisionLog(spec.get('capture'))

    def loader(values):
        holder['policy'] = eval_policy(native, arm, update, spec['noise_seed'])
        return holder['policy']

    def hook(controller, job, prefixes):
        wrap_controller(controller, holder['policy'])
        log.controller = controller
    hook.after_step = log.after_step
    return weights, loader, hook, log


def _policy_episode(spec, cfg, work):
    return _policy_run(spec, cfg, work, world=_check_spec(spec), stop=HORIZON_S, role=ROLE)


def _policy_run(spec, cfg, work, *, world, stop, role):
    from . import accel_plant, launch_production as production, protocol, rerun_cpus as rc
    from .rerun_tracks_backend import safety
    from .v2v_runner import patched
    native = production.backend_config(cfg)
    ck = spec['checkpoint']
    arm, update = ck['arm'], int(ck['update'])
    executor, loaded, rule, oracle = native['checkpoint'], None, None, None
    if spec['arm'] in ('U0', 'RULE', 'RULE_FAIR', 'MAPPO_SPEED') + ORACLE_ARMS + LEARNED_ARMS:
        kind = dict(U0='u0', RULE='rule', RULE_FAIR='rule', MAPPO_SPEED='mappo_comparator',
                    ORACLE='oracle', ORACLE_WM='oracle', LEARNED_TRIGGER='learned_trigger')[spec['arm']]
        if ck.get('kind') != kind or (arm, update) != ('A', 0):
            raise ValueError(f'{spec["arm"]} arm = its bound executor + the bound common u0 (zero residual)')
        # the bound zero-residual u0 (A_checkpoint_000.pt, sha re-verified), loaded by the SAME
        # production loaders with run_root = its directory (pre-run plant root or the clean run root)
        path = Path(ck['path'])
        if path.name != 'A_checkpoint_000.pt' or rc.sha(path) != ck['file_sha256']:
            raise ValueError('bound u0 residual changed after registration')
        native = dict(native, run_root=str(path.parent))
        executor, loaded = bound_executor(ck)        # U0/RULE: the frozen d9042cc5 export (pin unchanged)
        if spec['arm'] in ('RULE', 'RULE_FAIR'):
            rule = rule_hook(ck['rule'], ck['rule_sha256'])
        if spec['arm'] in ORACLE_ARMS:
            oracle = oracle_hook(ck, world, spec['arm'])
            rule = oracle
        if spec['arm'] in LEARNED_ARMS:
            oracle = learned_trigger_hook(ck, world)
            rule = oracle
    else:
        path = Path(native['run_root'])/f'{arm}_checkpoint_{update:03}.pt'
        if Path(ck['path']) != path or rc.sha(path) != ck['file_sha256']:
            raise ValueError('evaluation checkpoint changed after registration')
    weights, loader, hook, log = eval_components(native, spec)
    if rule is not None:
        rule_state, base_hook = rule[0], hook

        def hook(controller, job, prefixes):           # U0 hook first, then the S-REGIMEU0 rule (as regimeu0.worker)
            base_hook(controller, job, prefixes)
            rule_state(controller, job, prefixes)
        hook.after_step = base_hook.after_step
    job = dict(id=Path(work).name, path=str(work), world=world, role=role, arm=arm, weights=weights,
               runtime=native['runtime'], checkpoint=executor, noise_seed=spec['noise_seed'],
               deterministic=True, guided=False, v2v=True, sim_s=stop, stop=stop,
               warm_history=native['warm_history'], accel_bounds=native['accel_bounds'])
    # As launch_evaluation.episode: the original registered guards (both route map worlds to N1).
    vlog = VehicleLog(work)
    try:
        with patched(accel_plant, dict(guard=protocol.guard, install_seed_scope=protocol.install_seed_scope)), \
                logged_tick(vlog):
            if loaded is None:
                result = accel_plant.episode(job, controller_hook=hook, policy_loader=loader)
            else:
                with comparator_setup(loaded):
                    result = accel_plant.episode(job, controller_hook=hook, policy_loader=loader)
    finally:
        vehicle_log = vlog.close()
    extra = {}
    if spec['arm'] in ('B', 'R'):
        extra['decision_log'] = log.log
    if spec.get('capture') is not None:
        extra['decision'] = log.first
    if loaded is not None:
        extra['executor_sha256_loaded'] = loaded
    extra['vehicle_log'] = vehicle_log
    if oracle is not None and spec['arm'] in LEARNED_ARMS:
        extra['learned_trigger_sha256_loaded'] = oracle[1]
        extra['learned_trigger'] = oracle[0].receipts()
    elif oracle is not None:
        extra['oracle_sha256_loaded'] = oracle[1]
        extra['oracle'] = oracle[0].receipts()
    elif rule is not None:
        h = rule[0]
        extra['rule_sha256_loaded'] = rule[1]
        extra['rule'] = dict(strategy=h.strategy, activations=len(h.acts), issued=len(h.issued),
                             lane_held_steps=h.lane_held, members=sum(len(a['members']) for a in h.acts))
    return _record(spec, result['totals']['team_delay_vs'], safety(result, Path(work)/'observable.jsonl.gz'), **extra)


REGIME = dict(lane=2, zone_s_m=(-300., 75.44), window_s=(300, 360))


def closing_lane_regime(trace_path, *, lane=REGIME['lane'], zone=REGIME['zone_s_m'], window=REGIME['window_s']):
    """Report-only regime observable from the NATIVE trace: mean speed of the RSU-observable rows (CAV
    self states + locally detected neighbours) in WM lane 2 (the closing lane) within 300 m of the drop
    over t in [300, 360) s. The NATIVE arm is untreated, so the label never depends on a treatment."""
    import gzip
    speeds = []
    with gzip.open(trace_path, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            t = row.get('time_s', 0)
            if t < window[0]:
                continue
            if t >= window[1]:
                break
            speeds += [float(v['v_mps']) for v in row.get('vehicles', ())
                       if v.get('lane') == lane and zone[0] <= float(v.get('s_m', -1e9)) <= zone[1]]
    return dict(closing_lane_speed_mps=sum(speeds)/len(speeds) if speeds else None, n_obs=len(speeds),
                lane=lane, zone_s_m=list(zone), window_s=list(window), source='NATIVE RSU-observable rows')


def _native_episode(spec, cfg, work):
    return _native_run(spec, cfg, work, world=_check_spec(spec), stop=HORIZON_S, role=ROLE)


def _native_run(spec, cfg, work, *, world, stop, role):
    from tools.coopnative import plant as native
    from tools.coopnative.policy import Policy as NativePolicy
    from . import launch_production as production
    from .map_runtime import guard_world, install as map_install, patch_globals
    from .protocol import install_seed_scope
    from .replay_bank import ttc_events
    if spec['checkpoint'] != dict(kind='native'):
        raise ValueError('NATIVE spec must carry the native checkpoint marker')
    runtime = production.backend_config(cfg)['runtime']
    token = f'nm_finaleval NATIVE {spec["panel_sha256"]} {spec["job_id"]}'
    job = dict(id=Path(work).name, path=str(work), world=deepcopy(world), role=role, arm='native',
               weights=NativePolicy().state_dict(), runtime=runtime, noise_seed=spec['noise_seed'],
               sim_s=stop, stop=stop, approval_line=token)

    def approved(text):
        if text != token:
            raise ValueError('native evaluation token mismatch')

    def setup(rt):
        native.setup(rt)
        install_seed_scope()
        if map_install(job) is None:
            raise ValueError('NATIVE evaluation requires the wz3to2 map binding')
        patch_globals(run.__globals__, job)
        # the map binding REPLACES the plant's observations() (CAV-report rows); wrap whatever is bound now
        # (316461a4 wrapped before this patch, so the log stayed empty)
        observe = run.__globals__['observations']

        def observations(env):             # read-only ground-truth log, then the bound rows unchanged
            vlog.tick(env)
            return observe(env)
        run.__globals__['observations'] = observations
    vlog = VehicleLog(work)
    run = types.FunctionType(native.episode.__code__, dict(native.episode.__globals__,
                             approval_line=approved, setup=setup, guard=guard_world))
    try:
        result = run(job)
    finally:
        vehicle_log = vlog.close()
    if stop != HORIZON_S:                  # one-episode logcheck: the log is the product
        return dict(vehicle_log=vehicle_log)
    totals = result['metrics'][str(HORIZON_S-WARMUP_S)]
    vector = dict(collisions=int(totals['collisions']), teleports=int(totals['teleports']),
                  hard_brakes=int(totals['hard_brake']),
                  ttc_events=int(ttc_events(Path(work)/'observable.jsonl.gz', WARMUP_S, HORIZON_S)))
    return _record(spec, totals['team_delay_vs'], vector, regime=closing_lane_regime(Path(work)/'observable.jsonl.gz'),
                   vehicle_log=vehicle_log)


# ---------------------------------------------------------------- MAPPO_LANE (lane-capable comparator)

def load_lane_executor(path, sha256):
    """Deterministic load through the C4 deployment loader AFTER the frozen runtime setup; the
    export must carry its lane head (speed-only exports are refused here)."""
    import os
    import torch
    if _sha(path) != sha256:
        raise ValueError('MAPPO_LANE checkpoint differs from its bound sha256')
    if os.environ.get('C4_POLICY_STOCHASTIC') == '1' or float(os.environ.get('C4_POLICY_ACCEL_NOISE', '0') or 0) != 0:
        raise ValueError('deterministic deployment required (C4_POLICY_STOCHASTIC/ACCEL_NOISE set)')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('trainer') != 'mappo' or payload.get('lane_head') is not True or not any(
            str(k).startswith('lane_head.') for k in payload.get('policy', {})):
        raise ValueError('MAPPO_LANE requires a mappo export with its lane head')
    from wms.control.policy import load_policy_fn
    executor = load_policy_fn(str(path), device='cpu')
    if getattr(executor, 'deterministic', True) is not True:
        raise ValueError('MAPPO_LANE executor is not deterministic')
    executor.reset()
    return executor, dict(format=payload.get('format'), iteration=payload.get('iteration'), file_sha256=sha256,
                          loader='wms.control.policy.load_policy_fn', lane_change_mode=LANE_CHANGE_MODE)


def _flags(value):
    import numpy as np
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1).astype(bool)


def cav_only(state, cmd, lanes):
    """Executor output must command and steer present CAV slots only (clean3 lane-on rule)."""
    import numpy as np
    cmd, lanes = np.asarray(cmd, dtype=np.float64), np.asarray(lanes)
    if cmd.ndim != 2 or cmd.shape[0] != 1 or lanes.shape != cmd.shape:
        raise ValueError('MAPPO_LANE executor output must be [1,K] accel and [1,K] lanes')
    if lanes.dtype.kind not in 'iu':
        raise ValueError('MAPPO_LANE lane targets must be integers')
    flat = lanes.reshape(-1)
    if not set(np.unique(flat).tolist()) <= set(LANE_GRID):
        raise ValueError('lane target outside the WM lane grid {-1,0,1,2}')
    cav = _flags(state['is_cav'])
    if 'present' in state:
        cav = cav & _flags(state['present'])
    if cav.shape != flat.shape:
        raise ValueError('state/command slot count mismatch')
    if np.isinf(cmd).any():
        raise ValueError('infinite MAPPO_LANE command')
    commanded = np.isfinite(cmd.reshape(-1))
    if (commanded & ~cav).any():
        raise ValueError('acceleration command for a non-CAV slot')
    if (flat[~(cav & commanded)] != -1).any():
        raise ValueError('lane command for a non-CAV/uncommanded slot')
    return cmd, lanes.astype(np.int64), int((flat >= 0).sum())


def lane_tick(b, executor, trace, *, plant, cf, canonical, recorder=None):
    """One 1 s control tick: coopfree.backend.tick with the lane-aware executor and the env lane
    path (a lanes tensor every tick -> laneChangeMode 513 on commanded CAVs; HDVs untouched)."""
    import math
    import numpy as np
    import torch
    now, rows = float(b.env.now), cf.observations(b)
    ids = np.asarray(b.env.frame.vid_index)
    keep = np.zeros(len(ids), bool) if b.previous is None else (ids == b.previous) & (ids >= 0)
    b.previous = ids.copy()
    for obj in (executor.memory, executor.dwell):            # slot reuse never inherits memory/dwell
        if obj.present is not None:
            obj.present &= torch.as_tensor(keep, device=obj.present.device)[None]
    state = b.env.state()
    cmd, lanes = executor(state, with_lanes=True)
    cmd, lanes, requested = cav_only(state, cmd, lanes)
    b.command_digest.update(canonical(plant.exact(cmd)).encode())
    slot = {vid: i for i, vid in enumerate(b.env.slot_vehicle_ids())}
    flat_cmd, flat_lanes = cmd.reshape(-1), lanes.reshape(-1)
    if recorder is not None:
        recorder.tick(now, rows, {r['vid']: float(flat_cmd[slot[r['vid']]]) for r in rows if r['is_cav']},
                      {r['vid']: int(flat_lanes[slot[r['vid']]]) for r in rows if r['is_cav']})
    vehicles = []
    for r in rows:
        v = dict(r)
        if r['is_cav']:
            i = slot[r['vid']]
            a = float(flat_cmd[i])
            v.update(controller='MAPPO_LANE' if math.isfinite(a) else 'native',
                     issued_acceleration_mps2=a if math.isfinite(a) else None,
                     lane_target=int(flat_lanes[i]) if flat_lanes[i] >= 0 else None, command_present=math.isfinite(a))
        vehicles.append(v)
    if trace is not None:
        trace.write(canonical(dict(time_s=now, vehicles=vehicles, actions=[]))+'\n')
    _, info = b.env.step(cmd, lanes=lanes)
    interval = info['interval']
    road, source = interval['speed_deficit_vehicle_seconds'], interval['source_wait_vehicle_seconds']
    b.totals['road_delay_vs'] += road
    b.totals['source_wait_vs'] += source
    b.totals['team_delay_vs'] += road+source
    for key, field in (('collisions', 'collision_involvements'), ('teleports', 'starting_teleports'),
                       ('ttc_below_3_vs', 'ttc_below_3_vehicle_seconds'), ('hard_brake', 'braking_vehicle_seconds')):
        b.totals[key] += interval.get(field, 0.)
    b.observe(b.env)
    if info['done'] or any(interval.get(k, 0) for k in ('collision_involvements', 'starting_teleports',
                                                        'dropped_vehicle_steps')):
        raise ValueError('early termination/safety/slot anomaly (MAPPO_LANE)')
    return requested, dict(info.get('lane_commands', {}))


def _mappo_lane_episode(spec, cfg, work):
    """Amendment 17: the bound S-MAPPOWZL training execution path in a FRESH interpreter rooted at the bound training
    source tree (its own map/env binding, no G5 plant overlay); a binding without it is refused."""
    import subprocess
    import sys
    _check_spec(spec)
    ck = spec['checkpoint']
    if not ck.get('train_config') or not ck.get('train_src'):
        raise ValueError('MAPPO_LANE runs only on the bound S-MAPPOWZL training path (bind --train-config/--train-src)')
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    (work/'lane_spec.json').write_text(json.dumps(spec, sort_keys=True), encoding='utf-8')
    src = Path(ck['train_src'])
    env = train_subprocess_env(src)
    with (work/'lane_episode.log').open('w') as log:
        subprocess.run([sys.executable, '-B', '-m', 'tools.coopmwm.nm_eval_runner', 'lane-episode', '--spec',
                        str(work/'lane_spec.json'), '--work', str(work), '--out', str(work/'lane_record.json')],
                       cwd=str(src), env=env, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    return json.loads((work/'lane_record.json').read_text(encoding='utf-8'))


# ---------------------------------------------------------------- amendment 17 addendum: plant vs training env

ENVPAR_STOP = WARMUP_S+120
#: amendment 17c (MAIN): the training env is reset with the plant's horizon (901 s after the warm-up, as the leverprobe
#: Backend does); episodes still step only the registered seconds
TRAIN_ENV_HORIZON_S = 901
ENV_ATTRS = ('k', 'ttc', 'population', 'controller', 'physics_log', 'cav_vtype', 'extra_sumo_cli', 'v2v',
             'entry_queue_in_state', 'direct_accel_max_mps2', 'check_parity', 'mode', 'seed', 'demand', 'penetration',
             'warmup_s', 'horizon_s')


class _StartCaptured(Exception):
    pass


def _norm_bytes(data, roots):
    for root in roots:
        data = data.replace(str(root).encode(), b'<scene>').replace(str(root).replace('\\', '/').encode(), b'<scene>')
    import re as _re

    def repl(m):          # any other referenced existing file path -> <file:name:sha16> (host-independent)
        p = Path(m.group(1).decode(errors='replace'))
        return (b'"<file:'+p.name.encode()+b':'+_sha(p)[:16].encode()+b'>"') if p.is_absolute() and p.is_file() else m.group(0)
    return _re.sub(rb'"([^"<>]+?)"', repl, data)


def _norm_value(v):
    p = Path(str(v)) if isinstance(v, (str, Path)) else None
    if p is not None and p.is_absolute() and p.is_file():
        return f'<file:{p.name}:{_sha(p)[:16]}>'
    return v


def capture_start(run):
    """Run ``run()`` until SUMO would start (traci.start), capture the command and the env, never start SUMO."""
    import inspect
    import traci
    captured = {}
    original = traci.start

    def start(cmd, *a, **k):
        captured['cmd'] = [str(c) for c in cmd]
        for info in inspect.stack():
            env = info.frame.f_locals.get('self')
            if env is not None and hasattr(env, 'scenario_root') and hasattr(env, 'requests'):
                captured['env'] = env
                break
        raise _StartCaptured()
    traci.start = start
    try:
        run()
    except _StartCaptured:
        pass
    finally:
        traci.start = original
    if 'cmd' not in captured:
        raise ValueError('SUMO start was not reached')
    return captured


def start_receipt(captured):
    """Zero-SUMO receipt: normalised SUMO command line, scenario file hashes (scene dir normalised), env kwargs/config,
    and the shas of every loaded traffic_shepherd module and of the env class hierarchy."""
    import dataclasses
    import inspect
    import sys
    env = captured.get('env')
    files, cmd, roots = {}, [], set()
    for c in captured['cmd']:
        p = Path(c)
        if p.suffix and p.is_file() and p.suffix in ('.sumocfg', '.xml', '.gz') or p.name.endswith('.rou.xml'):
            roots.add(p.parent)
    for c in captured['cmd']:
        p = Path(c)
        cmd.append(f'<file:{p.name}>' if p.parent in roots and p.is_file() else
                   (f'<binary:{p.name}>' if c == captured['cmd'][0] else c))
    for root in sorted(roots, key=str):
        for f in sorted(root.iterdir(), key=lambda x: x.name):
            if f.is_file():
                files[f.name] = hashlib.sha256(_norm_bytes(f.read_bytes(), roots)).hexdigest()
    attrs = {k: repr(getattr(env, k, None)) for k in ENV_ATTRS} if env is not None else {}
    config = getattr(env, 'config', None)
    if config is not None and dataclasses.is_dataclass(config):
        attrs['config'] = json.dumps({k: _norm_value(v) for k, v in dataclasses.asdict(config).items()},
                                     sort_keys=True, default=repr)
    classes = {}
    if env is not None:
        for cls in type(env).__mro__:
            try:
                classes[f'{cls.__module__}.{cls.__qualname__}'] = _sha(inspect.getsourcefile(cls))
            except (TypeError, OSError):
                continue
    modules, module_paths = {}, {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if name.split('.')[0] in ('traffic_shepherd', 'tools', 'wms') and f and Path(f).is_file():
            modules[name] = _sha(f)
            module_paths[name] = str(Path(f).resolve())     # where each module was imported from (N5 .pth note)
    return dict(cmd=cmd, sumo_binary=captured['cmd'][0], files=files, env=attrs, env_classes=classes, modules=modules,
                module_paths=module_paths,
                env_class=type(env).__qualname__ if env is not None else None)


def _envpar_world(spec):
    from . import map_descriptor as md, map_scenarios as ms
    world = spec['world']
    ms.guard(world, PARITY_ROLE)                      # TRAIN worlds only
    if md.for_world(world).name != MAP:
        raise ValueError(f'process map is not {MAP}')
    if spec.get('kind') != 'envparity' or spec['arm'] != 'U0' or int(spec['sim_s']) != ENVPAR_STOP:
        raise ValueError('envparity spec: U0, TRAIN world, 300 + 120 s')
    return world


def _train_env_parts(ck):
    import hashlib as h
    if _sha(ck['train_config']) != ck['train_config_sha256']:
        raise ValueError('training config differs from its binding')
    manifest = lane_train_manifest(ck['train_src'])
    if h.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() != ck['train_src_sha256']:
        raise ValueError('training source tree differs from its binding')
    from tools.nm_mappo.sanity_eval import load_actor, load_cfg, resolve
    from wms.control.mappo.rollout import build_env
    raw = load_cfg(ck['train_config'])
    env_spec, _, _ = resolve(raw, fake=False)
    return raw, env_spec, load_actor, build_env


def start_receipt_train(spec):
    """Fresh interpreter in the training tree: the S-MAPPOWZL env (lane_writer lcexec) up to SUMO start."""
    world, ck = _envpar_world(spec), spec['checkpoint']
    raw, env_spec, _, build_env = _train_env_parts(ck)
    env = build_env(env_spec)
    captured = capture_start(lambda: env.reset(seed=int(world['seed']), demand_vph=int(world['demand']),
                                               penetration=float(world['penetration']), warmup_s=float(WARMUP_S),
                                               horizon_s=float(TRAIN_ENV_HORIZON_S)))
    return start_receipt(captured)


def start_receipt_plant(spec, cfg, work):
    """The plant U0 path (accel_plant + frozen runtime + G5 + N1) up to SUMO start."""
    world = _envpar_world(spec)
    captured = capture_start(lambda: _policy_run(spec, cfg, work, world=world, stop=ENVPAR_STOP, role=PARITY_ROLE))
    return start_receipt(captured)


def envparity_train(spec, work):
    """Fresh interpreter in the training tree: U0 (bound frozen export, deterministic, native lanes) for 120 s in the
    S-MAPPOWZL training env after the 300 s warm-up; ground-truth vehicle log + the plant reducers."""
    import numpy as np
    world, ck = _envpar_world(spec), spec['checkpoint']
    if _sha(ck['executor_path']) != ck['executor_sha256']:
        raise ValueError('U0 executor differs from its binding')
    raw, env_spec, load_actor, build_env = _train_env_parts(ck)
    exe, lane_control = load_actor(ck['executor_path'], raw.get('lanes'), None)
    env = build_env(env_spec)
    vlog = VehicleLog(work)
    totals = dict(team_delay_vs=0., collisions=0., teleports=0., hard_brake=0.)
    try:
        state = env.reset(seed=int(world['seed']), demand_vph=int(world['demand']), penetration=float(world['penetration']),
                          warmup_s=float(WARMUP_S), horizon_s=float(TRAIN_ENV_HORIZON_S))
        exe.reset()
        for _ in range(ENVPAR_STOP-WARMUP_S):
            vlog.tick(env)
            accel, lanes = exe(state, with_lanes=True)
            if lanes is not None and bool(np.any(np.asarray(lanes) >= 0)):
                raise ValueError('U0 must not issue lane commands (native lanes)')
            state, info = env.step(accel, lanes=None)
            interval = info.get('interval') or {}
            totals['team_delay_vs'] += interval.get('speed_deficit_vehicle_seconds', 0.)+interval.get(
                'source_wait_vehicle_seconds', 0.)
            totals['hard_brake'] += interval.get('braking_vehicle_seconds', 0.)
            totals['collisions'] += interval.get('collision_involvements', 0)
            totals['teleports'] += interval.get('starting_teleports', 0)
            if info.get('done') or info.get('terminated'):
                break
    finally:
        vehicle_log = vlog.close()
        if hasattr(env, 'close'):
            env.close()
    return dict({k: deepcopy(spec[k]) for k in IDENTITY}, status='traced', kind='envparity', path='train',
                J_veh_s=float(totals['team_delay_vs']), lane_control=lane_control,
                safety=dict(collisions=int(totals['collisions']), teleports=int(totals['teleports']),
                            hard_brakes=int(totals['hard_brake'])), vehicle_log=vehicle_log)


def envparity(spec, cfg, work):
    """One addendum parity episode: path 'plant' (the U0 plant episode, role check, stop 420) or 'train' (the
    S-MAPPOWZL training env in a fresh interpreter rooted at the bound training tree)."""
    import subprocess
    import sys
    _envpar_world(spec)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if spec['path'] == 'plant':
        rec = _policy_run(spec, cfg, work, world=spec['world'], stop=ENVPAR_STOP, role=PARITY_ROLE)
        return dict(rec, status='traced', kind='envparity', path='plant',
                    safety={k: rec['safety'][k] for k in ('collisions', 'teleports', 'hard_brakes')})
    src = Path(spec['checkpoint']['train_src'])
    (work/'envpar_spec.json').write_text(json.dumps(spec, sort_keys=True), encoding='utf-8')
    env = train_subprocess_env(src)
    with (work/'envparity.log').open('w') as log:
        subprocess.run([sys.executable, '-B', '-m', 'tools.coopmwm.nm_eval_runner', 'envparity-train', '--spec',
                        str(work/'envpar_spec.json'), '--work', str(work), '--out', str(work/'envpar_record.json')],
                       cwd=str(src), env=env, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    return json.loads((work/'envpar_record.json').read_text(encoding='utf-8'))


def shared_safety(result, trace, ttc_events=None):
    """The plant arms' reducer tools.coopmwm.rerun_tracks_backend.safety; ``ttc_events`` only for CPU fakes."""
    if ttc_events is None:
        from .rerun_tracks_backend import safety
        return safety(result, trace)
    totals = result['totals']
    return dict(collisions=int(totals['collisions']), teleports=int(totals['teleports']),
                hard_brakes=int(totals['hard_brake']), ttc_events=int(ttc_events(trace, WARMUP_S, HORIZON_S)))


def train_subprocess_env(src):
    """Environment of a training-tree subprocess (amendment 17b, MAIN): PYTHONPATH is built from the bound training
    tree ONLY (<train_src>, <train_src>/src) plus SUMO's own tools dir ($SUMO_HOME/tools, traci) - never the eval
    tree or any inherited entry, so no mixed import; no G5 plant overlay."""
    import os
    src = Path(src)
    entries = [str(src), str(src/'src')]
    if os.environ.get('SUMO_HOME'):
        entries.append(str(Path(os.environ['SUMO_HOME'])/'tools'))
    env = {k: v for k, v in os.environ.items() if k != 'COOPMWM_G5_RUNTIME'}
    env['PYTHONPATH'] = os.pathsep.join(entries)
    return env


def lane_train_manifest(root):
    """sha256 of the training-path files of a source tree (bound at bind-mappo, re-verified per episode)."""
    return {f: _sha(Path(root)/f) for f in LANE_TRAIN_FILES}


def lane_train_episode(spec, work, *, modules=None):
    """In the fresh interpreter: one held-out MAPPO_LANE episode on the training execution path. Same reducers as
    every arm: J = sum of the env interval speed deficit + source wait over [300, 1200); hard brakes = braking
    vehicle-seconds; collisions/teleports abort (as the plant); observable trace = the map CAV-report rows (TTC
    events); EVALUATION-ONLY ground-truth vehicle log."""
    import gzip
    import hashlib as h
    import numpy as np
    world, ck = _check_spec(spec), spec['checkpoint']
    if _sha(ck['executor_path']) != ck['executor_sha256'] or _sha(ck['train_config']) != ck['train_config_sha256']:
        raise ValueError('MAPPO_LANE executor or training config differs from its binding')
    manifest = lane_train_manifest(ck['train_src'])
    if h.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() != ck['train_src_sha256']:
        raise ValueError('MAPPO_LANE training source tree differs from its binding')
    if modules is None:
        from tools.nm_mappo.sanity_eval import load_actor, load_cfg, resolve
        from wms.control.mappo.rollout import build_env
        from tools.coopmwm import map_descriptor as md
        from tools.coopmwm.map_observations import cav_report_rows_with_length as rows_of
        from tools.coopmwm.replay_bank import ttc_events
    else:
        load_actor, load_cfg, resolve, build_env, md, rows_of, ttc_events = modules
    raw = load_cfg(ck['train_config'])
    env_spec, _, _ = resolve(raw, fake=False)
    if (env_spec.get('env_kwargs') or {}).get('lane_writer') != 'lcexec':
        raise ValueError('MAPPO_LANE training config does not bind the lcexec lane writer (never a native-lane writer)')
    lane_actions = (raw.get('policy') or {}).get('lane_actions')
    if lane_actions != LCEXEC_LANE_ACTIONS:
        raise ValueError(f'MAPPO_LANE lane_actions must be {LCEXEC_LANE_ACTIONS}')
    exe, lane_control = load_actor(ck['executor_path'], raw.get('lanes'), lane_actions)
    if lane_control != 'learned' or type(exe).__name__ != 'LaneActor':
        raise ValueError('MAPPO_LANE must load as the lane-capable ActorStep (LaneActor), never a native-lane executor')
    work = Path(work)
    env = build_env(env_spec)
    vlog = VehicleLog(work)
    totals = dict(team_delay_vs=0., road_delay_vs=0., source_wait_vs=0., collisions=0., teleports=0., hard_brake=0.)
    lane_ticks = 0
    try:
        state = env.reset(seed=int(world['seed']), demand_vph=int(world['demand']), penetration=float(world['penetration']),
                          warmup_s=float(WARMUP_S), horizon_s=float(TRAIN_ENV_HORIZON_S))
        exe.reset()
        desc = md.current()
        with gzip.open(work/'observable.jsonl.gz', 'xt', encoding='utf-8') as trace:
            for _ in range(HORIZON_S-WARMUP_S):
                vlog.tick(env)
                trace.write(json.dumps(dict(time_s=float(env.now), vehicles=rows_of(env, desc), actions=[]),
                                       sort_keys=True)+'\n')
                accel, lanes = exe(state, with_lanes=True)
                if lanes is not None and bool(np.any(np.asarray(lanes) >= 0)):
                    lane_ticks += 1
                state, info = env.step(accel, lanes=lanes)
                interval = info.get('interval') or {}
                road, source = interval.get('speed_deficit_vehicle_seconds', 0.), interval.get('source_wait_vehicle_seconds', 0.)
                totals['road_delay_vs'] += road
                totals['source_wait_vs'] += source
                totals['team_delay_vs'] += road+source
                totals['hard_brake'] += interval.get('braking_vehicle_seconds', 0.)
                totals['collisions'] += interval.get('collision_involvements', 0)
                totals['teleports'] += interval.get('starting_teleports', 0)
                if totals['collisions'] or totals['teleports'] or interval.get('dropped_vehicle_steps', 0):
                    raise ValueError('early termination/safety/slot anomaly (MAPPO_LANE)')
                if info.get('done') or info.get('terminated'):
                    break
    finally:
        vehicle_log = vlog.close()
        if hasattr(env, 'close'):
            env.close()
    # amendment 17c: the SAME shared reducer as every plant arm (rerun_tracks_backend.safety: collisions/teleports/
    # braking from the env intervals, TTC events from the observable trace) - never the env ttc flag
    safety = shared_safety(dict(totals=totals), work/'observable.jsonl.gz', ttc_events if modules is not None else None)
    return _record(spec, totals['team_delay_vs'], safety, executor_sha256_loaded=ck['executor_sha256'],
                   vehicle_log=vehicle_log, lane_request_ticks=lane_ticks, lane_path='training (lcexec ActorStep)',
                   env_horizon_s=TRAIN_ENV_HORIZON_S)


def _mappo_lane_run(spec, cfg, work, *, world, stop, role, recorder=None, vehicle_log=True):
    import gzip
    import os
    from collections import Counter
    from . import accel_plant, launch_production as production, protocol
    from .accel_storage import write
    from .rerun_tracks_backend import safety
    from .v2v_runner import patched
    ck = spec['checkpoint']
    if ck.get('kind') != 'mappo_lane':
        raise ValueError('MAPPO_LANE spec must carry the mappo_lane comparator binding')
    path, loaded = bound_executor(ck)
    native = production.backend_config(cfg)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    write(work/'CLAIM.json', dict(pid=os.getpid(), id=work.name, charged_sim_s=stop, arm='MAPPO_LANE'))
    job = dict(id=work.name, path=str(work), world=world, role=role, arm='MAPPO_LANE', runtime=native['runtime'],
               checkpoint=native['checkpoint'])
    # Same binding as every policy arm: frozen runtime (d9042cc5 pin unchanged) -> seed scope -> guard
    # -> G5 overlay -> N1 map install; the plant's own frozen executor is never used for actions here.
    with patched(accel_plant, dict(guard=protocol.guard, install_seed_scope=protocol.install_seed_scope)):
        plant = accel_plant.bind_runtime(job)
    from tools.coopfree import backend as cf
    from tools.coopfree.protocol import canonical
    plant.guard = lambda w: protocol.guard(w, role)
    executor, meta = load_lane_executor(path, loaded)
    b = plant.Backend(work, world, stop, native['checkpoint'])
    lane_commands, request_ticks = Counter(), 0
    vlog = VehicleLog(work) if vehicle_log else None
    try:
        with gzip.open(work/'observable.jsonl.gz', 'xt', encoding='utf-8') as trace:
            for _ in range(WARMUP_S, stop):
                if vlog is not None:
                    vlog.tick(b.env)
                requested, counts = lane_tick(b, executor, trace, plant=plant, cf=cf, canonical=canonical,
                                              recorder=recorder)
                lane_commands.update(counts)
                request_ticks += requested > 0
        result = dict(id=work.name, world=world, arm='MAPPO_LANE', totals=dict(b.totals), actual_sim_s=b.actual,
                      charged_sim_s=stop, lane_commands=dict(lane_commands), lane_request_ticks=request_ticks,
                      executor=meta)
        write(work/'RESULT.json', result)
    finally:
        b.close()
        extra = {} if vlog is None else dict(vehicle_log=vlog.close())
    return _record(spec, result['totals']['team_delay_vs'], safety(result, work/'observable.jsonl.gz'),
                   executor_sha256_loaded=loaded, lane_commands=dict(lane_commands), lane_request_ticks=request_ticks,
                   **extra)


def episode(spec, cfg, work):
    """One registered evaluation episode -> graded record (raises on any refusal/anomaly)."""
    if spec['arm'] == 'NATIVE':
        return _native_episode(spec, cfg, work)
    if spec['arm'] == 'MAPPO_LANE':
        return _mappo_lane_episode(spec, cfg, work)
    if spec['arm'] not in ('A', 'B', 'R', 'U0', 'RULE', 'RULE_FAIR', 'MAPPO_SPEED') + ORACLE_ARMS + LEARNED_ARMS:
        raise ValueError(f'unknown evaluation arm {spec["arm"]!r}')
    return _policy_episode(spec, cfg, work)


# ---------------------------------------------------------------- one-episode vehicle-log check

LOGCHECK_ARMS = ('NATIVE', 'U0', 'RULE', 'RULE_FAIR')


def logcheck(spec, cfg, work):
    """ONE short episode (TRAIN world, role 'check', native warmup + a few controlled seconds) of a pre-run arm
    whose only product is the ground-truth vehicle log - checked field by field by the launcher before any
    wave is dispatched (MAIN: every new mode gets a one-episode check first)."""
    from . import map_descriptor as md, map_scenarios as ms
    world = spec['world']
    ms.guard(world, PARITY_ROLE)
    if md.for_world(world).name != MAP:
        raise ValueError(f'process map is not {MAP}')
    stop = int(spec['sim_s'])
    if spec.get('kind') != 'logcheck' or spec['arm'] not in LOGCHECK_ARMS or not WARMUP_S < stop <= WARMUP_S+300:
        raise ValueError('logcheck: a pre-run arm, TRAIN world, 300 s warmup + <= 300 s')
    if spec['arm'] == 'NATIVE':
        out = _native_run(spec, cfg, work, world=world, stop=stop, role=PARITY_ROLE)
        vlog = out['vehicle_log']
    else:
        vlog = _policy_run(spec, cfg, work, world=world, stop=stop, role=PARITY_ROLE)['vehicle_log']
    import os
    return dict({k: deepcopy(spec[k]) for k in IDENTITY}, status='logged', kind='logcheck', vehicle_log=vlog,
                work=str(work), pid=os.getpid())


# ---------------------------------------------------------------- F11 parity (eval plant vs training env)

def _num(x):
    import math
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


class ParityTrace:
    """Per control tick: {vid: [s_m, v_mps, lane, command or None, lane target or None]} for CAVs."""

    def __init__(self):
        self.ticks = []

    def tick(self, now, rows, commands, lanes=None):
        lanes = lanes or {}
        cav = {}
        for r in rows:
            if r.get('is_cav'):
                lt = lanes.get(r['vid'])
                cav[str(r['vid'])] = [round(float(r['s_m']), 4), round(float(r['v_mps']), 4), int(r['lane']),
                                      _num(commands.get(r['vid'])), None if lt is None or lt < 0 else int(lt)]
        self.ticks.append(dict(t=int(round(float(now))), n=len(rows), cav=cav))


def _check_parity_spec(spec):
    from . import map_descriptor as md, map_scenarios as ms
    world = spec['world']
    ms.guard(world, PARITY_ROLE)                      # TRAIN worlds only
    if md.for_world(world).name != MAP:
        raise ValueError(f'process map is not {MAP}')
    if (spec.get('kind') != 'parity' or spec.get('path') not in ('plant', 'train')
            or spec['arm'] not in ('MAPPO_SPEED', 'MAPPO_LANE')):
        raise ValueError('parity spec: kind parity, path plant|train, a MAPPO comparator arm')
    if int(spec['sim_s']) != WARMUP_S+int(spec['rollout_s']) or spec['window'] != dict(start_s=WARMUP_S,
                                                                                         end_s=int(spec['sim_s'])):
        raise ValueError('parity window = native warmup + rollout_s')
    return world


def _parity_plant(spec, cfg, work):
    from . import accel_plant, launch_production as production, protocol, rerun_cpus as rc
    from .v2v_runner import patched
    world = _check_parity_spec(spec)
    stop, recorder, ck = int(spec['sim_s']), ParityTrace(), spec['checkpoint']
    if spec['arm'] == 'MAPPO_LANE':
        record = _mappo_lane_run(spec, cfg, work, world=world, stop=stop, role=PARITY_ROLE, recorder=recorder,
                                 vehicle_log=False)
        return dict(trace=recorder.ticks, executor_sha256_loaded=record['executor_sha256_loaded'], safety=record['safety'])
    native = production.backend_config(cfg)
    path = Path(ck['path'])
    if ck.get('kind') != 'mappo_comparator' or path.name != 'A_checkpoint_000.pt' or rc.sha(path) != ck['file_sha256']:
        raise ValueError('MAPPO_SPEED parity = the bound executor + the bound common u0 (zero residual)')
    native = dict(native, run_root=str(path.parent))
    executor, loaded = bound_executor(ck)
    weights, loader, hook, _ = eval_components(native, dict(spec, capture=None, noise_seed=world['seed']))
    base_hook = hook

    def hook(controller, job, prefixes):              # the MAPPO_SPEED episode hook + a read-only recorder
        base_hook(controller, job, prefixes)
        previous = controller.apply

        def apply(rows, commands, now):
            recorder.tick(now, rows, commands)
            return previous(rows, commands, now)
        controller.apply = apply
    hook.after_step = base_hook.after_step
    job = dict(id=Path(work).name, path=str(work), world=world, role=PARITY_ROLE, arm='A', weights=weights,
               runtime=native['runtime'], checkpoint=executor, noise_seed=world['seed'], deterministic=True,
               guided=False, v2v=True, sim_s=stop, stop=stop, warm_history=native['warm_history'],
               accel_bounds=native['accel_bounds'])
    with patched(accel_plant, dict(guard=protocol.guard, install_seed_scope=protocol.install_seed_scope)):
        with comparator_setup(loaded):
            result = accel_plant.episode(job, controller_hook=hook, policy_loader=loader)
    totals = result['totals']
    return dict(trace=recorder.ticks, executor_sha256_loaded=loaded,
                safety=dict(collisions=int(totals.get('collisions', 0)), teleports=int(totals.get('teleports', 0))))


def parity_train_trace(spec):
    """N7's training env path in THIS (fresh) process: resolve(config) -> build_env -> reset(seed, demand,
    penetration, warmup 300, horizon rollout_s) -> PolicyExecutor(state, with_lanes=True) -> step."""
    import numpy as np
    ck = spec['checkpoint']
    if _sha(ck['executor_path']) != ck['executor_sha256'] or _sha(spec['train_config']) != spec['train_config_sha256']:
        raise ValueError('parity train path: executor or training config differs from its binding')
    from tools.nm_mappo.sanity_eval import load_actor, load_cfg, resolve
    from wms.control.mappo.rollout import build_env
    raw = load_cfg(spec['train_config'])
    env_spec, binding, _ = resolve(raw, fake=False)
    world, recorder = spec['world'], ParityTrace()
    binding.check_seeds([world['seed']])
    env = build_env(env_spec)
    try:
        exe, lane_control = load_actor(ck['executor_path'], raw.get('lanes'))
        state = env.reset(seed=int(world['seed']), demand_vph=int(world['demand']), penetration=float(world['penetration']),
                          warmup_s=float(WARMUP_S), horizon_s=float(spec['rollout_s']))
        exe.reset()
        collisions = teleports = 0
        for _ in range(int(spec['rollout_s'])):
            accel, lanes = exe(state, with_lanes=True)
            if spec['arm'] == 'MAPPO_SPEED' and lanes is not None:
                raise ValueError('speed-only comparator produced lane commands')
            ids = env.slot_vehicle_ids()
            a = np.asarray(accel.detach().cpu() if hasattr(accel, 'detach') else accel, dtype=np.float64).reshape(-1)
            ln = (None if lanes is None else
                  np.asarray(lanes.detach().cpu() if hasattr(lanes, 'detach') else lanes).reshape(-1))
            cav = _flags(state['is_cav'])
            sm, vm = (np.asarray(state[k].detach().cpu() if hasattr(state[k], 'detach') else state[k],
                                 dtype=np.float64).reshape(-1) for k in ('s_m', 'v_mps'))
            lane = np.asarray(state['lane'].detach().cpu() if hasattr(state['lane'], 'detach') else state['lane']).reshape(-1)
            rows = [dict(vid=ids[i], is_cav=True, s_m=sm[i], v_mps=vm[i], lane=int(lane[i]))
                    for i in range(len(ids)) if ids[i] is not None and cav[i]]
            recorder.tick(env.now, rows, {ids[i]: a[i] for i in range(len(ids)) if ids[i] is not None},
                          None if ln is None else {ids[i]: int(ln[i]) for i in range(len(ids)) if ids[i] is not None})
            state, info = env.step(accel, lanes=lanes)
            summary = info.get('interval') or {}
            collisions += int(summary.get('collision_involvements', 0) or 0)
            teleports += int(summary.get('starting_teleports', 0) or 0)
            if info.get('done') or info.get('terminated'):
                break
    finally:
        if hasattr(env, 'close'):
            env.close()
    return dict(trace=recorder.ticks, executor_sha256_loaded=ck['executor_sha256'], lane_control=lane_control,
                safety=dict(collisions=collisions, teleports=teleports))


def _parity_train(spec, work):
    """The training path runs in a FRESH interpreter (trainer binding: map_runtime.install(leverprobe=False),
    no G5 plant overlay) so it never shares process state with plant jobs."""
    import os
    import subprocess
    import sys
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    (work/'parity_spec.json').write_text(json.dumps(spec, sort_keys=True), encoding='utf-8')
    env = {k: v for k, v in os.environ.items() if k != 'COOPMWM_G5_RUNTIME'}
    with (work/'parity_train.log').open('w') as log:
        subprocess.run([sys.executable, '-B', '-m', 'tools.coopmwm.nm_eval_runner', 'parity-train', '--spec',
                        str(work/'parity_spec.json'), '--out', str(work/'parity_train.json')], env=env, check=True,
                       stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    return json.loads((work/'parity_train.json').read_text(encoding='utf-8'))


def parity(spec, cfg, work):
    """One F11 parity trace -> record (identity + trace; never graded on J)."""
    out = _parity_plant(spec, cfg, work) if spec['path'] == 'plant' else _parity_train(spec, work)
    return dict({k: deepcopy(spec[k]) for k in IDENTITY}, status='traced', kind='parity', path=spec['path'],
                trace=out['trace'], executor_sha256_loaded=out['executor_sha256_loaded'],
                safety={k: int(v) for k, v in out['safety'].items()})


# ---------------------------------------------------------------- zero-physics checks

def mappo_load_check(path, sha256):
    """Zero physics: the comparator file hashes as bound and torch-loads; reports its layout."""
    import torch
    if _sha(path) != sha256:
        raise ValueError(f'{path} does not hash to {sha256}')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    keys = sorted(saved)[:20] if isinstance(saved, dict) else []
    return dict(status='loaded', sha256=sha256, type=type(saved).__name__, keys=keys,
                format=saved.get('format') if isinstance(saved, dict) else None)


def setup_check(runtime, checkpoint, sha256, binding=None):
    """cpuV, ZERO SUMO steps: the frozen-runtime setup the MAPPO episode will run (comparator pin
    = ``sha256``) plus the executor load, and the speed-only requirement. Never starts SUMO."""
    if _sha(checkpoint) != sha256:
        raise ValueError(f'{checkpoint} does not hash to {sha256}')
    if binding is not None:
        bound = read_binding(binding)
        if bound['sha256'] != sha256 or bound['path'] != str(checkpoint):
            raise ValueError('binding record names a different comparator')
    with comparator_setup(sha256) as setup:
        setup(runtime, checkpoint)
    from wms.control.policy import load_policy_fn
    exe = load_policy_fn(str(checkpoint), device='cpu')
    speed_only = bool(getattr(exe, 'native_lanes', False)) and not getattr(exe, 'residual_on_native', True)
    return dict(status='PASS' if speed_only else 'FAIL', sha256=sha256, frozen_pin=FROZEN_MAPPO_SHA,
                pin_swapped=sha256 != FROZEN_MAPPO_SHA, native_lanes=bool(getattr(exe, 'native_lanes', False)),
                residual_on_native=bool(getattr(exe, 'residual_on_native', True)), sumo_steps=0,
                reason=None if speed_only else 'speed-only direct MAPPO required (native lanes, no residual)')


def main(argv=None):
    p = argparse.ArgumentParser(description='N6 eval runner zero-physics checks')
    p.add_argument('command', choices=('setup-check', 'load-check', 'parity-train', 'lane-episode', 'envparity-train',
                                       'start-receipt'))
    p.add_argument('--path', choices=('plant', 'train'), help='start-receipt: which path')
    p.add_argument('--cfg', help='start-receipt --path plant: the plant CONFIG.json')
    p.add_argument('--work', help='lane-episode: the episode work dir')
    p.add_argument('--runtime', default='/home/ubuntu/e3final')
    p.add_argument('--checkpoint')
    p.add_argument('--sha256')
    p.add_argument('--binding', help='optional MAPPO_COMPARATOR.json to cross-check')
    p.add_argument('--spec', help='parity-train: the parity spec JSON (written by the launcher worker)')
    p.add_argument('--out', help='parity-train: output JSON (never overwritten)')
    a = p.parse_args(argv)
    if a.command in ('envparity-train', 'start-receipt'):
        spec = json.loads(Path(a.spec).read_text(encoding='utf-8'))
        if a.command == 'envparity-train':
            value = envparity_train(spec, a.work)
        elif a.path == 'train':
            value = start_receipt_train(spec)
        else:
            cfg = json.loads(Path(a.cfg).read_text(encoding='utf-8'))
            value = start_receipt_plant(spec, cfg, a.work)
        with open(a.out, 'x', encoding='utf-8') as f:
            json.dump(value, f, sort_keys=True)
        return 0
    if a.command == 'lane-episode':
        spec = json.loads(Path(a.spec).read_text(encoding='utf-8'))
        value = lane_train_episode(spec, a.work)
        with open(a.out, 'x', encoding='utf-8') as f:
            json.dump(value, f, sort_keys=True)
        return 0
    if a.command == 'parity-train':
        spec = json.loads(Path(a.spec).read_text(encoding='utf-8'))
        _check_parity_spec(spec)
        value = parity_train_trace(spec)
        with open(a.out, 'x', encoding='utf-8') as f:
            json.dump(value, f, sort_keys=True)
        return 0
    if not a.checkpoint or not a.sha256:
        p.error('--checkpoint and --sha256 required')
    out = (setup_check(a.runtime, a.checkpoint, a.sha256, a.binding) if a.command == 'setup-check'
           else mappo_load_check(a.checkpoint, a.sha256))
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0 if out['status'] in ('PASS', 'loaded') else 1


if __name__ == '__main__':
    raise SystemExit(main())
