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

MAPPO comparator: the bound executor (``bind-mappo``: path + sha256, re-verified here) replaces
the plant's frozen executor checkpoint. ``tools.leverprobe.backend.setup`` pins d9042cc5; for a
bound non-d9042cc5 comparator ONLY that pin is swapped, inside this worker, for the bound sha
(``comparator_setup``: same function code, every other frozen-runtime check unchanged). The
training path's pin is never touched. Records carry the sha actually loaded.

NATIVE: the qualified true-native plant (``tools.coopnative.plant``) with the N1 map install +
CAV-report rows; J = its ``metrics['900']['team_delay_vs']`` (identical reducer).

Decisions (B/R): descriptive capture only (every new 2+-member cooperative move); no
``decision_branch`` (a faithful branch needs the MoveExecutor state restored inside the CRN
prefix replay; not built, never faked).

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
FROZEN_MAPPO_SHA = 'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'


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
    from . import accel_plant, launch_production as production, protocol, rerun_cpus as rc
    from .rerun_tracks_backend import safety
    from .v2v_runner import patched
    world = _check_spec(spec)
    native = production.backend_config(cfg)
    ck = spec['checkpoint']
    arm, update = ck['arm'], int(ck['update'])
    path = Path(native['run_root'])/f'{arm}_checkpoint_{update:03}.pt'
    if Path(ck['path']) != path or rc.sha(path) != ck['file_sha256']:
        raise ValueError('evaluation checkpoint changed after registration')
    executor, loaded = native['checkpoint'], None
    if spec['arm'] == 'MAPPO':
        if ck.get('kind') != 'mappo_comparator' or (arm, update) != ('A', 0):
            raise ValueError('MAPPO arm = bound comparator executor + the common u0 (zero residual)')
        executor, loaded = bound_executor(ck)
    weights, loader, hook, log = eval_components(native, spec)
    job = dict(id=Path(work).name, path=str(work), world=world, role=ROLE, arm=arm, weights=weights,
               runtime=native['runtime'], checkpoint=executor, noise_seed=spec['noise_seed'],
               deterministic=True, guided=False, v2v=True, sim_s=HORIZON_S, stop=HORIZON_S,
               warm_history=native['warm_history'], accel_bounds=native['accel_bounds'])
    # As launch_evaluation.episode: the original registered guards (both route map worlds to N1).
    with patched(accel_plant, dict(guard=protocol.guard, install_seed_scope=protocol.install_seed_scope)):
        if loaded is None:
            result = accel_plant.episode(job, controller_hook=hook, policy_loader=loader)
        else:
            with comparator_setup(loaded):
                result = accel_plant.episode(job, controller_hook=hook, policy_loader=loader)
    extra = {}
    if spec['arm'] in ('B', 'R'):
        extra['decision_log'] = log.log
    if spec.get('capture') is not None:
        extra['decision'] = log.first
    if loaded is not None:
        extra['executor_sha256_loaded'] = loaded
    return _record(spec, result['totals']['team_delay_vs'], safety(result, Path(work)/'observable.jsonl.gz'), **extra)


def _native_episode(spec, cfg, work):
    from tools.coopnative import plant as native
    from tools.coopnative.policy import Policy as NativePolicy
    from . import launch_production as production
    from .map_runtime import guard_world, install as map_install, patch_globals
    from .protocol import install_seed_scope
    from .replay_bank import ttc_events
    world = _check_spec(spec)
    if spec['checkpoint'] != dict(kind='native'):
        raise ValueError('NATIVE spec must carry the native checkpoint marker')
    runtime = production.backend_config(cfg)['runtime']
    token = f'nm_finaleval NATIVE {spec["panel_sha256"]} {spec["job_id"]}'
    job = dict(id=Path(work).name, path=str(work), world=deepcopy(world), role=ROLE, arm='native',
               weights=NativePolicy().state_dict(), runtime=runtime, noise_seed=spec['noise_seed'],
               sim_s=HORIZON_S, stop=HORIZON_S, approval_line=token)

    def approved(text):
        if text != token:
            raise ValueError('native evaluation token mismatch')

    def setup(rt):
        native.setup(rt)
        install_seed_scope()
        if map_install(job) is None:
            raise ValueError('NATIVE evaluation requires the wz3to2 map binding')
        patch_globals(run.__globals__, job)
    run = types.FunctionType(native.episode.__code__, dict(native.episode.__globals__,
                             approval_line=approved, setup=setup, guard=guard_world))
    result = run(job)
    totals = result['metrics'][str(HORIZON_S-WARMUP_S)]
    vector = dict(collisions=int(totals['collisions']), teleports=int(totals['teleports']),
                  hard_brakes=int(totals['hard_brake']),
                  ttc_events=int(ttc_events(Path(work)/'observable.jsonl.gz', WARMUP_S, HORIZON_S)))
    return _record(spec, totals['team_delay_vs'], vector)


def episode(spec, cfg, work):
    """One registered evaluation episode -> graded record (raises on any refusal/anomaly)."""
    if spec['arm'] == 'NATIVE':
        return _native_episode(spec, cfg, work)
    if spec['arm'] not in ('A', 'B', 'R', 'MAPPO'):
        raise ValueError(f'unknown evaluation arm {spec["arm"]!r}')
    return _policy_episode(spec, cfg, work)


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
    p.add_argument('command', choices=('setup-check', 'load-check'))
    p.add_argument('--runtime', default='/home/ubuntu/e3final')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--binding', help='optional MAPPO_COMPARATOR.json to cross-check')
    a = p.parse_args(argv)
    out = (setup_check(a.runtime, a.checkpoint, a.sha256, a.binding) if a.command == 'setup-check'
           else mappo_load_check(a.checkpoint, a.sha256))
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0 if out['status'] in ('PASS', 'loaded') else 1


if __name__ == '__main__':
    raise SystemExit(main())
