"""wz3to2 final-evaluation runner (N6) for ``ops/nm_finaleval.py`` on the clean4 deployed source.

Contract: ``ops/nm_finaleval.RUNNER_CONTRACT``. One process = one map: every worker needs
``COOPMWM_MAP_SPEC`` = the staged wz3to2 MAP_SPEC.json (N1 ``map_descriptor``) and
``COOPMWM_G5_RUNTIME=1`` like the clean4 owners; the launcher sets nothing else.

* Policy arms (A/B/R u24; MAPPO = the bound comparator executor + the clean4 common u0, i.e. MAPPO
  with a zero residual): the clean3 final evaluation episode (``launch_evaluation.episode``)
  unchanged except the world registry, the role and, for MAPPO, the executor checkpoint:
  N1 map world on the registered evaluation panel, role ``final_test`` (N1 HELDOUT_ROLES), the
  same ``accel_plant.episode`` (N1 ``bind_runtime`` installs the map), deterministic, V2V, noise
  seed = world seed, 1200 s (native warmup 0-300 s, control 300-1200 s), J = plant
  ``totals['team_delay_vs']``, safety = ``rerun_tracks_backend.safety``. The MAPPO comparator
  (MAIN: N7's wz3to2 retrain, else the explicitly bound d9042cc5 fallback "u0 transferred, not
  retrained") is passed as path + sha256 and re-hashed before the plant loads it.
* NATIVE: the qualified true-native plant (``tools.coopnative.plant``; clean3 final-eval binding)
  with the N1 map install + CAV-report rows (the same steps as N1's ``tools.coopmwm.plant``
  native_reference patch); J = its ``metrics['900']['team_delay_vs']`` (identical reducer).
* Decisions (B/R): descriptive capture only. Every new 2-3-member cooperation move the
  EpisodePolicy starts (``sample['track2']['active_moves']``) is logged; on the registered decision
  subset the first one inside the capture window is returned as ``decision`` (no state). No
  ``decision_branch`` is provided: a faithful branch needs the EpisodePolicy MoveExecutor state
  restored at t0 inside the clean3 CRN prefix replay (not built), so the launcher reports the
  per-decision decomposition as unavailable instead of running an unqualified substitute.

HDVs are never commanded (the plants only command CAVs). Never run locally: this starts SUMO.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

MAP = 'wz3to2'
NATIVE = True
ROLE = 'final_test'
HORIZON_S = 1200
WARMUP_S = 300
IDENTITY = ('arm', 'world_id', 'world', 'checkpoint', 'window', 'panel_sha256')
LOG_LIMIT = 200


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


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def mappo_load_check(path, sha256):
    """Zero physics: the comparator file hashes as bound and torch-loads; reports its layout."""
    import torch
    if _sha(path) != sha256:
        raise ValueError(f'{path} does not hash to {sha256}')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    keys = sorted(saved)[:20] if isinstance(saved, dict) else []
    return dict(status='loaded', sha256=sha256, type=type(saved).__name__, keys=keys,
                format=saved.get('format') if isinstance(saved, dict) else None)


def _record(spec, J, safety, **extra):
    return dict({k: deepcopy(spec[k]) for k in IDENTITY}, status='graded', J_veh_s=float(J),
                safety={k: int(v) for k, v in safety.items()}, **extra)


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


def _policy_episode(spec, cfg, work):
    from . import accel_plant, launch_production as production, protocol, rerun_cpus as rc
    from . import track2_backend as p1
    from .rerun_tracks_backend import safety
    from .v2v_runner import patched
    world = _check_spec(spec)
    native = production.backend_config(cfg)
    ck = spec['checkpoint']
    arm, update = ck['arm'], int(ck['update'])
    path = Path(native['run_root'])/f'{arm}_checkpoint_{update:03}.pt'
    if Path(ck['path']) != path or rc.sha(path) != ck['file_sha256']:
        raise ValueError('evaluation checkpoint changed after registration')
    executor = native['checkpoint']
    if spec['arm'] == 'MAPPO':
        if ck.get('kind') != 'mappo_comparator' or (arm, update) != ('A', 0):
            raise ValueError('MAPPO arm = bound comparator executor + the common u0 (zero residual)')
        if _sha(ck['executor_path']) != ck['executor_sha256']:
            raise ValueError('MAPPO comparator checkpoint differs from its bound sha256')
        executor = ck['executor_path']
    weights = production.checkpoint_weights(native, arm, update)
    holder = {}
    log = DecisionLog(spec.get('capture'))

    def loader(values):
        holder['policy'] = production.load_policy(values, native)
        return holder['policy']

    def hook(controller, job, prefixes):
        policy = holder['policy']
        if isinstance(policy, p1.EpisodePolicy):
            original = controller.apply

            def apply(rows, commands, now):
                policy.observe(rows, now, command_commitments=controller.active_plans)
                return original(rows, commands, now)
            controller.apply = apply
        log.controller = controller
    hook.after_step = log.after_step
    job = dict(id=Path(work).name, path=str(work), world=world, role=ROLE, arm=arm, weights=weights,
               runtime=native['runtime'], checkpoint=executor, noise_seed=spec['noise_seed'],
               deterministic=True, guided=False, v2v=True, sim_s=HORIZON_S, stop=HORIZON_S,
               warm_history=native['warm_history'], accel_bounds=native['accel_bounds'])
    # As launch_evaluation.episode: the original registered guards (both route map worlds to N1).
    with patched(accel_plant, dict(guard=protocol.guard, install_seed_scope=protocol.install_seed_scope)):
        result = accel_plant.episode(job, controller_hook=hook, policy_loader=loader)
    extra = {}
    if spec['arm'] in ('B', 'R'):
        extra['decision_log'] = log.log
    if spec.get('capture') is not None:
        extra['decision'] = log.first
    return _record(spec, result['totals']['team_delay_vs'], safety(result, Path(work)/'observable.jsonl.gz'), **extra)


def _native_episode(spec, cfg, work):
    import types
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
