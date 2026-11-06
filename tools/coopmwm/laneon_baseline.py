"""Registered EXTRA frozen final-evaluation baseline: lane-on MAPPO ``aecd7d70``.

CLEAN3_QUOTE.json phase ``v2v_final_laneon_mappo_aecd7d70`` (owner 2026-11-06 01:16
"keep it as the baseline for now...", MAIN 01:16): the untouched lane-controlling
MAPPO export runs on the SAME 96 paired u24 worlds/seeds as A/B/R, full 1200 s
windows, DESCRIPTIVE ONLY - the registered primary comparator stays the speed-only
MAPPO (owner Oct30; REGISTRATION.md:4936) and the u24 synergy criterion is
unchanged. Charging uses the budgeted phase; nothing here gates the run.

The adapter loads the exact C4 export through the original C4 loader
(``wms.control.policy``: CavPolicy acceleration head + masked 3-way lane head +
its own group encoder/lane rules, deterministic argmax) with a hard SHA pin; a
lane-stripped (speed-only), foreign-trainer or byte-modified export is refused.
Execution keeps its original E3 registration regime: the executor's lane targets
flow ONLY through the env's existing lane command path -
``CavSumoEnv.step(accel, lanes=...)`` puts every finite-command CAV under
``laneChangeMode`` LC_MODE_POLICY = 513 on EVERY policy tick (stay-only ticks
included) and issues ``changeLane`` with SUMO's own gap/braking safety; HDVs and
non-commanded slots are never touched, and a lane target on a non-CAV slot is a
hard refusal. The paired bootstrap CIs vs each arm reuse the registered T1D
``planner_eval`` estimator (whole-world resamples, same seed/level); they are
reported, never gated.

CPU tests use fakes only. ``episode`` needs the frozen SUMO runtime and runs on
the qualified CPU host; it is never executed by the CPU test suite.
"""
from collections import Counter
from copy import deepcopy
import math
import os
from pathlib import Path

import torch

from . import rerun_cpus as rc
from .exact_grade import _seal, verify_sealed

SCHEMA = 'coopmwm/laneon_baseline/v1'
RECEIPT_SCHEMA = 'coopmwm/laneon_baseline/receipt/v1'
ANALYSIS_SCHEMA = 'coopmwm/laneon_baseline/analysis/v1'
#: registered baseline identity (clean3_restart.LANE_ON_BASELINE / AUTHORITY.md)
NAME = 'mappo_laneon_aecd7d70'
COMMIT = 'aecd7d70'
CHECKPOINT_SHA256 = 'aecd7d70ccce2995a10d03d9400f587040b1d3b435eebeab2561768ba57c3564'
#: CLEAN3_QUOTE.json allocated phase (96 x 1200 base + >=15% margin), separate
#: from the A/B/R v2v_final phase so every lane-on second charges its own budget.
PHASE = 'v2v_final_laneon_mappo_aecd7d70'
PHASE_ALLOCATED_SIM_S = 133_200
#: the env's existing lane path applies this laneChangeMode to commanded CAVs
#: (traffic_shepherd.c4.env.LC_MODE_POLICY, the checkpoint's E3 registration regime)
LANE_CHANGE_MODE = 513
ROLE = ('extra frozen final-evaluation baseline; descriptive only; '
        'primary comparator remains speed-only MAPPO')
AUTHORITY = 'owner 2026-11-06 01:16 "keep it as the baseline for now..."; MAIN 01:16'
EPISODES = rc.TEST_EPISODES
HORIZON_S = rc.EPISODE_SIM_S
WARMUP_S = 300
#: same counters/estimator family as the u24 paired contract (rerun_cpus.FINAL_EVALUATION)
SAFETY_COUNTERS = ('collisions', 'teleports', 'hard_brakes', 'ttc_events')
DESCRIPTIVE_NOTE = ('descriptive extra baseline; never a gate and never the primary '
                    'contrast; the registered primary comparator remains speed-only MAPPO')

OUTCOME_KEYS = ('J_veh_s', 'safety', 'lane_commands', 'executor', 'result_ref')
RECEIPT_KEYS = ('schema', 'baseline', 'phase', 'job_id', 'world_id', 'seed',
                'checkpoint_sha256', 'lane_change_mode', 'J_veh_s', 'safety',
                'lane_commands', 'episode_end_s', 'executor', 'result_ref',
                'final_worlds_sha256', 'receipt_sha256')
LANE_COMMAND_KEYS = ('issued', 'completed', 'not_completed', 'noop', 'deferred_edge_end',
                     'refused_junction', 'refused_not_allowed', 'refused_no_lane',
                     'refused_not_policy')


def registered_entry(cfg):
    """The single registered lane-on baseline entry (fail closed before any dispatch).

    Requires the clean3 ``extra_final_baselines`` registration with the exact
    quoted role/phase/SHA and a staged ``checkpoint`` path; a quoted-but-unbound
    slot is refused, never silently skipped (W4: do not treat the quoted slot as
    a completed evaluation).
    """
    entries = [b for b in (cfg.get('extra_final_baselines') or []) if isinstance(b, dict)]
    found = [b for b in entries if b.get('name') == NAME]
    if not found:
        raise ValueError('REFUSED: lane-on baseline not registered in this configuration')
    entry = found[0]
    if (len(found) != 1 or entry.get('role') != ROLE or entry.get('phase') != PHASE
            or entry.get('episodes') != EPISODES or entry.get('lane_change_mode') != LANE_CHANGE_MODE
            or entry.get('sha256') != CHECKPOINT_SHA256):
        raise ValueError('REFUSED: lane-on baseline registration differs from the '
                         'CLEAN3_QUOTE.json baseline')
    path = entry.get('checkpoint')
    if not isinstance(path, str) or not path:
        raise ValueError('REFUSED: lane-on baseline requires the staged aecd7d70 '
                         'checkpoint path (laneon=dict(checkpoint=...))')
    rc.guard_path(path)
    return deepcopy(entry)


def staged_checkpoint(entry):
    """Absolute path of the staged aecd7d70 export, SHA-verified BEFORE any charge."""
    path = Path(rc.guard_path(entry['checkpoint']))
    if not path.is_file() or rc.sha(path) != CHECKPOINT_SHA256:
        raise ValueError('REFUSED: staged lane-on checkpoint missing or its SHA differs '
                         'from the registered aecd7d70 export: '+str(path))
    return path


def load_executor(path, *, expected_sha256=CHECKPOINT_SHA256):
    """SHA-pinned load of the frozen lane-on MAPPO through the original C4 loader.

    Returns ``(PolicyExecutor, meta)``. Refuses a wrong-hash file, a lane-stripped
    (speed-only) export and any foreign trainer/format: only the exact E3
    lane-aware export ``aecd7d70`` may run as this baseline.
    """
    from wms.control.policy import PolicyExecutor, load_policy_components

    path = Path(rc.guard_path(path))
    if rc.sha(path) != expected_sha256:
        raise ValueError('REFUSED: lane-on MAPPO checkpoint SHA differs from the '
                         'registered aecd7d70 export: '+str(path))
    policy, encoder, lane_rules, payload = load_policy_components(str(path), device='cpu')
    if payload.get('format') != 'c4_dreamer/3' or payload.get('trainer') != 'mappo':
        raise ValueError('REFUSED: lane-on baseline requires the mappo c4_dreamer/3 export')
    if payload.get('lane_head') is not True:
        raise ValueError('REFUSED: lane-on baseline requires the 3-way lane head '
                         '(speed-only / lane-stripped exports are refused)')
    executor = PolicyExecutor(policy, encoder, lane_rules=lane_rules,
                              deterministic=True, device='cpu')
    if not executor.has_lane_head:
        raise ValueError('REFUSED: lane-on executor lost its lane head')
    meta = dict(format=payload['format'], trainer=payload['trainer'], lane_head=True,
                iteration=payload.get('iteration'), file_sha256=expected_sha256,
                accel_min_mps2=executor.accel_min, accel_max_mps2=executor.accel_max,
                lane_change_mode=LANE_CHANGE_MODE)
    return executor, meta


class LaneOnController:
    """Per-tick adapter: the frozen executor's lane-aware output through the env's
    existing lane path, with lane commands restricted to CAV slots.

    ``step(env)`` computes ``(accel, lanes) = executor.act_with_lane(env.state())``
    (the executor already applies its current- and target-lane safety) and hands
    both to ``env.step(accel, lanes=lanes)``; the env alone issues ``changeLane``
    and LC_MODE_POLICY. A lane target on a slot without a finite CAV command, or
    an off-grid lane value, is a hard refusal - lane commands are CAV-only.
    """

    def __init__(self, executor):
        self.executor = executor
        self.reset()

    def reset(self):
        self.executor.reset()
        self.lane_commands = Counter()
        self.ticks = 0

    def act(self, env):
        import numpy as np

        state = env.state()
        accel, lanes = self.executor.act_with_lane(state)
        accel = np.asarray(accel)
        lanes = np.asarray(lanes)
        if lanes.shape != accel.shape:
            raise ValueError('REFUSED: lane-on executor lane/accel shape mismatch')
        if not set(np.unique(lanes)).issubset({-1, 0, 1, 2}):
            raise ValueError('REFUSED: lane target outside the WM lane grid {-1,0,1,2}')
        commanded = np.isfinite(accel)
        cav = (np.asarray(state['is_cav']).reshape(-1).astype(bool)
               if 'is_cav' in state else np.ones(commanded.shape, bool))
        if (lanes[~(commanded & cav)] != -1).any():
            raise ValueError('REFUSED: lane command for a non-CAV/uncommanded slot')
        return accel, lanes

    def step(self, env):
        accel, lanes = self.act(env)
        _, info = env.step(accel, lanes=lanes)
        self.ticks += 1
        self.lane_commands.update(info.get('lane_commands', {}))
        return info


def specs(registration, entry):
    """Same-world paired job specs for the baseline: the IDENTICAL world dicts,
    world ids, full windows and panel SHA as the A/B/R ``planner_eval`` specs,
    with the SHA-bound aecd7d70 checkpoint in place of an arm checkpoint."""
    verify_sealed(registration, 'final_worlds_sha256')
    if registration.get('role') != 'train' or len(registration['worlds']) != EPISODES:
        raise ValueError(f'REFUSED: lane-on baseline requires the {EPISODES} registered '
                         'paired TRAIN worlds')
    checkpoint = dict(path=str(Path(rc.guard_path(entry['checkpoint']))),
                      file_sha256=CHECKPOINT_SHA256)
    return [(dict(branch='shared_final', baseline=NAME, arm=NAME,
                  world_id=row['world_id'], world=deepcopy(row['world']),
                  checkpoint=deepcopy(checkpoint), sim_s=HORIZON_S,
                  window=dict(start_s=0, end_s=HORIZON_S),
                  lane_change_mode=LANE_CHANGE_MODE,
                  final_worlds_sha256=registration['final_worlds_sha256']), {})
            for row in registration['worlds']]


def make_receipt(spec, outcome):
    """Validate an episode outcome against its spec and return the sealed receipt."""
    if not isinstance(outcome, dict):
        raise ValueError('REFUSED: lane-on outcome must be an object')
    if set(outcome)-set(OUTCOME_KEYS):
        raise ValueError(f'REFUSED: unregistered lane-on outcome keys '
                         f'{sorted(set(outcome)-set(OUTCOME_KEYS))}')
    if {'J_veh_s', 'safety'}-set(outcome):
        raise ValueError('REFUSED: lane-on outcome needs J_veh_s and safety')
    record = dict(schema=RECEIPT_SCHEMA, baseline=NAME, phase=PHASE,
                  job_id=f'{NAME}_{spec["world_id"]}', world_id=spec['world_id'],
                  seed=spec['world']['seed'], checkpoint_sha256=CHECKPOINT_SHA256,
                  lane_change_mode=LANE_CHANGE_MODE,
                  J_veh_s=_finite(outcome['J_veh_s']), safety=_safety(outcome['safety']),
                  lane_commands=_lane_commands(outcome.get('lane_commands', {})),
                  episode_end_s=spec['window']['end_s'],
                  executor=str(outcome.get('executor', '')),
                  result_ref=str(outcome.get('result_ref', '')),
                  final_worlds_sha256=spec['final_worlds_sha256'])
    record = _seal(record, 'receipt_sha256')
    validate_receipt(record, spec)
    return record


def validate_receipt(record, spec=None):
    """Sealed lane-on receipt integrity; a bound spec additionally enforces identity."""
    if record.get('schema') != RECEIPT_SCHEMA or record.get('baseline') != NAME:
        raise ValueError('REFUSED: not a lane-on baseline receipt')
    verify_sealed(record, 'receipt_sha256')
    if set(record) != set(RECEIPT_KEYS):
        raise ValueError('REFUSED: lane-on receipt keys differ from the schema')
    for key, constant in (('phase', PHASE), ('checkpoint_sha256', CHECKPOINT_SHA256),
                          ('lane_change_mode', LANE_CHANGE_MODE), ('baseline', NAME)):
        if record[key] != constant:
            raise ValueError(f'REFUSED: lane-on receipt {key} differs from the registration')
    if type(record.get('seed')) is not int or record['seed'] < 0:
        raise ValueError('REFUSED: lane-on receipt seed must be a nonnegative int')
    _finite(record['J_veh_s'])
    _safety(record['safety'])
    _lane_commands(record['lane_commands'])
    if type(record.get('episode_end_s')) is not int or record['episode_end_s'] != HORIZON_S:
        raise ValueError('REFUSED: lane-on receipt episode_end_s differs from the full window')
    if spec is not None:
        expected = dict(world_id=spec['world_id'], seed=spec['world']['seed'],
                        final_worlds_sha256=spec['final_worlds_sha256'],
                        lane_change_mode=spec['lane_change_mode'])
        for key, value in expected.items():
            if record[key] != value:
                raise ValueError(f'REFUSED: lane-on receipt {key} differs from its job')
    return True


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('REFUSED: lane-on J_veh_s must be a finite number')
    return float(value)


def _safety(value):
    if not isinstance(value, dict) or set(value) != set(SAFETY_COUNTERS):
        raise ValueError(f'REFUSED: lane-on safety keys must be exactly {sorted(SAFETY_COUNTERS)}')
    for key, count in value.items():
        if isinstance(count, bool) or type(count) is not int or count < 0:
            raise ValueError(f'REFUSED: lane-on safety.{key} must be a nonnegative int')
    return dict(value)


def _lane_commands(value):
    if not isinstance(value, dict) or set(value)-set(LANE_COMMAND_KEYS):
        raise ValueError('REFUSED: lane-on lane_commands keys differ from the env counters')
    for key, count in value.items():
        if isinstance(count, bool) or type(count) is not int or count < 0:
            raise ValueError(f'REFUSED: lane-on lane_commands.{key} must be a nonnegative int')
    return dict(value)


def analyse(*, registration, records, arm_records, contract):
    """Descriptive paired whole-world bootstrap CIs of the baseline minus EACH arm.

    Completeness is exact (every registered world exactly once in the baseline and
    in every arm); the estimator is the registered T1D ``planner_eval`` one on the
    u24 contract bootstrap. Reported only - never a gate, never the primary
    contrast. ``arm_records`` are the graded A/B/R results of the same-panel
    u24 paired receipt.
    """
    from .planner_eval import _paired_ci

    verify_sealed(registration, 'final_worlds_sha256')
    worlds = registration['worlds']
    want = {row['world_id']: row for row in worlds}
    have = _validated_records(records, want, registration['final_worlds_sha256'])
    if set(have) != set(want):
        raise ValueError('REFUSED: incomplete lane-on panel (every world exactly once): '
                         f'missing {sorted(set(want)-set(have))}')
    arms = list(contract['arms'])
    reference = _validated_arm_records(arm_records, arms, want,
                                       registration['final_worlds_sha256'])
    bootstrap = contract['bootstrap']
    alpha = 1.0-float(bootstrap['level'])
    paired_ci = {}
    for arm in arms:
        deltas = [have[w['world_id']]['J_veh_s']-reference[(arm, w['world_id'])]['J_veh_s']
                  for w in worlds]
        ci = _paired_ci(deltas, bootstrap['reps'], bootstrap['seed'], alpha)
        paired_ci[f'{NAME}-{arm}'] = dict(n=ci['n_worlds'], mean=ci['mean_delta_vs'],
                                          low=ci['ci95'][0], high=ci['ci95'][1], native=ci)
    safety = {NAME: {k: sum(have[w['world_id']]['safety'][k] for w in worlds)
                     for k in SAFETY_COUNTERS}}
    safety.update({arm: {k: sum(reference[(arm, w['world_id'])]['safety'][k] for w in worlds)
                         for k in SAFETY_COUNTERS} for arm in arms})
    analysis = dict(schema=ANALYSIS_SCHEMA, baseline=NAME, phase=PHASE,
                    descriptive=True, note=DESCRIPTIVE_NOTE, comparisons=sorted(paired_ci),
                    final_worlds_sha256=registration['final_worlds_sha256'],
                    paired_ci=paired_ci, safety=safety, bootstrap=dict(bootstrap),
                    charged_phase=PHASE)
    analysis['analysis_sha256'] = _seal(analysis, 'analysis_sha256')['analysis_sha256']
    return analysis


def _validated_records(records, worlds, final_worlds_sha256):
    have = {}
    for row in records:
        if row.get('arm') != NAME or row.get('baseline') != NAME:
            raise ValueError('REFUSED: lane-on record baseline identity differs')
        world_id = row.get('world_id')
        if world_id not in worlds:
            raise ValueError(f'REFUSED: lane-on record outside the registered panel: {world_id}')
        if world_id in have:
            raise ValueError(f'REFUSED: duplicate lane-on record for world {world_id}')
        if (row.get('status') != 'graded' or row.get('world') != worlds[world_id]['world']
                or row.get('window') != dict(start_s=0, end_s=HORIZON_S)
                or row.get('final_worlds_sha256') != final_worlds_sha256
                or row.get('lane_change_mode') != LANE_CHANGE_MODE):
            raise ValueError(f'REFUSED: lane-on record world/window/panel binding differs: {world_id}')
        checkpoint = row.get('checkpoint') or {}
        if checkpoint.get('file_sha256') != CHECKPOINT_SHA256:
            raise ValueError(f'REFUSED: lane-on record checkpoint SHA differs: {world_id}')
        if not isinstance(row.get('J_veh_s'), (int, float)) or not math.isfinite(row['J_veh_s']):
            raise ValueError(f'REFUSED: finite lane-on J_veh_s required: {world_id}')
        _safety(row.get('safety', {}))
        have[world_id] = row
    return have


def _validated_arm_records(records, arms, worlds, final_worlds_sha256):
    reference = {}
    for row in records:
        key = (row.get('arm'), row.get('world_id'))
        if key[0] not in arms or key[1] not in worlds:
            raise ValueError('REFUSED: arm record outside the same paired panel')
        if key in reference or row.get('status') != 'graded':
            raise ValueError('REFUSED: duplicate/ungraded arm record')
        if (row.get('world') != worlds[key[1]]['world']
                or row.get('final_worlds_sha256') != final_worlds_sha256):
            raise ValueError('REFUSED: arm record world/panel binding differs')
        if not isinstance(row.get('J_veh_s'), (int, float)) or not math.isfinite(row['J_veh_s']):
            raise ValueError('REFUSED: finite arm J_veh_s required')
        _safety(row.get('safety', {}))
        reference[key] = row
    missing = [(a, w) for a in arms for w in worlds if (a, w) not in reference]
    if missing:
        raise ValueError(f'REFUSED: same-panel arm records incomplete: {missing[0]}')
    return reference


# ---------------------------------------------------------------- execution (cpuV)

def episode(spec, cfg, work):
    """One full-window paired-world episode of the frozen lane-on baseline.

    Real SUMO execution on the qualified CPU host only (never in CPU tests):
    binds the SAME hash-verified frozen e3final runtime the A/B/R final episodes
    import (plus the clean3 G5 overlay when COOPMWM_G5_RUNTIME=1), loads the
    SHA-pinned export through the C4 loader and runs its full lane-aware path at
    the 1 s control cadence: native 300 s warmup, then 900 controlled ticks with
    ``env.step(accel, lanes=...)`` - the env's existing lane path, LC_MODE_POLICY
    513 on every commanded CAV every tick. Grading reuses the harness accounting
    (cumulative team_delay_vs J, collisions/teleports/hard-brake totals, TTC
    events from the observable tape). Exactly ``spec['sim_s']`` physics seconds
    may run (precharge guard).
    """
    import gzip
    import json
    import random

    import numpy as np

    from tools.coopnative.plant import accumulate, observations, setup as runtime_setup

    from .accel_protocol import guard, install_seed_scope

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    work = Path(rc.guard_path(work))
    work.mkdir(parents=True, exist_ok=True)
    executor, meta = load_executor(spec['checkpoint']['path'])
    world = spec['world']
    from . import protocol
    allowed = [w for index in (0, 1) for w in protocol.worlds('train', index)]
    if world not in allowed:
        raise ValueError('lane-on evaluation only accepts originally registered TRAIN worlds')
    runtime_setup(cfg['runtime'])
    from .g5_runtime import install as g5_install
    g5_install(cfg['runtime'])       # no-op unless COOPMWM_G5_RUNTIME=1; before env import
    guard(world, 'train')
    install_seed_scope()
    allowed_sim_s = int(spec['sim_s'])
    with (work/'CLAIM.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), id=f'{NAME}_{spec["world_id"]}',
                                     charged_sim_s=allowed_sim_s)))
    random.seed(world['seed'])
    np.random.seed(world['seed'])
    torch.manual_seed(world['seed'])
    from traffic_shepherd.c4.env import CavSumoEnv
    import traci
    env = None
    original, actual = traci.simulationStep, 0.0

    def counted(*args, **kwargs):
        nonlocal actual
        if actual+.5 > allowed_sim_s:
            raise ValueError('physics exceeds precharge')
        before = traci.simulation.getTime()
        result = original(*args, **kwargs)
        actual += traci.simulation.getTime()-before
        return result
    traci.simulationStep = counted
    controller = LaneOnController(executor)
    totals = dict.fromkeys(('team_delay_vs', 'road_delay_vs', 'source_wait_vs', 'collisions',
                            'teleports', 'ttc_below_3_vs', 'hard_brake'), 0.)
    metrics = {}
    try:
        env = CavSumoEnv(k_slots=192, controller='direct', ttc=True, physics_log=False,
                         population='textbook', scenario_root=work/'scenarios')
        env.reset(seed=world['seed'], demand_vph=world['demand'],
                  penetration=world['penetration'], warmup_s=WARMUP_S,
                  horizon_s=HORIZON_S-WARMUP_S+1, controller='direct')
        with gzip.open(work/'observable.jsonl.gz', 'xt', encoding='utf-8') as tape:
            for now in range(WARMUP_S, spec['window']['end_s']):
                rows = observations(env)
                info = controller.step(env)
                accumulate(totals, info['interval'])
                issued = getattr(getattr(env, 'ownership', None), '_last', {})
                for row in rows:
                    record = issued.get(row['vid'])
                    row.update(controller=NAME if row['is_cav'] else 'native',
                               issued_acceleration_mps2=(record.issued_acceleration_mps2
                                                         if record is not None else None))
                tape.write(json.dumps(dict(time_s=now, vehicles=rows, actions=[],
                                           lane_commands=info['lane_commands']),
                                      sort_keys=True)+'\n')
                if (info['done'] and now+1 != spec['window']['end_s']) or \
                        info['interval'].get('dropped_vehicle_steps', 0):
                    raise ValueError('early termination/slot anomaly')
                if now+1 in (900, 1200):
                    metrics[str(now+1-WARMUP_S)] = dict(totals)
    finally:
        if env is not None:
            env.close()
        traci.simulationStep = original
        (work/'CLOSE.json').write_text(json.dumps(dict(actual_sim_s=actual,
                                                       charged_sim_s=allowed_sim_s))+'\n',
                                       encoding='utf-8')
    from .replay_bank import ttc_events
    vector = dict(collisions=int(totals['collisions']), teleports=int(totals['teleports']),
                  hard_brakes=int(totals['hard_brake']),
                  ttc_events=int(ttc_events(work/'observable.jsonl.gz',
                                            spec['window']['start_s'], spec['window']['end_s'])))
    result = dict({k: deepcopy(spec[k]) for k in ('arm', 'world_id', 'world', 'checkpoint',
                                                  'window', 'final_worlds_sha256', 'baseline',
                                                  'lane_change_mode')},
                  status='graded', J_veh_s=float(totals['team_delay_vs']),
                  safety=vector, lane_commands=dict(controller.lane_commands),
                  executor=dict(module='tools.coopmwm.laneon_baseline', deterministic=True,
                                **meta), result_ref=str(work/'RESULT.json'))
    (work/'RESULT.json').write_text(json.dumps(dict(
        id=f'{NAME}_{spec["world_id"]}', world=world, arm=NAME, metrics=metrics,
        totals=dict(totals), actual_sim_s=actual, charged_sim_s=allowed_sim_s,
        lane_commands=dict(controller.lane_commands), safety=vector,
        lane_change_mode=LANE_CHANGE_MODE), sort_keys=True)+'\n', encoding='utf-8')
    return result
