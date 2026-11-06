"""Registered EXTRA frozen u24 baseline: lane-on MAPPO ``aecd7d70`` (descriptive only).

Owner 2026-11-06 01:16 ("keep it as the baseline for now...", MAIN 01:16) and
CLEAN3_QUOTE.json phase ``v2v_final_laneon_mappo_aecd7d70`` (96 x 1200 base, 133200
allocated): the untouched lane-controlling MAPPO export runs on the SAME 96 paired
u24 worlds as A/B/R, full 1200 s windows, and is reported with paired whole-world
bootstrap CIs against each arm. The registered primary comparator stays the
speed-only MAPPO d9042cc5 and the u24 synergy criterion is unchanged; nothing here
gates, pauses or edits the primary evaluation.

Separate entry point (W4 gap, CLEAN3_FINAL_BASELINE_WIRING_W4.md): the running clean3
source 3e5bc9f9 is hash-pinned and its monitor imports ``launch_evaluation`` lazily,
so this module changes no existing file. Option off (never invoked) = the u24 paired
evaluation and its receipt are byte-identical. It only READS the sealed
``evaluation/PAIRED_u024.json`` and writes ``evaluation/LANEON_u024.json``.

Budget: the running registration (``clean3_restart.PHASES``, sealed as
``CONFIG.json`` ``clean3_budget``) carries this baseline INSIDE ``v2v_final`` =
531600 = 398400 (A/B/R) + 133200 (quote phase ``v2v_final_laneon_mappo_aecd7d70``);
``ops/clean3_chain.coverage`` asserts v2v_final need = 4 x 96 x 1200. Charges go to
the phase holding the allocation in the sealed registration: the quote phase key
when a registration has one, else ``v2v_final`` with job ids ``laneon_u024_NNNN``
and a hard sub-cap of 133200 on those ids. The A/B/R u24 charges must already be
reserved, so the baseline can never crowd out the primary evaluation.

Execution: the SAME frozen plant the A/B/R final episodes use builds each world
(``tools.leverprobe.backend.setup``/``Backend`` with the configuration's runtime and
checkpoint binding, the original registered TRAIN guard, the bound G5 env). The
per-tick loop mirrors ``tools.coopfree.backend.tick`` (slot-reuse memory/dwell
masking, team-delay totals, observable tape, anomaly stop) except that commands come
from the SHA-pinned aecd7d70 executor with its 3-way lane head, and its lane targets
go through the env's existing lane path ``env.step(cmd, lanes=lanes)`` without
``lane_actors``: laneChangeMode 513 on every finite-command CAV every tick (stay-only
ticks included), SUMO's own gap/brake safety, i.e. its original E3 regime. Lane or
acceleration commands on any non-CAV slot are refused before the env sees them.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import math
import multiprocessing
import os
from pathlib import Path

from . import rerun_cpus as rc
from .clean3_restart import LANE_ON_BASELINE
from .exact_grade import _seal, verify_sealed
from .protocol import digest

NAME = LANE_ON_BASELINE['name']                  # 'mappo_laneon_aecd7d70'
COMMIT = LANE_ON_BASELINE['commit']              # 'aecd7d70'
#: S-LANEMAPPO independently hashed local + cpuV copies (LANE_MAPPO_FEASIBILITY.md 2.1)
CHECKPOINT_SHA256 = 'aecd7d70ccce2995a10d03d9400f587040b1d3b435eebeab2561768ba57c3564'
CHECKPOINT_FORMAT = 'c4_dreamer/3'
UPDATE = 24
PREFIX = f'laneon_u{UPDATE:03}'
PAIRED = f'evaluation/PAIRED_u{UPDATE:03}.json'
TARGET = f'evaluation/LANEON_u{UPDATE:03}.json'
FAILED = f'evaluation/LANEON_u{UPDATE:03}_FAILED.json'
PRIMARY_PREFIX = f'paired_u{UPDATE:03}'
QUOTE_PHASE = 'v2v_final_laneon_mappo_aecd7d70'
QUOTE_BASE_SIM_S = rc.TEST_EPISODES*rc.EPISODE_SIM_S       # 115200
QUOTE_ALLOCATED_SIM_S = 133_200                            # CLEAN3_QUOTE.json, >=15% margin
MERGED_PHASE = 'v2v_final'
MERGED_ALLOCATION_SIM_S = 398_400+QUOTE_ALLOCATED_SIM_S    # clean3_restart.PHASES v2v_final
#: traffic_shepherd.c4.env LC_MODE_POLICY, applied by the env when a lanes tensor is passed
LANE_CHANGE_MODE = 513
WARMUP_S = 300
LANE_GRID = (-1, 0, 1, 2)
SCHEMA = 'coopmwm/laneon_eval/v1'
RECEIPT_SCHEMA = 'coopmwm/laneon_eval/receipt/v1'
ANALYSIS_SCHEMA = 'coopmwm/laneon_eval/analysis/v1'
RECEIPT_KEYS = ('schema', 'baseline', 'job_id', 'world_id', 'world', 'seed', 'checkpoint_sha256',
                'lane_change_mode', 'window', 'final_worlds_sha256', 'charged_phase',
                'charged_sim_s', 'actual_sim_s', 'J_veh_s', 'safety', 'lane_commands',
                'lane_request_ticks', 'warmup_prefix_sha256', 'result_ref', 'receipt_sha256')
IDENTITY_KEYS = ('arm', 'world_id', 'world', 'checkpoint', 'window', 'final_worlds_sha256',
                 'lane_change_mode')
NOTE = ('EXTRA frozen lane-on MAPPO aecd7d70 baseline (owner 2026-11-06 01:16): descriptive '
        'only, never a gate and never the primary contrast; the registered primary comparator '
        'remains speed-only MAPPO d9042cc5 and the u24 synergy criterion is unchanged. '
        'Cross-action-regime comparison: lane-on vs speed-only-base arms.')
SIGN = 'delta = lane-on minus arm J_veh_s (team delay veh-s); negative = lane-on lower delay'


def _refuse(message):
    raise ValueError('REFUSED: '+message)


# ---------------------------------------------------------------- registration / identity

def registered_entry(cfg):
    """The clean3 ``extra_final_baselines`` slot, exactly as clean3_restart registered it."""
    found = [b for b in (cfg.get('extra_final_baselines') or [])
             if isinstance(b, dict) and b.get('name') == NAME]
    if not found:
        _refuse('lane-on MAPPO baseline is not registered in this configuration')
    if len(found) != 1:
        _refuse('lane-on MAPPO baseline registered more than once')
    entry = deepcopy(found[0])
    if {k: v for k, v in entry.items() if k != 'checkpoint'} != LANE_ON_BASELINE:
        _refuse('lane-on baseline registration differs from clean3_restart.LANE_ON_BASELINE')
    return entry


def staged_checkpoint(entry, path=None, *, expected_sha256=None):
    """Absolute path of the staged aecd7d70 export, SHA-verified BEFORE any charge."""
    expected = expected_sha256 or CHECKPOINT_SHA256
    bound = entry.get('checkpoint')
    if path is None and bound is None:
        _refuse('the staged aecd7d70 checkpoint path is required')
    if path is not None and bound is not None and Path(path) != Path(bound):
        _refuse('checkpoint path differs from the registered binding')
    path = Path(rc.guard_path(path if path is not None else bound))
    if not path.is_absolute():
        _refuse('absolute checkpoint path required (spawned workers)')
    if not path.is_file() or rc.sha(path) != expected:
        _refuse('lane-on checkpoint missing or its SHA differs from the registered aecd7d70 '
                'export: '+str(path))
    return path


def payload_meta(path):
    """Export metadata only (weights_only load; no network construction)."""
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != CHECKPOINT_FORMAT or payload.get('trainer') != 'mappo':
        _refuse('lane-on baseline requires the mappo c4_dreamer/3 export')
    if payload.get('lane_head') is not True or not any(
            str(k).startswith('lane_head.') for k in payload.get('policy', {})):
        _refuse('lane-on baseline requires the 3-way lane head (speed-only exports refused)')
    return dict(format=payload['format'], trainer=payload['trainer'], lane_head=True,
                iteration=payload.get('iteration'))


def load_executor(path, *, expected_sha256=None):
    """SHA-pinned deterministic load through the C4 deployment loader (wms.control.policy).

    In a worker this runs AFTER the frozen runtime setup, so ``wms`` resolves to the
    hash-verified e3final runtime that originally evaluated aecd7d70.
    """
    expected = expected_sha256 or CHECKPOINT_SHA256
    if rc.sha(path) != expected:
        _refuse('lane-on MAPPO checkpoint SHA differs from the registered aecd7d70 export: '
                +str(path))
    if (os.environ.get('C4_POLICY_STOCHASTIC') == '1'
            or float(os.environ.get('C4_POLICY_ACCEL_NOISE', '0') or 0) != 0):
        _refuse('deterministic deployment required (C4_POLICY_STOCHASTIC/ACCEL_NOISE set)')
    meta = payload_meta(path)
    from wms.control.policy import load_policy_fn
    executor = load_policy_fn(str(path), device='cpu')
    if getattr(executor, 'deterministic', True) is not True:
        _refuse('lane-on executor is not deterministic')
    if not any(k.startswith('lane_head.') for k in executor.policy.state_dict()):
        _refuse('lane-on executor lost its lane head')
    executor.reset()
    return executor, dict(meta, file_sha256=expected, deterministic=True,
                          loader='wms.control.policy.load_policy_fn',
                          lane_change_mode=LANE_CHANGE_MODE)


def charge_phase(native):
    """The registered phase that carries the lane-on allocation (never an unbudgeted one)."""
    phases = native['budget']['phases']
    if QUOTE_PHASE in phases:
        phase, allocation = QUOTE_PHASE, phases[QUOTE_PHASE]
    elif phases.get(MERGED_PHASE) == MERGED_ALLOCATION_SIM_S:
        phase, allocation = MERGED_PHASE, QUOTE_ALLOCATED_SIM_S
    else:
        _refuse('no registered budget allocation carries the lane-on baseline')
    if type(allocation) is not int or allocation < QUOTE_BASE_SIM_S:
        _refuse('lane-on allocation below its 96 x 1200 base')
    return dict(phase=phase, allocation_sim_s=allocation, quote_phase=QUOTE_PHASE,
                merged_into_v2v_final=phase == MERGED_PHASE, job_prefix=PREFIX,
                rule=('quote phase key when registered; else v2v_final (531600 = 398400 A/B/R '
                      '+ 133200 lane-on) with a 133200 sub-cap on laneon_u024_* ids'))


def specs(registration, checkpoint):
    """Lane-on job specs on the IDENTICAL registered panel the A/B/R specs use."""
    worlds = rc.final_world_registration(dict(final_evaluation=registration))
    if worlds != registration:
        _refuse('unbound lane-on paired world registration')
    cp = dict(path=str(checkpoint), file_sha256=CHECKPOINT_SHA256)
    horizon = rc.EPISODE_SIM_S
    return [(dict(branch='shared_final', arm=NAME, world_id=row['world_id'],
                  world=deepcopy(row['world']), checkpoint=deepcopy(cp), sim_s=horizon,
                  window=dict(start_s=0, end_s=horizon),
                  final_worlds_sha256=registration['final_worlds_sha256'],
                  lane_change_mode=LANE_CHANGE_MODE), {})
            for row in registration['worlds']]


def paired_reference(paired, registration, contract):
    """The sealed u24 A/B/R receipt, re-verified with the registered T1D harness."""
    from . import planner_eval
    verify_sealed(paired, 'receipt_sha256')
    if paired.get('update') != UPDATE or paired.get('status') != 'graded':
        _refuse('lane-on baseline requires the completed u24 paired evaluation')
    if paired.get('registration') != registration:
        _refuse('u24 paired panel differs from the registered final worlds')
    recomputed = planner_eval.analyse_track2(registration=registration,
        checkpoints=paired['checkpoints'], records=paired['results'], contract=contract)
    if digest(recomputed) != digest(paired['analysis']):
        _refuse('u24 paired analysis does not reproduce from its own results')
    return {(r['arm'], r['world_id']): r for r in paired['results']}


def same_worlds(pairs, reference, contract):
    """Every lane-on job is a registered world graded once per arm on the same panel."""
    ids = [spec['world_id'] for spec, _ in pairs]
    if len(set(ids)) != len(ids) or len(ids) != rc.TEST_EPISODES:
        _refuse('lane-on panel must be the 96 distinct registered worlds')
    for spec, _ in pairs:
        for arm in contract['arms']:
            row = reference.get((arm, spec['world_id']))
            if row is None or any(row.get(k) != spec[k] for k in
                                  ('world', 'window', 'final_worlds_sha256')):
                _refuse('lane-on world differs from the paired arm world: '
                        +f'{arm}/{spec["world_id"]}')
    if {w for _, w in reference} != set(ids):
        _refuse('paired arm panel differs from the lane-on panel')
    return dict(worlds=len(ids), seeds=sorted(spec['world']['seed'] for spec, _ in pairs),
                arms=list(contract['arms']))


# ---------------------------------------------------------------- per-tick adapter

def _flag(value):
    import numpy as np
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1).astype(bool)


def cav_only(state, cmd, lanes):
    """Validate the executor output: commands and lane targets on present CAV slots only."""
    import numpy as np
    cmd = np.asarray(cmd, dtype=np.float64)
    lanes = np.asarray(lanes)
    if cmd.ndim != 2 or cmd.shape[0] != 1 or lanes.shape != cmd.shape:
        _refuse('lane-on executor output must be [1,K] accel and [1,K] lanes')
    if lanes.dtype.kind not in 'iu':
        _refuse('lane-on lane targets must be integers')
    flat = lanes.reshape(-1)
    if not set(np.unique(flat).tolist()) <= set(LANE_GRID):
        _refuse('lane target outside the WM lane grid {-1,0,1,2}')
    cav = _flag(state['is_cav'])
    if 'present' in state:
        cav = cav & _flag(state['present'])
    if cav.shape != flat.shape:
        _refuse('state/command slot count mismatch')
    commanded = np.isfinite(cmd.reshape(-1))
    if np.isinf(cmd).any():
        _refuse('infinite lane-on command')
    if (commanded & ~cav).any():
        _refuse('acceleration command for a non-CAV slot')
    if (flat[~(cav & commanded)] != -1).any():
        _refuse('lane command for a non-CAV/uncommanded slot')
    return cmd, lanes.astype(np.int64), int((flat >= 0).sum())


def tick(b, executor, trace, *, plant, cf, canonical):
    """One 1 s control tick; tools.coopfree.backend.tick with the lane-aware executor."""
    import numpy as np
    import torch
    now, rows = float(b.env.now), cf.observations(b)
    ids = np.asarray(b.env.frame.vid_index)
    keep = np.zeros(len(ids), bool) if b.previous is None else (ids == b.previous) & (ids >= 0)
    b.previous = ids.copy()
    for obj in (executor.memory, executor.dwell):   # slot reuse never inherits memory/dwell
        if obj.present is not None:
            obj.present &= torch.as_tensor(keep, device=obj.present.device)[None]
    state = b.env.state()
    cmd, lanes = executor(state, with_lanes=True)
    cmd, lanes, requested = cav_only(state, cmd, lanes)
    b.command_digest.update(canonical(plant.exact(cmd)).encode())
    slot = {vid: i for i, vid in enumerate(b.env.slot_vehicle_ids())}
    flat_cmd, flat_lanes = cmd.reshape(-1), lanes.reshape(-1)
    vehicles = []
    for r in rows:
        v = dict(r)
        if r['is_cav']:
            i = slot[r['vid']]
            a = float(flat_cmd[i])
            v.update(controller=NAME if math.isfinite(a) else 'native',
                     issued_acceleration_mps2=a if math.isfinite(a) else None,
                     lane_target=int(flat_lanes[i]) if flat_lanes[i] >= 0 else None,
                     command_present=math.isfinite(a))
        vehicles.append(v)
    if trace is not None:
        trace.write(canonical(dict(time_s=now, vehicles=vehicles, actions=[]))+'\n')
    # The env's existing lane path: a lanes tensor EVERY tick (no lane_actors) puts every
    # finite-command CAV under LC_MODE_POLICY 513; HDVs are never touched by the env.
    _, info = b.env.step(cmd, lanes=lanes)
    interval = info['interval']
    road, source = interval['speed_deficit_vehicle_seconds'], interval['source_wait_vehicle_seconds']
    b.totals['road_delay_vs'] += road
    b.totals['source_wait_vs'] += source
    b.totals['team_delay_vs'] += road+source
    for key, field in (('collisions', 'collision_involvements'), ('teleports', 'starting_teleports'),
                       ('ttc_below_3_vs', 'ttc_below_3_vehicle_seconds'),
                       ('hard_brake', 'braking_vehicle_seconds')):
        b.totals[key] += interval.get(field, 0.)
    b.observe(b.env)
    if info['done'] or any(interval.get(k, 0) for k in
                           ('collision_involvements', 'starting_teleports', 'dropped_vehicle_steps')):
        raise ValueError('early termination/safety/slot anomaly (lane-on baseline)')
    return requested, dict(info.get('lane_commands', {}))


def episode(spec, cfg, work):
    """One full-window paired-world episode on the frozen plant (qualified CPU host only)."""
    import gzip
    from . import g5_runtime, protocol
    from .accel_storage import write
    world = spec['world']
    allowed = [w for index in (0, 1) for w in protocol.worlds('train', index)]
    if world not in allowed:
        _refuse('lane-on evaluation only accepts originally registered TRAIN worlds')
    if spec.get('arm') != NAME or spec.get('lane_change_mode') != LANE_CHANGE_MODE:
        _refuse('not a lane-on baseline job')
    if spec['checkpoint'].get('file_sha256') != CHECKPOINT_SHA256 or \
            rc.sha(spec['checkpoint']['path']) != CHECKPOINT_SHA256:
        _refuse('lane-on checkpoint changed after registration')
    if cfg.get('g5_runtime') and not g5_runtime.enabled():
        _refuse('A/B/R ran on the bound G5 runtime env; set COOPMWM_G5_RUNTIME=1')
    work = Path(rc.guard_path(work))
    sim_s = int(spec['sim_s'])
    work.mkdir(parents=True, exist_ok=True)
    write(work/'CLAIM.json', dict(pid=os.getpid(), id=work.name, charged_sim_s=sim_s,
                                  baseline=NAME))
    from tools.leverprobe import backend as plant
    from tools.coopfree import backend as cf
    from tools.coopfree.protocol import canonical
    # Same order as accel_plant.episode under launch_evaluation's original TRAIN guard.
    plant.setup(cfg['runtime'], cfg['checkpoint'])
    protocol.install_seed_scope()
    protocol.guard(world, 'train')
    g5_runtime.install(cfg['runtime'])
    plant.guard = lambda w: protocol.guard(w, 'train')
    executor, meta = load_executor(spec['checkpoint']['path'])
    b = plant.Backend(work, world, sim_s, cfg['checkpoint'])
    metrics, lane_commands, request_ticks = {}, Counter(), 0
    try:
        warmup_prefix = b.fingerprint()
        with gzip.open(work/'observable.jsonl.gz', 'xt', encoding='utf-8') as trace:
            for t in range(WARMUP_S, spec['window']['end_s']):
                requested, counts = tick(b, executor, trace, plant=plant, cf=cf,
                                         canonical=canonical)
                lane_commands.update(counts)
                request_ticks += requested > 0
                if t+1 in (900, 1200):
                    metrics[str(t+1-WARMUP_S)] = dict(b.totals)
        result = dict(id=work.name, world=world, arm=NAME, metrics=metrics,
                      prefixes={str(WARMUP_S): warmup_prefix}, final=b.fingerprint(),
                      totals=dict(b.totals), actual_sim_s=b.actual, charged_sim_s=sim_s,
                      lane_commands=dict(lane_commands), lane_request_ticks=request_ticks,
                      lane_change_mode=LANE_CHANGE_MODE, executor=meta)
        write(work/'RESULT.json', result)
    finally:
        b.close()
    from .rerun_tracks_backend import safety
    vector = safety(result, work/'observable.jsonl.gz')
    return dict({k: deepcopy(spec[k]) for k in IDENTITY_KEYS}, status='graded',
                J_veh_s=float(result['totals']['team_delay_vs']), safety=vector,
                lane_commands=dict(lane_commands), lane_request_ticks=request_ticks,
                actual_sim_s=float(result['actual_sim_s']),
                warmup_prefix_sha256=digest(warmup_prefix), executor=meta,
                result_ref=str(work/'RESULT.json'))


# ---------------------------------------------------------------- receipts

def _counters(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        _refuse(f'{label} keys must be exactly {sorted(keys)}')
    for key, count in value.items():
        if type(count) is not int or count < 0:
            _refuse(f'{label}.{key} must be a nonnegative int')
    return dict(value)


def _lane_counts(value):
    if not isinstance(value, dict):
        _refuse('lane_commands must be an object')
    for key, count in value.items():
        if not isinstance(key, str) or type(count) is not int or count < 0:
            _refuse('lane_commands must map names to nonnegative ints')
    return dict(value)


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _refuse(label+' must be a finite number')
    return float(value)


def make_receipt(spec, result, *, job_id, phase, safety_counters):
    """Sealed per-world receipt; the outcome must carry its job's exact identity."""
    if not isinstance(result, dict) or result.get('status') != 'graded':
        _refuse('graded lane-on result required: '+str(job_id))
    if any(result.get(k) != spec[k] for k in IDENTITY_KEYS):
        _refuse('lane-on result world/checkpoint/window binding differs: '+str(job_id))
    record = dict(schema=RECEIPT_SCHEMA, baseline=NAME, job_id=job_id,
        world_id=spec['world_id'], world=deepcopy(spec['world']), seed=spec['world']['seed'],
        checkpoint_sha256=spec['checkpoint']['file_sha256'], lane_change_mode=LANE_CHANGE_MODE,
        window=deepcopy(spec['window']), final_worlds_sha256=spec['final_worlds_sha256'],
        charged_phase=phase, charged_sim_s=int(spec['sim_s']),
        actual_sim_s=_finite(result.get('actual_sim_s'), 'actual_sim_s'),
        J_veh_s=_finite(result.get('J_veh_s'), 'J_veh_s'),
        safety=_counters(result.get('safety'), safety_counters, 'safety'),
        lane_commands=_lane_counts(result.get('lane_commands')),
        lane_request_ticks=result.get('lane_request_ticks'),
        warmup_prefix_sha256=result.get('warmup_prefix_sha256'),
        result_ref=str(result.get('result_ref', '')))
    record = _seal(record, 'receipt_sha256')
    validate_receipt(record, spec, safety_counters=safety_counters)
    return record


def validate_receipt(record, spec=None, *, safety_counters):
    if record.get('schema') != RECEIPT_SCHEMA or record.get('baseline') != NAME:
        _refuse('not a lane-on baseline receipt')
    verify_sealed(record, 'receipt_sha256')
    if set(record) != set(RECEIPT_KEYS):
        _refuse('lane-on receipt keys differ from the schema')
    if record['checkpoint_sha256'] != CHECKPOINT_SHA256 or record['lane_change_mode'] != LANE_CHANGE_MODE:
        _refuse('lane-on receipt checkpoint/lane mode differs from the registration')
    if record['window'] != dict(start_s=0, end_s=rc.EPISODE_SIM_S) or \
            record['charged_sim_s'] != rc.EPISODE_SIM_S:
        _refuse('lane-on receipt must cover the full registered window')
    if type(record['seed']) is not int or record['seed'] != record['world'].get('seed'):
        _refuse('lane-on receipt seed differs from its world')
    if not 0 <= _finite(record['actual_sim_s'], 'actual_sim_s') <= record['charged_sim_s']:
        _refuse('lane-on physics exceeds its charge')
    _finite(record['J_veh_s'], 'J_veh_s')
    _counters(record['safety'], safety_counters, 'safety')
    _lane_counts(record['lane_commands'])
    if type(record['lane_request_ticks']) is not int or record['lane_request_ticks'] < 0:
        _refuse('lane_request_ticks must be a nonnegative int')
    if not isinstance(record['warmup_prefix_sha256'], str) or len(record['warmup_prefix_sha256']) != 64:
        _refuse('warmup prefix digest required')
    if spec is not None and any(record[k] != v for k, v in (
            ('world_id', spec['world_id']), ('world', spec['world']),
            ('final_worlds_sha256', spec['final_worlds_sha256']))):
        _refuse('lane-on receipt world_id/world/panel differs from its job')
    return True


# ---------------------------------------------------------------- analysis

def arm_prefixes(root, paired):
    """Warmup (t=300) physical prefix digests of the A/B/R u24 jobs, read back from disk."""
    from .accel_storage import read
    out = {}
    for index, row in enumerate(paired['results']):
        path = Path(root)/'evaluation'/f'{PRIMARY_PREFIX}_{index:04}'/'RESULT.json'
        try:
            value = read(path)['prefixes'][str(WARMUP_S)]
        except (OSError, KeyError, TypeError, ValueError):
            continue
        out[(row['arm'], row['world_id'])] = digest(value)
    return out


def analyse(*, registration, receipts, paired, contract, prefixes=None):
    """Descriptive paired whole-world bootstrap CIs (lane-on minus EACH arm), sealed."""
    from .planner_eval import _paired_ci
    reference = paired_reference(paired, registration, contract)
    counters = list(contract['safety_counters'])
    want = {row['world_id']: row['world'] for row in registration['worlds']}
    have = {}
    for record in receipts:
        validate_receipt(record, safety_counters=counters)
        w = record['world_id']
        if w not in want or w in have or record['world'] != want[w] or \
                record['final_worlds_sha256'] != registration['final_worlds_sha256']:
            _refuse('lane-on receipt outside/duplicate on the registered panel: '+str(w))
        have[w] = record
    if set(have) != set(want):
        _refuse('every registered world must be graded exactly once (lane-on)')
    order = [row['world_id'] for row in registration['worlds']]
    boot = contract['bootstrap']
    alpha = 1-boot['level']
    paired_ci, safety_ci, means = {}, {}, {NAME: float(sum(have[w]['J_veh_s'] for w in order)/len(order))}
    for arm in contract['arms']:
        deltas = [have[w]['J_veh_s']-reference[arm, w]['J_veh_s'] for w in order]
        ci = _paired_ci(deltas, boot['reps'], boot['seed'], alpha)
        key = f'{NAME}-{arm}'
        paired_ci[key] = dict(n=ci['n_worlds'], mean=ci['mean_delta_vs'], low=ci['ci95'][0],
                              high=ci['ci95'][1], native=ci)
        safety_ci[key] = {}
        for counter in counters:
            d = [have[w]['safety'][counter]-reference[arm, w]['safety'][counter] for w in order]
            c = _paired_ci(d, boot['reps'], boot['seed'], alpha)
            safety_ci[key][counter] = dict(mean=c['mean_delta_vs'], low=c['ci95'][0], high=c['ci95'][1])
        means[arm] = float(sum(reference[arm, w]['J_veh_s'] for w in order)/len(order))
    safety = {NAME: {k: sum(have[w]['safety'][k] for w in order) for k in counters}}
    safety.update({arm: {k: sum(reference[arm, w]['safety'][k] for w in order) for k in counters}
                   for arm in contract['arms']})
    world_check = None
    if prefixes is not None:
        compared, mismatched, unavailable = 0, [], []
        for w in order:
            arms = [prefixes.get((arm, w)) for arm in contract['arms']]
            if any(p is None for p in arms):
                unavailable.append(w)
                continue
            compared += 1
            if len({have[w]['warmup_prefix_sha256'], *arms}) != 1:
                mismatched.append(w)
        world_check = dict(rule='t=300 Backend fingerprint digest equal across lane-on and A/B/R',
                           compared=compared, matched=compared-len(mismatched),
                           mismatched=mismatched, unavailable=unavailable)
    out = dict(schema=ANALYSIS_SCHEMA, baseline=NAME, descriptive=True, gated=False, note=NOTE,
        sign=SIGN, harness='tools.coopmwm.planner_eval._paired_ci (same as analyse_track2)',
        comparisons=list(paired_ci), final_worlds_sha256=registration['final_worlds_sha256'],
        paired_receipt_sha256=paired['receipt_sha256'], bootstrap=deepcopy(boot),
        n_worlds=len(order), mean_J_veh_s=means, paired_ci=paired_ci, safety=safety,
        safety_paired_ci=safety_ci, lane_commands=dict(sum((Counter(have[w]['lane_commands'])
                                                            for w in order), Counter())),
        warmup_prefix_check=world_check)
    return _seal(out, 'analysis_sha256')


# ---------------------------------------------------------------- orchestration

def _accounting(native):
    from .storage import ledger
    return ledger(Path(native.get('accounting_root', native['run_root'])))


def _prepare(cfg, checkpoint):
    """Every zero-physics check, in charge order. Returns the bound execution context."""
    from . import launch_production as production
    from .launch_training import read
    root = Path(cfg['run_root'])
    entry = registered_entry(cfg)
    path = staged_checkpoint(entry, checkpoint)
    meta = payload_meta(path)
    contract = deepcopy(cfg['final_evaluation'])
    if contract != rc.FINAL_EVALUATION:
        _refuse('final evaluation contract differs from the registered u24 contract')
    native = production.backend_config(cfg)
    charge = charge_phase(native)
    registration = rc.final_world_registration(read(root/'ISOLATED_REGISTRATION.json'))
    pairs = specs(registration, path)
    jobs = [dict(id=f'{PREFIX}_{i:04}', sim_s=int(spec['sim_s'])) for i, (spec, _) in enumerate(pairs)]
    return dict(root=root, entry=entry, path=path, meta=meta, contract=contract, native=native,
                charge=charge, registration=registration, pairs=pairs, jobs=jobs)


def dry(cfg, checkpoint=None):
    """Zero physics: registration, checkpoint SHA/payload, panel, budget phase and headroom."""
    ctx = _prepare(cfg, checkpoint)
    rows = _accounting(ctx['native'])
    phase, quoted = ctx['charge']['phase'], sum(j['sim_s'] for j in ctx['jobs'])
    allocated = ctx['native']['budget']['phases'][phase]
    spent = sum(r['sim_s'] for r in rows if r['phase'] == phase)
    primary = sum(r['sim_s'] for r in rows if r['id'].startswith(PRIMARY_PREFIX+'_'))
    pending_primary = 0 if primary else ctx['contract']['sim_s']
    return dict(dry_run=True, new_sim_s=0, baseline=NAME, checkpoint=str(ctx['path']),
        checkpoint_sha256=CHECKPOINT_SHA256, payload=ctx['meta'], charge=ctx['charge'],
        jobs=len(ctx['jobs']), quoted_sim_s=quoted, phase_allocated_sim_s=allocated,
        phase_spent_sim_s=spent, primary_u24_reserved_sim_s=primary,
        fits=spent+pending_primary+quoted <= allocated and quoted <= ctx['charge']['allocation_sim_s'],
        paired_u24_present=(ctx['root']/PAIRED).is_file(),
        receipt_present=(ctx['root']/TARGET).is_file(),
        final_worlds_sha256=ctx['registration']['final_worlds_sha256'],
        seeds=[spec['world']['seed'] for spec, _ in ctx['pairs']])


def verify_receipt_file(value, ctx):
    verify_sealed(value, 'receipt_sha256')
    if value.get('schema') != SCHEMA or value.get('status') != 'graded' or \
            value.get('registration') != ctx['registration'] or \
            value.get('checkpoint', {}).get('file_sha256') != CHECKPOINT_SHA256:
        _refuse('existing LANEON receipt does not bind this registration/checkpoint')
    return value


def _dispatch(ctx, workers):
    native, root = ctx['native'], ctx['root']
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(episode, spec, native, root/'evaluation'/job['id'])
                   for job, (spec, _) in zip(ctx['jobs'], ctx['pairs'])]
        for job, future in zip(ctx['jobs'], futures):
            try:
                results.append(future.result(timeout=3600))
            except Exception as error:      # recorded, never dropped/imputed/retried
                errors.append(dict(job=job['id'], error=repr(error)))
    return results, errors


def run(cfg, checkpoint=None, *, wait_s=0, workers=16):
    """Execute the registered lane-on baseline once on the u24 panel; idempotent receipt."""
    from . import track2_backend as p1
    from .launch_training import health, read, wait_for, write
    ctx = _prepare(cfg, checkpoint)           # SHA/payload/contract/phase before any charge
    root = ctx['root']
    if (root/TARGET).exists():
        return verify_receipt_file(read(root/TARGET), ctx)
    if (root/FAILED).exists():
        _refuse('a previous lane-on attempt failed; inspect it, never retry')
    paired_path = root/PAIRED
    if wait_s:
        paired = wait_for(paired_path, root, timeout=wait_s)
    elif paired_path.is_file():
        paired = read(paired_path)
    else:
        _refuse('the u24 paired A/B/R evaluation is not complete')
    reference = paired_reference(paired, ctx['registration'], ctx['contract'])
    worlds = same_worlds(ctx['pairs'], reference, ctx['contract'])
    rows = _accounting(ctx['native'])
    ids = {r['id'] for r in rows}
    if any(i.startswith(PREFIX+'_') for i in ids):
        _refuse('lane-on charges exist without a receipt; inspect, never retry')
    primary = {f'{PRIMARY_PREFIX}_{i:04}' for i in range(len(paired['results']))}
    if not primary <= ids:
        _refuse('the A/B/R u24 charges must be reserved before the lane-on baseline')
    quoted = sum(j['sim_s'] for j in ctx['jobs'])
    if quoted > ctx['charge']['allocation_sim_s']:
        _refuse('lane-on quote exceeds its registered allocation')
    health(root)
    p1._reserve(ctx['native'], ctx['jobs'], ctx['charge']['phase'])
    results, errors = _dispatch(ctx, workers)

    def failed(errors):
        write(root/FAILED, _seal(dict(schema=SCHEMA, status='failed', errors=errors,
            graded=len(results), charged_sim_s=quoted, charge=ctx['charge'],
            rule='charged and recorded; never dropped/imputed/retried'), 'receipt_sha256'))
        return ValueError('incomplete lane-on baseline; see '+FAILED)
    if errors or any(r.get('status') != 'graded' for r in results):
        raise failed(errors)
    counters = list(ctx['contract']['safety_counters'])
    try:
        receipts = [make_receipt(spec, result, job_id=job['id'], phase=ctx['charge']['phase'],
                                 safety_counters=counters)
                    for job, (spec, _), result in zip(ctx['jobs'], ctx['pairs'], results)]
        analysis = analyse(registration=ctx['registration'], receipts=receipts, paired=paired,
                           contract=ctx['contract'], prefixes=arm_prefixes(root, paired))
    except ValueError as error:
        raise failed([dict(job='receipts/analysis', error=repr(error))]) from error
    value = _seal(dict(schema=SCHEMA, update=UPDATE, status='graded', baseline=ctx['entry'],
        checkpoint=dict(path=str(ctx['path']), file_sha256=CHECKPOINT_SHA256, payload=ctx['meta']),
        lane_change_mode=LANE_CHANGE_MODE, charge=dict(ctx['charge'], jobs=len(ctx['jobs']),
        sim_s=quoted), registration=ctx['registration'], same_worlds=worlds,
        paired_receipt_sha256=paired['receipt_sha256'], results=results, receipts=receipts,
        analysis=analysis, descriptive=True, note=NOTE, new_sim_s=quoted), 'receipt_sha256')
    write(root/TARGET, value)
    return value


def summary(value):
    analysis = value['analysis']
    return dict(status=value['status'], receipt_sha256=value['receipt_sha256'],
                charge=value['charge'], mean_J_veh_s=analysis['mean_J_veh_s'],
                paired_ci={k: {x: v[x] for x in ('n', 'mean', 'low', 'high')}
                           for k, v in analysis['paired_ci'].items()},
                safety=analysis['safety'], warmup_prefix_check=analysis['warmup_prefix_check'])


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='clean3 u24 extra lane-on MAPPO aecd7d70 baseline')
    parser.add_argument('action', choices=('dry', 'run'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--wait-hours', type=float, default=0.)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args(argv)
    from .launch_training import checked_config
    cfg = checked_config(args.config)
    if args.action == 'dry':
        print(json.dumps(dry(cfg, args.checkpoint), sort_keys=True))
        return
    value = run(cfg, args.checkpoint, wait_s=int(args.wait_hours*3600), workers=args.workers)
    print(json.dumps(summary(value), sort_keys=True))


if __name__ == '__main__':
    main()
