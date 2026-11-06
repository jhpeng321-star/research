"""N1 base-policy sanity job on wz3to2: frozen speed-only MAPPO u0 vs native CAVs (cpuV; N4 operator).

NEVER run locally: this starts SUMO. Zero physics happens in ``plan`` and ``summarize``.

    export COOPMWM_MAP_SPEC=/home/ubuntu/<staged>/assets/research/newmap_candidates/wz3to2/MAP_SPEC.json
    export COOPMWM_G5_RUNTIME=1 CUDA_VISIBLE_DEVICES=
    nice -n 19 python -m tools.coopmwm.newmap_sanity plan --root /home/ubuntu/newmap_sanity_<tag> \
        --runtime /home/ubuntu/e3final --checkpoint /home/ubuntu/mappospeed/production2/policy_00169.pt \
        [--u0 /home/ubuntu/coopmwm20261106_clean3_r6/A_checkpoint_000.pt]
    nice -n 19 python -m tools.coopmwm.newmap_sanity run --root /home/ubuntu/newmap_sanity_<tag> --workers 8
    python -m tools.coopmwm.newmap_sanity summarize --root /home/ubuntu/newmap_sanity_<tag>

Arms per world (paired: same registered world, same scenario bytes, same SUMO seed):
* ``mappo``  - every CAV commanded by the frozen speed-only MAPPO executor (deterministic; the clean3
  base, i.e. MAPPO u0 with a zero residual), native lanes, the CommandOwnership chain;
* ``native`` - every CAV released to SUMO (same env, controller 'native');
* ``clean3_u0`` (only with --u0) - the real clean3 training episode path (accel_plant.episode,
  deterministic u0 weights, V2V wrapper when the checkpoint is widened, CAV-report rows): a
  runtime-integration check; it aborts on the plant's hard-event guard like training would.
Every episode is the 1200-s plant episode: native warmup 0-300 s, control 300-1200 s.
Metrics (control window, veh-s): team delay = on-road speed deficit + source waiting (the plant's
J), collisions, teleports, TTC<3 s exposure, hard braking, dropped slots, max present vehicles,
lane-command counts, CAV-report RSU coverage; per-second team delay series.

Registered pass rule (written into SANITY_PLAN.json before physics): MAPPO is the clean4 base iff
(a) every mappo episode completes with 0 collisions, 0 teleports and 0 dropped slot-steps;
(b) pooled paired delay ratio sum(J_mappo - J_native)/sum(J_native) <= +0.05;
(c) TTC<3 s exposure and hard braking each <= 1.5 x native + 10 veh-s (pooled).
Otherwise the base is native CAVs. n is small: the numbers are descriptive, the rule is a sanity gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

SCHEMA = 'coopmwm.newmap_sanity.v1'
SIM_S = 1200
WARMUP_S, HORIZON_S, STOP_S = 300, 901, 1200
DEFAULT_SEEDS = (989596, 989597, 989598, 989599)          # last TRAIN seeds (NEWMAP2 uses 989200-989202)
DEFAULT_CELLS = ((2800, 0.4), (3200, 0.4), (2800, 0.2), (3200, 0.6))
RULE = dict(max_pooled_delay_ratio=0.05, exposure_factor=1.5, exposure_slack_vs=10.0,
            text='MAPPO base iff every mappo episode has 0 collisions/teleports/dropped slot-steps, pooled '
                 'sum(J_mappo-J_native)/sum(J_native) <= +0.05, and pooled TTC<3 s exposure and hard braking '
                 'each <= 1.5 x native + 10 veh-s; else native CAVs')


def _write(path, value, mode='x'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding='utf-8') as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------- plan (zero physics)
def plan(root, *, runtime, checkpoint, u0=None, seeds=DEFAULT_SEEDS, cells=DEFAULT_CELLS, commit=None):
    from . import map_descriptor as md, map_scenarios as ms
    root = Path(root)
    if root.exists():
        raise ValueError('new sanity root required; never overwrite')
    desc = md.current()
    if desc.name != ms.MAP:
        raise ValueError(f'set {md.ENV} to the wz3to2 MAP_SPEC.json (process map is {desc.name})')
    if len(seeds) != len(cells):
        raise ValueError('one cell per seed (paired worlds)')
    worlds = [ms.world(int(s), int(d), float(p)) for s, (d, p) in zip(seeds, cells)]
    for w in worlds:
        ms.guard(w, 'sanity')
    arms = ['mappo', 'native'] + (['clean3_u0'] if u0 else [])
    jobs = [dict(id=f'{arm}_{w["seed"]}_{w["cell_key"]}', arm=arm, world=w, role='sanity',
                 runtime=str(runtime), checkpoint=str(checkpoint), u0=str(u0) if u0 else None,
                 sim_s=SIM_S, path=str(root/'jobs'/f'{arm}_{w["seed"]}_{w["cell_key"]}'))
            for w in worlds for arm in arms]
    body = dict(schema=SCHEMA, created_unix=time.time(), purpose='N1 base-policy sanity on wz3to2 (MAPPO u0 vs native)',
                map=desc.record(), map_descriptor_sha256=desc.digest, scenario_registry=ms.registry_record(),
                worlds=worlds, arms=arms, jobs=jobs, budget=dict(jobs=len(jobs), sim_s_per_job=SIM_S,
                total_sim_s=len(jobs)*SIM_S), rule=RULE, source_commit=commit,
                episode=dict(warmup_s=WARMUP_S, horizon_s=HORIZON_S, stop_s=STOP_S, control_window_s=[WARMUP_S, STOP_S]),
                hard_rules='TRAIN-range seeds only; no 964000-964199/962xxx/977xxx; nice 19 CPU only; '
                           'precharged per job, never retried; HDVs never commanded')
    body['plan_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    root.mkdir(parents=True)
    _write(root/'SANITY_PLAN.json', body)
    return body


# ------------------------------------------------------------------------- one episode (SUMO)
def _mask_executor(exe, env, previous):
    """leverprobe Backend.advance presence masking of the executor memory/dwell."""
    import numpy as np
    import torch
    ids = np.asarray(env.frame.vid_index)
    keep = np.zeros(len(ids), bool) if previous is None else (ids == previous) & (ids >= 0)
    for obj in (exe.memory, exe.dwell):
        if obj.present is not None:
            obj.present = obj.present & torch.as_tensor(keep, device=obj.present.device)[None]
    return ids.copy()


def _queue(frame, desc):
    """Closing-lane queue proxy from the frame (diagnostic only): WM lane 2, 300 m before the drop, v < 5."""
    import numpy as np
    lane = np.where(frame.junction, frame.lane_carry, frame.lane)
    drop = desc.frame.merge_zone_s_m[1]
    m = frame.present & (lane == desc.frame.merge_lane) & (frame.s >= drop-300.) & (frame.s <= drop) & (frame.v < 5.)
    return int(m.sum())


def run_controlled(job):
    """mappo / native arm: frozen runtime + map binding, counting loop (records, never aborts on events)."""
    import random
    import numpy as np
    import torch
    from tools.leverprobe import backend as exact
    exact.setup(job['runtime'], job['checkpoint'])          # frozen runtime + MAPPO hashes, nice 19, CPU only
    from .g5_runtime import install as g5
    g5(job['runtime'])
    from .map_runtime import guard_world, install
    guard_world(job['world'], job['role'])
    desc = install(job)
    if desc is None:
        raise ValueError('sanity job requires the wz3to2 map binding')
    import traci
    from traffic_shepherd.c4.env import CavSumoEnv
    from .map_observations import report_summary
    path = Path(job['path'])
    world, mappo = job['world'], job['arm'] == 'mappo'
    random.seed(world['seed']); np.random.seed(world['seed']); torch.manual_seed(world['seed'])
    original, used = traci.simulationStep, [0.]

    def counted(*args, **kwargs):
        if used[0]+.5 > job['sim_s']:
            raise ValueError('step exceeds precharge')
        before = float(traci.simulation.getTime())
        out = original(*args, **kwargs)
        used[0] += float(traci.simulation.getTime())-before
        return out
    traci.simulationStep = counted
    env = exe = None
    try:
        if mappo:
            from wms.control.policy import load_policy_fn
            exe = load_policy_fn(str(job['checkpoint']), device='cpu')
            if not exe.native_lanes or exe.residual_on_native:
                raise ValueError('speed-only direct MAPPO required')
            exe.reset()
        env = CavSumoEnv(k_slots=192, controller='direct' if mappo else 'native', ttc=True,
                         scenario_root=path/'scenarios', physics_log=False, population='textbook')
        env.reset(seed=world['seed'], demand_vph=world['demand'], penetration=world['penetration'],
                  warmup_s=WARMUP_S, horizon_s=HORIZON_S, controller='direct' if mappo else 'native')
        totals, lane_cmds, series, queue, coverage = Counter(), Counter(), [], [], []
        previous, max_present = None, 0
        for t in range(WARMUP_S, STOP_S):
            if mappo:
                previous = _mask_executor(exe, env, previous)
                cmd, lanes = exe(env.state(), with_lanes=True)
                if lanes is not None:
                    raise ValueError('native lanes required')
                _, info = env.step(cmd, lanes=None)
            else:
                _, info = env.step(None)
            iv = info['interval']
            road, source = float(iv.get('speed_deficit_vehicle_seconds', 0.)), float(iv.get('source_wait_vehicle_seconds', 0.))
            totals.update(road_delay_vs=road, source_wait_vs=source, team_delay_vs=road+source,
                          collisions=float(iv.get('collision_involvements', 0)), teleports=float(iv.get('starting_teleports', 0)),
                          ttc_below_3_vs=float(iv.get('ttc_below_3_vehicle_seconds', 0.)),
                          hard_brake=float(iv.get('braking_vehicle_seconds', 0.)),
                          dropped_vehicle_steps=float(iv.get('dropped_vehicle_steps', 0)), arrivals=float(iv.get('arrivals', 0)))
            lane_cmds.update(info.get('lane_commands') or {})
            series.append(round(road+source, 6))
            max_present = max(max_present, int(env.frame.present.sum()))
            if (t+1) % 10 == 0:
                queue.append([t+1, _queue(env.frame, desc)])
            if (t+1) % 60 == 0:
                coverage.append(report_summary(env.frame, desc, now_s=t+1))
            if t+1 in (900, 1200):
                totals[f'team_delay_vs_at_{t+1-WARMUP_S}'] = totals['team_delay_vs']
            if info['done']:
                break
        return dict(totals=dict(totals), series=series, closing_lane_queue=queue, rsu_coverage=coverage,
                    lane_commands=dict(lane_cmds), max_present=max_present, k_slots=env.k,
                    population=len(env.requests), designated_cavs=len(env.cav_ids),
                    scenario_routes_sha256=env.scenario_manifest['routes_sha256'], actual_sim_s=used[0])
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            traci.simulationStep = original


def run_clean3_u0(job):
    """The clean3 training episode path itself (accel_plant.episode), deterministic u0 weights."""
    import torch
    from . import accel_plant
    saved = torch.load(job['u0'], map_location='cpu', weights_only=False)
    weights = saved['policy'] if isinstance(saved, dict) and 'policy' in saved else saved
    v2v = int(weights['actor.0.weight'].shape[1]) > 33
    native = dict(id=job['id'], path=str(Path(job['path'])/'plant'), world=job['world'], role=job['role'], arm='A',
                  weights=weights, checkpoint=job['checkpoint'], runtime=job['runtime'], noise_seed=0,
                  deterministic=True, sim_s=job['sim_s'], stop=STOP_S, v2v=v2v, queue_root=str(Path(job['path']).parent))
    result = accel_plant.episode(native)
    return dict(totals=result['totals'], metrics=result['metrics'], actual_sim_s=result['actual_sim_s'],
                requests=result['requests'], overrides=result['overrides'], v2v=v2v,
                u0_sha256=_sha(job['u0']))


def run_job(job):
    path = Path(job['path'])
    path.mkdir(parents=True, exist_ok=False)
    _write(path/'CLAIM.json', dict(pid=os.getpid(), job=job['id'], charged_sim_s=job['sim_s'], unix=time.time()))
    started = time.time()
    try:
        out = run_clean3_u0(job) if job['arm'] == 'clean3_u0' else run_controlled(job)
        out.update(id=job['id'], arm=job['arm'], world=job['world'], status='ok', wall_s=round(time.time()-started, 2))
        _write(path/'RESULT.json', out)
        return dict(id=job['id'], status='ok')
    except Exception as exc:                                   # failure receipt, never retried
        import traceback
        _write(path/'FAILED.json', dict(id=job['id'], arm=job['arm'], world=job['world'], error=repr(exc),
                                       traceback=traceback.format_exc(), wall_s=round(time.time()-started, 2)))
        return dict(id=job['id'], status='failed', error=repr(exc))


def run(root, *, workers=8):
    import multiprocessing
    root = Path(root)
    body = _read(root/'SANITY_PLAN.json')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise ValueError('run under nice 19 with CUDA_VISIBLE_DEVICES= (CPU only)')
    ledger = root/'CHARGES.jsonl'
    if ledger.exists():
        raise ValueError('sanity jobs already charged; never retried (write a new root)')
    with ledger.open('x', encoding='utf-8') as f:            # precharge every job before any SUMO
        for job in body['jobs']:
            f.write(json.dumps(dict(id=job['id'], sim_s=job['sim_s'], unix=time.time()), sort_keys=True)+'\n')
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=int(workers), maxtasksperchild=1) as pool:
        status = pool.map(run_job, body['jobs'], chunksize=1)
    _write(root/'RUN_DONE.json', dict(status=status, unix=time.time()))
    return summarize(root)


# ------------------------------------------------------------------------- summary (zero physics)
def decide(pairs, mappo_complete):
    """Registered rule over paired (mappo, native) totals."""
    jm = sum(p['mappo']['team_delay_vs'] for p in pairs)
    jn = sum(p['native']['team_delay_vs'] for p in pairs)
    events = {k: sum(p['mappo'].get(k, 0.) for p in pairs) for k in ('collisions', 'teleports', 'dropped_vehicle_steps')}
    ttc = {a: sum(p[a].get('ttc_below_3_vs', 0.) for p in pairs) for a in ('mappo', 'native')}
    brake = {a: sum(p[a].get('hard_brake', 0.) for p in pairs) for a in ('mappo', 'native')}
    ratio = (jm-jn)/jn if jn > 0 else math.inf
    f, slack = RULE['exposure_factor'], RULE['exposure_slack_vs']
    checks = dict(complete=bool(mappo_complete and pairs), safety=not any(events.values()),
                  delay=ratio <= RULE['max_pooled_delay_ratio'],
                  ttc=ttc['mappo'] <= f*ttc['native']+slack, hard_brake=brake['mappo'] <= f*brake['native']+slack)
    return dict(base='mappo_u0' if all(checks.values()) else 'native', checks=checks,
                pooled=dict(team_delay_mappo_vs=jm, team_delay_native_vs=jn, delay_ratio=ratio,
                            mappo_minus_native_vs=jm-jn, events_mappo=events, ttc_below_3_vs=ttc, hard_brake_vs=brake))


def summarize(root):
    root = Path(root)
    body = _read(root/'SANITY_PLAN.json')
    results, failures = {}, {}
    for job in body['jobs']:
        p = Path(job['path'])
        if (p/'RESULT.json').exists():
            results[job['id']] = _read(p/'RESULT.json')
        elif (p/'FAILED.json').exists():
            failures[job['id']] = _read(p/'FAILED.json')
    pairs, per_world = [], []
    for w in body['worlds']:
        key = f'{w["seed"]}_{w["cell_key"]}'
        m, n = results.get('mappo_'+key), results.get('native_'+key)
        row = dict(world=w, mappo=m and m['totals'], native=n and n['totals'],
                   clean3_u0=(results.get('clean3_u0_'+key) or {}).get('totals'),
                   max_present=dict(mappo=m and m['max_present'], native=n and n['max_present']))
        if m and n:
            pairs.append(dict(mappo=m['totals'], native=n['totals']))
            row['mappo_minus_native_team_delay_vs'] = m['totals']['team_delay_vs']-n['totals']['team_delay_vs']
        per_world.append(row)
    complete = len(pairs) == len(body['worlds'])           # every world paired (mappo AND native)
    out = dict(schema=SCHEMA+'.summary', plan_sha256=body['plan_sha256'], rule=RULE, per_world=per_world,
               failures={k: v['error'] for k, v in failures.items()}, decision=decide(pairs, complete),
               charged_sim_s=body['budget']['total_sim_s'],
               actual_sim_s=sum(r.get('actual_sim_s', 0.) for r in results.values()),
               sign='mappo_minus_native > 0 = MAPPO adds delay (worse)', unix=time.time())
    _write(root/'SANITY_SUMMARY.json', out, mode='w')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.coopmwm.newmap_sanity')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--root', required=True)
    p.add_argument('--runtime', default='/home/ubuntu/e3final')
    p.add_argument('--checkpoint', default='/home/ubuntu/mappospeed/production2/policy_00169.pt')
    p.add_argument('--u0')
    p.add_argument('--worlds', type=int, default=len(DEFAULT_SEEDS))
    p.add_argument('--commit')
    r = sub.add_parser('run')
    r.add_argument('--root', required=True)
    r.add_argument('--workers', type=int, default=8)
    s = sub.add_parser('summarize')
    s.add_argument('--root', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'plan':
        n = int(a.worlds)
        if not 1 <= n <= len(DEFAULT_SEEDS):
            raise SystemExit(f'--worlds must be 1..{len(DEFAULT_SEEDS)}')
        out = plan(a.root, runtime=a.runtime, checkpoint=a.checkpoint, u0=a.u0, seeds=DEFAULT_SEEDS[:n],
                   cells=DEFAULT_CELLS[:n], commit=a.commit)
        print(json.dumps(dict(plan_sha256=out['plan_sha256'], jobs=len(out['jobs']), sim_s=out['budget']['total_sim_s'])))
    elif a.cmd == 'run':
        print(json.dumps(run(a.root, workers=a.workers)['decision']))
    else:
        print(json.dumps(summarize(a.root)['decision']))


if __name__ == '__main__':
    main()
