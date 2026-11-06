"""WM-B raw WM-data collection on wz3to2 (phase wm_wz_data; cpuV; SUMO).

NEVER run ``run`` locally: it starts SUMO. ``plan`` and ``summarize`` are zero-physics.

    export COOPMWM_MAP_SPEC=<root>/src/assets/research/newmap_candidates/wz3to2/MAP_SPEC.json
    export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    nice -n 19 python -m tools.wzwmdata.collect plan --root <root>/run
    nice -n 19 python -m tools.wzwmdata.collect run --root <root>/run --workers 12 [--only canary]
    python -m tools.wzwmdata.collect summarize --root <root>/run

Episodes are the original WM tape format: ``traffic_shepherd.c4.recorder`` write-once directory
``<job>/episode/`` (manifest.json, summary.json, episode.npz schema ``c4_sumo_episode/1``,
run/action_timeline.jsonl.gz, scenario/) recorded by the frozen e3final CavSumoEnv bound to wz3to2
(``tools.coopmwm.map_runtime.install``), plus ``<job>/extras.npz`` (per-physics-step entry queue /
source wait, per-second team delay) and ``<job>/RESULT.json``.

Arms (WMDATA worlds; every arm of a world shares the world, scenario bytes and SUMO seed):
* ``native``      every CAV released to SUMO;
* ``mappo``       frozen speed-only MAPPO (policy_00169, d9042cc5; deterministic; native lanes);
* ``coop_mappo``  MAPPO base + cooperation-move perturbations (members' accel overridden);
* ``coop_native`` native base + the same move library, members also hold their lane (policy lane
                  control applies to members only).
WMQUAL (WM-D set (a)): ``native`` 0-1500 s, split tag WMQUAL, separate ``wmqual/`` directory.

Ledger: ``<root>/ledger/CHARGES.jsonl`` is the PHASE ledger (cap 1,000,000 sim-s, shared with any
other wm_wz_data task id). Every job appends an ALLOC row (flock + fsync) with its full episode
length BEFORE SUMO starts; a job directory that exists is never re-run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import time

SCHEMA = 'wzwmdata.collect.v1'
PHASE_CAP_SIM_S = 1_000_000
TASK = 'WMB'
TASK_CAP_SIM_S = 620_000
ARMS = ('native', 'mappo_u0', 'coop_mappo_u0', 'coop_native')
MAPPO_ARMS = ('mappo_u0', 'coop_mappo_u0', 'mappo', 'coop_mappo')   # canary used the short tags
DEFAULT_RUNTIME = '/home/ubuntu/e3final'
DEFAULT_CHECKPOINT = '/home/ubuntu/mappospeed/production2/policy_00169.pt'
LOAD_MAX = 42.0              # cpuV 48 cores: start a job only below this 1-min load (clean4 first)
MAX_WORKERS = {'cpuV': 12, 'gpuA': 16}
MIN_FREE_GB = 25.0
FORBIDDEN = ('holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'sealed')


def _write(path, value, mode='x'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding='utf-8') as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _guard_path(p):
    low = str(p).lower()
    if any(x in low for x in FORBIDDEN):
        raise ValueError(f'restricted path refused: {p}')


# --------------------------------------------------------------------------- plan (zero physics)
def make_jobs(root, *, runtime, checkpoint, data_root=None, skip=()):
    from . import registry as reg
    root = Path(data_root or root)
    jobs = []
    order = {c: i for i, c in enumerate(reg.CELL_ORDER)}
    worlds = sorted(reg.wmdata_worlds(), key=lambda w: (order[(w['demand'], w['penetration'])], w['seed']))
    for w in worlds:
        for arm in ARMS:
            jid = f"{w['cell_key']}_s{w['seed']}_{arm}"
            jobs.append(dict(id=jid, arm=arm, world=w, split='TRAIN', block='WMDATA',
                             warmup_s=reg.WARMUP_S, end_s=reg.WMDATA_END_S, sim_s=reg.WMDATA_END_S,
                             runtime=str(runtime), checkpoint=str(checkpoint),
                             path=str(root / 'wmdata' / w['cell_key'] / f"s{w['seed']}" / arm)))
    for w in reg.wmqual_worlds():
        jid = f"WMQUAL_{w['cell_key']}_s{w['seed']}_native"
        jobs.append(dict(id=jid, arm='native', world=w, split='WMQUAL', block='WMQUAL',
                         warmup_s=reg.WARMUP_S, end_s=reg.WMQUAL_END_S, sim_s=reg.WMQUAL_END_S,
                         runtime=str(runtime), checkpoint=str(checkpoint),
                         path=str(root / 'wmqual' / w['cell_key'] / f"s{w['seed']}" / 'native')))
    # canary = the first WMDATA world (all four arms) + the first WMQUAL episode
    first = jobs[0]['world']['seed']
    for j in jobs:
        j['wave'] = 'canary' if (j['block'] == 'WMDATA' and j['world']['seed'] == first) else 'main'
    next(j for j in jobs if j['block'] == 'WMQUAL')['wave'] = 'canary'
    skip = set(skip)
    return [j for j in jobs if j['id'] not in skip and j['id'].replace('_u0', '') not in skip]


def plan(root, *, runtime=DEFAULT_RUNTIME, checkpoint=DEFAULT_CHECKPOINT, commit=None, data_root=None,
         skip=(), prior_charged_sim_s=0.0, prior_note=None, host='cpuV'):
    from . import registry as reg
    from tools.coopmwm import map_descriptor as md
    root = Path(root)
    _guard_path(root)
    if root.exists():
        raise ValueError('new collection root required; never overwrite')
    desc = md.current()
    if desc.name != reg.MAP:
        raise ValueError(f'set {md.ENV} to the wz3to2 MAP_SPEC.json (process map is {desc.name})')
    jobs = make_jobs(root, runtime=runtime, checkpoint=checkpoint, data_root=data_root, skip=skip)
    total = sum(j['sim_s'] for j in jobs)
    if total + float(prior_charged_sim_s) > TASK_CAP_SIM_S:
        raise ValueError(f'plan {total} sim-s exceeds the WM-B task cap {TASK_CAP_SIM_S}')
    body = dict(schema=SCHEMA, created_unix=time.time(), phase=reg.PHASE, task=TASK,
                purpose='wz3to2 WM training raw data (WMDATA) + WM-D qualification native tapes (WMQUAL)',
                map=desc.record(), map_descriptor_sha256=desc.digest, registry=reg.record(),
                arms=list(ARMS), jobs=jobs,
                budget=dict(jobs=len(jobs), total_sim_s=total,
                            wmdata_sim_s=sum(j['sim_s'] for j in jobs if j['block'] == 'WMDATA'),
                            wmqual_sim_s=sum(j['sim_s'] for j in jobs if j['block'] == 'WMQUAL'),
                            task_cap_sim_s=TASK_CAP_SIM_S, phase_cap_sim_s=PHASE_CAP_SIM_S),
                coop=dict(module='tools.wzwmdata.coop', key='wzwmdata_coop_v1:{seed}:{arm}'),
                source_commit=commit, runtime=str(runtime), checkpoint=str(checkpoint), host=host,
                data_root=str(data_root) if data_root else None, skipped_ids=sorted(set(skip)),
                prior_charged_sim_s=float(prior_charged_sim_s), prior_note=prior_note,
                hard_rules='registered seeds only; WMQUAL never in WM train/val; nice 19 CPU only; '
                           'precharged per job, never retried; HDVs never commanded; own PIDs only')
    body['plan_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    root.mkdir(parents=True)
    (root / 'ledger').mkdir()
    _write(root / 'PLAN.json', body)
    return body


def all_jobs(root):
    """PLAN.json jobs + jobs of every PLAN_AMEND_*.json (registered amendments, sorted by name)."""
    root = Path(root)
    jobs = list(_read(root / 'PLAN.json')['jobs'])
    for amend in sorted(root.glob('PLAN_AMEND_*.json')):
        jobs.extend(_read(amend)['jobs'])
    ids = [j['id'] for j in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate job ids across plan amendments')
    return jobs


def amend(root, *, name, reason, commit):
    """Explicit re-run jobs (new ids/paths) for every FAILED job; failed receipts stay charged."""
    root = Path(root)
    out = root / f'PLAN_AMEND_{name}.json'
    if out.exists():
        raise ValueError('amendment exists; never overwrite')
    jobs = []
    for job in all_jobs(root):
        if (Path(job['path']) / 'FAILED.json').exists():
            new = dict(job, id=f"{job['id']}_{name}", path=f"{job['path']}_{name}", wave='main',
                       rerun_of=job['id'])
            jobs.append(new)
    total = sum(j['sim_s'] for j in jobs)
    body = dict(schema=SCHEMA + '.amend', name=name, reason=reason, source_commit=commit, jobs=jobs,
                total_sim_s=total, unix=time.time())
    body['amend_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    _write(out, body)
    return body


# --------------------------------------------------------------------------- ledger
def _ledger_rows(path):
    if not Path(path).exists():
        return []
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def charge(root, job):
    """Append the ALLOC row under an exclusive lock; refuse above the phase or task cap."""
    import fcntl
    path = Path(root) / 'ledger' / 'CHARGES.jsonl'
    lock = Path(root) / 'ledger' / 'CHARGES.lock'
    with lock.open('a') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        rows = _ledger_rows(path)
        if any(r.get('id') == job['id'] for r in rows):
            raise ValueError(f"{job['id']} already charged; never retried")
        phase = sum(float(r['sim_s']) for r in rows)
        prior = float(_read(Path(root) / 'PLAN.json').get('prior_charged_sim_s', 0.0))
        task = prior + sum(float(r['sim_s']) for r in rows if r.get('task') == TASK)
        if phase + job['sim_s'] > PHASE_CAP_SIM_S or task + job['sim_s'] > TASK_CAP_SIM_S:
            raise ValueError(f'cap: phase {phase}+{job["sim_s"]} / task {task}')
        row = dict(phase='wm_wz_data', task=TASK, kind='ALLOC', id=job['id'], sim_s=job['sim_s'],
                   split=job['split'], unix=time.time(), pid=os.getpid(),
                   cumulative_phase_sim_s=phase + job['sim_s'])
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, sort_keys=True) + '\n')
            f.flush()
            os.fsync(f.fileno())
        fcntl.flock(lk, fcntl.LOCK_UN)
    return row


# --------------------------------------------------------------------------- one episode (SUMO)
def _mask_executor(exe, env, previous):
    """Presence masking of the MAPPO executor memory/dwell (= tools.coopmwm.newmap_sanity)."""
    import numpy as np
    import torch
    ids = np.asarray(env.frame.vid_index)
    keep = np.zeros(len(ids), bool) if previous is None else (ids == previous) & (ids >= 0)
    for obj in (exe.memory, exe.dwell):
        if obj.present is not None:
            obj.present = obj.present & torch.as_tensor(keep, device=obj.present.device)[None]
    return ids.copy()


def run_episode(job, root):
    import importlib
    import random
    import numpy as np
    import torch
    from tools.leverprobe import backend as exact
    exact.setup(job['runtime'], job['checkpoint'])        # frozen runtime + MAPPO hashes, nice 19, CPU
    from tools.coopmwm.map_runtime import install
    from . import registry as reg
    from .coop import CoopSchedule
    desc = install(job)
    if desc is None or desc.name != reg.MAP:
        raise ValueError('wz3to2 map binding required')
    env_mod = importlib.import_module('traffic_shepherd.c4.env')
    env_mod.build_scenario = reg.scenario_builder(desc)    # this phase's registry (same generator)
    env_mod.check_case = reg.check_case
    import traci
    from traffic_shepherd.c4.env import CavSumoEnv
    world, arm = job['world'], job['arm']
    reg.world(world['seed'], world['demand'], world['penetration'], block=job['block'])
    random.seed(world['seed'])
    np.random.seed(world['seed'])
    torch.manual_seed(world['seed'])
    original, used = traci.simulationStep, [0.]

    def counted(*args, **kwargs):
        if used[0] + .5 > job['sim_s'] + 1e-9:
            raise ValueError('step exceeds precharge')
        before = float(traci.simulation.getTime())
        out = original(*args, **kwargs)
        used[0] += float(traci.simulation.getTime()) - before
        return out
    traci.simulationStep = counted
    path = Path(job['path'])
    mappo = arm in MAPPO_ARMS
    mode = 'native' if arm == 'native' else 'direct'
    env = exe = None
    schedule = None
    if arm.startswith('coop_'):
        schedule = CoopSchedule(f"wzwmdata_coop_v1:{world['seed']}:{arm}",
                                drop_s=float(desc.frame.merge_zone_s_m[1]),
                                closing_lane=desc.frame.merge_lane,
                                adjacent_lane=desc.frame.main_lanes[1],
                                lanes=(desc.frame.main_lanes[0], desc.frame.main_lanes[1], desc.frame.merge_lane),
                                start_s=job['warmup_s'], end_s=job['end_s'])
    try:
        if mappo:
            from wms.control.policy import load_policy_fn
            exe = load_policy_fn(str(job['checkpoint']), device='cpu')
            if not exe.native_lanes or exe.residual_on_native:
                raise ValueError('speed-only direct MAPPO required')
            exe.reset()
        meta = dict(policy={'native': 'native', 'mappo_u0': 'mappo_speed_u0',
                            'coop_mappo_u0': 'mappo_speed_u0+coop', 'mappo': 'mappo_speed_u0',
                            'coop_mappo': 'mappo_speed_u0+coop', 'coop_native': 'native+coop'}[arm],
                    arm=arm, collector_job=job['id'], phase=reg.PHASE, task=TASK, split=job['split'],
                    block=job['block'], map=reg.MAP, cell_key=world['cell_key'],
                    mappo_sha256=exact.MAPPO_SHA if mappo else None, map_descriptor_sha256=desc.digest)
        env = CavSumoEnv(k_slots=192, controller=mode, ttc=True, scenario_root=path / 'tmp_scenarios',
                         physics_log=False, population='textbook')
        base_meta = env._meta                     # episode.npz meta_json also carries the map identity
        env._meta = lambda: dict(base_meta(), map=reg.MAP, map_descriptor_sha256=desc.digest, arm=arm,
                                 policy=meta['policy'], split=job['split'], block=job['block'],
                                 phase=reg.PHASE, task=TASK, cell_key=world['cell_key'],
                                 collector_job=job['id'])
        env.reset(seed=world['seed'], demand_vph=world['demand'], penetration=world['penetration'],
                  warmup_s=job['warmup_s'], horizon_s=job['end_s'] - job['warmup_s'], controller=mode,
                  episode_dir=path / 'episode', meta=meta)
        k = env.k
        totals, lane_cmds = Counter(), Counter()
        delay_series, eq_steps, sw_steps = [], [], []
        previous, max_present = None, 0
        for t in range(int(job['warmup_s']), int(job['end_s'])):
            cmd, lanes = None, None
            if mappo:
                previous = _mask_executor(exe, env, previous)
                out, lane_out = exe(env.state(), with_lanes=True)
                if lane_out is not None:
                    raise ValueError('native lanes required')
                cmd = out.detach().cpu().numpy().astype(np.float64).reshape(1, k) if torch.is_tensor(out) \
                    else np.asarray(out, np.float64).reshape(1, k)
            if schedule is not None:
                accel, hold = schedule.commands(float(t), env.frame, env.builder.vehicle_ids)
                if cmd is None:
                    cmd = np.full((1, k), np.nan)
                for slot, a in accel.items():
                    cmd[0, slot] = a
                if arm == 'coop_native' and accel:
                    lanes = np.full((1, k), -1.0)
                    for slot, ln in hold.items():
                        lanes[0, slot] = ln
            if mode == 'native':
                _, info = env.step(None)
            else:
                _, info = env.step(cmd, lanes=lanes)
            iv = info['interval']
            road = float(iv.get('speed_deficit_vehicle_seconds', 0.))
            source = float(iv.get('source_wait_vehicle_seconds', 0.))
            totals.update(road_delay_vs=road, source_wait_vs=source, team_delay_vs=road + source,
                          collisions=float(iv.get('collision_involvements', 0)),
                          teleports=float(iv.get('starting_teleports', 0)),
                          ttc_below_3_vs=float(iv.get('ttc_below_3_vehicle_seconds', 0.)),
                          hard_brake=float(iv.get('braking_vehicle_seconds', 0.)),
                          dropped_vehicle_steps=float(iv.get('dropped_vehicle_steps', 0)),
                          arrivals=float(iv.get('arrivals', 0)), departures=float(iv.get('departures', 0)))
            lane_cmds.update(info.get('lane_commands') or {})
            delay_series.append(road + source)
            eq_steps.extend(info.get('entry_queue_steps') or [])
            sw_steps.extend(info.get('source_wait_steps') or [])
            if env.frame is not None:
                max_present = max(max_present, int(np.asarray(env.frame.present).sum()))
            if info['done']:
                break
        summary = _read(path / 'episode' / 'summary.json')
        np.savez_compressed(path / 'extras.npz', team_delay_per_s=np.asarray(delay_series, np.float32),
                            entry_queue_per_step=np.asarray(eq_steps, np.float32),
                            source_wait_per_step=np.asarray(sw_steps, np.float32),
                            control_start_s=np.asarray(job['warmup_s'], np.float64))
        ep = path / 'episode' / 'episode.npz'
        return dict(totals=dict(totals), lane_commands=dict(lane_cmds), max_present=max_present,
                    k_slots=k, population=len(env.requests), designated_cavs=len(env.cav_ids),
                    scenario_routes_sha256=env.scenario_manifest['routes_sha256'],
                    actual_sim_s=used[0], coop=None if schedule is None else schedule.record(),
                    episode_npz_sha256=_sha(ep), episode_npz_bytes=ep.stat().st_size,
                    summary=dict(collisions=summary.get('collisions'), teleports=summary.get('teleports'),
                                 dropped_vehicle_steps=summary.get('dropped_vehicle_steps'),
                                 dropped_cav_steps=summary.get('dropped_cav_steps'),
                                 max_present_vehicles=summary.get('max_present_vehicles'),
                                 command_counts=summary.get('command_counts'),
                                 sim_s_per_wall_s=summary.get('sim_s_per_wall_s')))
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            traci.simulationStep = original
            tmp = path / 'tmp_scenarios'
            if tmp.exists() and not any(tmp.iterdir()):
                tmp.rmdir()


def _wait_capacity(root, data_path, load_max):
    import shutil
    while True:
        if (Path(root) / 'STOP').exists():
            return 'stop'
        load = os.getloadavg()[0]
        free = shutil.disk_usage(str(data_path)).free / 1e9
        root_free = shutil.disk_usage(str(root)).free / 1e9
        if (load_max <= 0 or load < load_max) and free >= MIN_FREE_GB and root_free >= 1.0:
            return None
        time.sleep(30)


def run_job(arg):
    job, root, load_max = arg
    root = Path(root)
    path = Path(job['path'])
    if path.exists():
        return dict(id=job['id'], status='skipped_exists')
    data_path = path.parent
    data_path.mkdir(parents=True, exist_ok=True)
    if _wait_capacity(root, data_path, load_max) == 'stop':
        return dict(id=job['id'], status='stopped')
    try:
        row = charge(root, job)
    except ValueError as exc:
        return dict(id=job['id'], status='refused', error=repr(exc))
    path.mkdir(parents=True, exist_ok=False)
    _write(path / 'CLAIM.json', dict(pid=os.getpid(), job=job['id'], charged_sim_s=job['sim_s'],
                                     ledger_row=row, unix=time.time()))
    started = time.time()
    try:
        out = run_episode(job, root)
        out.update(id=job['id'], arm=job['arm'], world=job['world'], split=job['split'],
                   block=job['block'], status='ok', wall_s=round(time.time() - started, 2),
                   charged_sim_s=job['sim_s'])
        _write(path / 'RESULT.json', out)
        return dict(id=job['id'], status='ok', wall_s=out['wall_s'])
    except Exception as exc:                                   # failure receipt, never retried
        import traceback
        _write(path / 'FAILED.json', dict(id=job['id'], arm=job['arm'], world=job['world'], error=repr(exc),
                                          traceback=traceback.format_exc(), charged_sim_s=job['sim_s'],
                                          wall_s=round(time.time() - started, 2)))
        return dict(id=job['id'], status='failed', error=repr(exc))


def run(root, *, workers=12, only=None, ids=None, load_max=LOAD_MAX):
    import multiprocessing
    root = Path(root)
    _guard_path(root)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise ValueError('run under nice 19 with CUDA_VISIBLE_DEVICES= (CPU only)')
    host = _read(root / 'PLAN.json').get('host', 'cpuV')
    if int(workers) > MAX_WORKERS.get(host, 12):
        raise ValueError(f'{host} WM-B limit: at most {MAX_WORKERS.get(host, 12)} workers')
    jobs = [j for j in all_jobs(root) if (only is None or j['wave'] == only) and (ids is None or j['id'] in ids)]
    stamp = time.strftime('%Y%m%dT%H%M%S')
    _write(root / f'RUN_START_{stamp}.json', dict(pid=os.getpid(), workers=workers, only=only, load_max=load_max,
                                                 n_jobs=len(jobs), unix=time.time()))
    ctx = multiprocessing.get_context('spawn')
    status = []
    with ctx.Pool(processes=int(workers), maxtasksperchild=1) as pool:
        for st in pool.imap_unordered(run_job, [(j, str(root), float(load_max)) for j in jobs], chunksize=1):
            status.append(st)
            with (root / 'run_progress.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(st, unix=time.time())) + '\n')
    _write(root / f'RUN_DONE_{stamp}.json', dict(status=status, unix=time.time()))
    return summarize(root)


# --------------------------------------------------------------------------- summary (zero physics)
def summarize(root):
    root = Path(root)
    body = _read(root / 'PLAN.json')
    rows = _ledger_rows(root / 'ledger' / 'CHARGES.jsonl')
    per_cell, per_arm, failures, sizes = {}, Counter(), {}, []
    safety = Counter()
    done = 0
    jobs = all_jobs(root)
    for job in jobs:
        p = Path(job['path'])
        if (p / 'RESULT.json').exists():
            r = _read(p / 'RESULT.json')
            done += 1
            key = f"{job['block']}:{job['world']['cell_key']}"
            c = per_cell.setdefault(key, Counter())
            c['episodes'] += 1
            c['sim_s'] += r['actual_sim_s']
            c[f"arm_{job['arm']}"] += 1
            per_arm[f"{job['block']}:{job['arm']}"] += 1
            sizes.append(r.get('episode_npz_bytes', 0))
            t = r['totals']
            for k in ('collisions', 'teleports', 'dropped_vehicle_steps'):
                safety[k] += t.get(k, 0)
            if r.get('coop'):
                for fam, n in r['coop']['by_family'].items():
                    c[f'moves_{fam}'] += n
            c['max_present'] = max(c.get('max_present', 0), r.get('max_present', 0))
        elif (p / 'FAILED.json').exists():
            failures[job['id']] = _read(p / 'FAILED.json')['error']
    out = dict(schema=SCHEMA + '.summary', plan_sha256=body['plan_sha256'], jobs_planned=len(jobs),
               jobs_done=done, failures=failures, per_cell={k: dict(v) for k, v in sorted(per_cell.items())},
               per_arm=dict(per_arm), safety=dict(safety),
               ledger=dict(rows=len(rows), phase_charged_sim_s=sum(float(r['sim_s']) for r in rows),
                           task_charged_sim_s=sum(float(r['sim_s']) for r in rows if r.get('task') == TASK)),
               episode_npz_bytes=dict(total=sum(sizes), mean=(sum(sizes) / len(sizes)) if sizes else None),
               unix=time.time())
    _write(root / 'SUMMARY.json', out, mode='w')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.wzwmdata.collect')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--root', required=True)
    p.add_argument('--runtime', default=DEFAULT_RUNTIME)
    p.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    p.add_argument('--commit')
    p.add_argument('--data-root')
    p.add_argument('--skip-ids-file')
    p.add_argument('--prior-charged', type=float, default=0.0)
    p.add_argument('--prior-note')
    p.add_argument('--host', default='cpuV', choices=sorted(MAX_WORKERS))
    r = sub.add_parser('run')
    r.add_argument('--root', required=True)
    r.add_argument('--workers', type=int, default=12)
    r.add_argument('--only', choices=('canary', 'main'))
    r.add_argument('--ids-file')
    r.add_argument('--load-max', type=float, default=LOAD_MAX)
    m = sub.add_parser('amend')
    m.add_argument('--root', required=True)
    m.add_argument('--name', required=True)
    m.add_argument('--reason', required=True)
    m.add_argument('--commit', required=True)
    s = sub.add_parser('summarize')
    s.add_argument('--root', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'plan':
        skip = () if not a.skip_ids_file else Path(a.skip_ids_file).read_text().split()
        out = plan(a.root, runtime=a.runtime, checkpoint=a.checkpoint, commit=a.commit, data_root=a.data_root,
                   skip=skip, prior_charged_sim_s=a.prior_charged, prior_note=a.prior_note, host=a.host)
        print(json.dumps(dict(plan_sha256=out['plan_sha256'], **out['budget'])))
    elif a.cmd == 'run':
        ids = None if not a.ids_file else set(Path(a.ids_file).read_text().split())
        out = run(a.root, workers=a.workers, only=a.only, ids=ids, load_max=a.load_max)
        print(json.dumps(dict(done=out['jobs_done'], failures=len(out['failures']), ledger=out['ledger'])))
    elif a.cmd == 'amend':
        out = amend(a.root, name=a.name, reason=a.reason, commit=a.commit)
        print(json.dumps(dict(amend_sha256=out['amend_sha256'], jobs=[j['id'] for j in out['jobs']],
                              total_sim_s=out['total_sim_s'])))
    else:
        out = summarize(a.root)
        print(json.dumps(dict(done=out['jobs_done'], failures=len(out['failures']), ledger=out['ledger'],
                              safety=out['safety'])))


if __name__ == '__main__':
    main()
