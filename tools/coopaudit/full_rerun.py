"""Corrected v2: fresh complete same-seed prefix for EVERY counterfactual branch."""
import argparse
import concurrent.futures
import contextlib
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from tools.coopaudit.accounting import Ledger, write
from tools.coopaudit.controllers import Controller, observations, wall_roster
from tools.coopaudit.design import canary, guard, select_group, sha, worlds
from tools.coopaudit.release import require_release
from tools.coopaudit.runner import ROOT, canonical, crn, initial_state, physics_rows, setup, trace_rows

ARMS = ('mappo','wall','sumocf')


def branch_plan(group, mechanics=False):
    if len(group)<2:
        return []
    branches=[('native','B',[]),('native','B_repeat',[])]
    if mechanics:
        branches += [('mappo','DISABLED',[]),('sumocf','DISABLED',[])]
    for c in ARMS:
        branches += [(c,'GROUP',group)]
        branches += [(c,f'ONLY:{i}',[v]) for i,v in enumerate(group)]
        if mechanics:
            branches += [(c,'GROUP_repeat',group)]
    return branches


def selected():
    selection=json.loads((ROOT/'CORRECTED_DESIGN.json').read_text())
    require_release(ROOT,selection)
    if selection['branch_mode']!='full_same_seed_prefix_rerun' or selection['controllers']!=list(ARMS):
        raise ValueError('not the registered corrected paired audit')
    if sha(selection['checkpoint'])!=selection['checkpoint_sha256']:
        raise ValueError('final checkpoint changed')
    return selection


def admission(selection):
    """Check controls/resources before every reservation, never signal peers."""
    storage=Path(selection['storage_root'])
    while True:
        # HOLD/STOP are checked again on every poll; no unguarded dispatch timer.
        require_release(ROOT,selection)
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:')))*1024
        load=os.getloadavg()[0]
        free=shutil.disk_usage(storage).free
        if load<=20 and available>=8*1024**3 and free>=1024**3:
            return
        write(ROOT/f'v2_WAIT_{os.getpid()}.json',dict(unix=time.time(),load=load,
                                                    memory_available=available,free_bytes=free))
        time.sleep(30)


def verify_prefix(env, expected, rows):
    if initial_state(env)!=expected['initial_state_sha256'] or crn(env)!=expected['crn_sha256']:
        raise ValueError('full-rerun prefix tensor/request identity mismatch')
    if 'observations_sha256' in expected and canonical(rows)!=expected['observations_sha256']:
        raise ValueError('full-rerun prefix observable identity mismatch')


def run_world(world, mechanics=False):
    selection=selected()
    inputs=setup()
    guard(world['seed'],world['demand'])
    from traffic_shepherd.c4 import branching as B
    from traffic_shepherd.c4 import evalprep as E
    from traffic_shepherd.c4.env import CavSumoEnv
    from observer import audit_lanes
    from tools.coopaudit.trained import trained_bound
    mechanics_name='v2_canary' if mechanics else 'v2'
    ident=f"{mechanics_name}_s{world['seed']}"
    out=Path(selection['storage_root'])/ident
    link=ROOT/'jobs'/ident
    if out.exists() or link.exists():
        raise ValueError('v2 world already started; no retry')
    if mechanics:
        previous=json.loads((ROOT/'jobs/canary_s969699_attempt2/FROZEN.json').read_text())
        group=previous['group']
        roster=previous['wall_roster']
        expected=dict(initial_state_sha256=previous['initial_state_sha256'],crn_sha256=previous['crn_sha256'],
                      observations_sha256=canonical(previous['observations']))
    else:
        entry=next(p for p in selection['panel'] if p['world']==world)
        path=ROOT/'jobs'/f"census_s{world['seed']}"/'FROZEN.json'
        if sha(path)!=entry['census_sha256']:
            raise ValueError('preserved census changed')
        previous=json.loads(path.read_text())
        group=select_group(previous['observations'],selection['group_size'])
        roster=wall_roster(previous['observations'],group)
        if group!=entry['group'] or roster!=entry['wall_roster']:
            raise ValueError('frozen corrected group changed')
        expected=dict(initial_state_sha256=entry['initial_state_sha256'],crn_sha256=entry['crn_sha256'],
                      observations_sha256=canonical(previous['observations']))
    frozen=dict(world=world,group=group,wall_roster=roster,**expected)
    plan=branch_plan(group,mechanics)
    admission(selection)
    out.mkdir()
    link.symlink_to(out,target_is_directory=True)
    write(out/'START.json',dict(world=world,pid=os.getpid(),unix=time.time(),branch_mode='full_prefix_rerun'))
    write(out/'FROZEN.json',frozen)
    (out/'traces').mkdir()
    progress=0
    metrics,hashes,effects={}, {}, {}
    env=None
    try:
        for controller,name,members in plan:
            admission(selection)
            key=controller+'_'+name
            Ledger(ROOT).charge(ident+'_'+key,900)
            write(out/f'START_{key}.json',dict(precharged_sim_s=900,pid=os.getpid(),unix=time.time()))
            branch_progress=0
            prefix=[]
            prefix_digest=hashlib.sha256()
            def hook(e):
                nonlocal progress,branch_progress
                now=int(e.now)
                progress += now-branch_progress
                branch_progress=now
                prefix_digest.update(json.dumps(physics_rows(e),separators=(',',':')).encode())
                if e.now>270:
                    prefix.extend(physics_rows(e))
            # SAVE_STATE_CLI preserves the exact census launch command only; no
            # saveState/loadState/snapshot call exists anywhere in this engine.
            env=CavSumoEnv(k_slots=192,controller='direct',extra_sumo_cli=B.SAVE_STATE_CLI,
                           scenario_root=out/'scenarios'/key,physics_log=False,population='textbook')
            env.reset(seed=world['seed'],demand_vph=world['demand'],penetration=world['penetration'],
                      warmup_s=300,horizon_s=601,controller='direct',warmup_hook=hook)
            if branch_progress!=300:
                progress+=300-branch_progress
                branch_progress=300
            verify_prefix(env,expected,observations(env))
            if metrics:
                if prefix_digest.hexdigest()!=first_prefix_sha:
                    raise ValueError('full native prefix trajectory not bitwise identical')
            else:
                first_prefix_sha=prefix_digest.hexdigest()
            B.prepare_branch_mode(env)
            bound=None
            if controller=='sumocf':
                bound=Controller('sumocf',inputs,roster)
                bound.bound=trained_bound(selection)
                bound.name='sumocf'
            elif controller!='native':
                bound=Controller(controller,inputs,roster)
            m=E.Measurement()
            m.path=out
            m.start(env)
            values={}
            crossed=set()
            local_crossing=None
            digest=hashlib.sha256()
            command_ticks=command_count=0
            ramp_ids=set()
            with gzip.open(out/'traces'/f'{key}.jsonl.gz','wt') as trace, audit_lanes() as lane_audit:
                for r in trace_rows(prefix,group,ramp_ids=ramp_ids):
                    trace.write(json.dumps(r)+'\n')
                for elapsed in range(1,601):
                    if elapsed<=180:
                        crossed.update(r['vid'] for r in observations(env) if 0<=r['s']<=75.44)
                    enabled=frozenset(members) if elapsed<=60 else frozenset()
                    a,kw,commands=bound.command(env,enabled) if bound else (None,{}, {})
                    _,info=env.step(a,**kw)
                    progress+=1
                    branch_progress+=1
                    m.update(info)
                    if elapsed==180:
                        local_crossing=sum(m.tracker.delay_by_vid.get(v,0.)+m.source.get(v,0.) for v in crossed)
                    observed=physics_rows(env)
                    digest.update(json.dumps(observed,separators=(',',':')).encode())
                    if elapsed<=90:
                        for r in trace_rows(observed,group,commands,ramp_ids):
                            trace.write(json.dumps(r)+'\n')
                    command_ticks+=bool(commands)
                    command_count+=len(commands)
                    if elapsed in (300,600):
                        values[str(elapsed)]=m.finish(save_trajectory=False)
                    if info['done'] or any(info['interval'].get(k,0) for k in
                                           ('collisions','teleports','dropped_vehicle_steps')):
                        raise ValueError('early end or safety anomaly')
                if any(lane_audit.values()):
                    raise ValueError('native lane audit failed')
            effects[key]=dict(local_cohort_delay_vs=local_crossing,local_crossed=sorted(crossed),
                              wall_events=bound.wall.events if bound and bound.wall else [],
                              wall_issued_commands=command_count if controller=='wall' else 0,
                              issued_commands=command_count,command_ticks=command_ticks,lane_audit=lane_audit)
            metrics[key],hashes[key]=values,digest.hexdigest()
            if controller=='sumocf':
                bound.bound.exe.close()
            env.close()
            env=None
            write(out/f'branch_{key}.json',dict(actual_sim_s=branch_progress,metrics=values,
                  trajectory_sha256=hashes[key],effects=effects[key],prefix_verified=True,
                  prefix_trajectory_sha256=prefix_digest.hexdigest()))
            if key=='native_B_repeat' or (mechanics and name=='DISABLED'):
                if metrics[key]!=metrics['native_B'] or hashes[key]!=hashes['native_B']:
                    raise ValueError('full-prefix native/no-op bitwise determinism failed')
            if mechanics and name=='GROUP_repeat':
                if metrics[key]!=metrics[controller+'_GROUP'] or hashes[key]!=hashes[controller+'_GROUP']:
                    raise ValueError('full-prefix controlled bitwise determinism failed')
        if mechanics and effects['wall_GROUP']['wall_issued_commands']==0:
            raise ValueError('wall positive control issued no commands')
        write(out/'result.json',dict(ok=True,frozen=frozen,mode=mechanics_name,window_s=60,
              branch_mode='full_prefix_rerun',selection_sha256=canonical(selection),actual_sim_s=progress,
              metrics=metrics,trajectory_sha256=hashes,effects=effects,missing_group=not group))
        return dict(seed=world['seed'],actual_sim_s=progress,group_size=len(group))
    except BaseException:
        write(out/'failure.json',dict(actual_sim_s=progress,traceback=traceback.format_exc()))
        write(ROOT/'STOP',dict(job=ident,unix=time.time(),reason='v2 failure; never retry'))
        raise
    finally:
        if env is not None:
            env.close()


def main():
    import fcntl
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('canary','main'))
    mode=p.parse_args().mode
    setup()
    selection=selected()
    with (ROOT/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if mode=='canary':
            Ledger(ROOT).admit_panel([selection['canary_charge_sim_s'],selection['main_charge_sim_s']])
            result=run_world(canary(),True)
            write(ROOT/'V2_CANARY_COMPLETE.json',dict(ok=True,**result))
        else:
            canary_receipt=json.loads((ROOT/'V2_CANARY_VERIFIED.json').read_text())
            if not canary_receipt['ok'] or not canary_receipt.get('commit'):
                raise ValueError('committed corrected canary required')
            Ledger(ROOT).admit_panel([selection['main_charge_sim_s']])
            write(ROOT/'V2_PANEL_ADMITTED.json',dict(total_sim_s=selection['main_charge_sim_s'],
                                                    selection_sha256=canonical(selection)))
            # At most six independent, single-thread worlds; every branch checks
            # priority flags/resources and takes the original atomic ledger lock.
            with concurrent.futures.ProcessPoolExecutor(max_workers=selection['max_workers']) as pool:
                futures=[pool.submit(run_world,w) for w in worlds()]
                for future in concurrent.futures.as_completed(futures):
                    result=future.result()
                    print(json.dumps(result),flush=True)
            write(ROOT/'V2_MAIN_COMPLETE.json',dict(ok=True,worlds=28,unix=time.time(),
                                                  selection_sha256=canonical(selection)))


if __name__=='__main__':
    main()
