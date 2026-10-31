"""Owned, precharged saved-state audit. Main dispatch requires a bound SNR selection."""
import argparse
import contextlib
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import shutil
import time
import traceback

from tools.coopaudit.accounting import Ledger, write
from tools.coopaudit.design import (CAP, CONTROLLERS, MAPPO_SHA, SOURCE_SHA, canary,
                                    checked, guard, select_group, sha, worlds, branch_charge)

ROOT = Path(os.environ.get('COOPAUDIT_ROOT','/root/coopaudit20261031'))
RUNTIME = Path('/root/autodl-tmp/c4/speedhead20261030/e3final')
WRAPPERS = Path('/root/wmdata2')


def setup():
    # Preserve this package's tools namespace while adding only frozen sibling tools.
    import tools
    tools.__path__.append(str(WRAPPERS/'tools'))
    sys.path[:0] = [str(RUNTIME/'runtime/src'),str(RUNTIME/'runtime'),str(WRAPPERS/'tools/speedhead'),
                   str(WRAPPERS/'tools/ctrlarm')]
    import torch
    torch.set_num_threads(1)
    if os.getpriority(os.PRIO_PROCESS,0) != 19 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('nice19 / CPU-only worker required')
    import importlib.metadata
    if importlib.metadata.version('eclipse-sumo') != '1.27.1':
        raise ValueError('wrong SUMO version')
    manifest = RUNTIME/'runtime/SOURCE_MANIFEST.json'
    if sha(manifest) != SOURCE_SHA:
        raise ValueError('runtime manifest changed')
    from tools.coopaudit.design import RESTRICTED
    entries = json.loads(manifest.read_text())
    for name,h in entries.items():
        if not RESTRICTED.search(name) and sha(RUNTIME/'runtime'/name) != h:
            raise ValueError('scientific runtime changed')
    inputs = dict(mappo='/root/autodl-tmp/c4/speedhead20261030/mappospeed_final169.pt',
                  ctrl0=str(RUNTIME/'inputs/policy_latest.pt'))
    expected = {'tools/ctrlarm/zero.py':'24095a1092eafb4c894804aa8142478a802eaea046142ec777e5f24f49f73596',
                'tools/speedhead/observer.py':'b22af4623b5fe7fbb9a8920b8e8417c3591efae66a52bae7b72f4d899b4a7874',
                'tools/speedhead/baselines.py':'c63b14bfaa49b993cade4608b0e69f714de6a286ea9cb1049816118541f4f3eb'}
    if any(sha(WRAPPERS/n)!=h for n,h in expected.items()) or sha(inputs['mappo'])!=MAPPO_SHA:
        raise ValueError('controller wrapper/hash mismatch')
    if sha(inputs['ctrl0'])!='64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c':
        raise ValueError('CTRL0 input changed')
    from observer import audit_lanes
    from zero import forced_base_commands, ZeroResidual
    if not shutil.which('sumo'):
        raise ValueError('SUMO not on PATH; refusing BEFORE any reservation')
    return inputs


def canonical(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def initial_state(env):
    return canonical({k:v.detach().cpu().tolist() for k,v in env.state().items() if hasattr(v,'detach')})


def crn(env):
    return canonical(sorted((r.vehicle_id,float(r.depart_s),r.route,r.vehicle_id in env.cav_ids)
                            for r in env.requests))


def physics_rows(env):
    import numpy as np
    raw = env.history.raw(3)
    rows = []
    for j in (1,2):
        for s in np.flatnonzero(raw['mask'][j]):
            kin = raw['kin'][j][s]
            onehot = kin[1:4]
            lane = int(onehot.argmax()) if onehot.max()>.5 else -1
            vid = env.builder.vehicle_ids[int(raw['vid_index'][j][s])]
            rows.append(dict(t=float(env.now-.5*(2-j)), vid=vid, s=float(kin[0]),
                             lane=lane, v=float(kin[4]), a=float(kin[5])))
    return sorted(rows,key=lambda r:(r['t'],r['vid']))


def trace_rows(rows, group, commands=None):
    by_t = {}
    for r in rows:
        by_t.setdefault(r['t'],[]).append(r)
    output = []
    for cohort in by_t.values():
        positions = [r['s'] for r in cohort if r['vid'] in group]
        for r in cohort:
            member = r['vid'] in group
            if member or (r['lane']==2 and (-300<=r['s']<=75.44 or
                                          any(abs(r['s']-s)<=300 for s in positions))):
                output.append(dict(r,member=member,requested_accel=(commands or {}).get(r['vid'])))
    return output


def wait_spare():
    while os.getloadavg()[0] > 20 or (ROOT/'YIELD').exists():
        if (ROOT/'STOP').exists():
            raise ValueError('owned STOP')
        write(ROOT/'WAIT.json',dict(load=os.getloadavg(),unix=time.time(),reason='priority/load'))
        time.sleep(60)
    if (ROOT/'STOP').exists():
        raise ValueError('owned STOP')


def execute(world, mode, selection=None, attempt=1):
    import numpy as np
    from traffic_shepherd.c4 import branching as B
    from traffic_shepherd.c4 import evalprep as E
    from traffic_shepherd.c4.env import CavSumoEnv
    from tools.coopaudit.controllers import Controller, observations, wall_roster, ctrl0_scope
    from observer import audit_lanes
    inputs = setup()
    guard(world['seed'],world['demand'])
    ledger = Ledger(ROOT)
    ident = f"{mode}_s{world['seed']}"
    if attempt!=1:
        if mode!='canary' or attempt!=2 or not (ROOT/'ZERO_STEP_REPAIR.json').exists():
            raise ValueError('only documented zero-step canary infrastructure repair allowed')
        ident += '_attempt2'
    out = ROOT/'jobs'/ident
    if out.exists():
        raise ValueError('attempt directory exists; never retry')
    wait_spare()
    ledger.charge(ident+'_prefix',300)
    out.mkdir(parents=True)
    progress = {'sim_s':0}
    write(out/'START.json',dict(world=world,mode=mode,pid=os.getpid(),unix=time.time()))
    prefix = []
    H = 180 if mode=='canary' else 600
    window = 60 if mode=='canary' else selection['window_s'] if selection else 60
    env = CavSumoEnv(k_slots=192,controller='direct',extra_sumo_cli=B.SAVE_STATE_CLI,
                     scenario_root=out/'scenarios',physics_log=False,population='textbook')
    try:
        def warmup_hook(e):
            progress['sim_s'] = e.now
            if e.now>270:
                prefix.extend(physics_rows(e))
        env.reset(seed=world['seed'],demand_vph=world['demand'],penetration=world['penetration'],
                  warmup_s=300,horizon_s=H+1,controller='direct',warmup_hook=warmup_hook)
        progress['sim_s'] = 300
        rows = observations(env)
        group = select_group(rows,size=selection['group_size'] if selection else 'all')
        roster = wall_roster(rows,group)
        if mode=='canary':
            # Geometry-only wall pair for the mechanics canary; main G remains SNR-selected.
            group = roster
            if not group:
                raise ValueError('canary has no geometrically feasible wall; no replacement')
        frozen = dict(world=world,group=group,wall_roster=roster,observations=rows,
                      initial_state_sha256=initial_state(env),crn_sha256=crn(env))
        write(out/'FROZEN.json',frozen)
        if mode=='census':
            write(out/'result.json',dict(ok=True,actual_sim_s=300,frozen=frozen))
            return
        if mode=='main':
            census = json.loads((ROOT/'jobs'/f"census_s{world['seed']}"/'FROZEN.json').read_text())
            if any(census[k]!=frozen[k] for k in ('group','initial_state_sha256','crn_sha256')):
                raise ValueError('census/group/prefix identity mismatch')
        state = out/'start.xml'
        B.save_state(env,state)
        snap = B.snapshot_env(env)
        names = ('wall','mappo') if mode=='canary' else CONTROLLERS
        branches = [('native','B',frozenset()),('native','B_repeat',frozenset())]
        for controller in names:
            branches += [(controller,'GROUP',frozenset(group))]
            branches += [(controller,f'ONLY:{i}',frozenset([v])) for i,v in enumerate(group)]
            if mode=='canary':
                branches += [(controller,'GROUP_repeat',frozenset(group))]
        if mode=='canary':
            branches.insert(0,('native','LIVE',frozenset()))
        charge = H*len(branches)
        if mode=='canary' and 300+charge > 20000:
            raise ValueError('canary subcap')
        ledger.charge(ident+'_branches',charge)
        metrics, hashes, effects = {}, {}, {}
        (out/'traces').mkdir()
        for controller,name,members in branches:
            if name!='LIVE':
                B.load_state(env,state,snap)
            B.prepare_branch_mode(env)
            if initial_state(env)!=frozen['initial_state_sha256'] or crn(env)!=frozen['crn_sha256']:
                raise ValueError('same-state/RNG identity mismatch')
            bound = Controller(controller,inputs,roster) if controller!='native' else None
            m = E.Measurement()
            m.path = out
            m.start(env)
            digest = hashlib.sha256()
            clocks = [180] if mode=='canary' else [300,600]
            values, local, cohort_delay, touched = {}, 0., {}, set()
            key = controller+'_'+name
            issued = 0
            scope = ctrl0_scope() if controller=='ctrl0' else contextlib.nullcontext()
            with gzip.open(out/'traces'/f'{key}.jsonl.gz','wt') as trace, scope, audit_lanes() as lane_audit:
                for r in trace_rows(prefix,group):
                    trace.write(json.dumps(r)+'\n')
                for elapsed in range(1,H+1):
                    enabled = members if elapsed<=window else frozenset()
                    a,kw,commands = bound.command(env,enabled) if bound else (None,{}, {})
                    _,info = env.step(a,**kw)
                    progress['sim_s'] += 1
                    m.update(info)
                    observed = physics_rows(env)
                    digest.update(json.dumps(observed,separators=(',',':')).encode())
                    if elapsed<=window+120:
                        for r in observed:
                            if -150<=r['s']<=75.44:
                                local += .5*max(0.,1-r['v']/31.29)
                                touched.add(r['vid'])
                            cohort_delay[r['vid']] = cohort_delay.get(r['vid'],0.)+.5*max(0.,1-r['v']/31.29)
                    if elapsed<=window+30:
                        for r in trace_rows(observed,group,commands):
                            trace.write(json.dumps(r)+'\n')
                    issued += len(commands)
                    if elapsed in clocks:
                        values[str(elapsed)] = m.finish(save_trajectory=False)
                    if info['done'] or any(info['interval'].get(k,0) for k in
                                           ('collisions','teleports','dropped_vehicle_steps')):
                        raise ValueError('early end or safety anomaly')
                if any(lane_audit.values()):
                    raise ValueError('controller lane-call audit failed')
            effects[key] = dict(local_zone_delay_vs=local,cohort_delay=cohort_delay,touched=sorted(touched),
                                local_cohort_delay_vs=sum(cohort_delay[v] for v in touched),
                                wall_events=bound.wall.events if bound and bound.wall else [],
                                wall_issued_commands=issued,lane_audit=lane_audit)
            metrics[key],hashes[key] = values,digest.hexdigest()
            write(out/f'branch_{key}.json',dict(metrics=values,trajectory_sha256=hashes[key],effects=effects[key]))
        if metrics['native_B']!=metrics['native_B_repeat'] or hashes['native_B']!=hashes['native_B_repeat']:
            raise ValueError('bitwise native branch determinism failed')
        if mode=='canary':
            if hashes['native_LIVE']!=hashes['native_B'] or metrics['native_LIVE']!=metrics['native_B']:
                raise ValueError('live-trunk/loadState fidelity failed')
            for c in names:
                if hashes[c+'_GROUP']!=hashes[c+'_GROUP_repeat'] or metrics[c+'_GROUP']!=metrics[c+'_GROUP_repeat']:
                    raise ValueError('controlled branch determinism failed')
            if effects['wall_GROUP']['wall_issued_commands']==0:
                raise ValueError('positive control issued no wall commands')
        write(out/'result.json',dict(ok=True,frozen=frozen,mode=mode,window_s=window,
                                    actual_sim_s=progress['sim_s'],metrics=metrics,
                                    trajectory_sha256=hashes,effects=effects))
    except BaseException:
        write(out/'failure.json',dict(actual_sim_s=progress['sim_s'],traceback=traceback.format_exc()))
        write(ROOT/'STOP',dict(job=ident,unix=time.time()))
        raise
    finally:
        env.close()


def main():
    import fcntl
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=('canary','census','main'))
    p.add_argument('--attempt',type=int,choices=(1,2),default=1)
    a = p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        setup()
        if a.mode=='canary':
            execute(canary(),'canary',attempt=a.attempt)
            return
        receipt = ROOT/'CANARY_VERIFIED.json'
        if not receipt.exists():
            raise ValueError('verified canary required')
        selected = json.loads((ROOT/'SELECTED_DESIGN.json').read_text())
        if selected['group_size'] not in (2,4,'all') or selected['window_s'] not in (10,30,60):
            raise ValueError('unregistered SNR selection')
        if not selected.get('sumocf_table_sha256') or not selected.get('source_commit'):
            raise ValueError('SNR provenance binding required')
        if a.mode=='main':
            charges = []
            for w in worlds():
                if (ROOT/'jobs'/f"main_s{w['seed']}"/'result.json').exists():
                    continue
                doc = json.loads((ROOT/'jobs'/f"census_s{w['seed']}"/'FROZEN.json').read_text())
                charges.append(300+branch_charge(len(doc['group'])))
            Ledger(ROOT).admit_panel(charges)
            write(ROOT/'PANEL_ADMITTED.json',dict(charges=charges,total=sum(charges),selection=selected))
        # Sequential admission leaves spare capacity to S-SUMOCF and other priority jobs.
        for w in worlds():
            result = ROOT/'jobs'/f"{a.mode}_s{w['seed']}"/'result.json'
            if result.exists():
                continue
            execute(w,a.mode,selected)
        write(ROOT/f'{a.mode.upper()}_COMPLETE.json',dict(ok=True,worlds=28,unix=time.time()))


if __name__=='__main__':
    # Import frozen modules only after setup; do not create an alternate environment.
    setup()
    main()
