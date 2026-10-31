"""Owned one-thread state-conditional ensemble. No assignment/truth file inspection."""
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(os.environ.get('ENSPROBE_ROOT','/home/ubuntu/ensprobe20261031'))
RUNTIME = Path('/home/ubuntu/e3final/runtime')
MAPPO = Path('/home/ubuntu/mappospeed/production2/policy_latest.pt')
SOURCE = 'b3c031adb6bc2b5c2bca160bea308f83ba5567b97c1cd0e853eddcd8367a8193'
WORLD = [(969040,.05,2400),(969041,.10,2700),(969042,.2,2400),(969043,.4,2700),(969044,.2,2700)]
RNG = list(range(969086,969094))
MEASURE = None


def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def priority_wait():
    """Read-only priority observation, never signal or modify a peer process."""
    while True:
        if (ROOT/'STOP').exists() or (ROOT/'YIELD').exists():
            if (ROOT/'STOP').exists():raise RuntimeError('durable STOP')
            time.sleep(5);continue
        busy = os.getloadavg()[0] > 20
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name)==os.getpid():continue
            try:
                cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').lower()
                state=(p/'stat').read_text().rsplit(')',1)[1].split()[0]
                if state=='R' and any(x in cmd for x in ('tools.leverprobe','tools.c1switch','tools.c1closed','tools.wmvalue')):
                    busy=True
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
        if not busy:return
        time.sleep(5)


class Ledger:
    def __init__(self):self.allowed=0.;self.actual=0.;self.ident=None

    def charge(self, ident, seconds):
        path=ROOT/'accounting.jsonl'
        rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        prior=json.loads((ROOT/'BINDING.json').read_text()).get('recovery',{})
        if ident in {r['id'] for r in rows}:
            # Explicit pre-outcome recovery: this precise reservation advanced ZERO ticks.
            assert ident==prior.get('unspent_branch_id') and seconds==600
            assert [r['sim_s'] for r in rows if r['id']==ident]==[600]
            write(ROOT/'UNSPENT_RESERVATION_RESUMED.json',dict(id=ident,sim_s=600,refund=0,new_charge=0))
            self.ident,self.allowed,self.actual=ident,600.,0.
            return
        assert sum(r['sim_s'] for r in rows)+seconds<=49800
        with path.open('a') as f:
            f.write(json.dumps(dict(id=ident,sim_s=seconds,unix=time.time()))+'\n');f.flush();os.fsync(f.fileno())
        self.ident,self.allowed,self.actual=ident,float(seconds),0.

    def receipt(self):
        return dict(id=self.ident,charged_sim_s=self.allowed,actual_sim_s=self.actual)


def physical(env):
    """Public kinematics/routes/queues only: never query vehicle type or parameters."""
    api=env.api
    rows={v:dict(lane=api.vehicle.getLaneID(v),route=list(api.vehicle.getRoute(v)),
                route_index=api.vehicle.getRouteIndex(v),speed=api.vehicle.getSpeed(v),
                lane_pos=api.vehicle.getLanePosition(v),xy=list(api.vehicle.getPosition(v)))
          for v in sorted(api.vehicle.getIDList())}
    due={vid:r for vid,r in env.ledger.requests.items() if r.depart_s<=env.now and vid not in env.ledger.inserted}
    waiting={v:dict(lane=api.vehicle.getLaneID(v),route=list(api.vehicle.getRoute(v)),
                    departure=api.vehicle.getDeparture(v)) for v in sorted(due)}
    assert all(r['lane']=='' and r['departure']<0 for r in waiting.values())
    assert len(waiting)==env.entry_queue_now()
    return dict(time=api.simulation.getTime(),vehicles=rows,waiting=waiting,
                pending=sorted(api.simulation.getPendingVehicles()),source_queue=env.entry_queue_now(),
                lane_queues={l:api.lane.getLastStepHaltingNumber(l) for l in api.lane.getIDList()})


def compare(trunk, branch):
    out=dict(ids_equal=set(trunk['vehicles'])==set(branch['vehicles']),max_position_m=0.,max_speed_mps=0.,
             max_xy_m=0.,lanes_equal=True,routes_equal=True,queues_equal=True,time_error_s=abs(trunk['time']-branch['time']))
    for v in set(trunk['vehicles']) & set(branch['vehicles']):
        a,b=trunk['vehicles'][v],branch['vehicles'][v]
        out['max_position_m']=max(out['max_position_m'],abs(a['lane_pos']-b['lane_pos']))
        out['max_speed_mps']=max(out['max_speed_mps'],abs(a['speed']-b['speed']))
        out['max_xy_m']=max(out['max_xy_m'],*(abs(x-y) for x,y in zip(a['xy'],b['xy'])))
        out['lanes_equal'] &= a['lane']==b['lane']
        out['routes_equal'] &= (a['route'],a['route_index'])==(b['route'],b['route_index'])
    # getPendingVehicles() is an insertion-attempt cache, empty immediately after loadState.
    # Compare the actual due waiting objects, their routes and not-yet-departed status instead.
    out['pending_api_equal']=trunk['pending']==branch['pending']
    out['queues_equal']=all(trunk[k]==branch[k] for k in ('waiting','source_queue','lane_queues'))
    out['pass']=all(out[k] for k in ('ids_equal','lanes_equal','routes_equal','queues_equal')) and \
        max(out['max_position_m'],out['max_speed_mps'],out['max_xy_m'])<=1e-6 and out['time_error_s']<=1e-9
    return out


class Measurement:
    def __init__(self, prefix=None):
        self.sums={k:[0.]*66 for k in ('upstream','merge','ramp','downstream')}
        self.counts={k:[0]*66 for k in self.sums}
        self.exact_prefix_bins=None
        if prefix:
            for k in self.sums:
                self.sums[k][:6]=prefix.sums[k][:6]
                self.counts[k][:6]=prefix.counts[k][:6]
            self.exact_prefix_bins=copy.deepcopy(prefix.exact_prefix_bins)
        self.road=self.source=0.;self.clocks={};self.safety={}
        self.tape=hashlib.sha256()

    def before(self, env, t):
        from traffic_shepherd.c4.geometry import lane_and_s
        if not 240<=t<900:return
        b=int((t-240)//10)
        for row in env._table.values():
            if row['lane_id'].startswith(':'):continue
            lane,s=lane_and_s(row['lane_id'],row['lane_position_m'])
            if lane not in (0,1,2) or s is None:continue
            region='ramp' if lane==2 else 'upstream' if s<0 else 'merge' if s<=75.44 else 'downstream'
            self.sums[region][b]+=float(row['speed_mps']);self.counts[region][b]+=1

    def after(self, env, t, before, interval):
        if t<=300:return
        speeds={v:r['speed_mps'] for v,r in env._table.items()}
        self.road += .5*sum(max(0.,1.-speeds.get(v,speed)/31.29) for v,speed in {**before,**speeds}.items())
        self.source += .5*env.entry_queue_now()
        # Entire physical continuation is hashed, not merely its final objective.
        self.tape.update(json.dumps([t,[(v,r['lane_id'],r['lane_position_m'],r['speed_mps'])
                                      for v,r in sorted(env._table.items())]],separators=(',',':')).encode())
        for key in ('collision_involvements','starting_teleports','dropped_vehicle_steps'):
            self.safety[key]=self.safety.get(key,0)+interval.get(key,0)
        elapsed=int(round(t-300))
        if t-300==elapsed and elapsed in (120,300,600):
            self.clocks[str(elapsed)]=dict(road=self.road,source=self.source,J=self.road+self.source)

    def result(self):
        bins={k:[s/n if n else None for s,n in zip(self.sums[k],self.counts[k])] for k in self.sums}
        if self.exact_prefix_bins:
            for k in bins:bins[k][:6]=self.exact_prefix_bins[k]
        return dict(clocks=self.clocks,safety=self.safety,tape_sha256=self.tape.hexdigest(),
                    speed_bins=bins,
                    bin_counts=self.counts)


def reconstruct_first(env, previous):
    """No physics: public t0 kinematics + due-waiting identities, current-frame policy API."""
    import traci
    from collections import Counter
    old=Path(previous)/'w969040'
    reference=json.loads((old/'TRUNK.json').read_text())
    retained=json.loads((old/'w969040_native_r969086_RESULT.json').read_text())
    assert retained['actual_sim_s']==600 and retained['gate']['pass']
    env.reset(seed=969040,demand_vph=2400,penetration=.05,warmup_s=0,horizon_s=901,controller='direct')
    traci.simulation.loadState(str(old/'physical_state.xml'))
    env.warmup_s=300.;env.horizon_s=601.;env.end_s=901.
    due={v for v,r in env.ledger.requests.items() if r.depart_s<=300}
    env.ledger.inserted=due-set(reference['physical']['waiting'])
    live=set(traci.vehicle.getIDList())
    env.ledger.arrived=env.ledger.inserted-live
    env.ledger.time=300.
    env.reader._subscribed=set();env._table=env.reader.read()
    env.frame=env.builder.build(300.,env._table);env.history.clear();env.history.append(env.frame)
    env._previous_row={'vehicles':env._table};env._previous_live_sub=set(live)
    env._previous_live=set(live);env.window_totals=Counter()
    env._window_ledger0=(0.,0.,len(env.ledger.arrived))
    for v in sorted(live & env.cav_ids):traci.vehicle.setSpeedMode(v,env.ownership.cav_speed_mode)
    env.ownership._mode_configured=live & env.cav_ids
    assert compare(reference['physical'],physical(env))['pass']
    prefix=Measurement()
    prefix.exact_prefix_bins={k:values[:6] for k,values in retained['speed_bins'].items()}
    for k in prefix.counts:prefix.counts[k][:6]=retained['bin_counts'][k][:6]
    return reference,retained,prefix


def setup():
    import re
    sys.path[:0]=[str(RUNTIME/'src'),str(RUNTIME)]
    assert os.getpriority(os.PRIO_PROCESS,0)==19 and os.environ['CUDA_VISIBLE_DEVICES']==''
    assert importlib.metadata.version('eclipse-sumo')=='1.27.1'
    assert sha(RUNTIME/'SOURCE_MANIFEST.json')==SOURCE
    bad=re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vehicle_params|driver_params|assignment|requests\.rou|driver_routes',re.I)
    manifest=json.loads((RUNTIME/'SOURCE_MANIFEST.json').read_text())
    assert len(manifest)==543
    checked=0
    for name,h in manifest.items():
        if not bad.search(name):assert sha(RUNTIME/name)==h;checked+=1
    assert sha(MAPPO)=='d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'
    import torch
    torch.set_num_threads(1)
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    for s,p,d in WORLD:
        require_development_case(s,d,'exposed');check_case(s,d)
    for s in RNG:require_development_case(s,2400,'exposed');check_case(s,2400)
    return dict(source_manifest=SOURCE,permitted_sources_verified=checked,worlds=WORLD,rng=RNG,new_sim_s=0)


def main():
    global MEASURE
    import fcntl
    lock=(ROOT/'WORKER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'STOP').exists() and not (ROOT/'COMPLETE.json').exists()
    write(ROOT/'PREFLIGHT.json',setup())
    import traci
    import numpy as np
    import torch
    from traffic_shepherd.c4.env import CavSumoEnv
    from traffic_shepherd.simulation import start_sumo
    from wms.control.policy import load_policy_fn
    from wms.control.lanes import LaneDwell
    ledger=Ledger();env=None
    original_step=traci.simulationStep
    def counted_step(*a,**kw):
        assert ledger.actual+.5<=ledger.allowed,'step without precharge'
        before=traci.simulation.getTime()
        result=original_step(*a,**kw)
        ledger.actual+=traci.simulation.getTime()-before
        return result
    traci.simulationStep=counted_step
    original_after=CavSumoEnv._after_physics
    def observe(e,after,actions,command_time,commanded):
        MEASURE.before(e,after-.5)
        before={v:r['speed_mps'] for v,r in getattr(e,'_table',{}).items()}
        interval=original_after(e,after,actions,command_time,commanded)
        MEASURE.after(e,after,before,interval)
        return interval
    CavSumoEnv._after_physics=observe
    try:
        for seed,penetration,demand in WORLD:
            priority_wait();out=ROOT/f'w{seed}';out.mkdir(exist_ok=False)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            recovery=json.loads((ROOT/'BINDING.json').read_text()).get('recovery',{})
            MEASURE=Measurement()
            env=CavSumoEnv(k_slots=192,controller='direct',scenario_root=out/'scenarios',
                           extra_sumo_cli=['--save-state.precision','17','--save-state.rng','false'],
                           physics_log=False,population='textbook')
            reconstruct=recovery.get('reconstruct_first',False) and seed==969040
            if reconstruct:
                reference,retained,prefix=reconstruct_first(env,recovery['previous_root'])
                trunk=reference['physical']
                write(out/'TRUNK.json',reference)
                write(out/'w969040_native_r969086_RESULT.json',retained)
                write(out/'w969040_native_r969086_GATE.json',retained['gate'])
                write(out/'ZERO_TICK_RECONSTRUCTION.json',dict(new_actual_sim_s=0,physical_gate=compare(trunk,physical(env)),
                       current_frame_only_policy_api=True,retained_prefix_and_first_result=recovery['previous_root']))
            else:
                ledger.charge(f'w{seed}_trunk_repeat' if recovery and seed==969040 else f'w{seed}_trunk',300)
                env.reset(seed=seed,demand_vph=demand,penetration=penetration,warmup_s=300,horizon_s=601,
                          controller='direct',warmup_hook=lambda e:priority_wait())
                assert ledger.actual==300
                trunk=physical(env);prefix=MEASURE
                write(out/'TRUNK.json',dict(world=seed,penetration=penetration,demand=demand,physical=trunk,**ledger.receipt()))
            if recovery and seed==969040 and not reconstruct:
                original=json.loads((Path(recovery['original_root'])/'w969040/TRUNK.json').read_text())['physical']
                assert all(original[k]==trunk[k] for k in original),'repeat warmup changed original physical state'
                write(out/'REPEAT_TRUNK_GATE.json',dict(pass_gate=True,max_physical_error=0.,new_actual_sim_s=300))
            statefile=out/'physical_state.xml'
            if reconstruct:statefile=Path(recovery['previous_root'])/'w969040/physical_state.xml'
            else:traci.simulation.saveState(str(statefile))  # no Python read/parse/hash of assignment-bearing XML
            shared={id(env.api):env.api,id(env._circles):env._circles}
            snap=copy.deepcopy(env.__dict__,dict(shared))
            for arm in ('native','mappo'):
                for rng in RNG:
                    if reconstruct and arm=='native' and rng==969086:continue  # immutable successful result reused
                    priority_wait();ident=f'w{seed}_{arm}_r{rng}'
                    ledger.charge(ident,600)
                    env.close()
                    start_sumo(env.binary,env.scene,seed=rng,config=env.config,api=traci,
                               extra_cli=env.extra_sumo_cli)
                    traci.simulation.loadState(str(statefile))
                    env.__dict__.update(copy.deepcopy(snap,dict(shared)))
                    env.reader._subscribed=set()
                    live=set(traci.vehicle.getIDList())
                    for vid in sorted(env.ownership._mode_configured & live):
                        traci.vehicle.setSpeedMode(vid,env.ownership.cav_speed_mode)
                    gate=compare(trunk,physical(env))
                    gate.update(sumoseed=traci.simulation.getOption('seed'),rng_saved=traci.simulation.getOption('save-state.rng'))
                    write(out/(ident+'_GATE.json'),gate)
                    assert gate['pass'],f'physical-state gate failed: {gate}'
                    assert int(gate['sumoseed'])==rng and gate['rng_saved']=='false'
                    env.mode='native' if arm=='native' else 'direct'
                    exe=load_policy_fn(MAPPO,device='cpu') if arm=='mappo' else None
                    dwell=LaneDwell(exe.lane_rules.dwell_s,1.) if exe else None
                    MEASURE=Measurement(prefix)
                    state=env.state()
                    for _ in range(600):
                        priority_wait()
                        ages=dwell.step(state) if dwell is not None else None
                        command,lanes=exe(state,with_lanes=True,dwell_age=ages) if exe else (None,None)
                        assert lanes is None,'speed-only MAPPO attempted lane control'
                        state,info=env.step(command,lanes=lanes)
                        assert not any(info['lane_commands'].values())
                        assert not any(MEASURE.safety.values()),'safety or slot anomaly'
                    assert ledger.actual==600 and env.now==900
                    write(out/(ident+'_RESULT.json'),dict(world=seed,arm=arm,rng=rng,penetration=penetration,
                                                        demand=demand,gate=gate,**ledger.receipt(),**MEASURE.result()))
                    print(json.dumps(dict(complete=ident,actual_sim_s=ledger.actual)),flush=True)
            env.close();env=None
        rows=[json.loads(s) for s in (ROOT/'accounting.jsonl').read_text().splitlines()]
        total=sum(r['sim_s'] for r in rows)
        assert total==49800
        write(ROOT/'COMPLETE.json',dict(charged_sim_s=total,actual_sim_s=total,branches=80,trunks=5,
                                        extra_native_prefix_sim_s=300))
    except BaseException:
        write(ROOT/'FAILURE.json',dict(traceback=traceback.format_exc(),**ledger.receipt()))
        with (ROOT/'STOP').open('x') as f:f.write('gate or implementation failure; no automatic retries\n')
        raise
    finally:
        if env is not None:env.close()
        CavSumoEnv._after_physics=original_after;traci.simulationStep=original_step


if __name__=='__main__':main()
