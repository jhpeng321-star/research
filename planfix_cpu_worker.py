"""One locked CPU worker: eight original startup gates, then one WM speed search.

No SUMO route exists in this worker. Its proposals are NOT admitted examples.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text())

def publish(path,value):
    with Path(path).open('x') as out:
        json.dump(value,out,sort_keys=True);out.flush();os.fsync(out.fileno())

def health(worker):
    c=Path('/home/ubuntu/coopmwm20261105_clean2');baseline=read(worker/'CLEAN2_BASELINE.json')
    launch=read(c/'LAUNCHED.json')
    assert sha(c/'LAUNCHED.json')==baseline['launch_file_sha256']
    assert sha(c/'CONFIG.json')==launch['config_file_sha256']
    assert not (c/'STOP.json').exists() and not (c/'FAULT.json').exists()
    assert not (worker/'STOP.json').exists()
    for role,owner in launch['owners'].items():
        p=Path('/proc')/str(owner['pid'])/'stat';f=p.read_text().split()
        assert f[21]==owner['start_ticks'] and f[2]!='Z', 'clean2 original owner no longer active'
    for role,old in baseline['pace'].items():
        paths=sorted(c.glob(role+'_UPDATE_*.json'))
        latest=read(paths[-1])
        if latest['update']>old['last_update']:
            previous=read(paths[-2])
            duration=latest['finished_unix']-previous['finished_unix']
            if duration>old['ceiling_s']:
                raise RuntimeError('own next WM evaluation held: clean2 update pace below frozen baseline')
    return launch

def main(worker):
    import fcntl,signal,subprocess
    worker=Path(worker);os.nice(19)
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpus[-1]})
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    lock=(worker/'CPU_WORKER.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),
        affinity=sorted(os.sched_getaffinity(0)),nice=os.nice(0),new_sim_s=0)
    publish(worker/'CPU_OWNER.json',owner)
    # Refuse a simulator at the process-launch boundary, while permitting
    # ordinary library/interpreter platform discovery during model imports.
    popen=subprocess.Popen
    def no_sumo(args,*pos,**kwargs):
        executable=str(args[0] if isinstance(args,(list,tuple)) else args).lower()
        if 'sumo' in executable:raise RuntimeError('CPU WM worker cannot launch SUMO')
        return popen(args,*pos,**kwargs)
    subprocess.Popen=no_sumo
    health(worker)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from planfix_cpu_runtime import install
    runtime=install(worker)
    import tools.coopmwm
    old=Path('/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa/pytools/tools/coopmwm')
    tools.coopmwm.__path__.append(str(old))
    from tools.coopmwm import fresh_worker,plan_service,warm_service,plan_wm
    from tools.coopmwm.protocol import digest
    original_evaluate=plan_wm.evaluate
    def guarded_evaluate(*args,**kwargs):
        health(worker)
        return original_evaluate(*args,**kwargs)
    plan_wm.evaluate=guarded_evaluate
    binding=read(worker/'GPU_BINDING.json')
    cached=[]
    def cpu_configured(bootstrap,fraction):
        if not cached:
            start=time.monotonic();cached.append(bootstrap())
            assert next(cached[0].model.parameters()).device.type=='cpu'
            publish(worker/'BOOTSTRAP_PASS.json',dict(wall_s=time.monotonic()-start,pid=os.getpid(),
                wm_sha256='b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436',
                physical_sha256=cached[0].export['sha256'],source_receipt=cached[0].receipt,new_sim_s=0))
        return cached[0]
    plan_service.configured_runtime=cpu_configured
    # All eight gate implementations and fresh-worker seals are the original
    # f631ce90 functions; only CUDA allocator setup is replaced by CPU bootstrap.
    stdin=sys.stdin;sys.stdin=[]
    try:warm_service.main(worker)
    finally:sys.stdin=stdin
    qualification=read(worker/'FRESH_QUALIFICATION.json')
    assert qualification['pass_'] and qualification['pid']==os.getpid()
    health(worker)
    # Prepare a registered TRAIN-world query from an existing causal bank frame.
    from tools.coopmwm.bank_v2v import trace_frames
    from tools.coopmwm.variety import decision_groups,plan_config
    w=worker.parent;inventory=read(w/'BANK_INVENTORY.json')
    item=next(x for x in inventory['rows'] if x['world_seed']==986026)
    frozen=w/'bank_verification_v5'/item['id']/'FROZEN_SNAPSHOT.pt'
    snapshot=torch.load(frozen,map_location='cpu',weights_only=False)
    assert snapshot['policy']['source']=='clean1 frozen MAPPO u0 native loader'
    assert 986000<=snapshot['episode']['world']['seed']<=986071
    frame=trace_frames(item['trace'])[int(item['t0'])]
    ego=item['move']['members'][0];anchor=next(r for r in frame['vehicles'] if r['vid']==ego)
    fields=('vid','s_m','v_mps','lane','is_cav','length_m')
    rows=[{k:r[k] for k in fields if k in r} for r in frame['vehicles'] if abs(r['s_m']-anchor['s_m'])<=100.]
    groups=decision_groups(rows,plan_config(binding['search_config']))
    group=next(g for g in groups if 2<=len(g['members'])<=4)
    record=dict(world_id=item['world_seed'],time_s=item['t0'],rows=rows,group=group['members'],
        policy_state={k:v.detach().cpu().tolist() for k,v in snapshot['weights'].items() if k.startswith(('actor.','broadcast.'))},
        warm_memory={},commands={},policy_residuals={})
    # Populate current MAPPO requests from the frozen original policy on this
    # observable roster. The zero-u0 residual must remain exactly zero.
    record['state_id']=digest(dict(source_snapshot=sha(frozen),ego=ego,rows=rows,group=record['group']))
    rt=cached[0];source=runtime.diagnostic_source(rt,record,0)
    acceleration,_=rt.mp(source.state)
    record['commands']={r['vid']:float(acceleration[0,i]) for i,r in enumerate(rows) if r['is_cav']}
    record['policy_residuals']={v:0. for v in record['commands']}
    from tools.coopmwm.policy_wm import load_actor
    actor=load_actor(record['policy_state'],'cpu');assert actor.zero_actor
    publish(worker/'SPEED_QUERY.json',dict(record=record,snapshot_path=str(frozen),snapshot_sha256=sha(frozen),
        roster_contract='one actual CAV sensing radius100m; causal kinematics only',
        native_command_source='original frozen MAPPO export on that observable roster',new_sim_s=0))
    health(worker)
    from tools.coopmwm.mppi_runtime import plan
    start=time.monotonic();publish(worker/'SPEED_SEARCH_STARTED.json',dict(owner=owner,record_sha256=digest(record),unix=time.time(),new_sim_s=0))
    result=plan(rt,record,None,binding)
    publish(worker/'SPEED_SEARCH_RESULT.json',dict(result=result,wall_s=time.monotonic()-start,owner=owner,
        all_eight_gates_same_pid=True,source_binding_sha256=sha(worker/'GPU_BINDING.json'),
        record_sha256=digest(record),new_sim_s=0,sumo_verified_plans=0,bank_switch_applied=False))

if __name__=='__main__':
    worker=Path(sys.argv[1])
    try:main(worker)
    except BaseException as error:
        import traceback
        publish(worker/'CPU_FAILED.json',dict(error=repr(error),traceback=traceback.format_exc(),unix=time.time(),new_sim_s=0))
        raise
