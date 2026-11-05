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

def native_requests(mp,state,rows):
    """The frozen export's paired API is act_with_lane, not bare __call__."""
    import numpy as np
    acceleration,_=mp.act_with_lane(state)
    acceleration=np.asarray(acceleration)
    assert acceleration.shape==(1,len(rows))
    commands={r['vid']:float(acceleration[0,i]) for i,r in enumerate(rows) if r['is_cav']}
    assert all(np.isfinite(v) for v in commands.values())
    return commands

def prepare_query(worker,rt,binding,runtime):
    """Cheap observable/base-policy ABI preflight BEFORE expensive WM gates.

    This prepares inputs, not candidates or WM scores. Restore policy memory
    and CPU RNG so the original numerical startup fixture is unchanged.
    """
    import torch
    from tools.coopmwm.bank_v2v import trace_frames
    from tools.coopmwm.variety import decision_groups,plan_config
    from tools.coopmwm.protocol import digest
    w=worker.parent;inventory=read(w/'BANK_INVENTORY.json')
    item=next(x for x in inventory['rows'] if x['world_seed']==986026)
    frozen=w/'bank_verification_v5'/item['id']/'FROZEN_SNAPSHOT.pt'
    snapshot=torch.load(frozen,map_location='cpu',weights_only=False)
    assert snapshot['policy']['source']=='clean1 frozen MAPPO u0 native loader'
    assert not any(k.startswith(('p1.','t2c.')) for k in snapshot['weights'])
    assert snapshot['policy']['weights_sha256']==digest({k:v.tolist() for k,v in snapshot['weights'].items()})
    assert 986000<=snapshot['episode']['world']['seed']<=986071
    frame=trace_frames(item['trace'])[int(item['t0'])]
    fields=('vid','s_m','v_mps','lane','is_cav','length_m')
    priority=item['move']['members'][0]
    egos=sorted((r for r in frame['vehicles'] if r.get('is_cav')),key=lambda r:(r['vid']!=priority,r['vid']))
    selected=None
    for anchor in egos:
        ego=anchor['vid']
        rows=[{k:r[k] for k in fields if k in r} for r in frame['vehicles'] if abs(r['s_m']-anchor['s_m'])<=100.]
        groups=decision_groups(rows,plan_config(binding['search_config']))
        group=next((g for g in groups if ego in g['members'] and 2<=len(g['members'])<=4),None)
        if group is not None:selected=(ego,rows,group);break
    if selected is None:raise ValueError('no ego-local registered2..4-member group in the frozen source context')
    ego,rows,group=selected
    record=dict(world_id=item['world_seed'],time_s=item['t0'],rows=rows,group=group['members'],
        policy_state={k:v.detach().cpu().tolist() for k,v in snapshot['weights'].items() if k.startswith(('actor.','broadcast.'))},
        warm_memory={},commands={},policy_residuals={})
    record['state_id']=digest(dict(source_snapshot=sha(frozen),ego=ego,rows=rows,group=record['group']))
    source=runtime.diagnostic_source(rt,record,0)
    try:
        with torch.random.fork_rng(devices=[]):
            record['commands']=native_requests(rt.mp,source.state,rows)
            from tools.coopmwm.policy_wm import load_actor,residuals
            actor=load_actor(record['policy_state'],'cpu')
            current=residuals(actor,source.state,[r['vid'] for r in rows])
            record['policy_residuals']={r['vid']:float(current[0,i]) for i,r in enumerate(rows) if r['is_cav']}
    finally:rt.mp.reset()
    publish(worker/'SPEED_QUERY.json',dict(record=record,snapshot_path=str(frozen),snapshot_sha256=sha(frozen),
        roster_contract='one actual CAV sensing radius100m; causal kinematics only',
        native_command_source='original frozen MAPPO act_with_lane on that observable roster',
        prepared_before_gates_without_wm_search=True,new_sim_s=0))
    return record

def training_finished(root,launch):
    """Accept idle-after-training only with all three exact sealed u24 outputs."""
    for role in ('A','B','R'):
        path=root/(role+'_UPDATE_024.json')
        if not path.exists():return False
        row=read(path);body={k:v for k,v in row.items() if k!='update_sha256'}
        sealed=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if row.get('update_sha256')!=sealed or row.get('update')!=24 or row.get('arm')!=role or row.get('config_sha256')!=launch['runtime_sha256']:return False
        checkpoint=root/(role+'_checkpoint_024.pt')
        if not checkpoint.exists() or sha(checkpoint)!=row.get('checkpoint_sha256'):return False
    return True

def probe_health(worker):
    """No model import or signal: the original pace predicate, plus proved idle."""
    c=Path('/home/ubuntu/coopmwm20261105_clean2');baseline=read(worker/'CLEAN2_BASELINE.json')
    launch=read(c/'LAUNCHED.json')
    assert sha(c/'LAUNCHED.json')==baseline['launch_file_sha256']
    assert sha(c/'CONFIG.json')==launch['config_file_sha256']
    assert not (c/'STOP.json').exists() and not (c/'FAULT.json').exists()
    assert not (worker/'STOP.json').exists() and not (worker.parent/'STOP.json').exists()
    alive=[]
    for role,owner in launch['owners'].items():
        p=Path('/proc')/str(owner['pid'])/'stat';f=p.read_text().split() if p.exists() else None
        alive.append(bool(f and f[21]==owner['start_ticks'] and f[2]!='Z'))
    if not any(alive) and training_finished(c,launch):
        return dict(blocked=False,reason='proved_completed_clean2_idle',pace={},launch=launch)
    if not all(alive):
        return dict(blocked=True,reason='wait_original_clean2_owners_or_sealed_complete',pace={},launch=launch)
    pace={}
    for role,old in baseline['pace'].items():
        paths=sorted(c.glob(role+'_UPDATE_*.json'))
        latest=read(paths[-1])
        previous=read(paths[-2]);duration=latest['finished_unix']-previous['finished_unix']
        pace[role]=dict(update=latest['update'],interval_s=duration,limit_s=old['ceiling_s'],
            over=latest['update']>old['last_update'] and duration>old['ceiling_s'])
    blocked=any(x['over'] for x in pace.values())
    return dict(blocked=blocked,reason='unchanged_pace_hold' if blocked else 'active_training_in_bounds',pace=pace,launch=launch)

def health(worker):
    while True:
        value=probe_health(worker)
        if not value['blocked']:return value['launch']
        if getattr(health,'last_reason',None)!=value['reason']:
            print('CPU_RESOURCE_WAIT '+json.dumps({k:v for k,v in value.items() if k!='launch'}),flush=True)
            health.last_reason=value['reason']
        time.sleep(60)  # preserve the same model/RNG/partial fresh qualification

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
            prepare_query(worker,cached[0],binding,runtime)
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
    rt=cached[0];record=read(worker/'SPEED_QUERY.json')['record']
    health(worker)
    from tools.coopmwm.mppi_runtime import plan
    start=time.monotonic();cpu_start=time.process_time();publish(worker/'SPEED_SEARCH_STARTED.json',dict(owner=owner,record_sha256=digest(record),unix=time.time(),new_sim_s=0))
    result=plan(rt,record,None,binding)
    publish(worker/'SPEED_SEARCH_RESULT.json',dict(result=result,wall_s=time.monotonic()-start,active_cpu_s=time.process_time()-cpu_start,
        wall_includes_resource_pauses=True,owner=owner,
        all_eight_gates_same_pid=True,source_binding_sha256=sha(worker/'GPU_BINDING.json'),
        record_sha256=digest(record),new_sim_s=0,sumo_verified_plans=0,bank_switch_applied=False))

if __name__=='__main__':
    worker=Path(sys.argv[1])
    try:main(worker)
    except BaseException as error:
        import traceback
        publish(worker/'CPU_FAILED.json',dict(error=repr(error),traceback=traceback.format_exc(),unix=time.time(),new_sim_s=0))
        raise
