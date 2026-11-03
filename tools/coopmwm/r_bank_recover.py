"""Recover missed R-u4 snapshots from saved observable tapes, with exact RNG replay.

No episode is rerun. Every regenerated own action must be bitwise identical before
publishing any state; the existing random-eight/exact verifier consumes the states.
"""
import copy
import gzip
import json
from pathlib import Path
import time
from types import SimpleNamespace
import torch
from .protocol import digest
from .storage import write
from .accel_storage import read


def regenerate(policy, result, noise_seed, until):
    generator=torch.Generator().manual_seed(noise_seed);memory={};events=[]
    for step in result['steps']:
        if step['time_s']>until:break
        saved=step['sample']
        assert not saved.get('behaviour',{}).get('executed'), 'not an own-policy episode'
        sample,memory=policy.sample(saved['context'],generator,False,memory)
        for k in ('own_action','action','previous_noise'):
            assert sample[k]==saved[k], 'saved R action/RNG mismatch: '+k
        if step['time_s']==until:return generator,memory,events,copy.deepcopy(saved)
        events.append(copy.deepcopy(step))
    raise ValueError('saved R decision second absent')


def prepare(root, job_path, binding):
    from .accel_learner import load
    from .warm_protocol import SEED
    from .warm_history import ObservableHistory,FIELDS
    from .mppi_search import group_components,final_horizon,REGISTERED
    from .exploration import regime
    from .policy_wm import actor_state
    from .sumo_verify import capture
    result=read(job_path/'RESULT.json');assert result['arm']=='R'
    assert 986000<=result['world']['seed']<=986071
    policy,_,_,payload=load(root/'R_checkpoint_003.pt');assert payload['update']==3
    noise_seed=int(digest(dict(seed=SEED,world=result['world']['seed'],wave='replay0735_warm_u004'))[:15],16)
    history=ObservableHistory()
    with gzip.open(job_path/'observable.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);t=int(row['time_s'])
            rows=[{k:v[k] for k in FIELDS} for v in row['vehicles']]
            history.push(t,rows)
            if t<360 or t%10 or t+final_horizon(REGISTERED['max_duration_s'])>binding['demand_phase_end_s']:continue
            groups=group_components(rows)
            if not groups:continue
            commands={a['vid']:a['mappo_raw'] for a in row['actions']}
            if any(v['is_cav'] and v['vid'] not in commands for v in rows):continue
            generator,memory,events,sample=regenerate(policy,result,noise_seed,t)
            job=dict(id=result['id'],world=result['world'],arm='R',update=4,role='train',
                training_seed=SEED,runtime=binding['runtime'],checkpoint=binding['checkpoint'],
                warm_history=True,accel_bounds=binding['accel_bounds'],noise_seed=noise_seed)
            controller=SimpleNamespace(policy=policy,events=events,memory=memory,active_plans=[])
            snap=capture(controller,result['prefixes'],job,t,sample,generator)
            record=dict(world_id=result['world']['seed'],time_s=t,rows=rows,group=groups[0]['members'],
                commands=commands,policy_residuals={v:float(a[0]) for v,a in
                    zip(sample['context']['members'],sample['own_action'])},history=history.snapshot(t),
                mode='centre',policy_state=actor_state(policy))
            record['state_id']=digest(dict(record=record,job=result['id']))
            meta=dict(state_id=record['state_id'],arm='R',job=result['id'],world=result['world'],t0=t,
                group=groups[0],regime=regime(rows),policy_update=3,record_sha256=digest(record),
                unix=time.time(),recovered_from_saved_own_episode=True,RNG_actions_bitwise=True)
            return snap,record,meta
    return None


def run(root, control):
    import fcntl,os,signal
    from .replay_start import config,binding,atomic
    root=Path(root);control=Path(control)
    if (control/'R_RECOVER_BINDING.json').exists():
        from .protocol import sha
        v=read(control/'R_RECOVER_BINDING.json');s=Path(v['source_root'])
        assert v['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
        assert Path(__file__).resolve()==s/'pytools/tools/coopmwm/r_bank_recover.py'
        cpu=read(s/'LS_CPU_RESULT.json');assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
        for n,h in cpu['source_hashes'].items():assert sha(s/n)==h,n
    else:v=config(root)
    cfg=binding(root,v,'R')
    lock=(control/'RECOVER.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGHUP,signal.SIG_IGN);torch.set_num_threads(1);os.nice(19)
    jobs=sorted((root/'jobs').glob('replay0735_warm_u004_R_*'))
    bank=root/'replay_bank/R';published=[]
    for job in jobs:
        while True:
            from .warm_health import assert_running
            assert_running(root,'R')
            if time.time()>v['gpu_deadline']:raise ValueError('registered recovery deadline')
            scope={p.stem for p in (bank/'snapshots').glob('*.json')}
            pending=sum(p.stem in scope and not (root/'centre_responses'/p.name).exists()
                        for p in (root/'centre_requests').glob('*.json'))
            if pending<8:break
            time.sleep(10)
        value=prepare(root,job,cfg)
        if value is None:continue
        snapshot,request,meta=value;key=request['state_id']
        (bank/'snapshots').mkdir(parents=True,exist_ok=True)
        path=bank/'snapshots'/(key+'.pt')
        assert not path.exists(),'adopt prior one-use recovery; never republish'
        torch.save(snapshot,path);write(path.with_suffix('.json'),meta)
        write(root/'centre_requests'/(key+'.json'),request)
        published.append(key)
        atomic(control/'STATUS.json',dict(unix=time.time(),published=published,new_episode_sim_s=0,
            arm='R',policy_update=3,source='saved R-u4 own-policy tapes',new_branch_phase='sumo_verification'))
    write(control/'COMPLETE.json',dict(unix=time.time(),published=published,new_episode_sim_s=0))


if __name__=='__main__':
    import sys
    run(sys.argv[1],sys.argv[2])
