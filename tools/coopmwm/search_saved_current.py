"""Restart current-policy search input from already-charged own-policy episodes.

No SUMO replay: regenerate action/RNG exactly, recover 60 observable seconds,
then feed the existing four-slot broker and exact bank grader. No second broker.
"""
import copy
import gzip
import json
import os
from pathlib import Path
import time


def prepare(root,job_path,binding):
    import torch
    from types import SimpleNamespace
    from tools.coopmwm.accel_storage import read
    from tools.coopmwm.accel_learner import load
    from tools.coopmwm.warm_protocol import SEED
    from tools.coopmwm.warm_history import ObservableHistory,FIELDS
    from tools.coopmwm.r_bank_recover import regenerate
    from tools.coopmwm.mppi_search import group_components,final_horizon,REGISTERED
    from tools.coopmwm.policy_wm import actor_state
    from tools.coopmwm.protocol import digest
    from tools.coopmwm.exploration import regime
    from tools.coopmwm.sumo_verify import capture
    result=read(job_path/'RESULT.json');assert result['arm']=='B'
    assert 986000<=result['world']['seed']<=986071
    wave=result['id'].rsplit('_B_',1)[0]
    update=int(wave.rsplit('u',1)[1]);policy_update=update-1
    assert policy_update>=16,'current-policy collection only'
    policy,_,_,payload=load(root/f'B_checkpoint_{policy_update:03}.pt')
    assert payload['update']==policy_update
    noise_seed=int(digest(dict(seed=SEED,world=result['world']['seed'],wave=wave))[:15],16)
    history=ObservableHistory()
    with gzip.open(job_path/'observable.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);t=int(row['time_s'])
            rows=[{k:v[k] for k in FIELDS} for v in row['vehicles']];history.push(t,rows)
            if t<360 or t%10 or t+final_horizon(REGISTERED['max_duration_s'])>binding['demand_phase_end_s']:continue
            groups=group_components(rows)
            # Owner18:29 ordering: >=3 first, inherited downstream/near-merge
            # order within that class. No state exclusion or behavioural cutoff.
            groups.sort(key=lambda g:len(g['members'])<3)
            if not groups:continue
            commands={a['vid']:a['mappo_raw'] for a in row['actions']}
            if any(v['is_cav'] and v['vid'] not in commands for v in rows):continue
            generator,memory,events,sample=regenerate(policy,result,noise_seed,t)
            job=dict(id=result['id'],world=result['world'],arm='B',update=update,role='train',
                training_seed=SEED,runtime=binding['runtime'],checkpoint=binding['checkpoint'],
                warm_history=True,accel_bounds=binding['accel_bounds'],noise_seed=noise_seed)
            controller=SimpleNamespace(policy=policy,events=events,memory=memory,active_plans=[])
            snapshot=capture(controller,result['prefixes'],job,t,sample,generator)
            record=dict(world_id=result['world']['seed'],time_s=t,rows=rows,group=groups[0]['members'],
                commands=commands,policy_residuals={v:float(a[0]) for v,a in
                    zip(sample['context']['members'],sample['own_action'])},history=history.snapshot(t),
                policy_state=actor_state(policy))
            meta=dict(arm='B',job=result['id'],world=result['world'],t0=t,group=groups[0],
                regime=regime(rows),policy_update=policy_update,unix=time.time(),
                recovered_from_saved_own_episode=True,RNG_actions_bitwise=True,
                new_episode_sim_s=0,source_wave=wave)
            return snapshot,record,meta
    return None


def run(root,control,preflight=False):
    import torch,fcntl
    from tools.coopmwm.storage import read,write
    from tools.coopmwm.protocol import digest,sha
    from tools.coopmwm.replay_start import owner,binding
    from tools.coopmwm.warm_health import assert_running
    from tools.coopmwm.niche_collect import search_config
    root=Path(root);control=Path(control)
    authority=read(control/('CPU_BINDING.json' if preflight else 'AUTHORITY.json'))
    assert sha(__file__)==authority['source_sha256']
    cfg=binding(root,read(root/'B_REPLAY_EXPOSURE_V2.json'),'B')
    torch.set_num_threads(1);os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    bank=root/'replay_bank/B';published=[];worlds=set()
    if not preflight:
        lock=(control/'RUN.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (control/'OWNER.json').exists(),'adopt existing search restart owner'
        write(control/'OWNER.json',owner())
    candidates=sorted((root/'jobs').glob('replay0735_warm_u0[12][0-9]_B_*/RESULT.json'),reverse=True)
    candidates=[p for p in candidates if int(p.parent.name.split('_u')[1][:3])>=17
        and (p.parent/'observable.jsonl.gz').exists()]
    # World metadata only: focus first; distinct worlds before repeats.
    def focus(path):
        from tools.coopmwm.accel_storage import read as read_episode
        result=read_episode(path);w=result['world']
        return w['demand']>=2400 and w['penetration']>=.2
    candidates.sort(key=lambda p:not focus(p))
    for p in candidates:
        if len(published)>=authority['max_queries']:break
        seed=int(p.parent.name.rsplit('_',1)[1])
        if seed in worlds:continue
        assert_running(root,'B')
        value=prepare(root,p.parent,cfg)
        if value is None:continue
        snapshot,record,meta=value
        if preflight:
            return dict(pass_=True,meta=meta,history_frames=len(record['history']),members=len(record['group']),new_sim_s=0)
        # Same policy snapshot can be queried in a new explicit recovery namespace;
        # no prior request is replayed. New t>=360 differs from live t300 captures.
        if any(read(q).get('job')==meta['job'] and read(q).get('t0')==meta['t0'] and
               read(q).get('group',{}).get('members')==record['group']
               for q in (bank/'snapshots').glob('*.json')):continue
        while True:
            scope={q.stem for q in (bank/'snapshots').glob('*.json')}
            pending=sum(q.stem in scope and not (root/'responses'/q.name).exists() for q in (root/'requests').glob('*.json'))
            if pending<8:break
            assert_running(root,'B');time.sleep(10)
        admissions=bank/'collection_admissions';admissions.mkdir(exist_ok=True)
        with (bank/'COLLECTION_ADMISSION.claim').open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX)
            ordinal=len(list(admissions.glob('*.json')));record['search_config']=search_config(ordinal)
            write(admissions/f'{ordinal:09}.json',dict(ordinal=ordinal,job=meta['job'],time_s=meta['t0'],
                group=record['group'],search_config=record['search_config'],source='MAIN18:29 saved current-policy recovery'))
        record['state_id']=digest(dict(record=record,job=meta['job'],recovery='MAIN1829'))
        key=record['state_id'];meta.update(state_id=key,record_sha256=digest(record))
        torch.save(snapshot,bank/'snapshots'/(key+'.pt'));write(bank/'snapshots'/(key+'.json'),meta)
        write(root/'quota_admissions'/(key+'.json'),dict(blocking=True,collection=True,unix=time.time()))
        write(root/'requests'/(key+'.json'),record)
        worlds.add(seed);published.append(key)
        with (control/'PROGRESS.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),key=key,meta=meta))+'\n')
    if not preflight:write(control/'COMPLETE.json',dict(unix=time.time(),published=published,new_episode_sim_s=0))
    return dict(published=published,new_sim_s=0)


if __name__=='__main__':
    import sys
    print(json.dumps(run(sys.argv[1],sys.argv[2],preflight='--preflight' in sys.argv)))
