"""Frozen-u3, nonblocking TRAIN snapshots and exact labels; never an optimizer."""
import copy
import gzip
import io
import json
import os
from pathlib import Path
import time
from .protocol import digest
from .warm_spool import immutable


def compressed_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    tmp=path.with_name(path.name+'.publishing')
    with tmp.open('xb') as f:
        f.write(gzip.compress(data,compresslevel=3,mtime=0));f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def install(controller,job,prefixes,generator):
    """Observe all eligible windows, publish compressed snapshots, always act on policy."""
    from .warm_history import ObservableHistory
    from .mppi_search import group_components
    from .sumo_verify import capture
    from .exploration import regime
    import torch
    root=Path(job['ls_collect']['root']);history=ObservableHistory()
    def call(rows,commands,now,sample):
        if int(now)%10 or now>900:return None
        groups=group_components(rows)
        if not groups:return None
        assert not controller.active_plans,'collection must execute no guide plans'
        key=f'{job["id"]}_t{int(now):04}'
        snapshot=capture(controller,prefixes,job,now,sample,generator)
        # Prefix and past samples are shared in the immutable completed episode;
        # defer their materialization until labeling instead of quadratically
        # duplicating every prefix at every window/group.
        snapshot.pop('prefixes');snapshot.pop('samples')
        snapshot['current_sample']=copy.deepcopy(sample)
        snapshot['collection_episode_path']=job['path']
        buffer=io.BytesIO();torch.save(snapshot,buffer)
        d=root/'snapshots';d.mkdir(parents=True,exist_ok=True)
        with (d/(key+'.pt.gz')).open('xb') as f:
            f.write(gzip.compress(buffer.getvalue(),compresslevel=3,mtime=0))
        common=dict(world_id=job['world']['seed'],time_s=int(now),rows=rows,commands=commands,
            policy_residuals={v:float(a[0]) for v,a in zip(sample['context']['members'],sample['own_action'])},
            history=history.snapshot(now),policy_state=job['policy_state'])
        if job.get('search_learning') is not None:common['search_learning']=job['search_learning']
        for i,g in enumerate(groups):
            record=dict(common,group=g['members']);record['state_id']=digest(dict(record=record,job=job['id']))
            envelope=dict(record=record,snapshot=key,episode_path=job['path'],group_index=i,
                regime=regime(rows),world_id=job['world']['seed'],time_s=int(now))
            compressed_json(root/'pending'/(record['state_id']+'.json.gz'),envelope)
        sample['behaviour']=dict(route='policy',executed=False,sample_probability=0.,q=0.,
            collection_snapshots=len(groups),nonblocking_collection=True)
        return None
    call.observe=lambda rows,now:history.push(now,rows)
    def prime(frames):
        for f in frames:history.push(f['time_s'],f['rows'],f['entrance_counts'])
    call.prime=prime
    controller.guide=call


def complete_snapshot(root,envelope):
    import torch
    from .accel_storage import read
    with gzip.open(Path(root)/'snapshots'/(envelope['snapshot']+'.pt.gz'),'rb') as f:
        s=torch.load(f,map_location='cpu',weights_only=False)
    ep=read(Path(envelope['episode_path'])/'RESULT.json');t0=s['t0']
    assert ep['id']==s['episode']['id'] and ep['world']==s['episode']['world']
    assert 986000<=ep['world']['seed']<=986071
    s['prefixes']={str(t):ep['prefixes'][str(t)] for t in range(300,t0+1)}
    s['samples']={str(x['time_s']):x['sample'] for x in ep['steps'] if x['time_s']<t0}
    s['samples'][str(t0)]=s.pop('current_sample')
    assert not any(x['sample'].get('behaviour',{}).get('executed') for x in ep['steps'])
    return s


def envelope_order(envelope):
    # All worlds' earliest group 0 before later groups/windows. No outcome score.
    return envelope['time_s'],envelope['group_index'],envelope['world_id'],envelope['record']['state_id']


def queue_metadata(path):
    """Keep only the index in RAM, never thousands of full histories/policies."""
    e=json.loads(gzip.decompress(Path(path).read_bytes()))
    return dict(path=str(path),world_id=e['world_id'],time_s=e['time_s'],
        group_index=e['group_index'],episode_path=e['episode_path'],key=e['record']['state_id'])


def coverage_order(e,counts):
    # New worlds first. Then middle/late states, with no outcome-based ordering.
    n=counts.get(e['world_id'],0);target=(300,600,900)[min(n,2)]
    return n,abs(e['time_s']-target),e['group_index'],e['world_id'],e['key']


def focus_order(e,counts,worlds,fraction):
    """Registered focus allocation, then unchanged coverage within each stratum."""
    focus={w['seed'] for w in worlds if w['demand'] in (2400,2700) and
        w['penetration'] in (.2,.4)}
    assert 0<=fraction<=1 and focus
    total=sum(counts.values());allocated=sum(n for w,n in counts.items() if w in focus)
    prefer_focus=allocated<fraction*(total+1)
    return (int((e['world_id'] in focus)!=prefer_focus),)+coverage_order(e,counts)


def label(root,envelope,response,config):
    from .sumo_verify import verify,SubprocessBranchRunner
    key=envelope['record']['state_id'];root=Path(root)
    assigned=root/'requests'/(key+'.json')
    if assigned.exists():
        record=json.loads(assigned.read_text())
        if 'search_config' in record:
            plain=dict(record);plain.pop('search_config')
            assert plain==envelope['record'],'niche metadata changed observable snapshot'
            assert response['record_sha256']==digest(record)
            envelope=dict(envelope,record=record)
    candidates=response.get('collection_top8')
    if not candidates or len(candidates)!=8:
        raise ValueError('collection search lacks the eight original final candidate profiles')
    feasibility=None
    if config.get('nullfix_active'):
        from .ls_feasible import shortlist,skipped
        candidates,feasibility=shortlist(envelope['record'],candidates,config['accel_bounds'])
        if not candidates:
            result=skipped(envelope['record'],envelope,feasibility)
            generation=config.get('label_generation')
            target=root/'labels'/(key+('_'+generation if generation else '')+'.json')
            target.parent.mkdir(parents=True,exist_ok=True)
            extra={k:response[k] for k in ('variation','search_config','nullfix') if k in response}
            immutable(target,dict(search_id=key,world_id=envelope['world_id'],
                record_sha256=digest(envelope['record']),verification=result,
                feasibility=feasibility,policy_frozen_u3=True,
                plans_executed_during_collection=False,**extra))
            return result
    snapshot=complete_snapshot(root,envelope)
    generation=config.get('label_generation')
    attempt=root/'label_attempts'/generation if generation else root
    (attempt/'label_charges').mkdir(parents=True,exist_ok=True)
    (root/'label_voids').mkdir(exist_ok=True)
    target=root/'labels'/(key+('_'+generation if generation else '')+'.json')
    if generation:
        from .ls_label_runtime import SmokeBranchRunner,grading_disk_ready
        runner=SmokeBranchRunner(attempt/'branches',workers=9,timeout_s=3600,
            smoke_path=root/('LABEL_SUMO_SMOKE_'+generation+'.json'))
        while not grading_disk_ready(root,generation):time.sleep(10)
    else:runner=SubprocessBranchRunner(root/'branches',workers=9,timeout_s=3600)
    def charge(vid,specs):
        # Complete worst-case nine-branch charge was made BEFORE the search.
        reservation=json.loads((root/'admissions'/(key+'.json')).read_text())
        assert len(specs)==len(candidates)+1 and sum(s['sim_s'] for s in specs)<=reservation['charged_sim_s']
        previous=root/'labels'/(key+'.json')
        if generation and previous.exists():
            from .ls_label_runtime import void_environment_failure
            from .accel_storage import reserve
            old=json.loads(previous.read_text());assert void_environment_failure(old),'not an authorized no-physics retry'
            immutable(root/'label_voids'/(key+'.json'),dict(original_label_sha256=digest(old),
                original_receipt_sha256=old['verification']['receipt_sha256'],actual_sim_s=0,
                no_outcome=True,charges_retained=True,reason='SUMO environment missing; MAIN23:32'))
            reserve(Path(config['run_root']),[dict(id='lslabel_'+key+'_'+generation,
                sim_s=sum(s['sim_s'] for s in specs))],'sumo_verification')
        immutable(attempt/'label_charges'/(key+'.json'),dict(verification_id=vid,
            reservation=reservation,branches=specs,no_double_charge=True))
    kw=dict(members_t0=feasibility['members_t0']) if feasibility else {}
    result=verify(snapshot,candidates,runner,charge,'wm',workers=9,root=attempt/'verification',**kw)
    extra={k:response[k] for k in ('variation','search_config','nullfix') if k in response}
    if feasibility is not None:extra['feasibility']=feasibility
    immutable(target,dict(search_id=key,world_id=envelope['world_id'],
        record_sha256=digest(envelope['record']),verification=result,
        policy_frozen_u3=True,plans_executed_during_collection=False,**extra))
    return result


def episode_worker(job):
    import torch
    from .accel_plant import episode
    torch.set_num_threads(1)
    from .warm_protocol import install_accounting
    install_accounting(job['queue_root'],job['ls_collect']['budget'])
    try:return episode(job)['id']
    finally:
        p=Path(job['path'])/'CLOSE.json'
        if not p.exists():immutable(p,dict(unix=time.time(),charges_retained=True))


def supervise(root):
    """One durable owner, concurrent episode collection/search/label queues."""
    import fcntl
    import signal
    import concurrent.futures as futures
    import multiprocessing
    import torch
    from .accel_storage import read,reserve,ledger
    from .warm_protocol import install_accounting
    root=Path(root);config=json.loads((root/'LS_COLLECT_BINDING.json').read_text())
    if (root/'LS_RECOVERY_CONFIG.json').exists():
        config.update(json.loads((root/'LS_RECOVERY_CONFIG.json').read_text()))
    if (root/'LS_VALIDITY_CONFIG.json').exists():
        config.update(json.loads((root/'LS_VALIDITY_CONFIG.json').read_text()))
    if (root/'LS_NICHE_CONFIG.json').exists():
        config.update(json.loads((root/'LS_NICHE_CONFIG.json').read_text()))
    if (root/'LS_LABEL_CONFIG.json').exists():
        config.update(json.loads((root/'LS_LABEL_CONFIG.json').read_text()))
    if (root/'LS_LABEL_DISK_CONFIG.json').exists():
        config.update(json.loads((root/'LS_LABEL_DISK_CONFIG.json').read_text()))
    if (root/'LS_FOCUS_CONFIG.json').exists():
        config.update(json.loads((root/'LS_FOCUS_CONFIG.json').read_text()))
    if (root/'LS_NULLFIX_CONFIG.json').exists():
        config.update(json.loads((root/'LS_NULLFIX_CONFIG.json').read_text()))
    if (root/'LS_REPLAY_DRAIN_CONFIG.json').exists():
        config.update(json.loads((root/'LS_REPLAY_DRAIN_CONFIG.json').read_text()))
    main=Path(config['run_root']);install_accounting(main,config['budget'])
    if config.get('operational_spend_cap'):
        from . import accel_storage
        from .accel_protocol import SUPERSEDED_CHARGE
        accel_storage._reserve.__globals__['CAP']=min(accel_storage._reserve.__globals__['CAP'],
            config['operational_spend_cap']-SUPERSEDED_CHARGE)
    if config.get('sumo_home'):
        from .ls_label_runtime import sumo_environment
        os.environ.update(sumo_environment(config['sumo_home']))
    with (root/'SUPERVISOR.claim').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGHUP,signal.SIG_IGN)
        immutable(root/('OWNER'+config.get('receipt_suffix','')+'.json'),dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time()))
        from .accel_learner import load
        from .policy_wm import actor_state
        policy,_,_,progress=load(main/'B_checkpoint_003.pt')
        assert progress['update']==3
        jobs=[]
        for w in config['worlds']:
            jid=f'lscollect1_B3_{w["seed"]}'
            jobs.append(dict(id=jid,path=str(main/'jobs'/jid),world=w,role='train',arm='B',update=3,
                training_seed=20261101,weights=policy.state_dict(),policy_state=actor_state(policy),
                runtime=config['runtime'],checkpoint=config['checkpoint'],noise_seed=20261101+w['seed'],
                deterministic=False,guided=False,health_guards=False,sim_s=1200,warm_history=True,
                accel_bounds=config['accel_bounds'],queue_root=str(main),ls_collect=dict(root=str(root),budget=config['budget'])))
        existing={c['id'] for c in ledger(main)}
        if not any(j['id'] in existing for j in jobs):reserve(main,jobs,'train')
        else:assert all(j['id'] in existing for j in jobs),'partial collection episode charge'
        for n in ('admissions','labels','label_charges','pending','requests','proposer_receipts','snapshots','label_failures'):(root/n).mkdir(exist_ok=True)
        for n in ('requests','responses','quota_admissions'):(main/n).mkdir(exist_ok=True)
        ctx=multiprocessing.get_context('spawn')
        with futures.ProcessPoolExecutor(max_workers=24,mp_context=ctx) as pool, futures.ThreadPoolExecutor(max_workers=2) as grades:
            episodes={}
            for j in jobs:
                p=Path(j['path'])
                if (p/'RESULT.json').exists():continue
                if (p/'CLAIM.json').exists():raise ValueError('interrupted collection episode: never replay claimed job')
                episodes[pool.submit(episode_worker,j)]=j['id']
            grading={};seen=set();pending={};started=time.time();last_status=0
            while True:
                if (root/'MAIN_STOP.json').exists():break
                for f,jid in episodes.items():
                    if f.done() and f.exception():raise f.exception()
                for p in (root/'pending').glob('*.json.gz'):
                    if p.stem in seen:continue
                    seen.add(p.stem)
                    e=queue_metadata(p);pending[e['key']]=e
                for key,f in list(grading.items()):
                    if f.done():
                        try:f.result()
                        except Exception as error:
                            import traceback
                            immutable(root/'label_failures'/(key+'.json'),dict(error=repr(error),
                                traceback=traceback.format_exc(),unix=time.time(),excluded=True,charges_retained=True))
                            print('ALERT label_failed '+key+' '+repr(error),flush=True)
                        del grading[key]
                admitted={p.stem for p in (root/'admissions').glob('*.json')}
                answered={k for k in admitted if (main/'responses'/(k+'.json')).exists()}
                # The GPU queue stays bounded. Episodes never consult this queue.
                inflight=admitted-answered
                if not config.get('drain_only') and len(admitted)<config['max_searches'] and len(inflight)<8:
                    counts={}
                    for k in admitted:
                        w=pending[k]['world_id'];counts[w]=counts.get(w,0)+1
                    from .ls_validity import feasible
                    todo=[e for k,e in pending.items() if k not in admitted and
                        (not config.get('arrival_admission') or feasible(e['time_s'],config['arrival_admission'])) and
                        (Path(e['episode_path'])/'RESULT.json').exists()]
                    for admission_offset in range(min(8-len(inflight),config['max_searches']-len(admitted),len(todo))):
                        order=(lambda e:focus_order(e,counts,config['worlds'],config['focus_fraction'])) if 'focus_fraction' in config else (lambda e:coverage_order(e,counts))
                        e=min(todo,key=order);todo.remove(e)
                        counts[e['world_id']]=counts.get(e['world_id'],0)+1
                        envelope=json.loads(gzip.decompress(Path(e['path']).read_bytes()))
                        r=copy.deepcopy(envelope['record'])
                        if config.get('niche_alternation'):
                            from .niche_collect import search_config
                            ordinal=len(admitted)+admission_offset-config['niche_alternation']['start_admission_count']
                            r['search_config']=search_config(ordinal)
                        key=r['state_id'];charge=9*min(1200,r['time_s']+420)
                        try:reserve(main,[dict(id='lslabel_'+key,sim_s=charge)],'sumo_verification')
                        except Exception as error:
                            immutable(root/'BUDGET_ALERT.json',dict(error=repr(error),key=key,unix=time.time(),
                                phase='sumo_verification',charged_sim_s=charge))
                            print('ALERT budget_blocked '+repr(error),flush=True);raise
                        immutable(root/'admissions'/(key+'.json'),dict(key=key,charged_sim_s=charge,unix=time.time()))
                        immutable(main/'quota_admissions'/(key+'.json'),dict(blocking=True,collection=True,unix=time.time()))
                        immutable(root/'requests'/(key+'.json'),r)
                        immutable(main/'requests'/(key+'.json'),r)
                for key in sorted(answered):
                    from .ls_label_runtime import label_path
                    if key in grading or label_path(root,key,config.get('label_generation')).exists() or (root/'label_failures'/(key+'.json')).exists():continue
                    response=read(main/'responses'/(key+'.json'))
                    if response.get('query_error'):
                        immutable(root/'label_failures'/(key+'.json'),dict(error=response['query_error'],
                            unix=time.time(),excluded=True,charges_retained=True))
                        print('ALERT query_code_declined '+key,flush=True);continue
                    receipt=response.get('proposer_receipt')
                    if receipt is None:raise ValueError('collection response has no full-pool receipt')
                    immutable(root/'proposer_receipts'/(key+'.json'),receipt)
                    limit=1 if config.get('label_generation') and not (root/('LABEL_SUMO_SMOKE_'+config['label_generation']+'.json')).exists() else 2
                    if len(grading)<limit:
                        envelope=json.loads(gzip.decompress(Path(pending[key]['path']).read_bytes()))
                        grading[key]=grades.submit(label,root,envelope,response,config)
                now=time.time()
                if now-last_status>=30:
                    from .ls_validity import exclusion_counts
                    labels=[json.loads(p.read_text()) for p in (root/'labels').glob('*.json')]
                    v=dict(unix=now,start_unix=started,episodes_completed=sum(f.done() for f in episodes)+len(jobs)-len(episodes),
                        snapshot_queries=len(pending),searches_requested=len(admitted),searches_finished=len(answered),
                        worlds=sorted({pending[k]['world_id'] for k in answered}),
                        admitted_worlds=sorted({pending[k]['world_id'] for k in admitted}),
                        episode_worlds=sorted(w['seed'] for w in config['worlds'] if (main/'jobs'/f'lscollect1_B3_{w["seed"]}'/'RESULT.json').exists()),
                         verified_lists=sum(x['verification']['status'] in ('accepted','rejected') for x in labels),
                         label_failures=len(list((root/'label_failures').glob('*.json'))),
                        verification_statuses=[x['verification']['status'] for x in labels],grading_inflight=len(grading),
                        charged_collection_sim_s=sum(c['sim_s'] for c in ledger(main) if c['id'].startswith(('lscollect1_','lslabel_'))),
                        budget_capacity_searches=config['max_searches'],training_paused=True,mirror_active=False,
                        arrival_admission=config.get('arrival_admission'),
                        excluded_by_time_bucket=exclusion_counts(pending,config['arrival_admission']) if config.get('arrival_admission') else {})
                    tmp=root/'LS_COLLECT_STATUS.publishing'
                    with tmp.open('w') as f:json.dump(v,f)
                    os.replace(tmp,root/'LS_COLLECT_STATUS.json');print(json.dumps(v),flush=True);last_status=now
                time.sleep(2)


if __name__=='__main__':
    import sys
    try:supervise(sys.argv[1])
    except BaseException as error:
        import traceback
        immutable(Path(sys.argv[1])/'LS_COLLECT_FAILED.json',dict(error=repr(error),traceback=traceback.format_exc(),unix=time.time()))
        raise
