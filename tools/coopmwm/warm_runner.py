"""Fresh warm namespace: preserve A/Adam/RNG; restart B/R once from MAPPOu0.

The early B/R endpoint is the registered u4 viability pause; fixed full-budget
endpoints remain A/B48,R24. No final-panel execution lives in this supervisor.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import numpy as np
import torch
from .accel_runner import timestamp,wave,routing,panel_report,diagnostic_task
from .accel_learner import Policy,load,save,update,evidence,viability,advantages,UNIT
from .accel_storage import read,write,charged
from .protocol import sha,digest
from .warm_protocol import budget,batch_worlds,worlds,SEED,install_accounting
from .exploration import trust
from . import search_learning
from .warm_health import assert_running,HealthPause,before_update,after_check,braking


def verify(root,arm):
    b=read(root/'WARM_BINDING.json')
    latest=root/'MAIN0224_EXACT_BUDGET.json'
    if latest.exists():
        amendment=read(latest)
        assert amendment['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
        b=dict(b,budget=amendment['budget'])
    assert b['budget']==budget()
    assert b['cpu']['pass_'] and not b['cpu']['skips'] and b['cpu']['sim_s']==0
    for name,h in b['hashes'].items():assert sha(root/name)==h,'immutable warm source '+name
    text=(root/'AUTHORITY.md').read_text()
    assert b['scope_amendment_sha256'] in text and b['release_amendment_sha256'] in text
    assert_running(root,arm)
    if arm in ('B','R'):
        release=read(root/'WARM_RELEASE.json')
        assert release['pass_'] and release['N_exec']>=10
        assert release['release_amendment_sha256'] in text
    v4=root/f'WARM_{arm}_RUNTIME_BINDING_V4.json'
    v3=v4 if v4.exists() else root/f'WARM_{arm}_RUNTIME_BINDING_V3.json'
    if arm in ('B','R') or v3.exists():
        runtime=read(v3 if v3.exists() else root/f'WARM_{arm}_RUNTIME_BINDING_V2.json')
        assert runtime['cpu']['pass_'] and not runtime['cpu']['skips']
        assert runtime['amendment_sha256'] in text
        source=Path(runtime['source_root'])
        assert Path(__file__).resolve()==source/'pytools/tools/coopmwm/warm_runner.py'
        for name,h in runtime['hashes'].items():assert sha(source/name)==h,arm+' runtime source '+name
        if v3.exists():
            awr=read(root/'AWR_BINDING.json');assert awr['amendment_sha256'] in text
            b=dict(b,**dict(runtime['operational'],awr=awr['config']))  # no duplicate-keyword TypeError
    install_accounting()
    return b


def wait(root,path,deadline,arm='B'):
    while not path.exists():
        assert_running(root,arm)
        if time.time()>deadline:raise ValueError('registered warm runtime deadline')
        time.sleep(2)
    return read(path)


def check(root,b,arm,policy,u):
    controls=read(root/'MAPPO_CHECK.json')['records']
    eps=wave(root,b,{arm:policy},worlds('check'),'check',f'warmcheck{u:03}_{arm}','checks')
    records=[{k:e[k] for k in ('id','world','arm','metrics')} for e in eps]
    write(root/f'{arm}_CHECK_RECORDS_{u:03}.json',records)
    panel_report(root,f'{arm}_CHECK_{u:03}.json',records,controls,'check')
    if arm=='B':
        a=wait(root,root/f'A_CHECK_RECORDS_{u:03}.json',b['gpu_deadline'])
        report=panel_report(root,f'AB_CHECK_{u:03}.json',a+records,controls,'check')
        if b.get('health_guards'):after_check(root,arm,u,report)


def random_reference(root,u,concurrent=False):
    """Pin latest completed B statistics once; never wait for B's current update."""
    if not concurrent:return root/f'B_RANDOM_QUOTA_{u:03}.json'
    receipt=root/f'R_MATCH_REFERENCE_{u:03}.json'
    if receipt.exists():return Path(read(receipt)['path'])
    choices=[p for p in root.glob('B_RANDOM_QUOTA_[0-9][0-9][0-9].json')
        if int(p.stem.rsplit('_',1)[1])<=u]
    if not choices:raise ValueError('no completed B quota for concurrent R')
    path=max(choices,key=lambda p:int(p.stem.rsplit('_',1)[1]))
    write(receipt,dict(path=str(path),B_reference_update=int(path.stem.rsplit('_',1)[1]),
        R_update=u,rule='MAIN1844 latest completed B; best-effort strata; no current-B wait'))
    return path


def run_arm(root,arm,hooks=None):
    import fcntl
    import os
    b=verify(root,arm)
    lock=(root/f'WARM_{arm}.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    suffix=b.get('runtime_receipt_suffix','_V3' if b.get('health_guards') else '')
    assert not (root/f'WARM_{arm}_STARTED{suffix}.json').exists(),'one-use warm arm already started'
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)));torch.set_num_threads(1)
    torch.manual_seed(SEED)
    if arm=='A':
        resume=b.get('A_resume_update',4)
        policy,opt,rng,payload=load(root/f'A_checkpoint_{resume:03}.pt');assert payload['update']==resume
        first,last=resume+1,b.get('A_stop_after_update',48)
    elif 'resume_update' in b:
        resume=b['resume_update']
        policy,opt,rng,payload=load(root/f'{arm}_checkpoint_{resume:03}.pt');assert payload['update']==resume
        first,last=resume+1,min(budget()['updates'][arm],b.get('stop_after_update',budget()['updates'][arm]))
    else:
        policy=Policy();opt=torch.optim.Adam(policy.parameters(),lr=3e-4)
        rng=torch.Generator().manual_seed(SEED);first,last=1,budget()['updates'][arm]
        save(root/f'{arm}_checkpoint_000.pt',policy,opt,rng,0,charged(root))
    write(root/f'WARM_{arm}_STARTED{suffix}.json',dict(timestamp_utc=timestamp(),first_update=first,
        pause_update=last,full_endpoint=budget()['updates'][arm],charged_sim_s=charged(root)))
    running=[];beta=1.;learned=search_learning.empty(arm,SEED,b['accel_bounds'])
    exact_rng=np.random.default_rng(SEED);exact_count=0
    if b.get('resume_progress'):
        progress=b['resume_progress'];running=progress['running'];beta=progress['beta']
        if arm=='B':learned=progress['learned']
        exact_rng.bit_generator.state=progress['exact_rng_state'];exact_count=progress['exact_count']
    diagnostics=ThreadPoolExecutor(max_workers=1)
    try:
        if arm=='A' and b.get('health_guards') and not b.get('skip_legacy_reporting'):
            # Finish any due reporting interrupted after the durable u16 checkpoint.
            if not (root/'A_CHECK_016.json').exists():check(root,b,arm,policy,16)
            if not (root/'A_DEV_016.json').exists():
                dev=wave(root,b,{arm:policy},worlds('dev'),'dev','warmdev016_A','dev')
                panel_report(root,'A_DEV_016.json',dev,read(root/'MAPPO_DEV.json')['records'],'dev')
        for u in range(first,last+1):
            assert_running(root,arm)
            start=time.monotonic();start_unix=time.time();extra=dict(update=u)
            if b.get('sumo_verify'):extra['sumo_verify']=b['sumo_verify']
            if b.get('quota_aware_admission'):extra['quota_aware_admission']=True
            if b.get('parallel_group_queries') and arm=='B':extra['parallel_group_queries']=True
            if b.get('bounded_fanout') and arm=='B':extra['bounded_fanout']=True
            # HARDEN-3 opt-in (MAIN decision): cap new in-flight group queries per window.
            if b.get('parallel_fanout_factor') and arm=='B':extra['parallel_fanout_factor']=b['parallel_fanout_factor']
            if b.get('health_guards'):extra.update(health_guards=True,safety_reference=b['safety_reference'])
            if arm=='B':extra['search_learning']=learned
            replay=bool(b.get('b_replay')) and arm in ('B','R')
            if replay:extra['b_replay']=True  # MAIN B-REPLAY: plain PPO episodes, bank imitation only
            # HARDEN-3 opt-in (MAIN decision): R update u may match B's PREVIOUS
            # completed update (lag 1) so R runs concurrently with B update u.
            quota_u=u-1 if arm=='R' and b.get('r_concurrent_lag1_quota') else u
            if arm=='R' and not replay:
                reference=random_reference(root,u,b.get('concurrent_R',False)) if b.get('concurrent_R') else root/f'B_RANDOM_QUOTA_{quota_u:03}.json'
                wait(root,reference,b['gpu_deadline'],arm)
                extra['random_quota']=str(reference)
            eps=wave(root,dict(b,beta=beta),{arm:policy},batch_worlds(u),'train',f'warm_u{u:03}','train',False,extra)
            collection_begin=(min([start_unix]+[(root/'jobs'/e['id']/'CLAIM.json').stat().st_mtime for e in eps])
                if b.get('parallel_group_queries') else start_unix)
            routes=routing(eps)
            deviation=None
            if arm=='B' and not replay and routes['executed']<b['N_exec']:
                from .quota_admission import first_update_deviation
                deviation=first_update_deviation(root,arm,u,routes['executed'],b['N_exec'])
            matching=None
            if arm=='R' and not replay:
                from .r_matching import report
                quota=read(Path(extra['random_quota']))
                matching=report(quota['tickets'],routes['durations'])
                matching['reference_quota']=extra['random_quota']
                if quota_u!=u:matching=dict(matching,matched_B_update=quota_u,rule='lag-1 concurrent R (opt-in)')
            ev=evidence(eps);running+=ev
            exact_receipts=[];weights=None;pairs=[]
            if replay:
                from .replay_bank import bank_pairs,imitation_decision_cap
                replay_beta=float(b.get('replay_beta',1.))
                cap=fair=None
                if arm=='R':
                    # MAIN fairness: R imitates at most as many admitted targets as B's bank
                    # had at the matching B update (lag 1 when R runs concurrently).
                    bu=u-1 if b.get('r_concurrent_lag1_quota') else u
                    bp=root/f'B_UPDATE_{bu:03}.json'
                    if bu==u:wait(root,bp,b['gpu_deadline'],arm)
                    cap=imitation_decision_cap(read(bp)) if bp.exists() else 0
                    fair=dict(matched_B_update=bu,B_bank_decisions=cap,B_update_receipt_found=bp.exists())
                pairs,weights=bank_pairs(root,arm,SEED,beta=replay_beta,sources=b.get('replay_sources',{}).get(arm,()),
                    max_decisions=cap,v2v=bool(b.get('v2v')),pair_flags=bool(b.get('imitation_trust')))
                from .replay_bank import bank_quality
                weights['bank_quality']=bank_quality(root,arm,b.get('replay_sources',{}).get(arm,()))
                if fair is not None:
                    have=weights['counts']['available_admitted'];fair.update(R_available=have,R_used=weights['counts']['used_admitted'],
                        shortfall=max(0,cap-have),excess_not_used=max(0,have-cap),padding_from_B=False)
                    weights['fairness']=fair
                tr=dict(beta=replay_beta,graded=weights['counts']['positive'],bank=weights['counts'],
                    mode='B-REPLAY: no executed plans; trust/beta fixed by binding')
            elif arm in ('B','R'):
                from . import exact_runtime as xr
                for ep in eps:xr.enqueue(root,b,ep,u,arm,beta)
                exact_receipts,weights,pairs,tr=xr.evidence(root,arm,beta)
            else:tr=dict(beta=0.)
            if b.get('health_guards'):
                before_update(root,arm,u,routes,
                    [dict(executed=True)]*tr['graded'] if arm in ('B','R') else running,tr,0 if replay else b['N_exec'],
                    effective_guide_beta=(b.get('sumo_verify') or {}).get('guide_beta'))
                if arm in ('B','R'):
                    a_records=[]
                    for w in batch_worlds(u):
                        ident=f'warm_u{u:03}_A_{w["seed"]}'
                        p=root/'jobs'/ident/'RESULT.json'
                        if p.exists():a_records.append(read(p))
                    braking(root,arm,u,eps,a_records)
            if arm=='B':
                learned=search_learning.exact_update(learned,exact_receipts,lambda p:read(Path(p)),arm,SEED)
                write(root/f'B_SEARCH_LEARNING_{u:03}.json',learned)
            write(root/f'{arm}_VIABILITY_{u:03}.json',dict(timestamp_utc=timestamp(),update=u,
                arm=arm,critic_diagnostic_only=viability(ev),cumulative_critic_diagnostic_only=viability(running),
                exact_trust=tr,routing=routes))
            # Snapshot the rollout policy BEFORE the optimizer for exact twins.
            state={k:v.detach().clone() for k,v in policy.state_dict().items()}
            if arm in ('B','R'):
                from .exact_grade import exact_update,lag_report
                imitation_trust=b.get('imitation_trust') if replay else None
                metrics=exact_update(policy,opt,eps,rng,pairs,b['awr']['coefficient'],
                    imitation_epochs=b.get('imitation_epochs',5) if replay else 0,
                    imitation_trust=imitation_trust,
                    cooperative=weights.get('pair_cooperative') if imitation_trust else None)
                grade_lag=None if replay else lag_report(xr.queue(root),u)
            else:
                metrics=update(policy,opt,eps,rng,**({'awr':b['awr']} if 'awr' in b else {}));grade_lag=None
            save(root/f'{arm}_checkpoint_{u:03}.pt',policy,opt,rng,u,charged(root))
            lesson=None
            if replay:
                # MAIN 2026-11-03 01:22: LESSON-LEARNED CHECK after every B/R update in B-REPLAY.
                # Never fatal for the run; charges bank_verification_ext (lesson_u<k>-prefixed).
                from .replay_bank import record_lesson
                cfg=b.get('sumo_verify') or {}
                def lesson_runner(work):
                    from .sumo_verify import SubprocessBranchRunner
                    return SubprocessBranchRunner(Path(work),workers=cfg.get('workers'),
                        timeout_s=cfg.get('timeout_s',3600))
                try:
                    lesson=record_lesson(root,arm,policy,root,u,lesson_runner,
                        sources=b.get('replay_sources',{}).get(arm,()),v2v=bool(b.get('v2v')))
                except Exception as error:
                    lesson=dict(arm=arm,update=u,error=repr(error),list_size=0,entries=[])
            if hooks is not None:
                # New V2V roots only: component-owned lesson/synergy integration.
                # Default callers never enter this path or invoke a new callback.
                lesson=hooks['diagnostics'](root,arm,u,policy,b,lesson)
            raw=[v*UNIT for e in eps for v,_ in advantages(e['steps'])]
            gpu_busy=None
            if b.get('parallel_group_queries') and arm=='B':
                from .tail_runtime import busy_report
                from .plan_client import worker_slots
                gpu_busy=busy_report(root,u,collection_begin,time.time(),[w['slot'] for w in worker_slots(root)])
            elapsed=time.time()-collection_begin if b.get('parallel_group_queries') else time.monotonic()-start
            write(root/f'{arm}_UPDATE_{u:03}.json',dict(timestamp_utc=timestamp(),update=u,arm=arm,
                training_seed=SEED,ppo=metrics,trust=tr,beta_used=beta if arm!='A' else 0.,
                effective_guide_beta=(b.get('sumo_verify') or {}).get('guide_beta'),
                proposal_routing=routes,team_only_noise=dict(advantage_variance_veh_s_squared=float(np.var(raw,ddof=1))),
                wall_s=elapsed,eta_final_hours=(budget()['updates'][arm]-u)*elapsed/3600,
                gpu_busy=gpu_busy,
                 exact_imitation=weights,grading_lag=grade_lag,
                 bank_imitation=next((m['bank_imitation'] for m in metrics if 'bank_imitation' in m),None),
                quota_deviation=deviation,random_matching=matching,
                concurrently_active_plans=max(s.get('simultaneously_active_plans',0) for e in eps for s in e['steps']),
                overlap_rechecks=[s['proposal']['overlap_recheck'] for e in eps for s in e['steps']
                    if s.get('proposal') and s['proposal'].get('overlap_recheck')],
                charged_sim_s=charged(root),reward_source='SUMO source-inclusive team J only'))
            beta=tr
            if arm=='B':
                write(root/f'B_RANDOM_QUOTA_{u:03}.json',dict(sampling=routes['sampling'],
                    tickets=[dict(t,id=i) for i,t in enumerate(routes['durations'])]))
                candidates=[(e,s) for e in eps for s in e['steps'] if s['time_s']<=600 and
                    s['sample'].get('behaviour',{}).get('plan_start')]
                for i in exact_rng.permutation(len(candidates))[:max(0,12-exact_count)]:
                    e,s=candidates[int(i)]
                    diagnostics.submit(diagnostic_task,root,b,state,e,s,exact_count);exact_count+=1
            if u%4==0 and not replay and not b.get('skip_legacy_reporting'):check(root,b,arm,policy,u)
            if u%16==0 and not b.get('skip_legacy_reporting'):
                dev=wave(root,b,{arm:policy},worlds('dev'),'dev',f'warmdev{u:03}_{arm}','dev')
                panel_report(root,f'{arm}_DEV_{u:03}.json',dev,read(root/'MAPPO_DEV.json')['records'],'dev')
        diagnostics.shutdown(wait=True)
        write(root/f'WARM_{arm}_PAUSED.json',dict(timestamp_utc=timestamp(),update=last,
            full_endpoint=budget()['updates'][arm],charged_sim_s=charged(root),
            reason='registered checkpoint boundary complete; final evaluation waits for fixed primary endpoints'))
    except BaseException as error:
        write(root/f'WARM_{arm}_FAILED{suffix}.json',dict(timestamp_utc=timestamp(),error=repr(error)))
        if not isinstance(error,HealthPause):
            guided_stop=('WARM_GUIDED_STOP_V5.json' if (root/'DNS_GUIDED_RELEASE_1651.json').exists() else
                'WARM_GUIDED_STOP_V4.json' if (root/'MATCH_GUIDED_RELEASE.json').exists() else
                'WARM_GUIDED_STOP_V3.json' if (root/'INCIDENT_GUIDED_RELEASE.json').exists() else
                'WARM_GUIDED_STOP_V2.json' if (root/'MAIN0508_GUIDED_RELEASE.json').exists() else 'WARM_GUIDED_STOP.json')
            stop=root/(guided_stop if arm in ('B','R') and b.get('health_guards') else 'STOP.json')
            if not stop.exists():write(stop,dict(arm=arm,error=repr(error),timestamp_utc=timestamp()))
        raise
    finally:
        diagnostics.shutdown(wait=True);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--arm',choices=('A','B','R'),required=True)
    a=p.parse_args();run_arm(a.root,a.arm)
