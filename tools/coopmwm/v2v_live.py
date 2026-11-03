"""Qualified adapters for isolated V2V roots sharing the authoritative ledger.

Bank input subdirectories are read-only views of the ongoing same-arm bank;
lesson, safety, optimizer and diagnostic outputs belong to the new run root.
"""
from pathlib import Path
import time
import types


def accounting(root, cfg):
    from . import accel_storage
    from .accel_protocol import SUPERSEDED_CHARGE
    from tools.coopnative import storage
    source=Path(cfg['accounting_root'])
    for name in ('ACCOUNTING.claim','CHARGES.jsonl'):
        if (Path(root)/name).resolve() != (source/name).resolve():
            raise ValueError('V2V must use the existing ledger AND its lock')
    budget=cfg['budget']
    if budget['total'] != 17471840 or sum(budget['phases'].values())>budget['total']:
        raise ValueError('V2V registered cap/allocation differs')
    accel_storage._reserve=types.FunctionType(storage.reserve.__code__,
        dict(storage.reserve.__globals__,CAP=budget['total']-SUPERSEDED_CHARGE,CAPS=budget['phases']))


def hooks(root, cfg):
    from . import replay_bank as rb, warm_health as health
    from . import bank_imitation, bank_statistics
    bank_statistics.install(rb)
    bank_imitation.run=bank_imitation.run_separate
    from .v2v_runtime import Policy, load_checkpoint
    from .accel_storage import read
    original_pairs,original_quality,original_lesson=rb.bank_pairs,rb.bank_quality,rb.record_lesson

    def configure(r,arm,binding):
        if Path(r)!=Path(root) or not binding['v2v'] or binding.get('sumo_verify'):
            raise ValueError('V2V own-policy episodes only')

    def safety(r,arm,u,episodes,ignored):
        if arm=='A':return
        reference=Path(r)/f'A_UPDATE_{u:03}.json'
        while not reference.exists():
            health.assert_running(r,arm)
            if time.time()>cfg['gpu_deadline']:raise ValueError('registered resource deadline')
            time.sleep(5)
        refs=[]
        for e in episodes:
            p=Path(r)/'jobs'/f'v2v1_warm_u{u:03}_A_{e["world"]["seed"]}'/'RESULT.json'
            if not p.exists():raise ValueError('missing same-world V2V A safety reference')
            refs.append(read(p))
        health.braking(r,arm,u,episodes,refs)

    return dict(configure=configure,accounting=accounting,policy_factory=Policy,
        load_checkpoint=load_checkpoint,bank_pairs=original_pairs,bank_quality=original_quality,
        record_lesson=original_lesson,health_before_update=health.before_update,safety=safety,
        diagnostics=lambda r,a,u,p,b,l:diagnostics(r,a,u,p,b,l))


def diagnostics(root, arm, update, policy, cfg, lesson):
    if update not in cfg['diagnostic_schedule'][arm]['isolated']:return lesson
    isolated(root,arm,update,policy,cfg)
    return lesson


def isolated(root, arm, update, policy, cfg):
    """Group/solos/background and a fresh legacy-u3 reference, fully precharged."""
    import torch
    from . import replay_bank as rb, policy_synergy as ps, sumo_verify as sv
    from .exact_grade import _seal
    from .storage import read,write
    from .warm_health import assert_running
    root=Path(root);base=root/'isolated'/f'u{update:03}_{arm}'
    output=root/f'POLICY_SYNERGY_u{update:03}_{arm}_V2V_ISOLATED.json'
    if output.exists():return read(output)
    base.mkdir(parents=True,exist_ok=True)
    if (base/'CLAIM.json').exists():raise ValueError('interrupted V2V diagnostic claim; never replay')
    confirmed=rb.confirmations(root,arm)
    decisions=[r for r in rb.bank_decisions(root,arm,sources=cfg['replay_sources'].get(arm,()))
        if confirmed.get(r['verification_id'],{}).get('admitted')
        and confirmed[r['verification_id']].get('cooperation',{}).get('cooperative') is True]
    links={read(p)['verification_id']:read(p) for p in (rb.bank_root(root,arm)/'decisions').glob('*.json')}
    old=torch.load(Path(cfg['legacy_root'])/f'{arm}_checkpoint_003.pt',weights_only=False)['policy']
    new={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    write(base/'CLAIM.json',dict(update=update,arm=arm,verification_ids=[r['verification_id'] for r in decisions],
        reference='original non-V2V arm u3; not this V2V arm u3',phase='v2v_isolated'))
    rows=[]
    for receipt in decisions:
        vid=receipt['verification_id']
        snapshot=rb._decision_snapshot(root,arm,receipt,links)
        pairs=ps.specs_for(snapshot,receipt,update,arm,new,old,isolated=True)
        reference=ps.specs_for(snapshot,receipt,update,arm,old,old)[0]
        pairs.append(reference)
        renamed=[]
        for i,(spec,snap) in enumerate(pairs):
            spec=dict(spec);spec.pop('branch_sha256')
            spec.update(verification_id=f'v2viso_u{update:03}_{arm}_{vid}_{i}',branch='reference' if i==len(pairs)-1 else spec['branch'])
            renamed.append((_seal(spec,'branch_sha256'),snap))
        runner=ps.BranchRunner(base/'branches',workers=1)
        blocked=runner.admit([s for s,_ in renamed])
        if blocked:raise RuntimeError(blocked)
        rb._confirm_charge(root,'v2v_isolated',f'v2v_isolated_u{update:03}_{arm}')(vid,[s for s,_ in renamed])
        results=[]
        from .bank_v2v import enabled
        for spec,snap in renamed:
            assert_running(root,arm)
            runner.prepare([spec],snap)
            with enabled(True):results.append(sv.run_branch(spec,snap,runner))
        ref=results.pop();background=results.pop()
        if ref['status']=='graded':
            measures=ps.metrics(ref['J_veh_s'],results,receipt['chosen_plan']['members'])
            measures['background_only_delta_veh_s']=(ref['J_veh_s']-background['J_veh_s']
                if background['status']=='graded' else None)
        else:measures=dict(status='incomplete',policy_cooperation_gain_veh_s=None)
        if background['status']!='graded':measures['status']='incomplete'
        row=dict(verification_id=vid,group=receipt['chosen_plan']['members'],results=results,
            reference=ref,background=background,window=receipt['window'],
            plan_cooperation_gain_veh_s=confirmed[vid]['cooperation']['cooperation_gain_veh_s'],**measures)
        write(base/(vid+'.json'),row);rows.append(row)
    result=_seal(dict(arm=arm,update=update,entries=rows,
        status='complete' if all(r['status']=='graded' for r in rows) else 'incomplete',
        phase='v2v_isolated',v2v=True,reference='legacy u3, identical snapshot/prefix/CRN',
        rule='group new vs each solo new; all other CAVs legacy u3; background-only separately'),
        'policy_synergy_sha256')
    write(output,result);return result


def run(root, cfg, release, arm):
    from .v2v_runner import run_arm
    run_arm(root,arm,cfg,release,hooks(root,cfg))


def resource_gate(root, cfg):
    import shutil
    from .warm_health import assert_running
    for arm in ('B','R'):assert_running(root,arm)
    free=shutil.disk_usage(root).free
    return dict(ready=free>=10*1024**3,disk_free_bytes=free,
        required_free_bytes=10*1024**3,rule='MAIN15:26/15:46 launch resource gate')


def configuration(root):
    from .storage import read
    from .protocol import sha
    root=Path(root);v=read(root/'V2V_RUNTIME.json')
    if v['amendment_sha256'] not in (root/'AUTHORITY.md').read_text():
        raise ValueError('V2V runtime authority missing')
    source=Path(v['config']['source_root']);cpu=read(source/'LS_CPU_RESULT.json')
    if sha(source/'LS_CPU_RESULT.json')!=v['cpu_result_sha256']:
        raise ValueError('V2V qualification receipt changed')
    if not cpu['pass_'] or cpu['sim_s'] or cpu['skips']:raise ValueError('V2V Linux release gate')
    # The qualification manifest contains two inherited generic assignment
    # implementation filenames. Do not reopen any assignment-named path at
    # runtime; their original metadata stays in the immutable CPU receipt.
    excluded=('pytools/traffic_shepherd/mixed_assignment.py',
              'runtime_src/traffic_shepherd/mixed_assignment.py')
    for name,h in cpu['source_hashes'].items():
        if name in excluded:continue
        if sha(source/name)!=h:raise ValueError('V2V immutable source mismatch: '+name)
    if Path(__file__).resolve()!=source/'pytools/tools/coopmwm/v2v_live.py':
        raise ValueError('V2V supervisor must run from qualified source')
    return v


def stage_arm(root, stage, arm):
    import copy
    from .storage import read,write
    root=Path(root);value=configuration(root);cfg=copy.deepcopy(value['config'])
    release=copy.deepcopy(value['release'])
    if stage=='smoke':
        cfg.update(qualification_stage='one_update_smoke',smoke_authority_sha256=value['amendment_sha256'],
            rehearsal_late_60min=True,stop_after_update=1,A_stop_after_update=1,
            runtime_receipt_suffix='_V2V_SMOKE')
    else:
        smoke=read(root/'V2V_REAL_SMOKES_PASS.json')
        if smoke['arms']!=['A','B','R'] or not smoke['pass_']:raise ValueError('all real smokes required')
        cfg.update(resume_update=1,A_resume_update=1,runtime_receipt_suffix='_V2V_LIVE',
            next_not_before_s=smoke['next_not_before_s'])
        release.update(real_update_smokes_pass=True,smoke_arms=smoke['arms'])
    run(root,cfg,release,arm)
    write(root/f'{stage}_{arm}_COMPLETE.json',dict(arm=arm,stage=stage,unix=time.time(),
        update=1 if stage=='smoke' else 24))


def spawn(root, stage, arm=None):
    import subprocess,sys
    from .storage import read,write
    from .replay_start import owner,alive
    label=stage+('_'+arm if arm else '')
    root=Path(root);dispatch=root/(label+'_DISPATCH.json')
    if dispatch.exists():
        d=read(dispatch)
        if not alive(d) and not (root/(label+'_COMPLETE.json')).exists():
            raise ValueError('owned V2V child exited: '+label)
        return d
    command=[sys.executable,'-B','-u','-m','tools.coopmwm.v2v_live',str(root),stage]
    if arm:command+=['--arm',arm]
    with (root/(label+'.log')).open('x') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    d=dict(owner(child.pid),argv=command,label=label);write(dispatch,d);return d


def watch(root):
    import fcntl,shutil
    from .storage import read,write
    from .protocol import sha
    from .replay_start import owner,alive,atomic
    root=Path(root);value=configuration(root);cfg=value['config']
    lock=(root/'WATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'WATCH_OWNER.json').exists():raise ValueError('adopt existing V2V watcher')
    write(root/'WATCH_OWNER.json',owner())
    children=[]
    while True:
        gate=resource_gate(root,cfg)
        atomic(root/'V2V_WATCH_STATUS.json',dict(gate,unix=time.time(),stage='resource_wait',children=children))
        if gate['ready']:break
        time.sleep(120)
    children=[spawn(root,'smoke',a) for a in ('A','B','R')]
    while not all((root/f'smoke_{a}_COMPLETE.json').exists() for a in ('A','B','R')):
        for child in children:
            if not alive(child) and not (root/(child['label']+'_COMPLETE.json')).exists():
                raise ValueError('V2V smoke failed: '+child['label'])
        atomic(root/'V2V_WATCH_STATUS.json',dict(unix=time.time(),stage='real_one_update_smokes',children=children))
        time.sleep(10)
    next_times={}
    for arm in ('A','B','R'):
        jobs=list((root/'jobs').glob('v2v1_warm_u001_'+arm+'_*/CLAIM.json'))
        if len(jobs)!=24 or not (root/f'{arm}_checkpoint_001.pt').exists():
            raise ValueError('real V2V smoke incomplete: '+arm)
        next_times[arm]=min(p.stat().st_mtime for p in jobs)+cfg['pace_s']
    write(root/'V2V_REAL_SMOKES_PASS.json',dict(pass_=True,arms=['A','B','R'],unix=time.time(),
        next_not_before_s=next_times,qualified_source=cfg['source_root'],
        smoke_checkpoints={a:sha(root/f'{a}_checkpoint_001.pt') for a in ('A','B','R')},
        simulation_reused_for_training=True,new_extra_smoke_sim_s=0))
    for arm in ('A','B','R'):
        path=root/f'WARM_{arm}_PAUSED.json'
        if read(path)['update']!=1:raise ValueError('smoke endpoint mismatch')
        saved=root/f'WARM_{arm}_PAUSED_AFTER_SMOKE.json'
        if saved.exists():raise ValueError('smoke endpoint already adopted')
        path.rename(saved)
    children=[spawn(root,'train',a) for a in ('A','B','R')]
    write(root/'V2V_STARTED.json',dict(unix=time.time(),children=children,first_update=1,
        resumed_after_smoke=2,final_update=24,common_u0_sha256=cfg['common_checkpoint_sha256'],
        funding_amendment_sha256=cfg['funding_amendment_sha256']))
    # Episode tests are separate durable children; the coordinator receives only
    # public hashes and never opens outcome values during training.
    while True:
        for child in children:
            if not alive(child) and not (root/(child['label']+'_COMPLETE.json')).exists():
                raise ValueError('V2V training child exited: '+child['label'])
        for update in (8,24):
            if all((root/f'{a}_UPDATE_{update:03}.json').exists() for a in ('A','B','R')):
                spawn(root,'test'+str(update))
        if all((root/f'train_{a}_COMPLETE.json').exists() for a in ('A','B','R')):
            if all((root/f'test{u}_COMPLETE.json').exists() for u in (8,24)):break
        atomic(root/'V2V_WATCH_STATUS.json',dict(unix=time.time(),stage='training',children=children))
        time.sleep(30)
    write(root/'WATCH_COMPLETE.json',dict(unix=time.time(),outcomes_closed=True))


def execute_test(root,cfg,spec):
    import os,torch
    from .accel_plant import episode
    from .final_runtime import public_completion
    from .protocol import sha,digest
    from .storage import write
    torch.set_num_threads(1);os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    weights=torch.load(Path(root)/f'{spec["arm"]}_checkpoint_{spec["checkpoint_update"]:03}.pt',weights_only=False)['policy']
    job=dict(spec,path=str(Path(root)/'jobs'/spec['id']),role='final_test',v2v=True,weights=weights,
        runtime=cfg['runtime'],checkpoint=cfg['checkpoint'],noise_seed=int(digest(spec)[:15],16),
        deterministic=True,guided=False,gpu_deadline=cfg['gpu_deadline'],accel_bounds=cfg['accel_bounds'])
    episode(job)
    meta=public_completion(spec,Path(job['path']))
    meta.update(checkpoint_update=spec['checkpoint_update'],checkpoint_sha256=sha(Path(root)/f'{spec["arm"]}_checkpoint_{spec["checkpoint_update"]:03}.pt'))
    write(Path(root)/'final_receipts'/(spec['id']+'.json'),meta)
    return meta


def episode_test(root, update):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from .warm_protocol import worlds
    from .storage import write
    from .accel_storage import reserve,ledger
    root=Path(root);cfg=configuration(root)['config'];accounting(root,cfg)
    if not all((root/f'{a}_UPDATE_{update:03}.json').exists() for a in ('A','B','R')):
        raise ValueError('fixed V2V checkpoints required')
    claim=root/f'test{update}_CLAIM.json'
    if claim.exists():raise ValueError('interrupted episode test; no replay')
    phase='v2v_interim' if update==8 else 'v2v_final'
    jobs=[dict(id=f'v2vtest{update:03}_{a}_{w["seed"]}',arm=a,world=w,sim_s=1200,
        phase=phase,kind='episode',checkpoint_update=update) for a in ('A','B','R') for w in worlds('final_test')]
    if {x['id'] for x in jobs}&{x['id'] for x in ledger(root)}:raise ValueError('test already charged')
    write(claim,dict(update=update,jobs=288,sim_s=345600,phase=phase,outcomes_closed=True))
    # Charge each complete episode immediately before submission, without
    # holding reservations during resource waits or retrying partial attempts.
    receipts=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for offset in range(0,len(jobs),2):
            pair=jobs[offset:offset+2]
            from .runner import admission
            admission(cfg)
            reserve(root,pair,phase)
            receipts.extend(pool.map(execute_test,[root]*len(pair),[cfg]*len(pair),pair))
    write(root/f'test{update}_COMPLETE.json',dict(update=update,unix=time.time(),jobs=len(receipts),
        phase=phase,outcomes_closed=True,public_receipt_ids=[r['id'] for r in receipts]))


if __name__=='__main__':
    import argparse,signal,traceback
    from .storage import write
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('stage',choices=('watch','smoke','train','test8','test24'))
    parser.add_argument('--arm',choices=('A','B','R'));args=parser.parse_args()
    try:
        if args.stage=='watch':watch(args.root)
        elif args.stage.startswith('test'):episode_test(args.root,int(args.stage[4:]))
        else:stage_arm(args.root,args.stage,args.arm)
    except BaseException as error:
        name=args.stage+('_'+args.arm if args.arm else '')
        write(args.root/(name+'_FAILED.json'),dict(unix=time.time(),error=repr(error),traceback=traceback.format_exc(),physics_retry=False))
        raise
