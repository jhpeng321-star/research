"""18:54 primary-first independent arms; immutable legacy B runs retained."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import os
import sys
import threading
import time
import torch
from . import runner as run,gate
from .storage import read,write,ledger,reconciled
from .protocol import UPDATES,CHECK_EVERY,TRAINING_SEEDS,batch_worlds,worlds,budget
from .policy import Policy
from .exploration import update as guided_update,evidence,trust
from tools.coopnative.learner import update as team_update,save,load


def stem(si,arm):return f'S{si}_{arm}'+('1854' if arm in ('B','R') else '')


def restored(path):
    with INIT_LOCK:return load(path)


def train_arm(root,b,si,arm,controls,peer_done=None):
    seed=TRAINING_SEEDS[si];prefix=stem(si,arm)
    b=dict(b,training_seed=seed,workers=(4 if arm=='A' else 8) if peer_done is not None else 12)
    start=run.checkpoint_path(root,b,si,arm,0)
    if not start.exists():
        # Protect the process-global CPU RNG during simultaneous initialization.
        with INIT_LOCK:
            torch.manual_seed(seed);p=Policy();o=torch.optim.Adam(p.parameters(),lr=3e-4)
            g=torch.Generator().manual_seed(seed);save(start,p,o,g,0,reconciled(root))
    for u in range(1,UPDATES+1):
        if (root/'STOP.json').exists():raise ValueError('STOP before arm update')
        begin=time.monotonic()
        if peer_done is not None and peer_done.is_set():b=dict(b,workers=12)
        policy,optimizer,g,_=restored(run.checkpoint_path(root,b,si,arm,u-1))
        checkpoint=run.checkpoint_path(root,b,si,arm,u)
        update=root/f'{prefix}_UPDATE_{u:03}.json'
        if not checkpoint.exists():
            beta=read(root/f'{prefix}_UPDATE_{u-1:03}.json')['trust'] if u>1 else 1.
            extra={}
            if arm in ('B','R'):
                gp=root/'gates'/f'{prefix}_u{u:03}'
                quota=(read(root/'gates'/f'{stem(si,"B")}_u{u:03}'/'VERDICT.json')['actual']
                       if arm=='R' else b['volume']['N_exec'])
                gate.initialize(gp,quota,b['volume']['sampling']);extra['gate_path']=str(gp)
            # A's existing physical IDs are retained, including update4/check/dev.
            ident=f's{si}_u{u:03}' if arm=='A' else f'v1854_s{si}_u{u:03}'
            episodes=run.wave(root,dict(b,beta=beta,sampling=b['volume'].get('sampling')),policy,
                batch_worlds(u,si),arm,'train',ident,'train',extra=extra)
            verdict=gate.verify(gp,episodes) if arm in ('B','R') else None
            metrics=(guided_update if arm in ('B','R') else team_update)(policy,optimizer,episodes,g)
            ev=evidence(episodes) if arm in ('B','R') else []
            if arm in ('B','R'):
                running=[r for old in range(1,u) for r in read(root/f'{prefix}_EVIDENCE_{old:03}.json')['records']]+ev
                write(root/f'{prefix}_EVIDENCE_{u:03}.json',dict(records=ev))
            tr=trust(beta,running) if arm in ('B','R') else dict(beta=0.)
            charged=reconciled(root);save(checkpoint,policy,optimizer,g,u,charged,ppo=metrics)
            wall=time.monotonic()-begin
            write(update,dict(update=u,arm=arm,training_seed=seed,ppo=metrics,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                team_only_noise=run.training_noise(episodes),cooperative_move_noise=run.cooperative_noise(episodes),
                training_sim_s=u*28800,charged_sim_s=charged,beta_used=beta if arm in ('B','R') else 0.,
                trust=tr,proposal_routing=run.proposal_report(episodes),guidance_gate=verdict,
                wall_s=wall,eta_remaining_hours=(UPDATES-u)*wall/3600,
                reward_source='SUMO team delay only; no WM reward',engineering_amendment='18:54'))
        else:policy,optimizer,g,_=restored(checkpoint)
        if u%CHECK_EVERY==0:
            ident=f's{si}_check{u:03}' if arm=='A' else f'v1854_s{si}_check{u:03}'
            rs=run.wave(root,b,policy,worlds('check'),arm,'check',ident,'checks',True)
            if not (root/f'{prefix}_CHECK_{u:03}.json').exists():run.report_panel(root,u,rs,controls,'check',si,prefix)
        if (si==0 and u==4) or u==UPDATES:
            ident=f's{si}_dev{u:03}' if arm=='A' else f'v1854_s{si}_dev{u:03}'
            rs=run.wave(root,b,policy,worlds('dev'),arm,'dev',ident,'dev',True)
            if not (root/f'{prefix}_DEV_{u:03}.json').exists():run.report_panel(root,u,rs,read(root/'MAPPO_DEV.json')['records'],'dev',si,prefix)
    if peer_done is not None:peer_done.set()
    if not (root/f'{prefix}_TRAINING_COMPLETE.json').exists():write(root/f'{prefix}_TRAINING_COMPLETE.json',dict(final_update=UPDATES,training_seed=seed,arm=arm))
    return policy


INIT_LOCK=threading.Lock()


def main(root,only_A=False):
    import fcntl
    b=run.verify(root);assert b['volume_amendment']
    claim=(root/'SUPERVISOR.claim').open('a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    torch.set_num_threads(1)
    try:
        if b.get('unstarted_precharged_jobs'):
            policy,_,_,_=restored(run.checkpoint_path(root,b,0,'A',4))
            run.wave(root,b,policy,worlds('dev'),'A','dev','s0_dev004','dev',True)
        reconciled(root)
        controls=run.wave(root,b,Policy(),worlds('check'),'mappo','check','control','checks',True)
        if only_A:
            train_arm(root,b,0,'A',controls)
            return
        assert b['volume']['release_guided'],'guided arms are held until measured qualification'
        for si in range(2):
            done=threading.Event()
            with ThreadPoolExecutor(max_workers=2) as pool:
                tasks={a:pool.submit(train_arm,root,b,si,a,controls,done) for a in ('A','B')}
                def failure(f):
                    if f.exception() and not (root/'STOP.json').exists():write(root/'STOP.json',dict(error=repr(f.exception()),charged_sim_s=sum(r['sim_s'] for r in ledger(root))))
                for f in tasks.values():f.add_done_callback(failure)
                policies={a:f.result() for a,f in tasks.items()}
            write(root/f'S{si}_TRAINING_COMPLETE.json',dict(final_update=UPDATES,active_arms=['A','B'],engineering_amendment='18:54'))
            run.final_test(root,b,policies,si)
            if si==0:
                if not (root/'R_SMOKE.json').exists():
                    smokes=run.wave(root,b,Policy(),worlds('dev')[:1],'R','dev','smoke','mechanics',True)
                    base=next(r for r in read(root/'MAPPO_DEV.json')['records'] if r['world']==smokes[0]['world'])
                    assert smokes[0]['metrics']==base['metrics']
                    write(root/'R_SMOKE.json',dict(pass_=True,sim_s=1200,MAPPO_baseline_reused=True))
                rp=train_arm(root,b,0,'R',controls)
                write(root/'R0_TRAINING_COMPLETE.json',dict(final_update=UPDATES,active_arms=['R']))
                run.random_final_test(root,b,rp)
        write(root/'COMPLETE.json',dict(budget=budget(),charged_sim_s=reconciled(root),engineering_amendment='18:54'))
    except BaseException as error:
        if not (root/'STOP.json').exists():write(root/'STOP.json',dict(error=repr(error),charged_sim_s=sum(r['sim_s'] for r in ledger(root))))
        raise
    finally:claim.close()


if __name__=='__main__':main(Path(sys.argv[1]),only_A='--only-A' in sys.argv[2:])
