"""One owned B-parent handover; children and assigned searches finish once."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from .quota_transition import state,terminate
from .warm_health import assert_running
from .warm_spool import immutable


def paths(root,update):
    from .warm_protocol import batch_worlds
    return [root/'jobs'/f'warm_u{update:03}_B_{w["seed"]}' for w in batch_worlds(update)]


def freeze(root):
    root=Path(root)
    assert not (root/'TAIL_PARENT_FROZEN.json').exists(),'adopt existing handover'
    owner=json.loads((root/'QUOTA_B_DISPATCH.json').read_text())
    assert state(owner,'tools.coopmwm.quota_runtime') not in ('exited','exiting','T')
    u=max([0]+[int(p.stem.rsplit('_',1)[1]) for p in root.glob('B_UPDATE_[0-9][0-9][0-9].json')])+1
    jobs=paths(root,u)
    if not all((p/'CLAIM.json').exists() for p in jobs) or (root/f'B_checkpoint_{u:03}.pt').exists():return None
    # Only parent supervision is paused. Every existing SUMO/GPU child continues.
    os.kill(owner['pid'],signal.SIGSTOP)
    limit=time.monotonic()+5
    while state(owner,'tools.coopmwm.quota_runtime')!='T':
        if time.monotonic()>limit:raise ValueError('owned parent did not stop')
        time.sleep(.05)
    if (root/f'B_checkpoint_{u:03}.pt').exists() or (root/f'B_UPDATE_{u:03}.json').exists():
        os.kill(owner['pid'],signal.SIGCONT)
        return None  # Boundary raced before STOP; wait for the next wave.
    v=dict(owner=owner,adopt_update=u,resume_update=u-1,unix=time.time(),
        checkpoint_sha256=__import__('hashlib').sha256((root/f'B_checkpoint_{u-1:03}.pt').read_bytes()).hexdigest(),
        existing_claims=[str(p) for p in jobs],children_continue=True,new_sumo_sim_s=0)
    immutable(root/'TAIL_PARENT_FROZEN.json',v)
    return v


def completed(root,frozen):
    jobs=paths(root,frozen['adopt_update'])
    if not all((p/'RESULT.json').exists() for p in jobs):return False
    for p in jobs:
        claim=json.loads((p/'CLAIM.json').read_text());proc=Path('/proc',str(claim['pid']))
        if not (proc/'stat').exists():continue
        s=(proc/'stat').read_text().split()
        # A stale claim PID is never used for a signal or interpreted as our child.
        if int(s[3])!=frozen['owner']['pid']:continue
        children=proc/'task'/str(claim['pid'])/'children'
        if children.exists():
            for child in children.read_text().split():
                cmd=Path('/proc',child,'cmdline')
                if cmd.exists() and b'sumo' in cmd.read_bytes().split(bytes([0]))[0]:return False
    return True


def handover(root,frozen,plan):
    root=Path(root);assert not (root/'TAIL_B_DISPATCH.json').exists(),'adopt existing new owner'
    assert completed(root,frozen);assert_running(root,'B')
    from .accel_learner import load
    from .exact_runtime import assert_live
    from .protocol import sha
    assert_live(root)
    owner=frozen['owner'];assert state(owner,'tools.coopmwm.quota_runtime')=='T'
    resume=frozen['resume_update'];u=frozen['adopt_update']
    assert sha(root/f'B_checkpoint_{resume:03}.pt')==frozen['checkpoint_sha256']
    _,_,_,payload=load(root/f'B_checkpoint_{resume:03}.pt');assert payload['update']==resume
    assert not (root/f'B_checkpoint_{u:03}.pt').exists() and not (root/f'B_UPDATE_{u:03}.json').exists()
    claims={json.loads((p/'CLAIM.json').read_text())['pid'] for p in paths(root,u)}
    children=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            s=(p/'stat').read_text().split()
            if int(s[3])!=owner['pid'] or s[2]=='Z':continue
            cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
            assert ('multiprocessing.resource_tracker' in cmd or
                (int(p.name) in claims and 'multiprocessing.spawn' in cmd)),'non-wave child; inspect before handover'
            children.append(dict(pid=int(p.name),start_ticks=s[21],command=cmd))
        except FileNotFoundError:continue
    binding=dict(plan,resume_update=resume,adopt_update=u)
    immutable(root/'TAIL_RUNTIME_BINDING.json',binding)
    terminate(owner,'tools.coopmwm.quota_runtime')
    for child in children:
        p=Path('/proc',str(child['pid']))
        if not (p/'stat').exists():continue
        s=(p/'stat').read_text().split();assert s[21]==child['start_ticks']
        if s[2]=='Z':continue
        assert (p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()==child['command']
        os.kill(child['pid'],signal.SIGTERM)
    immutable(root/'TAIL_HANDOVER_COMPLETE.json',dict(previous=owner,adopt_update=u,resume_update=resume,
        all24results_adopted=True,checkpoint_sha256=frozen['checkpoint_sha256'],ended_completed_workers=children,
        new_sumo_sim_s=0,physics_replay=False,unix=time.time()))
    target=root/'TAIL_B_DISPATCH.json';assert not target.exists(),'adopt existing new owner'
    source=plan['source_root']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',
        PYTHONPATH=source+'/pytools:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
    env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
    with (root/'TAIL_B.log').open('x') as log:
        p=subprocess.Popen(['nice','-n','19',os.sys.executable,'-B','-u','-m','tools.coopmwm.tail_runtime',str(root)],
            env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    s=Path('/proc',str(p.pid),'stat').read_text().split()
    value=dict(pid=p.pid,start_ticks=s[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        root=str(root),source=source,component='B',unix=time.time())
    immutable(target,value)
    return value


def watch(root):
    import fcntl
    root=Path(root);plan=json.loads((root/'TAIL_PLAN_BINDING.json').read_text())
    assert plan['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
    from .protocol import sha
    for name,h in plan['cpu']['source_hashes'].items():assert sha(Path(plan['source_root'])/name)==h,name
    lock=(root/'TAIL_TRANSITION.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    deadline=json.loads((root/'WARM_REMOTE_BINDING.json').read_text())['deadline_unix']
    try:
        frozen=None
        while frozen is None:
            assert_running(root,'B')
            if time.time()>deadline:raise ValueError('original deadline')
            frozen=freeze(root)
            if frozen is None:time.sleep(2)
        while not completed(root,frozen):
            assert_running(root,'B')
            if time.time()>deadline:raise ValueError('original deadline; existing parent preserved frozen')
            time.sleep(2)
        return handover(root,frozen,plan)
    except BaseException as error:
        immutable(root/'TAIL_TRANSITION_FAILED.json',dict(error=repr(error),unix=time.time(),no_retry=True));raise
    finally:lock.close()
