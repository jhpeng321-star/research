"""One-use remote parent/broker handover; old physics and inference finish once."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from .quota_runtime import binding,install
from .warm_spool import immutable
from .warm_health import assert_running


def state(owner,needle):
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id']
    p=Path('/proc')/str(owner['pid'])
    if not (p/'stat').exists():return 'exited'
    s=(p/'stat').read_text().split();assert s[21]==owner['start_ticks'],'PID identity changed'
    if s[2]=='Z':return 'exited'
    cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
    assert needle in cmd and owner['root'] in cmd,'command identity changed'
    return s[2]


def terminate(owner,needle):
    status=state(owner,needle)
    if status=='exited':return
    os.kill(owner['pid'],signal.SIGTERM)
    if status=='T':os.kill(owner['pid'],signal.SIGCONT)
    end=time.monotonic()+15
    while state(owner,needle)!='exited':
        if time.monotonic()>end:raise ValueError('owned parent did not exit; no duplicate dispatch')
        time.sleep(.1)


def launch(root,component,source):
    target=root/f'QUOTA_{component}_DISPATCH.json'
    assert not target.exists(),'one-use owner already dispatched; adopt receipt'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',
        PYTHONPATH=source+'/pytools:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
    env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
    with (root/f'QUOTA_{component}.log').open('x') as log:
        p=subprocess.Popen(['nice','-n','19',os.sys.executable,'-B','-u','-m','tools.coopmwm.quota_runtime',str(root),'--component',component],
            env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    s=Path(f'/proc/{p.pid}/stat').read_text().split()
    value=dict(pid=p.pid,start_ticks=s[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        root=str(root),source=source,component=component,unix=time.time())
    immutable(target,value)
    return value


def replace_broker(root):
    root=Path(root);b=binding(root);assert_running(root,'B')
    old=json.loads((root/'CAPACITY_BROKER_DISPATCH_V2.json').read_text())
    terminate(old,'tools.coopmwm.capacity_recovery')
    value=launch(root,'broker',b['source_root'])
    immutable(root/'MAIN0548_BROKER_HANDOVER.json',dict(previous=old,current=value,inference_replay=False,unix=time.time()))
    return value


def completed(root):
    previous=json.loads((root/'MAIN0508_REPAIR_BINDING_V2.json').read_text())
    paths=[root/'jobs'/f'warm_u001_B_{w["seed"]}' for w in previous['repair_plan']['worlds']]
    if not all((p/'RESULT.json').exists() for p in paths):return False
    # A RESULT is written before Backend.close; do not terminate a worker while
    # its final SUMO child is still closing, even though the result is complete.
    for p in paths:
        c=json.loads((p/'CLAIM.json').read_text());children=Path(f'/proc/{c["pid"]}/task/{c["pid"]}/children')
        if children.exists():
            for child in children.read_text().split():
                cmd=Path('/proc',child,'cmdline')
                if cmd.exists() and b'sumo' in cmd.read_bytes().split(bytes([0]))[0]:return False
    return True


def handover(root):
    root=Path(root);b=binding(root)
    assert completed(root),'current physics still open'
    assert_running(root,'B')
    wr,_=install(root)
    from .accel_learner import load
    from .exact_runtime import assert_live
    assert_live(root)
    for arm in ('B','R'):
        config=wr.verify(root,arm);_,_,_,payload=load(root/f'{arm}_checkpoint_000.pt')
        assert payload['update']==0 and config['quota_aware_admission']
    frozen=json.loads((root/'MAIN0548_FROZEN_PARENTS.json').read_text())
    # Old pool children hold completed results in their return pipes. Capture
    # identities under the OLD parent's ancestry before signalling each one.
    old_b=frozen['owners']['B']['pid'];children=[];ended=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            s=(p/'stat').read_text().split()
            if int(s[3])!=old_b or s[2]=='Z':continue
            cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
            assert 'multiprocessing.spawn' in cmd or 'multiprocessing.resource_tracker' in cmd,'unexpected owned child'
            children.append(dict(pid=int(p.name),start_ticks=s[21],command=cmd))
        except FileNotFoundError:continue
    for arm,owner in frozen['owners'].items():terminate(owner,'tools.coopmwm.capacity_recovery')
    for child in children:
        p=Path('/proc')/str(child['pid'])
        if not (p/'stat').exists():continue
        s=(p/'stat').read_text().split();assert s[21]==child['start_ticks']
        if s[2]=='Z':continue
        assert (p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()==child['command']
        os.kill(child['pid'],signal.SIGTERM);ended.append(child)
    immutable(root/'MAIN0548_HANDOVER_COMPLETE.json',dict(unix=time.time(),parents=frozen['owners'],ended_completed_workers=ended,
        source=b['source_root'],all24results_adopted=True,new_sumo_sim_s=0,optimizer_replay=False))
    owners={arm:launch(root,arm,b['source_root']) for arm in ('B','R')}
    immutable(root/'MAIN0548_TRAINING_CONTINUED.json',dict(unix=time.time(),owners=owners,B_u1_results_adopted=True,R_pipelined=True))
    return owners


def watch(root):
    import fcntl
    root=Path(root);b=binding(root)
    lock=(root/'MAIN0548_TRANSITION.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    deadline=json.loads((root/'WARM_REMOTE_BINDING.json').read_text())['deadline_unix']
    try:
        while not completed(root):
            assert_running(root,'B')
            if time.time()>deadline:raise ValueError('original deadline; no resource extension')
            time.sleep(5)
        time.sleep(2)
        handover(root)
    except BaseException as error:
        immutable(root/'MAIN0548_TRANSITION_FAILED.json',dict(error=repr(error),unix=time.time(),no_retry=True));raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('replace_broker','handover','watch'));p.add_argument('root',type=Path)
    a=p.parse_args();print(json.dumps(globals()[a.mode](a.root)))
