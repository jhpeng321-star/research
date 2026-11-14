"""Receipt-only V3 publisher. Remote training/inference never depend on this PID."""
import argparse
from datetime import datetime,timezone
import json
import subprocess
import sys
import time
import uuid
from . import ops
from .storage import read,write
from .protocol import sha,checked
from .warm_deploy import ROOT
from .warm_transport import QueueTransport


def commit(paths):
    paths=sorted({p.replace('\\','/') for p in paths})
    if not paths:return
    def git(*args):return subprocess.check_output(['git',*args],cwd=ops.LOCAL,text=True,encoding='utf-8')
    if git('diff','--cached','--name-only').strip():raise ValueError('other staged work; preserve it')
    # Observers retain every immutable receipt, so their complete path list grows
    # past Windows' CreateProcess limit. Inspect only changed intended files and
    # send exact staging pathspecs on stdin, never in an unbounded command line.
    pending=set(git('ls-files','--modified','--others','--exclude-standard','-z').split('\0'))
    paths=[p for p in paths if p in pending]
    if not paths:return
    print(git('status','--short','--untracked-files=no'),flush=True)
    batch=[]
    for path in paths:
        literal=':(literal)'+path
        if batch and len(subprocess.list2cmdline(['git','diff','--stat','--',*batch,literal]))>16000:
            print(git('diff','--stat','--',*batch),flush=True);batch=[]
        batch.append(literal)
    if batch:print(git('diff','--stat','--',*batch),flush=True)
    print(git('log','--oneline','-10'),flush=True)
    pathspecs='\0'.join(':(literal)'+p for p in paths)+'\0'
    subprocess.run(['git','add','--pathspec-from-file=-','--pathspec-file-nul'],
        input=pathspecs.encode('utf-8'),cwd=ops.LOCAL,check=True)
    if git('diff','--cached','--name-only').strip():
        subprocess.run(['git','commit','-m','coopmwm: publish durable warm training checks'],cwd=ops.LOCAL,check=True)


def run():
    import msvcrt
    from .warm_remote_ops import status
    from .warm_publish import publish
    from .warm_watch import backups
    lock=(ops.EVIDENCE/'WARM_V3_OBSERVER.claim').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'1');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    binding=read(ops.EVIDENCE/'WARM_V3_OBSERVER_BINDING.json')
    original=ops.ssh.remote
    def event(v):
        write(ops.EVIDENCE/('WARM_V3_TRANSPORT_'+uuid.uuid4().hex+'.json'),dict(v,unix=time.time()))
    transport=QueueTransport(original,ROOT,event)
    def remote(host,source,timeout=180):
        return transport.call(source,timeout=timeout) if host=='cpuQ' else original(host,source,timeout=timeout)
    ops.ssh.remote=remote
    try:
        while time.time()<binding['deadline_unix']:
            for name,h in binding['helper_hashes'].items():
                if sha(ops.LOCAL/name)!=h:raise ValueError('observer source changed '+name)
            live=status();changed=publish();print(json.dumps(live),flush=True)
            before={str(p.relative_to(ops.LOCAL)) for p in ops.EVIDENCE.glob('WARM_BACKUP_*.json')}
            backups()
            changed += [str(p.relative_to(ops.LOCAL)) for p in ops.EVIDENCE.glob('WARM_BACKUP_*.json') if str(p.relative_to(ops.LOCAL)) not in before]
            changed += [str(p.relative_to(ops.LOCAL)) for p in ops.EVIDENCE.glob('WARM_V3_TRANSPORT_*.json')]
            pauses=[k for k in live['receipts'] if k.startswith('PAUSE_') or k=='WARM_GUIDED_STOP.json']
            if pauses:
                p=ops.EVIDENCE/'WARM_V3_GUARD_PUBLICATION.json'
                if not p.exists():
                    write(p,dict(unix=time.time(),evidence={k:live['receipts'][k] for k in pauses},
                        action='B/R paused remotely; A unaffected; diagnosis/registered release required; no automatic resume'))
                    changed.append(str(p.relative_to(ops.LOCAL)))
            if live['updates']['B']>=4:
                p=ops.EVIDENCE/'WARM_V3_VIABILITY_PACKAGE.json';local=ops.EVIDENCE/'warm_live'
                if not p.exists() and (local/'AB_CHECK_004.json').exists():
                    names=[q.name for q in local.glob('*.json') if q.name.startswith(('B_VIABILITY_','B_UPDATE_','AB_CHECK_','ACCEL_EXACT_PLAN_'))]
                    write(p,dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),receipts=sorted(names),
                        checkpoint_rule='fixedu4 early viability pause; final panel unopened',charged_sim_s=live['charged_sim_s']))
                    doc=ops.LOCAL/'docs/paper_results/COOPMWM.md'
                    with doc.open('a',encoding='utf-8') as f:f.write('\n## B-u4 viability package available\n- SUMO critic contrasts, plan routing/type/rank, exact-twin coverage and B-vs-A curves are in `coopmwm/WARM_V3_VIABILITY_PACKAGE.json` and the linked immutable receipts. Registered owner-review pause; no final-panel outcomes.\n')
                    changed.extend([str(p.relative_to(ops.LOCAL)),str(doc.relative_to(ops.LOCAL))])
            commit(changed)
            failures=[k for k in live['receipts'] if '_FAILED_V3' in k]
            if live['active_transport_stop'] or failures or pauses:
                raise ValueError('durable pause/failure published; inspect remote evidence; no automatic restart')
            if all('WARM_'+a+'_PAUSED.json' in live['receipts'] for a in ('A','B','R')):
                p=ops.EVIDENCE/'WARM_V3_OBSERVER_COMPLETE.json';write(p,live);commit([str(p.relative_to(ops.LOCAL))]);return
            time.sleep(60)
        raise ValueError('registered observer deadline')
    except BaseException as error:
        p=ops.EVIDENCE/('WARM_V3_OBSERVER_EXIT_'+str(int(time.time()))+'.json')
        write(p,dict(unix=time.time(),error=repr(error),training_inference_depend_on_observer=False,
            disposition='remote owned PIDs/receipts preserved; no physics/inference retry or PID control'))
        commit([str(p.relative_to(ops.LOCAL))]);raise
    finally:ops.ssh.remote=original;lock.close()


def dispatch():
    target=ops.EVIDENCE/'WARM_V3_OBSERVER_DISPATCH.json';assert not target.exists()
    assert (ops.EVIDENCE/'WARM_B_DISPATCH_V3.json').exists() and (ops.EVIDENCE/'WARM_R_DISPATCH_V3.json').exists()
    names=('warm_v3_watch','warm_remote_ops','warm_publish','warm_watch','warm_storage','warm_transport')
    binding=dict(deadline_unix=read(ops.EVIDENCE/'WARM_A48_BINDING.json')['gpu_deadline'],
        helper_hashes={f'tools/coopmwm/{n}.py':sha(ops.LOCAL/f'tools/coopmwm/{n}.py') for n in names},
        contract='observe/publish/backup/commit only; bounded connection recovery; never dispatch/resume/stop physics or GPU work')
    write(ops.EVIDENCE/'WARM_V3_OBSERVER_BINDING.json',binding)
    with (ops.EVIDENCE/'WARM_V3_OBSERVER.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-X','utf8','-m','tools.coopmwm.warm_v3_watch','run'],cwd=ops.LOCAL,stdout=log,stderr=subprocess.STDOUT)
    value=dict(pid=p.pid,unix=time.time(),binding=binding);write(target,value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','dispatch'));globals()[p.parse_args().mode]()
