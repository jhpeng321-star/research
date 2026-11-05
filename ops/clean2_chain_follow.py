"""Single read-only local follower; cpuV chain remains independent of this laptop."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path('E:/agent-work/_sol')
REPO=Path('E:/agent-work/c4-rerun-launch')
RUN=Path('E:/research-dense/runs/active/C4-V7FT-20261017')
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.stderr.reconfigure(encoding='utf-8',errors='replace')


def write_once(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('existing receipt differs: '+str(path))
        return
    temporary=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    with temporary.open('x',encoding='utf-8') as f:
        json.dump(value,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
    os.link(temporary,path);temporary.unlink()


def main():
    dispatch=json.loads((REPO/'CLEAN2_CHAIN_DISPATCH.json').read_text())
    with (BASE/'CLEAN2_CHAIN_FOLLOW.lock').open('a+b') as lock:
        import msvcrt
        lock.seek(0,2)
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        write_once(BASE/'CLEAN2_CHAIN_FOLLOW_PID.json',dict(pid=os.getpid(),
            started=datetime.now().astimezone().isoformat(),source_commit=dispatch['source_commit']))
        seen=None
        while True:
            try:
                result=subprocess.run([sys.executable,'-B',str(BASE/'clean2_transport.py'),
                    str(BASE/'clean2_chain_progress_remote.py'),'--source-package','--timeout','25'],
                    capture_output=True,timeout=40)
                if result.returncode:raise RuntimeError(result.stderr.decode('utf-8',errors='replace'))
                snapshot=json.loads(result.stdout)
                status=snapshot['chain']['status']
                if status and status['source_commit']!=dispatch['source_commit']:
                    raise ValueError('chain source identity changed')
                if status and status['sequence']!=seen:
                    stamp=datetime.now().astimezone().isoformat()
                    lines=[stamp+' CLEAN2 chain '+status['step']+' '+status['status']+'.',
                        'CLEAN2 Source '+dispatch['source_commit']+'; cpuV chain PID'+str(dispatch['pid'])+'.',
                        'CLEAN2 '+('Fault '+str(status['detail']) if status['status']=='FAILED' else 'Exact receipt checks and one-shot gate remain mandatory.'),
                        'CLEAN2 Root '+dispatch['root']+'; ledger '+json.dumps(snapshot.get('charges',{}),sort_keys=True)+'.',
                        'CLEAN2 Detached cpuV chain remains laptop-independent; clean1 preserved stopped; no automatic retry.']
                    with (RUN/'CLEAN1_STATUS.md').open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n\n')
                    print('\n'.join(lines),flush=True);seen=status['sequence']
                launched=snapshot['receipts']['LAUNCHED.json']
                if launched:
                    if launched['runtime_commit']!=dispatch['source_commit']:raise ValueError('launch source identity changed')
                    write_once(REPO/'CLEAN2_LAUNCHED.json',launched)
                    marker='CLEAN2_LAUNCH_'+launched['runtime_commit']+'_'+str(launched['unix'])
                    registration=RUN/'REGISTRATION.md'
                    if marker not in registration.read_text(encoding='utf-8'):
                        with registration.open('a',encoding='utf-8') as f:
                            f.write('\n### '+datetime.now().astimezone().isoformat()+' — CLEAN2 LAUNCHED (read-only follower; system clock)\n'+
                                marker+'\nActual cpuV LAUNCHED.json: '+json.dumps(launched,sort_keys=True)+'\n'+
                                'Exact final smoke/fix/reporting and single zero-event gate PASS required by the chain. '+
                                'Clean1 retained stopped; own clean2 ledger; all original schedules/windows and u24 synergy criterion unchanged.\n')
                    sys.path.insert(0,str(BASE))
                    import clean2_learning_watch
                    clean2_learning_watch.main()
                    return
                if status and status['status']=='FAILED':return
            except Exception as error:
                print(datetime.now().astimezone().isoformat()+' CLEAN2 FOLLOW '+repr(error),flush=True)
            time.sleep(60)


if __name__=='__main__':main()
