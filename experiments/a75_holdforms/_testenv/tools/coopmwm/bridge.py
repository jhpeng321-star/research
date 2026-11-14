"""Owned persistent stdio WM service; one request at a time, no SUMO dispatch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from . import ops
from .deploy import ROOT,GPU_ROOT,EVIDENCE
from .protocol import digest
from .storage import read,write


def next_request():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({ROOT!r});assert r.is_dir()
if (r/'STOP.json').exists():print(json.dumps(dict(stop=True)))
else:
 paths=[p for p in (r/'requests').glob('*.json') if not (r/'responses'/p.name).exists()]
 # Diagnostic quartet requests have priority over exploration after the first wave.
 records=[json.loads(p.read_text()) for p in paths]
 records.sort(key=lambda r:(r.get('mode')=='proposal',r['state_id']))
 print(json.dumps(dict(record=records[0] if records else None)))
''',timeout=35)
    return json.loads(out)


def persist(record,result):
    assert result['record_sha256']==digest(record)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,sys,json
r=pathlib.Path({ROOT!r});sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.storage import read,write
result=json.loads({json.dumps(result)!r});p=r/'responses'/{(record['state_id']+'.json')!r}
if p.exists():assert read(p)==result
else:write(p,result)
print(json.dumps(dict(persisted=True,state_id={record['state_id']!r})))
''',timeout=45)
    return json.loads(out)


def run():
    # Byte-range locking works on the existing Windows environment; no daemon duplication.
    import msvcrt
    lock=(EVIDENCE/'BRIDGE.claim').open('a+b');lock.seek(0)
    if lock.read(1)==b'':lock.write(b'1');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    gpu=None
    try:
        while True:
            item=next_request()
            if item.get('stop'):break
            record=item['record']
            if record is None:
                time.sleep(5);continue
            if gpu is None:
                log=(EVIDENCE/'GPU_SERVICE.log').open('x',encoding='utf-8')
                command=('/root/miniconda3/bin/python -B -u -c '
                    '"import sys;sys.path.insert(0,\''+GPU_ROOT+'/pytools\');'
                    'from pathlib import Path;from tools.coopmwm.service import main;main(Path(\''+GPU_ROOT+'\'))"')
                gpu=subprocess.Popen(['ssh','-o','BatchMode=yes','gpuA',command],stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1)
                write(EVIDENCE/'GPU_SSH_OWNER.json',dict(pid=gpu.pid,host='gpuA',root=GPU_ROOT,unix=time.time()))
                while True:
                    line=gpu.stdout.readline()
                    if not line:raise RuntimeError('GPU service ended before readiness; preserve GPU_SERVICE.log')
                    if not line.startswith('COOPMWM_JSON '):continue
                    value=json.loads(line[len('COOPMWM_JSON '):])
                    if value['event']=='READY':
                        write(EVIDENCE/'GPU_READY.json',value);break
            begin=time.monotonic()
            gpu.stdin.write(json.dumps(record)+'\n');gpu.stdin.flush()
            while True:
                line=gpu.stdout.readline()
                if not line:raise RuntimeError('GPU service ended mid-request; no redispatch')
                if not line.startswith('COOPMWM_JSON '):continue
                value=json.loads(line[len('COOPMWM_JSON '):])
                if value['event']=='RESULT':break
            assert value['key']==record['state_id']
            persist(record,value['result'])
            write(EVIDENCE/'forecasts'/(record['state_id']+'.json'),dict(
                state_id=record['state_id'],mode=record.get('mode','diagnostic'),
                wall_s=time.monotonic()-begin,unix=time.time(),result_sha256=digest(value['result'])))
            print(json.dumps(dict(event='FORECAST',key=record['state_id'],wall_s=time.monotonic()-begin)),flush=True)
    except BaseException as error:
        dest=EVIDENCE/'BRIDGE_ERROR.json'
        if not dest.exists():write(dest,dict(error=repr(error),unix=time.time()))
        raise
    finally:
        # EOF only to our own service; never signal a peer PID.
        if gpu and gpu.stdin:gpu.stdin.close()
        lock.close()


def start():
    assert EVIDENCE.is_dir()
    assert not (EVIDENCE/'BRIDGE_DISPATCH.json').exists(),'inspect existing own dispatch'
    assert (EVIDENCE/'GPU_BINDING.json').exists()
    log=(EVIDENCE/'BRIDGE.log').open('x',encoding='utf-8')
    p=subprocess.Popen([sys.executable,'-B','-u','-m','tools.coopmwm.bridge','run'],cwd=ops.LOCAL,
        stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    receipt=dict(pid=p.pid,unix=time.time(),root=ROOT,gpu_root=GPU_ROOT)
    write(EVIDENCE/'BRIDGE_DISPATCH.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','run'));a=p.parse_args()
    start() if a.mode=='start' else run()
