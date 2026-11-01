"""Owned local SSH relay: gpuR inference, gpuA exclusive claims and ledger.

There is no retry or reassignment. A broken connection stops the live run.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.wmplan import ops

GROOT = '/root/wmplan20261101_v1'
TAG = os.environ.get('WMPLAN_RELAY_TAG','wmplan_gpuR')
if os.environ.get('WMPLAN_RUN_ROOT'):
    ops.ROOT = os.environ['WMPLAN_RUN_ROOT']
    ops.OUT = Path(os.environ['WMPLAN_RECEIPT_DIR'])


def next_message(process):
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError('owned gpuR SSH service ended; no replay')
        if line.startswith('WMPLAN_JSON '):
            return json.loads(line[len('WMPLAN_JSON '):])
        print(line.rstrip(),flush=True)


def claim():
    return json.loads(ops.remote(f'''import pathlib,sys,json,time
root=pathlib.Path({ops.ROOT!r});sys.path.insert(0,str(root/'pytools'))
from tools.coopplan.storage import write,read
out=dict(done=(root/'COMPLETE.json').is_file(),stopped=(root/'STOP.json').is_file(),record=None)
if not out['done'] and not out['stopped']:
 for p in sorted((root/'requests').glob('*.json')):
  try:write(root/'forecast_claims'/p.name,dict(worker=1,host='gpuR',relay_pid={os.getpid()},unix=time.time()))
  except FileExistsError:continue
  (root/'forecasts'/p.stem).mkdir()
  out['record']=read(p);break
print(json.dumps(out))
'''))


def deliver(message):
    # Both transports use the same atomic response publication and audit files.
    return ops.remote(f'''import pathlib,sys,json
root=pathlib.Path({ops.ROOT!r});sys.path.insert(0,str(root/'pytools'))
from tools.coopplan.storage import write
from tools.wmplan.predict import publish
m=json.loads({json.dumps(message)!r});key=m.pop('key');event=m.pop('event')
assert (root/'forecast_claims'/f'{{key}}.json').is_file()
if event=='FUTURE':write(root/'forecasts'/key/f'k{{m["k"]}}.json',m['result'])
elif event=='RESULT':
 assert len(list((root/'forecasts'/key).glob('k*.json')))==8
 publish(root/'responses'/f'{{key}}.json',m)
else:raise ValueError(event)
print(json.dumps(dict(delivered=event,key=key)))
''')


def main():
    assert (ops.OUT/'MIRROR.json').is_file()
    env = f'PYTHONPATH={GROOT}/pywm:{GROOT}/deps OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1'
    command = (f'cd {GROOT} && {env} nice -n 19 /root/miniconda3/bin/python -B -u '
               f'-m tools.wmplan.predict {GROOT} --worker 1 --stdio')
    assert (ops.LOCAL/'_out').is_dir()
    with (ops.LOCAL/'_out'/(TAG+'_stderr.log')).open('x',encoding='utf-8') as err:
        process = subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
                                    'gpuR',command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                   stderr=err,text=True,encoding='utf-8',bufsize=1)
        try:
            ready = next_message(process)
            if ready['event'] != 'READY':
                raise ValueError('gpuR startup protocol')
            owner = dict(**ready['owner'],relay_pid=os.getpid(),ssh_pid=process.pid,
                         unix=time.time(),root=GROOT)
            ops.save('GPU_R_READY.json',owner)
            print(ops.remote(f'''import pathlib,json
root=pathlib.Path({ops.ROOT!r})
with (root/'GPU_READY_1.json').open('x') as f:json.dump({owner!r},f)
print(json.dumps(dict(gpuR_ready=True)))
'''),flush=True)
            while True:
                state = claim()
                if state['done'] or state['stopped']:
                    break
                record = state['record']
                if record is None:
                    time.sleep(3)
                    continue
                print(json.dumps(dict(event='CLAIM',key=record['state_id'])),flush=True)
                process.stdin.write(json.dumps(record)+'\n');process.stdin.flush()
                for k in range(8):
                    message = next_message(process)
                    if (message['event'],message['key'],message['k']) != ('FUTURE',record['state_id'],k):
                        raise ValueError('gpuR future ordering/identity')
                    print(deliver(message),flush=True)
                message = next_message(process)
                if message['event'] != 'RESULT' or message['key'] != record['state_id']:
                    raise ValueError('gpuR response identity')
                print(deliver(message),flush=True)
            process.stdin.close()
            process.wait(timeout=60)
        except BaseException:
            failure = traceback.format_exc()
            ops.save('GPU_R_FAILURE.json',dict(error=failure,unix=time.time(),retry=False))
            print(ops.remote(f'''import pathlib,sys
root=pathlib.Path({ops.ROOT!r});sys.path.insert(0,str(root/'pytools'))
from tools.coopplan.storage import stop
stop(root,{failure!r})
print('durable STOP; no retry')
'''),flush=True)
            # Closing our own connection lets our stdio worker exit at EOF.
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            raise


def launch():
    assert (ops.LOCAL/'_out').is_dir()
    assert not (ops.OUT/'GPU_R_RELAY.json').exists()
    with (ops.LOCAL/'_out'/(TAG+'_relay.log')).open('x',encoding='utf-8') as log:
        p = subprocess.Popen([sys.executable,'-B','-u','-m','tools.wmplan.bridge'],
            cwd=ops.LOCAL,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
    receipt = dict(pid=p.pid,unix=time.time(),command='tools.wmplan.bridge')
    ops.save('GPU_R_RELAY.json',receipt)
    return receipt


if __name__ == '__main__':
    main()
