"""Explicit recovery of resource-stopped tasks; retain frozen scientific code."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=1)
        f.flush()
        os.fsync(f.fileno())


def prepare():
    from expand import host_state
    from operate import REMOTE
    from probe import ssh
    from watch import creation, digest
    states = {h: host_state(h) for h in ('gpuA', 'gpuR')}
    assert all(not s['alive'] and s['complete'] is not None for s in states.values())
    assert states['gpuA']['complete']['success']
    bridge = json.loads((HERE/'evidence/EXPAND_BRIDGE_LAUNCH.json').read_text())
    assert creation(bridge['pid']) != bridge['creation_time']
    collector = json.loads((HERE/'evidence/WATCH_LAUNCH.json').read_text())
    assert creation(collector['pid']) == collector['creation_time']
    assert all(digest(HERE/n) == v for n, v in collector['source_hashes'].items())
    done = set().union(*(set(s['outputs']) for s in states.values()))
    roster = {f'{m}_{p}_w{w}_k{k}_h300.json': [m, w, k, p]
              for w in range(40)
              for m, k, p in ([(m, k, 'common') for m in ('wm2', 'bbtraj', 'bbid', 'historical')
                               for k in range(8 if m == 'wm2' else 1)] + [('wm2', 0, 'production')])}
    failed = [x for x in states['gpuR']['complete']['results'] if x['status'] != 'complete']
    assert all(x['returncode'] == -15 and x['memory_guard_stop'] for x in failed)
    missing = sorted(set(roster)-done)
    assert set(missing) == {x['task']+'.json' for x in failed}
    assert len(done) == 467 and len(missing) == 13
    audits = {}
    for host in ('gpuA', 'gpuR'):
        audits[host] = json.loads(ssh(host, f'''from pathlib import Path
import json,sys,subprocess
sys.path.insert(0,{REMOTE!r});import expand_runner as r
p=Path({REMOTE!r});live=[]
for q in (p/'evidence').glob('*_OWNED.json'):
 x=json.loads(q.read_text());stat=Path('/proc/'+str(x['pid'])+'/stat')
 if stat.exists():
  s=stat.read_text().split(') ',1)[1].split()
  if s[19]==x['start_ticks'] and s[0]!='Z':live.append(dict(pid=x['pid'],task=x.get('task'),state=s[0]))
print(json.dumps(dict(live_workers=live,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.free','--format=csv,noheader'],text=True),apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True))))
'''))
        assert not audits[host]['live_workers']
    plan = dict(at=time.time(),completed_preserved=len(done),missing=missing,
                tasks=[roster[n] for n in missing],failed_attempts=failed,audits=audits,
                host='gpuR',gpu_lanes=2,budget_bytes=8_000_000_000,
                worker_sha256=digest(HERE/'worker.py'),collector=collector,
                runner_sha256=digest(__file__),bridge_dead=bridge,
                reason='Original guard-stopped tasks only; two lanes, serialized admission; no automatic retries or scientific changes')
    write(HERE/'evidence/TAIL_RECOVERY_PLAN.json', plan)
    print(json.dumps({k: plan[k] for k in ('completed_preserved','missing','gpu_lanes','audits')}, indent=1))


def launch():
    from operate import REMOTE, scp
    from probe import ssh
    plan = json.loads((HERE/'evidence/TAIL_RECOVERY_PLAN.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['runner_sha256']
    print(ssh('gpuR', f'''from pathlib import Path
p=Path({REMOTE!r});assert p.is_dir()
assert not (p/'TAIL_RECOVERY_PLAN.json').exists()
assert not (p/'recover_tail.py').exists()
print('verified absent recovery destinations')
'''))
    print(scp(__file__, f'gpuR:{REMOTE}/recover_tail.py'))
    print(scp(HERE/'evidence/TAIL_RECOVERY_PLAN.json', f'gpuR:{REMOTE}/TAIL_RECOVERY_PLAN.json'))
    receipt = json.loads(ssh('gpuR', f'''from pathlib import Path
import os,subprocess,json,time
p=Path({REMOTE!r});assert not (p/'evidence/TAIL_RECOVERY_LAUNCH.json').exists()
with (p/'evidence/TAIL_RECOVERY.log').open('x') as log:
 child=subprocess.Popen(['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(p/'recover_tail.py'),'run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}})
 r=dict(pid=child.pid,start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split(') ',1)[1].split()[19],started=time.time())
 with (p/'evidence/TAIL_RECOVERY_LAUNCH.json').open('x') as f:json.dump(r,f)
print(json.dumps(r))
'''))
    write(HERE/'evidence/TAIL_RECOVERY_LAUNCH.json', receipt)
    print(json.dumps(receipt))


def run():
    import fcntl
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import expand_runner as original
    root = HERE
    lock = (root/'EXPAND.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((root/'TAIL_RECOVERY_PLAN.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['runner_sha256']
    assert hashlib.sha256((root/'worker.py').read_bytes()).hexdigest() == plan['worker_sha256']
    assert not (root/'TAIL_RECOVERY_COMPLETE.json').exists()
    admission = threading.Lock()

    def one(task):
        model, world, future, plant = task
        name = f'{model}_{plant}_w{world}_k{future}_h300'
        output = root/'panel'/(name+'.json')
        assert not output.exists()
        # Serialize admission and allow CUDA accounting to settle before admitting
        # the second process. The frozen per-process allocator cap is 3 GiB.
        with admission:
            while True:
                free, total = original.memory()
                if free >= 5120 and total*2**20 + 4*2**30 <= plan['budget_bytes'] and os.getloadavg()[0] < 80:
                    break
                time.sleep(10)
            cmd = [sys.executable,'-B','-u',str(root/'worker.py'),'--model',model,
                   '--world',str(world),'--future',str(future),'--plant',plant,
                   '--seconds','300','--out','panel','--device','cuda']
            with (root/'evidence'/f'EXPAND_RETRY_{name}.log').open('x') as log:
                child = subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONPATH':'/root/autodl-tmp/c4/speedhead20261030/pythondeps',
                         'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
            ticks = original.incarnation(child.pid)
            owned = dict(pid=child.pid,start_ticks=ticks,device='cuda',task=name,command=cmd,started=time.time())
            write(root/'evidence'/f'EXPAND_RETRY_{name}_OWNED.json',owned)
            time.sleep(10)
        peak = 0
        guard = False
        while child.poll() is None:
            free, total = original.memory()
            peak = max(peak, total)
            if total*2**20 > plan['budget_bytes'] or free < 1024:
                guard = True
                if child.poll() is None and original.incarnation(child.pid) == ticks:
                    child.terminate()
                    child.wait(timeout=120)
                break
            time.sleep(2)
        code = child.wait()
        result = dict(**owned,returncode=code,finished=time.time(),peak_host_owned_mib=peak,
                      memory_guard_stop=guard,status='complete' if code == 0 and output.exists() else 'failed')
        if result['status'] == 'complete':
            x = json.loads(output.read_text())
            assert (x['model'],x['world'],x['future'],x['plant'],x['steps']) == (model,world,future,plant,600)
            assert x['identity']['source_sha256'] == plan['worker_sha256']
            result['output_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
        write(root/'evidence'/f'EXPAND_RETRY_{name}_attempt2.json',result)
        print(json.dumps(result),flush=True)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(one,t) for t in plan['tasks']]
        for f in as_completed(futures):
            results.append(f.result())
    write(root/'TAIL_RECOVERY_COMPLETE.json',dict(success=all(x['status']=='complete' for x in results),results=results,finished=time.time()))


def status():
    from operate import REMOTE
    from probe import ssh
    print(ssh('gpuR', f'''from pathlib import Path
import json,sys,os
sys.path.insert(0,{REMOTE!r});import expand_runner as r
p=Path({REMOTE!r});launch=json.loads((p/'evidence/TAIL_RECOVERY_LAUNCH.json').read_text())
attempts=[json.loads(q.read_text()) for q in (p/'evidence').glob('EXPAND_RETRY_*_attempt2.json')]
done=p/'TAIL_RECOVERY_COMPLETE.json'
print(json.dumps(dict(alive=r.incarnation(launch['pid'])==launch['start_ticks'],complete=json.loads(done.read_text()) if done.exists() else None,attempts=attempts,memory=r.memory(),load=os.getloadavg())))
'''))


def publish():
    from operate import REMOTE, scp
    from probe import ssh
    from expand import host_state, publish as original_publish, finish
    from watch import creation
    collector = json.loads((HERE/'evidence/WATCH_LAUNCH.json').read_text())
    assert creation(collector['pid']) == collector['creation_time']
    print(scp(f'gpuR:{REMOTE}/TAIL_RECOVERY_COMPLETE.json', HERE/'evidence/TAIL_RECOVERY_COMPLETE.json'))
    receipt = json.loads((HERE/'evidence/TAIL_RECOVERY_COMPLETE.json').read_text())
    plan = json.loads((HERE/'evidence/TAIL_RECOVERY_PLAN.json').read_text())
    assert receipt['success'] and len(receipt['results']) == len(plan['tasks'])
    assert {x['task']+'.json' for x in receipt['results']} == set(plan['missing'])
    states = {h: host_state(h) for h in ('gpuA', 'gpuR')}
    assert all(not s['alive'] for s in states.values())
    for x in receipt['results']:
        assert x['returncode'] == 0 and not x['memory_guard_stop']
        states['gpuR']['outputs'][x['task']+'.json'] = x['output_sha256']
    original_publish(states)
    assert len(host_state('gpuA')['outputs']) == 480
    finish()
    print('All 480 outputs published; existing sole collector retains finalization.')


if __name__ == '__main__':
    actions = dict(prepare=prepare,launch=launch,run=run,status=status,publish=publish)
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=actions)
    actions[parser.parse_args().action]()
