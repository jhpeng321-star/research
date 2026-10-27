"""S-E5FIX detached coordinator: adopt cpuP, repair cpuQ, enforce retained charges."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R = Path('/root/autodl-tmp/c4/restart20261026/e5')
CPU = '/home/ubuntu/e3long'
if __package__:
    from . import remote as B
else:
    import remote as B


def cpu(box, source):
    return B.ssh(box, source, timeout=180)


def snapshot(box):
    return cpu(box, f'''import pathlib,json,time
r=pathlib.Path({CPU!r}); op=r/'operations_e5'; shard={0 if box == 'cpuP' else 1}
out=r/'main/e3long_coopoff'/('shard'+str(shard))
name='RUNNER_STATUS.json' if shard==0 else 'FIX_RUNNER_STATUS.json'
state=json.loads((op/name).read_text()) if (op/name).exists() else {{}}
charges=[json.loads(s) for s in (out/'budget.jsonl').read_text().splitlines()]
rows=json.loads((out/'RESULTS.json').read_text()) if (out/'RESULTS.json').exists() else []
assert len(charges)==len({{r['id'] for r in charges}})
charged=sum(r['sim_s'] for r in charges); assert charged<=102600
print(json.dumps(dict(state=state,charged_sim_s=charged,n_results=len(rows),
 seeds=sorted(r['job']['seed'] for r in rows),
 pause=json.loads((op/'PAUSE.json').read_text()) if (op/'PAUSE.json').exists() else {{}})))
''')


def start(mode):
    return cpu('cpuQ', f'''import pathlib,json,subprocess,os
r=pathlib.Path({CPU!r}); op=r/'operations_e5'
name='FIX_'+{mode.upper()!r}+'_PID.json'; assert not (op/name).exists()
home='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo'
path=home+'/bin:/home/ubuntu/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin'
with (op/(name+'.log')).open('xb') as log:
 p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u',str(op/'fix_runner.py'),{mode!r}],
  cwd=r/'runtime',env={{**os.environ,'SUMO_HOME':home,'PATH':path,'OMP_NUM_THREADS':'1',
  'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}},
  start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
d=dict(pid=p.pid,sid=os.getsid(p.pid),start_ticks=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19])
assert d['sid']==p.pid
(op/name).write_text(json.dumps(d)); print(json.dumps(d))
''')


def cycle(phase, refresh_p=True):
    boxes = {}
    for box in ('cpuP', 'cpuQ'):
        if box == 'cpuQ' or refresh_p:
            B.heartbeat(box)
        B.pull(box)
        boxes[box] = snapshot(box)
    total = sum(d['charged_sim_s'] for d in boxes.values())
    assert total <= 205200 and total-22500 <= 182700
    doc = dict(status=phase, t=time.time(), boxes=boxes, charged_sim_s=total,
               new_charged_sim_s=total-22500, remaining_sim_s=205200-total)
    B.atomic(R/'FIX_STATUS.json', doc)
    assert not any(d['state'].get('status') == 'failed' for d in boxes.values()), boxes
    return doc


def drive():
    try:
        # cpuP stays stale-paused until the repaired cpuQ canary has passed.
        canary = start('canary')
        B.atomic(R/'FIX_CHILDREN.json', dict(cpuQ_canary=canary))
        while True:
            doc = cycle('CPUQ_CANARY', refresh_p=False)
            if doc['boxes']['cpuQ']['state'].get('status') == 'canary_complete':
                break
            time.sleep(15)
        proof = cpu('cpuQ', f"import pathlib; print(pathlib.Path({CPU+'/operations_e5/FIX_CANARY_VERIFIED.json'!r}).read_text())")
        assert proof['passed'] and proof['seed'] == 973001
        B.atomic(R/'FIX_CANARY_VERIFIED.json', proof)
        child = start('remainder')
        owned_p = cpu('cpuP', f"import pathlib; print(pathlib.Path({CPU+'/operations_e5/MAIN_PID.json'!r}).read_text())")
        B.atomic(R/'FIX_CHILDREN.json', dict(cpuP_adopted=owned_p, cpuQ_canary=canary, cpuQ=child))
        while True:
            doc = cycle('RUNNING')
            states = [d['state'].get('status') for d in doc['boxes'].values()]
            if all(s in ('complete', 'WAIT_OWNER_BUDGET') for s in states):
                break
            time.sleep(15)
        seeds = {s for d in doc['boxes'].values() for s in d['seeds']}
        missing = sorted(set(range(973000, 973228))-seeds)
        if missing:
            assert doc['charged_sim_s'] == 205200
            doc.update(status='WAIT_OWNER_BUDGET', missing_worlds=missing,
                       results_unavailable=True, additional_sim_s_required=900*len(missing))
            B.atomic(R/'FIX_BUDGET_RECEIPT.json', doc)
        else:
            B.analyze()
            doc.update(status='COMPLETE', results=str(R/'RESULTS.json'))
        B.atomic(R/'FIX_STATUS.json', doc)
        for box in ('cpuP', 'cpuQ'):
            cpu(box, f"import pathlib,json; pathlib.Path({CPU+'/operations_e5/FIX_FINAL.json'!r}).write_text(json.dumps({doc!r},indent=2)); print('{{}}')")
    except BaseException as error:
        B.atomic(R/'FIX_STATUS.json', dict(status='WAIT_OWNER', error=repr(error), t=time.time()))
        raise


def launch():
    assert not (R/'FIX_PID.json').exists(), 'single authorized repair coordinator'
    old = json.loads((R/'PID.json').read_text())
    proc = Path('/proc')/str(old['pid'])
    if proc.exists():
        stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        assert stat[19] != old['start_ticks'] or stat[0] == 'Z', 'old owned coordinator live'
    assert json.loads((R/'STATUS.json').read_text())['status'] == 'WAIT_OWNER'
    for box in ('cpuP', 'cpuQ'):
        assert snapshot(box)['charged_sim_s'] == (17100 if box == 'cpuP' else 5400)
    proof = cpu('cpuQ', f"import pathlib; print(pathlib.Path({CPU+'/operations_e5/FIX_PREFLIGHT.json'!r}).read_text())")
    assert proof['status'] == 'PASS' and proof['new_charged_sim_s'] == 0
    with (R/'fix_driver.log').open('xb') as log:
        p = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__)), 'drive'],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    doc = dict(pid=p.pid, sid=os.getsid(p.pid),
               start_ticks=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    assert doc['sid'] == p.pid
    B.atomic(R/'FIX_PID.json', doc)
    print(json.dumps(doc))


if __name__ == '__main__':
    assert sys.argv[1] in ('launch', 'drive')
    globals()[sys.argv[1]]()
