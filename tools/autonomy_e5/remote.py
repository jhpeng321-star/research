"""Optional E5 detached coordinator; explicit launch only after all preparation passes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R = Path('/root/autodl-tmp/c4/restart20261026/e5')
CPU = '/home/ubuntu/e3long'


def atomic(path, doc):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(path)


def ssh(box, source, timeout=120):
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', box,
                        '/home/ubuntu/miniconda3/bin/python -B -'], input=source,
                       text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f'{box}: {p.stderr[-4000:]} {p.stdout[-1000:]}')
    return json.loads(p.stdout.splitlines()[-1])


def pull(box):
    dest = R/box
    dest.mkdir(exist_ok=True)
    subprocess.run(['rsync', '-az', '--timeout=30', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
                    '--include', '*/', '--include', '*.json', '--include', '*.jsonl', '--exclude', '*',
                    box+':'+CPU+'/main/e3long_coopoff/', str(dest)+'/'], check=True, timeout=90)
    return dest


def start(box, shard, canary=False):
    return ssh(box, f"""import pathlib,json,sys,subprocess,os
r=pathlib.Path({CPU!r}); op=r/'operations_e5'; inputs=r/'inputs/coopoff_cycle1'
assert json.loads((op/'PREFLIGHT.json').read_text())['status']=='PREPARED_NOT_LAUNCHED'
out=r/'main/e3long_coopoff'/('shard'+str({shard})); op.mkdir(exist_ok=True); out.parent.mkdir(parents=True,exist_ok=True)
argv=[sys.executable,'-B','-u',str(op/'e5_runner.py'),'--inputs',str(inputs/'INPUTS.json'),
 '--main-go',str(inputs/'GO_E5.json'),'--out',str(out),'--shard',{str(shard)!r},'--workers','6','--phase','policy']
if {canary!r}: argv.append('--canary')
name='CANARY_PID.json' if {canary!r} else 'MAIN_PID.json'; assert not (op/name).exists()
with (op/(name+'.log')).open('xb') as log:
 p=subprocess.Popen(argv,cwd=r/'runtime',env={{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
  'MKL_NUM_THREADS':'1'}},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d={{'pid':p.pid,'start_ticks':pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().rsplit(')',1)[1].split()[19],
 'sid':os.getsid(p.pid),'output':str(out)}}
assert d['sid']==p.pid; (op/name).write_text(json.dumps(d)); print(json.dumps(d))
""")


def heartbeat(box):
    return ssh(box, f"""import pathlib,json,time
r=pathlib.Path({CPU!r}); op=r/'operations_e5'
(op/'MIRROR_HEARTBEAT.json').write_text(json.dumps({{'t':time.time()}}))
p=op/'RUNNER_STATUS.json'; print(p.read_text() if p.exists() else '{{}}')
""")


def analyze():
    import numpy as np
    on, off, cells = {}, {}, {}
    for box in ('cpuP', 'cpuQ'):
        data = ssh(box, f"""import json,pathlib
r=pathlib.Path({CPU!r}); shard={0 if box=='cpuP' else 1}
on=json.loads((r/'main/e3long_policy'/('shard'+str(shard))/'RESULTS.json').read_text())
off=json.loads((r/'main/e3long_coopoff'/('shard'+str(shard))/'RESULTS.json').read_text())
ledger=[json.loads(s) for s in (r/'main/e3long_coopoff'/('shard'+str(shard))/'budget.jsonl').read_text().splitlines()]
assert len(ledger)==114 and len({{x['id'] for x in ledger}})==114
print(json.dumps({{'on':on,'off':off,'charged_sim_s':sum(x['sim_s'] for x in ledger)}}))
""", timeout=180)
        assert data['charged_sim_s'] == 102600
        for label, dest in (('on', on), ('off', off)):
            for row in data[label]:
                seed = row['job']['seed']
                assert seed not in dest and row['ok']
                dest[seed] = row['metrics']
                cells[seed] = row['job']['cell']
    assert set(on) == set(off) == set(range(973000, 973228))
    assert all(on[s]['initial_state_sha256'] == off[s]['initial_state_sha256'] for s in on)
    for dest in (on, off):
        for metrics in dest.values():
            assert set(metrics['windows']) == {'120', '300', '600'}
            for horizon in (120, 300, 600):
                window = metrics['windows'][str(horizon)]
                assert window['t0_s'] == 300 and window['t_end_s'] == 300+horizon
                assert not window['dropped_vehicle_steps']
                assert abs(window['team_delay_vs']-window['road_delay_vs']-window['source_wait_vs']) < 1e-3
                assert abs(window['team_delay_vs']-window['mainline_delay_vs']-window['ramp_delay_vs']) < 1e-3
    rng = np.random.default_rng(20261027)
    groups = [sorted(s for s in on if cells[s] == c) for c in range(12)]
    assert all(len(g) == 19 for g in groups)
    samples = [rng.integers(0, 19, size=(5000, 19)) for _ in groups]
    rows = []
    for horizon in (120, 300, 600):
        for metric in ('team_delay_vs', 'road_delay_vs', 'source_wait_vs', 'throughput_vph',
                       'mainline_delay_vs', 'ramp_delay_vs', 'collisions', 'teleports'):
            values = [np.array([on[s]['windows'][str(horizon)][metric]-
                                off[s]['windows'][str(horizon)][metric] for s in g]) for g in groups]
            assert all(np.isfinite(v).all() for v in values)
            boot = np.mean([v[i].mean(1) for v, i in zip(values, samples)], axis=0)
            rows.append({'horizon_s': horizon, 'metric': metric, 'n_worlds': 228,
                         'contrast': 'ours-coopON(cycle1) - ours-coopOFF(cycle1)',
                         'mean': float(np.mean(values)), 'ci95': np.quantile(boot, [.025, .975]).tolist(),
                         'bootstrap': '5000 world-cluster resamples stratified by12cells; fixed seed20261027'})
    atomic(R/'RESULTS.json', rows)
    for box in ('cpuP', 'cpuQ'):
        ssh(box, f"import pathlib,json; p=pathlib.Path({CPU+'/operations_e5/RESULTS.json'!r}); p.write_text(json.dumps({rows!r},indent=2)); print('{{}}')")


def drive():
    try:
        for box in ('cpuP', 'cpuQ'):
            heartbeat(box)
        canary = start('cpuP', 0, True)
        atomic(R/'CANARY_PID.json', canary)
        while True:
            state = heartbeat('cpuP')
            pull('cpuP')
            if state.get('status') == 'failed':
                raise RuntimeError('canary failed; no remainder')
            if state.get('status') == 'canary_complete':
                break
            time.sleep(15)
        proof = ssh('cpuP', f"""import json,pathlib
r=pathlib.Path({CPU!r}); out=r/'main/e3long_coopoff/shard0'
row=json.loads((out/'RESULTS.json').read_text())[0]; assert row['ok'] and row['job']['seed']==973000
base=json.loads((r/'main/e3long_baseline/shard0/jobs/s973000_native/result.json').read_text())
assert row['metrics']['initial_state_sha256']==base['metrics']['initial_state_sha256']
assert set(row['metrics']['windows'])=={{'120','300','600'}}
proof={{'passed':True,'canary_seed':973000,'charged_sim_s':900}}
(r/'operations_e5/CANARY_VERIFIED.json').write_text(json.dumps(proof)); print(json.dumps(proof))
""")
        ssh('cpuQ', f"import pathlib,json; pathlib.Path({CPU+'/operations_e5/CANARY_VERIFIED.json'!r}).write_text(json.dumps({proof!r})); print('{{}}')")
        pids = {box: start(box, shard) for shard, box in enumerate(('cpuP', 'cpuQ'))}
        atomic(R/'CHILDREN.json', pids)
        while True:
            states = {}
            for box in ('cpuP', 'cpuQ'):
                states[box] = heartbeat(box)
                pull(box)
            atomic(R/'STATUS.json', {'t': time.time(), 'status': 'RUNNING', 'boxes': states})
            if any(s.get('status') == 'failed' for s in states.values()):
                raise RuntimeError('scientific/operational job failed; no simulation retry')
            if all(s.get('status') == 'complete' for s in states.values()):
                break
            time.sleep(15)
        analyze()
        atomic(R/'STATUS.json', {'status': 'COMPLETE', 'charged_sim_s': 205200, 'results': str(R/'RESULTS.json')})
    except Exception as error:
        atomic(R/'STATUS.json', {'status': 'WAIT_OWNER', 'error': repr(error)})
        raise


def launch():
    assert not (R/'PID.json').exists()
    # All readiness proof must exist on both boxes BEFORE any detached child is started.
    for box in ('cpuP', 'cpuQ'):
        doc = ssh(box, f"""import pathlib,json,hashlib,runpy,sys,os
r=pathlib.Path({CPU!r}); op=r/'operations_e5'; inputs=r/'inputs/coopoff_cycle1'
sys.path[:0]=[str(r/'runtime'),str(r/'runtime/src'),str(r/'operations')]; os.chdir(r/'runtime')
proof=json.loads((op/'PREFLIGHT.json').read_text())
assert hashlib.sha256((op/'e5_runner.py').read_bytes()).hexdigest()==proof['runner_sha256']
assert hashlib.sha256((inputs/'GO_E5.json').read_bytes()).hexdigest()==proof['go_sha256']
ns=runpy.run_path(str(op/'e5_runner.py'))
ns['validate'](json.loads((inputs/'INPUTS.json').read_text()),json.loads((inputs/'GO_E5.json').read_text()))
assert not (op/'MAIN_PID.json').exists() and not (op/'CANARY_PID.json').exists()
print(json.dumps(proof))
""")
        assert doc['status'] == 'PREPARED_NOT_LAUNCHED' and doc['charged_sim_s'] == 0
    with (R/'driver.log').open('xb') as log:
        p = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__)), 'drive'],
                             start_new_session=True, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT)
    doc = {'pid': p.pid, 'sid': os.getsid(p.pid),
           'start_ticks': Path(f'/proc/{p.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]}
    assert doc['sid'] == p.pid
    atomic(R/'PID.json', doc)
    print(json.dumps(doc))


if __name__ == '__main__':
    assert sys.argv[1] in ('drive', 'launch')
    globals()[sys.argv[1]]()
