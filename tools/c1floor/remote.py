"""Small explicit SSH transport; remote reads use named, screened paths."""
from pathlib import Path
import argparse
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = '/root/c1floor20261031'
DENIED = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
          'hetdata_truth', 'vtype')


def ssh(code, host='gpuR', python='/root/miniconda3/bin/python', timeout=180):
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                        host, python+' -B -'], input=code, text=True,
                       encoding='utf-8', capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stdout+p.stderr)
    return p.stdout


def scp(source, target):
    p = subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                        str(source), str(target)], capture_output=True, text=True,
                       encoding='utf-8', timeout=600)
    if p.returncode:
        raise RuntimeError(p.stdout+p.stderr)
    return p.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('inspect', 'stage', 'run', 'status', 'fetch', 'inputs', 'references', 'capacity-rearm', 'bundle', 'extract', 'setup', 'diagnose', 'stop-report', 'pin-metrics'))
    parser.add_argument('--phase', default='tests')
    a = parser.parse_args()
    if a.action == 'inspect':
        print(ssh('''import os,json,subprocess,time
from pathlib import Path
print('load',os.getloadavg())
print('cpus',os.cpu_count(),'affinity',len(os.sched_getaffinity(0)))
for n in ('cpu.max','cpu.stat'):
 p=Path('/sys/fs/cgroup')/n;print(n,p.read_text() if p.exists() else 'absent')
p=Path('/sys/fs/cgroup/cpu.stat')
if p.exists():
 def usage():return int(dict(s.split() for s in p.read_text().splitlines())['usage_usec'])
 before=usage(); t=time.monotonic(); time.sleep(5); print('actual_busy_cpus_5s',(usage()-before)/1e6/(time.monotonic()-t))
print('SUMO_OPTIONS', '\\n'.join(s for s in subprocess.check_output(['/root/miniconda3/bin/sumo','--help'],text=True).splitlines() if 'rng' in s.lower() or 'precision' in s.lower()))
for s in ('/root/wmcalib/data','/root/sumocf20261031','/root/coopaudit20261031'):
 p=Path(s); print(s, sorted(q.name for q in p.iterdir() if not any(x in q.name.lower() for x in '''+repr(DENIED)+''')) if p.is_dir() else 'MISSING')
for s in ('/root/sumocf20261031/STATUS.json','/root/sumocf20261031/SUPERVISOR.json'):
 p=Path(s); print(s,p.read_text() if p.exists() else 'PENDING')
'''))
    elif a.action == 'capacity-rearm':
        print(ssh(f'''from pathlib import Path
import os,json,signal,time
r=Path({REMOTE!r}); rows=[json.loads(s) for s in (r/'accounting.jsonl').read_text().splitlines()]
assert len(rows)==1 and rows[0]['id']=='s969400_native'
p=Path('/proc/508011'); assert p.is_dir()
assert p.joinpath('stat').read_text().split()[21]=='4206272443'
assert b'/root/c1floor20261031/run.py' in p.joinpath('cmdline').read_bytes()
assert not (r/'jobs/s969400_native_attempt2').exists()
os.kill(508011,signal.SIGTERM)
time.sleep(1)
print('own queued coordinator terminated before precharge; no SUMO children dispatched')
'''))
    elif a.action == 'pin-metrics':
        source=Path('E:/agent-work/c4-c1closed/tools/c1closed/analyze.py')
        assert (HERE/'evidence').is_dir()
        print(scp(source,f'gpuR:{REMOTE}/C1CLOSED_analyze_source.py'))
        print(scp(f'gpuR:{REMOTE}/C1CLOSED_analyze_source.py',HERE/'evidence/C1CLOSED_analyze_source.py'))
    elif a.action == 'stop-report':
        print(ssh(f'''from pathlib import Path
import os,json,hashlib
r=Path({REMOTE!r})
stop=r/'STOP'
if not stop.exists():
 with stop.open('x') as f:f.write('BLOCKED: same-seed saved-state canary drift; no alternative-seed or panel dispatch.\\n');f.flush();os.fsync(f.fileno())
rows=[json.loads(s) for s in (r/'accounting.jsonl').read_text().splitlines()]
failures=[json.loads(p.read_text()) for p in (r/'jobs').glob('*/failure.json')]
actual=sum(x['actual_sim_s'] for x in failures)
assert len(rows)==6 and sum(x['sim_s'] for x in rows)==9000 and actual==2400
assert not list((r/'jobs').glob('*/alt*.npz'))
owners=[json.loads(s) for s in (r/'PIDS.jsonl').read_text().splitlines()]
states=[]
for owner in owners:
 for k in ('worker_pid','coordinator_pid'):
  p=Path('/proc')/str(owner[k]);state=p.joinpath('stat').read_text().split()[2] if p.exists() else 'exited'
  assert state in ('exited','Z'),(owner[k],state)
  states.append(dict(pid=owner[k],state=state))
last=r/'jobs/s969400_native_attempt6'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=dict(status='BLOCKED_SAVED_STATE_CANARY',actual_sim_s=actual,charged_sim_s=9000,
 cap_actual_sim_s=30000,alternative_branches_run=0,panel_worlds_run=0,first_drift_time_s=300.5,
 first_drift_hdv_ids=[101,104],recorded_rows=45631,restored_rows=45227,
 original_reproduction=json.loads((last/'REPRODUCTION.json').read_text()),
 original_tape_sha256=sha(last/'traj_base.npz'),restored_tape_sha256=sha(last/'same_seed.npz'),
 saved_state_sha256=sha(last/'start.xml'),processes=states)
with (r/'BLOCKED.json').open('x') as f:json.dump(receipt,f,indent=1);f.flush();os.fsync(f.fileno())
(r/'STATUS.json').write_text(json.dumps(receipt,indent=1))
print(json.dumps(receipt,indent=1))
'''))
    elif a.action == 'diagnose':
        print(ssh(f'''from pathlib import Path
import numpy as np,json
r=Path({REMOTE!r})/'jobs/s969400_native_attempt6'
with np.load(r/'traj_base.npz',allow_pickle=False) as a,np.load(r/'same_seed.npz',allow_pickle=False) as b:
 meta=json.loads(str(a['meta']));cavs=set(meta['cavs'])
 ia=np.flatnonzero(a['t']<=600);ib=np.flatnonzero(b['t']<=600)
 ia=ia[np.lexsort((a['vid'][ia],a['t'][ia]))];ib=ib[np.lexsort((b['vid'][ib],b['t'][ib]))]
 print('rows',len(ia),len(ib))
 for t in sorted(set(a['t'])|set(b['t'])):
  aa={{int(a['vid'][j]):j for j in np.flatnonzero(a['t']==t)}}
  bb={{int(b['vid'][j]):j for j in np.flatnonzero(b['t']==t)}}
  missing=set(aa)^set(bb)
  bad=[v for v in aa.keys()&bb.keys() if any(a[k][aa[v]]!=b[k][bb[v]] for k in ('s','v','a','lane'))]
  if missing or bad:
   print('FIRST_DIVERGENCE',float(t),'missing',sorted(missing),'changed',len(bad))
   for v in sorted(bad)[:3]:print('observable',v,'is_cav',meta['vids'][v] in cavs,{{k:[float(a[k][aa[v]]),float(b[k][bb[v]])] for k in ('s','v','a','lane')}})
   break
 if len(ia)==len(ib):
  for k in ('t','vid','s','v','a','lane'):
   x,y=a[k][ia],b[k][ib];bad=np.flatnonzero(x!=y)
   print(k,'dtype',str(x.dtype),str(y.dtype),'unequal',len(bad),'max',float(np.max(np.abs(x.astype(float)-y))))
   if len(bad):
    j=int(bad[0]);print('first',float(a['t'][ia[j]]),int(a['vid'][ia[j]]),float(x[j]),float(y[j]))
'''))
    elif a.action == 'inputs':
        print(ssh('''from pathlib import Path
import json
p=Path('/root/wmcalib/data'); d=json.loads((p/'design.json').read_text())
print(json.dumps(d,indent=1))
q=p/'jobs/s969400_native'; print('files',sorted(x.name for x in q.iterdir()))
'''))
    elif a.action == 'stage':
        print(ssh(f'''from pathlib import Path
p=Path({REMOTE!r}); assert p.parent.is_dir(); p.mkdir(exist_ok=True)
for name in ('evidence','jobs','reference','logs'):
 (p/name).mkdir(exist_ok=True)
print(p)
'''))
        for name in ('worker.py', 'run.py', 'core.py', 'test_core.py', 'labels.py', 'design.json'):
            p = HERE/name
            if p.exists():
                print(scp(p, f'gpuR:{REMOTE}/{name}'))
        print(scp('E:/agent-work/c4-c1closed/tools/c1closed/labels.py',f'gpuR:{REMOTE}/labels.py'))
        print(scp('E:/agent-work/c4-c1closed/tools/c1closed/aggregates.py',f'gpuR:{REMOTE}/aggregates.py'))
        print(scp('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md',f'gpuR:{REMOTE}/REGISTRATION_INPUT.md'))
    elif a.action == 'references':
        print(ssh(f'''from pathlib import Path
p=Path({REMOTE!r})/'reference'; assert p.is_dir()
for s in range(969400,969424):
 (p/f's{{s}}_native').mkdir(exist_ok=True)
'''))
        # Remote-to-remote via the local transport: no source-machine allocation.
        for seed in range(969400,969424):
            for n in ('result.json','traj_base.npz'):
                p=subprocess.run(['scp','-3','-o','BatchMode=yes','-o','ConnectTimeout=15',
                   f'cpuQ:/home/ubuntu/wmdata20261031/jobs/s{seed}_native/{n}',
                   f'gpuR:{REMOTE}/reference/s{seed}_native/{n}'],capture_output=True,text=True,timeout=180)
                if p.returncode:raise RuntimeError(p.stdout+p.stderr)
            print('reference',seed,flush=True)
    elif a.action in ('extract','setup'):
        script=a.action+'.py'
        print(scp(HERE/script,f'gpuR:{REMOTE}/{script}'))
        print(ssh(f'''import subprocess,os
p=subprocess.run(['nice','-n','19','/root/miniconda3/bin/python','-B',{REMOTE!r}+'/'+{script!r}],text=True,capture_output=True,env={{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PATH':'/root/miniconda3/bin:'+os.environ['PATH']}})
print(p.stdout);print(p.stderr);raise SystemExit(p.returncode)
''',timeout=240))
    elif a.action == 'bundle':
        import tarfile
        import time
        name=f'evidence_{time.time_ns()}.tgz'
        print(ssh(f'''from pathlib import Path
import tarfile
r=Path({REMOTE!r})
files=[p for p in r.glob('*.json') if p.name not in ('LABELS.json',)]
files += list(r.glob('*.jsonl'))+list((r/'jobs').glob('*/*.json'))+list((r/'logs').glob('*.log'))
with tarfile.open(r/'evidence'/{name!r},'w:gz') as tar:
 for p in files:tar.add(p,arcname=str(p.relative_to(r)),recursive=False)
print('files',len(files))
'''))
        dest=HERE/'evidence';assert dest.is_dir()
        print(scp(f'gpuR:{REMOTE}/evidence/{name}',dest/name))
        target=dest/name.removesuffix('.tgz');target.mkdir()
        with tarfile.open(dest/name,'r:gz') as tar:
            for member in tar.getmembers():
                p=Path(member.name)
                assert not p.is_absolute() and '..' not in p.parts and member.isfile()
                assert not any(x in str(p).lower() for x in DENIED)
            tar.extractall(target,filter='data')
        print('saved',target)
    elif a.action == 'run':
        phase = a.phase
        assert phase in ('tests', 'preflight', 'canary', 'full')
        print(ssh(f'''import os,subprocess,json
from pathlib import Path
r=Path({REMOTE!r}); assert r.is_dir()
env={{**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PATH':'/root/miniconda3/bin:'+os.environ['PATH']}}
cmd=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(r/'run.py'),{phase!r}]
if {phase!r} in ('full','canary'):
 import time
 with (r/'logs'/({phase!r}+'_'+str(time.time_ns())+'.log')).open('x') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 print(json.dumps(dict(pid=p.pid,start_ticks=Path(f'/proc/{{p.pid}}/stat').read_text().split()[21])))
else:
 p=subprocess.run(cmd,env=env,text=True,capture_output=True)
 print(p.stdout); print(p.stderr); raise SystemExit(p.returncode)
''', timeout=240))
    elif a.action == 'status':
        print(ssh(f'''from pathlib import Path
import os,json
r=Path({REMOTE!r}); print('load',os.getloadavg())
for n in ('STATUS.json','COMPLETE.json','CANARY.json'):
 p=r/n; print(n,p.read_text() if p.exists() else 'PENDING')
print('results',len(list((r/'jobs').glob('*/result.json'))))
for p in (r/'jobs').glob('*/failure.json'):
 if (r/'STATUS.json').exists() and p.parent.name==json.loads((r/'STATUS.json').read_text())['job']:
  print(str(p),p.read_text())
for phase in ('canary','full'):
 logs=list((r/'logs').glob(phase+'_*.log'))
 if logs:
  p=max(logs,key=lambda p:p.name); print(p.name,p.read_text())
'''))
    else:
        dest = HERE/'evidence'
        assert dest.is_dir()
        for n in ('STATUS.json','COMPLETE.json','CANARY.json','PREFLIGHT_BINDING.json','accounting.jsonl'):
            print(scp(f'gpuR:{REMOTE}/{n}', dest/n))


if __name__ == '__main__':
    main()
