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
    parser.add_argument('action', choices=('inspect', 'stage', 'run', 'status', 'fetch', 'inputs', 'references'))
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
import os
r=Path({REMOTE!r}); print('load',os.getloadavg())
for n in ('STATUS.json','COMPLETE.json','CANARY.json','FAILURE.json'):
 p=r/n; print(n,p.read_text() if p.exists() else 'PENDING')
print('results',len(list((r/'jobs').glob('*/result.json'))))
for p in (r/'jobs').glob('*/failure.json'):
 print(str(p),p.read_text())
for n in ('canary.log','full.log'):
 p=r/'logs'/n
 if p.exists():print(n,p.read_text())
'''))
    else:
        dest = HERE/'evidence'
        assert dest.is_dir()
        for n in ('STATUS.json','COMPLETE.json','CANARY.json','PREFLIGHT_BINDING.json','accounting.jsonl'):
            print(scp(f'gpuR:{REMOTE}/{n}', dest/n))


if __name__ == '__main__':
    main()
