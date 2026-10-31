"""Named-path, recorded-state-only operations for S-C1PIDIAG."""
from pathlib import Path
import argparse
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/root/autodl-tmp/c4/c1pidiag20261031'
BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')


def allowed(value):
    if any(x in str(value).lower() for x in BAD):
        raise ValueError('excluded path or identity')
    return value


def ssh(code, timeout=120):
    import time
    for attempt in range(3):
        p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                            'gpuA', '/root/miniconda3/bin/python -B -'], input=code,
                           text=True, encoding='utf-8', capture_output=True, timeout=timeout)
        if p.returncode == 0 or 'Traceback' in p.stderr:
            break
        time.sleep(3 * (attempt + 1))
    if p.returncode:
        raise RuntimeError(p.stdout + p.stderr)
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('select', 'probe', 'read', 'list', 'inspect', 'stage', 'smoke', 'launch', 'status', 'fetch'))
    ap.add_argument('--path')
    args = ap.parse_args()
    if args.action == 'select':
        path = Path('E:/agent-work/c4-c1refresh/_out/c1refresh/INDEX.json')
        index = json.loads(path.read_text())
        rows = sorted(index['states'].values(), key=lambda e: (e['world'], e['t0_s'], e['sid']))
        panel = []
        for cohort, n, arm in [('pi_u_same', 40, 'pi'), ('wall_same', 30, 'wall')]:
            eligible = [e for e in rows if cohort in e['sets'] and
                        not any(b in str(e).lower() for b in BAD)]
            for i, e in enumerate(eligible[:n]):
                panel.append(dict(cohort=cohort, index=i, arm=arm, **e))
        import hashlib
        out = ROOT / 'docs/paper_results/c1pidiag'
        out.mkdir(parents=True, exist_ok=True)
        with (out/'PANEL.json').open('x') as f:
            json.dump(dict(rule='first 40 PI + first 30 wall sorted world,t0,sid; denied identities excluded',
                           index_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), states=panel), f, indent=2)
        print('fixed panel',len(panel), 'worlds',len({e['world'] for e in panel}))
    elif args.action in ('smoke', 'launch'):
        script = 'worker.py' if args.action == 'smoke' else 'dispatch.py'
        extra = ['--smoke'] if args.action == 'smoke' else []
        label = args.action.upper()+'2'
        print(ssh(f'''from pathlib import Path
import os,subprocess,json,time
p=Path({REMOTE!r});assert p.is_dir();assert os.getloadavg()[0] < 100
target=p/'evidence_v2/{label}_OWNED.json';assert not target.exists(), 'already launched'
cmd=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(p/{script!r})]+{extra!r}
env={{**os.environ,'PYTHONPATH':'/root/autodl-tmp/c4/speedhead20261030/pythondeps','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}}
with (p/'evidence_v2/{label}.log').open('x') as log:
 child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
 receipt=dict(pid=child.pid,start_ticks=Path(f'/proc/{{child.pid}}/stat').read_text().split(') ',1)[1].split()[19],command=cmd,load=os.getloadavg(),started=time.time())
 with target.open('x') as f:json.dump(receipt,f)
 print(json.dumps(receipt))
'''))
    elif args.action == 'status':
        print(ssh(f'''from pathlib import Path
import json,os
p=Path({REMOTE!r})
for name in ('SMOKE2.log','LAUNCH2.log'):
 q=p/'evidence_v2'/name
 if q.exists():print(name,q.read_text()[-12000:])
for name in ('STATUS.json','COMPLETE.json','TF_DONE.json','PREFLIGHT.json','SELECTED.json'):
 q=p/'evidence_v2'/name
 if q.exists():print(name,q.read_text())
print('units',sum(1 for _ in (p/'units_v2').glob('*.json')),'load',os.getloadavg())
'''))
    elif args.action == 'fetch':
        out=ROOT/'docs/paper_results/c1pidiag'
        assert out.is_dir()
        for name in ('evidence','evidence_v2'):
            p=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-r',f'gpuA:{REMOTE}/{name}',str(out)],text=True,capture_output=True,timeout=180)
            if p.returncode:raise RuntimeError(p.stderr)
        print('evidence fetched')
    elif args.action == 'stage':
        print(ssh(f'''from pathlib import Path
import hashlib
p=Path({REMOTE!r});assert p.parent.is_dir();p.mkdir(exist_ok=True)
for name in ('evidence','units','smoke','evidence_v2','units_v2'):(p/name).mkdir(exist_ok=True)
for name in ('worker.py','dispatch.py','operate.py'):
 q=p/name
 if q.exists():
  data=q.read_bytes(); backup=p/'evidence'/(name+'.'+hashlib.sha256(data).hexdigest())
  if not backup.exists():backup.write_bytes(data)
print('destination',p)
'''))
        for path in [ROOT/'docs/paper_results/c1pidiag/PANEL.json', *sorted((ROOT/'tools/c1pidiag').glob('*.py'))]:
            p = subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',str(path),f'gpuA:{REMOTE}/{path.name}'],capture_output=True,text=True,timeout=180)
            if p.returncode: raise RuntimeError(p.stderr)
        print('staged')
    elif args.action == 'inspect':
        print(ssh('''import sys,inspect
sys.path.insert(0,'/root/autodl-tmp/c4/speedhead20261030/tools/speedhead')
import wm_worker as w
from wms.model.wms import WmSModel
print('runtime',w.FIX)
print(inspect.getsource(w.load_members))
print(inspect.getsource(WmSModel.step))
print(inspect.signature(WmSModel.forward))
'''))
    elif args.action == 'probe':
        print(ssh('''from pathlib import Path
import gzip,json,os
p=Path('/root/autodl-tmp/c4/c1refresh')
idx=json.loads((p/'states/INDEX.json').read_text())
for cohort in ('pi_u_same','wall_same'):
 e=sorted((e for e in idx['states'].values() if cohort in e['sets']),key=lambda e:(e['world'],e['t0_s'],e['sid']))[0]
 q=p/'states/states'/(e['sid']+'.json.gz')
 with gzip.open(q,'rt') as f:r=json.load(f)
 print(cohort,e['sid'],'keys',list(r))
 print('metadata',json.dumps({k:v for k,v in r.items() if k not in ('history','recorded','vehicles','cav_controllers','truth','initial_state','counter')}))
 h=r.get('history') or {};print('history', {k:v for k,v in h.items() if k not in ('kin','mask','target_mask','ids')})
 print('history shapes',len(h.get('kin',[])),len(h.get('ids',[])))
print('cpus',os.cpu_count(),'load',os.getloadavg())
'''))
    else:
        path = allowed(args.path)
        code = f'from pathlib import Path\np=Path({path!r})\n'
        if args.action == 'read':
            code += 'print(p.read_text())'
        else:
            code += f'print([q.name for q in p.iterdir() if not any(b in q.name.lower() for b in {BAD!r})] if p.is_dir() else "MISSING")'
        print(ssh(code))


if __name__ == '__main__':
    main()
