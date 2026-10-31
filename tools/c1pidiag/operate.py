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
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                        'gpuA', '/root/miniconda3/bin/python -B -'], input=code,
                       text=True, encoding='utf-8', capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stdout + p.stderr)
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('select', 'probe', 'read', 'list'))
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
 print('metadata',json.dumps({k:v for k,v in r.items() if k not in ('history','recorded','vehicles','cav_controllers','truth')})[:10000])
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
