"""Read-only gpuA transport and isolated CPU task operations; never simulation."""
import argparse
import base64
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/root/autodl-tmp/c4/restart20261026'


def ssh(host, code, quiet=False):
    python = '/root/miniconda3/bin/python' if host == 'gpuA' else '/home/ubuntu/miniconda3/bin/python'
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
                             python + ' -B -'], input=code, text=True, capture_output=True, timeout=90)
    if not quiet:
        print(result.stdout, end='')
    if result.stderr:
        print(result.stderr)
    result.check_returncode()
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inventory', 'cpu', 'bindings', 'paths', 'copy', 'setup', 'peek', 'logs'])
    parser.add_argument('--only', default='metadata')
    args = parser.parse_args()
    if args.action == 'copy':
        files = {f'checkpoints/{label}.pt': f'{REMOTE}/arm/loop/policy/{rel}/policy_native_best.pt'
                 for label, rel in [('c00', 'c00'), ('c01_on', 'c01.a3'), ('c02', 'c02.h60b.r2')]}
        files['checkpoints/starts.pt'] = f'{REMOTE}/arm/loop/returns/returns_c01.a1.npz'
        files['checkpoints/nativehead.pt'] = f'{REMOTE}/runtime_c1/configs/native_cav_head_v5.json'
        for phase in ('c00', 'c01.a3', 'c02.h60b.r2'):
            for name in ('RUN_SUMMARY.json', 'updates.jsonl', 'native_gate_result.json'):
                files[f'metadata/{phase}_{name}'] = f'{REMOTE}/arm/loop/policy/{phase}/{name}'
        for name in ('policy.py', 'dreamer.py', 'policy14.py', 'policy14_cli.py', 'counterfactual.py',
                     'coopw.py', 'paircrit.py', 'longcritic.py', 'bridge.py'):
            files[f'deployed/{name}'] = f'{REMOTE}/runtime_c1/src/wms/control/{name}'
        for name in ('c4_policy14_c1_fallback.toml', 'c4_policy14_main.toml'):
            files[f'deployed/{name}'] = f'{REMOTE}/runtime_c1/configs/{name}'
        files = {k: v for k, v in files.items() if
                 (k == f'checkpoints/{args.only}.pt' if args.only != 'metadata' else
                  not k.startswith('checkpoints/') and not k.endswith('updates.jsonl'))}
        if args.only != 'metadata':
            parent = Path('E:/Temp/opencode')
            assert parent.is_dir()
            for rel, source in files.items():
                tmp = parent / ('groupgrad_' + Path(rel).name)
                subprocess.run(['scp', '-C', '-o', 'BatchMode=yes', 'gpuA:' + source, str(tmp)],
                               check=True, timeout=90)
                subprocess.run(['scp', '-C', '-o', 'BatchMode=yes', str(tmp),
                                'cpuP:/home/ubuntu/groupgrad/' + rel], check=True, timeout=90)
                import hashlib
                digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
                ssh('cpuP', f'''import pathlib,json,hashlib
r=pathlib.Path.home()/'groupgrad'; p=r/{rel!r}; assert hashlib.sha256(p.read_bytes()).hexdigest()=={digest!r}
doc=json.loads((r/'inputs.json').read_text()); doc[{rel!r}]={{'source':{source!r},'sha256':{digest!r}}}; (r/'inputs.json').write_text(json.dumps(doc,indent=2)); print({rel!r},{digest!r})
''')
            return
        code = f'''import pathlib,json,base64,hashlib
files={files!r}; out={{}}
for dest,src in files.items():
 p=pathlib.Path(src)
 if p.is_file():
  data=p.read_bytes(); out[dest]={{'source':src,'sha256':hashlib.sha256(data).hexdigest(),'bytes':base64.b64encode(data).decode()}}
print(json.dumps(out))
'''
        p = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'gpuA', '/root/miniconda3/bin/python -B -'],
                           input=code, text=True, capture_output=True, timeout=90)
        p.check_returncode()
        payload = json.loads(p.stdout)
        print('read-only gpuA files:', {k: v['sha256'] for k, v in payload.items()}, flush=True)
        data = base64.b64encode(json.dumps(payload).encode()).decode()
        ssh('cpuP', f'''import pathlib,json,base64,hashlib,shutil
r=pathlib.Path.home()/'groupgrad'; assert r.is_dir()
payload=json.loads(base64.b64decode({data!r})); receipt={{}}
for rel,v in payload.items():
 p=r/rel; p.parent.mkdir(parents=True,exist_ok=True); b=base64.b64decode(v['bytes']); assert hashlib.sha256(b).hexdigest()==v['sha256']; p.write_bytes(b)
 receipt[rel]={{k:v[k] for k in ('source','sha256')}}
p=r/'checkpoints/c01_off.pt'; p.parent.mkdir(exist_ok=True); src=pathlib.Path.home()/'e3long/inputs/coopoff_cycle1/policy_latest.pt'; shutil.copyfile(src,p)
receipt['checkpoints/c01_off.pt']={{'source':str(src),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}
prior=json.loads((r/'inputs.json').read_text()) if (r/'inputs.json').exists() else {{}}; prior.update(receipt)
(r/'inputs.json').write_text(json.dumps(prior,indent=2)); print(json.dumps(receipt,indent=2))
''')
    elif args.action == 'logs':
        code = f'''import pathlib,json,hashlib
r=pathlib.Path({REMOTE!r}); out={{}}
keys=['update','actor_loss','actor_grad_norm','actor_norm','return_scale','cf_weights','cf_term_grad_norms','cf_coverage','cf_curriculum_position','actor_grad_norms','action_audit','n_futures','imagination_horizon_s']
for phase in ['c00','c01','c01.a2','c01.a3','c02','c02.h60','c02.h60b','c02.h60b.r2']:
 p=r/'arm/loop/policy'/phase/'updates.jsonl'
 if not p.exists(): continue
 rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
 out[phase]={{'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'n':len(rows),'keys':list(rows[-1]) if rows else [],'rows':[{{k:v for k,v in x.items() if k in keys}} for x in rows]}}
print(json.dumps(out))
'''
        text = ssh('gpuA', code, quiet=True)
        (ROOT/'tools/groupgrad/logs.json').write_text(text, encoding='utf-8')
        print({k: v['n'] for k,v in json.loads(text).items()})
    elif args.action == 'setup':
        ssh('cpuP', '''import pathlib,tarfile,json,torch,os
r=pathlib.Path.home()/'groupgrad'; assert r.is_dir(); dest=r/'runtime'; dest.mkdir(exist_ok=True)
with tarfile.open(r/'source.tar') as t:
 members=[m for m in t.getmembers() if not any(s in m.name.lower() for s in ('sealed/','holdout','heldout','cal_split','chunk5','j4b','inc1'))]
 assert all(not m.name.startswith('/') and '..' not in pathlib.PurePosixPath(m.name).parts for m in members)
 t.extractall(dest,members=members)
os.environ['CUDA_VISIBLE_DEVICES']=''; torch.set_num_threads(1)
for label in ('c00','c01_on','c01_off','c02'):
 p=torch.load(r/'checkpoints'/f'{label}.pt',map_location='cpu',weights_only=False)
 print(label,list(p),{k:len(p[k]) for k in ('policy','group_encoder','critic')},'update',p.get('update_count'))
 print('config',p.get('config')); print('counterfactual',p.get('counterfactual'))
''')
    elif args.action == 'peek':
        ssh('cpuP', '''import pathlib,json
r=pathlib.Path.home()/'groupgrad'
for name in ['deployed/c4_policy14_c1_fallback.toml']:
 print(name,(r/name).read_text())
''')
    elif args.action == 'inventory':
        ssh('gpuA', f'''import pathlib,json
r=pathlib.Path({REMOTE!r})
for p in [r,r/'arm',r/'arm/loop',r/'arm/loop/policy',r/'arm/loop/policy/c02.h60b.r2',r/'arm/loop/policy/c00.a2',r/'arm/loop/policy/c01']:
 print(str(p), sorted(x.name for x in p.iterdir()) if p.is_dir() else 'MISSING')
''')
    elif args.action == 'paths':
        ssh('gpuA', f'''import pathlib,json,hashlib
r=pathlib.Path({REMOTE!r})
for p in [r/'arm/loop/policy/c00',r/'arm/mini',r/'e5',pathlib.Path('/root/autodl-tmp/c4/coopoff20261026')]:
 print(str(p), sorted(x.name for x in p.iterdir()) if p.is_dir() else 'MISSING')
for rel in ['arm/loop/policy/c00/RUN_SUMMARY.json','arm/loop/policy/c01.a3/RUN_SUMMARY.json','e5/GO.json']:
 p=r/rel
 if p.exists(): print(rel,p.read_text()[:6000])
''')
    elif args.action == 'bindings':
        ssh('gpuA', f'''import pathlib,json
r=pathlib.Path({REMOTE!r})
for rel in ['arm/loop/LOOP14_STATE.json','arm/loop/policy/c00/policy_native_best.pt','arm/loop/policy/c01.a3/policy_native_best.pt','arm/loop/policy/c02.h60b.r2/native_gate_result.json','e5/FIX_VERIFIED.json']:
 p=r/rel
 if p.suffix=='.json': print(rel,p.read_text()[:14000])
 else: print(rel,p.exists(),p.stat().st_size if p.exists() else None)
print('nativeinit_baseline', [x.name for x in (r/'nativeinit_baseline').iterdir()])
''')
    else:
        ssh('cpuP', '''import pathlib,sys,torch
p=pathlib.Path.home()/'groupgrad'; assert p.parent.is_dir(); p.mkdir(exist_ok=True)
print(sys.executable,torch.__version__,p)
''')


if __name__ == '__main__':
    main()
