"""Sequential, bounded SSH operations for S-RAMPFIX-1 (stage/sync/preflight/launch/verify).

Run from the repo worktree (branch ``c4/rampfix``):  ``python tools/rampfix/operate.py <action>``.
NEVER run before the longloss job OUT exists on gpuR (its artifacts are this job's input).
"""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent))
from common import ID, OLD, OUT, VARIANTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def remote(host, source, timeout=120):
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
                        '/root/miniconda3/bin/python -B -'], input=source,
                       text=True, encoding='utf-8', capture_output=True, timeout=timeout)
    print(p.stdout, end='')
    print(p.stderr, end='')
    p.check_returncode()


def state():
    remote('gpuR', f'''import pathlib,json,shutil,subprocess
p=pathlib.Path({str(OUT)!r}); old=pathlib.Path({str(OLD)!r})
print('RAMPFIX_EXISTS',p.exists(),'DISK_GIB',shutil.disk_usage(old.parent).free/2**30)
if p.exists():
 print('CONTENTS',[x.name for x in p.iterdir()])
 for n in ['LAUNCH.json','RUNNING.json','DONE.json','STATE.json','RESULTS.json']:
  q=p/n
  if q.is_file():print(n,q.read_text())
print('OLD_CONTENTS',[x.name for x in old.iterdir() if not any(t in x.name.lower() for t in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1'))])
print('GPU',subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout)
''')


def stage():
    # Only tracked implementation files from THIS branch's HEAD, never data trees.
    base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    files = subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', 'HEAD', '--', 'src/wms',
         'scripts/c4/loop14_long.py', 'configs/c4_wm_loop14_r16_ft.toml'],
        cwd=ROOT, text=True).splitlines()
    from guard import safe
    payload = {}
    for name in files:
        safe(name)
        content = subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT)
        payload['runtime/'+name] = base64.b64encode(content).decode()
    for path in sorted((ROOT/'tools/rampfix').glob('*.py')):
        safe(path)
        payload['code/'+path.name] = base64.b64encode(path.read_bytes()).decode()
    payload['RAMPFIX.md'] = base64.b64encode((ROOT/'RAMPFIX.md').read_bytes()).decode()
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    remote('gpuR', f'''import base64,json,pathlib,shutil,hashlib,os
p=pathlib.Path({str(OUT)!r});old=pathlib.Path({str(OLD)!r})
assert p.parent.is_dir() and not p.exists(),'existing task directory; inspect instead of duplicate staging'
assert old.is_dir() and (old/'DESIGN.json').is_file(),'longloss OUT (this job input) missing on gpuR'
assert shutil.disk_usage(p.parent).free>3.75*2**30
p.mkdir();(p/'code').mkdir()
data=json.loads(base64.b64decode({encoded!r}))
for name in ['code/guard.py','code/sitecustomize.py']:
 (p/name).write_bytes(base64.b64decode(data[name]))
import sys
sys.path.insert(0,str(p/'code'))
from guard import install,safe
install()
def ignore(root,names):
 return [n for n in names if n=='__pycache__' or any(t in str(pathlib.Path(root)/n).lower() for t in __import__('guard').BAD)]
shutil.copytree(old/'runtime',p/'runtime',ignore=ignore)
hashes={{}}
for name,b64 in data.items():
 q=safe(p/name);q.parent.mkdir(parents=True,exist_ok=True)
 raw=base64.b64decode(b64);q.write_bytes(raw);hashes[name]=hashlib.sha256(raw).hexdigest()
(p/'STAGING.json').write_text(json.dumps({{'base':{base!r},'id':{ID!r},'sha256':hashes,'free_gib':shutil.disk_usage(p).free/2**30}},indent=1))
print('STAGED',len(hashes),'FILES','FREE_GIB',shutil.disk_usage(p).free/2**30)
''', timeout=120)


def sync():
    payload = {p.name: base64.b64encode(p.read_bytes()).decode()
               for p in sorted((ROOT/'tools/rampfix').glob('*.py'))}
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    remote('gpuR', f'''import pathlib,base64,json,hashlib
p=pathlib.Path({str(OUT)!r})
assert p.is_dir() and not (p/'LAUNCH.lock').exists(),'never replace running code'
d=json.loads((p/'STAGING.json').read_text())
for name,b64 in json.loads(base64.b64decode({encoded!r})).items():
 raw=base64.b64decode(b64);(p/'code'/name).write_bytes(raw)
 d['sha256']['code/'+name]=hashlib.sha256(raw).hexdigest()
(p/'STAGING.json').write_text(json.dumps(d,indent=1))
print('PRELAUNCH_CODE_SYNCED')
''')


def preflight():
    from train import variant_overrides
    overrides = {v: variant_overrides(v) for v in VARIANTS}
    remote('gpuR', f'''import os,sys,pathlib,json,ast
p=pathlib.Path({str(OUT)!r});rt=p/'runtime'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(p/'code'),str(rt/'src'),str(rt)]
from guard import install,safe
install()
from common import environment,mirror,disk,write
for q in (p/'code').glob('*.py'):ast.parse(q.read_text(),filename=str(q))
from wms.config.load import load_train_config
sys.path.insert(0,str(rt/'scripts/c4'))
from loop14_long import wm_sets
sets=wm_sets({{'looplong':{{'enabled':True,'wm_loss':True}}}},{{'cycles':[{{'cycle':2}}]}})
cfg=load_train_config(rt/'configs/c4_wm_loop14_r16_ft.toml',sets)
assert cfg.loss.longwm_enabled and cfg.curriculum.unroll_s==(300.0,)
assert cfg.curriculum.truncation_steps_per_stage==(600,)
variants={overrides!r}
for variant,overs in variants.items():
 c=load_train_config(rt/'configs/c4_wm_loop14_r16_ft.toml',overs)
 if variant in ('mres','both'):
  assert c.model.mainline_interaction_residual and c.model.mainline_residual_hidden==64
  from wms.cli.assemble import build_physics, build_model
  m=build_model(c,build_physics(c))
  assert m.mainline_residual is not None
  last=m.mainline_residual.net[-1]
  assert float(last.weight.detach().abs().sum())==0.0 and float(last.bias.detach().abs().sum())==0.0
 if variant in ('lane5','both'):
  assert c.optim.lane_v5_long and c.optim.lane_v5_long_lr_scale==0.05
from paired_statistics import bias,compare
ref=json.loads((pathlib.Path({str(OLD)!r})/'wm1_gate.json').read_text())
r=compare(ref,ref)
assert r['absolute_300s_bias_reduction']==0 and r['reduction_ci95']==[0.,0.]
assert len(ref['rows'])==48 and r['n_worlds']==20
env=environment();assert env['C4_GPU_MEM_FRACTION']=='.85'
write('PREFLIGHT.json',{{'t':__import__('common').now(),'sets':sets,'variants':variants,
      'wm1_bias':bias(ref),'bootstrap_identity':r,'free_gib':disk(),
      'code_compile':'pass','zero_init_head':'pass','configuration':'pass'}})
mirror()
print((p/'PREFLIGHT.json').read_text())
''', timeout=180)


def launch():
    remote('gpuR', f'''import pathlib,subprocess,os,sys,json
p=pathlib.Path({str(OUT)!r});assert (p/'PREFLIGHT.json').is_file()
assert not any((p/n).exists() for n in ['LAUNCH.json','RUNNING.json','DONE.json'])
sys.path.insert(0,str(p/'code'))
from common import environment,write,disk,now
disk()
with (p/'LAUNCH.lock').open('x') as lock:lock.write(now())
with (p/'chain.log').open('x') as log:
 child=subprocess.Popen(['/root/miniconda3/bin/python','-B',str(p/'code/chain.py')],
  cwd=p,env=environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 token=pathlib.Path(f'/proc/{{child.pid}}/stat').read_text().rsplit(')',1)[1].split()[19]
 write('LAUNCH.json',{{'pid':child.pid,'start_ticks':token,'started':now(),'session':os.getsid(child.pid),
  'method':'Popen start_new_session (setsid), stdin /dev/null, persistent log','training_cap_each_s':10800}})
print((p/'LAUNCH.json').read_text())
''', timeout=45)


def verify():
    remote('gpuR', f'''import pathlib,json,os,subprocess
p=pathlib.Path({str(OUT)!r})
if not (p/'LAUNCH.json').is_file():
 print('NOT_LAUNCHED');raise SystemExit
r=json.loads((p/'LAUNCH.json').read_text());pid=r['pid']
s=pathlib.Path(f'/proc/{{pid}}/stat').read_text().rsplit(')',1)[1].split()
assert s[19]==r['start_ticks'] and s[0]!='Z'
assert int(s[1])==1 and os.getsid(pid)==pid and os.readlink(f'/proc/{{pid}}/fd/0')=='/dev/null'
print('DETACHED_VERIFIED',r)
for n in ['RUNNING.json','STATE.json','CHILD.json','PROGRESS.json','DESIGN.json','ERROR.json','DONE.json','RESULTS.json']:
 q=p/n
 if q.is_file():
  d=json.loads(q.read_text())
  if n=='DESIGN.json':d={{k:d[k] for k in ['loop_episodes','variants','pair_proofs','free_gib']}}
  print(n,json.dumps(d))
q=p/'chain.log';print('CHAIN_TAIL',q.read_text()[-3000:])
for v in {VARIANTS!r}:
 q=p/(v+'_train.log')
 if q.exists():print(v.upper()+'_TRAIN_TAIL',q.read_text()[-2000:])
print(subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout)
''', timeout=45)
    remote('gpuA', f'''import pathlib
p=pathlib.Path({str(OUT)!r})
for n in ['LAUNCH.json','RUNNING.json','PREFLIGHT.json','DONE.json']:
 q=p/n
 if q.is_file():print('MIRROR',n,q.read_text())
''', timeout=45)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['state', 'stage', 'sync', 'preflight',
                                       'launch', 'verify'])
    args = ap.parse_args()
    globals()[args.action]()
