"""Sequential, bounded SSH operations for S-WMCOMBINED-1 (stage/sync/preflight/launch/verify).

Run from the repo worktree (branch ``c4/wmcombined``):  ``python tools/wmcombined/operate.py <action>``.
NEVER launch before the S-LONGLOSSTEST job OUT carries DONE.json on gpuR (wait condition).
"""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent))
from common import E3, ID, OLD, OUT  # noqa: E402

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
p=pathlib.Path({(OUT).as_posix()!r}); old=pathlib.Path({(OLD).as_posix()!r}); e3=pathlib.Path({(E3).as_posix()!r})
print('WMCOMBINED_EXISTS',p.exists(),'DISK_GIB',round(shutil.disk_usage(e3.parent).free/2**30,3))
print('LONGLOSS_DONE',(old/'DONE.json').is_file(),'E3_OK',e3.is_dir(),
      'E3_SPLIT',(e3/'SPLIT.json').is_file())
if p.exists():
 print('CONTENTS',[x.name for x in p.iterdir()])
 for n in ['LAUNCH.json','RUNNING.json','DONE.json','STATE.json','RESULTS.json','ERROR.json']:
  q=p/n
  if q.is_file():print(n,q.read_text()[:2000])
print('GPU',subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout)
''')


def stage():
    # Only tracked implementation files from THIS branch's HEAD, never data trees.
    base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    files = subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', 'HEAD', '--', 'src/wms',
         'scripts/c4/loop14_long.py', 'configs/c4_wm_loop14_r16_ft.toml',
         'configs/c4_wm_combined.toml'],
        cwd=ROOT, text=True).splitlines()
    from guard import safe
    payload = {}
    for name in files:
        safe(name)
        content = subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT)
        payload['runtime/'+name] = base64.b64encode(content).decode()
    for path in sorted((ROOT/'tools/wmcombined').glob('*.py')):
        safe(path)
        payload['code/'+path.name] = base64.b64encode(path.read_bytes()).decode()
    for name in ('paired_statistics.py', 'amended.py'):
        payload['code/'+name] = base64.b64encode(
            (ROOT/'tools/longloss'/name).read_bytes()).decode()
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    remote('gpuR', f'''import base64,json,pathlib,shutil,hashlib,os
p=pathlib.Path({(OUT).as_posix()!r});old=pathlib.Path({(OLD).as_posix()!r});e3=pathlib.Path({(E3).as_posix()!r})
assert p.parent.is_dir() and not p.exists(),'existing task directory; inspect instead of duplicate staging'
assert (old/'DONE.json').is_file(),'S-LONGLOSSTEST DONE.json missing (wait condition)'
assert (old/'DESIGN.json').is_file() and e3.is_dir(),'longloss/e3data inputs missing'
assert shutil.disk_usage(p.parent).free>3.5*2**30
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
''', timeout=180)


def sync():
    payload = {p.name: base64.b64encode(p.read_bytes()).decode()
               for p in sorted((ROOT/'tools/wmcombined').glob('*.py'))}
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    remote('gpuR', f'''import pathlib,base64,json,hashlib
p=pathlib.Path({(OUT).as_posix()!r})
assert p.is_dir() and not (p/'LAUNCH.lock').exists(),'never replace running code'
d=json.loads((p/'STAGING.json').read_text())
for name,b64 in json.loads(base64.b64decode({encoded!r})).items():
 raw=base64.b64decode(b64);(p/'code'/name).write_bytes(raw)
 d['sha256']['code/'+name]=hashlib.sha256(raw).hexdigest()
(p/'STAGING.json').write_text(json.dumps(d,indent=1))
print('PRELAUNCH_CODE_SYNCED')
''')


def preflight():
    remote('gpuR', f'''import os,sys,pathlib,json,ast
p=pathlib.Path({(OUT).as_posix()!r});rt=p/'runtime'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(p/'code'),str(rt/'src'),str(rt)]
from guard import install,safe
install()
from common import environment,disk,write,now
for q in (p/'code').glob('*.py'):ast.parse(q.read_text(),filename=str(q))
from wms.config.load import load_train_config
# (1) variant configuration: the loop FT recipe + flags, e3 paired source inert when off
cfg=load_train_config(rt/'configs/c4_wm_combined.toml')
assert cfg.loss.longwm_extra_paired is None and not cfg.model.mainline_interaction_residual
names={{s.name for s in cfg.data.c4_sources}}
assert 'e3_pairs' in names and 'loop14_pairs' in names
sys.path.insert(0,str(p/'code'))
from train import variant_overrides
sets=['loss.longwm_enabled=true','curriculum.unroll_s=[300.0]',
 'curriculum.truncation_steps_per_stage=[600]','optim.batch_size_per_stage=[2]',
 'loss.paired_diff.couple_randomness=true','loss.paired_diff.horizon_mode="stage"']
over=['runtime.device="cuda"']+sets+variant_overrides('combined')
c=load_train_config(rt/'configs/c4_wm_combined.toml',over)
assert c.loss.longwm_enabled and c.loss.longwm_extra_paired=='e3_pairs'
assert c.model.mainline_interaction_residual and c.model.mainline_residual_hidden==64
assert c.optim.lane_v5_long and c.optim.lane_v5_long_lr_scale==0.05
# (2) zero-init head builds under the variant config
from wms.cli.assemble import build_physics, build_model
m=build_model(c,build_physics(c))
assert m.mainline_residual is not None
last=m.mainline_residual.net[-1]
assert float(last.weight.detach().abs().sum())==0.0 and float(last.bias.detach().abs().sum())==0.0
# (3) WM2 loads EXACTLY under the unchanged loop config (gate/scoring reference)
import importlib.util
spec=importlib.util.spec_from_file_location('acc',rt/'tools/loop14_checks/accept.py')
acc=importlib.util.module_from_spec(spec);spec.loader.exec_module(acc)
_,_,wm2=acc.load_arm(rt/'configs/c4_wm_loop14_r16_ft.toml',{(OUT/'wm2/wm_s.pt').as_posix()!r})
del wm2
# (4) E3 paired sources build for BOTH held-out roles (never 'fit')
from e3_score import build_source
cal=build_source({(E3).as_posix()!r},'calibration'); val=build_source({(E3).as_posix()!r},'validation')
assert len(cal.groups)==36 and len(val.groups)==36
# (5) e3 evaluator plumbing smoke on WM2 (CPU, one group, 20 steps)
from tools.r10_attribution import load_model
cfg2,settings,model,_rep=load_model(str(rt/'configs/c4_wm_loop14_r16_ft.toml'),str(p/'wm2/wm_s.pt'),
 ('data.entries_v2=true','loss.paired_diff.couple_randomness=true'))
import torch
doc=__import__('e3_score').evaluate(model,settings,{(E3).as_posix()!r},'calibration',
 torch.device('cpu'),indexes=[0],steps=20,horizons=(10.,))
h=doc['horizons']['10'];assert h['n_pairs']==3 and h['n_groups']==1
import math
assert all(math.isfinite(v) for v in (h['mean_signed_error'],h['mae']))
# (6) verdict-tool bootstrap identity on the reference format
from paired_statistics import bias,compare
ref=json.loads((pathlib.Path({(OLD).as_posix()!r})/'wm1_gate.json').read_text())
r=compare(ref,ref)
assert r['absolute_300s_bias_reduction']==0 and r['reduction_ci95']==[0.,0.]
assert len(ref['rows'])==48 and r['n_worlds']==20
env=environment();assert env['C4_GPU_MEM_FRACTION']=='.85'
write('PREFLIGHT.json',{{'t':now(),'config':'pass','zero_init_head':'pass',
 'wm2_exact_load':'pass','e3_roles':{{'calibration':len(cal.groups),'validation':len(val.groups)}},
 'e3_smoke':{{'pairs':h['n_pairs'],'mean_signed_error_10s':h['mean_signed_error']}},
 'wm1_bias':bias(ref),'bootstrap_identity':r,'free_gib':disk(),'code_compile':'pass'}})
import common
common.mirror()
print((p/'PREFLIGHT.json').read_text())
''', timeout=1500)


def prepare():
    # Detached-free, bounded, CPU: stage inputs (WM2 fetch + receipts) before launch.
    remote('gpuR', f'''import pathlib,sys,subprocess,os
p=pathlib.Path({(OUT).as_posix()!r});assert p.is_dir() and (p/'STAGING.json').is_file()
assert not (p/'DESIGN.json').exists(),'already prepared'
sys.path.insert(0,str(p/'code'))
from guard import install
install()
os.environ['CUDA_VISIBLE_DEVICES']=''
r=subprocess.run([sys.executable,'-B',str(p/'code/prepare.py')],cwd=str(p),
 env={{**os.environ,'PYTHONPATH':f'{{p}}/code:{{p}}/runtime/src:{{p}}/runtime'}},
 capture_output=True,text=True,timeout=1500)
print(r.stdout,end='');print(r.stderr,end='')
raise SystemExit(r.returncode)
''')


def dryrun():
    # Full assemble + optimizer-contract check on CPU (no checkpoints, no training).
    remote('gpuR', f'''import pathlib,sys,subprocess,os
p=pathlib.Path({(OUT).as_posix()!r});assert (p/'DESIGN.json').is_file(),'prepare first'
sys.path.insert(0,str(p/'code'))
from guard import install
install()
r=subprocess.run([sys.executable,'-B',str(p/'code/train.py'),'combined','--dry-run'],
 cwd=str(p/'runtime'),
 env={{**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':f'{{p}}/code:{{p}}/runtime/src:{{p}}/runtime'}},
 capture_output=True,text=True,timeout=4000)
print(r.stdout[-4000:],end='');print(r.stderr[-4000:],end='')
raise SystemExit(r.returncode)
''', timeout=4200)


def launch():
    remote('gpuR', f'''import pathlib,subprocess,os,sys,json
p=pathlib.Path({(OUT).as_posix()!r});assert (p/'PREFLIGHT.json').is_file()
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
  'method':'Popen start_new_session (setsid), stdin /dev/null, persistent log','training_cap_s':12600}})
print((p/'LAUNCH.json').read_text())
''', timeout=45)


def verify():
    remote('gpuR', f'''import pathlib,json,os,subprocess
p=pathlib.Path({(OUT).as_posix()!r})
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
  if n=='DESIGN.json':d={{k:d[k] for k in ['loop_episodes','variants','pair_proofs','wm2_sha256','free_gib']}}
  print(n,json.dumps(d)[:3000])
q=p/'chain.log';print('CHAIN_TAIL',q.read_text()[-3000:] if q.exists() else 'none')
tl=p/'combined/train_log.jsonl'
if tl.exists():print('TRAIN_TAIL',tl.read_text()[-1500:])
print(subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout)
''', timeout=45)
    remote('gpuA', f'''import pathlib
p=pathlib.Path({(OUT).as_posix()!r})
for n in ['LAUNCH.json','RUNNING.json','PREFLIGHT.json','DONE.json','RESULTS.json']:
 q=p/n
 if q.is_file():print('MIRROR',n,q.read_text()[:1500])
''', timeout=45)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['state', 'stage', 'sync', 'preflight', 'prepare',
                                       'dryrun', 'launch', 'verify'])
    args = ap.parse_args()
    globals()[args.action]()
