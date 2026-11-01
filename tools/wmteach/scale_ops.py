"""One-use continuation staging/dispatch and bounded read-only collection."""
import json
from pathlib import Path
import time

from tools.coopbandit.protocol import sha
from tools.wmteach.ops import remote,ROOT,REMOTE

SCALE='/root/autodl-tmp/wmteach_scale20261101_v1'
HERE=ROOT/'docs/paper_results/wmteach'


def stage():
    overrides={f'tools/wmteach/{name}':(ROOT/'tools/wmteach'/name).read_text(encoding='utf-8')
               for name in ('scale.py','test_scale.py','labels.py','student.py')}
    design=(ROOT/'docs/paper_results/WMTEACH_SCALE_DESIGN.md').read_text(encoding='utf-8')
    return remote(f'''import pathlib,json,hashlib,shutil
parent=pathlib.Path({REMOTE!r});root=pathlib.Path({SCALE!r})
assert root.parent.is_dir() and not root.exists(), 'one-use root exists; never redispatch'
assert shutil.disk_usage(root.parent).free>4*2**30
identity=json.loads((parent/'IDENTITY.json').read_text())
for name,d in identity['sources'].items():
 assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==d,name
root.mkdir()
files={{str(pathlib.Path(n).relative_to(parent)) : pathlib.Path(n).read_text()
       for n in identity['sources'] if pathlib.Path(n).is_relative_to(parent)}}
files.update(json.loads({json.dumps(overrides)!r}))
files['WMTEACH_SCALE_DESIGN.md']={design!r}
for name,text in files.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:f.write(text)
(root/'deps').symlink_to(parent/'deps',target_is_directory=True)
for name in ('states','labels'):(root/name).mkdir()
sources={{**identity['sources'],**{{str(root/n):hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}}}}
new=dict(schema='wmteach-scale/1',backend='FULL-v4',source=identity['source'],parent=str(parent),
 parent_identity_sha256=hashlib.sha256((parent/'IDENTITY.json').read_bytes()).hexdigest(),sources=sources,
 worlds=list(range(969400,969432)),states=5120,lineages=2,times=[300+7.5*i for i in range(1,81)],
 K=8,horizons=[120,300],settings='(clock_index+lineage)%4',new_sumo_sim_s=0,
 observation_contract='FIT-joint-population/cold-motion-controller/causal-imagined-request-counters',
 design_sha256=hashlib.sha256((root/'WMTEACH_SCALE_DESIGN.md').read_bytes()).hexdigest())
with (root/'IDENTITY.json').open('x') as f:json.dump(new,f,sort_keys=True)
print(json.dumps(dict(root=str(root),identity_sha256=hashlib.sha256((root/'IDENTITY.json').read_bytes()).hexdigest(),
 design_sha256=new['design_sha256'],sources=len(sources))))
''',timeout=90)


def cpu():
    return remote(f'''import pathlib,sys,unittest,json,os
root=pathlib.Path({SCALE!r});sys.path[:0]=[str(root),str(root/'deps')]
os.environ['CUDA_VISIBLE_DEVICES']=''
suite=unittest.defaultTestLoader.loadTestsFromNames(['tools.wmteach.test_cpu','tools.wmteach.test_scale'])
r=unittest.TextTestRunner().run(suite)
assert r.wasSuccessful() and not r.skipped
receipt=dict(pass_=True,tests=r.testsRun,new_sumo_sim_s=0)
with (root/'CPU.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
''',timeout=90)


def register():
    design=sha(ROOT/'docs/paper_results/WMTEACH_SCALE_DESIGN.md')
    line=(f'- S-WMTEACH OWNER1253 SCALE CONTINUATION REGISTERED BEFORE DISPATCH: designSHA256{design}; '
          'supersedes first32state student LIVE trigger before any physics; own collector41120 exact-owned stopped, '
          'initialGPU437822 retained/no replay. gpuA immutable /root/autodl-tmp/wmteach_scale20261101_v1, '
          'FULL-v4 source22cc32f6; wait existing12EXAM/twoH quality without recomputation, report H300 helpful '
          'argmax agreement/sign/rank/regret FIRST. 32FIT969400..969431 x2independent O0 imagined lineages '
          'x80clocks307.5..900 every7.5s=5120DISTINCT observable states; setting=(clockindex+lineage)%4, '
          'K8xO0..O4 paired futures batched/H120,H300/source-inclusive delay. Frozen FIT joint-driver population '
          'at descendants/cold motion+controller/causal imagined request totals,exposure,backlog; no generator '
          'latents or exact labels fit student. Inherited roster-only EXAM quality is weaker, not fully conditioned '
          'qualification. Bounded batched-vs-single firstfuture canary <=1e-3veh-s and conservation/finite gate; '
          'first result reusedONCE. Shared S-DISTILL feature/executor/GBT-MLP/confidence/nested4x3worldCV, '
          'fit FIRST complete5120corpus/H300 only. LIVE also needs >=3helpfuldiagnosticstates,>=2exactargmaxhits,'
          '>=20%helpfulagreement,Wilson95lower>5%,positive mean exact chosen saving on helpful states; '
          'gate fixed before QUALITY completed, never selects snapshot/model. Existing MAIN11:48 cap100000/'
          'sameDEV984000..984023/panelSHA72776ff1/exact58800upper retained; final policy+design hashes/portable '
          'parity/CPU24guards registration before one-use live. Strict8GiB/nice19/singlethread/newSUMO0 for '
          'teacher; no peerPIDaction/retries/deletions/newenv/push. WMDATA2quarantined/FULL-v5notREADY; '
          'await explicit immutable successor handoff before a separately versioned rerun.\n')
    registry=Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    if 'designSHA256'+design not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(line);f.flush()
    receipt=HERE/'SCALE_REGISTRATION.json'
    if not receipt.exists():
        with receipt.open('x',encoding='utf-8') as f:json.dump(dict(line=line,design_sha256=design),f);f.write('\n')
    return json.dumps(dict(path=str(receipt),design_sha256=design))


def launch():
    registration=json.loads((HERE/'SCALE_REGISTRATION.json').read_text(encoding='utf-8'))
    return remote(f'''import pathlib,json,subprocess,os,time,hashlib,shutil
root=pathlib.Path({SCALE!r})
assert not (root/'LAUNCH_ONCE.json').exists(), 'existing dispatch; never duplicate'
assert json.loads((root/'CPU.json').read_text())['pass_']
assert hashlib.sha256((root/'WMTEACH_SCALE_DESIGN.md').read_bytes()).hexdigest()=={registration['design_sha256']!r}
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>12*1024 and os.getloadavg()[0]<24 and shutil.disk_usage(root).free>4*2**30
with (root/'LAUNCH_ONCE.json').open('x') as f:json.dump(dict(unix=time.time(),parent=os.getpid()),f)
with (root/'RUN.log').open('x') as log:
 p=subprocess.Popen(['/root/miniconda3/bin/python','-B','-u','-m','tools.wmteach.scale',str(root)],
  cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
  env={{**os.environ,'PYTHONPATH':str(root/'deps'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}})
receipt=dict(pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],
 boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),
 root=str(root),strict_memory_gib=8,new_sumo_sim_s=0)
with (root/'DISPATCH.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
''')


def collect():
    data=json.loads(remote(f'''import pathlib,json,time,os,subprocess,shutil
root=pathlib.Path({SCALE!r});files={{}}
for name in ('IDENTITY.json','CPU.json','DISPATCH.json','QUALITY.json','HELPFUL_QUALITY.json',
 'QUALITY_PROVENANCE.json','CANARY_PASS.json','CORPUS.json','COMPLETE.json','FAILURE.json','student_scale_5120/SUMMARY.json'):
 p=root/name
 if p.is_file():files[name]=json.loads(p.read_text())
owner=files.get('DISPATCH.json');process=None
if owner:
 p=pathlib.Path(f'/proc/{{owner["pid"]}}/stat');st=p.read_text().split() if p.is_file() else None
 process=dict(pid=owner['pid'],state=st[2] if st else None,identity_matches=bool(st and st[21]==owner['start_ticks'] and
  pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id']))
print(json.dumps(dict(unix=time.time(),root=str(root),files=files,process=process,
 generated_states=len(list((root/'states').glob('*/*_t*.json'))),
 complete_lineages=len(list((root/'states').glob('*/COMPLETE.json'))),
 labelled_states=len(list((root/'labels').glob('*/LABEL_H300.json'))),
 log_tail=(root/'RUN.log').read_text()[-8000:] if (root/'RUN.log').is_file() else '',
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True),
 load=os.getloadavg(),disk_free=shutil.disk_usage(root).free,new_sumo_sim_s=0)))
''',timeout=90))
    dest=HERE/f'SCALE_SNAPSHOT_{int(data["unix"])}.json'
    with dest.open('x',encoding='utf-8') as f:json.dump(data,f,sort_keys=True,allow_nan=False);f.write('\n')
    return json.dumps(dict(path=str(dest),process=data['process'],generated=data['generated_states'],
                           labelled=data['labelled_states'],quality=data['files'].get('HELPFUL_QUALITY.json'),
                           log_tail=data['log_tail'][-2000:]))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','cpu','register','launch','collect'])
    args=parser.parse_args();print(globals()[args.action]())
