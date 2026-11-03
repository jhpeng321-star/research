"""Resume certified archive batches after an argv-size verification failure."""
import argparse
from . import archive_move_ops as parent


def setup():
    parent.ROOT='/home/ubuntu/coopmwm20261103_archive1450_v3'
    parent.CPU_RECORD='ARCHIVE_MOVE_CPU_1450_V3.json'
    parent.REG_RECORD='ARCHIVE_MOVE_REGISTRATION_1450_V3.json'
    parent.CLIENT_RECORD='ARCHIVE_MOVE_CLIENT_DISPATCH_1450_V3.json'
    parent.CLIENT_LOG='ARCHIVE_MOVE_CLIENT_1450_V3.log'
    parent.RECOVERY_NOTE=('Engineering recovery of V2 after28 certified batches: '
        'verification payload exceeded argv length. Send verification program through '
        'SSH stdin; adopt only exact manifest/batch-recorded source deletions after '
        'fresh destination SHA checks. Other sources retained/rechecked. Existing '
        'destination namespace adopted; no simulation/inference/repeated deleted-source transfer.\n')


def adopt():
    print(parent.remote('cpuQ',f'''from pathlib import Path
import json,shutil,hashlib,time
old=Path('/home/ubuntu/coopmwm20261103_archive1450_v2');new=Path({parent.ROOT!r})
assert json.loads((new/'CPU_RESULT.json').read_text())['pass_']
assert 'Argument list too long' in json.loads((old/'MOVE_FAILED.json').read_text())['error']
d=json.loads((old/'MOVE_OWNER.json').read_text());p=Path('/proc',str(d['pid']))
assert not p.exists() or (p.joinpath('stat').read_text().split()[21]==d['start_ticks'] and p.joinpath('stat').read_text().split()[2]=='Z')
src=old/'MANIFEST.json';h=hashlib.sha256(src.read_bytes()).hexdigest()
assert h==json.loads((old/'PREPARE_COMPLETE.json').read_text())['manifest_sha256']
for name in ('MANIFEST.json','PREPARE_COMPLETE.json'):
 if not (new/name).exists():shutil.copy2(old/name,new/name)
 assert (new/name).read_bytes()==(old/name).read_bytes()
resume=dict(prior_control=str(old),manifest_sha256=h,prior_batches=len(list(old.glob('ARCHIVE_MOVE_batch*.json'))))
if not (new/'RESUME.json').exists():
 with (new/'RESUME.json').open('x') as f:json.dump(resume,f)
assert json.loads((new/'RESUME.json').read_text())==resume
print(json.dumps(dict(adopted=True,**resume,new_sim_s=0)))''',timeout=45))


def completed():
    import json
    from .storage import write
    value=json.loads(parent.remote('cpuQ',f'''from pathlib import Path
import json,time,shutil
root=Path({parent.ROOT!r});done=json.loads((root/'COMPLETE.json').read_text())
manifest=json.loads((root/'MANIFEST.json').read_text());mapping={{e['relative']:e for e in manifest['entries']}}
categories={{}};seen=set()
for base in (Path('/home/ubuntu/coopmwm20261103_archive1450_v2'),root):
 for path in sorted(base.glob('ARCHIVE_MOVE_batch*.json')):
  batch=json.loads(path.read_text());assert batch['manifest_sha256']==done['manifest_sha256']
  checked={{e['relative']:e for e in batch['verified']}}
  for name in batch['deleted_sources']:
   assert name not in seen;seen.add(name);entry=mapping[name]
   assert checked[name]['sha256']==entry['sha256'] and checked[name]['bytes']==entry['bytes']
   row=categories.setdefault(entry['category'],dict(files=0,bytes=0))
   row['files']+=1;row['bytes']+=entry['bytes']
assert len(seen)==len(manifest['entries']) and sum(x['bytes'] for x in categories.values())==done['total_verified_moved_bytes']
print(json.dumps(dict(completion=done,categories=categories,files=len(seen),
 disk_free_bytes=shutil.disk_usage('/home/ubuntu').free,unix=time.time(),new_sim_s=0)))''',timeout=45))
    p=parent.ops.EVIDENCE/'ARCHIVE_MOVE_1450_COMPLETED.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def remaining():
    print(parent.remote('cpuQ',f'''import subprocess,json
code="""import importlib.util,json,shutil
from pathlib import Path
p=Path({parent.ROOT!r})/'archive_move.py'
s=importlib.util.spec_from_file_location('archive',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
found,skipped=m.candidates();categories={{}}
for name,category in found.items():
 row=categories.setdefault(category,dict(files=0,bytes=0));row['files']+=1;row['bytes']+=Path(name).stat().st_size
free=shutil.disk_usage('/home/ubuntu').free
print(json.dumps(dict(free_bytes=free,remaining_approved_categories=categories,skipped=skipped,
 required_free_bytes=10*1024**3,minimum_additional_space_bytes=max(0,10*1024**3-free-sum(x['bytes'] for x in categories.values())),
 no_hashing_or_moving=True,new_sim_s=0)))"""
p=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-c',code],capture_output=True,text=True,timeout=45)
if p.returncode:raise RuntimeError(p.stderr)
print(p.stdout)''',timeout=60))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('qualify','adopt','register','launch','status','completed','remaining'))
    args=parser.parse_args();setup()
    if args.action in ('adopt','completed','remaining'):globals()[args.action]()
    else:getattr(parent,args.action)()
