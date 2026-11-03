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


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('qualify','adopt','register','launch','status'))
    args=parser.parse_args();setup()
    adopt() if args.action=='adopt' else getattr(parent,args.action)()
