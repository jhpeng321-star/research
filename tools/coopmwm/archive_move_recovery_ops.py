"""Adopt the completed manifest after the zero-I/O gpuR interpreter failure."""
import argparse
from . import archive_move_ops as parent


def setup():
    parent.ROOT='/home/ubuntu/coopmwm20261103_archive1450_v2'
    parent.CPU_RECORD='ARCHIVE_MOVE_CPU_1450_V2.json'
    parent.PREPARE_RECORD='ARCHIVE_MOVE_PREPARE_DISPATCH_1450_V2.json'
    parent.REG_RECORD='ARCHIVE_MOVE_REGISTRATION_1450_V2.json'
    parent.CLIENT_RECORD='ARCHIVE_MOVE_CLIENT_DISPATCH_1450_V2.json'
    parent.CLIENT_LOG='ARCHIVE_MOVE_CLIENT_1450_V2.log'
    parent.RECOVERY_NOTE=('Engineering recovery: V1 failed at gpuR python3 PATH lookup BEFORE '
        'namespace creation/rsync/verification/deletion (0 batches). Preserve all V1 '
        'sources/owners/failure. Pin /root/miniconda3/bin/python; new source/control '
        'passes Linux tests, adopt unchanged SHA-checked manifest without rehashing '
        'or any physics replay. Destination namespace is confirmed absent before V2.\n')


def adopt():
    print(parent.remote('cpuQ',f'''from pathlib import Path
import json,shutil,hashlib,time
old=Path('/home/ubuntu/coopmwm20261103_archive1450');new=Path({parent.ROOT!r})
assert json.loads((new/'CPU_RESULT.json').read_text())['pass_']
assert 'python3: command not found' in json.loads((old/'MOVE_FAILED.json').read_text())['error']
assert not list(old.glob('ARCHIVE_MOVE_batch*.json')) and not (old/'COMPLETE.json').exists()
owner=json.loads((old/'MOVE_OWNER.json').read_text());p=Path('/proc',str(owner['pid']))
assert not p.exists() or p.joinpath('stat').read_text().split()[2]=='Z','adopt live old owner'
src=old/'MANIFEST.json';h=hashlib.sha256(src.read_bytes()).hexdigest()
assert h==json.loads((old/'PREPARE_COMPLETE.json').read_text())['manifest_sha256']
if not (new/'MANIFEST.json').exists():shutil.copy2(src,new/'MANIFEST.json')
assert hashlib.sha256((new/'MANIFEST.json').read_bytes()).hexdigest()==h
v=dict(json.loads((old/'PREPARE_COMPLETE.json').read_text()),adopted_from=str(src),adopted_unix=time.time(),newly_hashed_bytes=0,newly_deleted_bytes=0)
if not (new/'PREPARE_COMPLETE.json').exists():
 with (new/'PREPARE_COMPLETE.json').open('x') as f:json.dump(v,f)
print(json.dumps(dict(adopted=True,manifest_sha256=h,files=v['files'],bytes=v['bytes'],newly_hashed_bytes=0,newly_deleted_bytes=0)))''',timeout=45))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','adopt','status','register','launch'))
    a=p.parse_args();setup()
    adopt() if a.action=='adopt' else getattr(parent,a.action)()
