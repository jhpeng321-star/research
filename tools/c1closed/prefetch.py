"""Bulk transport of immutable own outputs; existing watch remains sole finalizer.

This pre-populates the exact cache used by the frozen sync.py so its final
per-file checksum pass reuses content rather than opening hundreds of SCPs.
Never launches inference, evaluates outcomes, writes COMPLETE, or signals PIDs.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import time

from operate import HERE, REMOTE, scp
from probe import ssh, allowed


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def destination(name):
    p=PurePosixPath(name)
    assert len(p.parts)==2 and p.parts[0] in ('panel','evidence')
    assert p.name not in ('.','..') and p.suffix=='.json'
    allowed(name)
    return HERE/'out'/Path(name) if p.parts[0]=='panel' else HERE/'evidence'/p.name


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=128)
    parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    assert 1<=args.limit<=256
    if args.launch:
        from watch import creation
        for receipt in (HERE/'evidence').glob('PREFETCH_*_LAUNCH.json'):
            r=json.loads(receipt.read_text())
            assert creation(r['pid'])!=r['creation_time'],'one existing bulk-copy batch is still running'
        tag=str(time.time_ns());logpath=HERE/'evidence'/('PREFETCH_'+tag+'.log')
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--limit',str(args.limit)]
        with logpath.open('x') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP,close_fds=True)
        identity=creation(child.pid);assert identity is not None
        receipt=dict(pid=child.pid,creation_time=identity,at=time.time(),command=command,log=str(logpath),
                     role='one-off immutable-cache copy only; no inference, analysis, finalization or signals')
        with (HERE/'evidence'/('PREFETCH_'+tag+'_LAUNCH.json')).open('x') as f:json.dump(receipt,f,indent=1)
        print(json.dumps(receipt));return
    manifest=json.loads(ssh('gpuA',f'''from pathlib import Path
import json,hashlib
p=Path({REMOTE!r});result={{}}
paths=list((p/'panel').glob('*_h300.json'))+list((p/'evidence').glob('*_attempt*.json'))
for q in paths:
 assert not q.is_symlink()
 try:json.loads(q.read_text())
 except json.JSONDecodeError:continue
 result[str(q.relative_to(p))]=hashlib.sha256(q.read_bytes()).hexdigest()
print(json.dumps(result))
'''))
    missing=[]
    for name,value in sorted(manifest.items()):
        target=destination(name)
        if target.exists():
            assert digest(target)==value, 'immutable cached result differs; inspect before replacement'
        else:missing.append(name)
    selected=missing[:args.limit]
    print(json.dumps(dict(remote_files=len(manifest),uncached=len(missing),batch=len(selected))),flush=True)
    if not selected:return
    hashes={name:manifest[name] for name in selected}
    tag=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    remote=REMOTE+'/prefetch/'+tag+'.tgz'
    archive=json.loads(ssh('gpuA',f'''from pathlib import Path
import hashlib,json,tarfile,os
os.nice(19)
p=Path({REMOTE!r});assert p.is_dir();dest=p/'prefetch';dest.mkdir(exist_ok=True)
hashes={hashes!r};archive=Path({remote!r})
for name,value in hashes.items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==value
if not archive.exists():
 with tarfile.open(archive,'x:gz',compresslevel=1) as tf:
  for name in hashes:tf.add(p/name,arcname=name,recursive=False)
print(json.dumps(dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size)))
'''))
    print(json.dumps(dict(archive_bytes=archive['bytes'],batch=len(selected))),flush=True)
    local=HERE/'out'/('prefetch_'+tag+'.tgz')
    print(scp(f'gpuA:{remote}',local),flush=True);assert digest(local)==archive['sha256']
    with tarfile.open(local,'r:gz') as tf:
        members=tf.getmembers();assert {m.name for m in members}==set(selected)
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk()
            target=destination(member.name);assert target.parent.is_dir()
            data=tf.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==hashes[member.name]
            if target.exists():assert digest(target)==hashes[member.name]
            else:
                with target.open('xb') as f:f.write(data)
    print(json.dumps(dict(cached=len(selected),archive_bytes=archive['bytes'],still_uncached=len(missing)-len(selected))),flush=True)


if __name__=='__main__':main()
