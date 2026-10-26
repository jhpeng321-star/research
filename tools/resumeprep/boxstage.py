"""Stream an explicit regular-file bundle box-to-box, with no laptop files/base64 argv.

The coordinator relays bytes through pipes; extraction and all staging happen on boxes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import PurePosixPath
import shlex
import subprocess
import sys

BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'hetdata_truth', 'vtype')


def checked(path):
    p = PurePosixPath(path)
    if '..' in p.parts or any(x in str(p).lower() for x in BAD):
        raise ValueError('restricted/traversing staging path')
    return str(p)


def commands(source_host, source_root, dest_host, dest_root, paths, materialize_links=False):
    for p in (source_root, dest_root, *paths):
        checked(p)
    if any(PurePosixPath(p).is_absolute() for p in paths):
        raise ValueError('bundle members must be relative')
    # Source walks only the explicit directories and prunes names BEFORE opening.
    source = '''import os,sys,tarfile,json,hashlib,io,pathlib,re
root=pathlib.Path(sys.argv[1]); names=json.loads(sys.argv[2]); bad=json.loads(sys.argv[3])
materialize=sys.argv[4]=='1'
files=[]
for name in names:
 p=root/name
 if p.is_file() and (materialize or not p.is_symlink()): files.append(p)
 elif p.is_dir():
  for base,dirs,leaves in os.walk(p,followlinks=False):
   dirs[:]=[d for d in dirs if not any(b in d.lower() for b in bad) and not (pathlib.Path(base)/d).is_symlink()]
   files.extend(pathlib.Path(base)/f for f in leaves if not any(b in f.lower() for b in bad) and (materialize or not (pathlib.Path(base)/f).is_symlink()))
 else: raise ValueError('missing/linked source '+name)
manifest={}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 for p in sorted(set(files)):
  resolved=p.resolve(strict=True)
  if any(b in str(resolved).lower() for b in bad) or not resolved.is_relative_to(root.resolve()): raise ValueError('unapproved source target')
  if any(964000<=int(s)<=964199 for s in re.findall(r'(?<![0-9])([0-9]{6})(?![0-9])',str(resolved)+'/'+str(p))): raise ValueError('evaluation seed refused')
  rel=str(p.relative_to(root)); data=p.read_bytes(); manifest[rel]=hashlib.sha256(data).hexdigest()
  info=tarfile.TarInfo(rel); info.size=len(data); tar.addfile(info,io.BytesIO(data))
 data=json.dumps(manifest,sort_keys=True).encode(); info=tarfile.TarInfo('BOX_MANIFEST.json'); info.size=len(data); tar.addfile(info,io.BytesIO(data))
'''
    dest = '''import sys,tarfile,pathlib,hashlib,json
root=pathlib.Path(sys.argv[1]); bad=json.loads(sys.argv[2]); root.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
 for m in tar:
  p=pathlib.PurePosixPath(m.name)
  if not m.isfile() or p.is_absolute() or '..' in p.parts or any(b in m.name.lower() for b in bad): raise ValueError('unsafe member')
  dst=root/m.name; dst.parent.mkdir(parents=True,exist_ok=True)
  with dst.open('xb') as f: f.write(tar.extractfile(m).read())
manifest=json.loads((root/'BOX_MANIFEST.json').read_text())
for name,sha in manifest.items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha: raise ValueError('hash mismatch')
print(json.dumps({'verified':len(manifest),'manifest_sha256':hashlib.sha256((root/'BOX_MANIFEST.json').read_bytes()).hexdigest()}))
'''
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
    def command(host, script, args):
        py = '/home/ubuntu/miniconda3/bin/python' if host in ('cpuP','cpuQ') else '/root/miniconda3/bin/python'
        return ([sys.executable,'-c',script,*args] if host=='local' else
                [*ssh,host,shlex.join([py,'-c',script,*args])])
    return (command(source_host,source,[source_root,json.dumps(paths),json.dumps(BAD),'1' if materialize_links else '0']),
            command(dest_host,dest,[dest_root,json.dumps(BAD)]))


def stage(*args):
    a, b = commands(*args)
    # Only bounded script text is in argv; payload is never materialized on Windows.
    with subprocess.Popen(a, stdout=subprocess.PIPE) as source:
        with subprocess.Popen(b, stdin=source.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as dest:
            source.stdout.close()
            try:
                out, err = dest.communicate(timeout=600)
                source.wait(timeout=30)
            except BaseException:
                dest.terminate()
                source.terminate()
                raise
            if source.returncode or dest.returncode:
                raise RuntimeError(f'box stage source={source.returncode} dest={dest.returncode}: {err.decode()}')
            return json.loads(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-host', required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--dest-host', required=True)
    p.add_argument('--dest-root', required=True)
    p.add_argument('--materialize-links', action='store_true')
    p.add_argument('paths', nargs='+')
    a = p.parse_args()
    print(json.dumps(stage(a.source_host, a.source_root, a.dest_host, a.dest_root, a.paths, a.materialize_links)))
