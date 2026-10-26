"""Run on gpuR: SHA-identical q views on gpuA, reusing existing verified arrays."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile

DATA = Path('/root/autodl-tmp/c4/v5/data')
BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'vtype')


def safe(path):
    text = str(path).lower()
    if any(b in text for b in BAD) or any(964000 <= int(s) <= 964199 for s in re.findall(r'(?<!\d)(\d{6})(?!\d)', text)):
        raise ValueError('restricted/evaluation q-view path')


def main():
    dest = sys.argv[1]
    safe(dest)
    manifest = {}
    for name in ('v5_native_q', 'genrand_plain_q'):
        root = DATA/name
        for base, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not any(b in d.lower() for b in BAD)]
            for name in files:
                p = Path(base)/name
                safe(p)
                target = p.resolve(strict=True)
                safe(target)
                if not target.is_relative_to(DATA):
                    raise ValueError('q-view escapes approved TRAIN data root')
                h = hashlib.sha256()
                with p.open('rb') as stream:
                    for block in iter(lambda: stream.read(1 << 20), b''):
                        h.update(block)
                manifest[str(p.relative_to(DATA))] = {
                    'sha256': h.hexdigest(), 'target': str(target.relative_to(DATA)), 'size': target.stat().st_size}
    script = '''import pathlib,json,sys,hashlib
root=pathlib.Path(sys.argv[1]); root.mkdir(parents=True,exist_ok=False)
data=pathlib.Path('/root/autodl-tmp/c4/v5/data'); manifest=json.load(sys.stdin); missing=[]; verified={}
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype')
for name,row in manifest.items():
 if any(b in (name+'/'+row['target']).lower() for b in bad): raise ValueError('restricted path')
 dst=root/name; candidate=data/row['target']; dst.parent.mkdir(parents=True,exist_ok=True)
 if candidate.is_file():
  resolved=candidate.resolve(strict=True)
  if not resolved.is_relative_to(data) or any(b in str(resolved).lower() for b in bad): raise ValueError('q target escape')
  if resolved not in verified:
   h=hashlib.sha256()
   with resolved.open('rb') as stream:
    for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
   verified[resolved]=h.hexdigest()
  if verified[resolved]!=row['sha256']: raise ValueError('existing-array SHA mismatch: '+name)
  dst.symlink_to(resolved)
 else: missing.append(name)
(root/'QVIEW_MANIFEST.json').write_text(json.dumps(manifest,sort_keys=True))
print(json.dumps({'missing':missing,'verified_links':len(manifest)-len(missing),'missing_bytes':sum(manifest[n]['size'] for n in missing)}))
'''
    result = subprocess.run(['ssh', 'gpuA', shlex.join(['/root/miniconda3/bin/python', '-c', script, dest])],
                            input=json.dumps(manifest).encode(), capture_output=True, check=True)
    plan = json.loads(result.stdout)
    print(json.dumps({k: v for k, v in plan.items() if k != 'missing'}), flush=True)
    # Only genuinely missing queue/metadata bytes cross the wire; all existing
    # training arrays were SHA-verified in place. This never substitutes a corpus.
    if plan['missing_bytes'] > 2 * 1024**3:
        raise RuntimeError('missing q-view content exceeds bounded staging capacity')
    receive = '''import sys,tarfile,pathlib,json,hashlib
root=pathlib.Path(sys.argv[1]); manifest=json.loads((root/'QVIEW_MANIFEST.json').read_text()); count=0
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
 for m in tar:
  if not m.isfile() or m.name not in manifest: raise ValueError('unsafe q member')
  p=root/m.name
  if p.exists() or p.is_symlink(): raise ValueError('q destination already exists')
  raw=tar.extractfile(m).read()
  if hashlib.sha256(raw).hexdigest()!=manifest[m.name]['sha256']: raise ValueError('q payload SHA mismatch')
  p.write_bytes(raw); count+=1
if any(not (root/n).is_file() for n in manifest): raise ValueError('incomplete q view')
receipt={'verified_files':len(manifest),'copied_missing_files':count,'manifest_sha256':hashlib.sha256((root/'QVIEW_MANIFEST.json').read_bytes()).hexdigest()}
(root/'QVIEW_VERIFIED.json').write_text(json.dumps(receipt)); print(json.dumps(receipt))
'''
    proc = subprocess.Popen(['ssh', 'gpuA', shlex.join(['/root/miniconda3/bin/python', '-c', receive, dest])], stdin=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdin, mode='w|') as tar:
        for name in plan['missing']:
            raw = (DATA/name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != manifest[name]['sha256']:
                raise ValueError('source changed during q staging')
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('q-view payload staging failed')


if __name__ == '__main__':
    main()
