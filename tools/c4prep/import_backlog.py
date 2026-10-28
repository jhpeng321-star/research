"""Transport permitted, already-reexported C0-C2 clocks and audit C3 on both CPUs.

The archive contains new clocks and metadata only. Original arrays on gpuA are
SHA-verified against S-BACKLOG and linked into NEW exclusive dataset roots.
No converter, scenario input, simulator, or peer process is invoked.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/longint'))
from operate import remote


def main():
    source = (ROOT / 'tools/c4prep/backlog_bundle_remote.py').read_bytes()
    bundles = []
    for host in ('cpuP', 'cpuQ'):
        p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
                            '/home/ubuntu/miniconda3/bin/python -B -' + (' audit-only' if host == 'cpuQ' else '')], input=source,
                           capture_output=True, timeout=45, check=True)
        bundles.append((host, p.stdout))
        print(json.dumps({'host': host, 'archive_bytes': len(p.stdout),
                          'sha256': hashlib.sha256(p.stdout).hexdigest()}), flush=True)
    lengths = [(h, len(b), hashlib.sha256(b).hexdigest()) for h, b in bundles]
    remote(f'''import hashlib,io,json,os,pathlib,tarfile,re,sys
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026')
dest=r/'arm/loop/c4prep/backlog_import_v2'
assert dest.parent.is_dir() and not dest.exists()
dest.mkdir()
bad=re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype',re.I)
receipts=[]
for host,n,digest in {lengths!r}:
 raw=sys.stdin.buffer.read(n);assert hashlib.sha256(raw).hexdigest()==digest
 part=dest/host;part.mkdir()
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:*') as tar:
  for item in tar:
   p=pathlib.PurePosixPath(item.name)
   assert item.isfile() and not p.is_absolute() and '..' not in p.parts and not bad.search(str(p))
   target=part/str(p);target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as f:f.write(tar.extractfile(item).read())
 receipts.append({{'host':host,'archive_sha256':digest,'tapes':json.loads((part/'C3_TAPE_AUDIT.json').read_text())}})
datasets=[]
for cycle in (0,1,2):
 name=f'pairs_c{{cycle:02}}.a1.backlog';p=dest/'cpuP'/name
 sums=json.loads((p/'SHA256SUMS.json').read_text())
 original=r/f'arm/loop/data/pairs_c{{cycle:02}}.a1'
 manifest=json.loads((p/'C4_EPISODES_MANIFEST.json').read_text())
 old=json.loads((original/'C4_EPISODES_MANIFEST.json').read_text())
 assert [e['episode'] for e in manifest['episodes']]==[e['episode'] for e in old['episodes']]
 linked=0
 for rel,digest in sums.items():
  assert not bad.search(rel) and not pathlib.PurePosixPath(rel).is_absolute() and '..' not in pathlib.PurePosixPath(rel).parts
  target=p/rel
  if not target.exists():
   src=original/rel;assert src.is_file() and not bad.search(str(src.resolve()))
   assert hashlib.sha256(src.read_bytes()).hexdigest()==digest
   target.parent.mkdir(parents=True,exist_ok=True);os.link(src,target);linked+=1
  assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
 datasets.append({{'cycle':cycle,'root':str(p),'episodes':len(manifest['episodes']),
  'original_arrays_verified_and_linked':linked,'manifest_sha256':hashlib.sha256((p/'C4_EPISODES_MANIFEST.json').read_bytes()).hexdigest()}})
doc={{'task':'S-C4PREP2','datasets':datasets,'c3_audits':receipts,'originals_preserved':True,'new_sim_s':0}}
(dest/'IMPORT.json').write_text(json.dumps(doc,indent=1))
print(json.dumps({{'datasets':datasets,'c3_tapes':sum(len(x['tapes']) for x in receipts),
 'c3_complete_clocks':sum(t['complete_clocks'] for x in receipts for t in x['tapes']), 'receipt':str(dest/'IMPORT.json')}}))
''', b''.join(b for _, b in bundles), timeout=55)


if __name__ == '__main__':
    main()
