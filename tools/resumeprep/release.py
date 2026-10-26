"""Box-local baseline staging + bounded committed source overlays. No model/data on laptop."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import io
import tarfile

from boxstage import checked

BASE='1aee086ecb667c69895c157b9ce9e01bb308801c'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='gpuR')
    p.add_argument('--source-host',default='gpuR')
    p.add_argument('--source-runtime',default='/root/autodl-tmp/c4/v5/mainstart/mainr16/restart/runtime')
    p.add_argument('--release',default='/root/resumeprep_release')
    p.add_argument('--resume-baseline', action='store_true')
    p.add_argument('--update-overlays', action='store_true')
    a=p.parse_args()
    for x in (a.source_runtime,a.release):
        checked(x)
    # Existing own probe directory already verified. The helper itself is small;
    # baseline archive creation/extraction both run on the box.
    if not (a.resume_baseline or a.update_overlays):
        subprocess.run(['scp',str(Path(__file__).with_name('boxstage.py')),a.source_host+':/root/resumeprep_probe/boxstage.py'],check=True)
    runtime=a.release+'/runtime'
    cmd=['/root/miniconda3/bin/python','/root/resumeprep_probe/boxstage.py','--source-host','local',
         '--source-root',a.source_runtime,'--dest-host',('local' if a.host==a.source_host else a.host),'--dest-root',runtime,
         'src','traffic_shepherd','configs','scripts','tools','tests']
    if not (a.resume_baseline or a.update_overlays):
        subprocess.run(['ssh',a.source_host,shlex.join(cmd)],check=True)
    names=subprocess.check_output(['git','diff','--name-only','--diff-filter=ACMRT',BASE,'HEAD'],text=True).splitlines()
    # One bounded source-only overlay stream avoids one fragile SSH session/file.
    # The baseline remains box-local; no archive is materialized on the laptop.
    payload=io.BytesIO()
    manifest={}
    import hashlib
    with tarfile.open(fileobj=payload,mode='w') as tar:
        for name in names:
            checked(name)
            data=subprocess.check_output(['git','show','HEAD:'+name])
            manifest[name]=hashlib.sha256(data).hexdigest()
            info=tarfile.TarInfo(name); info.size=len(data)
            tar.addfile(info,io.BytesIO(data))
    script='''import pathlib,sys,json,tarfile,hashlib,os
root=pathlib.Path(sys.argv[1]); changes=json.loads(sys.argv[2]); resume=sys.argv[3]=='1'
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype')
baseline=json.loads((root/'BOX_MANIFEST.json').read_text())
for name,sha in baseline.items():
 if name not in changes and hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha: raise ValueError('baseline hash mismatch: '+name)
if resume:
 receipt=root.parent/'OPERATIONAL_RETRY.json'
 with receipt.open('x') as f: json.dump({'continuation':'S-RESTART renewed','failure':'SSH255 during overlay after baseline verified1072','repair':'verify untouched baseline; replay committed overlays in one SHA-verified stream','status':'retry_armed'},f)
seen=set()
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
 for member in tar:
  p=pathlib.PurePosixPath(member.name)
  if not member.isfile() or p.is_absolute() or '..' in p.parts or any(b in member.name.lower() for b in bad) or member.name not in changes: raise ValueError('unsafe overlay')
  data=tar.extractfile(member).read()
  if hashlib.sha256(data).hexdigest()!=changes[member.name]: raise ValueError('overlay SHA mismatch')
  target=root/member.name; target.parent.mkdir(parents=True,exist_ok=True)
  tmp=target.with_name(target.name+'.restart-staging'); tmp.write_bytes(data); os.replace(tmp,target); seen.add(member.name)
if seen!=set(changes): raise ValueError('incomplete overlay')
(root.parent/'SOURCE_OVERLAY.json').write_text(json.dumps(changes,sort_keys=True))
print(json.dumps({'verified_baseline':len(baseline),'verified_overlays':len(seen),'overlay_manifest_sha256':hashlib.sha256(json.dumps(changes,sort_keys=True).encode()).hexdigest()}))
'''
    command=['ssh',a.host,shlex.join(['/root/miniconda3/bin/python','-c',script,runtime,json.dumps(manifest),'1' if a.resume_baseline else '0'])]
    subprocess.run(command,input=payload.getvalue(),check=True)
    print(json.dumps({'host':a.host,'runtime':runtime,'overlays':names,
                      'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}))


if __name__=='__main__':
    main()
