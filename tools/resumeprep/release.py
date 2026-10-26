"""Box-local baseline staging + bounded committed source overlays. No model/data on laptop."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from boxstage import checked

BASE='1aee086ecb667c69895c157b9ce9e01bb308801c'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='gpuR')
    p.add_argument('--source-host',default='gpuR')
    p.add_argument('--source-runtime',default='/root/autodl-tmp/c4/v5/mainstart/mainr16/restart/runtime')
    p.add_argument('--release',default='/root/resumeprep_release')
    a=p.parse_args()
    for x in (a.source_runtime,a.release):
        checked(x)
    # Existing own probe directory already verified. The helper itself is small;
    # baseline archive creation/extraction both run on the box.
    subprocess.run(['scp',str(Path(__file__).with_name('boxstage.py')),a.source_host+':/root/resumeprep_probe/boxstage.py'],check=True)
    runtime=a.release+'/runtime'
    cmd=['/root/miniconda3/bin/python','/root/resumeprep_probe/boxstage.py','--source-host','local',
         '--source-root',a.source_runtime,'--dest-host',('local' if a.host==a.source_host else a.host),'--dest-root',runtime,
         'src','traffic_shepherd','configs','scripts','tools','tests']
    subprocess.run(['ssh',a.source_host,shlex.join(cmd)],check=True)
    names=subprocess.check_output(['git','diff','--name-only','--diff-filter=ACMRT',BASE,'HEAD'],text=True).splitlines()
    script='import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(sys.stdin.buffer.read())'
    for name in names:
        checked(name)
        data=subprocess.check_output(['git','show','HEAD:'+name])
        # Each source file is streamed, never embedded in an SSH/base64 argument.
        command=['ssh',a.host,shlex.join(['/root/miniconda3/bin/python','-c',script,runtime+'/'+name])]
        subprocess.run(command,input=data,check=True)
    print(json.dumps({'host':a.host,'runtime':runtime,'overlays':names,
                      'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}))


if __name__=='__main__':
    main()
