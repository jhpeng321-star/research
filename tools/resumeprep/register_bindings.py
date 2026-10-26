"""Future operator step: append exact smoke hashes, then acknowledge on the box."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='gpuA')
    p.add_argument('--arm',default='/root/resumeprep_release/arm')
    p.add_argument('--register',default='E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    a=p.parse_args()
    raw=subprocess.check_output(['ssh',a.host,shlex.join(['cat',a.arm+'/SMOKE_BINDINGS.json'])])
    bindings=json.loads(raw)
    if bindings['seeds'] != [968142,968143] or bindings['sim_s']!=1922 or not bindings.get('native_gate'):
        raise ValueError('missing native-relative gate or wrong registered mechanics smoke')
    digest=hashlib.sha256(raw).hexdigest()
    text='\n\n### S-C0DIAG-SMOKE-1 exact pre-dispatch bindings\n\n'+json.dumps(bindings,indent=1)+'\n\nBindings SHA256: '+digest+'\n'
    with Path(a.register).open('a',encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    script="import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.open('xb').write(sys.stdin.buffer.read())"
    command=shlex.join(['/root/miniconda3/bin/python','-c',script,a.arm+'/SHARED_SMOKE_HASHES_APPENDED.json'])
    subprocess.run(['ssh',a.host,command],input=json.dumps({'bindings_sha256':digest}).encode(),check=True)
    print(json.dumps({'appended':a.register,'bindings_sha256':digest}))


if __name__=='__main__':
    main()
