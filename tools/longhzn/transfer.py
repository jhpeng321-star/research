"""Copy explicitly authorized artifacts through memory; no GPU compute."""
import argparse
import io
import json
import shlex
import subprocess
import tarfile

from operate import PYTHON

FILES = {
    'wm0.pt': '/root/autodl-tmp/c4/restart20261026/arm/loop/wm/c00/wm_s.pt',
    'wm1.pt': '/root/autodl-tmp/c4/restart20261026/arm/loop/wm/c01/wm_s.pt',
    'r16.pt': '/root/autodl-tmp/c4/restart20261026_inputs/core/main14_loop/mainr16_base/wm_s_best.pt',
    'r16_config.json': '/root/autodl-tmp/c4/restart20261026_inputs/core/main14_loop/mainr16_base/config.resolved.json',
    'original_gate.json': '/root/autodl-tmp/c4/restart20261026/arm/horizon_gate_current.json',
    'threshold.json': '/root/autodl-tmp/c4/restart20261026/runtime_c1/configs/coopw_horizon_gate_13E_b2.json',
    'horizon_gate.py': '/root/autodl-tmp/c4/restart20261026/runtime_c1/tools/coopw/horizon_gate.py',
    'wm1_config.toml': '/root/autodl-tmp/c4/restart20261026/runtime_c1/configs/c4_wm_loop14_r16_ft.toml',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('host', choices=('cpuP','cpuQ'))
    a = p.parse_args()
    source = "import io,tarfile,sys,json; files=json.loads(sys.argv[1]); ar=tarfile.open(fileobj=sys.stdout.buffer,mode='w|',dereference=True); "
    source += "[(ar.add(path,arcname=name,recursive=False)) for name,path in files.items()]; ar.close()"
    cmd = shlex.join([PYTHON['gpuA'], '-B', '-c', source, json.dumps(FILES)])
    data = subprocess.check_output(['ssh','-o','BatchMode=yes','gpuA',cmd], timeout=300)
    with tarfile.open(fileobj=io.BytesIO(data)) as ar:
        assert set(ar.getnames()) == set(FILES)
    reader = '''import hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path.home()/'longhzn'; assert root.is_dir()
proof={}
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read())) as ar:
 for entry in ar:
  assert entry.isfile() and '/' not in entry.name and '..' not in entry.name
  data=ar.extractfile(entry).read(); p=root/entry.name
  if p.exists(): assert p.read_bytes()==data
  else: p.write_bytes(data)
  proof[entry.name]=hashlib.sha256(data).hexdigest()
(root/'ARTIFACTS.json').write_text(json.dumps(proof,indent=1)); print(json.dumps(proof))
'''
    command = shlex.join([PYTHON[a.host], '-B','-c',reader])
    result = subprocess.run(['ssh','-o','BatchMode=yes',a.host,command],input=data,capture_output=True,timeout=600)
    print(result.stdout.decode())
    print(result.stderr.decode())
    result.check_returncode()


if __name__ == '__main__':
    main()
