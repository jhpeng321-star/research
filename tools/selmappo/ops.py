"""Immutable CPU staging only; never starts SUMO or a detached training job."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

from .protocol import digest, safe_path
from .remote import run

REG = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')


def cpu_check(host):
    files = {p.as_posix(): safe_path(p).read_text(encoding='utf-8')
             for p in sorted(Path('tools/selmappo').glob('*.py'))}
    files['REGISTRATION.md'] = safe_path(REG).read_text(encoding='utf-8')
    files['docs/paper_results/SELMAPPO_DESIGN.md'] = safe_path(
        'docs/paper_results/SELMAPPO_DESIGN.md').read_text(encoding='utf-8')
    binding = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    raw = base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    result = run(host, f'''
import base64,hashlib,json,os,subprocess,sys,zlib
from pathlib import Path
home=Path.home()
root=home/('selmappo_cpucheck_{binding[:16]}')
assert home.is_dir()
files=json.loads(zlib.decompress(base64.b64decode(sys.stdin.buffer.read())))
assert hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()=={binding!r}
if root.exists():
 print('EXISTING_CPU_STAGE',root)
 print((root/'CPU_RESULT.json').read_text() if (root/'CPU_RESULT.json').exists() else 'UNRESOLVED_STAGE_NO_RELAUNCH')
 sys.exit(0 if (root/'CPU_RESULT.json').exists() else 2)
root.mkdir()
for name,text in files.items():
 p=root/name
 assert p.is_relative_to(root)
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:f.write(text)
assert (home/'e3final/runtime/src').is_dir() if {host!r}=='cpuQ' else (Path('/root/autodl-tmp/c4/speedhead20261030/e3final/runtime/src')).is_dir()
src=home/'e3final/runtime/src' if {host!r}=='cpuQ' else Path('/root/autodl-tmp/c4/speedhead20261030/e3final/runtime/src')
env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+str(src),SELMAPPO_TEST_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
test=subprocess.run([sys.executable,'-m','unittest','tools.selmappo.test_cpu','-v'],cwd=root,env=env,capture_output=True,text=True,timeout=90)
print(test.stdout);print(test.stderr)
audit=subprocess.run([sys.executable,'-m','tools.selmappo.audit','--registration',str(root/'REGISTRATION.md'),'--root',str(home)],cwd=root,env=env,capture_output=True,text=True,timeout=90)
print(audit.stdout);print(audit.stderr)
record=dict(host={host!r},root=str(root),binding={binding!r},design_sha256={digest('docs/paper_results/SELMAPPO_DESIGN.md')!r},cpu_returncode=test.returncode,cpu_stdout=test.stdout,cpu_stderr=test.stderr,audit_returncode=audit.returncode,audit_stdout=audit.stdout,audit_stderr=audit.stderr,new_sumo_seconds=0,load=os.getloadavg())
with (root/'CPU_RESULT.json').open('x') as f:json.dump(record,f,indent=2)
print('CPU_RECEIPT',json.dumps(record))
sys.exit(0 if test.returncode==0 and audit.returncode==0 else 1)
''', stdin=raw, timeout=240)
    tagged = [line.removeprefix('CPU_RECEIPT ') for line in result.stdout.splitlines()
              if line.startswith('CPU_RECEIPT ')]
    if tagged:
        evidence = safe_path('docs/paper_results/selmappo')
        if not evidence.parent.is_dir():
            raise ValueError('missing intended evidence parent')
        evidence.mkdir(exist_ok=True)
        with (evidence/f'CPU_{host}_{binding[:16]}.json').open('x', encoding='utf-8') as stream:
            json.dump(json.loads(tagged[-1]), stream, indent=2)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('host', choices=('cpuQ', 'gpuR'))
    cpu_check(parser.parse_args().host)
