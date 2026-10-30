"""Deploy owner2305 only before exclusive same-phase dispatch; preserve old bytes."""
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    runtime = '/root/autodl-tmp/c4/candB_pilot_final_EXPLORATORY_20261030'
    root = '/root/autodl-tmp/c4/candB_pilot_final_run_EXPLORATORY_20261030'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    sources = {n: subprocess.check_output(['git', 'show', commit+':tools/candB/'+n]).decode()
               for n in ('pilot_recipe.py', 'pilot_telemetry_2305.py')}
    hashes = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()}
    code = f'''import hashlib,json,os,subprocess,time
from pathlib import Path
runtime=Path({runtime!r}); root=Path({root!r})
assert runtime.is_dir() and root.is_dir()
assert not (root/'continue_trainer_OWNERSHIP.json').exists(), 'dispatch already happened; refuse hot patch'
owner=json.loads((root/'continue_OWNERSHIP.json').read_text())
fields=(Path('/proc')/str(owner['pid'])/'stat').read_text().rsplit(')',1)[1].split()
assert fields[19]==str(owner['start_ticks']) and fields[0]!='Z'
prior=json.loads((root/'continue_RECIPE_revision2.json').read_text())
path=runtime/'tools/candB/pilot_recipe.py'
old=path.read_bytes()
assert hashlib.sha256(old).hexdigest()==prior['after_sha256']
assert not (runtime/'tools/candB/pilot_telemetry_2305.py').exists()
env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='',C4_ALLOWED_ROOTS='/root/autodl-tmp/c4')
subprocess.run(['/root/miniconda3/bin/python','-B','-m','tools.candB.pilot_authority',
    '--exploratory-owner-20261030','verify','authority_gen/PILOT_EXPLORATORY_AUTHORITY.json'],cwd=runtime,env=env,check=True)
assert not (root/'continue_trainer_OWNERSHIP.json').exists()
with (root/'continue_RECIPE_pre2305.py').open('xb') as stream: stream.write(old)
sources={sources!r}; hashes={hashes!r}
with (runtime/'tools/candB/pilot_telemetry_2305.py').open('x',newline='\\n') as stream:
    stream.write(sources['pilot_telemetry_2305.py'])
with path.open('w',newline='\\n') as stream: stream.write(sources['pilot_recipe.py'])
for name,digest in hashes.items():
    assert hashlib.sha256((runtime/'tools/candB'/name).read_bytes()).hexdigest()==digest
receipt=dict(label='EXPLORATORY',source_commit={commit!r},epoch=time.time(),sha256=hashes,
    previous_recipe_sha256=hashlib.sha256(old).hexdigest(),controller=owner,
    scientific_source_unchanged=True,baseline_recomputed=False,dispatch_created=False)
with (root/'continue_OVERLAY_2305.json').open('x') as stream: json.dump(receipt,stream,indent=1)
print(json.dumps(receipt))
'''
    result = subprocess.run(['ssh', 'gpuA', '/root/miniconda3/bin/python', '-'],
                            input=code, text=True, capture_output=True, timeout=50)
    with (Path(__file__).parent / 'pilot_deploy_2305_receipt.json').open('x') as stream:
        json.dump(dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr), stream, indent=1)
    print(result.stdout + result.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
