"""Revise only a known owned operational overlay before continuation dispatch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    name = 'pilot_continue.py'
    source = subprocess.check_output(['git', 'show', commit+':tools/candB/'+name]).decode()
    digest = hashlib.sha256(source.encode()).hexdigest()
    code = f'''import hashlib,json,os
from pathlib import Path
root=Path({args.root!r}); runtime=Path({args.runtime!r})
assert not (root/'continue_trainer_OWNERSHIP.json').exists(), 'continuation already dispatched'
prior=json.loads((root/'continue_OVERLAY.json').read_text())
path=runtime/'tools/candB'/{name!r}
old=hashlib.sha256(path.read_bytes()).hexdigest()
assert old == prior['sha256'][{name!r}], 'not our known operational overlay'
receipt=dict(label='EXPLORATORY',source_commit={commit!r},name={name!r},
    before_sha256=old,after_sha256={digest!r},scientific_source_unchanged=True)
with (root/{args.revision!r}).open('x') as stream: json.dump(receipt,stream,indent=1)
with path.open('w',newline='\\n') as stream: stream.write({source!r})
assert hashlib.sha256(path.read_bytes()).hexdigest()=={digest!r}
print(json.dumps(receipt))
'''
    result = subprocess.run(['ssh', 'gpuA', '/root/miniconda3/bin/python', '-'],
        input=code, text=True, capture_output=True, timeout=25)
    with (Path(__file__).parent / ('pilot_'+args.revision)).open('x') as stream:
        json.dump(dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr), stream, indent=1)
    print(result.stdout + result.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
