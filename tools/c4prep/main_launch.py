"""Later MAIN-only one-command launcher. Never invoked by S-C4PREP preparation.

Resolve the final immutable runtime pointer, then require its resolved config,
final entrant binding, validated anchor, and matching dry-run receipts.
"""
import argparse
import base64
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--entrant', choices=('on', 'off'), required=True)
    p.add_argument('--candidate', choices=('A',), default='A')
    a = p.parse_args()
    source = f'''import pathlib,json,os
out=pathlib.Path('/root/autodl-tmp/c4/restart20261026/arm/loop/c4prep')
r=json.loads((out/'CURRENT_RUNTIME.json').read_text())
code=pathlib.Path(r['runtime'])
assert json.loads((code/'SOURCE.json').read_text())['commit']==r['commit']
python='/root/miniconda3/bin/python'
os.execv(python,[python,'-B',str(code/'tools/c4prep/launch.py'),'start',
 '--config',str(out/'resolved_A_{a.entrant.upper()}'/'CONFIG.json')])
'''
    encoded = base64.b64encode(source.encode()).decode()
    command = shlex.join(['/root/miniconda3/bin/python', '-B', '-c',
                          f'import base64;exec(base64.b64decode({encoded!r}))'])
    return subprocess.call(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'gpuA', command])


if __name__ == '__main__':
    raise SystemExit(main())
