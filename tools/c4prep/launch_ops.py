"""Sequential, batched SSH transport for explicitly owned launch operations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('amend', 'amend-resume', 'inspect', 'fit', 'fit-status', 'fit-fallback', 'checks', 'checks-status', 'prepare', 'dry', 'diagnose', 'status', 'start', 'verify'))
    args = parser.parse_args()
    if args.action in ('amend', 'amend-resume'):
        filename = 'MAIN_1700.json' if args.action == 'amend-resume' else 'MAIN_1500.json'
        authority = (ROOT / 'tools/c4prep/evidence' / filename).read_text()
        source = 'import sys\nsys.argv=["amend",' + repr(authority) + ']\n'
        source += (ROOT / 'tools/c4prep/amend_launch_remote.py').read_text()
        if args.action == 'amend-resume':
            source += '\nimport sys\nsys.argv=["launch_remote","diagnose"]\n'
            source += (ROOT / 'tools/c4prep/launch_remote.py').read_text()
    else:
        source = 'import sys\nsys.argv=["launch_remote",' + repr(args.action) + ']\n'
        source += (ROOT / 'tools/c4prep/launch_remote.py').read_text()
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'ControlPersist=no',
        'gpuA', '/root/miniconda3/bin/python -B -'], input=source, text=True,
        capture_output=True, timeout=45)
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
