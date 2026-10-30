"""Durable CUDA-hidden check receipt, independent of SSH lifetime."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    began = time.monotonic()
    command = [sys.executable, '-B', '-m', 'pytest', '-q',
        'tools/candB/pilot_wmfast_test.py', 'tools/candB/pilot_lean_gate_test.py',
        'tools/candB/pilot_continue_test.py']
    result = subprocess.run(command, text=True, capture_output=True)
    with Path('pilot_WMFAST_CPU_RESULT.json').open('x') as stream:
        json.dump(dict(returncode=result.returncode, command=command,
            seconds=time.monotonic()-began, cuda_visible_devices='',
            stdout=result.stdout, stderr=result.stderr), stream, indent=1)
    print(result.stdout+result.stderr, flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
