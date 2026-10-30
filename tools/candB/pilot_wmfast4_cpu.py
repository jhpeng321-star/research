"""Durable CUDA-hidden proof for the S4 operational boundary specialization."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    command = [sys.executable, '-B', '-m', 'unittest',
        'tools.candB.pilot_wmfast4_test', 'tools.candB.pilot_wmfast_test',
        'tools.candB.pilot_lean_gate_test']
    began = time.time()
    result = subprocess.run(command, text=True, capture_output=True)
    doc = dict(returncode=result.returncode, seconds=time.time()-began,
        command=command, CUDA_VISIBLE_DEVICES='', stdout=result.stdout, stderr=result.stderr)
    with Path('pilot_WMFAST4_CPU_RESULT.json').open('x') as stream:
        json.dump(doc, stream, indent=1)
    print(json.dumps(doc))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
