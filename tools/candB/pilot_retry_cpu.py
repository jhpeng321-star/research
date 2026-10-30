"""Durable CUDA-hidden proof for the owner03:00 retry contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    tag = os.environ.get('PILOT_RETRY_CPU_TAG', '0')
    command = [sys.executable, '-B', '-m', 'unittest',
        'tools.candB.pilot_retry_test', 'tools.candB.pilot_replay_guard_test',
        'tools.candB.pilot_continue_test']
    began = time.time()
    result = subprocess.run(command, text=True, capture_output=True)
    doc = dict(returncode=result.returncode, seconds=time.time()-began,
        command=command, CUDA_VISIBLE_DEVICES='', stdout=result.stdout, stderr=result.stderr)
    with Path(f'pilot_RETRY_CPU_RESULT_{tag}.json').open('x') as stream:
        json.dump(doc, stream, indent=1)
    print(json.dumps(doc))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
