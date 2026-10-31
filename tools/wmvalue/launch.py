"""Launch one durable local inference/finalization chain; record its exact PID."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .audit import digest, safe
from .evaluate_frozen import FROZEN_SHA


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--gate', required=True)
    p.add_argument('--real', required=True)
    p.add_argument('--model', required=True)
    a = p.parse_args()
    root = safe(a.root)
    assert root.parent.is_dir() and digest(safe(a.model)) == FROZEN_SHA
    assert safe(a.real).is_file() and safe(a.gate).is_file()
    root.mkdir(exist_ok=True)
    if (root/'LAUNCH.json').exists():
        raise ValueError('existing dispatch receipt: inspect PID and task claims, never duplicate')
    cmd = [sys.executable, '-B', '-u', '-m', 'tools.wmvalue.orchestrate',
           '--root', str(root), '--tag', a.tag, '--gate', str(safe(a.gate)),
           '--real', str(safe(a.real)), '--model', str(safe(a.model))]
    with (root/'DISPATCH.log').open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    doc = dict(schema='wmvalue_dispatch/1', pid=proc.pid, command=cmd, time=time.time(),
        cpu_host='gpuA', cpu_lanes=4, gpu_host='gpuR', gpu_lanes=3,
        gpu_admission='no peer compute PIDs; own workers only', no_duplicate_claims=True,
        finalization='complete320tasks -> frozen400pairs -> report+registration+commit',
        frozen_model_sha256=FROZEN_SHA, real_arrays_sha256=digest(a.real),
        code_sha256={n: digest(Path(__file__).parent/n) for n in
                     ('orchestrate.py', 'worker.py', 'evaluate_frozen.py', 'finalize.py')})
    with (root/'LAUNCH.json').open('x') as stream:
        json.dump(doc, stream, indent=1, allow_nan=False)
    print(json.dumps(doc), flush=True)


if __name__ == '__main__':
    main()
