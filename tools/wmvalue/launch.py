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
    p.add_argument('--short', action='store_true')
    p.add_argument('--original')
    p.add_argument('--prefix-gate')
    a = p.parse_args()
    root = safe(a.root)
    assert root.parent.is_dir() and digest(safe(a.model)) == FROZEN_SHA
    assert safe(a.real).is_file() and safe(a.gate).is_file()
    if a.short:
        assert a.original and a.prefix_gate
        audit = json.loads(safe(a.prefix_gate).read_text())
        assert audit['verdict'] == 'PASS' and audit['exact_prefix_parity_futures'] == [0, 3]
        assert audit['runtime_sha256'] == digest(Path(__file__).with_name('runtime.py'))
    root.mkdir(exist_ok=True)
    if (root/'LAUNCH.json').exists():
        raise ValueError('existing dispatch receipt: inspect PID and task claims, never duplicate')
    cmd = [sys.executable, '-B', '-u', '-m',
           'tools.wmvalue.short_dispatch' if a.short else 'tools.wmvalue.orchestrate',
           '--root', str(root), '--tag', a.tag, '--gate', str(safe(a.gate)),
            '--real', str(safe(a.real)), '--model', str(safe(a.model))]
    if a.short:
        cmd.extend(['--original', str(safe(a.original))])
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
    if a.short:
        doc.update(mode='MAIN21:12 short-prefix-first', cpu_hosts=dict(gpuA=4, gpuR=8),
            gpu_hosts=dict(gpuA=3, gpuR=3), original=str(safe(a.original)),
            prefix_gate_sha256=digest(a.prefix_gate),
            finalization='complete320prefixes -> frozen400pairs B/C/D/E -> FAIL or complete A',
            short_code_sha256={n: digest(Path(__file__).parent/n) for n in
                ('short_dispatch.py', 'evaluate_short.py', 'finalize_short.py')})
    with (root/'LAUNCH.json').open('x') as stream:
        json.dump(doc, stream, indent=1, allow_nan=False)
    print(json.dumps(doc), flush=True)


if __name__ == '__main__':
    main()
