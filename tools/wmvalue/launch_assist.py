"""One-time disjoint cpuR assistant launch; the original dispatcher stays live."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .audit import digest, safe


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--gate', required=True)
    p.add_argument('--real', required=True)
    p.add_argument('--model', required=True)
    a = p.parse_args()
    root = safe(a.root)
    assert root.is_dir() and (root/'LAUNCH.json').is_file()
    assert not (root/'ASSIST_CPU_R_LAUNCH.json').exists(), 'inspect existing assist; never duplicate'
    cmd = [sys.executable, '-B', '-u', '-m', 'tools.wmvalue.assist_cpu',
           '--root', str(root), '--tag', a.tag, '--gate', str(safe(a.gate)),
           '--real', str(safe(a.real)), '--model', str(safe(a.model))]
    with (root/'ASSIST_CPU_R.log').open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    doc = dict(schema='wmvalue_cpu_assist/1', pid=proc.pid, time=time.time(), command=cmd,
        source_sha256=digest(Path(__file__).parent/'assist_cpu.py'), host='gpuR', lanes=8,
        disjoint_world=969439, tasks=40, original_dispatcher_preserved=True,
        authorization='REGISTRATION MAIN19:52 GPU R CPUs released to C1/WM',
        before_first_admission='10cores+40GiB free; subsequent<=8single-thread nice19 lanes')
    with (root/'ASSIST_CPU_R_LAUNCH.json').open('x') as stream:
        json.dump(doc, stream, indent=1)
    print(json.dumps(doc), flush=True)


if __name__ == '__main__':
    main()
