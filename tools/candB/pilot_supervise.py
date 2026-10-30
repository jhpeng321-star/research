"""Own-PID detached dry->fresh-pilot chain; memory polling no faster than 600s."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, doc):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(dict(label='EXPLORATORY', **doc), stream, indent=1, allow_nan=False)
        stream.write('\n')


def incarnation(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exploratory-owner-20261030', action='store_true', required=True)
    p.add_argument('--root', required=True, type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    root.mkdir(exist_ok=False)
    runtime = Path.cwd()
    write(root/'OWNERSHIP.json', dict(pid=os.getpid(), start_ticks=incarnation(os.getpid()),
        utc=datetime.now(timezone.utc).isoformat(), runtime=str(runtime), argv=sys.argv))
    # Wait on gpuA; no peer PID is signalled, queried, or claimed as ours.
    last_query = 0.
    for mode in ('dry', 'pilot'):
        while True:
            if (root/'STOP').exists():
                return 1
            wait = 600 - (time.monotonic()-last_query)
            if wait > 0:
                time.sleep(wait)
            last_query = time.monotonic()
            memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip()
            with (root/'GPU_ADMISSION.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(t=time.time(), label='EXPLORATORY', free_MiB=memory))+'\n')
            if int(memory) >= 20500:
                break
        output = root/f'{mode}_EXPLORATORY'
        command = ['bash', 'tools/candB/pilot_launch.sh', str(output), mode,
                   '--exploratory-owner-20261030']
        env = dict(os.environ, RUNTIME=str(runtime))
        # The supervisor just sampled memory; the launcher consumes that single
        # admission observation instead of issuing a second tight nvidia query.
        env['PILOT_ADMISSION_FREE_MIB'] = memory
        env['PILOT_ADMISSION_EPOCH'] = str(time.time())
        with (root/f'{mode}.log').open('x') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, env=env)
            write(root/f'{mode}_OWNERSHIP.json', dict(pid=child.pid,
                start_ticks=incarnation(child.pid), argv=command, output=str(output),
                utc=datetime.now(timezone.utc).isoformat()))
            rc = child.wait()
        summary = json.loads((output/'RUN_SUMMARY.json').read_text()) if (output/'RUN_SUMMARY.json').is_file() else {}
        write(root/f'{mode}_EXIT.json', dict(returncode=rc, status=summary.get('status'),
                                           error=summary.get('error')))
        if rc or (mode == 'dry' and summary.get('status') != 'completed_dry_run'):
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
