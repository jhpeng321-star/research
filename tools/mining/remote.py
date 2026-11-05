"""Bounded laptop transport; preserve each command and its complete result."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def run(host, code, command='python3 -B -', timeout=90):
    if host == 'gpuA':
        argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                '-o', 'ConnectionAttempts=1', '-o', 'ServerAliveInterval=10',
                '-o', 'ServerAliveCountMax=2', 'gpuA', command]
    else:
        helper = (
            "import sys; sys.dont_write_bytecode=True; "
            "sys.path.insert(0,'/mnt/e/agent-work/c4-rerun-launch'); "
            "from tools.coopmwm.cpuv_ssh import Transport; "
            f"r=Transport().run({command!r},timeout={timeout-10},"
            "input=sys.stdin.buffer.read(),capture_output=True); "
            "sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);"
            "sys.exit(r.returncode)"
        )
        argv = ['wsl', '-d', 'Ubuntu-22.04', '--', 'python3', '-c', helper]
    started = time.time()
    result = subprocess.run(argv, input=code.encode(), capture_output=True, timeout=timeout)
    return dict(host=host, command=command, unix=started,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                wall_s=time.time()-started, returncode=result.returncode,
                stdout=result.stdout.decode(errors='replace'),
                stderr=result.stderr.decode(errors='replace'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('host', choices=['gpuA', 'cpuV'])
    parser.add_argument('script', type=Path)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--command', default='python3 -B -')
    parser.add_argument('--timeout', type=int, default=90)
    args = parser.parse_args()
    assert args.receipt.parent.is_dir() and not args.receipt.exists()
    data = run(args.host, args.script.read_text(encoding='utf-8'), args.command, args.timeout)
    args.receipt.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(data), flush=True)
    raise SystemExit(data['returncode'])
