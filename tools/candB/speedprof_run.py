"""Durable own-PID/status/log receipt for bounded probes, including SSH disconnects."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    log = args.receipt.with_suffix('.log')
    start = time.time()
    with log.open('w') as stream:
        child = subprocess.Popen([sys.executable, '-B', *command], stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT)
        proc = Path('/proc')/str(child.pid)/'stat'
        ticks = proc.read_text().rsplit(')', 1)[1].split()[19]
        receipt = dict(pid=child.pid, start_ticks=ticks, parent_pid=os.getpid(),
                       start_epoch=start, command=command, log=str(log), status='running')
        args.receipt.write_text(json.dumps(receipt, indent=1)+'\n')
        code = child.wait()
    receipt.update(status='completed', returncode=code, seconds=time.time()-start)
    args.receipt.write_text(json.dumps(receipt, indent=1)+'\n')
    print(log.read_text(), end='')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
