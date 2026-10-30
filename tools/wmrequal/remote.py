"""Bounded gpuA-only transport; stdin scripts, retries with 15/45 s backoff."""
from pathlib import Path
import subprocess
import sys
import time


def remote(source, timeout=180):
    for attempt in range(3):
        result = subprocess.run(
            ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'gpuA',
             '/root/miniconda3/bin/python -B -'], input=source, text=True,
            encoding='utf-8', capture_output=True, timeout=timeout)
        if result.returncode != 255 or attempt == 2:
            print(result.stdout, end='')
            print(result.stderr, end='', file=sys.stderr)
            result.check_returncode()
            return result.stdout
        time.sleep((15, 45)[attempt])


if __name__ == '__main__':
    remote(Path(sys.argv[1]).read_text(encoding='utf-8'),
           int(sys.argv[2]) if len(sys.argv) > 2 else 180)
