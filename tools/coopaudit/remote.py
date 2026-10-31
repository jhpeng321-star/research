"""Short sequential SSH operations; existing interpreters only."""
import argparse
import base64
import subprocess
from pathlib import Path

PYTHON = {'gpuR': '/root/miniconda3/bin/python',
          'cpuQ': '/home/ubuntu/miniconda3/bin/python'}


def run(host, code, timeout=120):
    command = f"{PYTHON[host]} -"
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                           host, command], input=code, capture_output=True, text=True, timeout=timeout)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('host', choices=PYTHON)
    p.add_argument('code', nargs='?')
    p.add_argument('--file')
    p.add_argument('--out')
    a = p.parse_args()
    result = run(a.host, Path(a.file).read_text() if a.file else a.code)
    print(result.stdout, end='')
    print(result.stderr, end='')
    if a.out:
        path = Path(a.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.stdout)
    raise SystemExit(result.returncode)
