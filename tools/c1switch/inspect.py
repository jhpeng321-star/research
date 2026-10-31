"""Exact-path receipt discovery; never recursively open experimental inputs."""
import argparse
import json
from pathlib import Path
import re
import subprocess

REG = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['registration', 'remote'])
    a = p.parse_args()
    if a.action == 'registration':
        for i, line in enumerate(REG.read_text(encoding='utf-8').splitlines(), 1):
            if re.search(r'969[01]\d\d|C1SWITCH', line):
                print(f'{i}: {line}')
    else:
        code = '''import json, os
from pathlib import Path
roots = ['/root/c1switch20261031', '/root/c1switch', '/root/wmdata2/jobs']
out = {r: (sorted(p.name for p in Path(r).iterdir()) if Path(r).is_dir() else None) for r in roots}
out['load'] = os.getloadavg()
print(json.dumps(out))
'''
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                                 'gpuR', '/root/miniconda3/bin/python -'], input=code, text=True,
                                capture_output=True, timeout=25)
        print(result.stdout)
        print(result.stderr)
        result.check_returncode()


if __name__ == '__main__':
    main()
