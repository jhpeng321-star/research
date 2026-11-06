"""Guarded SSH inspection for NATIVELANE's authorised hosts (no shell quoting)."""
from __future__ import annotations

import argparse
import base64
import json
import subprocess


def remote(host, code, data=None, timeout=180):
    assert host in ("cpuX", "gpuB", "gpuR")
    payload = base64.b64encode(code.encode()).decode()
    python = "/root/miniconda3/bin/python" if host != "cpuX" else "/home/ubuntu/c4/venv/bin/python"
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host,
         f"{python} -c \"import base64;exec(base64.b64decode('{payload}'))\""],
        input=data, capture_output=True, timeout=timeout, check=False)
    if proc.returncode:
        raise RuntimeError(proc.stdout.decode(errors="replace") + proc.stderr.decode(errors="replace"))
    return proc.stdout


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("host", choices=("cpuX", "gpuB", "gpuR"))
    ap.add_argument("op", choices=("list", "read"))
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    code = '''
import json,re
from pathlib import Path
req=json.loads(INPUT)
bad=re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype",re.I)
for raw in req['paths']:
 assert not bad.search(raw),raw
 p=Path(raw).expanduser(); assert not bad.search(str(p.resolve())),p
 print('PATH',p)
 if req['op']=='list': print(json.dumps([x.name for x in p.iterdir() if not bad.search(x.name)],indent=1))
 else: print(p.read_text())
'''.replace("INPUT", repr(json.dumps(vars(a))))
    print(remote(a.host, code).decode())
