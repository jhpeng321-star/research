"""Explicit once-only pre-START infrastructure retry, preserving failed receipts."""
import argparse
import json
import os
import subprocess
import time

from .audit import digest, safe
from .orchestrate import fetch, run_task, save_new


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--original', required=True)
    p.add_argument('--key', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--gate', required=True)
    a = p.parse_args()
    root, original = safe(a.root), safe(a.original)
    claim_path = root/'claims'/(a.key+'.json')
    claim = json.loads(claim_path.read_text())
    log = root/'logs'/(a.key+'.log')
    error = root/'claims'/(a.key+'.error.json')
    assert error.is_file() and json.loads(error.read_text())['returncode'] == 255
    assert not (root/'claims'/(a.key+'.remote.json')).exists(), 'not a pre-START failure'
    assert log.read_text().strip().startswith('Connection closed by ')
    assert len(log.read_text().splitlines()) == 1, 'must audit all exception/START text'
    assert not (root/'tasks'/(a.key+'.json')).exists()
    assert claim['device'] == 'cuda' and claim['host'] == 'gpuA'
    legacy = []
    for path in (original/'claims').glob('*.remote.json'):
        meta = json.loads(path.read_text())
        owner = json.loads(path.with_name(path.name.replace('.remote.json', '.json')).read_text())
        if owner['host'] == claim['host']:
            legacy.append(int(meta['pid']))
    code = ("from pathlib import Path; import json\n"
        f"assert not Path({claim['remote']!r}).exists(), 'remote output exists: collect it'\n"
        f"allowed=set({legacy!r}); active=[]\n"
        "for p in Path('/proc').iterdir():\n"
        " if not p.name.isdigit(): continue\n"
        " try: args=(p/'cmdline').read_bytes().split(b'\\0')\n"
        " except (FileNotFoundError,PermissionError,ProcessLookupError): continue\n"
        " if args[:4]==[b'/root/miniconda3/bin/python',b'-B',b'-u',b'-']:\n"
        "  active.append(int(p.name)); assert int(p.name) in allowed, 'unknown stdin worker: do not retry'\n"
        "print(json.dumps(dict(no_unknown_stdin_workers=True,active_legacy_pids=active,remote_output=False)))\n")
    check = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', claim['host'],
        '/root/miniconda3/bin/python -B -'], input=code, text=True, capture_output=True,
        timeout=30, check=True)
    gpu = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', claim['host'],
        'nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=30, check=True)
    assert not gpu.stdout.strip(), 'peer GPU occupied: wait, no retry admission'
    parent = root/'infra_retry_once'
    parent.mkdir(exist_ok=True)
    retry = parent/a.key
    assert not retry.exists(), 'single retry already reserved/consumed'
    retry.mkdir()
    for name in ('tasks', 'claims', 'logs'):
        (retry/name).mkdir()
    save_new(retry/'AUTHORIZATION.json', dict(amendment='MAIN21:12', key=a.key,
        failed_log_sha256=digest(log), failed_error_sha256=digest(error),
        admission=json.loads(check.stdout), gpu_compute_pids=[], time=time.time(), pid=os.getpid()))
    task = tuple(claim['task'])
    run_task(retry, a.tag, task, claim['host'], 'cuda', digest(a.gate), short=True)
    fetch(claim['host'], claim['remote'], root/'tasks'/(a.key+'.json'))
    save_new(retry/'COMPLETE.json', dict(key=a.key, sha256=digest(root/'tasks'/(a.key+'.json'))))
    print('AUTHORIZED INFRA RETRY COMPLETE; failed claim/log retained', a.key, flush=True)


if __name__ == '__main__':
    main()
