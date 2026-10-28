"""Detached local evidence operations with owned PID-incarnation receipts."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'evidence_entrant/operations'


def incarnation(pid):
    if sys.platform == 'win32':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)]*4)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        times = [wintypes.FILETIME() for _ in range(4)]
        try:
            if not kernel.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
                return None
            return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        finally:
            kernel.CloseHandle(handle)
    stat = Path('/proc')/str(pid)/'stat'
    return stat.read_text().split()[21] if stat.exists() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.name.replace('_', '').isalnum()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command and command[0] in ('gather.py', 'summarize.py')
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = OUT/(args.name+'_OWNER.json')
    if args.worker:
        with (OUT/(args.name+'.log')).open('ab') as output:
            code = subprocess.run([sys.executable, str(ROOT/'tools/entrant'/command[0]), *command[1:]],
                                  cwd=ROOT, stdout=output, stderr=subprocess.STDOUT).returncode
        (OUT/(args.name+'_DONE.json')).write_text(json.dumps({'returncode': code, 'finished': time.time()}))
        raise SystemExit(code)
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if incarnation(old['pid']) == old['incarnation']:
            raise RuntimeError('named owned local operation already live')
    log = (OUT/(args.name+'.log')).open('ab')
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
    child = subprocess.Popen([sys.executable, str(Path(__file__)), '--worker', args.name, '--', *command],
                             cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             creationflags=flags)
    owner = dict(pid=child.pid, incarnation=incarnation(child.pid), started=time.time(), command=command)
    receipt.write_text(json.dumps(owner, indent=1))
    print(json.dumps(owner))


if __name__ == '__main__':
    main()
