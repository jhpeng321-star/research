"""Retire only this task's identity-verified, idle local F6 ask-mode master.

No remote command, training control, socket deletion, or process-name signal.
Run on the shared WSL user after the noninteractive F6 fix is committed.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from tools.coopmwm.cpuv_ssh import HOST, Transport, _deadline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    receipt = Path(args.receipt)
    assert receipt.parent.is_dir() and not receipt.exists()
    client = Transport()
    deadline = _deadline(35)
    result = dict(unix=time.time(), local_only=True, remote_commands=0, retired=None)
    with client.gate.claim(deadline=deadline):
        checked = client._owned_run(['ssh', *client._options(), '-O', 'check', HOST],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=5)
        if checked.returncode == 0:
            match = re.fullmatch(rb'Master running \(pid=(\d+)\)\s*', checked.stderr)
            assert match, checked.stderr
            pid = int(match[1])
            proc = Path('/proc')/str(pid)
            assert proc.stat().st_uid == os.getuid()
            argv = (proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
            expected = ['ssh', *client._options(master=True), '-M', '-N', '-f', HOST]
            legacy = ['ControlMaster=yes' if x == 'ControlMaster=auto' else x for x in expected]
            assert argv in (expected, legacy), 'not this task F6 master; refuse any signal'
            if argv == legacy:
                before = (proc/'stat').read_text().split()[21]
                handle = os.pidfd_open(pid)
                try:
                    assert (proc/'stat').read_text().split()[21] == before
                    assert (proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0') == legacy
                    # The shared gate has no active session here. pidfd pins this
                    # exact owned local transport process, never a research PID.
                    signal.pidfd_send_signal(handle, signal.SIGTERM)
                    limit = time.monotonic()+5
                    while client.socket.exists() and time.monotonic() < limit:
                        time.sleep(0.05)
                    assert not client.socket.exists(), 'master did not release its own socket'
                finally:
                    os.close(handle)
                result['retired'] = dict(pid=pid, start_ticks=before, argv=legacy)
        client._master(deadline)
        verified = client._owned_run(['ssh', *client._options(), '-O', 'check', HOST],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=5)
        assert verified.returncode == 0, verified.stderr
        result['master_check'] = verified.stderr.decode().strip()
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
