"""F6 shared pool with a local OpenSSH double-master-flag correction.

OpenSSH interprets `-o ControlMaster=yes -M` as ControlMaster=ask. In an
unattended process this rejects multiplexed sessions. Use auto before the
existing -M, producing yes. Preserve the shared gate/socket and all lifecycle,
rate, strict-host-key and no-direct-fallback guarantees. Never retire a master.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, '/mnt/e/agent-work/c4-fix-ssh')
from tools.coopmwm.cpuv_ssh import Transport


class OperatorTransport(Transport):
    def _options(self, *, master=False):
        options = super()._options(master=master)
        if master:
            options = ['ControlMaster=auto' if value == 'ControlMaster=yes' else value for value in options]
        return options


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--timeout', type=float, default=40)
    p.add_argument('--put', nargs=2, metavar=('LOCAL', 'REMOTE'))
    p.add_argument('command', nargs='?')
    a = p.parse_args()
    if bool(a.put) == bool(a.command):
        p.error('supply one command or --put LOCAL REMOTE')
    transport = OperatorTransport()
    result = (transport.copy(*a.put, upload=True, timeout=a.timeout) if a.put else
              transport.run(a.command, timeout=a.timeout))
    sys.exit(result.returncode)
