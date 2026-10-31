"""Read-only resource/path audit for INT; never launch or signal peer jobs."""
import argparse
import subprocess


def ssh(host, code, timeout=45):
    if host not in ('gpuA', 'gpuR', 'cpuQ'):
        raise ValueError(host)
    python = '/home/ubuntu/miniconda3/bin/python' if host == 'cpuQ' else '/root/miniconda3/bin/python'
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             host, python+' -B -'],
                            input=code, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stdout + '\n' + result.stderr)
    return result.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument('host', choices=('gpuA', 'gpuR', 'cpuQ'))
    args = p.parse_args()
    print(ssh(args.host, """
import os, json, subprocess
from pathlib import Path
print('LOAD', os.getloadavg(), 'CPUS', os.cpu_count())
print(subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.total,utilization.gpu',
                     '--format=csv'], capture_output=True, text=True).stdout)
paths = [
 '/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt',
 '/root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml',
 '/root/autodl-tmp/c4/setupaudit20261030/policy_000190.pt',
 '/root/autodl-tmp/c4/speedhead20261030/EXPORT_BOUND.json',
 '/root/autodl-tmp/c4/speedhead20261030/e3final/runtime',
 '/root/autodl-tmp/c4/speedhead20261030/pythondeps',
 '/root/wmcalib/data', '/root/wmcalib', '/root/rbint20261031']
for name in paths:
 p=Path(name)
 print('PATH', name, 'dir' if p.is_dir() else 'file' if p.is_file() else 'MISSING')
own=Path('/root/rbint20261031')
if own.is_dir():
 for receipt in sorted(own.glob('DISPATCH_*.json')):
  print('OWN_DISPATCH', receipt.name, receipt.read_text())
 for done in sorted(own.glob('*/COMPLETE.json')):
  print('OWN_COMPLETE', done.parent.name, done.read_text())
"""))


if __name__ == '__main__':
    main()
