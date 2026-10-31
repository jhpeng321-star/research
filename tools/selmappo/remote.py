"""Small explicit SSH operations; no process-name signaling or dataset reads."""
import base64
import subprocess
import sys


def run(host, code, *, stdin=None, timeout=60):
    encoded = base64.b64encode(code.encode()).decode()
    python = '/home/ubuntu/miniconda3/bin/python' if host == 'cpuQ' else '/root/miniconda3/bin/python'
    command = f"{python} -c 'import base64; exec(base64.b64decode(\"{encoded}\"))'"
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, command],
                            capture_output=True, text=True, input=stdin, timeout=timeout)
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    result.check_returncode()
    return result


if __name__ == '__main__':
    run(sys.argv[1], '''
import os,json
from pathlib import Path
p=Path.home()/'selmappo'
print(json.dumps(dict(root=str(p),exists=p.exists(),entries=[x.name for x in p.iterdir()] if p.exists() else [],load=os.getloadavg(),cpus=os.cpu_count())))
for base in (Path.home()/'selmappo', Path.home()/'selmappo20261031'):
 print('OWN_ROOT',str(base),base.exists())
 if base.exists():
  for f in base.glob('*.json'):
   print(str(f), f.read_text())
for proc in Path('/proc').iterdir():
 if proc.name.isdigit():
  try:
   args=(proc/'cmdline').read_bytes().split(bytes([0]))
   if any(b'tools.selmappo.train' in a for a in args):
    print('SELECTOR_PID',proc.name,[a.decode(errors='replace') for a in args])
  except (FileNotFoundError, PermissionError, ProcessLookupError):
   pass
if p.exists():
 for f in p.glob('*.json'):
  print(str(f), f.read_text())
config=Path.home()/'mappospeed/speed.toml'
print(config.read_text() if config.exists() else 'NO_LOCAL_MAPPO_CONFIG')
for root in (Path.home()/'c1switch20261031',Path.home()/'c1switch',Path.home()/'e3final',Path.home()/'wmdata2'):
 print(str(root), root.exists(), [x.name for x in root.iterdir()] if root.exists() else [])
print('MEMORY', Path('/proc/meminfo').read_text().splitlines()[:3])
''')
