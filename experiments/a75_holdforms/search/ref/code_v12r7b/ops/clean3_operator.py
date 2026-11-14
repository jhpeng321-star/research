"""Bounded F6 operator execution; retain the remote JSON receipt locally."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--script', required=True)
parser.add_argument('--receipt', required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
script = (repo/'ops'/args.script).resolve()
assert script.parent == (repo/'ops').resolve()
target = Path(args.receipt)
assert target.parent.is_dir() and not target.exists()
source = json.loads((repo/'CLEAN2_PACKAGE.json').read_text())['source_root']
remote = '. /home/ubuntu/cpus_env/env.sh '+shlex.quote(source)+' && exec /home/ubuntu/miniconda3/bin/python -B -'
helper = ("import sys;sys.dont_write_bytecode=True;sys.path.insert(0,'/mnt/e/agent-work/c4-rerun-launch');"
    "from tools.coopmwm.cpuv_ssh import Transport;"
    "r=Transport().run(sys.argv[1],timeout=55,input=sys.stdin.buffer.read(),capture_output=True);"
    "sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)")
result = subprocess.run(['wsl', '-d', 'Ubuntu-22.04', '--', 'python3', '-c', helper, remote],
    input=script.read_bytes(), capture_output=True, timeout=70)
if result.returncode:
    sys.stderr.buffer.write(result.stderr)
    raise SystemExit(result.returncode)
value = json.loads(result.stdout.decode('utf-8').replace('\x00', ''))
with target.open('x', encoding='utf-8') as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
print(json.dumps(dict(receipt=str(target), timestamp=value.get('timestamp'),
    status=value.get('status')), sort_keys=True))
