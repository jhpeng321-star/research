"""Send zero-physics operator sources over the accepted shared F6 transport."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
parser = argparse.ArgumentParser()
parser.add_argument('--receipt', required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
target = Path(args.receipt)
assert target.parent.is_dir() and not target.exists()
sources = {name: (repo/'tools/coopmwm'/f'{name}.py').read_text(encoding='utf-8')
    for name in ('launch_evaluation', 'clean3_budget')}
approval = json.loads((repo/'BUDGET_CLEAN3.json').read_text(encoding='utf-8'))
source = json.loads((repo/'CLEAN2_PACKAGE.json').read_text())['source_root']
code = ('OPERATOR_SOURCES='+repr(sources)+'\nOWNER_BUDGET='+repr(approval)+'\n'
    +(repo/'ops/clean3_quote_stopped_remote.py').read_text(encoding='utf-8')).encode()
remote = '. /home/ubuntu/cpus_env/env.sh '+shlex.quote(source)+' && exec /home/ubuntu/miniconda3/bin/python -B -'
helper = ("import sys;sys.dont_write_bytecode=True;sys.path.insert(0,'/mnt/e/agent-work/c4-rerun-launch');"
    "from tools.coopmwm.cpuv_ssh import Transport;"
    "r=Transport().run(sys.argv[1],timeout=55,input=sys.stdin.buffer.read(),capture_output=True);"
    "sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)")
result = subprocess.run(['wsl', '-d', 'Ubuntu-22.04', '--', 'python3', '-c', helper, remote],
    input=code, capture_output=True, timeout=70)
if result.returncode:
    print(result.stderr.decode('utf-8', errors='replace'), file=sys.stderr)
    raise SystemExit(result.returncode)
value = json.loads(result.stdout.decode('utf-8').replace('\x00', ''))
with target.open('x', encoding='utf-8') as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
quote = value['quote']
print(json.dumps(dict(receipt=str(target), timestamp=value['timestamp'], status=quote['status'],
    prior_charged_sim_s=quote['prior_charged_sim_s'], lesson_costs=quote['lesson_costs'],
    clean3_allocation_sim_s=quote['allocated_sim_s'],
    protected_plan_fix_sim_s=quote['plan_fix_unspent_reserved_sim_s'],
    mining=value['mining'], overall_bound_sim_s=value['overall_cumulative_bound_sim_s'],
    overall_shortfall_sim_s=value['overall_shortfall_sim_s'], launch_authorized=False), sort_keys=True))
