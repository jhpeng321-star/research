"""Start/adopt or poll one SHA-bound cpuV-local operator job via accepted F6."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('action', choices=('start', 'poll'))
parser.add_argument('--job', required=True)
parser.add_argument('--script')
parser.add_argument('--source')
parser.add_argument('--receipt', required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
assert args.job.replace('_', '').replace('-', '').isalnum()
target = Path(args.receipt)
assert target.parent.is_dir() and not target.exists()
source = args.source or json.loads((repo/'CLEAN2_PACKAGE.json').read_text())['source_root']
script = (repo/'ops'/args.script).read_text(encoding='utf-8') if args.script else None
assert args.action != 'start' or script is not None
payload = dict(action=args.action, job=args.job, source=source, script=script)
code = r'''
import hashlib,json,os,shlex,subprocess,time
from pathlib import Path
r=PAYLOAD
c=Path('/home/ubuntu/cpuv_setup_20261105')/r['job']
assert c.parent.is_dir()
if r['action']=='start':
    sha=hashlib.sha256(r['script'].encode()).hexdigest()
    if not c.exists():
        c.mkdir()
        with (c/'OPERATOR.py').open('x') as f:f.write(r['script'])
        worker="""import json,subprocess,time
from pathlib import Path
c=Path(CONTROL);start=time.time()
with (c/'job.log').open('x') as log:
    result=subprocess.run(['bash','-lc',COMMAND],stdout=log,stderr=subprocess.STDOUT)
with (c/'RESULT.json').open('x') as f:
    json.dump(dict(returncode=result.returncode,seconds=time.time()-start),f)
""".replace('CONTROL',repr(str(c))).replace('COMMAND',repr(
            '. /home/ubuntu/cpus_env/env.sh '+shlex.quote(r['source'])+
            ' && exec /home/ubuntu/miniconda3/bin/python -B -u '+shlex.quote(str(c/'OPERATOR.py'))))
        with (c/'DISPATCH_STARTED.json').open('x') as f:
            json.dump(dict(source=r['source'],script_sha256=sha,unix=time.time()),f)
        with (c/'coordinator.log').open('x') as log:
            p=subprocess.Popen(['/usr/bin/python3','-B','-u','-c',worker],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        owner=dict(pid=p.pid,start_ticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21])
        with (c/'OWNER.json').open('x') as f:json.dump(owner,f)
    assert hashlib.sha256((c/'OPERATOR.py').read_bytes()).hexdigest()==sha
assert c.is_dir()
owner=json.loads((c/'OWNER.json').read_text())
p=Path('/proc')/str(owner['pid'])/'stat'
fields=p.read_text().split() if p.exists() else []
alive=bool(fields and fields[2]!='Z' and fields[21]==owner['start_ticks'])
result=json.loads((c/'RESULT.json').read_text()) if (c/'RESULT.json').exists() else None
log=(c/'job.log').read_text(errors='replace') if (c/'job.log').exists() else ''
print(json.dumps(dict(control=str(c),owner=owner,alive=alive,result=result,
    dispatch=json.loads((c/'DISPATCH_STARTED.json').read_text()),
    log=log if result else log[-8000:],timestamp=time.time())))
'''.replace('PAYLOAD', repr(payload))
helper = ("import sys;sys.dont_write_bytecode=True;sys.path.insert(0,'/mnt/e/agent-work/c4-rerun-launch');"
    "from tools.coopmwm.cpuv_ssh import Transport;"
    "r=Transport().run(sys.argv[1],timeout=45,input=sys.stdin.buffer.read(),capture_output=True);"
    "sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)")
result = subprocess.run(['wsl', '-d', 'Ubuntu-22.04', '--', 'python3', '-c', helper,
    'python3 -c '+shlex.quote(code)], capture_output=True, timeout=60)
if result.returncode:
    raise RuntimeError(result.stderr.decode(errors='replace'))
value = json.loads(result.stdout.decode().replace('\x00', ''))
with target.open('x', encoding='utf-8') as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
print(json.dumps(dict(receipt=str(target),control=value['control'],alive=value['alive'],result=value['result'])))
