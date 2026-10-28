"""Launch only the registered owned queue, recording PID incarnation immediately."""
import argparse
from remote import ssh

p=argparse.ArgumentParser();p.add_argument('host',choices=('cpuP','cpuQ'))
p.add_argument('--canary',action='store_true');a=p.parse_args()
args=['--parity',str(int(a.host=='cpuQ')),'--workers','20']
if a.canary:args+=['--canary']
ssh(a.host, '''
import json,os,subprocess,time
from pathlib import Path
p=Path.home()/'entrant'; assert p.is_dir()
name='canary' if CANARY else 'queue'
receipt=p/(name+'_LAUNCH.json')
if receipt.exists():
    r=json.loads(receipt.read_text()); proc=Path('/proc')/str(r['pid'])
    if proc.exists() and proc.joinpath('stat').read_text().split()[21]==r['start_ticks']:
        raise RuntimeError('owned queue already live')
log=(p/(name+'.log')).open('ab')
cmd=[str(Path.home()/'miniconda3/bin/python'),str(p/'runtime/tools/entrant/coordinate.py')]+ARGS
child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21]
r={'pid':child.pid,'start_ticks':ticks,'argv':cmd,'launched':time.time()}
receipt.write_text(json.dumps(r));print(json.dumps(r))
'''.replace('CANARY',repr(a.canary)).replace('ARGS',repr(args)))
