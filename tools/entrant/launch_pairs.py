"""Score only the chosen combination on the registered pairs300 panel."""
import argparse
from remote import detach

parser = argparse.ArgumentParser()
parser.add_argument('host', choices=('cpuP', 'cpuQ'))
parser.add_argument('variant', choices=('baseline', 'ramp_adapt_off', 'entrant_on', 'entrant_on_ramp_off'))
args = parser.parse_args()

detach(args.host, 'pairs', '''
import fcntl,json,os,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
p=Path.home()/'entrant';assert p.is_dir()
if (p/'OWNER.json').exists():
    old=json.loads((p/'OWNER.json').read_text());stat=Path('/proc')/str(old['pid'])/'stat'
    assert not (stat.exists() and stat.read_text().split()[21]==old['start_ticks']),'E3 queue still live'
sys.path[:0]=[str(p/'runtime/tools/entrant')]
import coordinate as co
import run
run.setup();os.nice(10)
with (p/'PAIRS_QUEUE.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner={'pid':os.getpid(),'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
           'parity':PARITY,'workers':20,'panel':'pairs300','variant':VARIANT,'started':time.time()}
    co.atomic(p/'PAIRS_OWNER.json',owner)
    variants=list(dict.fromkeys(('baseline',VARIANT)))
    units=[(g,k,v,600,'pairs300',g==0 and k==0 and v.startswith('entrant_on'))
           for v in variants for g in range(48) if g%2==PARITY for k in range(8)]
    results=[]
    with ProcessPoolExecutor(max_workers=20) as pool:
        for future in as_completed([pool.submit(co.work,u) for u in units]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
            co.atomic(p/'PAIRS_PROGRESS.json',{'owner':owner,'done':len(results),'total':len(units),
                'errors':sum('error' in r for r in results),'updated':time.time()})
    co.atomic(p/'PAIRS_DONE.json',results)
'''.replace('PARITY', str(int(args.host == 'cpuQ'))).replace('VARIANT', repr(args.variant)))
