"""Single-owned, capped panel supervisor. Durable plan/charges precede dispatch."""
from pathlib import Path
import concurrent.futures as cf
import fcntl
import json
import os
import sys
import time
from worker import ROOT,run,write
from admission import amendments,effective

WORLDS=(969421,969422,969425,969426,969427,969428,969432,969435)


def jobs():
    return [dict(id=f's{w}_{variant}_{arm}_r{rng}',world=w,variant=variant,arm=arm,rng=rng,horizon=600,charge=601)
        for w in WORLDS for variant in ('default','inferred')
        for arm in ('native','mappo','pi') for rng in (101,202,303)]


def main():
    assert os.getpriority(os.PRIO_PROCESS,0)==19
    lock=(ROOT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    panel=jobs()
    plan=dict(schema='sumocore/1',jobs=panel,hard_cap=100000,
              panel_precharge=86544,engineering_cap=1000,code=str(Path(__file__).parent))
    if (ROOT/'plan.json').exists():
        old=json.loads((ROOT/'plan.json').read_text())
        assert {k:v for k,v in old.items() if k!='code'}=={k:v for k,v in plan.items() if k!='code'}
    else:
        assert not (ROOT/'ledger.jsonl').exists()
        write(ROOT/'plan.json',plan)
        with (ROOT/'ledger.jsonl').open('x') as f:
            for j in panel:f.write(json.dumps(dict(id=j['id'],charge=j['charge'],kind='panel'))+'\n')
            f.flush();os.fsync(f.fileno())
    phase=sys.argv[1]
    mapping=amendments();panel=[effective(j,mapping) for j in panel]
    todo=panel[:1] if phase=='canary' else panel
    if phase!='canary':assert (ROOT/'jobs'/panel[0]['id']/'result.json').exists()
    todo=[j for j in todo if not any((ROOT/'jobs'/j['id']/name).exists() for name in ('result.json','recovered_result.json'))]
    # Never duplicate a failed, partial or currently-running directory.
    assert not any((ROOT/'jobs'/j['id']).exists() for j in todo)
    with cf.ProcessPoolExecutor(max_workers=1 if phase=='canary' else 8) as pool:
        pending={}
        for job in todo:
            while os.getloadavg()[0]>24:
                amendments();time.sleep(20)
            amendments()
            pending[pool.submit(run,job)]=job
        for future in cf.as_completed(pending):
            try:print('COMPLETE',future.result(),flush=True)
            except BaseException:
                stop=ROOT/'STOP' if not (ROOT/'STOP').exists() else ROOT/('STOP.'+pending[future]['id'])
                if not stop.exists():write(stop,dict(failed=pending[future]['id']))
                for queued in pending:queued.cancel()
                raise
    print('FINISHED',phase,flush=True)


if __name__=='__main__':main()
