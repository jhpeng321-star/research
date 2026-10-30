"""Exclusive durable lean receipt bridge: launch, startup, every update and phase3 ACK."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.candB.pilot_lean_deploy import ROOT

DIRECTORY=Path(__file__).resolve().parent
LEDGER=Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')


def observe():
    code=f'''
import json,time
from pathlib import Path
root=Path({ROOT!r}); output=root/'pilot_EXPLORATORY-LEAN'
doc=dict(epoch=time.time(),root=str(root),files={{}},owners={{}},rows=[])
for name in ('OWNERSHIP.json','TRAINER_OWNERSHIP.json'):
    p=root/name
    if p.exists():
        owner=json.loads(p.read_text()); proc=Path('/proc')/str(owner['pid'])
        if (proc/'stat').exists():
            fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
            owner.update(same_incarnation=fields[19]==str(owner['start_ticks']),state=fields[0],
                cpu_ticks=int(fields[11])+int(fields[12]),argv=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode())
        doc['owners'][name]=owner
for name in ('GPU.jsonl','EXIT.json','supervisor.log','trainer.log'):
    p=root/name
    if p.exists(): doc['files'][name]=p.read_text()[-10000:]
for name in ('RUN_SUMMARY.json','LEAN_STARTUP.json','frozen_scoring_baseline.json'):
    p=output/name
    if p.exists(): doc['files'][name]=json.loads(p.read_text())
p=output/'updates.jsonl'
if p.exists():
    for line in p.read_text().splitlines():
        try: doc['rows'].append(json.loads(line))
        except ValueError: pass
print(json.dumps(doc))
'''
    result=subprocess.run(['ssh','gpuA','/root/miniconda3/bin/python','-'],input=code,
        text=True,capture_output=True,timeout=25)
    if result.returncode: raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def append(path,text):
    with path.open('a',encoding='utf-8') as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())


def commit(paths, message):
    worktree=DIRECTORY.parents[1]
    for command in (['git','status','--short'],['git','diff','--','tools/candB/PILOT.md'],['git','log','--oneline','-10']):
        subprocess.run(command,cwd=worktree,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=worktree).returncode:
        raise RuntimeError('other staged work present; preserve receipts pending commit')
    subprocess.run(['git','add','--',*[str(p.relative_to(worktree)) for p in paths]],cwd=worktree,check=True)
    subprocess.run(['git','commit','-m',message],cwd=worktree,check=True)


def record(doc):
    receipt=DIRECTORY/'pilot_lean_latest.json'
    receipt.write_text(json.dumps(doc,indent=1))
    for phase,row in [(0,doc['owners'].get('TRAINER_OWNERSHIP.json'))]+[(r['phase_update'],r) for r in doc['rows']]:
        if row is None: continue
        path=DIRECTORY/f'pilot_lean_phase{phase}.json'
        if path.exists(): continue
        with path.open('x') as stream:
            json.dump(dict(label='EXPLORATORY-LEAN',observed_utc=datetime.now(timezone.utc).isoformat(),
                receipt=row,startup=doc['files'].get('LEAN_STARTUP.json')),stream,indent=1)
        if phase==0:
            text=f'\n### EXPLORATORY-LEAN launch ACK\n- gpuA owned trainer `{row}`; original dry phase retired, sunk8530.66s.\n'
        else:
            text=(f'\n### EXPLORATORY-LEAN phase{phase}\n'
                f'- Startup={row.get("startup_s")}s; optimizer={row.get("optimizer_update_s")}s; including rescore={row.get("update_including_rescore_s")}s.\n'
                f'- Raw training pre-optimizer dJ={json.dumps(row.get("lean_training_raw_J"))}; independent post-update score={json.dumps(row.get("independent_J"))}; rescore skipped={row.get("independent_rescore_skipped")}.\n'
                f'- ES accepted pairs={row.get("eslong_accepted_pairs")}; gradients actor/dense/long={row.get("actor_grad_norm")}/{row.get("grad_norm_dense")}/{row.get("grad_norm_long")}.\n'
                f'- Executed residual={json.dumps(row.get("executed_residual_training"))}; group terms={json.dumps(row.get("group_credit_terms"))}; confident gains={json.dumps(row.get("confident_group_gain_rows"))}.\n'
                f'- Peak allocated/reserved={row.get("cuda_peak_allocated_bytes")}/{row.get("cuda_peak_reserved_bytes")}; worker peaks={json.dumps(row.get("pilot_worker_memory"))}; R4={json.dumps(row.get("R4"))}. Full row `{path.name}`.\n')
        append(DIRECTORY/'PILOT.md',text)
        if phase<=3 or phase%5==0:
            append(LEDGER,f'\n- S-PILOTB EXPLORATORY-LEAN '+('LAUNCH' if phase==0 else f'UPDATE{phase}')+
                f' {datetime.now(timezone.utc).isoformat()}: '+json.dumps(row)+'\n')
        paths=[path,DIRECTORY/'PILOT.md']
        if phase>=3 and not (DIRECTORY/'pilot_lean_update3_ACK.json').exists():
            ack=DIRECTORY/'pilot_lean_update3_ACK.json'
            with ack.open('x') as stream:
                json.dump(dict(label='EXPLORATORY-LEAN',phase_update=phase,observed_utc=datetime.now(timezone.utc).isoformat(),row=row),stream,indent=1)
            paths.append(ack)
        commit(paths,f'S-PILOTB: receipt lean '+('launch' if phase==0 else f'phase update {phase}'))
    return doc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spawn',action='store_true')
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if args.once:
        doc=observe()
        with (DIRECTORY/f'pilot_lean_snapshot_{int(time.time())}.json').open('x') as stream: json.dump(doc,stream,indent=1)
        print(json.dumps(doc)); return
    if args.spawn:
        with (DIRECTORY/'pilot_lean_OBSERVER_DISPATCH.json').open('x') as receipt:
            with (DIRECTORY/'pilot_lean_observer.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-B','-u','-m','tools.candB.pilot_lean_observe'],
                    cwd=DIRECTORY.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
            json.dump(dict(pid=child.pid,epoch=time.time(),root=ROOT,observer_only=True),receipt,indent=1)
        print(json.dumps(dict(observer_pid=child.pid))); return
    with (DIRECTORY/'pilot_lean_OBSERVER_OWNERSHIP.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(),epoch=time.time(),root=ROOT,poll_s=60,trainer_dispatch=False),stream,indent=1)
    while not (DIRECTORY/'pilot_lean_OBSERVER_STOP').exists():
        try:
            doc=record(observe())
            print(json.dumps(dict(epoch=time.time(),phases=[r['phase_update'] for r in doc['rows']],exit=doc['files'].get('EXIT.json'))),flush=True)
            if 'EXIT.json' in doc['files']: return
        except Exception as error:
            print(json.dumps(dict(epoch=time.time(),error=str(error))),flush=True)
        time.sleep(60)


if __name__=='__main__':
    main()
