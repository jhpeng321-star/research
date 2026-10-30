"""Sole durable telemetry/registration bridge for the CONT0525 S2 continuation pilot."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.candB.pilot_lean_observe import append, commit, DIRECTORY, LEDGER

ROOT = '/root/autodl-tmp/c4/candB_pilot_run_CONT0525_v1_20261031'
OUTPUT = ROOT+'/pilot_EXPLORATORY-LEAN_CONT0525'
LAUNCHED = {}


def observe():
    code = f'''
import json,time
from pathlib import Path
root=Path({ROOT!r}); doc=dict(epoch=time.time(),root=str(root),owners={{}},files={{}},output={{}},updates=[])
for name in ('OWNERSHIP.json','TRAINER_OWNERSHIP.json'):
    p=root/name
    if p.exists():
        owner=json.loads(p.read_text()); proc=Path('/proc')/str(owner['pid'])
        owner['alive']=False
        if (proc/'stat').exists():
            fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
            owner.update(alive=fields[19]==str(owner['start_ticks']),state=fields[0],
                cpu_ticks=int(fields[11])+int(fields[12]))
        doc['owners'][name]=owner
for name in ('EXIT.json','GPU.jsonl','supervisor.log','trainer.log'):
    p=root/name
    if p.exists(): doc['files'][name]=p.read_text()[-8000:]
out=Path({OUTPUT!r})
for file in ('RUN_SUMMARY.json','pilot_WMFAST_CONTINUATION_PROOF.json','frozen_scoring_baseline.json'):
    if (out/file).exists():
        try: doc['output'][file]=json.loads((out/file).read_text())
        except ValueError: pass
if (out/'updates.jsonl').exists():
    for line in (out/'updates.jsonl').read_text().splitlines():
        try: doc['updates'].append(json.loads(line))
        except ValueError: pass
print(json.dumps(doc))
'''
    result = subprocess.run(['ssh', '-o', 'ConnectTimeout=10', 'gpuA', '/root/miniconda3/bin/python', '-'],
        input=code, text=True, capture_output=True, timeout=25)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def record(doc):
    (DIRECTORY/'pilot_cont0525_latest.json').write_text(json.dumps(doc, indent=1))
    for phase_row in doc['updates']:
        phase = phase_row.get('phase_update')
        if phase is None:
            continue  # raw trainer rows carry no phase clock; skip like receipts do
        marker = DIRECTORY/f'pilot_cont0525_phase{phase}.json'
        if marker.exists():
            continue
        with marker.open('x') as stream:
            json.dump(dict(output=OUTPUT, row=phase_row,
                proof=doc['output'].get('pilot_WMFAST_CONTINUATION_PROOF.json')), stream, indent=1)
        text = (f'\n### EXPLORATORY-LEAN CONT0525 phase{phase}\n'
            f'- optimizer={phase_row.get("optimizer_update_s")}s; incl rescore={phase_row.get("update_including_rescore_s")}s.\n'
            f'- Raw training dJ={json.dumps(phase_row.get("lean_training_raw_J"))}; rescore skipped={phase_row.get("independent_rescore_skipped")}.\n'
            f'- ES accepted={phase_row.get("eslong_accepted_pairs")}; stack sides={phase_row.get("eslong_stack_sides")}; actor/dense/long norms={phase_row.get("actor_grad_norm")}/{phase_row.get("grad_norm_dense")}/{phase_row.get("grad_norm_long")}.\n'
            f'- Executed={json.dumps(phase_row.get("executed_residual_training"))}; group credit={json.dumps(phase_row.get("group_credit_terms"))}.\n'
            f'- Peak={phase_row.get("cuda_peak_allocated_bytes")}/{phase_row.get("cuda_peak_reserved_bytes")}; workers={json.dumps(phase_row.get("pilot_worker_memory"))}; R4={json.dumps(phase_row.get("R4"))}. Full row `{marker.name}`.\n')
        append(DIRECTORY/'PILOT.md', text)
        if phase <= 3 or phase % 5 == 0:
            append(LEDGER, f'\n- S-PILOTB EXPLORATORY-LEAN CONT0525 UPDATE{phase} '+
                f'{datetime.now(timezone.utc).isoformat()}: '+json.dumps(phase_row)+'\n')
        paths = [marker, DIRECTORY/'PILOT.md']
        if phase <= 2:
            ack = DIRECTORY/f'pilot_cont0525_update{phase}_ACK.json'
            if not ack.exists():
                with ack.open('x') as stream:
                    json.dump(dict(phase_update=phase, observed_utc=datetime.now(timezone.utc).isoformat(),
                        row=phase_row, epoch=time.time()), stream, indent=1)
                paths.append(ack)
        commit(paths, f'S-PILOTB: receipt CONT0525 phase update {phase}')


def main():
    parser_like = sys.argv[1:]
    if parser_like and parser_like[0] == '--spawn':
        with (DIRECTORY/'pilot_cont0525_WATCH_DISPATCH.json').open('x') as receipt:
            with (DIRECTORY/'pilot_cont0525_watch.log').open('x') as log:
                child = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'tools.candB.pilot_cont0525_watch'],
                    cwd=DIRECTORY.parents[1], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
                json.dump(dict(pid=child.pid, epoch=time.time(), root=ROOT), receipt, indent=1)
        print(json.dumps(dict(observer_pid=child.pid)))
        return
    if parser_like and parser_like[0] == '--once':
        doc = observe()
        with (DIRECTORY/f'pilot_cont0525_snapshot_{int(time.time())}.json').open('x') as stream:
            json.dump(doc, stream, indent=1)
        print(json.dumps(dict(epoch=doc['epoch'], updates=[r.get('phase_update') for r in doc['updates']],
            exit='EXIT.json' in doc['files'])))
        return
    with (DIRECTORY/'pilot_cont0525_WATCH_OWNERSHIP.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), epoch=time.time(), root=ROOT, observer_only=True), stream, indent=1)
    while not (DIRECTORY/'pilot_cont0525_WATCH_STOP').exists():
        try:
            doc = observe()
            record(doc)
            print(json.dumps(dict(epoch=doc['epoch'], updates=[r.get('phase_update') for r in doc['updates']],
                exit='EXIT.json' in doc['files'])), flush=True)
            if 'EXIT.json' in doc['files']:
                return
        except Exception as error:
            print(json.dumps(dict(epoch=time.time(), error=str(error))), flush=True)
        time.sleep(60)


if __name__ == '__main__':
    main()
