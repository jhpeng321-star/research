"""Sole durable telemetry/registration bridge for repair and same-phase WMFAST."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.candB.pilot_wmfast_dispatch import ROOT
from tools.candB.pilot_lean_observe import append, commit, DIRECTORY, LEDGER


def observe():
    code = f'''
import json,time
from pathlib import Path
root=Path({ROOT!r}); doc=dict(epoch=time.time(),root=str(root),owners={{}},outputs={{}},files={{}})
for name in ('OWNERSHIP.json','repair_TRAINER_OWNERSHIP.json','continue_TRAINER_OWNERSHIP.json'):
    p=root/name
    if p.exists():
        owner=json.loads(p.read_text()); proc=Path('/proc')/str(owner['pid'])
        owner['alive']=False
        if (proc/'stat').exists():
            fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
            owner.update(alive=fields[19]==str(owner['start_ticks']),state=fields[0],
                cpu_ticks=int(fields[11])+int(fields[12]),argv=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode())
        doc['owners'][name]=owner
for name in ('EXIT.json','repair_EXIT.json','continue_EXIT.json','BOUNDARY.json','GPU.jsonl','supervisor.log',
        'repair_trainer.log','continue_trainer.log'):
    p=root/name
    if p.exists(): doc['files'][name]=p.read_text()[-12000:]
for name in ('pilot_EXPLORATORY-LEAN','pilot_WMFAST_EXPLORATORY-LEAN'):
    output=root/name
    if not output.is_dir(): continue
    data=dict(rows=[],files={{}})
    for file in ('RUN_SUMMARY.json','LEAN_STARTUP.json','frozen_scoring_baseline.json',
        'pilot_STARTUP_REUSE_PROOF.json','pilot_WMFAST_CONTINUATION_PROOF.json'):
        if (output/file).exists():
            try: data['files'][file]=json.loads((output/file).read_text())
            except ValueError: pass
    if (output/'updates.jsonl').exists():
        for line in (output/'updates.jsonl').read_text().splitlines():
            try: data['rows'].append(json.loads(line))
            except ValueError: pass
    data['startup_scores']=[p.name for p in output.glob('pilot_startup_score_*.json')]
    doc['outputs'][name]=data
print(json.dumps(doc))
'''
    result = subprocess.run(['ssh','-o','ConnectTimeout=10','gpuA','/root/miniconda3/bin/python','-'],
        input=code, text=True, capture_output=True, timeout=25)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def record(doc):
    (DIRECTORY/'pilot_wmfast_latest.json').write_text(json.dumps(doc, indent=1))
    for kind in ('repair', 'continue'):
        owner = doc['owners'].get(kind+'_TRAINER_OWNERSHIP.json')
        marker = DIRECTORY/f'pilot_wmfast_{kind}_LAUNCH.json'
        if owner and not marker.exists():
            with marker.open('x') as stream:
                json.dump(owner, stream, indent=1)
            text = f'\n- S-PILOTB EXPLORATORY-LEAN {kind.upper()} LAUNCH {datetime.now(timezone.utc).isoformat()}: '+json.dumps(owner)+'\n'
            append(LEDGER, text)
            append(DIRECTORY/'PILOT.md', text)
            commit([marker,DIRECTORY/'PILOT.md'],f'S-PILOTB: receipt {kind} launch after lean gate repair')
    for output, data in doc['outputs'].items():
        for row in data['rows']:
            phase = row['phase_update']
            marker = DIRECTORY/f'pilot_wmfast_phase{phase}.json'
            if marker.exists():
                continue
            with marker.open('x') as stream:
                json.dump(dict(output=output,row=row,proof=data['files'].get('pilot_WMFAST_CONTINUATION_PROOF.json'),
                    startup=data['files'].get('LEAN_STARTUP.json')),stream,indent=1)
            text = (f'\n### EXPLORATORY-LEAN repaired/WMFAST phase{phase}\n'
                f'- Output `{output}`; startup={row.get("startup_s")}s; optimizer={row.get("optimizer_update_s")}s; incl rescore={row.get("update_including_rescore_s")}s.\n'
                f'- Raw training dJ={json.dumps(row.get("lean_training_raw_J"))}; independent={json.dumps(row.get("independent_J"))}; rescore skipped={row.get("independent_rescore_skipped")}.\n'
                f'- ES accepted={row.get("eslong_accepted_pairs")}; actor/dense/long norms={row.get("actor_grad_norm")}/{row.get("grad_norm_dense")}/{row.get("grad_norm_long")}.\n'
                f'- Executed={json.dumps(row.get("executed_residual_training"))}; group credit={json.dumps(row.get("group_credit_terms"))}; confident gains={json.dumps(row.get("confident_group_gain_rows"))}.\n'
                f'- Peak allocated/reserved={row.get("cuda_peak_allocated_bytes")}/{row.get("cuda_peak_reserved_bytes")}; worker peaks={json.dumps(row.get("pilot_worker_memory"))}; R4={json.dumps(row.get("R4"))}. Full row `{marker.name}`.\n')
            append(DIRECTORY/'PILOT.md',text)
            if phase <= 3 or phase % 5 == 0:
                append(LEDGER, f'\n- S-PILOTB EXPLORATORY-LEAN UPDATE{phase} {datetime.now(timezone.utc).isoformat()}: '+json.dumps(row)+'\n')
            paths = [marker,DIRECTORY/'PILOT.md']
            ack = DIRECTORY/'pilot_wmfast_update3_ACK.json'
            if phase >= 3 and not ack.exists():
                with ack.open('x') as stream:
                    json.dump(dict(phase_update=phase,row=row,output=output),stream,indent=1)
                paths.append(ack)
            commit(paths,f'S-PILOTB: receipt repaired/WMFAST phase update {phase}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spawn', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.once:
        doc = observe()
        with (DIRECTORY/f'pilot_wmfast_snapshot_{int(time.time())}.json').open('x') as stream:
            json.dump(doc,stream,indent=1)
        print(json.dumps(doc)); return
    if args.spawn:
        with (DIRECTORY/'pilot_wmfast_WATCH_DISPATCH.json').open('x') as receipt:
            with (DIRECTORY/'pilot_wmfast_watch.log').open('x') as log:
                child = subprocess.Popen([sys.executable,'-B','-u','-m','tools.candB.pilot_wmfast_watch'],
                    cwd=DIRECTORY.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
            json.dump(dict(pid=child.pid,epoch=time.time(),root=ROOT),receipt,indent=1)
        print(json.dumps(dict(observer_pid=child.pid))); return
    with (DIRECTORY/'pilot_wmfast_WATCH_OWNERSHIP.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(),epoch=time.time(),root=ROOT,observer_only=True),stream,indent=1)
    while not (DIRECTORY/'pilot_wmfast_WATCH_STOP').exists():
        try:
            doc = observe()
            record(doc)
            print(json.dumps(dict(epoch=doc['epoch'],phases={k:[r['phase_update'] for r in d['rows']]
                for k,d in doc['outputs'].items()},exit=doc['files'].get('EXIT.json'))),flush=True)
            if 'EXIT.json' in doc['files']:
                return
        except Exception as error:
            print(json.dumps(dict(epoch=time.time(),error=str(error))),flush=True)
        time.sleep(60)


if __name__ == '__main__':
    main()
