"""Read only this pilot's remote artifacts; append launch/5-update ledger entries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


REMOTE = r'''
import json, os, sys, time
from pathlib import Path
root = Path(sys.argv[1])
out = dict(label='EXPLORATORY', root=str(root), epoch=time.time(), stages={})
for name in ('OWNERSHIP.json', 'dry_OWNERSHIP.json', 'pilot_OWNERSHIP.json', 'dry_EXIT.json', 'pilot_EXIT.json'):
    path = root/name
    if path.is_file():
        doc = json.loads(path.read_text())
        if 'pid' in doc:
            proc = Path('/proc')/str(doc['pid'])/'stat'
            doc['same_incarnation_alive'] = proc.is_file() and proc.read_text().rsplit(')',1)[1].split()[19] == doc['start_ticks']
        out[name] = doc
admission = root/'GPU_ADMISSION.jsonl'
if admission.is_file():
    out['last_gpu_admission'] = json.loads(admission.read_text().splitlines()[-1])
for mode in ('dry', 'pilot'):
    stage = root/(mode+'_EXPLORATORY')
    doc = dict(rows=[])
    for name in ('RUN_SUMMARY.json', 'frozen_scoring_baseline.json', 'gate_panel_result.json'):
        path = stage/name
        if path.is_file():
            value = json.loads(path.read_text())
            if name == 'RUN_SUMMARY.json':
                value = {k:v for k,v in value.items() if k not in ('updates', 'native_curve')}
            doc[name] = value
    updates = stage/'updates.jsonl'
    if updates.is_file():
        for line in updates.read_text().splitlines():
            row = json.loads(line)
            keys = ('update','phase_update','t','label','actor_loss','critic_loss','actor_grad_norm',
                'critic_grad_norm','actor_module_grad_norms','grad_norm_dense','grad_norm_long',
                'grad_norm_long_es','grad_norm_combined','eslong_accepted_pairs','eslong_rejected_pairs',
                'eslong_sigma','eslong_pairs','independent_J','cuda_peak_allocated_bytes',
                'cuda_peak_reserved_bytes','optimizer_update_s','update_including_rescore_s','R4')
            doc['rows'].append({k:row[k] for k in keys if k in row})
    log = root/(mode+'.log')
    if log.is_file():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-6000))
            doc['log_tail'] = stream.read().decode(errors='replace')
    out['stages'][mode] = doc
print(json.dumps(out, allow_nan=False))
'''


def observe(host, root):
    result = subprocess.run(['ssh', host, '/root/miniconda3/bin/python', '-', root],
                            input=REMOTE, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def record(doc, directory, ledger):
    state_path = directory/'pilot_observe_state.json'
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    utc = datetime.now(timezone.utc).isoformat()
    entries = []
    for mode, stage in doc['stages'].items():
        owned = doc.get(mode+'_OWNERSHIP.json')
        if owned and not state.get(mode+'_launched'):
            entries.append(f'- S-PILOTB {mode.upper()} LAUNCH ACK {utc}: EXPLORATORY '
                f'gpuA PID{owned["pid"]}/start ticks{owned["start_ticks"]}, output `{owned["output"]}`. '
                f'Runtime `{doc["OWNERSHIP.json"]["runtime"]}`; '
                'only own PIDs, peer jobs untouched. No SUMO.')
            state[mode+'_launched'] = True
        for row in stage['rows']:
            phase = row.get('phase_update', 0)
            if phase and phase % 5 == 0 and phase > state.get(mode+'_last_registered', 0):
                score = row.get('independent_J', {})
                entries.append(f'- S-PILOTB {mode.upper()} UPDATE{phase} {utc}: EXPLORATORY; '
                    f'raw dJ dense={score.get("raw_dJ_vs_CTRL0_dense")}, '
                    f'300s={score.get("raw_dJ_vs_CTRL0_long")}; '
                    f'ES accepted={row.get("eslong_accepted_pairs")}, '
                    f'grad actor/dense/long={row.get("actor_grad_norm")}/{row.get("grad_norm_dense")}/{row.get("grad_norm_long")}; '
                    f'update incl rescore s={row.get("update_including_rescore_s")}; '
                    f'peak reserved bytes={row.get("cuda_peak_reserved_bytes")}; '
                    f'executed={json.dumps(score.get("executed_residual"), separators=(",", ":"))}; '
                    f'R4={json.dumps(row.get("R4"))}.')
                state[mode+'_last_registered'] = phase
    if entries and ledger:
        with ledger.open('a', encoding='utf-8') as stream:
            stream.write('\n'+'\n'.join(entries)+'\n')
    state_path.write_text(json.dumps(state, indent=1)+'\n')
    snapshot = directory/f'pilot_observation_{int(doc["epoch"])}.json'
    snapshot.write_text(json.dumps(doc, indent=1)+'\n')
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='gpuA')
    p.add_argument('--root', required=True)
    p.add_argument('--ledger', type=Path)
    p.add_argument('--follow-through-update', type=int)
    p.add_argument('--brief', action='store_true')
    p.add_argument('--peer-gpu', default=None, help='optional second host, sampled at the same >=600s cadence')
    args = p.parse_args()
    directory = Path(__file__).resolve().parent
    while True:
        doc = observe(args.host, args.root)
        if args.peer_gpu:
            peer = subprocess.run(['ssh', args.peer_gpu, 'nvidia-smi',
                '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
                text=True, capture_output=True, check=True)
            doc['peer_gpu'] = dict(host=args.peer_gpu, memory_MiB=peer.stdout.strip(), epoch=time.time())
        entries = record(doc, directory, args.ledger)
        if args.brief:
            report = dict(utc=datetime.now(timezone.utc).isoformat(),
                own_supervisor_alive=doc.get('OWNERSHIP.json', {}).get('same_incarnation_alive'),
                last_gpu=doc.get('last_gpu_admission'), ledger_entries=entries,
                peer_gpu=doc.get('peer_gpu'),
                stages={mode: dict(updates=len(stage['rows']),
                    status=stage.get('RUN_SUMMARY.json', {}).get('status'),
                    error=stage.get('RUN_SUMMARY.json', {}).get('error'),
                    own_pid=doc.get(mode+'_OWNERSHIP.json', {}).get('pid'))
                    for mode, stage in doc['stages'].items()})
        else:
            report = dict(snapshot=doc, ledger_entries=entries)
        print(json.dumps(report), flush=True)
        if not args.follow_through_update:
            break
        pilot_rows = doc['stages']['pilot']['rows']
        if (pilot_rows and pilot_rows[-1].get('phase_update', 0) >= args.follow_through_update
                or 'dry_EXIT.json' in doc and doc['dry_EXIT.json']['returncode']
                or 'pilot_EXIT.json' in doc):
            break
        time.sleep(600)


if __name__ == '__main__':
    main()
