"""Durable ten-minute own-run observer, PILOT telemetry, and ledger bridge."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.candB import pilot_continue_observe, pilot_observe


def append(path, text):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--spawn', action='store_true')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    lease = directory / 'pilot_watch_2240_OWNERSHIP.json'
    if args.spawn:
        if lease.exists():
            raise ValueError('observer lease already exists; verify its PID, never duplicate')
        with (directory / 'pilot_watch_2240.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'tools.candB.pilot_watch_2240',
                '--root', args.root, '--ledger', str(args.ledger)], cwd=directory.parents[1],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
        print(json.dumps(dict(pid=child.pid, root=args.root)))
        return
    with lease.open('x') as stream:
        json.dump(dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), root=args.root,
                       command=sys.argv, poll_s=600), stream, indent=1)
    recorded = set()
    while True:
        if (directory / 'pilot_watch_2240_STOP').exists():
            return
        try:
            doc = pilot_continue_observe.observe('gpuA', args.root)
        except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as error:
            print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                                  observation_error=str(error))), flush=True)
            time.sleep(600)
            continue
        entries = pilot_observe.record(doc, directory, args.ledger)
        for mode, stage in doc['stages'].items():
            for row in stage['rows']:
                phase = row['phase_update']
                key = (mode, phase)
                if key in recorded:
                    continue
                score = row.get('independent_J', {})
                line = (f'\n### Observed {mode} phase update {phase} — EXPLORATORY\n'
                    f'- Absolute update{row["update"]}; raw dJ vs CTRL0 dense='
                    f'{score.get("raw_dJ_vs_CTRL0_dense")}, 300s={score.get("raw_dJ_vs_CTRL0_long")}.\n'
                    f'- ES accepted/rejected pairs={row.get("eslong_accepted_pairs")}/'
                    f'{row.get("eslong_rejected_pairs")}; actor/dense/long/ES gradients='
                    f'{row.get("actor_grad_norm")}/{row.get("grad_norm_dense")}/'
                    f'{row.get("grad_norm_long")}/{row.get("grad_norm_long_es")}.\n'
                    f'- Optimizer s={row.get("optimizer_update_s")}; incl paired rescore s='
                    f'{row.get("update_including_rescore_s")}; peak allocated/reserved bytes='
                    f'{row.get("cuda_peak_allocated_bytes")}/{row.get("cuda_peak_reserved_bytes")}.\n'
                    f'- Executed residual: `{json.dumps(score.get("executed_residual"))}`.\n'
                    f'- Recipe: `{json.dumps(row.get("pilot_recipe"))}`; R4: `{json.dumps(row.get("R4"))}`.\n')
                append(directory / 'PILOT.md', line)
                recorded.add(key)
                if mode == 'pilot' and phase >= 3 and not (directory / 'pilot_update3_ACK.json').exists():
                    with (directory / 'pilot_update3_ACK.json').open('x') as stream:
                        json.dump(dict(label='EXPLORATORY', observed_utc=datetime.now(timezone.utc).isoformat(),
                            row=row, trainer=doc.get('continue_trainer_OWNERSHIP.json')), stream, indent=1)
            proof = stage.get('CONTINUATION_PROOF.json')
            if proof and not (directory / 'pilot_samephase_PROOF.json').exists():
                with (directory / 'pilot_samephase_PROOF.json').open('x') as stream:
                    json.dump(proof, stream, indent=1)
        print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), entries=entries,
            phase_updates={mode: [r['phase_update'] for r in stage['rows']]
                           for mode, stage in doc['stages'].items()},
            controller_alive=doc.get('continue_OWNERSHIP.json', {}).get('same_incarnation_alive'),
            trainer=doc.get('continue_trainer_OWNERSHIP.json'), exit=doc.get('continue_EXIT.json'))), flush=True)
        if 'continue_EXIT.json' in doc:
            return
        time.sleep(600)


if __name__ == '__main__':
    main()
