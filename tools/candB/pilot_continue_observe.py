"""Observe the retained dry phase plus its ordinary continuation; ledger every5."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from tools.candB import pilot_observe


def observe(host, root):
    code = pilot_observe.REMOTE.replace(
        "'dry_EXIT.json', 'pilot_EXIT.json')", "'dry_EXIT.json', 'pilot_EXIT.json', "
        "'continue_OWNERSHIP.json', 'continue_trainer_OWNERSHIP.json', 'continue_EXIT.json')")
    code = code.replace("root/(mode+'_EXPLORATORY')",
        "root/('pilot_samephase_EXPLORATORY' if mode == 'pilot' else 'dry_EXPLORATORY')")
    code = code.replace("root/(mode+'.log')", "root/('continue.log' if mode == 'pilot' else 'dry.log')")
    code = code.replace("'update_including_rescore_s','R4'", "'update_including_rescore_s','R4', 'pilot_recipe', 'eslong_forward_workers'")
    code = code.replace("'gate_panel_result.json')", "'gate_panel_result.json', 'CONTINUATION_PROOF.json')")
    code += "\n"
    result = subprocess.run(['ssh', host, '/root/miniconda3/bin/python', '-', root],
        input=code, text=True, capture_output=True, check=True, timeout=25)
    doc = json.loads(result.stdout)
    if 'continue_trainer_OWNERSHIP.json' in doc:
        doc['pilot_OWNERSHIP.json'] = doc['continue_trainer_OWNERSHIP.json']
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--host', default='gpuA')
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--follow-through-update', type=int)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    while True:
        doc = observe(args.host, args.root)
        entries = pilot_observe.record(doc, directory, args.ledger)
        report = dict(utc=datetime.now(timezone.utc).isoformat(), ledger_entries=entries,
            controller=doc.get('continue_OWNERSHIP.json'),
            trainer=doc.get('continue_trainer_OWNERSHIP.json'),
            stages={mode: dict(updates=len(stage['rows']),
                status=stage.get('RUN_SUMMARY.json', {}).get('status'),
                error=stage.get('RUN_SUMMARY.json', {}).get('error'),
                last_row=stage['rows'][-1] if stage['rows'] else None)
                for mode, stage in doc['stages'].items()})
        print(json.dumps(report), flush=True)
        pilot = doc['stages']['pilot']['rows']
        if (not args.follow_through_update or 'continue_EXIT.json' in doc or
                pilot and pilot[-1].get('phase_update', 0) >= args.follow_through_update):
            return
        time.sleep(600)


if __name__ == '__main__':
    main()
