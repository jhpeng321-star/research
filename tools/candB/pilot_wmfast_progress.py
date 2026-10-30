"""Bounded wait on the sole observer's durable receipt; no GPU/network query."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', type=int, default=0)
    args = parser.parse_args()
    assert 0 <= args.wait <= 110
    directory = Path(__file__).parent
    until = time.monotonic()+args.wait
    while True:
        doc = json.loads((directory/'pilot_wmfast_latest.json').read_text())
        if time.monotonic() >= until or (directory/'pilot_wmfast_update3_ACK.json').exists() or 'EXIT.json' in doc['files']:
            break
        time.sleep(2)
    print(json.dumps(dict(observed_epoch=doc['epoch'], owners={k:dict(pid=o['pid'],alive=o.get('alive'),
        cpu_ticks=o.get('cpu_ticks'),state=o.get('state')) for k,o in doc['owners'].items()},
        outputs={k:dict(phases=[r['phase_update'] for r in d['rows']],scores=len(d['startup_scores']),
            startup=d['files'].get('LEAN_STARTUP.json'),baseline_durable='frozen_scoring_baseline.json' in d['files'],
            status=d['files'].get('RUN_SUMMARY.json',{}).get('status'),
            error=d['files'].get('RUN_SUMMARY.json',{}).get('error'),
            latest_update=d['rows'][-1] if d['rows'] else None) for k,d in doc['outputs'].items()},
        gpu_sample=doc['files'].get('GPU.jsonl','').splitlines()[-1:],
        supervisor_log=doc['files'].get('supervisor.log','')[-1500:],exit=doc['files'].get('EXIT.json'))))


if __name__ == '__main__':
    main()
