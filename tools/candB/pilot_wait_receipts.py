"""Bounded local-only monitor of the existing sole observer's pilot receipts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', type=int, default=110)
    parser.add_argument('--phase', type=int, default=1)
    args = parser.parse_args()
    assert 0 <= args.wait <= 110 and 1 <= args.phase <= 150
    path = Path(__file__).with_name('pilot_wmfast_latest.json')
    until = time.monotonic() + args.wait
    while True:
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError:
            if time.monotonic() >= until:
                raise
            time.sleep(1)
            continue
        outputs = doc['outputs']
        reached = any(any(r['phase_update'] >= args.phase for r in d['rows'])
            for d in outputs.values())
        fatal = any(d['files'].get('RUN_SUMMARY.json', {}).get('status') == 'failed'
            and d['files']['RUN_SUMMARY.json'].get('error') !=
            'RuntimeError: own-run STOP requested; checkpoint saved' for d in outputs.values())
        if reached or fatal or 'EXIT.json' in doc['files'] or time.monotonic() >= until:
            break
        time.sleep(2)
    result = dict(observed_utc=datetime.fromtimestamp(doc['epoch'], timezone.utc).isoformat(),
        age_s=round(time.time()-doc['epoch'], 1), outputs={},
        owners={k:dict(pid=v['pid'], alive=v.get('alive'), state=v.get('state'),
            cpu_ticks=v.get('cpu_ticks')) for k, v in doc['owners'].items()},
        gpu_sample=doc['files'].get('GPU.jsonl', '').splitlines()[-1:],
        exit=doc['files'].get('EXIT.json'))
    for name, data in outputs.items():
        summary = data['files'].get('RUN_SUMMARY.json', {})
        result['outputs'][name] = dict(status=summary.get('status'), error=summary.get('error'),
            baseline_durable='frozen_scoring_baseline.json' in data['files'],
            phases=[dict(phase=r['phase_update'], optimizer_s=r.get('optimizer_update_s'),
                total_s=r.get('update_including_rescore_s'), raw=r.get('lean_training_raw_J'),
                actor_norm=r.get('actor_grad_norm'), es_pairs=r.get('eslong_accepted_pairs'))
                for r in data['rows']],
            continuation_proof=data['files'].get('pilot_WMFAST_CONTINUATION_PROOF.json'))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
