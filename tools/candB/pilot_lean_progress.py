"""Read only the durable lean observer receipt; optional bounded wait, no GPU query."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait',type=int,default=0)
    args=parser.parse_args()
    if not 0<=args.wait<=110:
        raise ValueError('bounded wait must be0..110 seconds')
    directory=Path(__file__).parent
    end=time.monotonic()+args.wait
    while True:
        doc=json.loads((directory/'pilot_lean_latest.json').read_text())
        if (directory/'pilot_lean_update3_ACK.json').exists() or 'EXIT.json' in doc['files'] or time.monotonic()>=end:
            break
        time.sleep(2)
    trainer=doc['owners'].get('TRAINER_OWNERSHIP.json',{})
    baseline=doc['files'].get('frozen_scoring_baseline.json')
    print(json.dumps(dict(observed_epoch=doc['epoch'],trainer_pid=trainer.get('pid'),
        same_incarnation=trainer.get('same_incarnation'),state=trainer.get('state'),cpu_ticks=trainer.get('cpu_ticks'),
        startup=doc['files'].get('LEAN_STARTUP.json'),baseline_durable=baseline is not None,
        phases=[r['phase_update'] for r in doc['rows']],exit=doc['files'].get('EXIT.json'),
        log_tail=doc['files'].get('trainer.log','')[-2500:],
        gpu_sample=doc['files'].get('GPU.jsonl','').splitlines()[-1:])))


if __name__=='__main__':
    main()
