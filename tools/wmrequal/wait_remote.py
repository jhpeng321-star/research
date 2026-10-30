"""Finite task-owned completion watch. No SSH polling loop or peer access."""
from pathlib import Path
import json
import time

p=Path('/root/autodl-tmp/c4/wmrequal20261030')
deadline=time.monotonic()+16*3600
last=None
while True:
    if (p/'OWNER_2230_STOP.json').is_file():
        print('OWNER_CUTOFF', (p/'OWNER_2230_STOP.json').read_text(), flush=True)
        break
    counts={panel:len(list((p/'fast'/panel).glob('batch_*.json')))
            for panel in ('validation','calibration','panel120')}
    if counts!=last:
        print('PROGRESS',time.time(),counts,flush=True)
        last=counts
    if (p/'DENSE_RESUME_ERROR.json').is_file():
        raise RuntimeError((p/'DENSE_RESUME_ERROR.json').read_text())
    if (p/'DENSE_RESUME_DONE.json').is_file():
        assert counts==dict(validation=36,calibration=36,panel120=60),counts
        print('COMPLETE',(p/'DENSE_RESUME_DONE.json').read_text(),flush=True)
        break
    if time.monotonic()>deadline:
        raise TimeoutError('finite dense scoring chain incomplete')
    time.sleep(300)
