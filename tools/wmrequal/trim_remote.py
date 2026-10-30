"""Apply owner 22:30 cutoff to the recorded task-owned dense chain only."""
from pathlib import Path
import json
import os
import signal
import time

p = Path('/root/autodl-tmp/c4/wmrequal20261030')
assert (p/'fast/validation/DONE.json').is_file()
receipt = p/'OWNER_2230_STOP.json'
if receipt.exists():
    print(receipt.read_text())
else:
    parent = json.loads((p/'DENSE_RESUME_LAUNCH.json').read_text())
    children = [json.loads(s) for s in (p/'DENSE_RESUME_CHILDREN.jsonl').read_text().splitlines()]
    stopped = []
    for r in [parent, *children]:
        proc = Path(f'/proc/{r["pid"]}')
        if not (proc/'stat').is_file():
            continue
        fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            continue
        assert fields[19] == r['start_ticks'], 'PID incarnation differs'
        cmd = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert str(p) in cmd and ('score_fast.py' in cmd or 'DENSE_RESUME_CHILDREN' in cmd)
        if r is not parent:
            assert '--panel calibration' in cmd or '--panel panel120' in cmd
        os.kill(r['pid'], signal.SIGTERM)
        stopped.append(dict(pid=r['pid'], start_ticks=r['start_ticks'], cmd=cmd))
    time.sleep(3)
    for r in stopped:
        q = Path(f'/proc/{r["pid"]}/stat')
        if q.is_file():
            fields = q.read_text().rsplit(')', 1)[1].split()
            assert fields[19] != r['start_ticks'] or fields[0] == 'Z', 'owned job still active'
    result = dict(time=time.time(), owner_addendum='2026-10-30 22:30',
                  reason='validation complete; stop excluded calibration/PANEL120; retain rows',
                  stopped=stopped,
                  completed_batches={panel: len(list((p/'fast'/panel).glob('batch_*.json')))
                                     for panel in ('validation', 'pairs300', 'calibration', 'panel120')})
    receipt.write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(result, indent=1))
