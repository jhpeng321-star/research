"""Verified batched-feedback adapter over the immutable first scorer."""
from pathlib import Path
import hashlib
import os
import sys

ROOT=Path('/root/autodl-tmp/c4/wmrequal20261030')
RT=Path('/root/autodl-tmp/c4/restart20261026/runtime_longint')
sys.path[:0]=[str(ROOT),str(ROOT/'tools/wmrequal'),str(ROOT/'src'),str(RT)]
import score
from grouped_feedback import feedback
import wms.control.entrant as entrant

if os.environ.get('WMREQUAL_SERIAL_PARITY')!='1':
    entrant.feedback_from_specs=feedback
original=score.digest
def digest(path):
    raw=original(path)
    if Path(path).name=='SOURCE.json':
        raw=hashlib.sha256((raw+original(ROOT/'tools/wmrequal/grouped_feedback.py')+
                            original(ROOT/'tools/wmrequal/score_fast.py')).encode()).hexdigest()
    return raw
score.digest=digest
if os.environ.get('WMREQUAL_PARITY')=='1':
    score.HORIZONS=(10,)
    destination=ROOT/('parity_serial' if os.environ.get('WMREQUAL_SERIAL_PARITY')=='1' else 'parity_grouped')
else:
    destination=ROOT/'fast'
destination.mkdir(exist_ok=True)
for name in ('src','tools','REAL.json','SOURCE.json'):
    path=destination/name
    if not path.exists():path.symlink_to(ROOT/name,target_is_directory=name in ('src','tools'))
score.OUT=destination
score.main()
