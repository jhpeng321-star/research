"""S-SPEED2 compact residual attention, tested against a full dense K8 batch."""
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
entrant.feedback_from_specs=feedback
os.environ['WMS_SPEED_COMPACT_SLOTS']='1'
original=score.digest
def digest(path):
    raw=original(path)
    if Path(path).name=='SOURCE.json':
        raw=hashlib.sha256((raw+original(ROOT/'tools/wmrequal/grouped_feedback.py')+
                            original(ROOT/'tools/wmrequal/score_compact.py')).encode()).hexdigest()
    return raw
score.digest=digest
destination=ROOT/'compact'
destination.mkdir(exist_ok=True)
for name in ('src','tools','REAL.json','SOURCE.json'):
    path=destination/name
    if not path.exists():path.symlink_to(ROOT/name,target_is_directory=name in ('src','tools'))
score.OUT=destination
if os.environ.get('WMREQUAL_COMPACT_PROBE')=='1':
    source_for=score.source_for
    def first(*args,**kwargs):
        source,indexes=source_for(*args,**kwargs)
        return source,indexes[:1]
    score.source_for=first
score.main()
