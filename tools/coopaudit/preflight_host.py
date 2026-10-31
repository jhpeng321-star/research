"""Read/hash exact registered scientific inputs and run existing guards; zero SUMO."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

gpu = os.getuid() == 0
root = Path('/root/autodl-tmp/c4/speedhead20261030/e3final' if gpu else '/home/ubuntu/e3final')
wrappers = Path('/root/wmdata2' if gpu else '/home/ubuntu/wmdata20261031')
sys.path[:0] = [str(wrappers), str(root/'runtime/src'), str(root/'runtime'), str(wrappers/'tools/speedhead')]
bad = re.compile('sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype|driver_params|vehicle_params|assignment', re.I)
def sha(p):
    assert not bad.search(str(p))
    return hashlib.sha256(p.read_bytes()).hexdigest()

manifest = root/'runtime/SOURCE_MANIFEST.json'
assert sha(manifest) == 'b3c031adb6bc2b5c2bca160bea308f83ba5567b97c1cd0e853eddcd8367a8193'
entries = json.loads(manifest.read_text())
assert len(entries) == 543
permitted = {n:h for n,h in entries.items() if not bad.search(n)}
assert all(sha(root/'runtime'/n) == h for n,h in permitted.items())
from traffic_shepherd.seeds import require_development_case
from traffic_shepherd.c4.seeds import check_case
cells = ((.2,2400),(.2,2700),(.4,2400),(.4,2700))
cases = [(969650+4*r+c,d) for r in range(6) for c,(p,d) in enumerate(cells)]
cases += [(969674,2400),(969675,2700),(969676,1900),(969677,1900),(969699,2700)]
for seed,demand in cases:
    require_development_case(seed,demand,'exposed')
    check_case(seed,demand)
paths = [wrappers/'tools/ctrlarm/zero.py', wrappers/'tools/speedhead/observer.py',
         wrappers/'tools/speedhead/baselines.py', root/'inputs/policy_latest.pt',
         Path('/root/autodl-tmp/c4/speedhead20261030/mappospeed_final169.pt' if gpu else
              '/home/ubuntu/mappospeed/production2/policy_latest.pt')]
print(json.dumps(dict(runtime=str(root), wrappers=str(wrappers), source=sha(manifest),
                     files_verified=len(permitted), excluded_count=len(entries)-len(permitted),
                     guard_cases=len(cases), hashes={str(p):sha(p) for p in paths}), indent=1))
