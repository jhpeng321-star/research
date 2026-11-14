"""Zero-physics comparison of r10 mined bank examples and practice-pool identity."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

source = Path('/home/ubuntu/coopmwm20261106_clean3_src_ac9e4691')
root = Path('/home/ubuntu/coopmwm20261106_clean3_r4')
sys.path.insert(0, str(source/'pytools'))
from tools.coopmwm import bank_breadth as breadth, clean3_mined as mined, launch_production as production
from tools.coopmwm.protocol import digest


def forbidden(*args, **kwargs):
    raise AssertionError('read-only audit forbids process/simulation dispatch')


subprocess.Popen = forbidden
cfg = json.loads((root/'CONFIG.json').read_text())
native = production.backend_config(cfg, root=root/'smoke', smoke=True)
out = dict(timestamp=datetime.now(timezone.utc).isoformat(), status='ZERO_PHYSICS_POOL_AUDIT', arms={})
for arm in ('B', 'R'):
    found, wanted = mined.examples(native, arm)
    pool = breadth.practice_pool(native, arm)
    present = {(row['verification_id'], row['example_id']) for row in pool if row.get('mined')}
    targets = {(row['verification_id'], row['example_id']) for row in found}
    groups = defaultdict(list)
    original_visit = breadth.visit
    def collect(*args, **kwargs):
        consumer = args[2]
        def observe(**item):
            if '_g5_mined' in item['cfg']:
                snap, binding = breadth.verified_restart(item['cfg']['run_root'], arm,
                    item['receipt'], item['example'], item['plan'], item['cfg'].get('_mined_binding'))
                from tools.coopmwm.sumo_verify import snapshot_sha
                from tools.coopmwm.moves import from_plan
                context = digest(dict(snapshot=snapshot_sha(snap), family=from_plan(item['plan']).family))
                groups[context].append(dict(verification_id=item['receipt']['verification_id'],
                    example_id=item['example']['example_id'], plan_sha256=digest(item['plan'])))
            consumer(**item)
        return original_visit(args[0], args[1], observe, **kwargs)
    breadth.visit = collect
    try:
        repeated = breadth.practice_pool(native, arm)
    finally:
        breadth.visit = original_visit
    assert pool == repeated
    out['arms'][arm] = dict(mined_examples=found, pool_mined=[{k: row[k] for k in (
        'verification_id', 'example_id', 'context_id', 'mined_plan_sha256')} for row in pool if row.get('mined')],
        missing=sorted(targets-present), duplicate_contexts={k: rows for k, rows in groups.items() if len(rows)>1})
print(json.dumps(out, sort_keys=True))
