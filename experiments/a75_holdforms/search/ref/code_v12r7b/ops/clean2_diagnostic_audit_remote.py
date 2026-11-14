"""Read-only audit of a completed clean2 replicated test; never dispatch work.

Send with clean2_transport.py --source-package --research-env --action 12/24.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from tools.coopmwm import launch_training, track2_monitor
from tools.coopmwm.exact_grade import verify_sealed
from tools.coopmwm.protocol import digest


update = int(ACTION)
assert update in (6, 12, 24)
root = Path('/home/ubuntu/coopmwm20261105_clean2')
cfg = launch_training.checked_config(root/'CONFIG.json')
launch_training.health(root)
monitor = launch_training.read(root/f'LEARNING_MONITOR_u{update:03}.json')
verify_sealed(monitor, 'receipt_sha256')
assert monitor['config_sha256'] == digest(cfg)
path = root/f'track2_diagnostics/replicated_u{update:03}_live.json'
row = launch_training.read(path)
verify_sealed(row, 'receipt_sha256')
assert row['status'] == 'complete' and row['dry'] is False
assert row['update'] == update and row['kind'] == 'replicated'
assert row['binding']['bootstrap'] == track2_monitor.BOOTSTRAP
assert row['criterion_metric'] == track2_monitor.SYNERGY
assert Counter(record['arm'] for record in row['records']) == {'B': 5, 'R': 5}
assert row['cells'] == len(row['records']) == 10
manifest = row['branch_manifest']
assert digest(manifest) == row['quote_sha256']
assert sum(branch['sim_s'] for branch in manifest) == row['quoted_sim_s'] == row['new_sim_s']
assert row['accounting']['reserved'] is True
assert row['accounting']['sim_s'] == row['quoted_sim_s']
assert row['accounting']['job_ids'] == [branch['id'] for branch in manifest]
results = [result for record in row['records'] for entry in record['entries']
           for result in entry['results']]
assert len(results) == row['branches'] == len(manifest)
assert all(record['status'] == 'graded' for record in row['records'])
assert all(result['status'] == 'graded' for result in results)
safety = {key: sum(result['safety'][key] for result in results)
          for key in ('collisions', 'teleports', 'hard_brakes', 'ttc_events')}
assert safety['collisions'] == safety['teleports'] == 0
recomputed = track2_monitor._analyse(row['records'], update, row['binding']['bootstrap'])
assert all(row[key] == value for key, value in recomputed.items())
print(json.dumps(dict(schema='coopmwm.clean2.diagnostic_audit.v1', read_only=True,
    timestamp=datetime.now(timezone.utc).isoformat(), update=update,
    config_sha256=digest(cfg), monitor_sha256=monitor['receipt_sha256'],
    receipt_sha256=row['receipt_sha256'], cells=row['cells'], branches=row['branches'],
    quoted_sim_s=row['quoted_sim_s'], safety=safety, analysis_recomputed=True,
    metric_reports=row['metric_reports'], criterion_metric=row['criterion_metric'],
    verdict=row['verdict']), sort_keys=True, allow_nan=False))
