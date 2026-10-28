"""Print compact measured bounds; full differences stay in REPORT.json."""
from pathlib import Path
import json
import sys

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent/'evidence/REPORT.json'
report = json.loads(path.read_text())
interest = {'actor_loss', 'actor_grad_norm', 'returns', 'return_scale', 'reward',
            'independent_J.J', 'independent_J.road_delay', 'independent_J.native_relative_gain',
            'pessimistic_paired_credit', 'mean_paired_credit'}
for name, result in report.items():
    summary = {k: result.get(k) for k in ('s_per_update', 'peak_owned_gpu_mib', 'projected_150_hours')}
    summary['parity'] = {k: result.get('parity', {}).get(k) for k in
                         ('compared_leaves', 'bitwise_identical', 'n_diff')}
    summary['important_metric_differences'] = [d for d in result.get('parity', {}).get('differences', [])
                                               if d['key'] in interest]
    for field in ('six_serial_parity', 'nine_N2_state_parity', 'six_serial_state_parity'):
        if field not in result:
            continue
        data = result[field]
        differences = data.get('differences', [])
        entry = {k: data.get(k) for k in ('bitwise_identical', 'n_diff', 'compared_tensors')}
        if 'state' in field:
            for prefix in ('policy.', 'group_encoder.', 'optimizer.'):
                candidates = [d for d in differences if d['key'].startswith(prefix) and 'max_absolute' in d]
                if candidates:
                    entry[prefix] = max(candidates, key=lambda d: d['max_absolute'])
        summary[field] = entry
    print(name, json.dumps(summary, indent=2))
