"""Compare one update's metrics row against the reference row (same update number).

usage: parity.py REF_updates.jsonl NEW_updates.jsonl UPDATE
Exact = every numeric leaf equal (bitwise for floats via repr); timing/memory keys ignored.
"""
import json, math, sys

IGNORE = {'t', 'wall_time_s', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes'}


def rows(path, update):
    found = [json.loads(l) for l in open(path) if l.strip()]
    found = [r for r in found if r.get('update') == update]
    return found[-1] if found else None


def leaves(value, prefix=''):
    if isinstance(value, dict):
        for k, v in value.items():
            if k in IGNORE:
                continue
            yield from leaves(v, f'{prefix}{k}.')
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from leaves(v, f'{prefix}{i}.')
    else:
        yield prefix.rstrip('.'), value


ref, new = rows(sys.argv[1], int(sys.argv[3])), rows(sys.argv[2], int(sys.argv[3]))
a, b = dict(leaves(ref)), dict(leaves(new))
diff, worst = [], 0.0
for k in sorted(set(a) | set(b)):
    x, y = a.get(k), b.get(k)
    if x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)):
        continue
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        rel = abs(x - y) / max(abs(x), abs(y), 1e-30)
        worst = max(worst, rel)
        diff.append((k, x, y, rel))
    else:
        diff.append((k, x, y, None))
print(json.dumps({'update': int(sys.argv[3]), 'compared_leaves': len(set(a) & set(b)),
                  'only_ref': sorted(set(a) - set(b))[:20], 'only_new': sorted(set(b) - set(a))[:20],
                  'n_diff': len(diff), 'max_rel_diff': worst, 'bitwise_identical': not diff,
                  'wall_time_ref': ref.get('wall_time_s'), 'wall_time_new': new.get('wall_time_s'),
                  'peak_alloc_new': new.get('cuda_peak_allocated_bytes'),
                  'peak_reserved_new': new.get('cuda_peak_reserved_bytes'),
                  'diffs': diff[:25]}, indent=1, default=str))
