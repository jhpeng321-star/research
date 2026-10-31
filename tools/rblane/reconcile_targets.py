"""Preserve initial mismatch and reconcile the helper's first-tape-frame boundary."""
import csv
import hashlib
import json
from pathlib import Path

from .operate import RECEIPTS


def main():
    original = RECEIPTS / 'TARGET_CROSSCHECK.json'
    row = json.loads(original.read_text())
    target = Path('E:/agent-work/rb-shared/targets/lc_events.csv')
    assert hashlib.sha256(target.read_bytes()).hexdigest() == row['target_sha256']
    times = []
    with target.open(newline='') as f:
        for event in csv.DictReader(f):
            assert 969400 <= int(event['seed']) <= 969431
            times.append(float(event['t_s']))
    # The helper uses traj_base rows starting at 300.5, so it cannot detect
    # changes from 300.0 to the first row. Our views contain both states.
    boundary = [e for e in row['missing'] if e[3] == 300.5]
    unexpected = [e for e in row['missing'] if e[3] != 300.5]
    assert not any(t == 300.5 for t in times)
    assert not unexpected
    assert row['matched'] + len(boundary) == row['scoring_events']
    result = dict(status='PASS_RECONCILED', retained_original_sha256=hashlib.sha256(
        original.read_bytes()).hexdigest(), matched_nonboundary=row['matched'],
        expected_first_frame_boundary_missing=len(boundary), unexpected_missing=0,
        helper_earliest_detected_event_s=min(times), helper_raw_tape_start_s=300.5,
        view_start_s=240, scoring_events=row['scoring_events'],
        position_abs_max_m=row['position_abs_max_m'],
        note='Original MISMATCH receipt retained. All 35 absent targets are genuine view events at the helper first tape row; all 3220 later events agree.',
        new_inference=0)
    out = RECEIPTS / 'TARGET_RECONCILIATION.json'
    if out.exists():
        raise RuntimeError('already reconciled')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
