"""Independent panel completeness, source-binding and exact MOBIL-reuse checks."""
import hashlib
import json
from pathlib import Path

from .operate import RECEIPTS, ROOT
from .validate import aggregate, ARMS


def main():
    first_dir, last_dir = (RECEIPTS / n for n in ('panel_results', 'panel2_results'))
    summary = json.loads((last_dir / 'SUMMARY.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted(last_dir.glob('w*.json'))]
    assert len(rows) == 160
    assert {(r['seed'], r['arm']) for r in rows} == {
        (seed, arm) for seed in range(969400, 969432) for arm in ARMS}
    assert summary['status'] == 'COMPLETE' and summary['worlds'] == 32
    assert summary['aggregate'] == aggregate(rows)
    for row in rows:
        old = json.loads((first_dir / (row['episode'] + '.json')).read_text())
        assert row['input_sha256'] == old['input_sha256']
        assert row['scores']['mobil'] == old['scores']['mobil']
        assert row['recorded'] == old['recorded']
        assert row['exposure'] == old['exposure']
    for remote_path, digest in summary['source_sha256'].items():
        for suffix in ('src/wms/physics/lc2013.py', 'tools/rblane/validate.py'):
            if remote_path.endswith(suffix):
                assert hashlib.sha256((ROOT / suffix).read_bytes()).hexdigest() == digest
    complete = json.loads((RECEIPTS / 'PANEL2_COMPLETE.json').read_text())
    assert complete['complete'] and not complete['alive']
    targets = json.loads((RECEIPTS / 'TARGET_RECONCILIATION.json').read_text())
    assert targets['status'] == 'PASS_RECONCILED' and targets['unexpected_missing'] == 0
    result = dict(status='PASS', episodes=160, worlds=32, arms=5,
                  aggregate_reassembly='exact', current_model_and_scorer_hashes='exact',
                  same_cohort_and_all_input_hashes='exact', mobil_score_reuse='exact',
                  target_events_reconciled=targets['scoring_events'],
                  summary_sha256=hashlib.sha256((last_dir/'SUMMARY.json').read_bytes()).hexdigest())
    out = RECEIPTS / 'VERIFIED.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
