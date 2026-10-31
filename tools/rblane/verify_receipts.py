"""Independent panel completeness, source-binding and exact MOBIL-reuse checks."""
import hashlib
import json
import argparse
import re

from .operate import RECEIPTS, remote
from .validate import aggregate, ARMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', default='panel2')
    args = ap.parse_args()
    if not re.fullmatch(r'panel(?:[2-9]|[1-9][0-9]+)', args.kind):
        ap.error('expected numbered panel')
    first_dir, last_dir = RECEIPTS / 'panel_results', RECEIPTS / (args.kind + '_results')
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
        if summary.get('motion_parameter_sha256') is None:
            assert row['scores']['mobil'] == old['scores']['mobil']
        else:
            assert summary['mobil_reuse'] is None
        assert row['recorded'] == old['recorded']
        assert row['exposure'] == old['exposure']
    dispatch = json.loads((RECEIPTS / (args.kind.upper() + '_DISPATCH.json')).read_text())
    # Verify immutable dispatched bytes, not a later candidate in this worktree.
    binding = json.loads(remote(f'''import pathlib,hashlib,zipfile,json
p=pathlib.Path({dispatch['root']!r})
assert hashlib.sha256((p/'code.zip').read_bytes()).hexdigest()=={dispatch['bundle_sha256']!r}
summary=json.loads((p/'results/SUMMARY.json').read_text())
with zipfile.ZipFile(p/'code.zip') as z:
    for name,digest in summary['source_sha256'].items():
        rel=pathlib.Path(name).relative_to(p).as_posix()
        assert hashlib.sha256(z.read(rel)).hexdigest()==digest
        assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==digest
    if summary.get('motion_parameter_sha256') is not None:
        data=z.read('inputs/idm_pooled.json')
        assert hashlib.sha256(data).hexdigest()==summary['motion_parameter_sha256']
        assert json.loads(data)['parameters']==summary['motion_parameters']
print(json.dumps(dict(source_bundle_sha256={dispatch['bundle_sha256']!r},
                     summary_sha256=hashlib.sha256((p/'results/SUMMARY.json').read_bytes()).hexdigest())))
'''))
    assert binding['summary_sha256'] == hashlib.sha256((last_dir/'SUMMARY.json').read_bytes()).hexdigest()
    complete = json.loads((RECEIPTS / (args.kind.upper() + '_COMPLETE.json')).read_text())
    assert complete['complete'] and not complete['alive']
    targets = json.loads((RECEIPTS / 'TARGET_RECONCILIATION.json').read_text())
    assert targets['status'] == 'PASS_RECONCILED' and targets['unexpected_missing'] == 0
    result = dict(status='PASS', episodes=160, worlds=32, arms=5,
                  aggregate_reassembly='exact', dispatched_model_and_scorer_hashes='exact',
                  same_cohort_and_all_input_hashes='exact',
                  mobil_score_reuse=('exact' if summary.get('motion_parameter_sha256') is None
                                     else 'new paired scoring under new motion-only physical point'),
                  target_events_reconciled=targets['scoring_events'],
                  **binding)
    for row in rows:
        scores = row['scores']['lc2013']
        if 'by_motivation' in scores:
            reasons = list(scores['by_motivation'].values())
            for key in ('predicted', 'matched'):
                assert sum(r['exact'][key] for r in reasons) == scores['exact'][key]
    result['exact_motivation_partition'] = 'exact' if 'by_motivation' in rows[0]['scores']['lc2013'] else 'not saved'
    out = RECEIPTS / (args.kind.upper() + '_VERIFIED.json')
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
