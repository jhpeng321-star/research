"""Independent panel completeness, source-binding and exact MOBIL-reuse checks."""
import hashlib
import json
import argparse
import re
import math

from .operate import RECEIPTS, remote
from .validate import aggregate, ARMS


def differences(left, right, path='aggregate'):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in left.keys() | right.keys():
            if key not in left or key not in right:
                result.append((path+'.'+key, 'missing key'))
            else:
                result.extend(differences(left[key], right[key], path+'.'+key))
        return result
    return [] if left == right else [(path, left, right)]


def check_local_reassembly(expected, rebuilt, nrows):
    issues = differences(expected, rebuilt)
    for issue in issues:
        assert len(issue) == 3, issue
        path, a, b = issue
        # Python3.12 changed built-in float sum. Only the non-dyadic position
        # transform can expose this here; counts, precision and timing stay exact.
        assert path.endswith(('.position_abs_sum_m', '.position_mae_m')), issue
        assert isinstance(a, float) and isinstance(b, float) and math.isfinite(a) and math.isfinite(b), issue
        assert abs(a-b) <= nrows * max(math.ulp(a), math.ulp(b)), issue
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', default='panel2')
    args = ap.parse_args()
    if not re.fullmatch(r'panel(?:[2-9]|[1-9][0-9]+)', args.kind):
        ap.error('expected numbered panel')
    first_dir, last_dir = RECEIPTS / 'panel_results', RECEIPTS / (args.kind + '_results')
    summary = json.loads((last_dir / 'SUMMARY.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted(last_dir.glob('w*.json'))]
    arms = summary.get('diagnostic_arms',list(ARMS))
    assert arms in (list(ARMS),['native'])
    expected = 32*len(arms)
    assert len(rows) == expected
    assert {(r['seed'], r['arm']) for r in rows} == {
        (seed, arm) for seed in range(969400, 969432) for arm in arms}
    assert summary['status'] == 'COMPLETE' and summary['worlds'] == 32
    rebuilt = aggregate(rows)
    local_roundoff = check_local_reassembly(summary['aggregate'], rebuilt, len(rows))
    reused_dir = None
    if summary['mobil_reuse'] is not None:
        for receipt in RECEIPTS.glob('PANEL*_COMPLETE.json'):
            completed = json.loads(receipt.read_text())
            if completed['root'] + '/results' == summary['mobil_reuse']:
                assert completed['complete'] and not completed['alive']
                reused_dir = RECEIPTS / (receipt.name.removesuffix('_COMPLETE.json').lower() + '_results')
                previous = json.loads((reused_dir/'SUMMARY.json').read_text())
                for key in ('physics', 'snapshot', 'horizon_s'):
                    assert previous[key] == summary[key]
                assert previous.get('motion_parameter_sha256') == summary.get('motion_parameter_sha256')
                assert previous.get('encoder_source_sha256') == summary.get('encoder_source_sha256')
                assert set(arms).issubset(previous.get('diagnostic_arms',list(ARMS)))
                assert previous.get('motion_points_reader_sha256') == summary.get('motion_points_reader_sha256')
                break
        assert reused_dir is not None, 'unknown MOBIL reuse source'
    for row in rows:
        old = json.loads((first_dir / (row['episode'] + '.json')).read_text())
        assert row['input_sha256'] == old['input_sha256']
        if reused_dir is not None:
            reused = json.loads((reused_dir/(row['episode'] + '.json')).read_text())
            assert row['scores']['mobil'] == reused['scores']['mobil']
        assert row['recorded'] == old['recorded']
        assert row['exposure'] == old['exposure']
    dispatch = json.loads((RECEIPTS / (args.kind.upper() + '_DISPATCH.json')).read_text())
    # Verify immutable dispatched bytes, not a later candidate in this worktree.
    binding = json.loads(remote(f'''import pathlib,hashlib,zipfile,json,sys,os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
p=pathlib.Path({dispatch['root']!r})
assert hashlib.sha256((p/'code.zip').read_bytes()).hexdigest()=={dispatch['bundle_sha256']!r}
summary=json.loads((p/'results/SUMMARY.json').read_text())
with zipfile.ZipFile(p/'code.zip') as z:
    for name,digest in summary['source_sha256'].items():
        rel=pathlib.Path(name).relative_to(p).as_posix()
        assert hashlib.sha256(z.read(rel)).hexdigest()==digest
        assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==digest
    if summary.get('encoder_checkpoint_sha256') is not None:
        assert hashlib.sha256(z.read('inputs/wm_s.pt')).hexdigest()==summary['encoder_checkpoint_sha256']
        assert summary['motion_parameter_sha256']==summary['encoder_checkpoint_sha256']
        if summary.get('encoder_source_sha256') is not None:
            sys.path[:0]=[str(p/'src'),str(p)]
            from tools.rblane.encoder import encoder_source_hashes
            assert encoder_source_hashes()==summary['encoder_source_sha256']
    elif summary.get('motion_points_sha256') is not None:
        data=z.read('inputs/motion_point_table.json')
        assert hashlib.sha256(data).hexdigest()==summary['motion_points_sha256']==summary['motion_parameter_sha256']
        sys.path[:0]=[str(p/'src'),str(p)]
        from tools.rblane.motion_points import reader_identity
        assert reader_identity()==summary['motion_points_reader_sha256']
    elif summary.get('motion_parameter_sha256') is not None:
        data=z.read('inputs/idm_pooled.json')
        assert hashlib.sha256(data).hexdigest()==summary['motion_parameter_sha256']
        assert json.loads(data)['parameters']==summary['motion_parameters']
    if summary.get('pre_signed_source_sha256') is not None:
        assert hashlib.sha256(z.read('inputs/lc2013_before_signed.py')).hexdigest()==summary['pre_signed_source_sha256']
        assert summary['pre_signed_source_sha256']=='c1aa89fafb69ee18d34ba6b7c75445a6985a10096d41237439e8dd06189e1921'
sys.path[:0]=[str(p/'src'),str(p)]
from tools.rblane.validate import aggregate
rows=[json.loads(f.read_text()) for f in sorted((p/'results').glob('w*.json'))]
assert summary['aggregate']==aggregate(rows), 'original-runtime exact reassembly failed'
print(json.dumps(dict(source_bundle_sha256={dispatch['bundle_sha256']!r},
                     original_runtime_exact_reassembly=True,original_python=sys.version.split()[0],
                     summary_sha256=hashlib.sha256((p/'results/SUMMARY.json').read_bytes()).hexdigest())))
'''))
    assert binding['summary_sha256'] == hashlib.sha256((last_dir/'SUMMARY.json').read_bytes()).hexdigest()
    complete = json.loads((RECEIPTS / (args.kind.upper() + '_COMPLETE.json')).read_text())
    assert complete['complete'] and not complete['alive']
    targets = json.loads((RECEIPTS / 'TARGET_RECONCILIATION.json').read_text())
    assert targets['status'] == 'PASS_RECONCILED' and targets['unexpected_missing'] == 0
    result = dict(status='PASS', episodes=expected, worlds=32, arms=len(arms),
                  aggregate_reassembly='exact in original runtime', local_position_roundoff=local_roundoff,
                  dispatched_model_and_scorer_hashes='exact',
                  same_cohort_and_all_input_hashes='exact',
                  mobil_score_reuse='exact' if reused_dir is not None else 'new paired score for this setting/transform',
                  target_events_reconciled=sum(r['recorded'] for r in rows),
                  **binding)
    for row in rows:
        for model, scores in row['scores'].items():
            if model.startswith('lc2013') and 'by_motivation' in scores:
                reasons = list(scores['by_motivation'].values())
                for key in ('predicted', 'matched'):
                    assert sum(r['exact'][key] for r in reasons) == scores['exact'][key]
    result['exact_motivation_partition'] = 'exact' if 'by_motivation' in rows[0]['scores']['lc2013'] else 'not saved'
    if 'lc2013_before_signed' in summary['aggregate']:
        for transition in ('merge', 'overtake'):
            assert (summary['aggregate']['lc2013']['by_transition'][transition]
                    == summary['aggregate']['lc2013_before_signed']['by_transition'][transition])
        result['paired_ablation_nontarget_transition_control'] = 'exact'
    out = RECEIPTS / (args.kind.upper() + '_VERIFIED.json')
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
