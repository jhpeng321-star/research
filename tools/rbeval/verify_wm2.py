"""Verify the completed frozen WM2 report and its local immutable task mirror.

This performs no rollout or rescoring. It rechecks report/task/identity hashes,
the exact BIC events and observable contracts before publishing a NEW receipt.
"""
import argparse
import csv
import math
from pathlib import Path, PurePosixPath

from .core import HERE, SEEDS, digest, load, safe, save_new
from .finalize import panel_state
from .score import read_predictions


def verify_report_hashes(receipt, report):
    expected = receipt['report_hashes']
    assert set(expected) == {p.name for p in report.iterdir() if p.is_file()}, 'report file set changed'
    for name, sha in expected.items():
        assert PurePosixPath(name).name == name and '\\' not in name
        assert digest(report/name) == sha, ('report hash mismatch', name)
    assert load(report/'gate.json') == receipt['gate'], 'gate receipt mismatch'


def point_delta(a, b):
    """Cross-platform reduction roundoff is not a changed forecast or score."""
    if a == b:
        return 0.
    try:
        x, y = float(a), float(b)
    except ValueError:
        raise AssertionError(('point value changed', a, b)) from None
    assert math.isfinite(x) and math.isfinite(y) and math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-8), (
        'point value changed beyond floating-point tolerance', a, b)
    return abs(x-y)


def verify(evidence):
    evidence = safe(evidence)
    receipt_path = evidence/'FINALIZED.json'
    result = load(receipt_path)
    assert result['schema'] == 'rbeval-finalized/1'
    assert (result['tasks'], result['worlds'], result['episodes']) == (160, 20, 800)
    remote = PurePosixPath(result['report'])
    assert remote.parent == PurePosixPath('/root/rbeval20261031/reports')
    report = evidence/'reports'/remote.name
    verify_report_hashes(result, report)
    originals = [p for p in evidence.glob('FINALIZER_[0-9]*.json')
                 if load(p).get('sources') == result['sources']]
    assert originals, 'missing original frozen scorer identity'
    state = panel_state(evidence)
    assert state['complete'] and state['tasks'] == 160 and not state['blocked']
    real = load(HERE/'real_observables.json')['episodes']
    _, current = read_predictions([evidence/f'wm2_shard{i}' for i in range(4)], real, partial=False)
    original = load(report/'audit.json')
    assert original['model'] == current['model']
    assert original['complete_worlds'] == current['complete_worlds'] == list(SEEDS)
    assert not original['missing_tasks'] and not current['missing_tasks']
    assert original['tasks'] == current['tasks'] == 160

    def task_hashes(audit):
        hashes = {}
        for row in audit['inputs']:
            # Remote paths stay recorded; compare their stable shard/task suffix.
            path = PurePosixPath(row['task'].replace('\\', '/'))
            key = (path.parent.name, path.name)
            assert key not in hashes, 'duplicate audited task'
            hashes[key] = (row['sha256'], row['identity_sha256'])
        return hashes

    assert task_hashes(original) == task_hashes(current), 'scored input set changed'
    return dict(schema='rbeval-wm2-verified/1', tasks=160, worlds=20,
        arm_futures=800, K=8, t0=300, horizons=[120, 300, 600],
        complete_worlds=list(SEEDS), model=current['model']['model'],
        wm_sha256=current['model']['wm_sha256'],
        finalization_receipt_sha256=digest(receipt_path),
        original_finalizer_receipts={p.name: digest(p) for p in originals},
        remote_report=result['report'], local_report=str(report),
        report_hashes=result['report_hashes'], frozen_scoring_sources=result['sources'],
        finished_unix=result['finished_unix'], gate=result['gate'],
        checks=dict(all_160_completion_hashes=True, all_report_hashes=True,
            exact_scored_input_set=True, common_model_contract=True,
            observable_starts_and_prehistory=True, speed_support=True,
            delay_decomposition=True, registered_bic_decisions=7200),
        verifier_sha256=digest(Path(__file__)),
        note='Original frozen full-panel report; no rollout or scoring repeated. '
             'Partial-report bootstrap correction leaves full20-world draws unchanged.')


def verify_correction(original, corrected):
    """Bind a separate correction and prove that forecast point data did not change."""
    corrected = safe(corrected)
    report = Path(original['local_report'])
    gate = load(corrected/'gate.json')
    assert gate['bootstrap_revision'] == 'scored-only-singleton-roundoff-v2'
    assert gate['complete_worlds'] == 20
    for key, value in original['gate'].items():
        assert gate[key] == value, ('unexpected gate change', key)
    old_audit, new_audit = load(report/'audit.json'), load(corrected/'audit.json')
    assert old_audit['model'] == new_audit['model']
    assert [(r['sha256'], r['identity_sha256']) for r in old_audit['inputs']] == [
        (r['sha256'], r['identity_sha256']) for r in new_audit['inputs']]
    assert new_audit['bootstrap_worlds'] == list(SEEDS)
    unchanged = ('distribution_world_bins', 'distribution_worlds', 'events_worlds',
                 'timing_worlds', 'energy_diagnostic', 'expected_shifts_worlds',
                 'expected_order', 'calibration')
    changed = {
        'distribution_summary': {f'wm_minus_{base}_{suffix}' for base in ('pooled', 'cell')
                                 for suffix in ('low', 'high', 'uncertainty_available')},
        'rates': {'paired_rate_low', 'paired_rate_high', 'paired_rate_uncertainty_available', 'paired_rate_maxstat_critical'},
        'timing_summary': {'low', 'high', 'uncertainty_available', 'maxstat_critical'},
        'onset_cdf_bands': {'low', 'high', 'uncertainty_available', 'maxstat_critical', 'wm_inside_real_band'},
        'expected_shifts': {'real_low', 'real_high', 'bias_low', 'bias_high', 'predicted_inside_real_ci',
                            'real_sign_resolved', 'resolved_sign_agrees'},
    }
    checked, max_delta = 0, 0.
    changed.update({name: set() for name in unchanged})
    for name, allowed in changed.items():
        def rows(root):
            with (root/(name+'.csv')).open(newline='', encoding='utf-8') as stream:
                return list(csv.DictReader(stream))
        old, new = rows(report), rows(corrected)
        assert len(old) == len(new)
        for a, b in zip(old, new):
            assert set(a) == set(b), ('point column set changed', name)
            for k in a:
                if k not in allowed:
                    max_delta = max(max_delta, point_delta(a[k], b[k]))
            if b['split'] == 'development12':
                assert all(not b[k] for k in allowed if k.endswith(('_low', '_high')) or k in ('low', 'high'))
            checked += 1
    return dict(schema='rbeval-corrected-report/1', report=str(corrected), gate=gate,
        report_hashes={p.name: digest(p) for p in corrected.iterdir() if p.is_file()},
        sources={name: digest(HERE/name) for name in ('score.py', 'core.py', 'references.json', 'real_observables.json')},
        point_data_files_equivalent=list(unchanged), point_rows_checked=checked,
        max_absolute_numeric_point_difference=max_delta,
        numeric_comparison=dict(relative_tolerance=1e-12, absolute_tolerance=1e-8,
                                reason='Linux versus Windows floating-point reductions'),
        development12_singleton_CIs_unavailable=True,
        supersedes='Original bootstrap CIs only; original forecasts, detector decisions, point estimates and Wilson intervals retained.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--out', required=True, help='NEW JSON receipt, existing parent')
    parser.add_argument('--corrected-report', help='NEW separately scored correction to verify and bind')
    args = parser.parse_args()
    out = safe(args.out)
    assert out.parent.is_dir() and not out.exists()
    result = verify(args.evidence)
    if args.corrected_report:
        result['corrected'] = verify_correction(result, args.corrected_report)
    save_new(out, result)
    print('VERIFIED', result['tasks'], 'tasks;', len(result['report_hashes']),
          'report files;', result['gate']['verdict'], ';', out)


if __name__ == '__main__':
    main()
