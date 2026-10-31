"""Precision/recall from already verified motion-only LC fit counts; no inference.

Uses only explicit publication artifacts. Never opens tapes, executes a fitter,
chooses a physical profile, or dispatches work. Ratios below are ratios of
expected counts for persistent profile draws, not mean per-run precision.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .operate import RECEIPTS, ROOT

ORDER = ['lcStrategic', 'lcCooperative', 'lcSpeedGain', 'lcKeepRight', 'lcAssertive']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    assert args.receipt.endswith('.json') and Path(args.receipt).name == args.receipt
    output = RECEIPTS / args.receipt
    assert output.parent.is_dir() and not output.exists()
    from .validate import FORBIDDEN
    assert not any(term in str(args.fit.resolve()).lower() for term in FORBIDDEN)
    allowed = {'lc2013_population.json', 'lc2013_posteriors.json', 'lc2013_shrinkage.json',
               'validation.json', 'provenance.json', 'lane_events.json'}
    publication = json.loads((args.fit / 'PUBLICATION.json').read_text())
    assert publication['status'] == 'PASS'
    assert sha(args.fit / 'VERIFIED.json') == publication['verified_sha256']
    verified = json.loads((args.fit / 'VERIFIED.json').read_text())
    assert verified['status'] == 'PASS' and not verified['adopted']
    assert set(verified['artifact_sha256']) == allowed
    for name, digest in verified['artifact_sha256'].items():
        assert sha(args.fit / name) == digest, f'changed verified artifact:{name}'
    population = json.loads((args.fit / 'lc2013_population.json').read_text())
    assert population['fit_interval_s'] == [240, 300]
    assert population['fit_worlds'] == list(range(969400, 969432))
    assert population['parameter_order'] == ORDER and population['snapshot_version'] == 'v2'
    provenance = json.loads((args.fit / 'provenance.json').read_text())
    assert [r['world'] for r in provenance['sources']] == population['fit_worlds']
    assert all(r['episode'].endswith('__native') for r in provenance['sources'])
    rows = verified['world_scores']
    assert [r['world'] for r in rows] == population['fit_worlds']
    reports = {}
    for name in ('raw_physics_population', 'raw_physics_shrunk_posterior', 'raw_physics_source_default'):
        total = verified['aggregate'][name]
        for key in ('n', 'events', 'expected_event_count', 'expected_correct_direction_event_count',
                    'event_brier_sum', 'categorical_brier_sum'):
            actual = math.fsum(r[name][key] for r in rows)
            assert math.isclose(actual, total[key], rel_tol=1e-12, abs_tol=1e-10), (name, key)
        proposals = total['expected_event_count']
        matches = total['expected_correct_direction_event_count']
        events = total['events']
        assert 0 <= matches <= min(proposals, events)
        reports[name] = dict(scored_steps=total['n'], recorded_events=events,
            expected_positive_steps=proposals, expected_exact_directed_matches=matches,
            expected_false_positive_steps=proposals - matches,
            precision_ratio_of_expected_counts=matches / proposals if proposals else None,
            recall_ratio_of_expected_counts=matches / events,
            event_brier=total['event_brier'], always_stay_event_brier=events / total['n'])
    support, weights = population['support'], population['weights']
    assert len(support) == len(weights) and math.isclose(math.fsum(weights), 1, abs_tol=1e-12)
    assert all(len(r) == 5 and all(math.isfinite(v) and v >= 0 for v in r[:4])
               and 0 <= r[1] <= 1 and math.isfinite(r[4]) and r[4] > 0 for r in support)
    assert all(math.isfinite(w) and w >= 0 for w in weights)
    boundary_mass = {name: math.fsum(w for row, w in zip(support, weights) if row[i] == 0)
                     for i, name in enumerate(ORDER[:4])}
    current = {name: sha(ROOT / 'src/wms/physics' / name)
               for name in verified['physical_source_sha256']}
    result = dict(status='PASS', computation='verified-count aggregation only; zero model/fitter/dispatch runs',
        fit_directory=str(args.fit.resolve()), publication_timestamp=publication['timestamp'],
        verified_sha256=publication['verified_sha256'], artifact_sha256=verified['artifact_sha256'],
        physical_source_sha256=verified['physical_source_sha256'], current_source_sha256=current,
        same_source_as_current=current == verified['physical_source_sha256'],
        parameters='published joint mixture, persistent per driver; no averaged-parameter execution',
        inference='native FIT240..300 only; internal empirical Bayes; later300..360 conditional scoring',
        scores=reports, fitted_population_zero_boundary_mass=boundary_mass,
        per_motivation=None, per_motivation_unavailable='upstream publication lacks motivation-stratified counts',
        adopted=False, scope='different frozen source/pooled IDM from LANE panels; not a current-panel replacement or autonomous qualification')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'same_source_as_current', 'scores',
                                          'fitted_population_zero_boundary_mass', 'adopted')}, indent=2))


if __name__ == '__main__':
    main()
