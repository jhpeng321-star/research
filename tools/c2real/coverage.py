"""Audit the collected-input boundary without simulating or rerunning analysis."""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

from analyze import ROOT, HOME, safe, read, doc, INPUTS


def main():
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    rows = list(csv.DictReader((HOME/'C2REAL.csv').open()))
    census = doc(HOME/'CENSUS.json')['cycles']
    result = {'cycles': {}, 'source': 'existing main arm/loop only',
              'new_simulations': 0, 'new_predictions': 0}
    for c in range(3):
        journal = doc(ROOT/f'arm/loop/worlds_c{c:02}.json')
        raw = safe(ROOT/f'arm/loop/worlds/c{c:02}')
        dirs = sorted(p.name for p in raw.iterdir())
        expected_dirs = sorted(Path(w['artifact']).parent.name for w in journal['worlds'].values())
        assert set(expected_dirs) <= set(dirs)
        failed_attempts = {}
        for name in sorted(set(dirs)-set(expected_dirs)):
            attempt = safe(raw/name)
            assert not (attempt/'artifact').exists(), name
            summaries = list((attempt/'collect').glob('*_COLLECT_SUMMARY.json'))
            assert len(summaries) == 1, name
            summary = doc(summaries[0])
            assert summary['episodes_complete'] == summary['branches'] == summary['pair_sim_seconds'] == 0
            assert not summary['pair_dJ'], name
            failed_attempts[name] = {'episodes_failed': summary['episodes_failed'],
                                     'pair_sim_seconds': summary['pair_sim_seconds'],
                                     'branches': summary['branches']}
        starts = {(int(r['world']), int(r['start_s'])) for r in rows
                  if r['level'] == 'start' and r['cycle'] == str(c)}
        credit_horizons, prediction_horizons, gate_horizons = Counter(), Counter(), Counter()
        for seed, w in journal['worlds'].items():
            artifact = safe(w['artifact'])
            for record in map(json.loads, read(artifact/'SUMOCREDIT_POINTS.jsonl').decode().splitlines()):
                assert record['seed'] == int(seed) and not 964000 <= record['seed'] <= 964199
                credit_horizons[record['horizon_s']] += 1
                if record.get('imagined') and record.get('sumo'):
                    assert (record['seed'], record['t_s']) in starts
                    if {'ALL', 'B'} <= record['imagined'].keys():
                        prediction_horizons[record['horizon_s']] += 1
            for record in map(json.loads, read(artifact/'GATE_CALIBRATION.jsonl').decode().splitlines()):
                assert (record['seed'], record['t_s']) in starts
                gate_horizons[record['horizon_s']] += 1
        tags = [f'c00.a{a}' for a in range(1, 5)] if c == 0 else [f'c{c:02}.a1']
        conversions = {}
        fingerprints = []
        for tag in tags:
            receipt = doc(ROOT/f'arm/loop/returns/returns_{tag}.receipt.json')
            native = doc(ROOT/f'arm/loop/returns/native_returns_{tag}.receipt.json')
            manifest = doc(ROOT/f'arm/loop/data/pairs_{tag}/C4_EPISODES_MANIFEST.json')
            fingerprint = []
            for arm, contents in [('policy', receipt), ('native', native)]:
                fingerprint.extend((arm, e['pair_key'], e['sha256'], e['J']) for e in contents['entries'])
            fingerprints.append(fingerprint)
            conversions[tag] = {'policy_starts': len(receipt['entries']),
                                'native_starts': len(native['entries']),
                                'tapes': len(manifest['episodes'])}
            assert len(receipt['entries']) == len(native['entries']) == len(starts)
            assert len(manifest['episodes']) == 2*len(starts)
        assert all(f == fingerprints[-1] for f in fingerprints)
        assert len(starts) == 2*len(expected_dirs) == 2*census[str(c)]['collected_worlds']
        assert 60 not in prediction_horizons and 60 not in gate_horizons
        result['cycles'][c] = dict(registered_worlds=len(journal['cases']), collected_worlds=len(expected_dirs),
            directories=dirs, missing_worlds=census[str(c)]['missing'],
            failed_attempts_without_paired_outcomes=failed_attempts,
            conversions=conversions, duplicate_conversion_retries=len(tags)-1,
            credit_record_horizons=dict(credit_horizons),
            matched_ALL_B_prediction_horizons=dict(prediction_horizons),
            gate_candidate_record_horizons=dict(gate_horizons))
    one = {int(s) for s in census['1']['worlds']}
    two = {int(s) for s in census['2']['worlds']}
    assert not one & two
    result['cycle1_cycle2_worlds_disjoint'] = True
    result['wm60_status'] = 'unavailable: no collected 60s predictions in main credit/gate records'
    result['c00_status'] = 'four collected worlds; a1-a4 duplicate policy AND native conversions'
    result['input_hashes'] = INPUTS
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HOME/'COVERAGE.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'input_hashes'}, indent=2))
    for r in rows:
        if r['level'] == 'behaviour_estimate' and r['arm'] in ('policy', 'native'):
            print(json.dumps({k: r[k] for k in ('cycle', 'metric', 'arm', 'value', 'ci_low', 'ci_high')}))
        if r['level'] == 'contrast' and r['metric'].startswith('WM'):
            print(json.dumps({k: r[k] for k in ('metric', 'value', 'ci_low', 'ci_high')}))


if __name__ == '__main__':
    main()
