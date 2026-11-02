"""Verify retained recovery evidence without rescoring, fitting or inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from tools.rbcal.outcomes import DEV
from tools.wmdecide.rbeval_adapter import safe_path, sha256
from tools.wmdecide.report import write_json


def verify(report, out):
    report, out = safe_path(report), safe_path(out)
    if out.exists() or not out.parent.is_dir():
        raise ValueError('new verification receipt required')
    result = json.loads((report/'RESULTS.json').read_bytes())
    assert not result['outcome_qualification'] and result['parameter_fits'] == result['new_inference'] == 0
    assert result['reserved_tasks_opened'] == 0 and not result['probability_artifact_refitted']
    worlds = result['scored_worlds']
    assert worlds and set(worlds) <= set(DEV)
    admission = result['admission']
    keys, conservation = set(), 0
    for item in admission['tasks']:
        key = item['world'], item['future']
        assert key not in keys and key[0] in DEV and key[1] in range(8)
        keys.add(key)
        path = safe_path(item['source'])
        assert sha256(path) == item['sha256']
        if item['world'] in worlds:
            task = json.loads(path.read_bytes())
            assert (task['seed'], task['future']) == key
            assert task['future_sampling']['population_audit']['conserved']
            conservation += 1
    assert len(keys) == admission['retained_tasks']
    assert conservation == admission['scored_tasks'] == len(worlds)*8
    assert all((world, k) in keys for world in worlds for k in range(8))
    source = result['source']['sources']
    fit = Path('docs/paper_results/rb2cal/v1/fit/calibration.json')
    # Compare both frozen forms: source staging/checkout can normalize newlines.
    frozen_fit = subprocess.check_output(['git', 'show', '0711a026:'+fit.as_posix()])
    lf_fit = frozen_fit.replace(b'\r\n', b'\n')
    expected_hash = source['/root/rbeval20261031/rb2cal/code_0711a02646bb/'+fit.as_posix()]
    representations = {hashlib.sha256(b).hexdigest() for b in (frozen_fit, lf_fit, lf_fit.replace(b'\n', b'\r\n'))}
    assert expected_hash in representations, (expected_hash, representations)
    assert json.loads(fit.read_bytes()) == json.loads(frozen_fit)
    bound_files = {'RESULTS.json': sha256(report/'RESULTS.json')}
    for variant in ('physical', 'baseline_matched'):
        root = report/variant
        data = json.loads((root/'observations.json').read_bytes())
        assert all({r['seed'] for r in rows} == set(worlds) for rows in data.values())
        assert len(data['distributions']) == len(worlds)*15*3  # five arms + ten unordered pairs
        assert len(data['coverage']) == len(worlds)*15*3*4
        assert all(r['K'] == 8 for r in data['distributions'])
        summaries = json.loads((root/'summary.json').read_bytes())
        assert all(not r['uncertainty_available'] and r['low'] is None and r['high'] is None for r in summaries)
        with (root/'reliability.csv').open(newline='', encoding='utf-8') as stream:
            diagrams = list(csv.DictReader(stream))
        assert all(r['observed_low'] == r['observed_high'] == '' and r['uncertainty_available'] == 'False' for r in diagrams)
        for name in ('observations.json', 'summary.json', 'reliability.csv',
                     'reliability_preference.svg', 'reliability_selected_preference.svg', 'audit.json'):
            bound_files[f'{variant}/{name}'] = sha256(root/name)
    receipt = dict(state='VERIFIED_PARTIAL_ONLY', outcome_qualification=False,
        scored_worlds=worlds, retained_tasks=len(keys), scored_tasks=conservation,
        excluded_incomplete_tasks=len(keys)-conservation,
        population_conservation_checks_passed=conservation, learned_artifact_sha256=expected_hash,
        local_artifact_semantically_matches_frozen=True,
        all_world_confidence_intervals_suppressed=True, hashes=bound_files)
    write_json(out, receipt)
    print(json.dumps(receipt, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    verify(args.report, args.out)
