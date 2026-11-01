"""Independent checks of cached-only BDRANK outputs and real-world pairing."""
from collections import Counter, defaultdict
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/paper_results/bdrank'


def rows(name):
    with (OUT/(name+'.csv')).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def grouped(data, keys):
    result = defaultdict(list)
    for r in data:
        result[tuple(r[k] for k in keys)].append(r)
    return result


def close(x, y, tol=1e-7):
    assert abs(float(x)-float(y)) < tol, (x, y)


def main():
    checks = Counter()
    real = rows('real_episodes')
    assert len(real) == 4500
    assert len({(r['seed'], r['arm']) for r in real}) == 500
    assert Counter(r['split'] for r in real) == {'dev32': 1440, 'reserved8': 360, 'replication50': 2700}
    source = ROOT.parent/'c4-wmbreakdown/docs/paper_results/wmbreakdown/real_events.csv'
    with source.open(encoding='utf-8', newline='') as f:
        old = {(r['seed'], r['arm'], r['region'], r['horizon_s']): r for r in csv.DictReader(f)}
    for r in real:
        o = old[r['seed'], r['arm'], r['region'], r['horizon_s']]
        assert (r['status'] == 'event') == (o['event'] == 'True')
        assert (r['status'] == 'preexisting') == (o['preexisting'] == 'True')
        close(r['team_delay_vs'], float(r['road_delay_vs'])+float(r['source_wait_vs']))
        close(r['team_delay_vs'], o['team_delay_vs'])
        assert r['duration_s'] == r['queue_extent_m'] == 'NA'
        if r['status'] == 'event':
            close(r['depth_mps'], -float(o['drop_mps']))
        else:
            assert r['depth_mps'] == 'NA'
            close(r['depth_expected'], 0)
        checks['real_rows_match_prior_detector_and_clock'] += 1

    # Independently express the all-pair through-origin fit as weighted within-world FE.
    fits = rows('cost_variance_decomposition')
    lookup = {(r['corpus'], r['split'], r['region'], r['horizon_s']): r
              for r in fits if r['level'] == 'pooled'}
    for key, rs in grouped(real, ['corpus', 'split', 'region', 'horizon_s']).items():
        X, y = [], []
        for _, world in grouped(rs, ['seed']).items():
            world = [r for r in world if r['status'] != 'preexisting']
            if len(world) < 2:
                continue
            xx = np.array([[float(r['occurrence']), (float(r['onset_s']) if r['status'] == 'event' else 0)/float(r['horizon_s']),
                            float(r['depth_expected'])] for r in world])
            yy = np.array([float(r['team_delay_vs']) for r in world])
            weight = np.sqrt(len(world))
            X.extend(weight*(xx-xx.mean(0)))
            y.extend(weight*(yy-yy.mean()))
        X, y = np.array(X), np.array(y)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        f = lookup[key]
        for j, k in enumerate(('beta_occurrence_veh_s', 'beta_timing_per_horizon_veh_s', 'beta_depth_veh_s_per_mps')):
            close(f[k], beta[j], 1e-6)
        residual = y-X@beta
        close(f['R2_full'], 1-residual@residual/(y@y))
        close(sum(float(f['shapley_'+k]) for k in ('occurrence', 'timing', 'severity')), f['R2_full'])
        close(float(f['residual_fraction'])+float(f['R2_full']), 1)
        checks['pair_regression_equals_independent_weighted_world_FE'] += 1

    pred = rows('controller_predictions')
    assert len(pred) == 900
    by = {(r['seed'], r['arm'], r['region'], r['horizon_s']): r for r in pred}
    draws = rows('controller_future_draws')
    assert len(draws) == 7200
    for key, rs in grouped(draws, ['seed', 'arm', 'region', 'horizon_s']).items():
        assert len(rs) == 8 and {int(r['future']) for r in rs} == set(range(8))
        p = by[key]
        for signal in ('occurrence', 'onset_rmst', 'depth_expected', 'mean_delay'):
            close(p['pred_'+signal], np.mean([float(r[signal]) for r in rs]))
        close(8*float(p['pred_occurrence']), round(8*float(p['pred_occurrence'])))
        checks['complete_K8_aggregate'] += 1

    # Validate categories and direction with independent product-of-differences arithmetic.
    for name in ('controller', 'option'):
        pairs = rows(name+'_pairs')
        for p in pairs:
            values = [p[k] for k in ('predicted_a', 'predicted_b', 'real_a', 'real_b')]
            if 'NA' in values:
                assert p['category'] == 'missing'
                continue
            a, b, c, d = map(float, values)
            px = a-b
            rx = c-d
            ptol = 1e-8 if 'delay' in p['signal'] or 'depth' in p['signal'] else 0
            rtol = 1e-8 if 'delay' in p['target'] or 'depth' in p['target'] else 0
            ptie, rtie = abs(px) <= ptol, abs(rx) <= rtol
            if ptie and rtie:
                expected = 'both_tie'
            elif ptie:
                expected = 'predicted_only_tie'
            elif rtie:
                expected = 'real_only_tie'
            else:
                opposite = p['signal'].startswith('onset') != p['target'].startswith('onset')
                expected = 'concordant' if (px*rx < 0 if opposite else px*rx > 0) else 'discordant'
            assert p['category'] == expected
            checks['pair_categories_and_tie_direction'] += 1
        summary = rows(name+'_concordance')
        for r in summary:
            close(sum(int(r[k]) for k in ('concordant', 'discordant', 'predicted_only_ties', 'real_only_ties', 'both_ties', 'missing')), r['pairs'])
            den = int(r['concordant'])+int(r['discordant'])
            if den:
                close(r['concordance'], int(r['concordant'])/den)
            if r['ci_status'] == 'NA_singleton_stratum':
                assert r['ci_low'] == r['ci_high'] == 'NA'
            checks['concordance_denominators'] += 1

    prefer = ('mappo', 'native', 'ctrl0', 'pi', 'circles')
    decisions = rows('controller_decisions')
    assert len(decisions) == 300
    for r in decisions:
        candidates = [by[r['seed'], a, 'upstream', r['horizon_s']] for a in prefer]
        rule = r['rule']
        def score(p):
            if rule == 'probability':
                return (float(p['pred_occurrence']),)
            if rule == 'onset':
                return (-float(p['pred_onset_rmst']),)
            if rule == 'severity':
                return (float(p['pred_depth_expected']),)
            if rule == 'delay':
                return (float(p['pred_mean_delay']),)
            return (float(p['pred_occurrence']), float(p['pred_mean_delay']))
        chosen = min(candidates, key=score)
        assert r['chosen'] == chosen['arm']
        ties = [p['arm'] for p in candidates if score(p) == score(chosen)]
        assert json.loads(r['tied_actions']) == ties
        actual = float(chosen['real_mean_delay'])
        baseline = float(candidates[0]['real_mean_delay'])
        best = min(float(p['real_mean_delay']) for p in candidates)
        close(r['saving'], baseline-actual)
        close(r['regret'], actual-best)
        close(float(r['saving'])+float(r['regret']), baseline-best)
        assert float(r['regret']) >= -1e-8
        checks['frozen_action_selection_and_regret_identity'] += 1
    selection = json.loads((OUT/'development_selection.json').read_text())
    assert selection['reserved_labels_used'] is False
    for h, r in selection['horizon_selection'].items():
        means = {k: np.mean([float(x['saving']) for x in decisions if x['split'] == 'development12'
                             and x['horizon_s'] == h and x['rule'] == k]) for k in r['development_mean_savings']}
        assert r['rule'] == max(means, key=means.get)
        for rule, value in means.items():
            close(value, r['development_mean_savings'][rule])
        checks['development_only_selection'] += 1

    # Replicated-world uncertainty is unchanged by duplicating every within-world row.
    # This checks clustering on the actual source panel rather than fictitious arm independence.
    import importlib.util
    spec = importlib.util.spec_from_file_location('bdrank_analysis_check', ROOT/'tools/bdrank/analyze.py')
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    frame = [r for r in real if r['split'] == 'dev32' and r['region'] == 'upstream' and r['horizon_s'] == '600' and r['arm'] == 'mappo']
    sample = []
    for r in frame:
        s = int(r['seed'])
        analysis.WORLD_META[s] = (r['demand'], r['penetration'], r['split'])
        sample.append(dict(seed=s, value=float(r['team_delay_vs'])))
    ci = analysis.interval(sample, 'value')
    duplicate = analysis.interval(sample+sample, 'value')
    assert ci['ci_status'] == 'fixed_cell_world_percentile'
    close(ci['ci_low'], duplicate['ci_low'])
    close(ci['ci_high'], duplicate['ci_high'])
    singleton = analysis.interval(sample[:1], 'value')
    assert singleton['ci_low'] is None and singleton['ci_status'] == 'NA_singleton_stratum'
    checks['world_bootstrap_duplicate_row_invariance_and_singleton_NA'] += 1

    option = rows('option_decisions')
    for model, name in [('FULL-v4', 'SNAPSHOT_1793521615.json'), ('FULL-v5', 'V5_QUALITY_SNAPSHOT_1793521619.json')]:
        src = json.loads((ROOT.parent/'c4-wmteach/docs/paper_results/wmteach'/name).read_text())
        for h in ('120', '300'):
            rs = [r for r in option if r['model'] == model and r['horizon_s'] == h]
            assert len(rs) == 12 and len({r['seed'] for r in rs}) == 12
            close(np.mean([float(r['saving']) for r in rs]), src['files']['QUALITY.json'][h]['exact_chosen_saving'])
            close(np.mean([float(r['regret']) for r in rs]), src['files']['QUALITY.json'][h]['exact_regret'])
            checks['option_results_match_complete_source_quality'] += 1
    report = ROOT/'docs/paper_results/BDRANK.md'
    if report.exists():
        assert len(report.read_text(encoding='utf-8').splitlines()) <= 60
        checks['report_line_cap'] += 1
    # All source hashes are rechecked; no metadata source paths are opened.
    manifest = json.loads((OUT/'input_manifest.json').read_text())
    bad = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')
    for r in manifest['inputs']:
        assert not any(s in r['path'].lower() for s in bad)
        assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() == r['sha256']
        checks['input_hashes'] += 1
    outputs = {}
    for p in sorted(OUT.iterdir()):
        if p.suffix in ('.csv', '.json', '.md') and p.name != 'verification.json':
            outputs[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    if report.exists():
        outputs['../BDRANK.md'] = hashlib.sha256(report.read_bytes()).hexdigest()
    # Explicit imports prove the analysis/verification paths cannot enter simulator/model runtimes.
    code_hashes = {}
    allowed = {'ast', 'collections', 'csv', 'hashlib', 'importlib', 'io', 'itertools', 'json', 'numpy', 'pathlib'}
    for name in ('analyze.py', 'verify.py'):
        p = ROOT/'tools/bdrank'/name
        source_text = p.read_text(encoding='utf-8')
        for n in ast.walk(ast.parse(source_text)):
            if isinstance(n, ast.Import):
                assert all(a.name.split('.')[0] in allowed for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                assert n.module.split('.')[0] in allowed
        checks['analysis_import_whitelist'] += 1
    for name in ('analyze.py', 'collect_option_metadata.py', 'inspect_cache.py', 'render.py', 'summarize.py', 'verify.py', 'README.md'):
        p = ROOT/'tools/bdrank'/name
        code_hashes[str(p.relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(p.read_bytes()).hexdigest()
    registration = (OUT/'REGISTRATION.md').read_text(encoding='utf-8').strip()
    shared = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    assert shared.read_text(encoding='utf-8').count(registration) == 1
    checks['one_shared_registration_line_matches_committed_copy'] += 1
    result = dict(pass_=True, checks=dict(checks), outputs=outputs, analysis_source_sha256=code_hashes,
                  registration_line_sha256=hashlib.sha256(registration.encode()).hexdigest(),
                  new_sumo_sim_s=0, new_inference=0)
    (OUT/'verification.json').write_text(json.dumps(result, indent=1)+'\n', encoding='utf-8')
    print(json.dumps(result['checks'], indent=1))


if __name__ == '__main__':
    main()
