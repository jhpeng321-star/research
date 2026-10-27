"""Independent scalar primary bootstrap and full artifact/sign/accounting checks."""
import csv
import hashlib
import json

import numpy as np

from tools.autonomy_e5.summarize import EVIDENCE, PAPER, ROOT


def main():
    manifest = json.loads((EVIDENCE/'MANIFEST.json').read_text())
    for group, base in (('artifacts', EVIDENCE), ('evidence', EVIDENCE), ('scripts', ROOT/'tools/autonomy_e5')):
        for name, sha in manifest[group].items():
            assert hashlib.sha256((base/name).read_bytes()).hexdigest() == sha
    for name in ('E5_COOPOFF_C1.csv', 'E5_COOPOFF_C1_worlds.csv', 'E5_COOPOFF_C1.md'):
        assert (PAPER/name).read_bytes() == (EVIDENCE/name).read_bytes()
    data = [json.loads((EVIDENCE/b/'ANALYSIS_DATA.json').read_text()) for b in ('cpuP', 'cpuQ')]
    rows = {(r['seed'], r['arm']): r for d in data for r in d['rows']}
    assert len(rows) == 912
    for box, d in zip(('cpuP', 'cpuQ'), data):
        status = json.loads((EVIDENCE/box/'ANALYSIS_STATUS.json').read_text())
        assert hashlib.sha256((EVIDENCE/box/'ANALYSIS_DATA.json').read_bytes()).hexdigest() == status['data_sha256']
        identity = json.loads((EVIDENCE/box/'IDENTITY_AUDIT.json').read_text())
        assert identity['worlds'] == len(identity['summaries']) == 114
        assert all(r['trajectories_byte_identical'] and r['metrics_exactly_identical'] for r in identity['summaries'])
        assert identity['checkpoint_sha256'][0] != identity['checkpoint_sha256'][1]
        assert identity['tensor_comparison']['policy']['different'] == ['residual_head.weight', 'residual_head.bias']
        telemetry = json.loads((EVIDENCE/box/'TELEMETRY_AUDIT.json').read_text())
        assert telemetry['aggregates']['issued_acceleration_mps2']['worlds_different'] == 114
        assert telemetry['aggregates']['lane_command_target']['worlds_different'] == 0
        for a in ('coopON', 'coopOFF', 'native', 'mappo'):
            assert len([r for r in d['rows'] if r['arm'] == a]) == 114
    charged = sum(x['charged_sim_s'] for d in data for x in d['proof']['charges'].values())
    actual = sum(x['actual_sim_s'] for d in data for x in d['proof']['charges'].values())
    assert (charged, actual) == (826200, 820800)
    authority = json.loads((EVIDENCE/'cpuQ/DONE_AUTHORITY.json').read_text())
    for name, field in (('done_runner.py', 'done_runner_sha256'), ('remote_done.py', 'remote_done_sha256')):
        assert hashlib.sha256((ROOT/'tools/autonomy_e5'/name).read_bytes()).hexdigest() == authority[field]
    assert hashlib.sha256((EVIDENCE/'cpuQ/E5DONE_AMENDMENT.md').read_bytes()).hexdigest() == authority['amendment_sha256']
    estimates = json.loads((EVIDENCE/'RESULTS.json').read_text())['estimates']
    ix = {(r['kind'], r['metric'], r['treatment'], r['reference'], r['horizon_s'], r['share'], r['demand']): r for r in estimates}
    assert len(ix) == len(estimates) == 15960
    for key, r in ix.items():
        kind, metric, a, b, h, share, demand = key
        if kind == 'physical_contrast' and metric == 'team_delay_vs':
            reverse = ix['saved_delay', metric, a, b, h, share, demand]
            np.testing.assert_allclose([reverse['value'], reverse['ci_low'], reverse['ci_high']],
                                       [-r['value'], -r['ci_high'], -r['ci_low']], atol=1e-10)
        if a == 'coopON' and b == 'coopOFF' and kind.endswith('contrast'):
            assert r['value'] == r['ci_low'] == r['ci_high'] == 0
    # Independent implementation: scalar draws, concatenate selected world values.
    rng = np.random.default_rng(20261027)
    groups = [[s for s in range(973000, 973228) if (s-973000) % 12 == c] for c in range(12)]
    samples = [rng.integers(0, 19, size=(5000, 19)) for _ in groups]
    for h in (120, 300, 600):
        values = [[rows[s, 'coopON']['windows'][str(h)]['team_delay_vs']-
                   rows[s, 'native']['windows'][str(h)]['team_delay_vs'] for s in g] for g in groups]
        boot = [np.mean([v[int(j)] for v, ids in zip(values, samples) for j in ids[i]]) for i in range(5000)]
        expected = ix['physical_contrast', 'team_delay_vs', 'coopON', 'native', h, 'All', 'All']
        np.testing.assert_allclose([np.mean(values), *np.quantile(boot, [.025, .975])],
                                   [expected['value'], expected['ci_low'], expected['ci_high']], atol=1e-9)
    with (EVIDENCE/'E5_COOPOFF_C1_worlds.csv').open(newline='') as f:
        assert len(list(csv.DictReader(f))) == 2736
    mirror = json.loads((EVIDENCE/'MIRROR_VERIFIED.json').read_text())
    assert mirror['status'] == 'PASS' and mirror['all_complete_coopoff_results_clocks_ledgers_mirrored']
    proof = dict(status='PASS', paired_worlds=228, arm_worlds=912, clocks=[120, 300, 600],
                 bootstrap_draws=5000, scalar_primary_reproduction=True, sign_reversal_all_scopes=True,
                 on_off_all_recorded_outcomes_exactly_equal=True, production_charged_sim_s=charged,
                 production_actual_sim_s=actual, n_estimates=len(estimates), per_world_rows=2736,
                 report_external_copies_byte_identical=True, completion_scripts_bound_to_authority=True,
                 mirror=mirror)
    (EVIDENCE/'VERIFIED.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
