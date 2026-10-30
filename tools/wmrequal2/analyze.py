"""CPU-only per-arm requalification from exactly 36 retained validation batches."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.wmrequal.summarize import (
    METRICS, OLD_SIGMA, aggregate, boot_sign, rows_for,
)

OLD = ROOT / 'tools/wmrequal/evidence'
HERE = Path(__file__).resolve().parent
ARMS = ('mappo', 'ours_c1', 'coopoff_c1')
HORIZONS = tuple(OLD_SIGMA)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    # Git's Windows checkout uses CRLF; original Linux receipts hash LF bytes.
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def enrich(rows, result):
    """Keep explicit sign denominators and harm CIs (zero-real calls still count)."""
    for mode in (False, True):
        rs = [r for r in rows if r['mix'] == mode]
        for h, summary in result['ON' if mode else 'OFF'].items():
            med = np.array([r['horizons'][h]['joint_median'] for r in rs])
            real = np.array([r['horizons'][h]['real'] for r in rs])
            draws = np.array([r['horizons'][h]['draws'] for r in rs])
            e = med - real
            np.testing.assert_allclose(e[:, 0], e[:, 2:5].sum(1), atol=.02, rtol=0)
            summary['J_median']['sign_n'] = int((real[:, 0] != 0).sum())
            summary['J_median']['sign_ci95'] = boot_sign(rs, med[:, 0], real[:, 0], np.ones(len(rs), bool))
            summary['error_squared_sum'] = float(np.square(e[:, 0]).sum())
            summary['max_component_partition_error'] = float(abs(e[:, 0] - e[:, 2:5].sum(1)).max())
            for label in ('conditional_zero_replicate_sigma', 'conditional_legacy_scale_sensitivity'):
                calls = summary[label]
                threshold = calls['threshold']
                gain = (med[:, 0] < -threshold) & (np.quantile(-draws[:, :, 0], .1, axis=1) > 0)
                harm = (med[:, 0] > threshold) & (np.quantile(draws[:, :, 0], .1, axis=1) > 0)
                for name, select in (('gain', gain), ('harm', harm), ('confident', gain | harm)):
                    signable = select & (real[:, 0] != 0)
                    correct = signable & (np.sign(med[:, 0]) == np.sign(real[:, 0]))
                    calls[name + '_sign_n'] = int(signable.sum())
                    calls[name + '_correct'] = int(correct.sum())
                    calls[name + '_zero_real'] = int((select & (real[:, 0] == 0)).sum())
                    calls[name + '_ci95'] = boot_sign(rs, med[:, 0], real[:, 0], select)
    return result


def build():
    previous = load(OLD / 'SUMMARY.json')
    real = load(OLD / 'REAL.json')['panels']['validation']
    canonical = load(OLD / 'LEGACY_REAL.json')['validation']
    runtime = load(HERE / 'evidence/RUNTIME.json')
    assert runtime['files']['src/wms/control/bridge.py']['functions']['make_cav_controller'][0]['defaults']['policy_ramp_merge'] == 'False'
    for arm in ARMS:
        assert set(runtime['controller_design']['controllers'][arm]) == {'checkpoint', 'sha256'}
    files = [OLD / 'validation' / f'batch_{i:03d}.json' for i in range(36)]
    lookup = {(r['pair_key'], r['arm']): r for r in canonical}
    rows, draws = rows_for('validation', real, files, lookup)
    assert len(rows) == 216 and len(draws) == 2304
    # Rebuild rather than trust cached summaries; bind receipts and reproduce ALL
    # old aggregate fields before introducing any new arm filtering.
    for file in files:
        receipt = load(file)['receipt']
        assert receipt['K'] == 8 and receipt['steps'] == 600
        assert receipt['physical_exits'] and receipt['queue_state']
        assert receipt['real_sha256'] == sha(OLD / 'REAL.json')
        assert [receipt[k + '_sha256'] for k in ('source', 'config', 'wm')] == previous['panels']['validation']['provenance']['source_config_wm_sha256']
    assert rows == previous['rows']
    pooled = aggregate(rows)
    assert pooled == {mode: previous['panels']['validation'][mode] for mode in ('OFF', 'ON')}
    by_arm = {arm: enrich([r for r in rows if r['arm'] == arm],
                         aggregate([r for r in rows if r['arm'] == arm])) for arm in ARMS}
    pooled = enrich(rows, pooled)
    no_mappo_rows = [r for r in rows if r['arm'] != 'mappo']
    no_mappo = enrich(no_mappo_rows, aggregate(no_mappo_rows))
    inputs = [OLD / 'SUMMARY.json', OLD / 'REAL.json', OLD / 'LEGACY_REAL.json',
              HERE / 'evidence/RUNTIME.json', *files]
    return dict(schema='wmrequal2_per_arm/1', metrics=list(METRICS),
        contract=dict(worlds=36, contrasts=108, K=8, mix='ON and OFF',
            error='predicted minus recorded arm-native accrued J, veh-s',
            components='source + road s<=75.44m + road s>75.44m; shared total-J median weights',
            statistic='median of 8 paired draws; calls also require q10 paired gain/harm > 0',
            bootstrap='2000 world-cluster draws, seed 0; zero real excluded from sign denominators only',
            training_perturbation_sigma=None, H_G_training=None, operational_H_G_training=0),
        exact_original_rows_reproduced=True, exact_original_aggregate_reproduced=True,
        affected_arms_policy_ramp_merge_true=[],
        original_ramp_ownership_exposure_arms=list(ARMS),
        per_arm=by_arm, all_arms=pooled,
        excluding_specific_bridge_bug=dict(removed_arms=[], retained_arms=list(ARMS),
            identical_to_all_arms=True),
        no_mappo_sensitivity=no_mappo,
        excluding_all_blocked_original_ramp_arms=dict(retained_arms=[], n=0,
            statistics=None, verdict='NOT QUALIFIED: no execution-matched contrasts'),
        rescore=dict(specific_bridge_bug='not applicable: no True-path validation arms',
            timeline_native_lane_repair='separate repair required; no corrected scores in existing rows'),
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs})


def show(result):
    for arm, values in [*result['per_arm'].items(), ('all', result['all_arms']),
                         ('no_mappo', result['no_mappo_sensitivity'])]:
        print(arm)
        for h in HORIZONS:
            r = values['ON'][h]
            j = r['J_median']
            c = r['conditional_zero_replicate_sigma']
            decomp = ' '.join(f"{m}={v['bias']:.1f}/{v['mae']:.1f}/{100*v['mse_covariance_share']:.1f}%"
                              for m, v in r['decomposition'].items())
            print(f"  {h}: {j['sign']:.3f}/{j['mae']:.1f}/{j['correlation']:.3f}/{j['bias']:+.1f}"
                  f" gain={c['gain']}:{c['gain_correct']}/{c['gain_sign_n']}"
                  f" harm={c['harm']}:{c['harm_correct']}/{c['harm_sign_n']} {decomp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    result = build()
    out = HERE / 'evidence/ANALYSIS.json'
    if args.verify:
        assert load(out) == result, 'saved analysis does not reproduce'
        print('EXACT_REPRODUCTION_OK')
    else:
        assert out.parent.is_dir()
        with out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=1, allow_nan=False)
            stream.write('\n')
    show(result)


if __name__ == '__main__':
    main()
