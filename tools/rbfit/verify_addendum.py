"""Verify materialized posterior algebra and the real committed UNC interface."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def load_posterior(root):
    compressed = root/'idm_posteriors.json.gz'
    if compressed.is_file():
        with gzip.open(compressed, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads((root/'idm_posteriors.json').read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', required=True, type=Path)
    parser.add_argument('--new', required=True, type=Path)
    parser.add_argument('--lane', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    original_population = (args.fit/'idm_population.json').read_bytes()
    assert (args.new/'idm_population.json').read_bytes() == original_population
    population = json.loads(original_population)
    policy = json.loads((args.new/'idm_shrinkage.json').read_bytes())
    assert policy['idm_population_sha256'] == hashlib.sha256(original_population).hexdigest()
    assert policy['fit_worlds'] == list(range(969400, 969432))
    assert policy['prefix_interval_s'] == [240, 270] and policy['calibration_interval_s'] == [270, 300]
    assert policy['conditioning_end_s'] == 300
    old, new = load_posterior(args.fit), load_posterior(args.new)
    assert new['shrinkage_applied'] is True and new['conditioning_end_s'] == 300
    assert set(old['drivers']) == set(new['drivers'])
    prior, support = np.array(population['weights']), np.array(population['support'])
    maximum_error, fallbacks = 0., 0
    for key, driver in new['drivers'].items():
        assert 969400 <= int(key.split(':')[0]) <= 969431
        raw = old['drivers'][key]
        posterior = prior.copy()
        if 'weights' in raw:
            posterior[:] = 0
            posterior[np.array(raw['support_indices'])] = raw['weights']
        n = raw['n_observations']
        r = 0. if policy['population_only'] or not n else n/(n+policy['prior_count'])
        expected = (1-r)*prior+r*posterior
        actual = np.zeros_like(prior)
        indices, weights = np.array(driver['support_indices']), np.array(driver['weights'])
        assert len(indices) == len(np.unique(indices)) and (indices >= 0).all() and (indices < len(prior)).all()
        assert np.isfinite(weights).all() and (weights >= 0).all() and abs(weights.sum()-1) < 1e-12
        actual[indices] = weights
        maximum_error = max(maximum_error, float(abs(actual-expected).max()))
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-13)
        mu = actual@np.log(support)
        np.testing.assert_allclose(mu, driver['log_mean'], atol=1e-12)
        cov = (np.log(support)-mu).T@(actual[:, None]*(np.log(support)-mu))
        np.testing.assert_allclose(cov, driver['log_cov'], atol=1e-12)
        assert np.linalg.eigvalsh(cov).min() >= -1e-12
        if not n:
            fallbacks += 1
            np.testing.assert_allclose(actual, prior, atol=1e-13)
    provenance = json.loads((args.new/'provenance.json').read_bytes())
    assert provenance['idm_posteriors_gzip_sha256'] == hashlib.sha256(
        (args.new/'idm_posteriors.json.gz').read_bytes()).hexdigest()

    from wms.control.fitted_hidden import FittedHiddenState
    from wms.control.hidden_state import plan_hidden_state_futures
    from wms.physics.motion_fit import MotionFit
    from tests.unit.test_hidden_state_futures import fixture
    reader = FittedHiddenState.from_directory(args.new)
    motion_reader = MotionFit(args.new)
    generator = torch.Generator().manual_seed(91)
    for key in new['drivers']:
        world, vid = map(int, key.split(':'))
        values = reader.draw(world, vid, generator=generator)
        assert all(np.isfinite(v) and v > 0 for v in values.values())
    # Real exported artifacts through the committed future-planner (paired rows).
    world = 969400
    vids = [int(k.split(':')[1]) for k in new['drivers'] if k.startswith(f'{world}:')][:3]
    model, state, histories, points = fixture(rows=2)
    ids = torch.tensor([vids+[999999, -1]]).expand(2, -1)
    global_rng = torch.get_rng_state().clone()
    kwargs = dict(future_sampling='hidden_state', n_samples=8, seed=190,
                  row_keys=[world, world], fitted_state=reader, vehicle_ids=ids, t0_s=300)
    futures = plan_hidden_state_futures(model, state, histories, **kwargs)
    replay = plan_hidden_state_futures(model, state, histories, **kwargs)
    for a, b in zip(futures, replay):
        for name, params in a.kwargs['driver_params'].items():
            assert torch.equal(params[0], params[1])
            assert torch.equal(params, b.kwargs['driver_params'][name])
            assert torch.equal(params[:, 3:], points[name][:, 3:])
    assert torch.equal(global_rng, torch.get_rng_state())
    assert motion_reader.posterior['shrinkage_applied']
    # Match generic, ID-free runtime conditioning against an actual FIT driver.
    # This proves the identity bridge can be bypassed without slot-index guesses.
    from idm_fit import native_episode, read_episode, extract_rows, next_speed
    arrays, _ = read_episode(native_episode(Path('/home/ubuntu/wmdata20261031/views'), world), world)
    motion = extract_rows(arrays)[0]
    checked_motion = None
    for key, driver in new['drivers'].items():
        if key.startswith(f'{world}:') and driver['n_observations']:
            vid = int(key.split(':')[1])
            rr = motion[(motion[:, 0] < 300) & (motion[:, 1] == vid)]
            prediction = next_speed(rr[:, 3, None], rr[:, 5, None], rr[:, 6, None], support, rr[0, 9])
            conditioned = motion_reader.condition_motion(
                *[torch.tensor(v) for v in (prediction, rr[:, 3], rr[:, 4], rr[:, 9], rr[:, 0]+rr[:, 9])],
                t0_s=300, policy_path=args.new/'idm_shrinkage.json')
            dense = np.zeros(len(prior)); dense[conditioned['support_indices']] = conditioned['weights']
            expected = np.zeros(len(prior)); expected[driver['support_indices']] = driver['weights']
            np.testing.assert_allclose(dense, expected, atol=1e-11)
            checked_motion = {'fit_identity_for_verification_only': key, 'n_intervals': len(rr),
                              'max_weight_error': float(abs(dense-expected).max())}
            break
    assert checked_motion is not None
    try:
        MotionFit(args.new, shrinkage_path=args.new/'idm_shrinkage.json')
    except ValueError as error:
        assert 'already applied' in str(error)
    else:
        raise AssertionError('Double shrinkage accepted')

    lc = json.loads((args.lane/'lc2013_population.json').read_bytes())
    ls = json.loads((args.lane/'lc2013_posteriors.json').read_bytes())
    np.testing.assert_allclose(sum(lc['weights']), 1, atol=1e-12)
    assert lc['n_drivers'] == len(ls['drivers']) == 3130
    assert lc['n_events'] == 658
    assert np.diff(lc['likelihood_trace']).min() >= -1e-7
    for driver in ls['drivers'].values():
        weights = np.array(driver['weights'])
        assert weights.shape == (len(lc['support']),) and (weights >= 0).all()
        assert np.isfinite(weights).all() and abs(weights.sum()-1) < 1e-12
    # This separate LC block remains opt-in; test parser/draw without adoption.
    lane_reader = FittedHiddenState.from_directory(args.new, lane_directory=args.lane)
    assert len(lane_reader.draw(world, vids[0], generator=generator, lane=True)) == 5
    processes = []
    for name in ('lane_v2', 'shrinkage_v1'):
        receipt = args.new.parent/(name+'_process.json')
        r = json.loads(receipt.read_bytes())
        proc = Path('/proc')/str(r['pid'])
        state = (proc/'stat').read_text().rsplit(')', 1)[1].split()[0] if proc.exists() else 'exited'
        assert state in ('Z', 'exited'), f'Own job still active: {r["pid"]}'
        processes.append(dict(name=name, pid=r['pid'], state=state))
    result = dict(status='PASS', drivers=len(new['drivers']), zero_evidence_population_fallbacks=fallbacks,
        maximum_posterior_algebra_error=maximum_error, unc_source='f8b6faea',
        paired_replayed_real_artifact_futures=len(futures)*2, global_rng_unchanged=True,
        generic_causal_motion_parity=checked_motion,
        lane_source_sha256=lc['lc2013_source_sha256'], lane_adopted=False, processes=processes,
        python=sys.version, artifact_sha256={str(p.relative_to(args.new.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in (args.new, args.lane) for p in directory.iterdir() if p.is_file()})
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
