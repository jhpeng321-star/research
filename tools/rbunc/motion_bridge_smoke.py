"""Compatibility smoke using actual FIT artifacts and its causal inference API.

Synthetic motion only: not a traffic score, coverage test, or population refit.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json

import torch

import wms.control.fitted_hidden as fitted_hidden
from wms.control.fitted_hidden import FittedHiddenState, IDM_ORDER, read_artifact


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fit-source', type=Path, required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert a.out.parent.is_dir() and not a.out.exists()
    # Policy/artifact reads use the same path guard as the production reader.
    _, policy_sha = read_artifact(a.policy)
    spec = importlib.util.spec_from_file_location('rbunc_peer_motion_fit', a.fit_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    fit = module.MotionFit(a.directory)
    reader = FittedHiddenState.from_directory(a.directory)
    support = reader.support
    before = torch.get_rng_state().clone()
    # Positive free-flow IDM synthetic predictions; exercise genuine API shapes,
    # conditioning/policy and exact finite weights without recording access.
    n = 8
    speed = torch.full((n,), 10., dtype=torch.float64)
    dt = torch.full((n,), .5, dtype=torch.float64)
    prediction = support[:, 2][None] * (1-(speed[:, None]/support[:, 0][None])**4)
    next_speed = speed[:, None] + dt[:, None]*prediction
    end = torch.arange(n, dtype=torch.float64)*.5+311.5
    observed = next_speed[:, 0]
    distribution = fit.condition_motion(next_speed, speed, observed, dt, end,
        t0_s=315., policy_path=a.policy)
    causal = reader.with_motion_posteriors({(969400, 101):distribution}, t0_s=315.)
    actual_support, actual_weights = causal._distribution(
        causal.population, causal.posterior['drivers']['969400:101'], IDM_ORDER)
    expected_indices = torch.tensor(distribution['support_indices'])
    assert torch.equal(actual_support, support[expected_indices])
    assert torch.equal(actual_weights, torch.tensor(distribution['weights'], dtype=torch.float64))
    # No-evidence causal inference must be population, not the old saved101 fit.
    empty = fit.condition_motion(next_speed[:0], speed[:0], observed[:0], dt[:0], end[:0],
        t0_s=315., policy_path=a.policy)
    prior = reader.with_motion_posteriors({(969400, 101):empty}, t0_s=315.)
    empty_support, empty_weights = prior._distribution(prior.population,
        prior.posterior['drivers']['969400:101'], IDM_ORDER)
    active = reader.weights > 0
    assert torch.equal(empty_support, support[active])
    torch.testing.assert_close(empty_weights, (reader.weights/reader.weights.sum())[active],
                               rtol=1e-14, atol=1e-16)
    assert set(causal.posterior['drivers']) == {'969400:101'}
    assert torch.equal(before, torch.get_rng_state())
    result = dict(status='PASS', scientific_rollouts=0, synthetic_intervals=n,
        exact_causal_support_weights=True, empty_history_is_population=True,
        saved_id_records_cleared=True, global_rng_unchanged=True,
        fitted_artifacts=reader.provenance, causal_snapshot=causal.provenance,
        policy_sha256=policy_sha,
        fit_source_sha256=hashlib.sha256(a.fit_source.read_bytes()).hexdigest(),
        adapter_source_sha256=hashlib.sha256(Path(fitted_hidden.__file__).read_bytes()).hexdigest())
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=1, allow_nan=False)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
