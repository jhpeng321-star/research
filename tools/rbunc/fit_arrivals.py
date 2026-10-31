"""Fit only observed native FIT entrance counters; no driver labels or outcomes."""
from pathlib import Path
import argparse
import json
import os

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import chi2

from fit_probe import BASE, digest, install_guard, safe
from wms.control.arrival_fit import request_bins


def diagnostics(counts):
    """Condition on each world rate before assessing excess count dispersion."""
    counts = np.asarray(counts, dtype=float)
    results = []
    for e in range(2):
        x = counts[:, :, e]
        mu = np.broadcast_to(x.mean(1, keepdims=True), x.shape)
        poisson_ll = float((x*np.log(mu)-mu-gammaln(x+1)).sum())
        pearson = float(((x-mu)**2/mu).sum())
        df = x.shape[0]*(x.shape[1]-1)
        def negative_ll(log_alpha):
            r = np.exp(-log_alpha)
            return -float((gammaln(x+r)-gammaln(r)-gammaln(x+1)
                + r*np.log(r/(r+mu))+x*np.log(mu/(r+mu))).sum())
        opt = minimize_scalar(negative_ll, bounds=(-16., 5.), method='bounded')
        nb_ll = -float(opt.fun)
        bic_delta = np.log(x.size)-2*(nb_ll-poisson_ll)
        # Common temporal profile, conditioned on each world's demand scale.
        profile = x.sum(0)/x.sum()
        temporal_mu = x.sum(1, keepdims=True)*profile[None]
        temporal_ll = float((x*np.log(np.maximum(temporal_mu, np.finfo(float).tiny))
                             -temporal_mu-gammaln(x+1)).sum())
        results.append(dict(entrance=e, pearson=pearson, df=df,
            pearson_dispersion=pearson/df, overdispersion_p=float(chi2.sf(pearson, df)),
            nb_alpha=float(np.exp(opt.x)), nb_minus_poisson_bic=float(bic_delta),
            extra_dispersion_bic_selected=bool(bic_delta < 0),
            empirical_time_rate_multiplier=(profile*x.shape[1]).tolist(),
            time_varying_minus_constant_bic=float((x.shape[1]-1)*np.log(x.size)
                                                 -2*(temporal_ll-poisson_ll))))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.out.is_dir()
    os.nice(19)
    torch.set_num_threads(1)
    assert os.getloadavg()[0] < 24
    install_guard()
    worlds, hashes, counts = {}, {}, []
    for world in range(969400, 969432):
        path = safe(BASE/f's{world}_native/start.pt')
        start = torch.load(path, map_location='cpu', weights_only=False)
        assert start['t0'] == 300 and start['seed'] == world
        observed = request_bins(start['past_entries'], start['observed_queue_history'])
        assert sum(observed['final_backlog']) == start['queue']
        worlds[str(world)] = observed
        hashes[str(world)] = digest(path)
        counts.append(observed['counts'])
    source = Path('/home/ubuntu/e3final/runtime/traffic_shepherd/scenarios.py')
    assert digest(source) == '91df383db377b58026c3150e36bddcf0f05d74d689242eff2f082bc4c3ad5424'
    artifact = dict(schema='wm_phys.observed_arrivals.v1', generator_family='piecewise_constant_poisson',
        conditioning_end_s=300, observed_phase_end_s=1200,
        counts_source='causal cumulative insertions plus entrance queue',
        rate_posterior='Gamma(count+0.5, rate=exposure), Jeffreys Poisson prior; once per future',
        bin_s=10, worlds=worlds, diagnostics=diagnostics(counts),
        source_sha256=digest(source), input_sha256=hashes, fitter_sha256=digest(__file__),
        qualification='FIT request process only; no traffic-outcome calibration')
    with (args.out/'observed_arrivals.json').open('x') as stream:
        json.dump(artifact, stream, indent=1, allow_nan=False)
    print(json.dumps(dict(worlds=len(worlds), diagnostics=artifact['diagnostics']), indent=1))


if __name__ == '__main__':
    main()
