"""Finite-support physical IDM population ML and exact discrete driver Bayes.

Support is generated from the initial Gaussian motion fit, NOT driver labels.
All mixture masses and acceleration residual scale maximize the observable
driver-sequence likelihood (EM). This avoids Gaussian posterior extrapolation
in weakly identified directions. It is approximate only in finite support.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp, ndtri, xlogy
from scipy.stats import qmc

from idm_fit import (FIT_WORLDS, NAMES, extract_rows, native_episode, next_speed,
                     read_episode, validate, write_json)


def sse_job(job):
    key, rows, support = job
    predicted = next_speed(rows[:, 3, None], rows[:, 5, None], rows[:, 6, None], support,
                           rows[0, 9])
    residual = (predicted - rows[:, 4, None]) / rows[:, 9, None]
    return key, np.square(residual).sum(0), len(rows)


def mixture_em(sse, n, sigma, iterations=500):
    weights = np.full(sse.shape[1], 1 / sse.shape[1])
    trace = []
    for _ in range(iterations):
        with np.errstate(divide="ignore"):
            joint = -sse/(2*sigma*sigma) + np.log(weights)[None,:]
        marginal = logsumexp(joint, axis=1)
        posterior = np.exp(joint-marginal[:,None])
        nll = float(np.sum(n*(np.log(sigma)+0.5*np.log(2*np.pi))-marginal))
        weights = posterior.mean(0)
        sigma = float(np.sqrt(np.sum(posterior*sse)/n.sum()))
        trace.append({"iteration": len(trace), "negative_log_likelihood": nll,
                      "sigma_accel_mps2": sigma, "effective_profiles": float(np.exp(-np.sum(xlogy(weights,weights))))})
        if len(trace) > 1 and abs(trace[-1]["negative_log_likelihood"]-trace[-2]["negative_log_likelihood"]) < 1e-5:
            break
    with np.errstate(divide="ignore"):
        joint = -sse/(2*sigma*sigma) + np.log(weights)[None,:]
    posterior = np.exp(joint-logsumexp(joint,axis=1)[:,None])
    return weights, sigma, posterior, trace


def moments(support, weights):
    z = np.log(support)
    mu = weights @ z
    cov = (z-mu).T @ (weights[:,None]*(z-mu))
    median = []
    for j in range(support.shape[1]):
        order = support[:,j].argsort()
        median.append(support[order[np.searchsorted(np.cumsum(weights[order]),0.5)],j])
    return {"log_mean": mu.tolist(), "log_cov": cov.tolist(),
            "median": dict(zip(NAMES, map(float,median))),
            "geometric_mean": dict(zip(NAMES,np.exp(mu).tolist())),
            "modal_parameters": dict(zip(NAMES,support[weights.argmax()].tolist()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sobol-power", type=int, default=10)
    args = parser.parse_args()
    if not args.out.parent.is_dir():
        raise ValueError("Output parent must exist")
    args.out.mkdir(exist_ok=False)
    started = time.time()
    initial = json.loads((args.initial/"idm_population.json").read_text())
    z = np.array(initial["log_mean"])
    root = np.linalg.cholesky(np.array(initial["log_cov"]))
    q = qmc.Sobol(5,scramble=True,seed=20261031).random_base2(args.sobol_power)
    support = np.vstack(([initial["pooled_fit_median"][k] for k in NAMES], np.exp(z),
                          np.exp(z + ndtri(q) @ root.T)))
    groups, worlds, history, sources = {}, {}, {}, []
    for world in FIT_WORLDS:
        arrays, provenance = read_episode(native_episode(args.root,world),world)
        rows,hist,_ = extract_rows(arrays)
        worlds[world] = rows
        sources.append(provenance)
        for identity,duration in hist.items():
            key = f"{world}:{identity}"
            history[key] = duration
            rr = rows[(rows[:,0]<300)&(rows[:,1]==identity)]
            if len(rr):
                groups[key] = rr
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        result = list(executor.map(sse_job, ((k,r,support) for k,r in groups.items()), chunksize=16))
    keys = [r[0] for r in result]
    sse, n = np.stack([r[1] for r in result]), np.array([r[2] for r in result])
    weights,sigma,post,trace = mixture_em(sse,n,float(initial["sigma_accel_mps2"]))
    # Preserve exact nonzero support; no behavioral/probability pruning rule.
    keep = weights>0
    weights, support, post = weights[keep],support[keep],post[:,keep]
    pop = {"schema":"wm_phys.idm_population.v1", "parameter_order":list(NAMES),
           "distribution":"finite_support_mixture", "support":support.tolist(), "weights":weights.tolist(),
           **moments(support,weights), "delta":4, "sigma_accel_mps2":sigma,
           "n_drivers":len(keys), "n_observations":int(n.sum()), "fit_worlds":list(FIT_WORLDS),
           "fit_interval_s":[240,300], "likelihood":"Gaussian acceleration; exact discrete-sequence marginal ML EM",
           "finite_support_note":"Sobol log-normal support inferred from Gaussian physical fit; weights/noise ML, not global continuous ML",
           "initial_population_sha256":hashlib.sha256((args.initial/"idm_population.json").read_bytes()).hexdigest(),
           "em_trace":trace, "pooled_fit_median":initial["pooled_fit_median"]}
    drivers = {}
    for key,w in zip(keys,post):
        indices = np.flatnonzero(w>0)
        mm = moments(support,w)
        drivers[key] = {**mm,"support_indices":indices.tolist(),"weights":w[indices].tolist(),
                        "n_observations":len(groups[key]),"observed_history_s":history[key],"delta":4,
                        "information_gain_nats":float(np.sum(xlogy(w,w/weights)))}
    for key,duration in history.items():
        if key not in drivers:
            drivers[key] = {**moments(support,weights),"n_observations":0,
                            "observed_history_s":duration,"delta":4}
    write_json(args.out/"idm_population.json",pop)
    write_json(args.out/"idm_posteriors.json",{"schema":"wm_phys.idm_posteriors.v1",
        "parameter_order":list(NAMES),"conditioning_end_s":300,"identity_key":"world:observable_vid",
        "distribution":"finite_support_mixture","fallback":"idm_population.json","drivers":drivers})
    pooled_mu = np.log([pop["pooled_fit_median"][k] for k in NAMES])
    result = validate(worlds,{k:np.log([v["modal_parameters"][n] for n in NAMES]) for k,v in drivers.items()},
                      np.log([pop["modal_parameters"][n] for n in NAMES]),pooled_mu)
    result["parameter_point_estimates"]="Joint modal physical profile, not independent marginal medians"
    result["elapsed_s"]=time.time()-started
    write_json(args.out/"validation.json",result)
    write_json(args.out/"provenance.json",{"sources":sources})
    print(json.dumps({"profiles":len(support),"sigma":sigma,"acceleration":result["acceleration_mps2"]}),flush=True)


if __name__=="__main__":
    os.environ.setdefault("OMP_NUM_THREADS","1")
    main()
