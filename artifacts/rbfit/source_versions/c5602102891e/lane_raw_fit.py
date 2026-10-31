"""Fit physical LC profile probabilities by observable one-hot least squares.

Unlike a free emission matrix, this objective scores the LC2013 decisions that
will actually run. Uses completed frozen-source decision caches; no additional
LC scoring, simulation, neural training, or controller outcomes. A symmetric
categorical observation likelihood conditions each persistent driver profile;
its evidence temperature is learned from causal prefix/suffix prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, xlogy


def fit_simplex(gram, correct, initial=None):
    """Convex categorical Brier least squares; exact simplex physical mixture."""
    n = len(correct)
    initial = np.full(n,1/n) if initial is None else np.asarray(initial)
    def objective(w):
        return float(w@gram@w-2*correct@w+1)
    def gradient(w):
        return 2*(gram@w-correct)
    result = minimize(objective,initial,jac=gradient,method="SLSQP",bounds=[(0,1)]*n,
        constraints={"type":"eq","fun":lambda w:w.sum()-1,"jac":lambda w:np.ones(n)},
        options={"ftol":1e-12,"maxiter":2000})
    w = np.maximum(result.x,0)
    # Numerical roundoff removal only, not a behavioral/probability cutoff.
    w[w <= np.finfo(float).eps*n] = 0
    w /= w.sum()
    gap = float(w@gradient(w)-gradient(w).min())
    if gap > 1e-6:
        raise RuntimeError(f"Simplex least-squares solution not converged: KKT gap={gap}")
    return w,{"success":bool(result.success),"message":str(result.message),
        "iterations":int(result.nit),"categorical_brier":objective(w),"simplex_kkt_gap":gap}


def bayes_weights(wrong, weights, beta):
    if beta == 0:
        return np.broadcast_to(weights,wrong.shape).copy()
    log_prior = np.full_like(weights,-np.inf)
    np.log(weights,out=log_prior,where=weights>0)
    logits = log_prior-beta*wrong
    return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))


def raw_probabilities(predictions, driver_weights):
    return np.stack([np.sum(driver_weights*(predictions==a),axis=1) for a in range(3)],axis=1)


def metrics(probability, actual):
    truth = np.eye(3)[actual]
    return {"n":len(actual),"events":int((actual!=0).sum()),
        "categorical_brier":float(np.mean(np.sum((probability-truth)**2,axis=1))),
        "event_brier":float(np.mean((1-probability[:,0]-(actual!=0))**2)),
        "negative_log_score":float(-np.log(np.maximum(probability[np.arange(len(actual)),actual],
                                                       np.finfo(float).tiny)).mean()),
        "expected_event_count":float(np.sum(1-probability[:,0]))}


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        raise ValueError("New output under existing parent required")
    verification = json.loads((args.cache/"VERIFIED.json").read_text())
    if verification["status"] != "PASS":
        raise ValueError("Independently verified completed LC cache required")
    pop = json.loads((args.cache/"lc2013_population.json").read_text())
    if not set(pop["fit_worlds"]) <= set(range(969400,969432)) or pop["fit_interval_s"] != [240,300]:
        raise ValueError("Only native FIT causal cache permitted")
    for name,expected in verification["artifact_sha256"].items():
        if hashlib.sha256((args.cache/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Verified LC cache artifact hash mismatch")
    args.out.mkdir(exist_ok=False)
    start = time.time()
    support = np.array(pop["support"])
    c = len(support)
    gram,correct,n_total = np.zeros((c,c)),np.zeros(c),0
    records,all_counts,all_prefix,all_evidence,all_prefix_evidence,keys = [],[],[],[],[],[]
    source_counts = np.zeros((3,3))
    for world in pop["fit_worlds"]:
        with np.load(args.cache/f"observations_{world}.npz",allow_pickle=False) as archive:
            r = {name:archive[name] for name in ("vid","time_s","source","actual","prediction","keys",
                "fit_counts","prefix_counts","fit_evidence","prefix_evidence")}
        fit = r["time_s"] <= 300
        prediction,actual = r["prediction"][fit],r["actual"][fit]
        for a in range(3):
            matrix = (prediction==a).astype(float)
            gram += matrix.T@matrix
        correct += np.sum(prediction==actual[:,None],axis=0)
        n_total += len(actual)
        np.add.at(source_counts,(r["source"][fit],actual),1)
        all_counts.append(r["fit_counts"])
        all_prefix.append(r["prefix_counts"])
        all_evidence.append(r["fit_evidence"])
        all_prefix_evidence.append(r["prefix_evidence"])
        keys.extend(f"{world}:{int(v)}" for v in r["keys"])
        records.append((world,r))
    weights,solver = fit_simplex(gram/n_total,correct/n_total,np.array(pop["weights"]))
    counts,prefix = np.concatenate(all_counts),np.concatenate(all_prefix)
    exposure = counts[:,0].sum((1,2,3))
    prefix_exposure = prefix[:,0].sum((1,2,3))
    wrong = exposure[:,None]-np.einsum("dcsaa->dc",counts)
    prefix_wrong = prefix_exposure[:,None]-np.einsum("dcsaa->dc",prefix)
    evidence,prefix_evidence = np.concatenate(all_evidence),np.concatenate(all_prefix_evidence)
    active = weights>0
    calibration = []
    offset = 0
    for _,r in records:
        selected = (r["time_s"]>270)&(r["time_s"]<=300)
        driver = offset+np.searchsorted(r["keys"],r["vid"][selected])
        calibration.append((r["prediction"][selected][:,active],r["actual"][selected],driver))
        offset += len(r["keys"])
    def objective(beta):
        posterior = bayes_weights(prefix_wrong[:,active],weights[active],beta)
        total,n = 0.0,0
        for p,y,driver in calibration:
            probability = raw_probabilities(p,posterior[driver])
            total += np.sum((probability-np.eye(3)[y])**2)
            n += len(y)
        return float(total/n)
    # Search coordinate is numerical only. Max beta is finite floating-point
    # log-range; no physical threshold or manually prescribed driver setting.
    max_beta = -np.log(np.finfo(float).tiny)
    def beta_at(x):
        return min(max_beta,x/(1-x)) if x<1 else max_beta
    grid = np.linspace(0,1,17)
    grid_scores = np.array([objective(beta_at(x)) for x in grid])
    j = int(grid_scores.argmin())
    lo,hi = grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]
    optimum = minimize_scalar(lambda x:objective(beta_at(x)),bounds=(lo,hi),method="bounded")
    score,x = min((float(grid_scores[j]),float(grid[j])),(float(optimum.fun),float(optimum.x)))
    beta = beta_at(x)
    posterior = bayes_weights(wrong,weights,beta)
    # Symmetric source-conditioned categorical discrepancy. Only driver
    # inference uses this; sampled physical models execute their raw decisions.
    error = np.exp(-beta)
    emission = np.tile((np.eye(3)+(1-np.eye(3))*error)[None],(3,1,1))/(1+2*error)
    source_rate = np.divide(source_counts,source_counts.sum(-1,keepdims=True),
        out=np.full_like(source_counts,1/3),where=source_counts.sum(-1,keepdims=True)>0)
    later = {k:[] for k in ("population","shrunk_posterior","source_rate")}
    raw_later = {k:[] for k in ("population","posterior","source_default")}
    y_later,offset = [],0
    for _,r in records:
        selected = r["time_s"]>300
        p,s,y = r["prediction"][selected],r["source"][selected],r["actual"][selected]
        driver = offset+np.searchsorted(r["keys"],r["vid"][selected])
        for label,ww in (("population",np.tile(weights,(len(y),1))),("shrunk_posterior",posterior[driver])):
            conditional = emission[s[:,None],p,:]
            later[label].append(np.sum(ww[...,None]*conditional,axis=1))
            raw_later["posterior" if label=="shrunk_posterior" else label].append(raw_probabilities(p,ww))
        later["source_rate"].append(source_rate[s])
        raw_later["source_default"].append(np.eye(3)[p[:,0]])
        y_later.append(y)
        offset += len(r["keys"])
    y_later = np.concatenate(y_later)
    population = dict(pop,weights=weights.tolist(),mean=dict(zip(pop["parameter_order"],(weights@support).tolist())),
        modal_parameters=dict(zip(pop["parameter_order"],support[weights.argmax()].tolist())),
        estimation="physical categorical one-hot least squares; causal symmetric-likelihood driver Bayes",
        population_least_squares=solver,numerically_active_profiles=int(active.sum()),
        observation_emission={"axes":["source_lane","predicted_decision","observed_decision"],
            "probabilities":emission.tolist(),"inverse_temperature":beta,
            "note":"Learned causal-history discrepancy for posterior inference only; NEVER runtime noise"})
    population.pop("likelihood_trace",None)
    population.pop("final_log_likelihood",None)
    write_json(args.out/"lc2013_population.json",population)
    drivers = {}
    for i,key in enumerate(keys):
        if exposure[i] == 0:
            continue
        w = posterior[i]
        # Full support retained for the existing UNC/LC reader contract.
        drivers[key] = {"weights":w.tolist(),"n_observations":int(exposure[i]),
            "n_informative_intervals":int(evidence[i]),
            "information_gain_nats":float(np.sum(xlogy(w[active],w[active]/weights[active]))),
            "posterior_fraction":float(evidence[i]>0)}
    write_json(args.out/"lc2013_posteriors.json",{"schema":"wm_phys.lc2013_posteriors.v1",
        "conditioning_end_s":300,"identity_key":"world:observable_vid","drivers":drivers,
        "fallback":"lc2013_population.json","shrinkage_already_applied":True})
    write_json(args.out/"lc2013_shrinkage.json",{"schema":"wm_phys.lc2013_shrinkage.v1","prior_count":0.0,
        "population_only":False,"inverse_temperature":beta,"calibration_intervals_s":[[240,270],[270,300]],
        "calibration_objective":"raw physical categorical Brier, not emission probability Brier",
        "categorical_brier":score,"raw_brier":score,"population_brier":float(grid_scores[0]),
        "internal_empirical_bayes":True,"already_applied":True})
    report = {"elapsed_s":time.time()-start,"n_fit_intervals":n_total,
        "later_FIT_directional_scores":{k:metrics(np.concatenate(v),y_later) for k,v in later.items()},
        "later_FIT_raw_physical_scores":{k:metrics(np.concatenate(v),y_later) for k,v in raw_later.items()},
        "n_zero_information_drivers":sum(d["n_informative_intervals"]==0 for d in drivers.values()),
        "adopted":False,"note":"Held-in conditional motion only; INT/EVAL free-running gate required"}
    write_json(args.out/"validation.json",report)
    provenance = json.loads((args.cache/"provenance.json").read_text())
    provenance.update(estimator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reused_decisions=str(args.cache),cache_verification_sha256=hashlib.sha256((args.cache/"VERIFIED.json").read_bytes()).hexdigest())
    write_json(args.out/"provenance.json",provenance)
    # Reuse completed observations/events via explicit links, without copying or
    # rescoring. They remain immutable in their original version directory.
    for world in pop["fit_worlds"]:
        (args.out/f"observations_{world}.npz").symlink_to((args.cache/f"observations_{world}.npz").resolve())
    (args.out/"lane_events.json").symlink_to((args.cache/"lane_events.json").resolve())
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
