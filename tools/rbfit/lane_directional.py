"""Direction-aware LC2013 empirical Bayes from observable native FIT motion.

One physical profile per driver. Source-lane conditioned categorical emissions
score stay/target-left/target-right, including non-events and wrong directions.
Emissions model discrepancy only and never become runtime behavioral noise.
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
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, xlogy
from scipy.stats import qmc

from idm_fit import FIT_WORLDS, NAMES, native_episode, read_episode, write_json
from lane_fit import LC_NAMES


def posterior_from_counts(counts, weights, emission):
    ll = np.einsum("dcspa,spa->dc", counts, np.log(emission))
    joint = ll + np.log(np.maximum(weights, np.finfo(float).tiny))
    norm = logsumexp(joint, axis=1)
    return np.exp(joint - norm[:, None]), norm


def emissions_from_counts(counts):
    """MLE with machine-epsilon numerical support; unseen rows use source rate."""
    source_rate = counts.sum(axis=1)
    global_rate = source_rate.sum(axis=0)
    if global_rate.sum() == 0:
        raise ValueError("No observable exposure")
    global_rate = global_rate / global_rate.sum()
    source_rate = np.divide(source_rate, source_rate.sum(-1, keepdims=True),
                            out=np.broadcast_to(global_rate, source_rate.shape).copy(),
                            where=source_rate.sum(-1, keepdims=True) > 0)
    result = np.divide(counts, counts.sum(-1, keepdims=True),
                       out=np.broadcast_to(source_rate[:, None], counts.shape).copy(),
                       where=counts.sum(-1, keepdims=True) > 0)
    result = np.maximum(result, np.finfo(float).eps)
    return result / result.sum(-1, keepdims=True)


def fit_directional(counts, iterations=300):
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 5 or counts.shape[2:] != (3, 3, 3) or (counts < 0).any():
        raise ValueError("Expected nonnegative [driver,profile,source,proposal,event] counts")
    # Exposure/observed outcomes must not depend on a candidate profile.
    observed = counts.sum(axis=3)
    if not np.array_equal(observed, np.broadcast_to(observed[:, :1], observed.shape)):
        raise ValueError("Candidate-dependent observation exposure")
    weights = np.full(counts.shape[1], 1 / counts.shape[1])
    emission = emissions_from_counts(counts.mean(axis=1).sum(axis=0))
    trace = []
    for _ in range(iterations):
        posterior, norm = posterior_from_counts(counts, weights, emission)
        trace.append(float(norm.sum()))
        if len(trace) > 1 and abs(trace[-1] - trace[-2]) < 1e-6:
            break
        weights = np.maximum(posterior.mean(0), np.finfo(float).tiny)
        weights /= weights.sum()
        emission = emissions_from_counts(np.einsum("dc,dcspa->spa", posterior, counts))
    posterior, norm = posterior_from_counts(counts, weights, emission)
    return weights, emission, posterior, trace, float(norm.sum())


def decision_labels(source, next_lane):
    """WM0=left,1=right,2=ramp; actual/predicted label0 is staying."""
    result = np.where(next_lane == source, 0, next_lane + 1)
    if ((result < 0) | (result > 2)).any():
        raise ValueError("LC2013 must not target the ramp")
    return result.astype(np.uint8)


def count_records(record, keys, candidates, selected):
    out = np.zeros((len(keys), candidates, 3, 3, 3), dtype=np.int64)
    if not np.any(selected):
        return out
    driver = np.searchsorted(keys, record["vid"][selected])
    prediction = record["prediction"][selected]
    np.add.at(out, (driver[:, None], np.arange(candidates)[None, :],
                   record["source"][selected, None], prediction,
                   record["actual"][selected, None]), 1)
    return out


def reuse_columns(support, prior_support):
    """Exact physical vectors only; no nearest-profile substitutions."""
    matches = (support[:,None,:] == prior_support[None,:,:]).all(-1)
    found = matches.any(-1)
    return np.flatnonzero(~found), np.where(found, matches.argmax(-1), -1)


def merge_predictions(record, previous, prior_columns, new_columns):
    for name in ("vid", "time_s", "source", "actual"):
        if not np.array_equal(record[name], previous[name]):
            raise ValueError("Cached observable record alignment changed")
    combined = np.empty((len(record["vid"]),len(prior_columns)), dtype=np.uint8)
    cached = np.flatnonzero(prior_columns >= 0)
    combined[:,cached] = previous["prediction"][:,prior_columns[cached]]
    combined[:,new_columns] = record["prediction"]
    return dict(record, prediction=combined)


def episode_job(job):
    import torch
    from wms.physics.config import PhysicsConfig
    from wms.physics.lc2013 import lane_change

    root, world, support, idm, out, reuse = job
    torch.set_num_threads(1)
    candidates = len(support)
    previous, new_columns, prior_columns = None, np.arange(candidates), None
    if reuse is not None:
        old_support = np.array(json.loads((Path(reuse)/"lc2013_population.json").read_text())["support"])
        new_columns, prior_columns = reuse_columns(support, old_support)
        if len(new_columns) == 0:
            raise ValueError("No new physical profiles: reuse the completed fit directly")
        with np.load(Path(reuse)/f"observations_{world}.npz", allow_pickle=False) as archive:
            previous = {key: archive[key] for key in ("vid", "time_s", "source", "actual", "prediction")}
    active_support = support[new_columns]
    arrays, provenance = read_episode(native_episode(Path(root), world), world)
    t, kin, mask, cav, ids = (arrays[k] for k in
                             ("time_s", "veh_kin", "veh_mask", "is_cav", "vid"))
    active_candidates, slots = len(active_support), kin.shape[1]
    physics = PhysicsConfig(sumo_fidelity=True, dt_s=float(t[1]-t[0]))
    params = {name: torch.as_tensor(active_support[:, j:j+1]).expand(active_candidates, slots)
              for j, name in enumerate(LC_NAMES)}
    params.update({name: torch.full((active_candidates, slots), idm[name], dtype=torch.float64)
                   for name in NAMES})
    memory = None
    chunks = {key: [] for key in ("vid", "time_s", "source", "actual", "prediction")}
    events = []
    for ti in np.flatnonzero((t[:-1] >= 240) & (t[1:] <= 360)):
        normal0 = kin[ti, :, 1:4].sum(-1) == 1
        normal1 = kin[ti+1, :, 1:4].sum(-1) == 1
        same = mask[ti].astype(bool) & mask[ti+1].astype(bool) & (ids[ti] == ids[ti+1])
        present = same & normal0 & normal1 & (ids[ti] >= 0)
        eligible = present & ~cav[ti].astype(bool) & ~cav[ti+1].astype(bool)
        source = kin[ti, :, 1:4].argmax(-1)
        destination = kin[ti+1, :, 1:4].argmax(-1)
        # Same teacher-forced post-longitudinal geometry as lane_v2. Recorded
        # future positions/speeds are a conditioning target, not an open-loop test.
        moved = np.array(kin[ti], copy=True)
        moved[:, [0, 4, 5]] = np.asarray(kin[ti+1])[:, [0, 4, 5]]
        memory = lane_change(torch.as_tensor(moved).expand(active_candidates, slots, 7),
            torch.as_tensor(present).expand(active_candidates, slots), physics,
            memory=memory, identity=torch.as_tensor(np.array(ids[ti], copy=True)).expand(active_candidates, slots),
            decision_mask=torch.as_tensor(eligible).expand(active_candidates, slots), driver_params=params)
        selected = np.flatnonzero(eligible & (destination != 2) | (eligible & (source == destination)))
        predicted = decision_labels(source[None, selected], memory["next_lane"].numpy()[:, selected]).T
        actual = decision_labels(source[selected], destination[selected])
        chunks["vid"].append(np.asarray(ids[ti, selected], dtype=np.int64))
        chunks["time_s"].append(np.full(len(selected), t[ti+1]))
        chunks["source"].append(source[selected].astype(np.uint8))
        chunks["actual"].append(actual)
        chunks["prediction"].append(predicted)
        for j in selected[actual != 0]:
            events.append({"identity": f"{world}:{int(ids[ti,j])}", "time_s": float(t[ti+1]),
                           "position_m": float(kin[ti+1,j,0]), "from_lane": int(source[j]),
                           "to_lane": int(destination[j])})
    record = {key: np.concatenate(value) for key, value in chunks.items()}
    if previous is not None:
        record = merge_predictions(record, previous, prior_columns, new_columns)
    keys = np.unique(record["vid"][record["time_s"] <= 300])
    # Validation entrants use population; retain them in records, not posteriors.
    all_keys = np.unique(record["vid"])
    counts = {name: count_records(record, all_keys, candidates, selection) for name, selection in (
        ("fit", record["time_s"] <= 300), ("prefix", record["time_s"] <= 270))}
    informative = np.ptp(record["prediction"], axis=1) != 0
    evidence = {name: np.bincount(np.searchsorted(all_keys, record["vid"][selection & informative]),
                                 minlength=len(all_keys)) for name, selection in (
        ("fit", record["time_s"] <= 300), ("prefix", record["time_s"] <= 270))}
    path = Path(out) / f"observations_{world}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **record, keys=all_keys, fit_counts=counts["fit"],
                            prefix_counts=counts["prefix"], fit_evidence=evidence["fit"],
                            prefix_evidence=evidence["prefix"])
    print(json.dumps({"world": world, "drivers": len(keys), "events": len(events),
                      "records": len(record["vid"])}), flush=True)
    return world, all_keys, counts, evidence, events, provenance


def shrink_weights(posterior, population, n, strength):
    fraction = np.divide(n, n + strength, out=np.zeros_like(n, dtype=float), where=n+strength > 0)
    return fraction[:, None]*posterior + (1-fraction[:, None])*population


def record_probabilities(record, driver_keys, weights, emission, selected):
    ids = np.searchsorted(driver_keys, record["vid"][selected])
    proposed = record["prediction"][selected]
    source = record["source"][selected]
    conditional = emission[source[:, None], proposed, :]
    return np.einsum("nc,nca->na", weights[ids], conditional)


def score(probability, actual):
    if len(actual) == 0:
        return {"n": 0, "categorical_brier": None, "negative_log_score": None}
    truth = np.eye(3)[actual]
    return {"n": len(actual), "events": int((actual != 0).sum()),
            "categorical_brier": float(np.mean(np.sum((probability-truth)**2, axis=1))),
            "event_brier": float(np.mean((1-probability[:,0]-(actual != 0))**2)),
            "negative_log_score": float(-np.log(np.maximum(
                probability[np.arange(len(actual)), actual], np.finfo(float).tiny)).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--idm-pooled", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sobol-power", type=int, default=6)
    parser.add_argument("--worlds", type=int, nargs="+", default=list(FIT_WORLDS))
    parser.add_argument("--support-json", type=Path, help="Explicit source-domain numerical search support")
    parser.add_argument("--reuse", type=Path, help="Completed identical-source/IDM decision cache")
    args = parser.parse_args()
    if not args.worlds or len(set(args.worlds)) != len(args.worlds) or not set(args.worlds) <= set(FIT_WORLDS):
        raise ValueError("Unique native FIT worlds required before any data read")
    if not args.out.parent.is_dir():
        raise ValueError("Expected existing output parent")
    args.out.mkdir(exist_ok=False)
    start = time.time()
    pooled = json.loads(args.idm_pooled.read_text())
    if pooled["schema"] != "wm_phys.idm_pooled.v1":
        raise ValueError("Expected fitted pooled IDM parameters")
    points = qmc.Sobol(5, scramble=True, seed=20261031).random_base2(args.sobol_power)
    support = np.concatenate((np.ones((1, 5)), np.exp(4*points-2)))
    if args.support_json is not None:
        support = np.array(json.loads(args.support_json.read_text())["support"], dtype=float)
        if (support.ndim != 2 or support.shape[1] != 5 or not np.isfinite(support).all()
                or not (support[:,:4] >= 0).all() or not (support[:,4] > 0).all()
                or not (support[:,1] <= 1).all()):
            raise ValueError("Invalid source-domain physical LC support")
        if len(np.unique(support,axis=0)) != len(support):
            raise ValueError("Duplicate numerical physical profiles")
    if args.reuse is not None:
        prior = json.loads((args.reuse/"provenance.json").read_text())
        for name in ("lc2013", "lc2013_geometry", "config"):
            path = Path(__import__("wms.physics."+name, fromlist=["__file__"]).__file__)
            if hashlib.sha256(path.read_bytes()).hexdigest() != prior["physical_source_sha256"][path.name]:
                raise ValueError("Cached physical source mismatch")
        if hashlib.sha256(args.idm_pooled.read_bytes()).hexdigest() != prior["idm_pooled_sha256"]:
            raise ValueError("Cached IDM candidate mismatch")
    jobs = [(str(args.root), w, support, pooled["parameters"], str(args.out),
             str(args.reuse) if args.reuse is not None else None) for w in args.worlds]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results.extend(executor.map(episode_job, jobs, chunksize=1))
    keys = [f"{w}:{int(vid)}" for w, kk, *_ in results for vid in kk]
    counts = np.concatenate([r[2]["fit"] for r in results])
    prefix = np.concatenate([r[2]["prefix"] for r in results])
    n = np.concatenate([r[3]["fit"] for r in results])
    n_prefix = np.concatenate([r[3]["prefix"] for r in results])
    weights, emission, posterior, trace, ll = fit_directional(counts)
    before, _ = posterior_from_counts(prefix, weights, emission)
    # Calibrate on the second half of the CAUSAL history. Freeze before later
    # 300..360 scoring. Full pre300 population means internal empirical Bayes.
    calibration = []
    offset = 0
    source_counts = counts[:, 0].sum(axis=(0, 2))
    baseline = emissions_from_counts(np.repeat(source_counts[:, None], 3, axis=1))[:, 0]
    for w, kk, *_ in results:
        with np.load(args.out/f"observations_{w}.npz", allow_pickle=False) as archive:
            r = {k: archive[k] for k in ("vid", "time_s", "source", "actual", "prediction")}
        selected = (r["time_s"] > 270) & (r["time_s"] <= 300)
        raw = record_probabilities(r, kk, before[offset:offset+len(kk)], emission, selected)
        pop = record_probabilities(r, kk, np.tile(weights, (len(kk),1)), emission, selected)
        driver = np.searchsorted(kk, r["vid"][selected])
        calibration.append((raw, pop, n_prefix[offset:offset+len(kk)][driver], r["actual"][selected]))
        offset += len(kk)
    def objective(strength):
        total, length = 0.0, 0
        for raw, pop, nn, actual in calibration:
            fraction = np.divide(nn, nn+strength, out=np.zeros_like(nn, dtype=float), where=nn+strength > 0)
            pred = fraction[:,None]*raw + (1-fraction[:,None])*pop
            total += np.sum((pred-np.eye(3)[actual])**2)
            length += len(actual)
        return float(total/length)
    # Parameterize strength over [0,infinity] by a bounded mixture coordinate;
    # both endpoints are evaluated explicitly, no behavioral clipping constant.
    scale = max(1.0, float(n_prefix.mean()))
    def strength_at(x):
        return scale*x/(1-x) if x < 1 else float("inf")
    opt = minimize_scalar(lambda x: objective(strength_at(x)), bounds=(0,1), method="bounded")
    choices = [(objective(0), 0.0), (objective(float("inf")), 1.0), (opt.fun, float(opt.x))]
    calibration_score, x = min(choices)
    strength = strength_at(x)
    shrunk = shrink_weights(posterior, weights, n, strength)
    pop = {"schema": "wm_phys.lc2013_population.v1", "parameter_order": list(LC_NAMES),
        "distribution": "finite_support_mixture", "support": support.tolist(), "weights": weights.tolist(),
        "mean": dict(zip(LC_NAMES, (weights@support).tolist())),
        "modal_parameters": dict(zip(LC_NAMES, support[weights.argmax()].tolist())),
        "fit_interval_s": [240,300], "fit_worlds": args.worlds, "n_drivers": int((counts[:,0].sum((1,2,3))>0).sum()),
        "observation_emission": {"axes": ["source_lane", "predicted_decision", "observed_decision"],
            "decision_labels": ["stay", "target_left_0", "target_right_1"],
            "probabilities": emission.tolist(), "note": "Discrepancy likelihood only; NEVER runtime event noise"},
        "likelihood_trace": trace, "final_log_likelihood": ll,
        "requires_boundary_aware_lc_reader": bool((support == 0).any()),
        "idm_parameters": pooled["parameters"], "known_limits": ["finite support search",
            "highway LC2013 subset", "left-censored wishes at240", "stepwise discrepancy independence",
            "teacher-forced geometry", "internal empirical Bayes, not independent calibration"]}
    source_paths = [Path(__import__("wms.physics."+name, fromlist=["__file__"]).__file__)
                    for name in ("lc2013", "lc2013_geometry", "config")]
    source_sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    pop["lc2013_source_sha256"] = source_sha["lc2013.py"]
    write_json(args.out/"lc2013_population.json", pop)
    drivers = {}
    for i, key in enumerate(keys):
        exposure = int(counts[i,0].sum())
        if exposure == 0:
            continue
        drivers[key] = {"weights": shrunk[i].tolist(), "n_observations": exposure,
            "n_informative_intervals": int(n[i]),
            "information_gain_nats": float(np.sum(xlogy(shrunk[i], shrunk[i]/weights))),
            "posterior_fraction": float(n[i]/(n[i]+strength)) if n[i]+strength > 0 else 0.0}
    write_json(args.out/"lc2013_posteriors.json", {"schema": "wm_phys.lc2013_posteriors.v1",
        "conditioning_end_s": 300, "identity_key": "world:observable_vid", "drivers": drivers,
        "fallback": "lc2013_population.json", "shrinkage_already_applied": True})
    write_json(args.out/"lc2013_shrinkage.json", {"schema": "wm_phys.lc2013_shrinkage.v1",
        "prior_count": None if np.isinf(strength) else strength, "population_only": bool(np.isinf(strength)),
        "evidence": "intervals whose LC2013 decisions differ across physical profiles",
        "calibration_intervals_s": [[240,270],[270,300]], "categorical_brier": calibration_score,
        "raw_brier": choices[0][0], "population_brier": choices[1][0],
        "internal_empirical_bayes": True, "already_applied": True})
    later = {k: [] for k in ("population", "raw_posterior", "shrunk_posterior", "source_rate")}
    observed = []
    offset = 0
    for w, kk, *_ in results:
        with np.load(args.out/f"observations_{w}.npz", allow_pickle=False) as archive:
            r = {k: archive[k] for k in ("vid", "time_s", "source", "actual", "prediction")}
        selected = r["time_s"] > 300
        for label, distribution in (("population", np.tile(weights, (len(kk),1))),
                ("raw_posterior", posterior[offset:offset+len(kk)]),
                ("shrunk_posterior", shrunk[offset:offset+len(kk)])):
            later[label].append(record_probabilities(r, kk, distribution, emission, selected))
        later["source_rate"].append(baseline[r["source"][selected]])
        observed.append(r["actual"][selected])
        offset += len(kk)
    report = {"elapsed_s": time.time()-start, "n_fit_intervals": int(counts[:,0].sum()),
        "later_FIT_directional_scores": {k: score(np.concatenate(v), np.concatenate(observed)) for k,v in later.items()},
        "n_zero_information_drivers": sum(d["n_informative_intervals"] == 0 for d in drivers.values()),
        "adopted": False, "note": "Requires free-running INT/EVAL evidence before adoption"}
    write_json(args.out/"validation.json", report)
    write_json(args.out/"lane_events.json", {"events": [e for r in results for e in r[4]]})
    write_json(args.out/"provenance.json", {"sources": [r[5] for r in results], "physical_source_sha256": source_sha,
        "estimator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reused_decisions": str(args.reuse) if args.reuse is not None else None,
        "support_sha256": hashlib.sha256(args.support_json.read_bytes()).hexdigest() if args.support_json else None,
        "idm_pooled_sha256": hashlib.sha256(args.idm_pooled.read_bytes()).hexdigest()})
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
