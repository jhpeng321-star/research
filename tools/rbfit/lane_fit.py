"""Finite-support LC2013 physical-parameter likelihood fit, native FIT only.

Each driver has ONE persistent parameter vector, not independent step noise.
Mixture masses, proposal emission probabilities and driver posterior masses are
estimated by EM from observable lane events at their recorded times/positions.
The numerical support is an explicitly approximate search, not a new LC rule.
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
from scipy.special import logsumexp, xlogy
from scipy.stats import qmc

from idm_fit import FIT_WORLDS, NAMES, native_episode, read_episode, write_json

LC_NAMES = ("lcStrategic", "lcCooperative", "lcSpeedGain", "lcKeepRight", "lcAssertive")


def fit_mixture(counts, iterations=200):
    """counts[D,C,4] = exact-step TP/FP/FN/TN; full driver-sequence mixture."""
    counts = np.asarray(counts, dtype=float)
    weights = np.full(counts.shape[1], 1 / counts.shape[1])
    baseline = counts[:, 0].sum(0)
    # Data-derived initialization; no hand-set event/merge probability floor.
    total = baseline.sum()
    rate = (baseline[0] + baseline[2]) / total
    p0, p1 = max(np.finfo(float).eps, rate / 2), min(1 - np.finfo(float).eps, rate * 2)
    trace = []
    for _ in range(iterations):
        ll = (xlogy(counts[..., 0], p1) + xlogy(counts[..., 1], 1 - p1)
              + xlogy(counts[..., 2], p0) + xlogy(counts[..., 3], 1 - p0))
        joint = ll + np.log(weights)
        normalizer = logsumexp(joint, axis=1)
        responsibilities = np.exp(joint - normalizer[:, None])
        weighted_counts = np.einsum("dc,dck->k", responsibilities, counts)
        new_p1 = weighted_counts[0] / max(np.finfo(float).tiny, weighted_counts[:2].sum())
        new_p0 = weighted_counts[2] / max(np.finfo(float).tiny, weighted_counts[2:].sum())
        weights = np.maximum(responsibilities.mean(0), np.finfo(float).tiny)
        weights /= weights.sum()
        p0, p1 = np.clip([new_p0, new_p1], np.finfo(float).eps, 1 - np.finfo(float).eps)
        trace.append(float(normalizer.sum()))
        if len(trace) > 1 and abs(trace[-1] - trace[-2]) < 1e-6:
            break
    # Recompute with EXPORTED weights/emissions, including zero-exposure drivers.
    ll = (xlogy(counts[..., 0], p1) + xlogy(counts[..., 1], 1 - p1)
          + xlogy(counts[..., 2], p0) + xlogy(counts[..., 3], 1 - p0))
    joint = ll + np.log(weights)
    posterior = np.exp(joint - logsumexp(joint, axis=1)[:, None])
    return weights, float(p0), float(p1), posterior, trace


def score_counts(counts, weights, p0, p1):
    """Expected Brier and conditional step log score under posterior mixture.

    TP+FP is constant only within candidate; reconstruct proposal indicators
    from saved probabilities in episode_job for the exact mixture score.
    """
    expected = np.einsum("dc,dck->k", weights, counts)
    tp, fp, fn, tn = expected
    n = expected.sum()
    return {"n": int(round(n)), "expected_tp": float(tp), "expected_fp": float(fp),
            "expected_fn": float(fn), "expected_tn": float(tn),
            "random_parameter_brier": float((tp * (1-p1)**2 + fp*p1**2
                                             + fn*(1-p0)**2 + tn*p0**2) / n),
            "expected_component_log_score": float((xlogy(tp, p1) + xlogy(fp, 1-p1)
                + xlogy(fn, p0) + xlogy(tn, 1-p0)) / n)}


def episode_job(job):
    import torch
    from wms.physics.config import PhysicsConfig
    from wms.physics.lc2013 import lane_change

    torch.set_num_threads(1)
    root, world, support, idm_pop, idm_drivers = job
    path = native_episode(Path(root), world)
    arrays, provenance = read_episode(path, world)
    t, kin, mask, cav, ids = (arrays[k] for k in
                             ("time_s", "veh_kin", "veh_mask", "is_cav", "vid"))
    candidates, slots = len(support), kin.shape[1]
    physics = PhysicsConfig(sumo_fidelity=True, dt_s=float(t[1] - t[0]))
    memory = None
    counts = {"fit": {}, "validation": {}}
    events = []
    p = {name: torch.as_tensor(support[:, j:j+1]).expand(candidates, slots)
         for j, name in enumerate(LC_NAMES)}
    records = {"fit": [], "validation": []}
    for ti in np.flatnonzero((t[:-1] >= 240) & (t[1:] <= 360)):
        normal0 = kin[ti, :, 1:4].sum(-1) == 1
        normal1 = kin[ti+1, :, 1:4].sum(-1) == 1
        same = mask[ti].astype(bool) & mask[ti+1].astype(bool) & (ids[ti] == ids[ti+1])
        present = same & normal0 & normal1 & (ids[ti] >= 0)
        eligible = present & ~cav[ti].astype(bool) & ~cav[ti+1].astype(bool)
        source = kin[ti, :, 1:4].argmax(-1)
        destination = kin[ti+1, :, 1:4].argmax(-1)
        moved = np.array(kin[ti], copy=True)
        moved[:, [0, 4, 5]] = np.asarray(kin[ti+1])[:, [0, 4, 5]]
        for name in NAMES:
            values = [idm_drivers.get(f"{world}:{int(vid)}", {}).get("median", idm_pop)[name]
                      for vid in ids[ti]]
            p[name] = torch.as_tensor(values, dtype=torch.float64).expand(candidates, slots)
        k = torch.as_tensor(moved).expand(candidates, slots, 7)
        m = torch.as_tensor(present).expand(candidates, slots)
        identity = torch.as_tensor(np.array(ids[ti], copy=True)).expand(candidates, slots)
        decisions = torch.as_tensor(eligible).expand(candidates, slots)
        memory = lane_change(k, m, physics, memory=memory, identity=identity,
                             decision_mask=decisions, driver_params=p)
        predicted = memory["next_lane"].numpy() != source[None, :]
        actual = destination != source
        split = "fit" if t[ti+1] <= 300 else "validation"
        for j in np.flatnonzero(eligible):
            key = f"{world}:{int(ids[ti, j])}"
            c = counts[split].setdefault(key, np.zeros((candidates, 4), dtype=np.int64))
            channel = np.where(predicted[:, j], 0 if actual[j] else 1, 2 if actual[j] else 3)
            c[np.arange(candidates), channel] += 1
            records[split].append((key, bool(actual[j]), np.packbits(predicted[:, j])))
            if actual[j]:
                events.append({"identity": key, "time_s": float(t[ti+1]), "position_m": float(kin[ti+1,j,0]),
                               "from_lane": int(source[j]), "to_lane": int(destination[j]), "split": split})
    print(json.dumps({"world": world, "drivers": len(counts["fit"]), "events": len(events)}), flush=True)
    return world, counts, records, events, provenance


def evaluate(records, weights, posteriors, p0, p1, candidates):
    errors, nll = {"population": [], "posterior": [], "event_rate": []}, {k: [] for k in
                                                                                   ("population", "posterior", "event_rate")}
    event_rate = sum(y for _, y, _ in records["fit"]) / len(records["fit"])
    for key, y, bits in records["validation"]:
        predictions = np.unpackbits(bits)[:candidates]
        for name, w in (("population", weights), ("posterior", posteriors.get(key, weights))):
            probability = p0 + (p1-p0) * float(predictions @ w)
            errors[name].append((probability-y)**2)
            nll[name].append(float(-xlogy(y, probability)-xlogy(1-y, 1-probability)))
        errors["event_rate"].append((event_rate-y)**2)
        nll["event_rate"].append(float(-xlogy(y, event_rate)-xlogy(1-y, 1-event_rate)))
    return {name: {"n": len(e), "brier": float(np.mean(e)), "negative_log_score": float(np.mean(nll[name]))}
            for name, e in errors.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--idm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sobol-power", type=int, default=5)
    parser.add_argument("--idm-mode", choices=("pooled", "posterior"), default="pooled")
    parser.add_argument("--idm-pooled", type=Path, help="Explicit wm_phys.idm_pooled.v1 candidate")
    parser.add_argument("--worlds", type=int, nargs="+", default=list(FIT_WORLDS))
    args = parser.parse_args()
    if not set(args.worlds) <= set(FIT_WORLDS):
        raise ValueError("Non-FIT world rejected before data reads")
    if not args.out.parent.is_dir():
        raise ValueError("Output parent must exist")
    args.out.mkdir(exist_ok=False)
    start = time.time()
    # Source SUMO defaults=1; support log-radius 2 is a numerical search domain,
    # not a threshold in the LC model. Finite support limits identification.
    points = qmc.Sobol(len(LC_NAMES), scramble=True, seed=20261031).random_base2(args.sobol_power)
    support = np.concatenate((np.ones((1, len(LC_NAMES))), np.exp(4*points-2)), axis=0)
    idm_artifact = json.loads((args.idm/"idm_population.json").read_text())
    idm_pop = idm_artifact["pooled_fit_median" if args.idm_mode == "pooled" else "median"]
    if args.idm_pooled is not None:
        if args.idm_mode != "pooled":
            raise ValueError("Explicit pooled candidate requires pooled IDM mode")
        pooled = json.loads(args.idm_pooled.read_text())
        if pooled["schema"] != "wm_phys.idm_pooled.v1":
            raise ValueError("Unsupported pooled IDM schema")
        idm_pop = pooled["parameters"]
    idm_drivers = ({} if args.idm_mode == "pooled" else
                   json.loads((args.idm/"idm_posteriors.json").read_text())["drivers"])
    jobs = ((str(args.root), w, support, idm_pop, idm_drivers) for w in args.worlds)
    counts, records, events, provenance = {"fit": {}, "validation": {}}, {"fit": [], "validation": []}, [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for _, cc, rr, ee, pp in executor.map(episode_job, jobs, chunksize=1):
            for split in counts:
                counts[split].update(cc[split])
                records[split].extend(rr[split])
            events.extend(ee)
            provenance.append(pp)
    keys = sorted(counts["fit"])
    fit_counts = np.stack([counts["fit"][key] for key in keys])
    weights, p0, p1, posterior, trace = fit_mixture(fit_counts)
    posterior_by_key = dict(zip(keys, posterior))
    mean = weights @ support
    log_mean = weights @ np.log(support)
    covariance = (np.log(support)-log_mean).T @ (weights[:,None]*(np.log(support)-log_mean))
    module_path = Path(__import__("wms.physics.lc2013", fromlist=["__file__"]).__file__)
    pop = {"schema": "wm_phys.lc2013_population.v1", "parameter_order": list(LC_NAMES),
           "distribution": "finite_support_mixture", "support": support.tolist(), "weights": weights.tolist(),
           "mean": dict(zip(LC_NAMES, mean.tolist())), "log_mean": log_mean.tolist(), "log_cov": covariance.tolist(),
           "modal_parameters": dict(zip(LC_NAMES, support[weights.argmax()].tolist())),
           "observation_emission": {"event_given_proposal": p1, "event_without_proposal": p0,
                                    "note": "Discrepancy likelihood only; do NOT use as runtime event noise"},
           "likelihood_trace": trace, "n_drivers": len(keys),
           "n_events": sum(e["split"] == "fit" for e in events),
           "fit_interval_s": [240,300], "fit_worlds": args.worlds,
           "idm_mode": args.idm_mode,
           "idm_parameters": idm_pop,
           "lc2013_source_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
           "known_limits": ["finite numerical support, not unrestricted ML", "module is highway LC2013 subset",
                            "wish state left-censored/zero at t=240", "stepwise emission conditional independence",
                            "IDM conditioned on causal fitted medians, not jointly integrated"]}
    write_json(args.out/"lc2013_population.json", pop)
    driver_json = {k: {"weights": w.tolist(), "n_observations": int(counts["fit"][k][0].sum()),
                      "information_gain_nats": float(np.sum(xlogy(w, w/weights)))}
                   for k,w in posterior_by_key.items()}
    write_json(args.out/"lc2013_posteriors.json", {"schema": "wm_phys.lc2013_posteriors.v1",
        "conditioning_end_s": 300, "identity_key": "world:observable_vid", "drivers": driver_json,
        "fallback": "lc2013_population.json"})
    write_json(args.out/"lane_events.json", {"events": events})
    write_json(args.out/"provenance.json", {"sources": provenance, "idm_artifact_dir": str(args.idm),
        "idm_pooled_sha256": (hashlib.sha256(args.idm_pooled.read_bytes()).hexdigest()
                               if args.idm_pooled is not None else None),
        "idm_population_sha256": hashlib.sha256((args.idm/"idm_population.json").read_bytes()).hexdigest(),
        "idm_posteriors_sha256": hashlib.sha256((args.idm/"idm_posteriors.json").read_bytes()).hexdigest()})
    report = {"elapsed_s": time.time()-start, "n_fit_intervals": len(records["fit"]),
              "n_validation_intervals": len(records["validation"]),
              "n_validation_events": sum(e["split"] == "validation" for e in events),
              "later_FIT_event_scores": evaluate(records, weights, posterior_by_key, p0, p1, len(support)),
              "population_training_confusion": score_counts(fit_counts, np.tile(weights,(len(keys),1)),p0,p1),
              "posterior_training_confusion": score_counts(fit_counts, posterior,p0,p1)}
    write_json(args.out/"validation.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
