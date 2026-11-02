"""Independently verify LC exports against saved observable decision records.

Does not import fitter algebra. Rebuilds sequence likelihoods directly from
decision arrays, verifies causal evidence shrinkage and separately scores RAW
physical proposals (without the observation-discrepancy emissions).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def probabilities(predictions, source, weights, emission=None):
    components = np.eye(3)[predictions] if emission is None else emission[source[:,None], predictions]
    return np.sum(weights[...,None]*components, axis=1)


def scores(probability, actual):
    truth = np.eye(3)[actual]
    return {"n": len(actual), "events": int((actual != 0).sum()),
        "categorical_brier_sum": float(np.sum((probability-truth)**2)),
        "event_brier_sum": float(np.sum((1-probability[:,0]-(actual != 0))**2)),
        "negative_log_sum": float(-np.log(np.maximum(probability[np.arange(len(actual)),actual],
                                                     np.finfo(float).tiny)).sum()),
        "expected_event_count": float(np.sum(1-probability[:,0])),
        "expected_correct_direction_event_count": float(np.sum(probability[np.arange(len(actual)),actual]*(actual!=0)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        raise ValueError("Verification receipt must be NEW under existing parent")
    root = args.fit
    pop = json.loads((root/"lc2013_population.json").read_text())
    post = json.loads((root/"lc2013_posteriors.json").read_text())
    policy = json.loads((root/"lc2013_shrinkage.json").read_text())
    validation = json.loads((root/"validation.json").read_text())
    provenance = json.loads((root/"provenance.json").read_text())
    support, weights = np.array(pop["support"]), np.array(pop["weights"])
    emission = np.array(pop["observation_emission"]["probabilities"])
    assert support.shape == (len(weights),5) and np.isfinite(support).all()
    assert (support[:,:4] >= 0).all() and (support[:,4] > 0).all()
    if pop.get("requires_boundary_aware_lc_reader"):
        assert (support[:,1] <= 1).all()
    assert (weights >= 0).all() and np.isfinite(weights).all()
    np.testing.assert_allclose(weights.sum(), 1, rtol=0, atol=1e-12)
    assert emission.shape == (3,3,3) and (emission > 0).all()
    np.testing.assert_allclose(emission.sum(-1), 1, rtol=0, atol=1e-12)
    assert post["conditioning_end_s"] == 300 and post["shrinkage_already_applied"] is True
    assert pop["fit_interval_s"] == [240,300]
    assert set(pop["fit_worlds"]) <= set(range(969400,969432))
    assert provenance["physical_source_sha256"]["lc2013.py"] == pop["lc2013_source_sha256"]
    strength = np.inf if policy["population_only"] else policy["prior_count"]
    assert strength >= 0
    all_rows, seen, max_error = [], set(), 0.0
    source_counts = np.zeros((3,3), dtype=int)
    raw_calibration = np.zeros(2)
    perdriver_means = []
    n_no_evidence, n_informative, zero_exposure = 0, 0, 0
    for world in pop["fit_worlds"]:
        with np.load(root/f"observations_{world}.npz", allow_pickle=False) as archive:
            r = {key: archive[key] for key in ("vid","time_s","source","actual","prediction")}
        assert (r["time_s"] > 240).all() and (r["time_s"] <= 360).all()
        assert r["prediction"].shape == (len(r["vid"]),len(weights))
        assert ((r["prediction"] >= 0) & (r["prediction"] <= 2)).all()
        assert ((r["source"] >= 0) & (r["source"] <= 2)).all()
        assert ((r["actual"] >= 0) & (r["actual"] <= 2)).all()
        fit = r["time_s"] <= 300
        np.add.at(source_counts, (r["source"][fit], r["actual"][fit]), 1)
        ids = np.unique(r["vid"])
        distribution = np.tile(weights, (len(ids),1))
        before = distribution.copy()
        nn = np.zeros(len(ids), dtype=int)
        for j, vid in enumerate(ids):
            observed = r["vid"] == vid
            sequence = observed & fit
            prefix = observed & (r["time_s"] <= 270)
            key = f"{world}:{int(vid)}"
            evidence = np.any(r["prediction"] != r["prediction"][:, :1],axis=1)
            n = int(np.sum(sequence & evidence))
            nn[j] = np.sum(prefix & evidence)
            def posterior(selected):
                log_prior = np.full_like(weights,-np.inf)
                np.log(weights,out=log_prior,where=weights>0)
                log_joint = log_prior+np.log(emission[
                    r["source"][selected,None],r["prediction"][selected],r["actual"][selected,None]]).sum(0)
                result = np.exp(log_joint-log_joint.max())
                return result/result.sum()
            raw, before[j] = posterior(sequence), posterior(prefix)
            fraction = n/(n+strength) if n+strength > 0 else 0.0
            expected = fraction*raw+(1-fraction)*weights
            if not np.any(sequence):
                assert key not in post["drivers"]
                zero_exposure += 1
                continue
            actual = np.array(post["drivers"][key]["weights"])
            max_error = max(max_error, float(np.max(np.abs(actual-expected))))
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)
            assert post["drivers"][key]["n_observations"] == int(sequence.sum())
            assert post["drivers"][key]["n_informative_intervals"] == n
            if n == 0:
                np.testing.assert_allclose(actual, weights, rtol=1e-11, atol=1e-12)
                n_no_evidence += 1
            else:
                n_informative += 1
            distribution[j] = actual
            perdriver_means.append(actual@support)
            seen.add(key)
        driver = np.searchsorted(ids, r["vid"])
        later = ~fit
        row = {"world": world}
        for label, w in (("population",np.tile(weights,(len(ids),1))), ("shrunk_posterior",distribution)):
            for family, e in (("likelihood", emission),("raw_physics",None)):
                probability = probabilities(r["prediction"][later],r["source"][later],w[driver[later]],e)
                row[f"{family}_{label}"] = scores(probability,r["actual"][later])
        default = np.eye(3)[r["prediction"][later,0]]
        row["raw_physics_source_default"] = scores(default,r["actual"][later])
        # Independently recompute the selected causal prefix-calibration score.
        suffix = (r["time_s"] > 270) & fit
        calibration_emission = (None if policy.get("calibration_objective","").startswith("raw physical") else emission)
        p_raw = probabilities(r["prediction"][suffix],r["source"][suffix],before[driver[suffix]],calibration_emission)
        p_pop = probabilities(r["prediction"][suffix],r["source"][suffix],np.tile(weights,(suffix.sum(),1)),calibration_emission)
        n_prefix = nn[driver[suffix]]
        f = np.divide(n_prefix,n_prefix+strength,out=np.zeros_like(n_prefix,dtype=float),where=n_prefix+strength>0)
        p = f[:,None]*p_raw+(1-f[:,None])*p_pop
        raw_calibration += [np.sum((p-np.eye(3)[r["actual"][suffix]])**2),suffix.sum()]
        all_rows.append(row)
    assert seen == set(post["drivers"])
    np.testing.assert_allclose(raw_calibration[0]/raw_calibration[1],policy["categorical_brier"],atol=1e-12)
    aggregate = {}
    for name in all_rows[0]:
        if name == "world":
            continue
        total = {k: sum(r[name][k] for r in all_rows) for k in all_rows[0][name]}
        n = total["n"]
        aggregate[name] = dict(total, categorical_brier=total["categorical_brier_sum"]/n,
                               event_brier=total["event_brier_sum"]/n,
                               negative_log_score=total["negative_log_sum"]/n)
    for label in ("population","shrunk_posterior"):
        saved = validation["later_FIT_directional_scores"][label]
        for metric in ("categorical_brier","event_brier","negative_log_score"):
            np.testing.assert_allclose(saved[metric],aggregate["likelihood_"+label][metric],atol=1e-12)
    # World-cluster bootstrap is descriptive internal-FIT uncertainty only.
    rng = np.random.default_rng(20261101)
    bootstrap = []
    for _ in range(2000):
        selected = rng.integers(0,len(all_rows),len(all_rows))
        a = sum(all_rows[i]["likelihood_shrunk_posterior"]["categorical_brier_sum"] for i in selected)
        b = sum(all_rows[i]["likelihood_population"]["categorical_brier_sum"] for i in selected)
        n = sum(all_rows[i]["likelihood_population"]["n"] for i in selected)
        bootstrap.append((a-b)/n)
    mean = weights@support
    receipt = {"status":"PASS", "drivers":len(seen), "zero_information_population_fallbacks":n_no_evidence,
        "informative_drivers":n_informative,"validation_only_population_fallbacks":zero_exposure,
        "maximum_direct_sequence_posterior_error":max_error,
        "aggregate":aggregate, "world_scores":all_rows,
        "internal_FIT_posterior_minus_population_categorical_brier_ci95": np.quantile(bootstrap,[.025,.975]).tolist(),
        "population_physical_mean":mean.tolist(),
        "population_physical_sd":np.sqrt(weights@((support-mean)**2)).tolist(),
        "across_driver_posterior_mean_sd":np.std(perdriver_means,axis=0).tolist(),
        "adopted":False,"physical_source_sha256":provenance["physical_source_sha256"],
        "artifact_sha256":{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
            "lc2013_population.json","lc2013_posteriors.json","lc2013_shrinkage.json",
            "validation.json","provenance.json","lane_events.json")}}
    with args.out.open("x") as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:v for k,v in receipt.items() if k not in ("world_scores","artifact_sha256")}))


if __name__ == "__main__":
    main()
