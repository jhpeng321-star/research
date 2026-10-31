"""Verify published physical parameter artifacts without accessing any tapes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts",type=Path,default=Path("artifacts/rbfit"))
    args=parser.parse_args()
    root=args.artifacts
    all_checks=[]
    for name in ("gaussian_v1","finite_mixture_v4"):
        path=root/name
        pop=load(path/"idm_population.json")
        if name=="finite_mixture_v4":
            payload=(path/"idm_posteriors.json.gz").read_bytes()
            raw=gzip.decompress(payload)
            receipt=load(path/"posterior_publication.json")
            assert hashlib.sha256(raw).hexdigest()==receipt["raw_sha256"]
            assert hashlib.sha256(payload).hexdigest()==receipt["compressed_sha256"]
            post=json.loads(raw)
            support=np.array(pop["support"])
            weights=np.array(pop["weights"])
            assert np.isfinite(support).all() and (support>0).all()
            assert np.isclose(weights.sum(),1) and (weights>0).all()
            trace=[r["negative_log_likelihood"] for r in pop["em_trace"]]
            assert np.diff(trace).max(initial=0)<1e-7
        else:
            post=load(path/"idm_posteriors.json")
        assert pop["fit_interval_s"]==[240,300]
        assert pop["fit_worlds"]==list(range(969400,969432))
        assert post["conditioning_end_s"]==300
        provenance=load(path/"provenance.json")["sources"]
        assert sorted(p["world"] for p in provenance)==list(range(969400,969432))
        for source in provenance:
            assert source["episode"].endswith("__native")
            assert set(source["sha256"])=={"episode.json","time_s.npy","veh_kin.npy","veh_mask.npy","is_cav.npy","vid.npy"}
        for driver in post["drivers"].values():
            z=np.array(driver["log_mean"])
            cov=np.array(driver["log_cov"])
            assert np.isfinite(z).all() and np.isfinite(cov).all()
            assert np.linalg.eigvalsh(cov).min()>-1e-8
            assert 0<driver["observed_history_s"]<=60
            if "weights" in driver:
                assert np.isclose(sum(driver["weights"]),1)
                assert max(driver["support_indices"])<len(pop["support"])
        all_checks.append({"artifact":name,"drivers":len(post["drivers"]),
                           "fitted_drivers":pop["n_drivers"],"observations":pop["n_observations"]})
    lane=load(root/"lane_v1/lc2013_population.json")
    lp=load(root/"lane_v1/lc2013_posteriors.json")
    assert np.isclose(sum(lane["weights"]),1)
    assert np.diff(lane["likelihood_trace"]).min(initial=0)>-1e-7
    assert len(lp["drivers"])==lane["n_drivers"]
    assert lane["fit_interval_s"]==[240,300]
    report={"status":"PASS","checks":all_checks,"lc_drivers":lane["n_drivers"],
            "lc_fit_events":lane["n_events"],"source_worlds":32,"native_only":True,
            "causal_window_s":[240,300],"future_validation_s":[300,360],
            "qualification":"Integrity/schema/numerics only; validation limitations remain"}
    (root/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":
    main()
