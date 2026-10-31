"""Check expected accelerations and physical-profile CRPS on later native FIT.

CRPS/coverage concern the latent physical-profile distribution only; Gaussian
model-discrepancy residual noise is NOT injected into runtime accelerations.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from idm_fit import (FIT_WORLDS,extract_rows,native_episode,next_speed,observed,
                     read_episode,summarize,write_json)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--fit",type=Path,required=True)
    args=parser.parse_args()
    pop=json.loads((args.fit/"idm_population.json").read_text())
    drivers=json.loads((args.fit/"idm_posteriors.json").read_text())["drivers"]
    support=np.array(pop["support"])
    prior=np.array(pop["weights"])
    scores={k:{"error":[],"crps":[],"covered":[],"width":[]} for k in ("population","posterior")}
    for world in FIT_WORLDS:
        arrays,_=read_episode(native_episode(args.root,world),world)
        rows=extract_rows(arrays)[0]
        rows=rows[(rows[:,0]>=300)&(rows[:,0]<360)]
        for identity in np.unique(rows[:,1]).astype(int):
            rr=rows[rows[:,1]==identity]
            values=(next_speed(rr[:,3,None],rr[:,5,None],rr[:,6,None],support,rr[0,9])
                    -rr[:,3,None])/rr[:,9,None]
            order=values.argsort(axis=1)
            x=np.take_along_axis(values,order,axis=1)
            y=observed(rr)
            d=drivers.get(f"{world}:{identity}",{})
            posterior=prior.copy()
            if "weights" in d:
                posterior=np.zeros(len(prior))
                posterior[np.array(d["support_indices"])]=d["weights"]
            for name,w in (("population",prior),("posterior",posterior)):
                sw=w[order]
                cw=sw.cumsum(axis=1)
                crps=np.sum(w*abs(values-y[:,None]),axis=1)-np.sum(sw*x*(2*cw-sw-1),axis=1)
                lo=x[np.arange(len(x)),(cw>=0.025).argmax(axis=1)]
                hi=x[np.arange(len(x)),(cw>=0.975).argmax(axis=1)]
                scores[name]["error"].extend(values@w-y)
                scores[name]["crps"].extend(crps)
                scores[name]["covered"].extend((y>=lo)&(y<=hi))
                scores[name]["width"].extend(hi-lo)
    report={name:{"expected_acceleration_error_mps2":summarize(s["error"]),
        "physical_profile_acceleration_crps":float(np.mean(s["crps"])),
        "physical_profile_95pct_coverage":float(np.mean(s["covered"])),
        "mean_interval_width_mps2":float(np.mean(s["width"]))} for name,s in scores.items()}
    report["contract"]="Native FIT [300,360), prior/posterior fit only [240,300]; no residual acceleration noise"
    report["qualification"]="Diagnostic; not outcome-uncertainty calibration or free-run traffic qualification"
    out=args.fit/"predictive_validation.json"
    if out.exists():
        raise ValueError("Preserve existing predictive check")
    write_json(out,report)
    print(json.dumps(report))


if __name__=="__main__":
    main()
