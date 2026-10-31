"""Final all-row pooled physical least-squares fit for the deterministic variant."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from idm_fit import (FIT_WORLDS,NAMES,extract_rows,native_episode,observed,
                     prediction,read_episode,validate,write_json)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--initial",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if not args.out.parent.is_dir():
        raise ValueError("Output parent must exist")
    args.out.mkdir(exist_ok=False)
    old=json.loads((args.initial/"idm_population.json").read_text())["pooled_fit_median"]
    worlds,sources={},[]
    for world in FIT_WORLDS:
        a,p=read_episode(native_episode(args.root,world),world)
        worlds[world]=extract_rows(a)[0]
        sources.append(p)
    rows=np.concatenate([r[r[:,0]<300] for r in worlds.values()])
    result=least_squares(lambda z:prediction(rows,z)-observed(rows),np.log([old[k] for k in NAMES]),
                          bounds=(-10,10),max_nfev=200,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    payload={"schema":"wm_phys.idm_pooled.v1","parameters":dict(zip(NAMES,np.exp(result.x).tolist())),
        "delta":4,"emergency_decel_mps2":9,"idm_stepping_s":0.25,"observation_step_s":0.5,
        "fit_interval_s":[240,300],"fit_worlds":list(FIT_WORLDS),"n_observations":len(rows),
        "optimizer_success":bool(result.success),"optimizer_message":result.message,
        "training_acceleration_rmse_mps2":float(np.sqrt(np.mean(result.fun**2))),
        "uncertainty_note":"Pooled point estimate, not a driver population spread or calibrated uncertainty"}
    write_json(args.out/"idm_pooled.json",payload)
    report=validate(worlds,{},result.x,result.x)
    report["contract"]+="; population/posterior/pooled labels all refer to the same pooled point fit here"
    write_json(args.out/"validation_recorded_endpoints.json",report)
    write_json(args.out/"provenance.json",{"sources":sources})
    print(json.dumps({"fit":payload,"validation":report["acceleration_mps2"]}))


if __name__=="__main__":
    main()
