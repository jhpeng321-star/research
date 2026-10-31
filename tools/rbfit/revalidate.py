"""Later native FIT verification against recorded endpoint coordinates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from idm_fit import (FIT_WORLDS,NAMES,extract_rows,native_episode,read_episode,
                     validate,write_json)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--fit",type=Path,required=True)
    args=parser.parse_args()
    pop=json.loads((args.fit/"idm_population.json").read_text())
    post=json.loads((args.fit/"idm_posteriors.json").read_text())["drivers"]
    worlds={}
    for world in FIT_WORLDS:
        arrays,_=read_episode(native_episode(args.root,world),world)
        worlds[world]=extract_rows(arrays)[0]
    def point(d):
        return (np.log([d["modal_parameters"][k] for k in NAMES])
                if "modal_parameters" in d else np.array(d["log_mean"]))
    report=validate(worlds,{k:point(v) for k,v in post.items()},point(pop),
                    np.log([pop["pooled_fit_median"][k] for k in NAMES]))
    report["position_target"]="actual recorded next_x; original Euler-reconstructed score retained separately"
    report["point_estimate"]="joint modal profile for finite mixture; log-mean for log-normal"
    out=args.fit/"validation_recorded_endpoints.json"
    if out.exists():
        raise ValueError("Preserve existing verification")
    write_json(out,report)
    print(json.dumps(report["acceleration_mps2"]))


if __name__=="__main__":
    main()
