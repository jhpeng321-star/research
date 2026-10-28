"""Zero-SUMO same-state sensitivity replay; NOT measured combined-law parity.

Old native tapes supply the observed native next acceleration only as an offline
target. Both branches share a TRAIN-inferred cap because old tapes omit SUMO's
actual prospective cap. Do not use this output as a real-SUMO deployment receipt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay import executor, sha, states_from_episode, summary
from wms.control.continuous import native_relative, project_native, scalar_native_relative, residual_cap_mask
from wms.control.execution import authority_guard, contract, inferred_follow_cap
from wms.control.native_cav import head_native_accel
from wms.model.cav import EXEC_CAP_KEY


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("data","policy","driver","out","authority"):
        ap.add_argument("--"+name,type=Path,required=True)
    args = ap.parse_args()
    os.environ["C4_EXECUTION_PARITY"] = "1"
    os.environ["C4_EXECUTION_DRIVER"] = str(args.driver.resolve())
    os.environ["C4_POLICY_STOCHASTIC"] = "0"
    os.environ["C4_POLICY_ACCEL_NOISE"] = "0"
    exe,_ = executor(args.policy)
    results,hashes = {},{}
    all_error,all_head,all_r,all_delta,all_project,all_target_project,all_caps = [],[],[],[],[],[],[]
    for seed in range(977000,977020,2):
        path = args.data/f"e3final/main/e3final/shard0/jobs/s{seed}_native/deploy/episode/episode.npz"
        hashes[str(path.relative_to(args.data))] = sha(path)
        with np.load(path,allow_pickle=False) as tape:
            z = dict(tape)
        state_at = states_from_episode(z)
        errors,heads,residuals,deltas,projections,target_projections,caps = [],[],[],[],[],[],[]
        for t in z["control_time_s"]:
            if t<300 or int(round(t-300))%10:
                continue
            i = int(np.searchsorted(z["time_s"],t))
            if i+1>=len(z["time_s"]):
                continue
            st = state_at(i)
            keep = torch.nonzero(st["present"][0]).flatten()
            ix = keep.numpy()
            st = {k:v.index_select(1,keep) for k,v in st.items()}
            valid = (z["is_cav"][i,ix]&z["present"][i+1,ix]&(z["lane"][i,ix]>=0)
                     &(z["vid_index"][i,ix]==z["vid_index"][i+1,ix]))
            real = (z["v_mps"][i+1,ix].astype(float)-z["v_mps"][i,ix])/.5
            with torch.no_grad():
                r = torch.as_tensor(exe(st).residual)
                st[EXEC_CAP_KEY] = inferred_follow_cap(st,None)
                head = head_native_accel(st)
                base = project_native(st,head)
                target_base = project_native(st,torch.as_tensor(real[None],dtype=torch.float32))
                predicted = native_relative(st,r,base)
                cap_mask = residual_cap_mask(st,r,base)
            v = st["v_mps"][0].numpy()
            cap = st[EXEC_CAP_KEY][0].numpy()
            reference = np.array([scalar_native_relative(float(vv),float(vv)+.5*float(aa),
                       float(cc),float(rr)) for vv,aa,cc,rr in
                       zip(v,target_base[0].numpy(),cap,r[0].numpy())])
            errors.extend((predicted[0].numpy()-reference)[valid].tolist())
            heads.extend((head[0].numpy()-real)[valid].tolist())
            residuals.extend(r[0].numpy()[valid].tolist())
            deltas.extend((predicted-base)[0].numpy()[valid].tolist())
            projections.extend((base-head)[0].numpy()[valid].tolist())
            target_projections.extend((target_base[0].numpy()-real)[valid].tolist())
            caps.extend(cap_mask[0].numpy()[valid].tolist())
        results[str(seed)] = {"n":len(errors),"prediction_minus_proxy":summary(errors),
            "head_minus_recorded_native":summary(heads),
            "wm_native_projection":summary(projections),
            "recorded_native_projection":summary(target_projections),
            "fully_capped_or_zero_fraction":float((np.asarray(deltas)==0.).mean()),
            "any_capped_or_zero_fraction":float((np.asarray(caps)|(np.asarray(residuals)==0.)).mean()),
            "positive_residual_fraction":float((np.asarray(residuals)>0.).mean())}
        for dest,source in ((all_error,errors),(all_head,heads),(all_r,residuals),
                            (all_delta,deltas),(all_project,projections),(all_target_project,target_projections),
                            (all_caps,caps)):
            dest.extend(source)
        print(seed,len(errors),flush=True)
    rng = np.random.default_rng(20261028)
    draws = rng.integers(0,10,(5000,10))
    intervals = {}
    for key in ("prediction_minus_proxy","head_minus_recorded_native"):
        x = np.array([w[key]["rmse"] for w in results.values()])
        intervals[key] = {"world_mean_rmse":float(x.mean()),
                         "ci95":np.quantile(x[draws].mean(1),[.025,.975]).tolist()}
    result = {"schema":"execparity_continuous_recorded_state/1","contract":contract(),
        "checkpoint_sha256":sha(args.policy),"driver_sha256":sha(args.driver),
        "sampling":"every tenth post-warmup control tick, native tapes, stable CAV identity",
        "worlds":results,"recordings":hashes,"prediction_minus_proxy":summary(all_error),
        "native_head_error":summary(all_head),"wm_native_projection":summary(all_project),
        "recorded_native_projection":summary(all_target_project),"world_bootstrap":intervals,
        "fully_capped_or_zero_fraction":float((np.asarray(all_delta)==0.).mean()),
        "any_capped_or_zero_fraction":float((np.asarray(all_caps)|(np.asarray(all_r)==0.)).mean()),
        "proxy_within_native_head_error":summary(all_error)["rmse"]<=summary(all_head)["rmse"],
        "measured_real_sumo":False,"deployment_passed":False,
        "limitations":["target uses observed native next speed only OFFLINE, never an actor feature",
            "shared TRAIN-inferred cap is not a measured SUMO cap; shared model error cancels",
            "old tapes cannot validate new-law realised acceleration or backend timing",
            "frozen stock SUMO lacks the same-step native-residual setter"]}
    authority = authority_guard(all_r,executed_delta=all_delta,capped=all_caps,checkpoint_sha256=sha(args.policy),
        recordings_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())
    authority["recordings"] = hashes
    authority["sampling"] = result["sampling"]
    for path,value in ((args.out,result),(args.authority,authority)):
        assert path.parent.is_dir()
        path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("worlds","recordings")},indent=2))


if __name__ == "__main__":
    main()
