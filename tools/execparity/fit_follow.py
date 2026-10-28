"""Learn CAV IDM parameters from observable TRAIN-val commanded tapes only, CPU.

This is a system-identification fit, never a policy/reward/value fit. The fit/score
world partition is native_cav_fit's pre-existing SHA1 70/30 definition. E3/E5 and
the forbidden 964 block are refused before any episode array is opened. The five
parameters are estimated, not copied from SUMO assignments or a fixed driver table.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay import ROOT, sha, summary
from wms.control.execution import idm_follow_speed
from wms.control.contract import MERGE_S1_M

BAD = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype", re.I)
NAMES = ("accel","decel","time_headway","min_gap","v0")


def guard(path):
    p = Path(path)
    if BAD.search(str(p)) or BAD.search(str(p.resolve())):
        raise ValueError("prohibited path")
    return p


def partition(world):
    return int(hashlib.sha1(world.encode()).hexdigest()[:8],16)%10 < 7


def rows(root, episodes):
    result, bindings = [], {}
    for ep in episodes:
        seed = int(ep["seed"])
        if 964000<=seed<=964199 or 973000<=seed<=973227 or 977000<=seed<=977999:
            raise ValueError("evaluation seed refused for driver fitting")
        directory = guard(root / ep["dir"])
        files = {n:guard(directory / (n+".npy")) for n in
                 ("veh_kin","veh_mask","is_cav","target_mask","controller",
                  "issued_acceleration_mps2","control_time_s","time_s")}
        data = {n:np.load(p,allow_pickle=False) for n,p in files.items()}
        bindings[ep["world"]] = {n:sha(p) for n,p in files.items()}
        kin,mask,cav,target = (data[n] for n in ("veh_kin","veh_mask","is_cav","target_mask"))
        mask,cav,target = mask.astype(bool),cav.astype(bool),target.astype(bool)
        # Fixed sampling, every tenth second; no filtering by target behaviour.
        for i in range(0,len(kin)-1,20):
            valid = mask[i] & mask[i+1] & cav[i] & target[i]
            c = int(np.searchsorted(data["control_time_s"],data["time_s"][i]))
            if c>=len(data["control_time_s"]) or data["control_time_s"][c]!=data["time_s"][i]:
                continue
            valid &= data["controller"][c]==3
            command = data["issued_acceleration_mps2"][c]
            valid &= np.isfinite(command)
            now = kin[i]
            lane = np.where(now[:,1:4].sum(-1)>0,now[:,1:4].argmax(-1),-1)
            ds = now[None,:,0]-now[:,None,0]
            ahead = (mask[i,:,None] & mask[i,None,:] & (lane[:,None]==lane[None,:]) & (ds>0))
            dist = np.where(ahead,ds,np.inf)
            leaders = dist.argmin(-1)
            has = np.isfinite(dist.min(-1))
            gap = np.where(has,dist.min(-1)-now[leaders,6],np.inf)
            lead_v = np.where(has,now[leaders,4],now[:,4])
            real = (kin[i+1,:,4].astype(float)-now[:,4])/.5
            valid &= lane>=0
            result.append(np.c_[now[valid,4],lead_v[valid],gap[valid],
                                  now[valid,0],lane[valid],command[valid],real[valid]])
    return torch.as_tensor(np.concatenate(result),dtype=torch.float64),bindings


def predict(x, values):
    params = {n:values[j].expand_as(x[:,0]) for j,n in enumerate(NAMES)}
    v,leader,gap,s,lane,command,_ = x.T
    target = idm_follow_speed(v,leader,gap,params)
    remaining = (MERGE_S1_M-s).clamp_min(0.)
    stop = idm_follow_speed(v,torch.zeros_like(v),remaining,params,respect_min_gap=False)
    stop = torch.minimum(stop,remaining/.5)
    target = torch.where(lane==2,torch.minimum(target,stop),target)
    target = torch.minimum(target,params["v0"])
    return torch.maximum(torch.minimum((target-v)/.5,command),-v/.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args = ap.parse_args()
    root,out = guard(args.data),guard(args.out)
    manifest = root/"C4_EPISODES_MANIFEST.json"
    episodes = [e for e in json.loads(manifest.read_text())["episodes"]
                if e["split_role"]=="val" and e["controller"]=="direct"
                and guard(root/e["dir"]).is_dir()]
    fit_eps = [e for e in episodes if partition(e["world"])]
    score_eps = [e for e in episodes if not partition(e["world"])]
    assert fit_eps and score_eps
    train,fit_hashes = rows(root,fit_eps)
    score,score_hashes = rows(root,score_eps)
    # Data-derived starts only. All five outputs remain learned positive scalars.
    init = torch.ones(5,dtype=train.dtype)
    init[-1] = train[:,0].max()
    raw = torch.nn.Parameter(init.clamp_min(torch.finfo(train.dtype).eps).log())
    opt = torch.optim.LBFGS([raw],max_iter=200,line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10,tolerance_change=1e-12)
    def closure():
        opt.zero_grad()
        loss = (predict(train,raw.exp())-train[:,-1]).square().mean()
        loss.backward()
        return loss
    opt.step(closure)
    values = raw.detach().exp()
    with torch.no_grad():
        fit_error = summary((predict(train,values)-train[:,-1]).numpy())
        score_error = summary((predict(score,values)-score[:,-1]).numpy())
    receipt = {"schema":"execparity_learned_idm/1","parameters":dict(zip(NAMES,values.tolist())),
               "algorithm":"SUMO IDM _v, dt .5, stepping .25, exponent 4; ramp lane-end stop",
               "source_manifest_sha256":sha(manifest),"fit_worlds":fit_hashes,"score_worlds":score_hashes,
               "fit_controller":"direct",
               "objective":"unweighted executed min(command,follow cap) acceleration MSE on fit worlds only",
               "partition":"existing native_cav_fit SHA1(world) mod10 <7 fit, >=7 score",
               "fit_error":fit_error,"score_error":score_error,
               "evaluation_data_used":False,"initialisation":"unit optimiser scalars; desired speed from observed maximum; no fixed driver table"}
    assert out.parent.is_dir() and not out.exists(),"freeze a new artifact; never overwrite a fit"
    out.write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k not in ("fit_worlds","score_worlds")},indent=2))


if __name__ == "__main__": main()
