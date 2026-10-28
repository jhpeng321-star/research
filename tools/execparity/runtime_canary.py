"""CPU native/MAPPO command identity and residual runtime-port source checks.

This is deliberately NOT a claimed SUMO trajectory canary. It compares frozen
E3-final and candidate runtime code/outputs on the same recorded observable states.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay import ROOT, executor, sha, states_from_episode


def function_tree(path,name):
    nodes = ast.parse(path.read_text(encoding="utf-8")).body
    return ast.dump(next(n for n in nodes if getattr(n,"name",None)==name),include_attributes=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e3-source",type=Path,required=True)
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--mappo",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args = ap.parse_args()
    old_path = args.e3_source/"src/wms/control/policy.py"
    spec = importlib.util.spec_from_file_location("wms.control._execparity_e3policy",old_path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    p,e,rules,_ = old.load_policy_components(str(args.mappo),device="cpu")
    reference = old.PolicyExecutor(p,e,lane_rules=rules,deterministic=True,device="cpu")
    candidate,_ = executor(args.mappo)
    assert sha(args.mappo).startswith("aecd7d70")
    ticks,rows = 0,0
    for seed in range(977000,977020,2):
        reference.reset(); candidate.reset()
        path = args.data/f"e3final/main/e3final/shard0/jobs/s{seed}_mappo/deploy/episode/episode.npz"
        with np.load(path,allow_pickle=False) as tape: z = dict(tape)
        state_at = states_from_episode(z)
        for t in z["control_time_s"]:
            if t<300 or int(round(t-300))%10: continue
            st = state_at(int(np.searchsorted(z["time_s"],t)))
            for k in ("present","is_cav","junction"): st[k] = st[k].bool()
            keep = torch.nonzero(st["present"][0]).flatten()
            st = {k:v.index_select(1,keep) for k,v in st.items()}
            a,lanes = reference(st,with_lanes=True)
            b,other = candidate(st,with_lanes=True)
            assert np.array_equal(a,b,equal_nan=True)
            assert (lanes is None and other is None) or np.array_equal(lanes,other)
            ticks+=1; rows+=int(st["is_cav"].sum())
    rel = "traffic_shepherd/c4/branching.py"
    ports = {}
    for name in ("split_output","BranchController","BranchTracker"):
        before = function_tree(args.e3_source/rel,name)
        after = function_tree(ROOT/rel,name)
        ports[name] = before==after
        assert ports[name],name
    from traffic_shepherd.c4.branching import BranchController
    native = BranchController("native")
    assert native.command(None,None,None,None)==(None,{})
    result = {"schema":"execparity_runtime_canary/1","mappo_sha256":sha(args.mappo),
              "source_policy_sha256":sha(old_path),"candidate_policy_sha256":sha(ROOT/"src/wms/control/policy.py"),
              "mappo_sampled_ticks":ticks,"mappo_cav_rows":rows,"commands_and_lanes_bit_identical":True,
              "native_releases_without_command":True,"e3_runtime_ports_ast_identical":ports,
              "sumo_trajectory_canary_run":False,"passed":True}
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
