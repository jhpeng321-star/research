"""Observable-only writer/lane-change census on the fetched tapes, with world CIs."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    worlds = {}
    for seed in range(977000,977020,2):
        worlds[str(seed)] = {}
        for arm in ("ours","native","mappo"):
            p = args.data / f"e3final/main/e3final/shard0/jobs/s{seed}_{arm}/deploy/episode/episode.npz"
            with np.load(p,allow_pickle=False) as tape: z = dict(tape)
            meta = json.loads(str(z["meta_json"]))
            counts = dict(stable_cav_ticks=0,controller_native=0,requested_finite=0,
                          native_with_finite_request=0,issued_not_requested_0p01=0,
                          realised_not_issued_0p01=0,changed_lane=0)
            for c,t in enumerate(z["control_time_s"]):
                if t < 300: continue
                i = int(np.searchsorted(z["time_s"],t))
                if i+1 >= len(z["time_s"]): continue
                m = (z["present"][i] & z["is_cav"][i] & (z["lane"][i]>=0)
                     & z["present"][i+1] & (z["vid_index"][i]==z["vid_index"][i+1]))
                rq,iss,ctrl = (z[k][c] for k in
                              ("requested_acceleration_mps2","issued_acceleration_mps2","controller"))
                real = (z["v_mps"][i+1].astype(float)-z["v_mps"][i])/.5
                f = m & np.isfinite(rq)
                counts["stable_cav_ticks"] += int(m.sum())
                counts["controller_native"] += int((m & (ctrl==0)).sum())
                counts["requested_finite"] += int(f.sum())
                counts["native_with_finite_request"] += int((f & (ctrl==0)).sum())
                counts["issued_not_requested_0p01"] += int((f & (np.abs(iss-rq)>.01)).sum())
                counts["realised_not_issued_0p01"] += int((f & (np.abs(real-iss)>.01)).sum())
                counts["changed_lane"] += int((m & (z["lane"][i+1]!=z["lane"][i])).sum())
            worlds[str(seed)][arm] = {"metadata":meta,"counts":counts}
    rng = np.random.default_rng(20261028)
    draws = rng.integers(0,10,(5000,10))
    aggregate = {}
    for arm in ("ours","native","mappo"):
        rows = [w[arm]["counts"] for w in worlds.values()]
        aggregate[arm] = {}
        for key in ("controller_native","native_with_finite_request","issued_not_requested_0p01",
                    "realised_not_issued_0p01","changed_lane"):
            denominator = "requested_finite" if "requested" in key or "issued" in key else "stable_cav_ticks"
            x = np.array([r[key]/max(1,r[denominator]) for r in rows])
            aggregate[arm][key] = {"n":sum(r[key] for r in rows),"denominator":denominator,
                                   "mean":float(x.mean()),"ci95":np.quantile(x[draws].mean(1),[.025,.975]).tolist()}
    args.out.write_text(json.dumps({"schema":"execparity_ledger/1","worlds":worlds,
                                   "equal_world_bootstrap":aggregate},indent=2)+"\n")
    print(json.dumps(aggregate,indent=2))


if __name__ == "__main__": main()
