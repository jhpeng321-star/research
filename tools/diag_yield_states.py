"""S-YIELD: build WM state records from converted TRAIN episodes (r9_train), no SUMO / model.
Every 8th episode (sorted by name) of each source, t0 in {450, 750}; records use the c4_episode_states layout
(initial_state, 60 s history, recorded native future for the t0 roster). Output: gzip json."""
import glob, gzip, json, sys
import numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, "src"); sys.path.insert(0, ".")
from diag_yield_extract import ROOT, lanes
import c4_episode_states as ces
ces.counter_at = lambda ep, j0: [[0.0] * 30, [0.0] * 30]   # entries OFF in the WM rollouts; no torch locally

OUT = sys.argv[1]
T0S = [450.0, 750.0]
recs = []
for src in ("v5_native", "genrand_plain"):
    paths = sorted(glob.glob(f"{ROOT}/{src}/episodes/*"))[::8]
    for p in paths:
        k = np.load(p + "/veh_kin.npy"); m = np.load(p + "/veh_mask.npy").astype(bool)
        vid = np.load(p + "/vid.npy").astype(np.int64); cav = np.load(p + "/is_cav.npy").astype(bool)
        t = np.load(p + "/time_s.npy"); meta = json.load(open(p + "/episode.json", encoding="utf-8-sig"))
        lane = lanes(k)
        nv = int(vid.max()) + 1
        is_cav = np.zeros(nv, bool)
        for kk in range(k.shape[1]):
            sel = m[:, kk] & cav[:, kk]
            is_cav[vid[sel, kk]] = True
        ep = {"time_s": t, "s_m": k[..., 0], "lane": lane, "v_mps": k[..., 4], "a_mps2": k[..., 5], "present": m,
              "vid_index": np.where(m, vid, -1), "vehicle_ids": np.array([f"v{i}" for i in range(nv)]),
              "cav_vehicle_ids": np.array([f"v{i}" for i in np.flatnonzero(is_cav)]), "meta": meta}
        for t0 in T0S:
            r = ces.episode_record(ep, t0_s=t0, arm="native", world=meta["world"] + f"@{int(t0)}")
            r["source"] = src; r["episode"] = meta["episode"]; r["t0_s"] = t0
            recs.append(r)
with gzip.open(OUT, "wt") as f:
    json.dump(recs, f)
print(len(recs), "records")
