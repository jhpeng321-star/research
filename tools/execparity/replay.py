"""Zero-SUMO replay; observed outcomes and reconstructed/counterfactual values stay labelled."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from wms.control.policy import PolicyExecutor, load_policy_components
from wms.control.native_cav import head_native_accel
from wms.control.nativeinit import compose_native
from wms.control.safety import safe_accel

torch.set_num_threads(1)


def states_from_episode(z):
    def state(i):
        p = z["present"][i]
        out = {key: torch.as_tensor(np.where(p, z[key][i], 0.)[None], dtype=torch.float32)
               for key in ("s_m", "v_mps", "a_mps2")}
        out.update({"present": torch.as_tensor(p[None], dtype=torch.bool),
                    "is_cav": torch.as_tensor((z["is_cav"][i] & p)[None], dtype=torch.bool),
                    "lane": torch.as_tensor(z["lane"][i][None].astype(np.int64)),
                    "junction": torch.as_tensor(z["junction"][i][None], dtype=torch.bool),
                    "lane_carry": torch.as_tensor(
                        (z["lane_carry"][i] if "lane_carry" in z else z["lane"][i]
                         )[None].astype(np.int64)),
                    "length_m": torch.full((1, len(p)), 5., dtype=torch.float32)})
        return out
    return state


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def executor(path):
    p, e, rules, payload = load_policy_components(str(path), device="cpu")
    return PolicyExecutor(p, e, lane_rules=rules, deterministic=True, device="cpu"), payload


def summary(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": len(x), "mean": float(x.mean()), "rmse": float(np.sqrt((x*x).mean())),
            "mae": float(np.abs(x).mean()),
            "q00_05_50_95_100": np.quantile(x, [0, .05, .5, .95, 1]).tolist()}


def world(z, on, off, arm):
    state = states_from_episode(z)
    times, vids = z["time_s"], z["vid_index"]
    rows = []
    for c, t in enumerate(z["control_time_s"]):
        if t < 300:
            continue
        i = int(np.searchsorted(times, t))
        if i + 2 >= len(times):
            continue
        assert abs(times[i] - t) < 1e-6
        st = state(i)
        keep = torch.nonzero(st["present"][0]).flatten()
        st = {key: value.index_select(1, keep) for key, value in st.items()}
        with torch.no_grad():
            ron, roff = on(st).residual[0], off(st).residual[0]
            native = head_native_accel(st)
            old_imag = safe_accel(st, compose_native(native, torch.as_tensor(ron)[None], -3., 2.6),
                                  accel_max=2.6)[0].numpy()
            head = native[0].numpy()
        valid = (z["present"][i] & z["is_cav"][i] & (z["lane"][i] >= 0)
                 & z["present"][i+1] & z["present"][i+2]
                 & (vids[i] == vids[i+1]) & (vids[i] == vids[i+2]))
        for local_k, k in enumerate(keep.numpy()):
            if not valid[k]:
                continue
            rq = float(z["requested_acceleration_mps2"][c, k])
            issued = float(z["issued_acceleration_mps2"][c, k])
            a1 = (float(z["v_mps"][i+1, k])-float(z["v_mps"][i, k])) / .5
            a2 = (float(z["v_mps"][i+2, k])-float(z["v_mps"][i+1, k])) / .5
            rows.append([ron[local_k], roff[local_k], rq, issued, a1, a2, head[local_k], old_imag[local_k],
                         z["lane"][i+1,k] != z["lane"][i,k]])
    a = np.asarray(rows)
    r, off_r, rq, issued, a1, a2, head, old_imag, lc = a.T
    out = {"n": len(a), "residual_on": summary(r), "residual_off": summary(off_r),
           "positive_on": float((r > 0).mean()), "positive_off": float((off_r > 0).mean()),
           "on_minus_off": summary(r-off_r), "realised_a1": summary(a1),
           "realised_minus_native_head": summary(a1-head),
           "lane_change_fraction": float(lc.mean()),
           "native_head_error": summary(head-a1)}
    if arm != "native":
        finite = np.isfinite(rq) & np.isfinite(issued)
        out.update({"positive_command": float((rq[finite] > 0).mean()),
                    "requested": summary(rq), "issued_minus_requested": summary(issued-rq),
                    "realised_minus_requested": summary(a1-rq),
                    "within_0p01_command": float((np.abs(a1[finite]-rq[finite]) < .01).mean()),
                    "below_command_0p01": float((a1[finite] < rq[finite]-.01).mean()),
                    "realised_above_1p2": float((a1 > 1.2+1e-4).mean())})
    if arm == "ours":
        # Invert compose only when the logged command is strictly inside its envelope.
        # At the upper/lower bound rq-r is NOT an identifiable native follow command.
        identifiable = np.isfinite(rq) & (rq < 1.2-1e-5) & (rq > -3.+1e-5)
        follow = rq[identifiable] - r[identifiable]
        d = a1[identifiable]-follow
        out.update({"follow_identifiable_n": int(identifiable.sum()),
                    "realised_minus_follow_reconstructed": summary(d),
                    "at_or_below_follow_0p001": float((d <= .001).mean()),
                    "positive_residual_not_realised": float((d < r[identifiable]-.01).mean()),
                    "off_cap_implied_fraction": float((a1[identifiable] <= follow+off_r[identifiable]-.01).mean()),
                    "old_imagination_error": summary(old_imag-a1),
                    "follow_reconstruction_error": summary(-d)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--off", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    on_path = args.inputs / "policy_latest.pt"
    assert sha(on_path).startswith("64f70af4") and sha(args.off).startswith("e1fa4881")
    assert sha(args.inputs / "mappo_v5_final.pt").startswith("aecd7d70")
    on, _ = executor(on_path)
    off, _ = executor(args.off)
    results, hashes = {}, {}
    for seed in range(977000, 977020, 2):
        results[str(seed)] = {}
        for arm in ("ours", "native", "mappo"):
            path = args.data / f"e3final/main/e3final/shard0/jobs/s{seed}_{arm}/deploy/episode/episode.npz"
            hashes[str(path.relative_to(args.data))] = sha(path)
            with np.load(path, allow_pickle=False) as z:
                results[str(seed)][arm] = world(dict(z), on, off, arm)
        print(seed, results[str(seed)]["ours"]["n"], flush=True)
    # Equal-world estimates and cluster bootstrap: vehicles/ticks are not independent worlds.
    metrics = ("positive_on", "positive_off", "positive_command", "within_0p01_command",
               "below_command_0p01", "at_or_below_follow_0p001", "positive_residual_not_realised",
               "off_cap_implied_fraction", "lane_change_fraction", "realised_above_1p2")
    rng = np.random.default_rng(20261028)
    draws = rng.integers(0, len(results), (5000, len(results)))
    aggregate = {}
    for arm in ("ours", "native", "mappo"):
        aggregate[arm] = {"worlds": len(results), "n": sum(w[arm]["n"] for w in results.values())}
        for metric in metrics:
            if metric not in next(iter(results.values()))[arm]:
                continue
            values = np.array([w[arm][metric] for w in results.values()])
            aggregate[arm][metric] = {"mean": float(values.mean()),
                                      "ci95": np.quantile(values[draws].mean(1), [.025,.975]).tolist()}
        for key in ("realised_minus_follow_reconstructed", "realised_minus_native_head",
                    "requested", "realised_a1", "issued_minus_requested", "follow_reconstruction_error"):
            if key not in next(iter(results.values()))[arm]:
                continue
            values = np.array([w[arm][key]["mean"] for w in results.values()])
            aggregate[arm][key] = {"mean": float(values.mean()),
                                     "ci95": np.quantile(values[draws].mean(1), [.025,.975]).tolist()}
    output = {"schema": "execparity_replay/1", "recordings": hashes,
              "checkpoint_sha256": {"on": sha(on_path), "off": sha(args.off),
                                    "mappo": sha(args.inputs / "mappo_v5_final.pt")},
              "worlds": results, "equal_world_bootstrap": aggregate,
              "limitations": ["coop-off has no E3-final tape; replay residual and cap-implied prediction only",
                  "follow baseline reconstructed only for unclipped C1 commands; not true native yielding",
                  "observed command shortfall does not identify safety vs ownership vs junction cause",
                  "10 adjacent cpuP even worlds are a limited audit panel, not a new performance evaluation"]}
    assert args.out.parent.is_dir()
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
