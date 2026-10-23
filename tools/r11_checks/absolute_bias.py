"""R11 M8 (R11_REVIEW C3): absolute target-follower bias around the merge + the registered keep-10b rule.

Reads the outputs of the reused tools/r10b_checks run (``<shm>/roll/<arm>/*.npz``, ``<shm>/os/onestep_<arm>.npz``).
Label mapping of tools/r11_checks/run_all.sh: ``r9`` = round-10b reference, ``r10b`` = round 11, ``headzero`` = round 11
with the head zeroed; ``sumo`` = recorded futures of the same states (rollout reference).

Computed per arm:
* one-step (teacher-forced, recorded states): model minus recorded acceleration of the target follower per event time
  over [-6, +2] s (pre-merge: partner rows by time until the partner's merge; post-merge: cut-in follower rows by time
  since the insertion), the mean over [-3, 0) s, and the bias of the kernel-matched UNPARTNERED lane-1 controls of the
  same window for the partner rows of the last 3 s (headway coordinates, the R11 loss's matcher);
* rollout: mean acceleration of the merge follower per event time minus the recorded (sumo) mean, and over [-3, 0) s;
* behaviour: merges per run, stuck ramp cars (time-mean ramp cars in the merge zone; share of times with a ramp car
  waiting > 60 s), TTC < 1.5 s share, min gap at insertion (p5), post-insertion hard brakes (share of merge followers
  whose post-insertion minimum acceleration < -3 m/s^2).

Registered rule (R11_REVIEW C3, R11_REVIEW_PACKET section 7):
  KEEP ROUND 10b (reject round 11) if round 11's mean absolute bias over [-3, 0) s exceeds round 10b's by > 0.3 m/s^2
  (rollout or one-step) AND any of merge count, stuck ramp cars, TTC < 1.5 s share, min gap at insertion, post-insertion
  hard brakes is worse than in round 10b.  M5 (collisions / TTC worse -> reject the round) is unchanged (analyze.py).
  Under-yield flag (a read): round 11's M1 mean over [-3, 0) s > -0.1.
Usage: absolute_bias.py --shm DIR --out MD --json JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for extra in (HERE, os.path.join(HERE, "..", "r10b_checks")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import analyze as A  # noqa: E402  (tools/r10b_checks, reused unchanged)

REF, R11, HZ = "r9", "r10b", "headzero"
LABEL = {REF: "round 10b", R11: "round 11", HZ: "round 11 head-zero", "sumo": "recorded"}
OFFSETS = (-6, -5, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2)
END_M = 75.44
TH_BIAS = 0.3
TH_UNDER = -0.1


def onestep_profile(d):
    bias = d["fin"] - d["rec"]
    part = (d["has"] == 1) & np.isfinite(d["tm"])
    post = (d["ev"] >= 0) & np.isfinite(d["te"])
    prof = {}
    for o in OFFSETS:
        sel = (part & (d["tm"] > -o - 0.25) & (d["tm"] <= -o + 0.25)) if o < 0 else \
            (post & (d["te"] >= o - 0.25) & (d["te"] < o + 0.25))
        prof[o] = A.boot_mean(bias[sel], d["win"][sel], 200)
    last3 = part & (d["tm"] > 0) & (d["tm"] <= 3.0)
    return prof, A.boot_mean(bias[last3], d["win"][last3], 300), last3


def matched_control_bias(d, last3):
    """Mean (model - recorded) of the kernel-matched unpartnered lane-1 controls of the same window, for the last-3-s
    partner rows (R11 loss matcher: v, time headway, closing speed, position)."""
    import torch
    from wms.train.losses.follower_courtesy import matched_control_residual

    gap = np.where(np.isfinite(d["gapL"]), d["gapL"], np.nan)
    hw = np.where(np.isfinite(gap), gap / np.maximum(d["v"], 0.5), 99.0)
    phi = np.stack([d["v"] / 10.0, np.arcsinh(hw), d["closing"] / 5.0, np.arcsinh((END_M - d["s"]) / 20.0)], -1)
    ctrl = (d["has"] == 0) & (d["lane"] == 1) & (d["rampbehind"] == 0)
    resid = d["rec"] - d["fin"]
    out, row_bias = [], []
    for w in np.unique(d["win"][last3]):
        rows = last3 & (d["win"] == w)
        cw = ctrl & (d["win"] == w) & np.isfinite(resid)
        if not cw.any():
            continue
        r_hat, _ = matched_control_residual(torch.tensor(phi[rows], dtype=torch.float32),
                                            torch.tensor(phi[cw], dtype=torch.float32),
                                            torch.tensor(resid[cw], dtype=torch.float32))
        out.append(-r_hat.numpy())
        row_bias.append((d["fin"] - d["rec"])[rows])
    if not out:
        return {"control_bias": float("nan"), "partner_bias": float("nan"), "n": 0}
    cb, pb = np.concatenate(out), np.concatenate(row_bias)
    return {"control_bias": float(cb.mean()), "partner_bias": float(pb.mean()),
            "adjacency_part": float((pb - cb).mean()), "n": int(len(cb))}


def rollout_profile(R):
    ev, me = R["ev"], R["me"]
    hdv = me[:, 4] == 0
    prof, pooled = {}, []
    for o in OFFSETS:
        oi = list(A.OF).index(int(round(o * 2)))
        ok = hdv & (ev[:, oi, 3] == 1) & np.isfinite(ev[:, oi, 2])
        prof[o] = A.boot_mean(ev[ok, oi, 2], me[ok, 0], 200)
    for o in A.M1_T:
        oi = list(A.OF).index(int(round(o * 2)))
        ok = hdv & (ev[:, oi, 3] == 1) & np.isfinite(ev[:, oi, 2])
        pooled.append((ev[ok, oi, 2], me[ok, 0]))
    vals = np.concatenate([p[0] for p in pooled]) if pooled else np.zeros(0)
    grp = np.concatenate([p[1] for p in pooled]) if pooled else np.zeros(0)
    return prof, A.boot_mean(vals, grp, 300)


def behaviour(R, safety):
    t = R["t"]
    key = t["ep"] * 100000 + t["t"]
    _, first = np.unique(key, return_index=True)
    nzone = t["nzone"][first]
    wait = {}
    for k, w in zip(key, t["waitR"]):
        if np.isfinite(w):
            wait[k] = max(wait.get(k, 0.0), w)
    ev, me = R["ev"], R["me"]
    hd = me[:, 4] == 0 if len(me) else np.zeros(0, bool)
    i0 = list(A.OF).index(0)
    post = ev[hd][:, i0:i0 + 13, 2] if len(ev) else np.zeros((0, 13))
    okp = ev[hd][:, i0:i0 + 13, 3] == 1 if len(ev) else np.zeros((0, 13), bool)
    mins = np.nanmin(np.where(okp, post, np.nan), axis=1) if len(post) else np.zeros(0)
    return {"merges_per_run": float(len(ev)) / max(R["n_runs"], 1),
            "stuck_mean_ramp_in_zone": float(np.mean(nzone)) if len(nzone) else float("nan"),
            "stuck_share_wait_gt60": float(np.mean([w > 60.0 for w in wait.values()])) if wait else 0.0,
            "ttc_lt_1p5_share": safety.get("ttc<1.5", float("nan")),
            "min_gap_ins_p5": safety.get("min_gap_ins_p5", float("nan")),
            "post_ins_hard_brake_share": float(np.mean(mins[np.isfinite(mins)] < -3.0)) if np.isfinite(mins).any()
            else float("nan")}


#: behaviour metric -> +1 if larger is worse, -1 if smaller is worse
WORSE_SIGN = {"merges_per_run": -1, "stuck_mean_ramp_in_zone": +1, "stuck_share_wait_gt60": +1,
              "ttc_lt_1p5_share": +1, "min_gap_ins_p5": -1, "post_ins_hard_brake_share": +1}


def decide(abs_rollout, abs_onestep, beh, m1_r11):
    deltas = {}
    for name, src in (("rollout", abs_rollout), ("one_step", abs_onestep)):
        if R11 in src and REF in src and np.isfinite(src[R11] - src[REF]):
            deltas[name] = float(src[R11] - src[REF])
    bias_trigger = any(np.isfinite(v) and v > TH_BIAS for v in deltas.values())
    worse = []
    if R11 in beh and REF in beh:
        for key, sign in WORSE_SIGN.items():
            a, b = beh[R11].get(key), beh[REF].get(key)
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) and sign * (a - b) > 0:
                worse.append(key)
    keep10b = bias_trigger and bool(worse)
    under = m1_r11 is not None and np.isfinite(m1_r11) and m1_r11 > TH_UNDER
    return {"abs_bias_delta_r11_minus_10b": deltas, "bias_trigger": bias_trigger, "worse_than_10b": worse,
            "verdict": "KEEP-ROUND-10B (reject round 11)" if keep10b else
            ("PASS" if deltas else "MISSING"),
            "under_yield_flag": bool(under), "m1_mean_m3_0_round11": m1_r11}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", required=True)
    args = ap.parse_args(argv)
    res = {"onestep": {}, "rollout": {}, "behaviour": {}, "control": {}}
    abs_os, abs_ro = {}, {}
    for arm in (REF, R11, HZ):
        v = A.load_onestep(args.shm, arm)
        if v is None or "tm" not in v[0]:
            continue
        prof, last3, sel = onestep_profile(v[0])
        res["onestep"][arm] = {"profile": prof, "mean_m3_0": last3}
        res["control"][arm] = matched_control_bias(v[0], sel)
        abs_os[arm] = last3[0]
    rolls = {arm: A.load_arm_rolls(args.shm, arm) for arm in ("sumo", REF, R11, HZ)}
    rolls = {k: v for k, v in rolls.items() if v is not None}
    safety = A.rollout_safety(rolls) if rolls else {}
    sumo = rollout_profile(rolls["sumo"]) if "sumo" in rolls else None
    for arm in (REF, R11, HZ):
        if arm not in rolls:
            continue
        prof, pooled = rollout_profile(rolls[arm])
        if sumo is not None:
            prof_b = {o: (prof[o][0] - sumo[0][o][0], prof[o][3]) for o in OFFSETS}
            abs_ro[arm] = pooled[0] - sumo[1][0]
        else:
            prof_b = {}
        res["rollout"][arm] = {"accel_profile": prof, "abs_bias_profile": prof_b, "mean_m3_0": pooled,
                               "abs_bias_m3_0": abs_ro.get(arm, float("nan"))}
        res["behaviour"][arm] = behaviour(rolls[arm], {k: v for k, v in safety.get(arm, {}).items()
                                                       if not k.startswith("_")})
    if sumo is not None:
        res["rollout"]["sumo"] = {"accel_profile": sumo[0], "mean_m3_0": sumo[1]}
        res["behaviour"]["sumo"] = behaviour(rolls["sumo"], {k: v for k, v in safety.get("sumo", {}).items()
                                                             if not k.startswith("_")})
    m1_r11 = None
    try:
        m1 = A.m1(rolls, A.cells_fn())
        m1_r11 = m1.get(R11, {}).get("mean_m3_0", (float("nan"),))[0]
        res["m1_mean_m3_0"] = {arm: m1[arm]["mean_m3_0"] for arm in m1}
    except Exception as exc:  # the M1 read is also in analyze.py's verdict
        res["m1_error"] = str(exc)
    res["decision"] = decide(abs_ro, abs_os, res["behaviour"], m1_r11)
    with open(args.json, "w", encoding="utf-8") as stream:
        json.dump(res, stream, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    L = ["# R11 M8: absolute target-follower bias and the keep-10b rule (R11_REVIEW C3)\n",
         f"Arms: {', '.join(f'{k} = {v}' for k, v in LABEL.items())}.\n",
         f"**Decision: {res['decision']['verdict']}**; bias deltas (round 11 - round 10b, [-3, 0) s): "
         f"{json.dumps({k: round(v, 3) for k, v in res['decision']['abs_bias_delta_r11_minus_10b'].items()})}; "
         f"worse than 10b: {res['decision']['worse_than_10b'] or 'none'}; under-yield flag: "
         f"{res['decision']['under_yield_flag']} (M1 mean {m1_r11}).\n",
         "\n## One-step absolute bias (model - recorded), target follower, by event time (s)\n",
         "| arm | " + " | ".join(str(o) for o in OFFSETS) + " | mean [-3,0) | matched-control bias (last 3 s) |",
         "|---" * (len(OFFSETS) + 3) + "|"]
    for arm, r in res["onestep"].items():
        c = res["control"].get(arm, {})
        L.append(f"| {LABEL[arm]} | " + " | ".join(A.fmt(*r["profile"][o][:1]) for o in OFFSETS)
                 + f" | {A.fmt(*r['mean_m3_0'])} | {A.fmt(c.get('control_bias', float('nan')))} |")
    L += ["\n## Rollout absolute bias (model - recorded mean of the merge follower), by event time (s)\n",
          "| arm | " + " | ".join(str(o) for o in OFFSETS) + " | mean [-3,0) |", "|---" * (len(OFFSETS) + 2) + "|"]
    for arm, r in res["rollout"].items():
        if arm == "sumo":
            continue
        L.append(f"| {LABEL[arm]} | " + " | ".join(A.fmt(r["abs_bias_profile"].get(o, (float('nan'),))[0])
                                                  for o in OFFSETS) + f" | {A.fmt(r['abs_bias_m3_0'])} |")
    L += ["\n## Behaviour (rollouts)\n", "| arm | " + " | ".join(WORSE_SIGN) + " |", "|---" * (len(WORSE_SIGN) + 1) + "|"]
    for arm, b in res["behaviour"].items():
        L.append(f"| {LABEL[arm]} | " + " | ".join(f"{b[k]:.4g}" if np.isfinite(b[k]) else "n/a" for k in WORSE_SIGN) + " |")
    with open(args.out, "w", encoding="utf-8") as stream:
        stream.write("\n".join(L) + "\n")
    print(f"M8: {res['decision']['verdict']} under-yield={res['decision']['under_yield_flag']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
