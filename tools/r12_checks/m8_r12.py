"""R12 M8 (+ seed-replicated M5 read): absolute target-follower bias and behaviour, round 12 vs ROUND 9.

Reads the reused tools/r10b_checks outputs of tools/r12_checks/run_all.sh: ``<shm>/os/onestep_<arm>.npz`` (one-step,
filled replay presence for every arm), ``<shm>/roll/<arm>/s<state>_k<k>.npz`` (seed set 1) and optionally
``<shm2>/roll/<arm>/...`` (seed set 2, run_rollouts.py --seed-offset; SEEDS2=1).  Label mapping: ``r9`` = ROUND 9
(reference), ``r10b`` = ROUND 12, ``headzero`` = round 12 head-zero, ``sumo`` = recorded futures of the same states.
Reuses tools/r11_checks/absolute_bias.py (one-step profile, WORSE_SIGN) and tools/r10b_checks/analyze.py (loaders, M1).

Per seed set and POOLED (both sets; clusters = states, so both seed sets of one state resample together):
* rollout absolute bias of the merge follower over [-3, 0) s = arm mean accel - recorded (sumo) mean accel;
* behaviour: merges per run, stuck ramp cars (time-mean ramp cars in the merge zone; share of times with a ramp wait
  > 60 s), TTC < 1.5 s share, min bumper gap at insertion p5, post-insertion hard brakes (share of HDV merge followers
  with post-insertion min accel < -3) - each with a state-cluster bootstrap 95 % CI, plus the paired CI of
  round 12 - round 9.  NOTE: these are computed PER RUN (time-means within a run, then pooled), so they can differ
  slightly from absolute_bias.behaviour, which merges the k samples of one state at equal times;
* M5: TTC < 1.5 s, overlap (gap < 0) and hard-brake (a < -3) row shares, round 12 (and head-zero) minus round 9,
  paired state bootstrap CI (as analyze.paired_diff);
* lane-change READ: mainline 0<->1 lane changes per run (HDVs; all vehicles) from the full rollout lane arrays
  (run_rollouts.py --lane-changes, key lc01), for round 12 / head-zero / round 9 and the recorded (sumo) reference.

REGISTERED RULES (R12; evaluated on the POOLED point estimates, literal; per-seed-set verdicts are reported and a
disagreement between seed sets is flagged as seed-inconsistent):
  M8  REJECT ROUND 12 if its mean ABSOLUTE target-follower bias over [-3, 0) s is NOT below round 9's in magnitude
      (|r12| >= |r9|; rollout OR one-step) AND any of merges per run, stuck ramp cars (either metric), TTC < 1.5 s
      share, min gap at insertion p5, post-insertion hard brakes is worse than round 9's.
  M8-under (read)  flag under-yield if round 12's rollout M1 mean over [-3, 0) s (C3-own matched excess) > -0.1.
  M5  REJECT ROUND 12 if, for the TTC < 1.5 s share or the overlap (gap < 0) share, the POOLED two-seed point estimate
      of (round 12 - round 9) is > 0 (worse) OR its paired 95 % state-bootstrap CI lies above 0.  Per-seed-set verdicts
      use the same rule and are reported.  (analyze.py's single-seed M5 proxy on seed set 1 stays in the reused M1-M7
      verdict.)
  LC  (read) mainline lane changes per run; no rule.
Usage: m8_r12.py --shm DIR [--shm2 DIR] --out MD --json JSON [--boot N]   (always exits 0)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for extra in (HERE, os.path.join(HERE, "..", "r11_checks"), os.path.join(HERE, "..", "r10b_checks")):
    if extra not in sys.path:
        sys.path.insert(0, extra)
if os.path.join(HERE, "..") not in sys.path:      # tools/ (diag_yield_common for the M1 read) as a fallback
    sys.path.append(os.path.join(HERE, ".."))

import absolute_bias as AB  # noqa: E402  (tools/r11_checks, reused unchanged)
import analyze as A  # noqa: E402  (tools/r10b_checks, reused unchanged)

REF, R12, HZ = "r9", "r10b", "headzero"
ARMS = (REF, R12, HZ)
LABEL = {REF: "round 9", R12: "round 12", HZ: "round 12 head-zero", "sumo": "recorded"}
WORSE_SIGN = dict(AB.WORSE_SIGN)
TH_UNDER = -0.1
I0 = list(A.OF).index(0)
M1_IDX = [list(A.OF).index(int(round(o * 2))) for o in A.M1_T]
ADD_KEYS = ("runs", "n_ev", "zone_sum", "zone_cnt", "wait_gt60", "wait_cnt", "nrows", "ttc", "ovl", "hb",
            "post_hb", "post_n", "m1_sum", "m1_cnt", "lc_hdv", "lc_all", "lc_runs")


# ------------------------------------------------------------------------------------------------ per-run statistics
def run_stats(rows, ev, me, lc01=None) -> dict:
    """Additive statistics of one rollout run (rows: lane-1 HDV rows in analyze.C layout; ev/me: merge events;
    lc01: [HDV 0<->1 lane changes, all-vehicle changes, HDV vehicle-steps] or None if the run file has no lc01)."""
    rows = np.asarray(rows, np.float64).reshape(-1, len(A.C))
    t = {c: rows[:, i] for i, c in enumerate(A.C)}
    st = {k: 0.0 for k in ADD_KEYS}
    st["runs"] = 1.0
    st["n_ev"] = float(len(ev))
    if lc01 is not None:
        st["lc_hdv"], st["lc_all"], st["lc_runs"] = float(lc01[0]), float(lc01[1]), 1.0
    if len(rows):
        tt, first = np.unique(t["t"], return_index=True)
        st["zone_sum"], st["zone_cnt"] = float(np.sum(t["nzone"][first])), float(len(first))
        wait = {}
        for k, w in zip(t["t"], t["waitR"]):
            if np.isfinite(w):
                wait[k] = max(wait.get(k, 0.0), w)
        st["wait_gt60"], st["wait_cnt"] = float(sum(w > 60.0 for w in wait.values())), float(len(wait))
        hdv1 = (t["cav"] == 0) & (t["lane"] == 1)
        has = hdv1 & (t["hasL"] == 1)
        clos = -t["dvL"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ttc = np.where(has & (clos > 1e-3), t["gapL"] / np.maximum(clos, 1e-3), np.inf)
        st["nrows"] = float(hdv1.sum())
        st["ttc"] = float((ttc < 1.5)[hdv1].sum())
        st["ovl"] = float((has & (t["gapL"] < 0)).sum())
        st["hb"] = float((hdv1 & (t["a"] < -3.0)).sum())
    gaps = np.zeros(0)
    if len(ev):
        hd = me[:, 4] == 0
        g = ev[hd, I0, 11]
        gaps = g[np.isfinite(g)]
        post = ev[hd][:, I0:I0 + 13, 2]
        ok = ev[hd][:, I0:I0 + 13, 3] == 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mins = np.nanmin(np.where(ok, post, np.nan), axis=1) if len(post) else np.zeros(0)
        fin = np.isfinite(mins)
        st["post_hb"], st["post_n"] = float((mins[fin] < -3.0).sum()), float(fin.sum())
        for oi in M1_IDX:
            okm = hd & (ev[:, oi, 3] == 1) & np.isfinite(ev[:, oi, 2])
            st["m1_sum"] += float(ev[okm, oi, 2].sum())
            st["m1_cnt"] += float(okm.sum())
    st["gaps"] = gaps
    return st


def load_set(roll_dirs, arm) -> dict:
    """{state index: aggregated run statistics} over every run file of ``arm`` in the given roll dirs."""
    out: dict = {}
    for d in roll_dirs:
        for f in sorted(glob.glob(os.path.join(d, arm, "*.npz"))):
            sid = int(re.search(r"s(\d+)_k", os.path.basename(f)).group(1))
            z = np.load(f)
            st = run_stats(z["rows"], z["ev"], z["me"], z["lc01"] if "lc01" in z.files else None)
            if sid not in out:
                out[sid] = st
            else:
                acc = out[sid]
                for k in ADD_KEYS:
                    acc[k] += st[k]
                acc["gaps"] = np.concatenate([acc["gaps"], st["gaps"]])
    return out


def _ratio(a, b, empty=float("nan")):
    return float(a / b) if b > 0 else empty


def metrics(per_state: dict, states, weights=None, sumo: dict | None = None) -> dict:
    """Behaviour / safety metrics over ``states`` (with multiplicities ``weights``)."""
    w = np.ones(len(states)) if weights is None else np.asarray(weights, float)
    tot = {k: float(sum(wi * per_state[s][k] for s, wi in zip(states, w) if s in per_state)) for k in ADD_KEYS}
    gaps = [np.repeat(per_state[s]["gaps"], int(wi)) for s, wi in zip(states, w) if s in per_state and wi > 0]
    gaps = np.concatenate(gaps) if gaps else np.zeros(0)
    m = {"merges_per_run": _ratio(tot["n_ev"], tot["runs"]),
         "stuck_mean_ramp_in_zone": _ratio(tot["zone_sum"], tot["zone_cnt"]),
         "stuck_share_wait_gt60": _ratio(tot["wait_gt60"], tot["wait_cnt"], 0.0),
         "ttc_lt_1p5_share": _ratio(tot["ttc"], tot["nrows"]),
         "min_gap_ins_p5": float(np.percentile(gaps, 5)) if len(gaps) else float("nan"),
         "post_ins_hard_brake_share": _ratio(tot["post_hb"], tot["post_n"]),
         "overlap_share": _ratio(tot["ovl"], tot["nrows"]),
         "hard_brake_share": _ratio(tot["hb"], tot["nrows"]),
         "m1_raw_mean": _ratio(tot["m1_sum"], tot["m1_cnt"]),
         "lane_changes_01_hdv_per_run": _ratio(tot["lc_hdv"], tot["lc_runs"]),
         "lane_changes_01_all_per_run": _ratio(tot["lc_all"], tot["lc_runs"])}
    if sumo is not None:
        ss = float(sum(wi * sumo[s]["m1_sum"] for s, wi in zip(states, w) if s in sumo))
        sc = float(sum(wi * sumo[s]["m1_cnt"] for s, wi in zip(states, w) if s in sumo))
        m["abs_bias_rollout_m3_0"] = m["m1_raw_mean"] - _ratio(ss, sc)
    return m


LC_KEYS = ("lane_changes_01_hdv_per_run", "lane_changes_01_all_per_run")
BEH_KEYS = tuple(WORSE_SIGN) + ("abs_bias_rollout_m3_0", "overlap_share", "hard_brake_share") + LC_KEYS


def bootstrap(sets: dict, sumo: dict | None, n_boot: int, seed: int = 0) -> dict:
    """Point estimate + state-cluster CI per arm, and the paired CI of every arm minus round 9."""
    rng = np.random.default_rng(seed)
    states = sorted(set().union(*[set(v) for v in sets.values()])) if sets else []
    if sumo is not None:
        states = [s for s in states if s in sumo] or states
    out = {arm: {"point": metrics(ps, states, sumo=sumo), "boot": {k: [] for k in BEH_KEYS}} for arm, ps in sets.items()}
    diffs = {arm: {k: [] for k in BEH_KEYS} for arm in sets if arm != REF and REF in sets}
    for _ in range(n_boot if states else 0):
        pick = rng.integers(0, len(states), len(states))
        cnt = np.bincount(pick, minlength=len(states))
        sel = [states[i] for i in np.flatnonzero(cnt)]
        wts = cnt[cnt > 0]
        vals = {arm: metrics(ps, sel, wts, sumo) for arm, ps in sets.items()}
        for arm in sets:
            for k in BEH_KEYS:
                out[arm]["boot"][k].append(vals[arm].get(k, float("nan")))
        for arm in diffs:
            for k in BEH_KEYS:
                diffs[arm][k].append(vals[arm].get(k, float("nan")) - vals[REF].get(k, float("nan")))
    res = {"n_states": len(states)}
    for arm, r in out.items():
        res[arm] = {k: _ci(r["point"].get(k, float("nan")), r["boot"][k]) for k in BEH_KEYS}
    for arm, dd in diffs.items():
        res[f"{arm}-{REF}"] = {k: _ci(out[arm]["point"].get(k, float("nan")) - out[REF]["point"].get(k, float("nan")), v)
                               for k, v in dd.items()}
    return res


def _ci(point, samples):
    x = np.asarray(samples, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return [float(point), float("nan"), float("nan")]
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(point), float(lo), float(hi)]


# ------------------------------------------------------------------------------------------------ rules
def decide(abs_ro: dict, abs_os: dict, beh: dict, m1_r12) -> dict:
    """M8 rule on point estimates: {arm: signed bias} for rollout / one-step, {arm: {metric: value}} behaviour."""
    trig = {}
    for name, src in (("rollout", abs_ro), ("one_step", abs_os)):
        a, b = src.get(R12), src.get(REF)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            trig[name] = bool(not abs(a) < abs(b))
    bias_trigger = any(trig.values())
    worse = []
    if R12 in beh and REF in beh:
        for key, sign in WORSE_SIGN.items():
            a, b = beh[R12].get(key), beh[REF].get(key)
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) and sign * (a - b) > 0:
                worse.append(key)
    reject = bias_trigger and bool(worse)
    under = m1_r12 is not None and np.isfinite(m1_r12) and m1_r12 > TH_UNDER
    return {"bias_not_below_round9": trig, "bias_trigger": bias_trigger, "worse_than_round9": worse,
            "verdict": "REJECT ROUND 12" if reject else ("PASS" if trig else "MISSING"),
            "under_yield_flag": bool(under), "m1_mean_m3_0_round12": m1_r12}


def m5_decide(boot: dict) -> dict:
    """M5: worse if the (round 12 - round 9) point estimate is > 0 OR its paired CI lies above 0."""
    d = boot.get(f"{R12}-{REF}", {})
    point = [k for k in ("ttc_lt_1p5_share", "overlap_share") if k in d and np.isfinite(d[k][0]) and d[k][0] > 0]
    ci = [k for k in ("ttc_lt_1p5_share", "overlap_share") if k in d and np.isfinite(d[k][1]) and d[k][1] > 0]
    bad = sorted(set(point) | set(ci))
    return {"verdict": "REJECT ROUND 12" if bad else ("PASS" if d else "MISSING"), "worse": bad,
            "worse_point_gt_0": point, "worse_ci_above_0": ci}


def combine(verdicts: dict) -> str:
    pooled = verdicts.get("pooled", "MISSING")
    others = {k: v for k, v in verdicts.items() if k != "pooled"}
    if len(others) > 1 and len(set(others.values())) > 1:
        return f"{pooled} (seed-inconsistent: " + ", ".join(f"{k} {v}" for k, v in others.items()) + ")"
    return pooled


# ------------------------------------------------------------------------------------------------ main
def merged_rolls(roots, arm):
    parts = [A.load_arm_rolls(r, arm) for r in roots]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    cat = lambda xs: np.concatenate([x for x in xs if len(x)]) if any(len(x) for x in xs) else xs[0]  # noqa: E731
    return {"t": {c: np.concatenate([p["t"][c] for p in parts]) for c in A.C}, "ev": cat([p["ev"] for p in parts]),
            "me": cat([p["me"] for p in parts]), "head": cat([p["head"] for p in parts]),
            "jerk_sw": cat([p["jerk_sw"] for p in parts]), "jerk_all": sum((p["jerk_all"] for p in parts), []),
            "misc": np.concatenate([p["misc"] for p in parts]), "n_runs": sum(p["n_runs"] for p in parts)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shm", required=True)
    ap.add_argument("--shm2", default=None, help="seed set 2 root (contains roll/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args(argv)
    res = {"labels": LABEL, "onestep": {}, "sets": {}, "decision": {}, "m5": {}}
    abs_os = {}
    for arm in ARMS:
        v = A.load_onestep(args.shm, arm)
        if v is None or "tm" not in v[0]:
            continue
        _, last3, _ = AB.onestep_profile(v[0])
        res["onestep"][arm] = {"mean_m3_0": list(last3)}
        abs_os[arm] = last3[0]
    roots = {"seed set 1": [os.path.join(args.shm, "roll")]}
    if args.shm2 and os.path.isdir(os.path.join(args.shm2, "roll")):
        roots["seed set 2"] = [os.path.join(args.shm2, "roll")]
        roots["pooled"] = roots["seed set 1"] + roots["seed set 2"]
    else:
        roots["pooled"] = roots["seed set 1"]
    m1_by = {}
    for name, dirs in roots.items():
        sets = {arm: load_set(dirs, arm) for arm in ARMS}
        sets = {k: v for k, v in sets.items() if v}
        sumo = load_set(dirs, "sumo") or None
        boot = bootstrap(sets, sumo, args.boot)
        beh = {arm: {k: boot[arm][k][0] for k in BEH_KEYS} for arm in sets}
        abs_ro = {arm: boot[arm]["abs_bias_rollout_m3_0"][0] for arm in sets if sumo is not None}
        m1_r12 = None
        try:
            rolls = {arm: merged_rolls([os.path.dirname(d) for d in dirs], arm) for arm in ("sumo",) + ARMS}
            rolls = {k: v for k, v in rolls.items() if v is not None}
            m1 = A.m1(rolls, A.cells_fn())
            m1_r12 = m1.get(R12, {}).get("mean_m3_0", (float("nan"),))[0]
            m1_by[name] = {arm: list(m1[arm]["mean_m3_0"]) for arm in m1}
        except Exception as exc:   # the M1 read needs tools/diag_yield_common on the path
            m1_by[name] = {"error": repr(exc)}
        res["sets"][name] = {"bootstrap": boot, "n_runs": {arm: int(sum(s["runs"] for s in sets[arm].values()))
                                                          for arm in sets}}
        res["decision"][name] = decide(abs_ro, abs_os, beh, m1_r12)
        if sumo is not None:   # lane-change read of the recorded reference (one run per state)
            res["sets"][name]["sumo_lane_changes"] = {
                k: v for k, v in bootstrap({"sumo": sumo}, None, args.boot).get("sumo", {}).items() if k in LC_KEYS}
        res["m5"][name] = m5_decide(boot)
    res["m1_reads"] = m1_by
    res["M8_verdict"] = combine({k: v["verdict"] for k, v in res["decision"].items()})
    res["M5_verdict"] = combine({k: v["verdict"] for k, v in res["m5"].items()})
    res["under_yield_flag_pooled"] = res["decision"]["pooled"]["under_yield_flag"]
    with open(args.json, "w", encoding="utf-8") as stream:
        json.dump(res, stream, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    write_md(args.out, res)
    print(f"M8 R12: {res['M8_verdict']}; M5 (seed-replicated): {res['M5_verdict']}; "
          f"under-yield={res['under_yield_flag_pooled']}")
    return 0


def _f(c, p=3):
    m, lo, hi = c
    if not np.isfinite(m):
        return "n/a"
    return f"{m:+.{p}f}" + (f" [{lo:+.{p}f}, {hi:+.{p}f}]" if np.isfinite(lo) else "")


def write_md(path, res):
    L = ["# R12 M8: absolute target-follower bias, behaviour and seed-replicated M5 (round 12 vs round 9)\n",
         f"Arms: {', '.join(f'{k} = {v}' for k, v in LABEL.items())}. Seed sets: {', '.join(res['sets'])}. "
         "CIs: state-cluster bootstrap 95 %.\n",
         f"**M8 decision (pooled): {res['M8_verdict']}**; **M5 (seed-replicated, pooled CI): {res['M5_verdict']}**; "
         f"under-yield flag (pooled, read): {res['under_yield_flag_pooled']}.\n",
         "Rules: M8 rejects round 12 if |abs bias [-3,0) s| is not below round 9's (rollout or one-step) AND any behaviour "
         "metric is worse than round 9's (point estimates, pooled seed sets); M5 rejects round 12 if the paired CI of the "
         "TTC < 1.5 s or overlap share difference (round 12 - round 9) lies above 0.\n",
         "## One-step absolute bias, target follower, mean over [-3, 0) s (filled replay presence)\n",
         "| arm | mean [95 % CI] (n) |", "|---|---|"]
    for arm, r in res["onestep"].items():
        L.append(f"| {LABEL[arm]} | {A.fmt(*r['mean_m3_0'])} |")
    for name, s in res["sets"].items():
        b = s["bootstrap"]
        d = res["decision"][name]
        L += [f"\n## {name} ({b.get('n_states', 0)} states; runs {s['n_runs']})\n",
              f"M8 {d['verdict']} (bias not below round 9: {d['bias_not_below_round9']}; worse: "
              f"{d['worse_than_round9'] or 'none'}; M1 round 12 {d['m1_mean_m3_0_round12']}); M5 {res['m5'][name]['verdict']} "
              f"{res['m5'][name]['worse'] or ''}\n",
              "| metric | " + " | ".join(LABEL[a] for a in ARMS if a in b) + " | round 12 - round 9 | head-zero - round 9 |",
              "|---|" + "---|" * (sum(a in b for a in ARMS) + 2)]
        for k in BEH_KEYS:
            cells = [_f(b[a][k]) for a in ARMS if a in b]
            cells.append(_f(b[f"{R12}-{REF}"][k]) if f"{R12}-{REF}" in b else "n/a")
            cells.append(_f(b[f"{HZ}-{REF}"][k]) if f"{HZ}-{REF}" in b else "n/a")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["\n## Lane-change READ: mainline 0<->1 lane changes per run (pooled seed sets; state-bootstrap 95 % CI)\n",
          "| arm | HDV changes / run | all-vehicle changes / run |", "|---|---|---|"]
    pooled = res["sets"].get("pooled", {})
    pb = pooled.get("bootstrap", {})
    for a in ARMS:
        if a in pb:
            L.append(f"| {LABEL[a]} | {_f(pb[a][LC_KEYS[0]], 2)} | {_f(pb[a][LC_KEYS[1]], 2)} |")
    sl = pooled.get("sumo_lane_changes")
    if sl:
        L.append(f"| recorded | {_f(sl[LC_KEYS[0]], 2)} | {_f(sl[LC_KEYS[1]], 2)} |")
    L.append("\nM1 reads (C3-own matched excess, mean [-3, 0) s): " + json.dumps(
        {k: ({a: round(v[0], 3) for a, v in r.items()} if "error" not in r else r) for k, r in res["m1_reads"].items()},
        default=str))
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(L) + "\n")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # exit 0 always; the verdict file says what failed
        print(f"M8 R12: ERROR {exc!r}")
        if "--out" in sys.argv[:-1]:
            with open(sys.argv[sys.argv.index("--out") + 1], "w", encoding="utf-8") as stream:
                stream.write(f"# R12 M8\n\n**M8 decision: ERROR** ({exc!r}); rules not evaluated.\n")
        raise SystemExit(0)
