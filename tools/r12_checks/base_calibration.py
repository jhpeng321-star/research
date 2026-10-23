"""R12 BASE checks: one-step base calibration of round 12 vs round 9 (BASE_FOLLOWER_DIAG.md sections 3, 7, 8).

Reads ``<os>/onestep_<arm>.npz`` written by tools/r10b_checks/run_onestep.py with ``--replay-present filled --base-cols``
(every arm replayed on the SAME filled-presence replay state).  Label mapping of tools/r12_checks/run_all.sh:
``r9`` = ROUND 9 (reference), ``r10b`` = ROUND 12, ``headzero`` = round 12 with follower_courtesy.net.4.* zeroed.
numpy only (analyze.boot_mean for the window-cluster bootstrap CI).

Row classes (lane-1 HDV transition rows; bias = model - recorded next-row acceleration, m/s^2; "gap" = own-lane leader
bumper gap in the replay state = column gapF when present, else gapL; no leader = non-finite gap):
  slow_start            unpartnered (has == 0), v < 2 m/s, 6 <= gap <= 30 m
  far_upstream_slow     slow_start and s < -400 m
  free_road             5 < v < 20 m/s and (gap > 60 m or no leader)
  last3_target_follower partner rows with 0 < t_merge <= 3 s (column tm)
  far_partner           partner rows with dA > 30 m          (M3 negative control)
  d_end_gt40            partner rows with d_end > 40 m       (M3 negative control)
  alongside             partner rows with bumper gap < 1 m   (M3 negative control)
  recorded_braking      rec < -1 m/s^2
  after_merge_zone      s > 75.44 m
  all_lane1             every lane-1 row

REGISTERED BASE REJECT RULES (evaluation rules, not model rules; magnitudes = |mean bias|, point estimates, literal):
  BASE-1  REJECT round 12 if its slow-start |mean bias| > 0.3 m/s^2 OR it is not smaller than round 9's.
  BASE-2  REJECT round 12 if its free-road |mean bias| or its far-upstream slow-start |mean bias| exceeds
          max(0.3, round 9's |mean bias| of the same class + 0.1) m/s^2 ("near 0", and not worse than round 9).
  BASE-3  REJECT round 12 if its recorded-braking |mean bias| exceeds round 9's by > 0.1 m/s^2.
  BASE-4  (read) last-3-s target-follower absolute bias for round 12, round-12 head-zero and round 9; expectation
          (BASE_FOLLOWER_DIAG): about +0.8 with the head zeroed, within +-0.3 with the head.
  ENC     (read) encoded IDM accel / decel percentiles p1/p10/p50/p90/p99 per arm (columns dacc/ddec, the values the
          IDM core received); flag if > 20 % of rows lie within 1 % of the physical range width of a bound
          (a 0.3-4.0, b 0.8-6.0; meaningful for physical-mode arms; legacy arms are read only).
A class with fewer than 5 finite rows gives an UNDECIDED rule (never a silent PASS).
Usage: base_calibration.py --os DIR --out MD --json JSON [--boot N]   (always exits 0)
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

REF, R12, HZ = "r9", "r10b", "headzero"
ARMS = (REF, R12, HZ)
LABEL = {REF: "round 9", R12: "round 12", HZ: "round 12 head-zero"}
END_M = 75.44
TH = {"BASE1": 0.3, "BASE2": 0.3, "BASE2_REF_MARGIN": 0.1, "BASE3": 0.1, "BASE4": 0.3, "ENC_SHARE": 0.20, "ENC_TOL": 0.01}
BOUNDS = {"accel": (0.3, 4.0), "decel": (0.8, 6.0)}
CLASSES = ("slow_start", "far_upstream_slow", "free_road", "last3_target_follower", "far_partner", "d_end_gt40",
           "alongside", "recorded_braking", "after_merge_zone", "all_lane1")


def load_rows(os_dir: str, arm: str):
    f = os.path.join(os_dir, f"onestep_{arm}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    cols = [str(c) for c in z["cols"]]
    r = z["rows"]
    return {c: r[:, i].astype(np.float64) for i, c in enumerate(cols)}


def leader_gap(d):
    return d["gapF"] if "gapF" in d else d["gapL"]


def row_classes(d) -> dict:
    """Boolean row masks of the registered classes (lane-1 rows only)."""
    lane1 = d["lane"] == 1
    has = d["has"] == 1
    gap = leader_gap(d)
    no_leader = ~np.isfinite(gap)
    v, s = d["v"], d["s"]
    with np.errstate(invalid="ignore"):
        slow = lane1 & (d["has"] == 0) & (v < 2.0) & (gap >= 6.0) & (gap <= 30.0)
        tm = d.get("tm", np.full(len(v), np.nan))
        return {
            "slow_start": slow,
            "far_upstream_slow": slow & (s < -400.0),
            "free_road": lane1 & (v > 5.0) & (v < 20.0) & ((gap > 60.0) | no_leader),
            "last3_target_follower": lane1 & has & (tm > 0.0) & (tm <= 3.0),
            "far_partner": lane1 & has & (d["dA"] > 30.0),
            "d_end_gt40": lane1 & has & (d["dend"] > 40.0),
            "alongside": lane1 & has & (d["bgap"] < 1.0),
            "recorded_braking": lane1 & (d["rec"] < -1.0),
            "after_merge_zone": lane1 & (s > END_M),
            "all_lane1": lane1,
        }


def class_stats(d, boot: int = 300) -> dict:
    bias = d["fin"] - d["rec"]
    out = {}
    for name, sel in row_classes(d).items():
        sel = sel & np.isfinite(bias)
        m, lo, hi, n = A.boot_mean(bias[sel], d["win"][sel], boot)
        out[name] = {"bias": m, "lo": lo, "hi": hi, "n": int(sel.sum()),
                     "mae": float(np.mean(np.abs(bias[sel]))) if sel.sum() >= 5 else float("nan"),
                     "rec": float(np.mean(d["rec"][sel])) if sel.any() else float("nan"),
                     "model": float(np.mean(d["fin"][sel])) if sel.any() else float("nan")}
    return out


def encoded_params(d) -> dict:
    """Percentiles of the encoded accel / decel over lane-1 rows and the share within 1 % of a range bound."""
    out = {}
    lane1 = d["lane"] == 1
    for col, key in (("dacc", "accel"), ("ddec", "decel")):
        if col not in d:
            out[key] = {"available": False}
            continue
        x = d[col][lane1 & np.isfinite(d[col])]
        if len(x) == 0:
            out[key] = {"available": False}
            continue
        lo, hi = BOUNDS[key]
        tol = TH["ENC_TOL"] * (hi - lo)
        share = float(np.mean((np.abs(x - lo) <= tol) | (np.abs(x - hi) <= tol)))
        out[key] = {"available": True, "n": int(len(x)),
                    "pct": {f"p{p}": float(np.percentile(x, p)) for p in (1, 10, 50, 90, 99)},
                    "share_near_bound": share, "flag": bool(share > TH["ENC_SHARE"])}
    return out


def _mag(stats, arm, cls):
    v = stats.get(arm, {}).get(cls, {}).get("bias", float("nan"))
    return abs(v) if np.isfinite(v) else float("nan")


def decide(stats: dict) -> dict:
    """Apply BASE-1..3 (+ the BASE-4 read) to ``class_stats`` per arm.  Returns {rule: (verdict, detail)}."""
    res = {}
    s12, s9 = _mag(stats, R12, "slow_start"), _mag(stats, REF, "slow_start")
    if not np.isfinite(s12) or not np.isfinite(s9):
        res["BASE-1"] = ("UNDECIDED", f"slow-start |bias| round 12 {s12:.3f}, round 9 {s9:.3f} (missing / n < 5)")
    else:
        bad = s12 > TH["BASE1"] or not s12 < s9
        res["BASE-1"] = ("REJECT" if bad else "PASS",
                         f"slow-start |bias| round 12 {s12:.3f} vs threshold {TH['BASE1']} and round 9 {s9:.3f}")
    fr, fu = _mag(stats, R12, "free_road"), _mag(stats, R12, "far_upstream_slow")

    def limit2(cls):
        ref = _mag(stats, REF, cls)
        return max(TH["BASE2"], ref + TH["BASE2_REF_MARGIN"]) if np.isfinite(ref) else TH["BASE2"]

    lim = {"free-road": limit2("free_road"), "far-upstream slow-start": limit2("far_upstream_slow")}
    bad2 = [f"{nm} {v:.3f} > {lim[nm]:.3f}" for nm, v in (("free-road", fr), ("far-upstream slow-start", fu))
            if np.isfinite(v) and v > lim[nm]]
    if bad2:
        res["BASE-2"] = ("REJECT", "; ".join(bad2))
    elif not (np.isfinite(fr) and np.isfinite(fu)):
        res["BASE-2"] = ("UNDECIDED", f"free-road |bias| {fr:.3f}, far-upstream slow-start |bias| {fu:.3f} (missing / n < 5)")
    else:
        res["BASE-2"] = ("PASS", f"free-road |bias| {fr:.3f} <= {lim['free-road']:.3f}, far-upstream slow-start "
                                 f"|bias| {fu:.3f} <= {lim['far-upstream slow-start']:.3f}")
    b12, b9 = _mag(stats, R12, "recorded_braking"), _mag(stats, REF, "recorded_braking")
    if not np.isfinite(b12) or not np.isfinite(b9):
        res["BASE-3"] = ("UNDECIDED", f"braking |bias| round 12 {b12:.3f}, round 9 {b9:.3f} (missing / n < 5)")
    else:
        res["BASE-3"] = ("REJECT" if b12 - b9 > TH["BASE3"] else "PASS",
                         f"recorded-braking |bias| round 12 {b12:.3f} - round 9 {b9:.3f} = {b12 - b9:+.3f} (reject if > {TH['BASE3']})")
    reads = {arm: stats.get(arm, {}).get("last3_target_follower", {}).get("bias", float("nan")) for arm in ARMS}
    r = reads[R12]
    res["BASE-4"] = ("READ", "last-3-s target-follower bias: " + ", ".join(
        f"{LABEL[a]} {v:+.3f}" if np.isfinite(v) else f"{LABEL[a]} n/a" for a, v in reads.items())
        + (f"; round 12 within +-{TH['BASE4']}: {abs(r) <= TH['BASE4']}" if np.isfinite(r) else ""))
    return res


def overall(rules: dict) -> str:
    rej = [k for k in ("BASE-1", "BASE-2", "BASE-3") if rules.get(k, ("",))[0] == "REJECT"]
    und = [k for k in ("BASE-1", "BASE-2", "BASE-3") if rules.get(k, ("",))[0] == "UNDECIDED"]
    if rej:
        return f"REJECT ROUND 12 ({', '.join(rej)})"
    if und:
        return f"UNDECIDED ({', '.join(und)} lack rows)"
    return "PASS (BASE-1..3)"


def fmtb(c):
    if c is None or not np.isfinite(c.get("bias", float("nan"))):
        return f"n/a (n={c.get('n', 0) if c else 0})"
    return f"{A.fmt(c['bias'], c['lo'], c['hi'])} MAE {c['mae']:.2f} (n={c['n']})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--os", required=True, help="directory with onestep_<arm>.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--boot", type=int, default=300)
    args = ap.parse_args(argv)
    stats, enc, meta = {}, {}, {}
    for arm in ARMS:
        d = load_rows(args.os, arm)
        if d is None:
            continue
        stats[arm] = class_stats(d, args.boot)
        enc[arm] = encoded_params(d)
        gapF = d.get("gapF")
        if gapF is not None:
            with np.errstate(invalid="ignore"):
                hidden = np.isfinite(gapF) & (~np.isfinite(d["gapL"]) | (gapF < d["gapL"] - 0.5))
            meta[arm] = {"base_cols": True, "share_rows_leader_only_in_filled_presence": float(np.mean(hidden[d["lane"] == 1]))}
        else:
            meta[arm] = {"base_cols": False}
    done = os.path.join(args.os, "ONESTEP_DONE.json")
    replay = json.load(open(done)).get("replay_present", "loss_valid") if os.path.exists(done) else "unknown"
    rules = decide(stats)
    verdict = overall(rules)
    res = {"verdict": verdict, "rules": {k: list(v) for k, v in rules.items()}, "classes": stats, "encoded": enc,
           "meta": meta, "replay_present": replay, "thresholds": TH, "labels": LABEL}
    with open(args.json, "w", encoding="utf-8") as stream:
        json.dump(res, stream, indent=1, default=float)
    L = ["# R12 BASE checks: one-step base calibration (round 12 vs round 9)\n",
         f"Arms: {', '.join(f'{k} = {v}' for k, v in LABEL.items())}. Replay presence: **{replay}** (same replay state for "
         "every arm). Bias = model - recorded next-row acceleration (m/s^2), mean [95 % CI over windows], MAE.\n",
         f"**BASE verdict: {verdict}**\n",
         "| rule | result | detail |", "|---|---|---|"]
    L += [f"| {k} | **{v[0]}** | {v[1]} |" for k, v in rules.items()]
    L += ["\nRegistered rules (evaluation rules, not model rules; |.| = magnitude of the mean bias, point estimates):",
          f"BASE-1 reject if slow-start |bias| > {TH['BASE1']} or not smaller than round 9's; BASE-2 reject if free-road or "
          f"far-upstream slow-start |bias| > max({TH['BASE2']}, round 9's + {TH['BASE2_REF_MARGIN']}); BASE-3 reject if recorded-braking |bias| exceeds round 9's by "
          f"> {TH['BASE3']}; BASE-4 read (expectation: head-zero about +0.8, round 12 within +-{TH['BASE4']}).\n",
          "## Bias by row class\n", "| class | " + " | ".join(LABEL[a] for a in ARMS if a in stats) + " | recorded mean |",
          "|---|" + "---|" * (sum(a in stats for a in ARMS) + 1)]
    for cls in CLASSES:
        rec = next((stats[a][cls]["rec"] for a in ARMS if a in stats), float("nan"))
        L.append(f"| {cls} | " + " | ".join(fmtb(stats[a].get(cls)) for a in ARMS if a in stats) + f" | {rec:+.2f} |")
    L += ["\n## Encoded IDM accel / decel (values the core received; lane-1 rows)\n",
          "| arm | param | p1 | p10 | p50 | p90 | p99 | share within 1 % of a bound | flag (> 20 %) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for arm, e in enc.items():
        for key, r in e.items():
            if not r.get("available"):
                L.append(f"| {LABEL[arm]} | {key} | n/a | | | | | | |")
                continue
            p = r["pct"]
            L.append(f"| {LABEL[arm]} | {key} | {p['p1']:.2f} | {p['p10']:.2f} | {p['p50']:.2f} | {p['p90']:.2f} | "
                     f"{p['p99']:.2f} | {r['share_near_bound']:.3f} | {'FLAG' if r['flag'] else 'no'} |")
    L.append(f"\nPhysical bounds used for the flag: accel {BOUNDS['accel']}, decel {BOUNDS['decel']} (round 9 is a legacy-"
             "encoder arm; its share is a read only).")
    for arm, m in meta.items():
        if m.get("base_cols"):
            L.append(f"- {LABEL[arm]}: share of lane-1 rows whose replay leader exists only with the filled presence (or is "
                     f"nearer than the loss_valid leader): {m['share_rows_leader_only_in_filled_presence']:.3f}")
    with open(args.out, "w", encoding="utf-8") as stream:
        stream.write("\n".join(L) + "\n")
    print(f"BASE: {verdict}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # exit 0 always; the verdict file says what failed
        print(f"BASE: ERROR {exc!r}")
        if "--out" in sys.argv[:-1]:
            with open(sys.argv[sys.argv.index("--out") + 1], "w", encoding="utf-8") as stream:
                stream.write(f"# R12 BASE checks\n\n**BASE verdict: ERROR** ({exc!r}); rules not evaluated.\n")
        raise SystemExit(0)
