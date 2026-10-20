"""V8WM3 inference-only quick check (copy of runs/active/C4-V7FT-20261017/v8wm2_check/v8wm2_check.py
+ the V8WM3 merge_impatience switches: imp=1, imp_d, imp_v, imp_th, imp_tg, imp_s0, imp_T).

V8WM2 inference-only quick check (CPU, no SUMO, no training).

v5m (its own physics.json: merge_end_wait + mainline_merge_yield ON) rolled on every 2nd
wall_same h60 state (72), wall arm, same semantics as ramp_diag/ramp_ablate.py (CAVs
teacher-forced, entries off, recorded SUMO exits end slots, lane seed 0, 120 s), with the
V8WM2 PhysicsConfig fields set at inference per variant:
  fw=hazard          forced_window_mode
  fmin               forced_window_min_hazard
  scope=zone         mainline_merge_yield_scope
  mew_s0 / mew_T     merge_end_wait_s0_follow_m / merge_end_wait_t_follow_s

R5TOOLS addition (default unchanged): ``--k K`` repeats every state with fixed seeds 0..K-1
(lane_seed = seed) and the summary adds per-state blow-up probabilities, p90/p95 severity
tails, scored-roster/NaN counts and paired new/recovered failures vs the first variant.

MERGECAL / V9ENTRY additions (defaults unchanged):
  gm=<g>             inference-time merge-head gate (lane_v5_gates[2]; STUCK_DIAG "gate 0.4")
  --entries queue    entries ON seeded with the recorded SUMO entrance backlog and demand
                     (wms.eval.entry_queue; HDV entrants, sumo_like_v2 insertion); default off.
                     Entrants occupy slots after the t0 roster; all metrics stay on the t0 roster.

Run from the c4/v8wm2 worktree (the code under test):
  cd E:/agent-work/c4-v8wm2; PYTHONPATH="src;.;tools" python <this> --var "base=<ckpt>" ...
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np  # noqa: E402

sys.path.insert(0, "tools")
import wm_eval_common as wc  # noqa: E402

SET = "E:/agent-work/c4-v6merge/_ref/states_h60/wall_same.json"
DT = 0.5
G: dict = {}


def parse_var(spec):
    name, rest = spec.split("=", 1)
    path, _, opts = rest.partition("|")
    o = dict(kv.split("=") for kv in filter(None, opts.split(",")))
    return name, path, o


def init(var_specs, entries="off", wall_members=False, wall_paths=None):
    import torch
    torch.set_num_threads(1)
    G["p"] = {}
    G["wall_members"] = bool(wall_members)
    G["wall_paths"] = wall_paths or {}
    for name, path, o in var_specs:
        p, cap = wc.load_predictor(wc.guard(path), "AS_IS", hdv_exits="sumo",
                                   entries=entries, wall_members=wall_members)
        m = p.model
        ch = {}
        if "fw" in o:
            ch["forced_window_mode"] = o["fw"]
        if "fmin" in o:
            ch["forced_window_min_hazard"] = float(o["fmin"])
        if "scope" in o:
            ch["mainline_merge_yield_scope"] = o["scope"]
        if "mew_s0" in o:
            ch["merge_end_wait_s0_follow_m"] = float(o["mew_s0"])
        if "mew_T" in o:
            ch["merge_end_wait_t_follow_s"] = float(o["mew_T"])
        if "imp" in o:
            ch["merge_impatience"] = o["imp"] not in ("0", "false", "off")
        for key, field in (("imp_d", "merge_impatience_distance_m"),
                           ("imp_v", "merge_impatience_speed_mps"),
                           ("imp_th", "merge_impatience_hazard_tau_s"),
                           ("imp_tg", "merge_impatience_gap_tau_s"),
                           ("imp_s0", "merge_impatience_s0_min_m"),
                           ("imp_T", "merge_impatience_t_min_s")):
            if key in o:
                ch[field] = float(o[key])
        if ch:
            new = dataclasses.replace(m.physics, **ch)
            for owner, attr in wc._physics_holders(m):
                setattr(owner, attr, new)
        if "gm" in o:            # merge-head gate threshold (inference-time lane_v5_gates[2])
            g = tuple(m.lane_v5_gates)
            m.lane_v5_gates = (g[0], g[1], float(o["gm"]))
        if "zt" in o:            # V10BRKPROBE: zero the temporal residual acceleration
            m.diag_zero_temporal = o["zt"] not in ("0", "false", "off")
        if "rb" in o:            # V10BRKPROBE: hard bound on the combined residual
            m.diag_residual_bound_mps2 = float(o["rb"])
        if "tau" in o:           # V10BRKPROBE: causal low-pass time constant (s)
            m.diag_temporal_tau_s = float(o["tau"])
        G["p"][name] = (p, cap, o)


def first_off2(lane, msk, k):
    idx = np.flatnonzero(msk[:, k] & (lane[:, k] != 2))
    return float(idx[0] * DT) if idx.size else np.inf


def off2_dist(pos, lane, msk, k, end=75.44):
    """MERGECAL: distance to the lane end (m) at the last ramp-lane row before the first row off
    lane 2 (NaN when the car never leaves it or leaves at t0)."""
    idx = np.flatnonzero(msk[:, k] & (lane[:, k] != 2))
    return float(end - pos[idx[0] - 1, k]) if idx.size and idx[0] > 0 else np.nan


def work(payload):
    idx, rec, arm, sample = payload
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles
    if G.get("wall_members") and G.get("wall_paths"):
        entry = G["wall_paths"].get(str(rec.get("key", ""))) or G["wall_paths"].get(idx)
        if entry:
            rec = dict(rec)
            rec["wall_member_paths"] = entry.get("members", entry)
    out = []
    for name, (p, cap, o) in G["p"].items():
        try:
            n_steps = 240
            t0, ids, kin, is_cav = t0_vehicles(rec)
            teacher, present_seq, _ = recorded_teacher(rec, t0, ids, kin, n_steps, arm,
                                                       p.backend.tools)
            if teacher is None:
                out.append({"idx": idx, "var": name, "sample": sample, "error": "no arm"})
                continue
            if getattr(p, "flags", None) is not None:
                p.flags = dataclasses.replace(p.flags, lane_seed=int(sample))
            p.predict_states(rec, {**rec.get("cav_controllers", {}), "arm": arm},
                             n_steps * DT, int(sample))
            r = cap["last"]
            n = len(ids)
            tr = teacher[0].numpy()
            rp = present_seq[0].numpy().astype(bool)
            pos = np.concatenate([tr[:1, :, 0], r["positions"][0, :, :n].numpy()], 0)
            lan = np.concatenate([tr[:1, :, 1], r["lanes"][0, :, :n].numpy()], 0)
            msk = np.concatenate([rp[:1], r["masks"][0, :, :n].numpy().astype(bool)], 0)
            hdv = ~np.asarray(is_cav, bool)
            ramp = [k for k in range(n) if hdv[k] and int(tr[0, k, 1]) == 2]
            res = {"idx": idx, "var": name, "sample": sample, "error": None,
                   "t_wm": np.array([first_off2(lan, msk, k) for k in ramp]),
                   "t_rec": np.array([first_off2(tr[:, :, 1], rp, k) for k in ramp]),
                   "d_wm": np.array([off2_dist(pos, lan, msk, k) for k in ramp]),
                   "d_rec": np.array([off2_dist(tr[:, :, 0], tr[:, :, 1], rp, k) for k in ramp])}
            for ts in (30, 60, 120):
                i = int(ts / DT)
                both = msk[i] & rp[i] & hdv
                d = np.abs(pos[i] - tr[i, :, 0])
                res[f"ds{ts}"] = float(np.median(d[both])) if both.any() else np.nan
                rb = both.copy()
                rb[[k for k in range(n) if k not in ramp]] = False
                res[f"ramp_ds{ts}"] = float(np.median(d[rb])) if rb.any() else np.nan
                res[f"ramp_ds_all{ts}"] = d[rb]
            out.append(res)
        except Exception as exc:  # noqa: BLE001
            out.append({"idx": idx, "var": name, "sample": sample,
                        "error": f"{type(exc).__name__}: {exc}"})
    return out


def is_blowup(row) -> bool:
    """The official blow-up definition: median |ds| > 50 m at 60 s and > 100 m at 120 s."""
    return bool(row.get("ds60", np.nan) > 50 and row.get("ds120", np.nan) > 100)


def state_blow_probability(rows) -> float:
    """Fraction of successful samples of one state that blow up (NaN without samples)."""
    ok = [r for r in rows if not r.get("error")]
    return float(sum(is_blowup(r) for r in ok) / len(ok)) if ok else np.nan


def severity_tail(values, q: float):
    """Finite-sample quantile (``None`` when no finite value)."""
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return float(np.percentile(vals, q)) if vals.size else None


def tail_text(values, q: float) -> str:
    """Formatted severity quantile (``n/a`` without finite samples)."""
    value = severity_tail(values, q)
    return "n/a" if value is None else f"{value:.1f}"


def paired_failure_counts(base_probs, var_probs) -> tuple[int, int]:
    """(new, recovered) state-level failures vs the base variant (probability >= 0.5)."""
    new = recovered = 0
    for idx in set(base_probs) & set(var_probs):
        b, v = base_probs[idx], var_probs[idx]
        if not np.isfinite(b) or not np.isfinite(v):
            continue
        if v >= 0.5 and b < 0.5:
            new += 1
        elif b >= 0.5 and v < 0.5:
            recovered += 1
    return new, recovered


def summarize(path):
    d = pickle.load(open(path, "rb"))
    names = [s[0] for s in d["specs"]]
    by = {n: {} for n in names}
    for r in d["results"]:
        by[r["var"]].setdefault(r["idx"], []).append(r)
    k = int(d.get("k", 1))
    multi = any(len(rows) > 1 for rows in by[names[0]].values()) or k > 1
    base_rows = by[names[0]]
    lines = ["| variant | states | ramp HDVs | stuck in lane 2 (SUMO) | median dep. WM-SUMO s "
             "| median abs dep. err s | early>2 s / late>2 s | HDV ds60 | HDV ds120 "
             "| ramp ds60 (per-state med / pooled) | ramp ds120 (per-state / pooled) "
             "| blow-ups | paired ds60 / ds120 vs base |",
             "|---" * 13 + "|"]
    if not multi:
        # k=1: unchanged single-sample table.
        by1 = {n: {idx: rows[0] for idx, rows in by[n].items()} for n in names}
        base = by1[names[0]]
        for n in names:
            rr = by1[n]
            tw = np.concatenate([r["t_wm"] for r in rr.values()])
            ts = np.concatenate([r["t_rec"] for r in rr.values()])
            ok = np.isfinite(tw) & np.isfinite(ts)
            dt = tw[ok] - ts[ok]
            ds60 = np.array([r["ds60"] for r in rr.values()])
            ds120 = np.array([r["ds120"] for r in rr.values()])
            r60 = np.array([r["ramp_ds60"] for r in rr.values()])
            r120 = np.array([r["ramp_ds120"] for r in rr.values()])
            p60 = np.concatenate([r["ramp_ds_all60"] for r in rr.values()])
            p120 = np.concatenate([r["ramp_ds_all120"] for r in rr.values()])
            blow = sum(1 for r in rr.values() if r["ds60"] > 50 and r["ds120"] > 100)
            keys = sorted(set(rr) & set(base))
            q60 = np.nanmedian([rr[k]["ds60"] - base[k]["ds60"] for k in keys])
            q120 = np.nanmedian([rr[k]["ds120"] - base[k]["ds120"] for k in keys])
            lines.append(
                f"| {n} | {len(rr)} | {len(tw)} | {int(np.isinf(tw).sum())} "
                f"({int(np.isinf(ts).sum())})"
                f" | {np.median(dt):+.1f} | {np.median(np.abs(dt)):.1f} | {(dt < -2).sum()} / "
                f"{(dt > 2).sum()} | {np.nanmedian(ds60):.1f} | {np.nanmedian(ds120):.1f} | "
                f"{np.nanmedian(r60):.1f} / {np.median(p60) if p60.size else np.nan:.1f} | "
                f"{np.nanmedian(r120):.1f} / {np.median(p120) if p120.size else np.nan:.1f} | "
                f"{blow}/{len(rr)} | {q60:+.2f} / {q120:+.2f} |")
        return "\n".join(lines)
    # k>1: pool samples for severity, count states for blow-ups, add per-state/tail sections.
    probs = {n: {idx: state_blow_probability(rows) for idx, rows in by[n].items()}
             for n in names}
    base_probs = probs[names[0]]
    tails = []
    for n in names:
        states = len(by[n])
        ok_rows = [r for rows in by[n].values() for r in rows if not r.get("error")]
        tw = np.concatenate([r["t_wm"] for r in ok_rows]) if ok_rows else np.array([])
        ts = np.concatenate([r["t_rec"] for r in ok_rows]) if ok_rows else np.array([])
        ok = np.isfinite(tw) & np.isfinite(ts)
        dt = tw[ok] - ts[ok]
        ds60 = np.array([r["ds60"] for r in ok_rows])
        ds120 = np.array([r["ds120"] for r in ok_rows])
        r60 = np.array([r["ramp_ds60"] for r in ok_rows])
        r120 = np.array([r["ramp_ds120"] for r in ok_rows])
        p60 = (np.concatenate([r["ramp_ds_all60"] for r in ok_rows])
               if ok_rows else np.array([]))
        p120 = (np.concatenate([r["ramp_ds_all120"] for r in ok_rows])
                if ok_rows else np.array([]))
        blow = sum(1 for p in probs[n].values() if np.isfinite(p) and p >= 0.5)
        keys = sorted(set(probs[n]) & set(base_probs))
        prob_delta = np.nanmedian([probs[n][k] - base_probs[k] for k in keys])
        ds60_state = np.array([np.nanmedian([r["ds60"] for r in by[n][idx]
                                             if not r.get("error")]) for idx in keys])
        base60_state = np.array([np.nanmedian([r["ds60"] for r in base_rows[idx]
                                               if not r.get("error")]) for idx in keys])
        ds120_state = np.array([np.nanmedian([r["ds120"] for r in by[n][idx]
                                              if not r.get("error")]) for idx in keys])
        base120_state = np.array([np.nanmedian([r["ds120"] for r in base_rows[idx]
                                                if not r.get("error")]) for idx in keys])
        q60_m = np.nanmedian(ds60_state - base60_state) if keys else np.nan
        q120_m = np.nanmedian(ds120_state - base120_state) if keys else np.nan
        lines.append(
            f"| {n} | {states} | {len(tw)} | {int(np.isinf(tw).sum())} "
            f"({int(np.isinf(ts).sum())})"
            f" | {np.median(dt):+.1f} | {np.median(np.abs(dt)):.1f} | {(dt < -2).sum()} / "
            f"{(dt > 2).sum()} | {np.nanmedian(ds60):.1f} | {np.nanmedian(ds120):.1f} | "
            f"{np.nanmedian(r60):.1f} / {np.median(p60) if p60.size else np.nan:.1f} | "
            f"{np.nanmedian(r120):.1f} / {np.median(p120) if p120.size else np.nan:.1f} | "
            f"{blow}/{states} | {q60_m:+.2f} / {q120_m:+.2f} |")
        nan_samples = sum(1 for r in ok_rows
                          if not (np.isfinite(r.get("ds60", np.nan))
                                  and np.isfinite(r.get("ds120", np.nan))))
        unscorable = sorted(idx for idx, rows in by[n].items()
                            if not any(len(r["t_wm"]) for r in rows if not r.get("error")))
        scored = states - len(unscorable)
        new, recovered = paired_failure_counts(base_probs, probs[n])
        finite_probs = [p for p in probs[n].values() if np.isfinite(p)]
        ds_tail = ", ".join(
            f"ds{h} {tail_text([r.get(f'ds{h}', np.nan) for r in ok_rows], 90)}/"
            f"{tail_text([r.get(f'ds{h}', np.nan) for r in ok_rows], 95)}"
            for h in (30, 60, 120))
        ramp_tail = ", ".join(
            f"ramp_ds{h} {tail_text([r.get(f'ramp_ds{h}', np.nan) for r in ok_rows], 90)}/"
            f"{tail_text([r.get(f'ramp_ds{h}', np.nan) for r in ok_rows], 95)}"
            for h in (30, 60, 120))
        tails += [
            f"- **{n}** per-state blow-up probability: mean "
            f"{np.mean(finite_probs) if finite_probs else np.nan:.2f}, median "
            f"{np.median(finite_probs) if finite_probs else np.nan:.2f}, max "
            f"{max(finite_probs) if finite_probs else np.nan:.2f}.",
            f"  severity tails (p90/p95): {ds_tail}",
            f"  ramp tails (p90/p95): {ramp_tail}",
            f"  scored roster: {scored}/{states} states; unscorable {unscorable[:8]}"
            f"{'...' if len(unscorable) > 8 else ''}; NaN/unscorable samples {nan_samples}.",
            f"  paired vs {names[0]}: {new} new failures, {recovered} recovered "
            f"(state blow probability >= 0.5); median change {prob_delta:+.2f}.",
        ]
    return "\n".join(lines + ["", f"k={k} details:"] + tails)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", action="append", default=[])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--entries", choices=("off", "queue"), default="off",
                    help="off (frozen default) / queue = V9ENTRY entries ON with the recorded "
                         "SUMO entrance queue")
    ap.add_argument("--k", type=int, default=1,
                    help="samples per state with fixed seeds 0..k-1 (default 1 = unchanged)")
    ap.add_argument("--wall-members", choices=("off", "insert"), default="off",
                    help="off (frozen default) / insert = add wall_timeline members that enter "
                         "after t0 as teacher-forced CAV slots at their recorded first "
                         "appearance (needs --wall-member-paths)")
    ap.add_argument("--wall-member-paths", default="",
                    help="JSON sidecar from tools/wall_member_paths.py; only member paths that "
                         "the original state export did not keep are used")
    a = ap.parse_args()
    if a.k < 1:
        raise SystemExit("--k must be >= 1")
    if a.summarize:
        print(summarize(a.out))
        return
    specs = [parse_var(s) for s in a.var]
    from wms.eval.normalize import iter_windows
    recs = [(i, r) for i, r in enumerate(iter_windows(wc.guard(SET))) if i % a.every == 0]
    if a.limit:
        recs = recs[: a.limit]
    tasks = [(i, r, "wall", s) for i, r in recs for s in range(a.k)]
    wall_paths = {}
    if a.wall_members == "insert":
        if not a.wall_member_paths:
            raise SystemExit("--wall-members insert needs --wall-member-paths")
        with open(a.wall_member_paths, encoding="utf-8") as fh:
            wall_paths = json.load(fh).get("windows", {})
        print(f"wall-member paths: {len(wall_paths)} windows", flush=True)
    t0 = time.time()
    res = []
    if a.workers <= 1:
        init(specs, a.entries, a.wall_members == "insert", wall_paths)
        for t in tasks:
            res.extend(work(t))
    else:
        from multiprocessing import Pool
        with Pool(a.workers, initializer=init,
                  initargs=(specs, a.entries, a.wall_members == "insert",
                            wall_paths)) as pool:
            for j, rr in enumerate(pool.imap_unordered(work, tasks, chunksize=1)):
                res.extend(rr)
                if (j + 1) % 8 == 0:
                    print(f"{j+1}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    with open(a.out, "wb") as fh:
        pickle.dump({"specs": specs, "results": res, "entries": a.entries, "k": a.k,
                     "wall_members": a.wall_members}, fh)
    print(f"done {time.time()-t0:.0f}s errors {sum(1 for r in res if r.get('error'))}", flush=True)
    for r in [r for r in res if r.get("error")][:3]:
        print(r)
    print(summarize(a.out))


if __name__ == "__main__":
    main()
