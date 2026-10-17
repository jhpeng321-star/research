"""V8LANE lane-behaviour evaluation of a hazard_v5 WM with several lane seeds (inference only).

A stochastic lane model is scored distributionally, not by one sample's median |ds|:

* **Counts** per state (t0-roster HDVs, both-present steps): overtakes (lane 1 -> 0), returns
  (0 -> 1), ramp merges (2 -> 1) of the WM (mean over seeds) vs SUMO.
* **Lane shares**: lane-0 share among the matched HDVs on the mainline (SUMO lane 0/1) at 30 /
  60 / 120 s, WM (mean over seeds) vs SUMO, and the right-lane share (WM lane == SUMO lane).
* **Position error** at 60 / 120 s (per state the median over matched HDVs, then the median over
  states): ``single`` = seed row 0 only (what a single-sample scorer sees); ``avg`` = the mean
  over seeds of the single-sample per-state value (expected single-sample error); ``ens`` = the
  error of the seed-mean position (the ensemble point forecast).  Blow-ups use the scorer's AND
  rule (> 50 m at 60 s and > 100 m at 120 s) on ``single`` and on ``avg``.

The N lane seeds run as N batch rows of ONE rollout (physics deterministic; only the lane draws
differ; lane seed = row of the private generator).  Wall arm of a states file, CAVs
teacher-forced, entries off, recorded SUMO exits end slots (the semantics of
``lane_diag/lane_ablate.py`` and ``wm_diag/roll_diag.py``).

Variant spec: ``name=<ckpt>|key=value,...`` with keys ``mainline`` (default/sample/budget/
budget_prop), ``bo`` / ``br`` (overtake / return logit offsets), ``gates`` (g_o:g_r:g_m),
``seeds`` (lane seeds = batch rows; default ``--seeds``).

    PYTHONPATH="src;.;tools" python tools/lane_fix_eval.py --every 3 --workers 6 --seeds 4 \\
        --var "base=<ckpt>|seeds=1" --var "bud=<ckpt>|mainline=budget,bo=-0.29,br=-0.45" \\
        --out out/fix.pkl
    python tools/lane_fix_eval.py --summarise out/fix.pkl [--ref base]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DT = 0.5
TIMES = (30, 60, 120)
G: dict = {}


def parse_var(spec: str):
    name, rest = spec.split("=", 1)
    path, _, opts = rest.partition("|")
    o = {}
    for kv in filter(None, opts.split(",")):
        k, v = kv.split("=", 1)
        o[k] = v
    return name, path, o


def _expand(obj, b: int):
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.expand(b, *obj.shape[1:]).clone() if obj.ndim >= 1 and obj.shape[0] == 1 else obj
    if isinstance(obj, dict):
        return {k: _expand(v, b) for k, v in obj.items()}
    return obj


def apply_variant(model, o: dict) -> None:
    """Inference patches of one variant (model attributes; nothing is saved)."""
    if "mainline" in o:
        model.lane_v5_mainline = o["mainline"]
    if "gates" in o:
        model.lane_v5_gates = tuple(float(x) for x in o["gates"].split(":"))
    if "bo" in o or "br" in o:
        cur = tuple(getattr(model, "lane_logit_offsets", (0.0, 0.0, 0.0)))
        model.lane_logit_offsets = (float(o.get("bo", cur[0])), float(o.get("br", cur[1])),
                                    float(cur[2]))


def init(var_specs, default_seeds: int):
    import torch

    import wm_eval_common as wc
    torch.set_num_threads(1)
    G["p"] = {}
    for name, path, o in var_specs:
        p, cap = wc.load_predictor(wc.guard(path), "AS_IS", hdv_exits="sumo")
        apply_variant(p.model, o)
        n = int(o.get("seeds", default_seeds))
        inner = p.model.rollout

        def rollout(state, _inner=inner, _n=n, **kw):
            return _inner(_expand(state, _n), **{k: _expand(v, _n) for k, v in kw.items()})
        p.model.rollout = rollout
        G["p"][name] = (p, cap, o, n)


def _changes(lane, msk, hdv):
    both = msk[1:] & msk[:-1]
    a, b = lane[:-1], lane[1:]
    ch = both & (a != b) & hdv[None]
    return (int((ch & (a == 1) & (b == 0)).sum()), int((ch & (a == 0) & (b == 1)).sum()),
            int((ch & (a == 2)).sum()))


def _med(x, m):
    return float(np.median(x[m])) if m.any() else np.nan


def work(payload):
    idx, rec, arm = payload
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles
    out = []
    for name, (p, cap, o, n_seeds) in G["p"].items():
        try:
            n_steps = 240
            t0, ids, kin, is_cav = t0_vehicles(rec)
            teacher, present_seq, _ = recorded_teacher(rec, t0, ids, kin, n_steps, arm,
                                                       p.backend.tools)
            if teacher is None:
                out.append({"idx": idx, "var": name, "error": "no arm"})
                continue
            p.predict_states(rec, {**rec.get("cav_controllers", {}), "arm": arm}, n_steps * DT, 0)
            r = cap["last"]
            n = len(ids)
            tr = teacher[0].numpy()
            rp = present_seq[0].numpy().astype(bool)
            hdv = ~np.asarray(is_cav, bool)
            rlan = tr[:, :, 1].astype(int)
            pos_s = r["positions"][:, :, :n].numpy()            # [S, T, N]
            lan_s = r["lanes"][:, :, :n].numpy().astype(int)
            msk_s = r["masks"][:, :, :n].numpy().astype(bool)
            S = pos_s.shape[0]
            pos = np.concatenate([np.repeat(tr[None, :1, :, 0], S, 0), pos_s], 1)
            lan = np.concatenate([np.repeat(rlan[None, :1], S, 0), lan_s], 1)
            msk = np.concatenate([np.repeat(rp[None, :1], S, 0), msk_s], 1)
            res = {"idx": idx, "var": name, "error": None, "world": rec.get("world"),
                   "seeds": S, "n_hdv": int(hdv.sum())}
            cnt = np.array([_changes(lan[s], msk[s], hdv) for s in range(S)], float)
            res["wm_ovt"], res["wm_ret"], res["wm_mrg"] = cnt.mean(0).tolist()
            res["wm_ovt_s"], res["wm_ret_s"] = cnt[:, 0].tolist(), cnt[:, 1].tolist()
            res["rec_ovt"], res["rec_ret"], res["rec_mrg"] = _changes(rlan, rp, hdv)
            # lane-0 share time series (every 10 s) on the matched mainline HDVs
            occ_w, occ_r = [], []
            for i in range(0, n_steps + 1, 20):
                ml = msk[:, i] & rp[i][None] & hdv[None] & (rlan[i] <= 1)[None]
                occ_w.append(float(np.mean([(lan[s, i][ml[s]] == 0).mean() if ml[s].any()
                                            else np.nan for s in range(S)])))
                m0 = rp[i] & hdv & (rlan[i] <= 1)
                occ_r.append(float((rlan[i][m0] == 0).mean()) if m0.any() else np.nan)
            res["occ_wm_series"], res["occ_rec_series"] = occ_w, occ_r
            for ts in TIMES:
                i = int(ts / DT)
                both = msk[:, i] & rp[i][None] & hdv[None]                         # [S, N]
                d = np.abs(pos[:, i] - tr[i, :, 0][None])
                per = np.array([_med(d[s], both[s]) for s in range(S)])
                res[f"ds{ts}_single"] = float(per[0])
                res[f"ds{ts}_avg"] = float(np.nanmean(per)) if np.isfinite(per).any() else np.nan
                allb = both.all(0)
                ens = np.abs(np.where(both, pos[:, i], 0.0).sum(0) / np.maximum(both.sum(0), 1)
                             - tr[i, :, 0])
                res[f"ds{ts}_ens"] = _med(ens, allb)
                ml = both & (rlan[i] <= 1)[None]
                res[f"occ0_wm{ts}"] = float(np.nanmean([(lan[s, i][ml[s]] == 0).mean()
                                                        if ml[s].any() else np.nan
                                                        for s in range(S)]))
                res[f"right{ts}"] = float(np.nanmean([(lan[s, i][ml[s]] == rlan[i][ml[s]]).mean()
                                                      if ml[s].any() else np.nan
                                                      for s in range(S)]))
                m0 = rp[i] & hdv & (rlan[i] <= 1)
                res[f"occ0_rec{ts}"] = float((rlan[i][m0] == 0).mean()) if m0.any() else np.nan
                res[f"blow{ts}_s"] = per.tolist()
            out.append(res)
        except Exception as exc:  # noqa: BLE001
            import traceback
            out.append({"idx": idx, "var": name,
                        "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"})
    return out


# ------------------------------------------------------------------ summary
def _ci_median_diff(x, y, worlds, n_boot=2000, seed=0):
    """Median paired difference and a world-cluster bootstrap 95 % CI."""
    d = np.asarray(x) - np.asarray(y)
    ok = np.isfinite(d)
    d, w = d[ok], np.asarray(worlds)[ok]
    if not len(d):
        return np.nan, np.nan, np.nan
    uw = np.unique(w)
    rng = np.random.default_rng(seed)
    meds = []
    groups = {u: d[w == u] for u in uw}
    for _ in range(n_boot):
        pick = rng.choice(uw, len(uw))
        meds.append(np.median(np.concatenate([groups[u] for u in pick])))
    return float(np.median(d)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def summarise(path: str, ref: str | None) -> str:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    res = [r for r in data["results"] if not r.get("error")]
    names = [s[0] for s in data["specs"]]
    by = {nm: {r["idx"]: r for r in res if r["var"] == nm} for nm in names}
    common = sorted(set.intersection(*(set(v) for v in by.values() if v)))
    rows = []
    hdr = ("| variant | seeds | ovt / ret / mrg per state (SUMO {so:.1f} / {sr:.1f} / {sm:.1f}) "
           "| lane-0 share 60 / 120 s (SUMO {o60:.3f} / {o120:.3f}) | right lane 60 / 120 s "
           "| ds60 single / avg / ens | ds120 single / avg / ens | blow-ups single / avg |")
    first = by[names[0]]
    rows.append(hdr.format(
        so=np.mean([first[i]["rec_ovt"] for i in common]),
        sr=np.mean([first[i]["rec_ret"] for i in common]),
        sm=np.mean([first[i]["rec_mrg"] for i in common]),
        o60=np.nanmean([first[i]["occ0_rec60"] for i in common]),
        o120=np.nanmean([first[i]["occ0_rec120"] for i in common])))
    rows.append("|" + "---|" * 8)
    for nm in names:
        rr = [by[nm][i] for i in common]
        if not rr:
            continue

        def m(key, f=np.nanmean):
            return f([r[key] for r in rr])

        def blow(kind):
            return sum(1 for r in rr if r[f"ds60_{kind}"] > 50 and r[f"ds120_{kind}"] > 100)
        rows.append(
            f"| {nm} | {rr[0]['seeds']} | {m('wm_ovt'):.2f} / {m('wm_ret'):.2f} / {m('wm_mrg'):.2f} "
            f"| {m('occ0_wm60'):.3f} / {m('occ0_wm120'):.3f} | {m('right60'):.3f} / {m('right120'):.3f} "
            f"| {m('ds60_single', np.nanmedian):.1f} / {m('ds60_avg', np.nanmedian):.1f} / "
            f"{m('ds60_ens', np.nanmedian):.1f} "
            f"| {m('ds120_single', np.nanmedian):.1f} / {m('ds120_avg', np.nanmedian):.1f} / "
            f"{m('ds120_ens', np.nanmedian):.1f} | {blow('single')} / {blow('avg')} |")
    out = [f"States: {len(common)} (common to all variants). File: {os.path.basename(path)}", "",
           *rows]
    if ref and ref in by:
        out += ["", f"Paired per-state difference vs `{ref}` (median [world-cluster 95 % CI]; "
                "+ = worse):", "",
                "| variant | d ds60 avg | d ds120 avg | d ds60 single | d ds120 single |",
                "|---|---|---|---|---|"]
        for nm in names:
            if nm == ref:
                continue
            worlds = [by[nm][i].get("world") for i in common]
            cells = []
            for key in ("ds60_avg", "ds120_avg", "ds60_single", "ds120_single"):
                med, lo, hi = _ci_median_diff([by[nm][i][key] for i in common],
                                              [by[ref][i][key] for i in common], worlds)
                cells.append(f"{med:+.1f} [{lo:+.1f}, {hi:+.1f}]")
            out.append(f"| {nm} | " + " | ".join(cells) + " |")
    out += ["", "Lane-0 share every 20 s (t = 0, 20, ..., 120 s), WM vs SUMO:", ""]
    for nm in names:
        rr = [by[nm][i] for i in common]
        if rr:
            w = np.nanmean([r["occ_wm_series"] for r in rr], 0)
            out.append(f"- {nm}: " + " ".join(f"{x:.3f}" for x in w[::2]))
    rr = [by[names[0]][i] for i in common]
    out.append("- SUMO: " + " ".join(f"{x:.3f}" for x in np.nanmean(
        [r["occ_rec_series"] for r in rr], 0)[::2]))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--var", action="append", default=[])
    ap.add_argument("--states", default="E:/agent-work/c4-v6merge/_ref/states_h60/wall_same.json")
    ap.add_argument("--arm", default="wall")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--summarise")
    ap.add_argument("--ref")
    a = ap.parse_args()
    if a.summarise:
        print(summarise(a.summarise, a.ref))
        return
    import wm_eval_common as wc
    from wms.eval.normalize import iter_windows
    specs = [parse_var(s) for s in a.var]
    recs = [(i, r) for i, r in enumerate(iter_windows(wc.guard(a.states)))
            if i % a.every == a.offset]
    if a.limit:
        recs = recs[: a.limit]
    tasks = [(i, r, a.arm) for i, r in recs]
    t0 = time.time()
    res = []
    if a.workers <= 1:
        init(specs, a.seeds)
        for t in tasks:
            res.extend(work(t))
    else:
        from multiprocessing import Pool
        with Pool(a.workers, initializer=init, initargs=(specs, a.seeds)) as pool:
            for j, rr in enumerate(pool.imap_unordered(work, tasks, chunksize=1)):
                res.extend(rr)
                if (j + 1) % 6 == 0:
                    print(f"{j+1}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    with open(a.out, "wb") as fh:
        pickle.dump({"specs": specs, "results": res}, fh)
    print(f"done {time.time()-t0:.0f}s errors {sum(1 for r in res if r.get('error'))}", flush=True)
    for r in [r for r in res if r.get("error")][:3]:
        print(r)


if __name__ == "__main__":
    main()
