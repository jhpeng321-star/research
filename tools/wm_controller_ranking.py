"""Does the WM rank CAV controllers like SUMO?  (C4 v7 WM evaluation, owner 2026-10-17.)

For every seed/demand/penetration, all controllers start from the identical t0 = 300 s state
(warmup is always native; the identity is checked and reported).  For each controller the CAVs
are teacher-forced along that controller's RECORDED SUMO trajectory and the WM predicts the HDVs
for H = 120 s (60 s metrics are the prefix of the same rollout).  Entries are OFF: the WM state
holds the t0 roster only; WM and SUMO metrics are both computed on the HDVs of the t0 roster
(SUMO all-HDV values including post-t0 entrants are reported for context, ``sumo_full``).

HDV metrics (t0-roster HDVs, steps 1..H/0.5):
* ``delay``  - delay proxy [veh-s] = sum over present HDV-steps upstream of s = 300 m of
  0.5 * max(0, 1 - v / 31.29) (the collector's speed-deficit definition, speed limit 31.29 m/s).
  Presence is exogenous in the WM rollout (SUMO exits at s ~ 539-553 m end a WM vehicle); the
  300 m cut keeps those exits from setting the WM's exposure - each model's own positions decide
  how long a vehicle counts.  ``exit_leaks`` (WM only) counts HDVs a SUMO exit removed while the
  WM still had them upstream of the cut (should be ~0); ``delay_500`` / ``exit_leaks_500`` = the
  same with a 500 m cut (v5m lags SUMO downstream, so ~20 leaks per state at 120 s: biased);
* ``discharge`` - HDVs first seen upstream of s = 75.44 m (merge-zone end) that cross it within H;
* ``ds_speed`` - vehicle-step mean HDV speed in [75.44, 600) m (downstream of the merge);
* ``us_speed`` - same in [-400, 0) m (approach), for context.

Report (per penetration and pooled; bootstrap 95 % CI over seeds, a seed = cluster):
(a) per controller WM vs SUMO; (b) controller difference vs the reference (native), WM vs SUMO,
and its error; (c) Spearman rank agreement between WM and SUMO controller means (complete seeds
only), bootstrap CI over seeds; plus, per pair of models, paired differences of |Delta error| and
of Spearman.

  PYTHONPATH="src;." python tools/wm_controller_ranking.py \
      --root E:/research-dense/runs/active/C4-DIAG-20261016/ranking \
      --ckpt v5m=E:/research-dense/runs/active/C4-V5-20261014/fetched/c4_wm_v5/wm_s_best.pt \
      --mode ON --workers 8 --json ranking_v5m.json --md RESULTS_part.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import c4_episode_states as E  # noqa: E402
import wm_eval_common as C  # noqa: E402

DEFAULT_ROOT = "E:/research-dense/runs/active/C4-DIAG-20261016/ranking"
METRICS = ("delay", "delay_500", "discharge", "ds_speed", "us_speed")
LEAK_KEY = {"delay": "exit_leaks", "delay_500": "exit_leaks_500"}
DELAY_END_M = 300.0       # primary delay corridor end (exposure-clean, see the docstring)
G: dict = {}


def hkey(h: float) -> str:
    return str(int(round(float(h))))


def hdv_metrics(pos, vel, msk, hdv, h: float) -> dict:
    return {"delay": C.delay_proxy(vel, msk, hdv, h, pos=pos, end=DELAY_END_M),
            "delay_500": C.delay_proxy(vel, msk, hdv, h, pos=pos, end=500.0),
            "discharge": C.discharge(pos, msk, hdv, h),
            "ds_speed": C.region_speed(pos, vel, msk, hdv, h, C.MERGE_END_M, 600.0),
            "us_speed": C.region_speed(pos, vel, msk, hdv, h, -400.0, 0.0)}


def init(ckpts, device, threads, hdv_exits="sumo", entries="off"):
    G["models"] = {name: C.load_predictor(path, mode, device, threads, hdv_exits, entries)
                   for name, path, mode in ckpts}


def work(payload):
    job, ctrl_dirs, ref, t0, horizons = payload
    seed, demand, pen = (int(x) for x in E.JOB_RE.match(job).groups())
    out = {"job": job, "seed": seed, "demand": demand, "penetration": pen / 1000.0, "ctrls": {}}
    ref_frame = None
    hmax = max(horizons)
    for ctrl, d in ctrl_dirs:
        res = {"dir": str(d)}
        try:
            ep = E.load_episode(d)
            rec = E.episode_record(ep, t0_s=t0, arm=ctrl, horizon_steps=int(round(hmax / C.DT_S)))
        except Exception as exc:  # noqa: BLE001 - recorded per controller
            res["error"] = repr(exc)
            out["ctrls"][ctrl] = res
            continue
        if ctrl == ref or ref_frame is None:
            ref_frame = ref_frame or rec["initial_state"]
        res["identical_t0"] = E.frames_identical(rec["initial_state"], ref_frame)
        res["sumo_full"] = {hkey(h): E.sumo_full_metrics(ep, t0, h) for h in horizons}
        for name, (p, cap) in G["models"].items():
            t = time.time()
            r = C.roll(p, cap, rec, ctrl, hmax)
            hdv = ~r["is_cav"]
            res.setdefault("n_hdv", int(hdv.sum()))
            res.setdefault("n_cav", int(r["is_cav"].sum()))
            res.setdefault("sumo", {hkey(h): hdv_metrics(r["rec_pos"], r["rec_vel"],
                                                             r["rec_msk"], hdv, h)
                                    for h in horizons})
            res[name] = {hkey(h): {**hdv_metrics(r["wm_pos"], r["wm_vel"], r["wm_msk"], hdv, h),
                                   "exit_leaks": C.exit_leaks(r["wm_pos"], r["rec_msk"], hdv, h,
                                                              end=DELAY_END_M),
                                   "exit_leaks_500": C.exit_leaks(r["wm_pos"], r["rec_msk"], hdv,
                                                                  h, end=500.0)}
                         for h in horizons}
            res.setdefault("sec", {})[name] = time.time() - t
        out["ctrls"][ctrl] = res
    return out


# ----------------------------------------------------------------------------- summary
def _table(results, ctrls, src, h, metric):
    """[n_jobs, n_ctrls] array of ``src`` (model name / 'sumo' / 'sumo_full') values."""
    a = np.full((len(results), len(ctrls)), np.nan)
    for i, r in enumerate(results):
        for j, c in enumerate(ctrls):
            v = r["ctrls"].get(c, {}).get(src, {}).get(h, {}).get(metric)
            if v is not None:
                a[i, j] = v
    return a


def _boot_spearman(wm, su, groups, n_boot, seed=0):
    """Spearman of controller means (columns) between wm and su; CI by resampling groups."""
    point = C.spearman(np.nanmean(wm, 0), np.nanmean(su, 0))
    labels = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in labels]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(labels), len(labels))])
        bs.append(C.spearman(np.nanmean(wm[idx], 0), np.nanmean(su[idx], 0)))
    bs = np.asarray(bs)
    bs = bs[np.isfinite(bs)]
    if not bs.size:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def summarize(results, ctrls, names, ref, horizons, n_boot=2000):
    pens = sorted({r["penetration"] for r in results})
    out = {}
    for pen in [*pens, "pooled"]:
        rs = [r for r in results if pen == "pooled" or r["penetration"] == pen]
        # complete cases: every controller present, identical t0, rolled by every model
        rs = [r for r in rs if all(c in r["ctrls"] and "sumo" in r["ctrls"][c]
                                   and r["ctrls"][c].get("identical_t0") for c in ctrls)]
        if not rs:
            continue
        groups = np.array([r["seed"] for r in rs])
        blk = {"n_states": len(rs), "n_seeds": int(np.unique(groups).size), "h": {}}
        for h in map(hkey, horizons):
            hb = {}
            for metric in METRICS:
                su = _table(rs, ctrls, "sumo", h, metric)
                sf = _table(rs, ctrls, "sumo_full", h, metric) if metric == "discharge" else None
                jr = ctrls.index(ref)
                mb = {"sumo_mean": dict(zip(ctrls, np.nanmean(su, 0).tolist())),
                      "sumo_delta": {c: C.bootstrap_mean(su[:, j] - su[:, jr], n_boot, groups=groups)
                                     for j, c in enumerate(ctrls) if c != ref},
                      "models": {}}
                if sf is not None:
                    mb["sumo_full_mean"] = dict(zip(ctrls, np.nanmean(sf, 0).tolist()))
                    mb["sumo_full_delta"] = {c: C.bootstrap_mean(sf[:, j] - sf[:, jr], n_boot,
                                                                 groups=groups)
                                             for j, c in enumerate(ctrls) if c != ref}
                derr = {}
                for name in names:
                    wm = _table(rs, ctrls, name, h, metric)
                    d_wm = wm - wm[:, [jr]]
                    d_su = su - su[:, [jr]]
                    derr[name] = d_wm - d_su
                    nonref = [j for j, c in enumerate(ctrls) if c != ref]
                    mb["models"][name] = {
                        "wm_mean": dict(zip(ctrls, np.nanmean(wm, 0).tolist())),
                        "err": {c: C.bootstrap_mean(wm[:, j] - su[:, j], n_boot, groups=groups)
                                for j, c in enumerate(ctrls)},
                        "wm_delta": {c: C.bootstrap_mean(d_wm[:, j], n_boot, groups=groups)
                                     for j, c in enumerate(ctrls) if c != ref},
                        "delta_err": {c: C.bootstrap_mean(derr[name][:, j], n_boot, groups=groups)
                                      for j, c in enumerate(ctrls) if c != ref},
                        "mean_abs_delta_err": C.bootstrap_mean(
                            np.abs(derr[name][:, nonref]).mean(1), n_boot, groups=groups),
                        "sign_agree": {c: float(np.mean(np.sign(np.nanmean(d_wm[:, j]))
                                                        == np.sign(np.nanmean(d_su[:, j]))))
                                       for j, c in enumerate(ctrls) if c != ref},
                        "spearman": _boot_spearman(wm, su, groups, n_boot),
                        "spearman_per_state_mean": float(np.nanmean(
                            [C.spearman(wm[i], su[i]) for i in range(len(rs))])),
                        "exit_leaks_per_state": (dict(zip(ctrls, np.nanmean(
                            _table(rs, ctrls, name, h, LEAK_KEY[metric]), 0).tolist()))
                            if metric in LEAK_KEY else None),
                    }
                mb["pairs"] = {}
                for na, nb in itertools.combinations(names, 2):
                    wa, wb = _table(rs, ctrls, na, h, metric), _table(rs, ctrls, nb, h, metric)
                    nonref = [j for j, c in enumerate(ctrls) if c != ref]
                    diff = np.abs(derr[na][:, nonref]).mean(1) - np.abs(derr[nb][:, nonref]).mean(1)
                    sp = []
                    labels = np.unique(groups)
                    members = [np.flatnonzero(groups == g) for g in labels]
                    rng = np.random.default_rng(1)
                    for _ in range(n_boot):
                        idx = np.concatenate([members[i] for i in
                                              rng.integers(0, len(labels), len(labels))])
                        s_m = np.nanmean(su[idx], 0)
                        sp.append(C.spearman(np.nanmean(wa[idx], 0), s_m)
                                  - C.spearman(np.nanmean(wb[idx], 0), s_m))
                    sp = np.asarray(sp)
                    sp = sp[np.isfinite(sp)]
                    point = (C.spearman(np.nanmean(wa, 0), np.nanmean(su, 0))
                             - C.spearman(np.nanmean(wb, 0), np.nanmean(su, 0)))
                    mb["pairs"][f"{na}-{nb}"] = {
                        "mean_abs_delta_err_diff": C.bootstrap_mean(diff, n_boot, groups=groups),
                        "spearman_diff": (point, float(np.percentile(sp, 2.5)) if sp.size else np.nan,
                                          float(np.percentile(sp, 97.5)) if sp.size else np.nan)}
                hb[metric] = mb
            blk["h"][h] = hb
        out[str(pen)] = blk
    return out


UNITS = {"delay": "veh-s", "delay_500": "veh-s", "discharge": "veh", "ds_speed": "m/s", "us_speed": "m/s"}


def markdown(summ, ctrls, names, ref, horizons) -> str:
    L = []
    for pen, blk in summ.items():
        label = "pooled (both penetrations)" if pen == "pooled" else f"penetration {pen}"
        L.append(f"\n### {label}: {blk['n_states']} states, {blk['n_seeds']} seeds\n")
        for h in map(hkey, horizons):
            for metric in METRICS:
                mb = blk["h"][h][metric]
                L.append(f"\n**{metric} [{UNITS[metric]}], H = {h} s**\n")
                head = "| controller | SUMO | " + " | ".join(
                    f"{n} WM | {n} WM-SUMO | {n} Δ vs {ref} WM | SUMO Δ vs {ref} | {n} Δ error"
                    for n in names) + " |"
                L.append(head)
                L.append("|" + "---|" * (2 + 5 * len(names)))
                for c in ctrls:
                    row = [c, f"{mb['sumo_mean'][c]:.2f}"]
                    for n in names:
                        m = mb["models"][n]
                        row += [f"{m['wm_mean'][c]:.2f}", C.fmt_ci(m["err"][c])]
                        if c == ref:
                            row += ["-", "-", "-"]
                        else:
                            row += [C.fmt_ci(m["wm_delta"][c]), C.fmt_ci(mb["sumo_delta"][c]),
                                    C.fmt_ci(m["delta_err"][c])]
                    L.append("| " + " | ".join(row) + " |")
                for n in names:
                    m = mb["models"][n]
                    sp = m["spearman"]
                    L.append(f"\n{n}: Spearman(WM, SUMO controller means) = {sp[0]:+.2f} "
                             f"[{sp[1]:+.2f}, {sp[2]:+.2f}]; per-state mean "
                             f"{m['spearman_per_state_mean']:+.2f}; mean |Δ error| over "
                             f"non-{ref} controllers {C.fmt_ci(m['mean_abs_delta_err'])}")
                    if metric in LEAK_KEY:
                        L.append(f"\n{n}: HDVs removed by a SUMO exit while the WM had them "
                                 "upstream of the delay end, per state: " + ", ".join(
                                     f"{c} {m['exit_leaks_per_state'][c]:.2f}" for c in ctrls))
                for pair, d in mb["pairs"].items():
                    L.append(f"\npaired {pair}: mean |Δ error| A-B "
                             f"{C.fmt_ci(d['mean_abs_delta_err_diff'])} (neg = A better); "
                             f"Spearman A-B {C.fmt_ci(d['spearman_diff'])}")
                if "sumo_full_mean" in mb:
                    L.append("\nSUMO all HDVs incl. post-t0 entrants (context): " + ", ".join(
                        f"{c} {mb['sumo_full_mean'][c]:.1f}" for c in ctrls))
    return "\n".join(L) + "\n"


def merge_jsons(paths):
    """Merge saved ranking outputs: same jobs/controllers, model results joined by name."""
    ckpts, ctrls, counts, by_job = [], None, None, {}
    for path in paths:
        doc = json.load(open(C.guard(path), encoding="utf-8"))
        new = [(c["name"], c["path"], c["mode"]) for c in doc["ckpts"]]
        clash = {n for n, _, _ in new} & {n for n, _, _ in ckpts}
        if clash:
            raise SystemExit(f"model name(s) {clash} appear in more than one --from-json file")
        ckpts += new
        ctrls = ctrls or doc["controllers"]
        counts = counts or doc["episode_counts"]
        for job in doc["jobs"]:
            have = by_job.setdefault(job["job"], job)
            if have is job:
                continue
            for c, x in job["ctrls"].items():
                if c not in have["ctrls"]:
                    continue
                if x.get("sumo") != have["ctrls"][c].get("sumo"):
                    raise SystemExit(f"{job['job']}/{c}: SUMO reference differs between files")
                for n, _, _ in new:
                    if n in x:
                        have["ctrls"][c][n] = x[n]
    names = [n for n, _, _ in ckpts]
    results = [j for j in sorted(by_job.values(), key=lambda r: r["job"])]
    for r in results:   # a job/controller is complete only if every merged model rolled it
        for c, x in list(r["ctrls"].items()):
            if any(n not in x for n in names):
                x.pop("sumo", None)
    return ckpts, ctrls, counts, results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="dir with one collect output dir per controller (<root>/<controller>)")
    ap.add_argument("--controllers", default=None,
                    help="comma list (default: every subdir of --root with episodes); "
                         "NAME=DIR entries point a controller at another collect dir")
    ap.add_argument("--reference", default="native")
    ap.add_argument("--ckpt", action="append", default=[], help="NAME=path[@MODE]")
    ap.add_argument("--from-json", action="append", default=[],
                    help="re-summarise saved --json outputs (repeatable: model results of the "
                         "same jobs are merged by name, e.g. v5m from the laptop + v7ft from a "
                         "GPU box) instead of rolling out")
    ap.add_argument("--mode", default="ON")
    ap.add_argument("--t0", type=float, default=300.0)
    ap.add_argument("--horizons", default="60,120")
    ap.add_argument("--limit", type=int, default=None, help="max jobs (smoke)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--entries", choices=tuple(C.ENTRY_MODES), default="off",
                    help="vehicles entering after t0: off (frozen default) / queue = V9ENTRY entries "
                         "ON seeded with the recorded SUMO entrance backlog + demand")
    ap.add_argument("--hdv-exits", choices=("sumo", "none"), default="sumo",
                    help="sumo = recorded SUMO exits end HDV slots too (predictor default); "
                         "none = only CAV slots follow recorded presence")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None, help="write the markdown tables here")
    a = ap.parse_args(argv)
    horizons = [float(x) for x in a.horizons.split(",")]
    if a.from_json:
        ckpts, ctrls, counts, results = merge_jsons(a.from_json)
        if a.reference not in ctrls:
            raise SystemExit(f"reference {a.reference} not among {ctrls}")
        ctrls = [a.reference] + [c for c in ctrls if c != a.reference]
        root = Path(a.root)
    else:
        if not a.ckpt:
            raise SystemExit("--ckpt is required unless --from-json is given")
        ckpts = [C.parse_ckpt(s, a.mode.upper()) for s in a.ckpt]
        root = Path(C.guard(a.root))
        if a.controllers:
            specs = [s.split("=", 1) if "=" in s else (s, str(root / s))
                     for s in a.controllers.split(",")]
        else:
            specs = [(d.name, str(d)) for d in sorted(root.iterdir())
                     if d.is_dir() and (d / "episodes").is_dir()]
        jobs_by_ctrl = {name: E.find_jobs(d) for name, d in specs}
        ctrls = [name for name, _ in specs]
        if a.reference not in ctrls:
            raise SystemExit(f"reference {a.reference} not among {ctrls}")
        ctrls = [a.reference] + [c for c in ctrls if c != a.reference]
        all_jobs = sorted(set().union(*[set(j) for j in jobs_by_ctrl.values()]))
        payloads = []
        for job in all_jobs:
            dirs = [(c, jobs_by_ctrl[c][job]) for c in ctrls if job in jobs_by_ctrl[c]]
            payloads.append((job, dirs, a.reference, a.t0, horizons))
        if a.limit:
            payloads = payloads[:a.limit]
        counts = {c: len(jobs_by_ctrl[c]) for c in ctrls}
        print(f"[ranking] controllers {counts}; {len(payloads)} jobs; models {ckpts}", flush=True)
        t = time.time()
        if a.workers <= 1:
            init(ckpts, a.device, a.threads, a.hdv_exits, a.entries)
            results = [work(x) for x in payloads]
        else:
            with Pool(a.workers, initializer=init, initargs=(ckpts, a.device, a.threads, a.hdv_exits, a.entries)) as pool:
                results = list(pool.imap_unordered(work, payloads, chunksize=1))
        results.sort(key=lambda r: r["job"])
        print(f"[ranking] rollouts done in {time.time() - t:.0f} s", flush=True)
    names = [c[0] for c in ckpts]
    not_identical = [(r["job"], c) for r in results for c, x in r["ctrls"].items()
                     if x.get("identical_t0") is False]
    errors = [(r["job"], c, x["error"]) for r in results for c, x in r["ctrls"].items()
              if "error" in x]
    summ = summarize(results, ctrls, names, a.reference, horizons, a.bootstrap)
    md = markdown(summ, ctrls, names, a.reference, horizons)
    header = (f"controllers (complete episodes): {counts}\n"
              f"t0-identity failures: {not_identical or 'none'}; conversion errors: {errors or 'none'}\n")
    print(header + md)
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(C.guard(a.md)).write_text(header + md, encoding="utf-8")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        with open(C.guard(a.json), "w", encoding="utf-8") as fh:
            json.dump({"schema": "wm_controller_ranking/1", "root": str(root), "t0_s": a.t0,
                       "horizons": horizons, "reference": a.reference, "controllers": ctrls,
                       "entries": a.entries,
                       "episode_counts": counts,
                       "ckpts": [{"name": n, "path": p, "mode": m} for n, p, m in ckpts],
                       "t0_identity_failures": not_identical, "errors": errors,
                       "summary": summ, "jobs": results}, fh, indent=1)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
