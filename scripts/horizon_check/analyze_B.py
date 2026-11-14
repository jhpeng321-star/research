"""HORIZON-CHECK part B analysis: decision accuracy of WM-imagined arm differences vs SUMO.

    python scripts/horizon_check/analyze_B.py --root runs/active/C4-HORIZON-20261013/B --out <json>

Sign convention (as traffic_shepherd.c4.branching): D_X = improvement of arm X over all-native
B (cost_B - cost_X for costs, value_X - value_B for benefits); POSITIVE = BETTER.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HORIZONS = ("30", "60", "120")
# (label, SUMO scope, SUMO key, imagined key, sign)
METRICS = (
    ("delay_all", "all", "road_delay_vs", "road_delay_vs", -1),
    ("team_delay_all", "all", "team_delay_vs", "road_delay_vs", -1),
    ("delay_cohort", "cohort", "delay", "road_delay_vs", -1),
    ("reward_all", "all", "reward", "reward", +1),
    ("reward_cohort", "cohort", "reward", "reward", +1),
    ("throughput_all", "all", "throughput_soft", "throughput_soft", +1),
    ("throughput_cohort", "cohort", "throughput", "throughput_soft", +1),
    ("hard_brake_all", "all", "hard_brake", "hard_brake", -1),
    ("hard_brake_cohort", "cohort", "hard_brake", "hard_brake", -1),
)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return None
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return pearson(ra, rb)


def agree(sim, img, thresh=None) -> dict:
    sim, img = np.asarray(sim, float), np.asarray(img, float)
    out = {"n": int(sim.size)}
    if sim.size == 0:
        return out
    out.update(sign_agreement=float(np.mean(np.sign(sim) == np.sign(img))),
               pearson=pearson(sim, img), spearman=spearman(sim, img),
               sumo_median_abs=float(np.median(np.abs(sim))),
               imag_median_abs=float(np.median(np.abs(img))),
               slope_img_on_sumo=(float(np.polyfit(sim, img, 1)[0]) if sim.size >= 3 and sim.std() > 0
                                  else None))
    if sim.size >= 4:
        big = np.abs(sim) >= np.median(np.abs(sim))
        out["sign_agreement_top_half"] = float(np.mean(np.sign(sim[big]) == np.sign(img[big])))
    if thresh is not None:
        big = np.abs(sim) >= thresh
        out["decisive_threshold"] = thresh
        out["n_decisive"] = int(big.sum())
        out["sign_agreement_decisive"] = (float(np.mean(np.sign(sim[big]) == np.sign(img[big])))
                                          if big.any() else None)
    return out


def _modes(manifest: dict) -> dict:
    """tag -> driver mode.  Specs without ',mode' predate parse_wms: those episodes ran
    imagine_multi's then-default coop_imagine driver SAMPLING."""
    out = {}
    for spec in manifest["args"]["wm"]:
        tag, rest = spec.split("=", 1)
        parts = rest.split(",")
        out[tag] = parts[1] if len(parts) > 1 else "sample"
    return out


def load(root: Path, pattern: str = "s*_p*") -> list:
    """Points of every complete-or-partial episode (ABORTED_* dirs excluded), tags renamed so
    that sampling-mode imaginations carry an 's' suffix, reimagine_*.jsonl tags merged."""
    pts = []
    for f in sorted(root.glob(f"{pattern}/points.jsonl")):
        ep = f.parent
        modes = _modes(json.loads((ep / "MANIFEST.json").read_text(encoding="utf-8")))
        rename = {t: (t + "s" if m == "sample" and not t.endswith("s") else t)
                  for t, m in modes.items()}
        extra = {}
        for g in sorted(ep.glob("reimagine_*.jsonl")):
            for line in g.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if not r.get("error"):
                        extra[int(r["t_s"])] = r
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            for blk in p["controllers"].values():
                blk["imagined"] = {rename.get(t, t): v for t, v in blk["imagined"].items()}
            p["B_hdv_abs_ds_median_m"] = {rename.get(t, t): v for t, v in
                                          (p.get("B_hdv_abs_ds_median_m") or {}).items()}
            r = extra.get(int(p["t_s"]))
            if r:
                for cname, tags in r["controllers"].items():
                    p["controllers"][cname]["imagined"].update(tags)
                p["B_hdv_abs_ds_median_m"].update(r["B_hdv_abs_ds_median_m"])
            pts.append(p)
    return pts


def sumo_d(p, cname, arm, h, scope, key, sign):
    x = p["controllers"][cname]["sumo"][arm][h][scope][key]
    b = p["sumo_B"][h][scope][key]
    return sign * (x - b)


def pairs(points, cname, tag, h, metric, kinds=None):
    label, scope, skey, ikey, sign = metric
    sim, img, meta = [], [], []
    for p in points:
        blk = p["controllers"].get(cname)
        if not blk or "error" in blk["imagined"].get(tag, {"error": 1}):
            continue
        im = blk["imagined"][tag]["h"][h]["D"]
        for arm in blk["arms"][1:]:
            kind = arm.split(":")[0]
            if kinds and kind not in kinds:
                continue
            sim.append(sumo_d(p, cname, arm, h, scope, skey, sign))
            img.append(im[arm][ikey])
            meta.append((p["seed"], p["penetration"], p["t_s"], arm))
    return sim, img, meta


def decisions(points, cname, tag, h, metric) -> dict:
    """Per point: argmax over {B, arms} of imagined D vs SUMO D; regret in SUMO units."""
    label, scope, skey, ikey, sign = metric
    top1, regret, regret_rel, allb = [], [], [], []
    for p in points:
        blk = p["controllers"].get(cname)
        if not blk or "error" in blk["imagined"].get(tag, {"error": 1}):
            continue
        im = blk["imagined"][tag]["h"][h]["D"]
        arms = blk["arms"]          # includes B (D = 0)
        s = np.array([0.0 if a == "B" else sumo_d(p, cname, a, h, scope, skey, sign) for a in arms])
        i = np.array([0.0 if a == "B" else im[a][ikey] for a in arms])
        best, pick = int(s.argmax()), int(i.argmax())
        top1.append(best == pick)
        regret.append(float(s[best] - s[pick]))
        spread = float(s.max() - s.min())
        regret_rel.append(float((s[best] - s[pick]) / spread) if spread > 0 else 0.0)
        j = arms.index("ALL")
        allb.append(np.sign(s[j]) == np.sign(i[j]))
    if not top1:
        return {"n": 0}
    return {"n": len(top1), "top1_agreement": float(np.mean(top1)),
            "mean_regret": float(np.mean(regret)), "median_regret": float(np.median(regret)),
            "mean_regret_rel_to_spread": float(np.mean(regret_rel)),
            "ALL_vs_B_sign_agreement": float(np.mean(allb))}


def stats(pred, rec) -> dict:
    p, r = np.asarray(pred, float), np.asarray(rec, float)
    ok = np.isfinite(p) & np.isfinite(r)
    p, r = p[ok], r[ok]
    if p.size == 0:
        return {"n": 0}
    mae = float(np.abs(p - r).mean())
    den = float(np.abs(r).mean())
    return {"n": int(p.size), "mae": mae, "bias": float((p - r).mean()),
            "rel_err": mae / den if den > 0 else None, "pearson": pearson(p, r)}


def aggregate_B(points, tag) -> dict:
    """Imagined all-native B vs SUMO B: zone aggregates (vs all traffic / vs t0 cohort),
    persistence baseline, cumulative road delay level, per-vehicle |ds|, cohort share."""
    out = {}
    cname = "damping"     # imagined B is controller-independent up to sampling; use one
    for h in HORIZONS:
        rows = defaultdict(list)
        for p in points:
            blk = p["controllers"].get(cname, {})
            im = blk.get("imagined", {}).get(tag)
            if not im or "error" in im:
                continue
            zi = im["h"][h]["zone_by_arm"]["B"]
            za, zc, z0 = p["sumo_B"][h]["zone_all"], p["sumo_B"][h]["zone_cohort"], p["zone_t0"]
            for key in ("count", "mean_speed", "queued_cells"):
                rows[f"{key}|img"].append(zi[key] if zi[key] is not None else np.nan)
                rows[f"{key}|all"].append(za[key] if za[key] is not None else np.nan)
                rows[f"{key}|cohort"].append(zc[key] if zc[key] is not None else np.nan)
                rows[f"{key}|t0"].append(z0[key] if z0[key] is not None else np.nan)
            rows["delay|img"].append(im["h"][h]["metrics"]["B"]["road_delay_vs"])
            rows["delay|all"].append(p["sumo_B"][h]["all"]["road_delay_vs"])
            rows["delay|cohort"].append(p["sumo_B"][h]["cohort"]["delay"])
            rows["team_delay|all"].append(p["sumo_B"][h]["all"]["team_delay_vs"])
            rows["reward|img"].append(im["h"][h]["metrics"]["B"]["reward"])
            rows["reward|all"].append(p["sumo_B"][h]["all"]["reward"])
            rows["reward|cohort"].append(p["sumo_B"][h]["cohort"]["reward"])
            rows["cohort_share_network"].append(p["sumo_B"][h]["n_present_cohort"]
                                                / max(p["sumo_B"][h]["n_present"], 1))
            rows["cohort_share_zone"].append(zc["count"] / za["count"] if za["count"] else np.nan)
            ds = (p.get("B_hdv_abs_ds_median_m", {}).get(tag, {}).get(cname, {}) or {}).get(h)
            rows["ds"].append(ds if ds is not None else np.nan)
        blk = {}
        for key in ("count", "mean_speed", "queued_cells"):
            blk[f"zone_{key}"] = {
                "img_vs_all_traffic": stats(rows[f"{key}|img"], rows[f"{key}|all"]),
                "img_vs_t0_cohort": stats(rows[f"{key}|img"], rows[f"{key}|cohort"]),
                "persistence_t0_vs_all_traffic": stats(rows[f"{key}|t0"], rows[f"{key}|all"])}
        blk["cum_road_delay_B"] = {"img_vs_all": stats(rows["delay|img"], rows["delay|all"]),
                                   "img_vs_cohort": stats(rows["delay|img"], rows["delay|cohort"]),
                                   "sumo_cohort_vs_all": stats(rows["delay|cohort"], rows["delay|all"])}
        blk["cum_reward_B"] = {"img_vs_all": stats(rows["reward|img"], rows["reward|all"]),
                               "img_vs_cohort": stats(rows["reward|img"], rows["reward|cohort"])}
        blk["source_wait_share_of_team_delay"] = float(np.nanmean(
            1 - np.asarray(rows["delay|all"]) / np.maximum(np.asarray(rows["team_delay|all"]), 1e-9)))
        blk["cohort_share_of_network_vehicles"] = float(np.nanmean(rows["cohort_share_network"]))
        blk["cohort_share_of_zone_vehicles"] = float(np.nanmean(rows["cohort_share_zone"]))
        d = np.asarray(rows["ds"], float)
        blk["B_hdv_abs_ds_median_m"] = float(np.nanmedian(d)) if np.isfinite(d).any() else None
        blk["n_points"] = len(rows["ds"])
        out[h] = blk
    return out


def _round(x, nd: int = 4):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", default="s*_p*", help="episode directory pattern")
    ap.add_argument("--tags", nargs="+", default=None, help="WM tags to analyse (default all)")
    ap.add_argument("--metrics", nargs="+", default=None, help="metric labels (default all)")
    args = ap.parse_args()
    points = load(Path(args.root), args.glob)
    ctrls = sorted({c for p in points for c in p["controllers"]})
    tags = sorted({t for p in points for c in p["controllers"].values() for t in c["imagined"]})
    if args.tags:
        tags = [t for t in tags if t in args.tags]
    metrics = [m for m in METRICS if not args.metrics or m[0] in args.metrics]
    res = {"schema": "c4_horizon_check_B_analysis/1", "n_points": len(points),
           "episodes": sorted({(p["seed"], p["penetration"]) for p in points}),
           "controllers": ctrls, "wms": tags, "convention": "D positive = better than all-native B",
           "pairs": {}, "decisions": {}, "aggregate_B": {}, "imag_errors": []}
    for p in points:
        for c, blk in p["controllers"].items():
            for t, im in blk["imagined"].items():
                if "error" in im:
                    res["imag_errors"].append([p["seed"], p["penetration"], p["t_s"], c, t,
                                               im["error"][:200]])
    for tag in tags:
        res["aggregate_B"][tag] = aggregate_B(points, tag)
        for c in ctrls + ["ALL_CONTROLLERS"]:
            cs = ctrls if c == "ALL_CONTROLLERS" else [c]
            for h in HORIZONS:
                for metric in metrics:
                    sim, img = [], []
                    by_kind = {}
                    for cc in cs:
                        s_, i_, _ = pairs(points, cc, tag, h, metric)
                        sim += s_
                        img += i_
                        for kind in ("ALL", "GROUP", "ONLY"):
                            a, b, _ = pairs(points, cc, tag, h, metric, kinds=(kind,))
                            by_kind.setdefault(kind, ([], []))
                            by_kind[kind][0].extend(a)
                            by_kind[kind][1].extend(b)
                    thr = 0.1 * float(np.median(np.abs(sim))) if sim else None
                    res["pairs"].setdefault(tag, {}).setdefault(c, {}).setdefault(h, {})[metric[0]] = {
                        "all_arms": agree(sim, img, thr),
                        **{kind: agree(*v) for kind, v in by_kind.items()}}
                    if c != "ALL_CONTROLLERS":
                        res["decisions"].setdefault(tag, {}).setdefault(c, {}).setdefault(h, {})[
                            metric[0]] = decisions(points, c, tag, h, metric)
    Path(args.out).write_text(json.dumps(_round(res), separators=(",", ":")), encoding="utf-8")
    print(f"{len(points)} points, controllers {ctrls}, wms {tags}, errors {len(res['imag_errors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
