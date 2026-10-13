"""GAPRULE: derive the speed-dependent lane-change gap acceptance from TRAIN data only.

    python tools/gaprule_fit.py collect OUT.npz [--config configs/c4_wm.toml] [--neg-frac 0.05]
                                                [--data-root E:/research-dense] [--limit N]
    python tools/gaprule_fit.py fit OUT.npz FIT.json

``collect`` iterates the TRAIN windows of the C4 WM config (pack_v3 native + pack_dspeed +
switching direct-mix TRAIN part; the same lists the WM trains on - never wall_same, never a
held-out / CAL list) and records, for every HDV step t and every legal adjacent target lane
(no change into the ramp lane 2; 2 -> 1 only), the WM-frame target-lane front/rear bumper
gaps and speeds at t (same ``lane_neighbour`` the physics mask uses) and whether SUMO moved
the car into that lane at t+1.  All recorded changes are kept; non-changes are subsampled
(``--neg-frac``, weight 1/neg_frac).  The old (fixed) mask decision with default driver
parameters is stored for comparison.

``fit`` chooses the speed-rule constants (see ``wms.physics.mobil``):
    lead   gap >= s0_lead + T_lead * v      + max(0, v^2  - v_lead^2) / (2 b_lead)
    follow gap >= s0_foll + T_foll * v_foll + max(0, v_foll^2 - v^2) / (2 b_foll)
by a small grid search that maximises the recall of recorded changes subject to a bounded
admission of DANGEROUS non-change states (definition in ``dangerous``) - TRAIN only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

CATS = {(2, 1): 0, (1, 0): 1, (0, 1): 2}
CAT_NAMES = ("merge", "overtake", "return")
#: A cut-in into the target lane is DANGEROUS when, in the configuration after the step (the
#: one-step-anticipated gaps the speed rule also judges), a target-lane car overlaps the ego,
#: the bumper gap to the target-lane leader or follower is below DANGER_GAP_M, the follower
#: (resp. the ego) would need more than DANGER_DECEL_MPS2 of constant deceleration to avoid
#: hitting the other car (closing speed^2 / (2 gap); comfortable decel of the SUMO vTypes is
#: 4.2-4.5 m/s2), or the time headway of the follower (resp. the ego) is below
#: DANGER_HEADWAY_S while it drives faster than DANGER_HEADWAY_MIN_V_MPS.
DANGER_GAP_M = 1.0
DANGER_DECEL_MPS2 = 4.5
DANGER_HEADWAY_S = 0.3
DANGER_HEADWAY_MIN_V_MPS = 5.0


def dangerous(v, gl, vl, gf, vf, overlap=None):
    """Rule-independent danger flag of a cut-in into the target lane (numpy arrays)."""
    gl_c = np.maximum(gl, 0.3)
    gf_c = np.maximum(gf, 0.3)
    b_lead = np.maximum(v - vl, 0.0) ** 2 / (2.0 * gl_c)
    b_foll = np.maximum(vf - v, 0.0) ** 2 / (2.0 * gf_c)
    out = (gl < DANGER_GAP_M) | (gf < DANGER_GAP_M) | (b_lead > DANGER_DECEL_MPS2) \
        | (b_foll > DANGER_DECEL_MPS2) \
        | ((v > DANGER_HEADWAY_MIN_V_MPS) & (gl < DANGER_HEADWAY_S * v)) \
        | ((vf > DANGER_HEADWAY_MIN_V_MPS) & (gf < DANGER_HEADWAY_S * vf))
    return out if overlap is None else out | overlap


def speed_rule(v, gl, vl, gf, vf, c, overlap=None):
    """Speed-rule feasibility (numpy), constants ``c`` as in PhysicsConfig.gap_*."""
    req_l = c["gap_s0_lead_m"] + c["gap_t_lead_s"] * v \
        + np.maximum(v ** 2 - vl ** 2, 0.0) / (2.0 * c["gap_b_lead_mps2"])
    req_f = c["gap_s0_follow_m"] + c["gap_t_follow_s"] * vf \
        + np.maximum(vf ** 2 - v ** 2, 0.0) / (2.0 * c["gap_b_follow_mps2"])
    ok = (gl >= req_l) & (gf >= req_f)
    return ok if overlap is None else ok & ~overlap


# --------------------------------------------------------------------- collect
def cmd_collect(args) -> int:
    from wmlane_smoke import point_data

    if args.data_root:
        point_data(Path(args.data_root))
    from wms.cli.assemble import build_datasets
    from wms.config.load import load_train_config
    from wms.physics.config import PhysicsConfig
    from wms.physics.mobil import (MASKED_LOGIT, anticipated_kin, mobil_merge_logits,
                                   target_overlap)
    from wms.physics.neighbours import lane_neighbour

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    cfg = load_train_config(args.config, ["runtime.device=cpu"])
    scratch = Path(args.out).with_suffix("").as_posix() + "_ds"
    train, _, _ = build_datasets(cfg, Path(scratch))
    p = PhysicsConfig(sumo_fidelity=True, lane_rule="learned_discretionary")
    rng = np.random.default_rng(int(args.seed) + 1000 * int(args.offset))
    cols = {k: [] for k in ("v", "s", "lane", "target", "gl", "vl", "gf", "vf", "length",
                            "label", "weight", "old_ok", "window",
                            "va", "gla", "vla", "gfa", "vfa", "ova")}
    n = len(train) if args.limit is None else min(len(train), int(args.limit))
    junction_rows = 0
    for w in range(int(args.offset), n, int(args.stride)):
        item = train[w]
        st = torch.as_tensor(item["states"]).float()
        pres = torch.as_tensor(item["present"]).bool()
        cav = torch.as_tensor(item["is_cav"]).bool()
        onehot = st[..., 1:4]
        lane = onehot.argmax(-1)
        # Junction transits in these packs are carried rows with a FROZEN s while v > 0 and a
        # spurious lane-0 one-hot (edge boundaries s = 0 / 75.44 m): they produce fake
        # 1 -> 0 -> 1 "changes" and fake lane-0 neighbours.  Drop such rows (ego and neighbours).
        s_all, v_all = st[..., 0], st[..., 4]
        frozen = torch.zeros_like(pres)
        same = (s_all[1:] - s_all[:-1]).abs() < 1e-4
        frozen[1:] |= same
        frozen[:-1] |= same
        junction_like = pres & frozen & (v_all > 0.2)
        lane_ok = pres & (onehot.sum(-1) > 0.5) & ~junction_like
        n_junction = int((pres & ~cav & junction_like).sum())
        kin, p0 = st[:-1], lane_ok[:-1]
        with torch.no_grad():
            old = mobil_merge_logits(kin, p0, p) > MASKED_LOGIT + 1e-6      # [T,K,3]
            nb, nba = {}, {}
            kin_a = anticipated_kin(kin, p.gap_anticipation_s)
            for tl in (0, 1):
                q = torch.full(lane[:-1].shape, tl)
                nb[tl] = (*lane_neighbour(kin, p0, q, p, ahead=True),
                          *lane_neighbour(kin, p0, q, p, ahead=False))
                nba[tl] = (*lane_neighbour(kin_a, p0, q, p, ahead=True),
                           *lane_neighbour(kin_a, p0, q, p, ahead=False),
                           target_overlap(kin_a, p0, q, p.gap_overlap_margin_m))
        hdv = lane_ok & ~cav
        both = (hdv[:-1] & hdv[1:]).numpy()
        l0, l1 = lane[:-1].numpy(), lane[1:].numpy()
        s_np, v_np = kin[..., 0].numpy(), kin[..., 4].numpy()
        for d in (-1, 1):
            tgt = l0 + d
            legal = both & (tgt >= 0) & (tgt <= 1) & ~((l0 == 0) & (tgt == 2))
            legal &= (l0 != 2) | (tgt == 1)
            label = legal & (l1 == tgt)
            keep = label | (legal & (rng.random(legal.shape) < float(args.neg_frac)))
            ti, ki = np.nonzero(keep)
            if ti.size == 0:
                continue
            tg = tgt[ti, ki]
            gl = np.where(tg == 0, nb[0][0].numpy()[ti, ki], nb[1][0].numpy()[ti, ki])
            vl = np.where(tg == 0, nb[0][1].numpy()[ti, ki], nb[1][1].numpy()[ti, ki])
            gf = np.where(tg == 0, nb[0][2].numpy()[ti, ki], nb[1][2].numpy()[ti, ki])
            vf = np.where(tg == 0, nb[0][3].numpy()[ti, ki], nb[1][3].numpy()[ti, ki])
            lab = label[ti, ki]

            def pick(table, c):
                return np.where(tg == 0, table[0][c].numpy()[ti, ki],
                                table[1][c].numpy()[ti, ki])

            row = {"va": kin_a[..., 4].numpy()[ti, ki], "gla": pick(nba, 0),
                   "vla": pick(nba, 1), "gfa": pick(nba, 2), "vfa": pick(nba, 3),
                   "ova": pick(nba, 4)}
            row.update({"v": v_np[ti, ki], "s": s_np[ti, ki], "lane": l0[ti, ki], "target": tg,
                   "gl": gl, "vl": vl, "gf": gf, "vf": vf,
                   "length": kin[..., 6].numpy()[ti, ki], "label": lab,
                   "weight": np.where(lab, 1.0, 1.0 / float(args.neg_frac)),
                   "old_ok": old.numpy()[ti, ki, tg], "window": np.full(ti.size, w)})
            for key, value in row.items():
                cols[key].append(value)
        junction_rows += n_junction
        if w % 100 == 0:
            print(f"window {w}/{n}", flush=True)
    out = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez_compressed(args.out, **out)
    print(f"rows {out['v'].size}, recorded changes {int(out['label'].sum())}, "
          f"junction-like HDV rows dropped {junction_rows}")
    return 0


# ------------------------------------------------------------------------- fit
def _load(paths):
    shards = [dict(np.load(x)) for x in paths.split(",")]
    d = {k: np.concatenate([s[k] for s in shards]) for k in shards[0]}
    cat = np.full(d["v"].shape, -1)
    for (a, b), c in CATS.items():
        cat[(d["lane"] == a) & (d["target"] == b)] = c
    d["cat"] = cat
    d["ova"] = d["ova"].astype(bool)
    # Danger is a property of the configuration after the step (anticipated gaps + overlap).
    d["dang"] = dangerous(d["va"], d["gla"], d["vla"], d["gfa"], d["vfa"], d["ova"])
    return d


def _features(d, timing):
    """``(v, gl, vl, gf, vf, overlap)`` judged at t or on the one-step-anticipated state."""
    if timing == "t":
        return d["v"], d["gl"], d["vl"], d["gf"], d["vf"], None
    return d["va"], d["gla"], d["vla"], d["gfa"], d["vfa"], d["ova"]


def evaluate(d, ok) -> dict:
    """Recall per category and false-feasible (weighted non-change) rates of a mask ``ok``."""
    lab, w = d["label"].astype(bool), d["weight"]
    neg = ~lab
    dang = d["dang"]
    out = {"recall_all": float(ok[lab].mean()) if lab.any() else None}
    for c, name in enumerate(CAT_NAMES):
        m = lab & (d["cat"] == c)
        out[f"recall_{name}"] = float(ok[m].mean()) if m.any() else None
        out[f"n_changes_{name}"] = int(m.sum())
    wn = w[neg]
    out["feasible_nonchange"] = float((wn * ok[neg]).sum() / wn.sum())
    out["dangerous_nonchange_frac"] = float((wn * dang[neg]).sum() / wn.sum())
    out["feasible_and_dangerous_nonchange"] = float((wn * (ok & dang)[neg]).sum() / wn.sum())
    md = neg & dang
    out["feasible_given_dangerous_nonchange"] = float(
        (w[md] * ok[md]).sum() / max(w[md].sum(), 1e-9))
    out["recorded_change_dangerous_frac"] = float(dang[lab].mean()) if lab.any() else None
    return out


GRID = {
    "gap_s0_lead_m": [0.5, 1.0, 1.5, 2.0, 3.0],
    "gap_t_lead_s": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "gap_b_lead_mps2": [2.0, 3.0, 4.5, 6.0],
    "gap_s0_follow_m": [0.5, 1.0, 1.5, 2.0, 3.0],
    "gap_t_follow_s": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "gap_b_follow_mps2": [2.0, 3.0, 4.5, 6.0],
}


def _lead_ok(f, s0, tt, b):
    v, gl, vl = f[0], f[1], f[2]
    return gl >= s0 + tt * v + np.maximum(v ** 2 - vl ** 2, 0.0) / (2 * b)


def _follow_ok(f, s0, tt, b):
    v, gf, vf = f[0], f[3], f[4]
    return gf >= s0 + tt * vf + np.maximum(vf ** 2 - v ** 2, 0.0) / (2 * b)


def cmd_fit(args) -> int:
    d = _load(args.npz)
    lab = d["label"].astype(bool)
    print(f"rows {lab.size}, changes {int(lab.sum())} "
          + ", ".join(f"{n} {int((lab & (d['cat'] == c)).sum())}"
                      for c, n in enumerate(CAT_NAMES)))
    descr = {}
    for c, name in enumerate(CAT_NAMES):
        m = lab & (d["cat"] == c)
        if m.any():
            descr[name] = {k: [round(float(x), 2)
                               for x in np.percentile(d[k][m], [1, 5, 25, 50, 75])]
                           for k in ("v", "gl", "gf", "vl", "vf", "s", "gla", "gfa")}
    old = evaluate(d, d["old_ok"].astype(bool))
    # Separable search: precompute the lead and the follow condition of every grid point.
    lead_keys = list(itertools.product(GRID["gap_s0_lead_m"], GRID["gap_t_lead_s"],
                                       GRID["gap_b_lead_mps2"]))
    foll_keys = list(itertools.product(GRID["gap_s0_follow_m"], GRID["gap_t_follow_s"],
                                       GRID["gap_b_follow_mps2"]))
    f = _features(d, args.timing)
    clear = np.ones_like(lab) if f[5] is None else ~f[5]
    lead = {k: _lead_ok(f, *k) & clear for k in lead_keys}
    foll = {k: _follow_ok(f, *k) for k in foll_keys}
    neg = ~lab
    w = d["weight"]
    dang = d["dang"]
    cat_masks = [lab & (d["cat"] == c) for c in range(len(CAT_NAMES))]
    wn_sum = w[neg].sum()
    md = neg & dang
    wd_sum = max(w[md].sum(), 1e-9)
    table = []
    for lk in lead_keys:
        for fk in foll_keys:
            ok = lead[lk] & foll[fk]
            recs = [float(ok[m].mean()) for m in cat_masks if m.any()]
            table.append({
                "gap_s0_lead_m": lk[0], "gap_t_lead_s": lk[1], "gap_b_lead_mps2": lk[2],
                "gap_s0_follow_m": fk[0], "gap_t_follow_s": fk[1], "gap_b_follow_mps2": fk[2],
                "mean_cat_recall": float(np.mean(recs)),
                "feasible_nonchange": float((w * ok)[neg].sum() / wn_sum),
                "feasible_given_dangerous_nonchange": float((w * ok)[md].sum() / wd_sum)})
    # Selection (TRAIN only): admissible = admits at most `max_danger` of the dangerous
    # non-change states; among those, the knee = the SMALLEST feasible-but-unused rate whose
    # category-averaged recall is within `recall_slack` of the best admissible recall
    # (i.e. the most conservative rule that keeps essentially all attainable recall).
    adm = [r for r in table if r["feasible_given_dangerous_nonchange"] <= float(args.max_danger)]
    best_rec = max(r["mean_cat_recall"] for r in adm)
    near = [r for r in adm if r["mean_cat_recall"] >= best_rec - float(args.recall_slack)]
    chosen = min(near, key=lambda r: (r["feasible_nonchange"], -r["mean_cat_recall"]))
    c = {k: chosen[k] for k in GRID}
    result = {"source": "TRAIN windows of configs/c4_wm.toml (tools/gaprule_fit.py collect); "
                        "junction-transit rows (frozen s) dropped",
              "danger_definition": {"gap_m": DANGER_GAP_M, "decel_mps2": DANGER_DECEL_MPS2,
                                    "headway_s": DANGER_HEADWAY_S,
                                    "headway_min_v_mps": DANGER_HEADWAY_MIN_V_MPS,
                                    "judged_on": "one-step-anticipated configuration + overlap"},
              "max_feasible_given_dangerous": float(args.max_danger),
              "recall_slack": float(args.recall_slack),
              "best_admissible_mean_cat_recall": best_rec,
              "recorded_change_percentiles_1_5_25_50_75": descr,
              "old_fixed_mask_default_params": old,
              "chosen_constants": c,
              "timing": args.timing,
              "chosen_eval": evaluate(d, speed_rule(*f[:5], c, overlap=f[5])),
              "n_grid": len(table)}
    adm.sort(key=lambda r: (-round(r["mean_cat_recall"], 3), r["feasible_nonchange"]))
    result["top10_admissible_by_recall"] = adm[:10]
    near.sort(key=lambda r: r["feasible_nonchange"])
    result["top10_knee_candidates"] = near[:10]
    Path(args.fit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.fit).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("old_fixed_mask_default_params", "chosen_constants",
                                             "chosen_eval", "best_admissible_mean_cat_recall")},
                     indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("out")
    c.add_argument("--config", default="configs/c4_wm.toml")
    c.add_argument("--neg-frac", type=float, default=0.05)
    c.add_argument("--data-root", default="E:/research-dense")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--offset", type=int, default=0, help="window slice start (parallel shards)")
    c.add_argument("--stride", type=int, default=1, help="window slice step (parallel shards)")
    f = sub.add_parser("fit")
    f.add_argument("npz", help="comma-separated shard files")
    f.add_argument("fit")
    f.add_argument("--max-danger", type=float, default=0.02)
    f.add_argument("--recall-slack", type=float, default=0.01)
    f.add_argument("--timing", default="anticipated", choices=("anticipated", "t"))
    args = ap.parse_args()
    return cmd_collect(args) if args.cmd == "collect" else cmd_fit(args)


if __name__ == "__main__":
    raise SystemExit(main())
