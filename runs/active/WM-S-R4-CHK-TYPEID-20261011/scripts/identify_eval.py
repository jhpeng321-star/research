"""Identification accuracy vs history length and traffic condition (evaluation).

Sets: wallc1_states_h60/wall_same.json (5 s dense history, 67 states) and
wallc1_states_h60_real/wall_same.json (60 s native history, 120 states).
Population: HDVs of the t0 roster (the slots a rollout would parameterise);
truth from out/truth_full.json (evaluation-only, never an identifier input).

Writes out/ident_eval.json (tables + confusion matrices) and out/ident_rows.pkl.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from typeid import TYPE_NAMES, pilot_scale, posteriors, prep  # noqa: E402
from wms.eval.normalize import initial_frame, iter_windows  # noqa: E402

TRUTH = json.loads((HERE.parent / "out/truth_full.json").read_text(encoding="utf-8"))
SETS = {
    "wall_same_h60": (ROOT / "data_cache/wallc1_states_h60/wall_same.json", (5,)),
    "wall_same_real": (ROOT / "data_cache/wallc1_states_h60_real/wall_same.json",
                       (5, 10, 20, 30, 60)),
}
OUT = HERE.parent / "out"
CONF_KEYS = ("free", "car_following", "mixed", "jam")


def condition(share: float, mean_v: float) -> str:
    if not np.isfinite(mean_v):
        return "no_data"
    if mean_v < 5.0:
        return "jam"
    if share >= 0.5:
        return "car_following"
    if share <= 0.1:
        return "free"
    return "mixed"


def main() -> int:
    rows = []
    results = {"population": "t0-roster HDVs", "sets": {}, "rows_file": "ident_rows.pkl"}
    for set_name, (set_file, lengths) in SETS.items():
        records = []
        for rec in iter_windows(set_file):
            d = prep(rec["history"])
            init = initial_frame(rec)
            cav = {str(r["id"]): bool(r.get("is_cav", False))
                   for r in init["vehicles"] if r.get("present", True)}
            records.append((rec, d, cav))
        scale = pilot_scale([d for _, d, _ in records])
        per_len = {L: {"n": 0, "correct": 0, "conf": Counter(), "conf_true": Counter(),
                       "confusion": Counter(), "by_share_bin": Counter(),
                       "by_share_bin_n": Counter(), "nodata": 0,
                       "post": []} for L in lengths}
        for idx, (rec, d, cav) in enumerate(records):
            world = str(rec.get("world"))
            per_length, stats = posteriors(d, lengths, scale)
            col = {x: i for i, x in enumerate(d["ids"])}
            veh_types = TRUTH[world]["vehicles"]
            row = {"idx": idx, "world": world, "key": rec.get("key"),
                   "t0": float(rec.get("t0_s", 0.0)), "veh": {}}
            for vid, is_cav in cav.items():
                if is_cav or vid not in col:
                    continue
                j = col[vid]
                true_t = veh_types.get(vid)
                cond = condition(stats["leader_share"][j], stats["mean_v"][j])
                entry = {"true": true_t, "cond": cond,
                         "leader_share": stats["leader_share"][j],
                         "mean_v": stats["mean_v"][j], "L": {}}
                for L in lengths:
                    out_j = per_length[L][j]
                    entry["L"][L] = out_j
                    blk = per_len[L]
                    if true_t is None:
                        continue
                    blk["n"] += 1
                    hit = out_j["argmax"] == true_t
                    blk["correct"] += int(hit)
                    blk["confusion"][(true_t, out_j["argmax"])] += 1
                    if out_j["n_ticks"] == 0:
                        blk["nodata"] += 1
                    blk["conf"][cond] += 1
                    blk["conf_true"][cond] += int(hit)
                    share = stats["leader_share"][j]
                    b = 0 if share < 0.1 else (1 if share < 0.5 else 2)
                    blk["by_share_bin"][b] += 1
                    blk["by_share_bin_n"][b] += int(hit)
                    blk["post"].append(max(out_j["posterior"].values()))
                row["veh"][vid] = entry
            rows.append(row)
        set_block = {"pilot_scale": scale}
        for L in lengths:
            blk = per_len[L]
            n = blk["n"]
            set_block[str(L)] = {
                "n": n, "accuracy": blk["correct"] / n if n else None,
                "no_evidence": blk["nodata"],
                "confusion_rows_true_cols_pred": {
                    f"{a}|{b}": blk["confusion"][(a, b)]
                    for a in TYPE_NAMES for b in TYPE_NAMES},
                "per_class_recall": {
                    a: (blk["confusion"][(a, a)]
                        / sum(blk["confusion"][(a, b)] for b in TYPE_NAMES)
                        if sum(blk["confusion"][(a, b)] for b in TYPE_NAMES) else None)
                    for a in TYPE_NAMES},
                "predicted_share": {
                    b: (sum(blk["confusion"][(a, b)] for a in TYPE_NAMES) / n if n else None)
                    for b in TYPE_NAMES},
                "by_condition": {c: {"n": blk["conf"][c],
                                     "accuracy": (blk["conf_true"][c] / blk["conf"][c]
                                                  if blk["conf"][c] else None)}
                                 for c in CONF_KEYS},
                "by_leader_share_bin": {
                    b: {"n": blk["by_share_bin"][b],
                        "accuracy": (blk["by_share_bin_n"][b] / blk["by_share_bin"][b]
                                     if blk["by_share_bin"][b] else None)}
                    for b in (0, 1, 2)},
                "mean_max_posterior": float(np.mean(blk["post"])) if blk["post"] else None,
            }
        results["sets"][set_name] = set_block
        print(set_name, "scale", scale, flush=True)
        for L in lengths:
            b = set_block[str(L)]
            print(f"  L={L:2d}s acc={b['accuracy']:.4f} recall={b['per_class_recall']}",
                  flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "ident_rows.pkl", "wb") as fh:
        pickle.dump(rows, fh, protocol=4)
    (OUT / "ident_eval.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'ident_eval.json'} and ident_rows.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
