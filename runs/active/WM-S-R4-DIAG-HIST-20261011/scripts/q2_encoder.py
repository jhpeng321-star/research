"""Q2: R4 driver-encoder outputs on real-history vs standard states vs TRAIN windows.

Collects the 9 inferred IDM/MOBIL parameters, the fused style embedding, shrinkage,
uncertainty and per-slot valid-history length for:
  * real-history quick-20 and full set (121 native ticks),
  * standard h60 quick-20 and full set (11 ticks, run-start limited),
  * a sampled TRAIN distribution (trainer-identical featurisation, 75 % start shifts).
Writes out/q2_encoder.json including per-parameter / per-dimension training 5-95 %
ranges consumed by the Q4 intervention test.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_lib2 import (DIAG, OUT, SETS, dump_json, encode_state, encode_train_window,
                       ensure_env, iter_records, make_predictor, q, quick20,
                       train_window_iter)

ensure_env()
import numpy as np  # noqa: E402

PNAMES = ("v0", "time_headway", "min_gap", "accel", "decel", "politeness",
          "mobil_threshold", "merge_gap", "merge_urgency")
LOWER = (5.0, 0.2, 0.5, 0.3, 0.5, 0.0, 0.0, 2.0, 0.1)
UPPER = (60.0, 4.0, 10.0, 5.0, 10.0, 1.0, 2.0, 30.0, 2.0)
N_TRAIN = 220


def collect_states(predictor, records, label):
    rows = []
    for i, record in enumerate(records):
        enc = encode_state(predictor, record)
        hdv = ~enc["is_cav"]
        rows.append({
            "world": str(record.get("world")), "t0_s": float(record.get("t0_s", 0.0)),
            "key": str(record.get("key")),
            "n_hdv": int(hdv.sum()),
            "params": enc["values"][hdv].tolist(),           # [n,9]
            "embedding": enc["embedding"][hdv].tolist(),      # [n,102]
            "shrinkage": enc["shrinkage"][hdv].tolist(),
            "uncertainty": enc["uncertainty"][hdv].tolist(),
            "hist_valid_ticks": enc["hist_valid_ticks"][hdv].tolist(),
            "ids": [v for v, h in zip(enc["ids"], hdv) if h],
        })
        if (i + 1) % 20 == 0:
            print(f"[{label}] {i + 1}/{len(records)}", flush=True)
    return rows


def stack_params(rows):
    return np.asarray([p for r in rows for p in r["params"]], float)


def stack_emb(rows):
    return np.asarray([e for r in rows for e in r["embedding"]], float)


def stack_ticks(rows):
    return np.asarray([t for r in rows for t in
                       r.get("hist_valid_ticks", r.get("valid_ticks"))], float)


def main() -> None:
    t0 = time.time()
    predictor = make_predictor("R4")

    real_q20 = quick20(SETS["real_q20"])
    std_q20 = quick20(SETS["std_q20"])
    data = {"real_q20": collect_states(predictor, real_q20, "real_q20"),
            "std_q20": collect_states(predictor, std_q20, "std_q20")}

    # training distribution
    train_rows = []
    gen = train_window_iter(N_TRAIN, seed=0)
    n_shift = n_zero = 0
    for win in gen:
        enc = encode_train_window(predictor, win)
        hdv = win["present"] & ~win["is_cav"]
        if hdv.sum() < 5:
            continue
        train_rows.append({
            "world": win["world"], "shift_ticks": win["shift"], "n_hdv": int(hdv.sum()),
            "params": enc["values"][hdv].tolist(),
            "embedding": enc["embedding"][hdv].tolist(),
            "shrinkage": enc["shrinkage"][hdv].tolist(),
            "valid_ticks": win["feat_mask"][hdv].sum(-1).astype(int).tolist(),
        })
        n_shift += int(win["shift"] > 0)
        n_zero += int(win["shift"] == 0)
    print(f"train windows: {len(train_rows)} ({n_shift} shifted / {n_zero} unshifted)",
          flush=True)

    tp, te = stack_params(train_rows), stack_emb(train_rows)
    tt = stack_ticks(train_rows)
    train_stats = {
        "n_windows": len(train_rows), "n_shifted": n_shift, "n_unshifted": n_zero,
        "n_hdv_slots": int(tp.shape[0]),
        "valid_ticks": q(tt, (5, 25, 50, 75, 95, 100)),
        "share_slots_full_61": float((tt >= 61).mean()),
        "share_slots_lt_5ticks": float((tt < 5).mean()),
        "params": {name: {"lower": LOWER[i], "upper": UPPER[i],
                          "quantiles": q(tp[:, i], (1, 5, 25, 50, 75, 95, 99)),
                          "mean": float(tp[:, i].mean()),
                          "std": float(tp[:, i].std()),
                          "share_at_lower": float((np.abs(tp[:, i] - LOWER[i]) < 1e-4).mean()),
                          "share_at_upper": float((np.abs(tp[:, i] - UPPER[i]) < 1e-4).mean())}
                   for i, name in enumerate(PNAMES)},
        "embedding": {"dim": int(te.shape[1]),
                      "q05": np.percentile(te, 5, axis=0).tolist(),
                      "q95": np.percentile(te, 95, axis=0).tolist(),
                      "median": np.median(te, axis=0).tolist(),
                      "mean": te.mean(axis=0).tolist(),
                      "std": te.std(axis=0).tolist(),
                      "q01": np.percentile(te, 1, axis=0).tolist(),
                      "q99": np.percentile(te, 99, axis=0).tolist()},
        "shrinkage": q(np.asarray([s for r in train_rows for s in r["shrinkage"]], float)),
    }

    emb_lo = np.asarray(train_stats["embedding"]["q05"])
    emb_hi = np.asarray(train_stats["embedding"]["q95"])
    emb_med = np.asarray(train_stats["embedding"]["median"])
    emb_mu = np.asarray(train_stats["embedding"]["mean"])
    emb_sd = np.asarray(train_stats["embedding"]["std"])
    p_lo = np.asarray([train_stats["params"][n]["quantiles"]["5"] for n in PNAMES])
    p_hi = np.asarray([train_stats["params"][n]["quantiles"]["95"] for n in PNAMES])

    def summarize_group(rows):
        parr, earr = stack_params(rows), stack_emb(rows)
        tarr = stack_ticks(rows)
        z = (earr - emb_mu[None, :]) / np.maximum(emb_sd[None, :], 1e-6)
        per_state = []
        for r in rows:
            e = np.asarray(r["embedding"], float)
            p = np.asarray(r["params"], float)
            zr = (e - emb_mu[None, :]) / np.maximum(emb_sd[None, :], 1e-6)
            per_state.append({
                "world": r["world"], "t0_s": r["t0_s"], "n_hdv": r["n_hdv"],
                "emb_frac_dims_outside_5_95": float(((e < emb_lo[None, :]) |
                                                     (e > emb_hi[None, :])).mean()),
                "emb_mean_abs_z": float(np.abs(zr).mean()),
                "emb_p95_abs_z": float(np.percentile(np.abs(zr), 95)),
                "param_frac_outside_5_95": float(((p < p_lo[None, :]) |
                                                  (p > p_hi[None, :])).mean()),
            })
        return {
            "n_states": len(rows), "n_hdv_slots": int(parr.shape[0]),
            "valid_ticks": q(tarr, (5, 25, 50, 75, 95, 100)),
            "share_slots_full_61": float((tarr >= 61).mean()),
            "share_slots_lt_5ticks": float((tarr < 5).mean()),
            "params": {name: {"quantiles": q(parr[:, i], (1, 5, 25, 50, 75, 95, 99)),
                              "share_at_lower": float((np.abs(parr[:, i] - LOWER[i]) < 1e-4).mean()),
                              "share_at_upper": float((np.abs(parr[:, i] - UPPER[i]) < 1e-4).mean()),
                              "share_outside_train_5_95": float(
                                  ((parr[:, i] < p_lo[i]) | (parr[:, i] > p_hi[i])).mean()),
                              "train_q05": float(p_lo[i]), "train_q95": float(p_hi[i])}
                       for i, name in enumerate(PNAMES)},
            "embedding": {
                "frac_slots_outside_any_dim": float(
                    (((earr < emb_lo[None, :]) | (earr > emb_hi[None, :])).any(1)).mean()),
                "mean_frac_dims_outside_5_95": float(
                    ((earr < emb_lo[None, :]) | (earr > emb_hi[None, :])).mean()),
                "mean_abs_z": float(np.abs(z).mean()),
                "p95_abs_z": float(np.percentile(np.abs(z), 95)),
                "max_abs_z": float(np.abs(z).max()),
                "per_dim_share_outside": (((earr < emb_lo[None, :]) |
                                           (earr > emb_hi[None, :])).mean(0)).tolist()},
            "shrinkage": q(np.asarray([s for r in rows for s in r["shrinkage"]], float)),
            "uncertainty_mean": float(np.asarray(
                [u for r in rows for u in r["uncertainty"]], float).mean()),
            "per_state": per_state,
        }

    out = {"train": train_stats,
           "real_q20": summarize_group(data["real_q20"]),
           "std_q20": summarize_group(data["std_q20"]),
           "valid_ticks_train_hist": q(tt)}
    # compact slot-level dump for Q3 blow-up look-ups
    dump_json({"real_q20": data["real_q20"], "std_q20": data["std_q20"]},
              OUT / "q2_slots.json")
    slim = {k: v for k, v in out.items() if k != "embedding_raw"}
    dump_json(slim, OUT / "q2_encoder.json")
    print(f"q2 done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
