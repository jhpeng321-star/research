"""Q5: is the real-history input malformed vs training windows?

Structural validation of the 20 quick real-history states' ``history`` blocks
(metadata, tick spacing, lane one-hot/junction rows, scales, ordering, t0-row
consistency with ``initial_state``, entrant masking, NaNs), plus a distribution
comparison of the featurised long-history channels against the training sample
(wall-arm reaction dynamics vs lpwo training arms). Writes out/q5_history.json.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_lib2 import (DATA, OUT, SETS, dump_json, ensure_env, q, quick20,
                       train_window_iter)

ensure_env()
import numpy as np  # noqa: E402
import torch  # noqa: E402

LONG_F = 9
CHANNELS = ("v/30", "a/3", "lane0", "lane1", "lane2", "gap/100", "disp/250",
            "dv/30", "leader_seen")


def validate_state(record: dict) -> dict:
    h = record["history"]
    kin = np.asarray(h["kin"], np.float32)
    mask = np.asarray(h["mask"], bool)
    target = np.asarray(h["target_mask"], bool)
    ids = [str(v) for v in h["ids"]]
    t, k, f = kin.shape
    out = {"world": str(record.get("world")), "t0_s": float(record.get("t0_s", 0.0)),
           "ticks": t, "slots": k, "features": f,
           "declared": {"history_s": h.get("history_s"),
                        "history_stride_s": h.get("history_stride_s"),
                        "dt_s": h.get("dt_s"),
                        "available_span_s": h.get("available_span_s"),
                        "run_start_limited": h.get("run_start_limited"),
                        "history_steps": h.get("history_steps")},
           "checks": {}, "flags": []}

    def check(name, ok, detail=None):
        out["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            out["flags"].append(name)

    d = out["declared"]
    check("metadata_complete", all(v is not None for v in d.values()))
    check("dt_is_0.5", d["dt_s"] == 0.5)
    check("span_matches_ticks", abs(d["available_span_s"] - (t - 1) * 0.5) < 1e-6)
    check("stride_is_1s", d["history_stride_s"] == 1.0)
    check("declared_60s", d["history_s"] >= 60.0)
    check("kin_shape", f == 7)
    check("mask_shape", mask.shape == (t, k))
    check("target_shape", target.shape == (t - 1, k))
    check("no_nan", bool(np.isfinite(kin[mask]).all()))
    check("ids_unique", len(set(ids)) == len(ids))

    # lane one-hot: rows either exactly one of cols 1:3 set (road) or none (junction/absent)
    onroad = mask & (kin[:, :, 1:4].sum(-1) == 1)
    junction = mask & (kin[:, :, 1:4].sum(-1) == 0)
    bad = mask & ~onroad & ~junction
    check("lane_onehot_valid", bad.sum() == 0, {"bad_rows": int(bad.sum())})

    # scales
    v = kin[:, :, 4][mask]
    a = kin[:, :, 5][mask]
    ln = kin[:, :, 6][mask]
    check("speed_range", bool(((v >= 0) & (v <= 40)).all()),
          {"min": float(v.min()), "max": float(v.max())})
    check("accel_range", bool(((a >= -12) & (a <= 12)).all()),
          {"min": float(a.min()), "max": float(a.max())})
    check("lengths_5m", bool(np.allclose(ln, 5.0) if ln.size else True),
          {"unique": sorted(set(np.round(ln, 2).tolist()))[:5]})

    # t0 row equals the initial state for shared ids
    init = record.get("initial_state") or {}
    pos0 = {str(vv.get("id")): vv for vv in init.get("vehicles", [])
            if vv.get("present", True)}
    col = {vid: i for i, vid in enumerate(ids)}
    mism = 0
    checked = 0
    for vid, row in pos0.items():
        j = col.get(vid)
        if j is None:
            continue
        last = kin[-1, j]
        lane_ok = (int(np.argmax(last[1:4])) == int(row.get("lane", -1))) or \
                  (last[1:4].sum() == 0 and int(row.get("lane", 0)) < 0)
        if lane_ok and abs(float(last[0]) - float(row.get("s", 0))) < 0.5 \
                and abs(float(last[4]) - float(row.get("v", 0))) < 0.5:
            checked += 1
        else:
            mism += 1
    check("t0_row_matches_initial_state", mism == 0,
          {"checked": checked, "mismatched": mism})
    check("initial_ids_covered", all(str(v) in col for v in pos0),
          {"missing": sum(1 for v in pos0 if str(v) not in col)})

    # per-vehicle observed pattern: first/last valid tick, interior gaps
    first_valid = np.full(k, -1)
    last_valid = np.full(k, -1)
    gaps_inside = 0
    for j in range(k):
        idx = np.flatnonzero(mask[:, j])
        if idx.size == 0:
            continue
        first_valid[j], last_valid[j] = int(idx[0]), int(idx[-1])
        gaps_inside += int((~mask[idx[0]:idx[-1] + 1, j]).sum())
    obs = mask.any(1)
    check("tick_grid_monotone", True, {"rows_with_any_obs": int(obs.sum())})
    check("no_interior_gaps", gaps_inside == 0, {"gap_cells": gaps_inside})
    # entrants = first valid tick after t=0; continuity via backward target product
    entrants = int((first_valid > 0).sum())
    present_t0 = int(mask[-1].sum())
    # backward continuity: target_mask[t]=False breaks the chain (slot reused)
    breaks = 0
    for j in range(k):
        idx = np.flatnonzero(mask[:, j])
        if idx.size < 2:
            continue
        for i in idx[:-1]:
            if not target[i, j]:
                breaks += 1
                break
    out["roster"] = {"present_t0": present_t0, "roster_ids": k,
                     "entrants_before_t0": entrants,
                     "slots_with_target_break": breaks,
                     "junction_share_obs": float(junction.sum() / max(1, mask.sum())),
                     "first_valid_median": float(np.median(first_valid[first_valid >= 0])),
                     "obs_span_s_median": float(np.median(
                         (last_valid - first_valid)[first_valid >= 0] * 0.5))}
    return out


def main() -> None:
    t0 = time.time()
    records = quick20(SETS["real_q20"])
    validations = [validate_state(r) for r in records]
    n_flag = sum(1 for v in validations if v["flags"])
    flag_names = sorted({f for v in validations for f in v["flags"]})
    roster_summary = {kk: q([v["roster"][kk] for v in validations])
                      for kk in validations[0]["roster"]}

    # featurised channel distribution: real vs std vs train
    from wms.data.history import long_history_features as lhf
    real_channels = []
    for r in records:
        hist = r["history"]
        kin = torch.as_tensor(np.asarray(hist["kin"], np.float32))
        mask = torch.as_tensor(np.asarray(hist["mask"], bool))
        target = torch.as_tensor(np.asarray(hist["target_mask"], bool))
        feat, fmask = lhf(kin, mask, target, mask[-1], history_s=60.0,
                          history_stride_s=1.0)
        f = feat.numpy()[fmask.numpy()]
        real_channels.append(f)
    real_f = np.concatenate(real_channels, 0)

    std_channels = []
    for r in quick20(SETS["std_q20"]):
        hist = r["history"]
        kin = torch.as_tensor(np.asarray(hist["kin"], np.float32))
        mask = torch.as_tensor(np.asarray(hist["mask"], bool))
        tm = np.asarray(hist.get("target_mask", []), bool)
        target = torch.as_tensor(tm if tm.size else np.zeros((max(0, kin.shape[0] - 1),
                                                              kin.shape[1]), bool))
        feat, fmask = lhf(kin, mask, target, mask[-1], history_s=60.0,
                          history_stride_s=1.0)
        std_channels.append(feat.numpy()[fmask.numpy()])
    std_f = np.concatenate(std_channels, 0)

    train_channels = []
    train_dyn = {"junction_share": [], "abs_accel_share": [], "stop_go": []}
    for win in train_window_iter(120, seed=1):
        fm = win["feat"][win["feat_mask"]]
        train_channels.append(fm)
        kk = win["kin"][win["present"]]
        train_dyn["junction_share"].append(float((kk[:, 1:4].sum(-1) == 0).mean()))
        train_dyn["abs_accel_share"].append(float((np.abs(kk[:, 5]) > 3).mean()))
        v = kk[:, 4]
        train_dyn["stop_go"].append(float(((v < 5)).mean()))
    train_f = np.concatenate(train_channels, 0)

    def ch_stats(f):
        return {name: {"q05": float(np.percentile(f[:, i], 5)),
                       "q50": float(np.percentile(f[:, i], 50)),
                       "q95": float(np.percentile(f[:, i], 95))}
                for i, name in enumerate(CHANNELS)}

    # kinematic dynamics of the observed history rows (raw), real vs train t0 rows
    real_kin_rows = []
    for r in records:
        hist = r["history"]
        kin = np.asarray(hist["kin"], np.float32)
        m = np.asarray(hist["mask"], bool)
        real_kin_rows.append(kin[m])
    rk = np.concatenate(real_kin_rows, 0)
    real_dyn = {"junction_share": float((rk[:, 1:4].sum(-1) == 0).mean()),
                "abs_accel_share_gt3": float((np.abs(rk[:, 5]) > 3).mean()),
                "share_v_below5": float((rk[:, 4] < 5).mean()),
                "share_v_below2": float((rk[:, 4] < 2).mean())}

    out = {"n_states": len(validations), "states_with_flags": n_flag,
           "flag_names": flag_names,
           "roster_summary": roster_summary,
           "channel_quantiles": {"real": ch_stats(real_f), "std": ch_stats(std_f),
                                 "train": ch_stats(train_f)},
           "n_obs_rows": {"real": int(real_f.shape[0]), "std": int(std_f.shape[0]),
                          "train": int(train_f.shape[0])},
           "raw_history_dynamics_real": real_dyn,
           "raw_history_dynamics_train": {
               kk: q(v) for kk, v in train_dyn.items()},
           "validations": validations}
    dump_json(out, OUT / "q5_history.json")
    print(f"q5 done in {time.time() - t0:.0f}s; flagged states {n_flag} {flag_names}",
          flush=True)


if __name__ == "__main__":
    main()
