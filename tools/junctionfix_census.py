"""JUNCTIONFIX census: junction-transit rows of the C4 TRAIN windows, junction_fix off vs ffill.

Usage (worktree root as CWD):
    PYTHONPATH=src python tools/junctionfix_census.py OUT.json [--config configs/c4_wm_v3.toml]
        [--data-root E:/research-dense] [--stride 1] [--limit N] [--seed 0]

Builds the TRAIN datasets of the config ONCE (TRAIN windows only: pack_v3 native + pack_dspeed +
the switching direct-mix TRAIN part; never a held-out / CAL / eval file) and decodes every
``stride``-th window twice, with the loader's junction_fix "off" and "ffill" (same start shift:
the global torch RNG is re-seeded per window and mode).  Counts per window (HDV = not CAV):

* raw lane changes as a ``present``-masked reader sees them (states one-hot argmax, both ends
  present; the GAPRULE census view), split by category and by whether an end is a junction row;
* the changes the per-car losses actually see (both ends ``loss_valid``);
* junction rows (present & not loss_valid & teacher lane -1 in OFF), frozen-s junction rows,
  t0 junction vehicles, CAV teacher rows with lane -1, loss_valid targets of slots that start
  on a junction, dropped t0 slots, aggregate (road_present) rows with recorded lane -1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

CATS = {(1, 0): "overtake_1to0", (0, 1): "return_0to1", (2, 1): "merge_2to1"}


def _loaders(dataset):
    """Every ``_PackWindowLoader`` under a (mixed) dataset."""
    out = []
    for src in getattr(dataset, "sources", None) or []:
        out += _loaders(src.dataset)
    loader = getattr(dataset, "_loader", None)
    if loader is not None and hasattr(loader, "junction_fix"):
        out.append(loader)
    return out


def _changes(lanes, valid):
    prev, nxt = lanes[:-1], lanes[1:]
    both = valid[:-1] & valid[1:]
    return both & (prev != nxt), prev, nxt


def window_stats(rec: dict, mode: str, junction_off: np.ndarray | None = None) -> Counter:
    c = Counter()
    st, teach = rec["states"], rec["cav_teacher"]
    pr, lv = rec["present"].astype(bool), rec["loss_valid"].astype(bool)
    cav = rec["is_cav"].astype(bool)
    t_n = st.shape[0]
    lanes = st[..., 1:4].argmax(-1)
    tlane = np.rint(teach[:t_n, :, 1]).astype(np.int64)
    jn = pr & ~lv if junction_off is None else junction_off & pr
    hdv_p = pr & ~cav
    hdv_v = lv & ~cav
    ch, prev, nxt = _changes(lanes, hdv_p)
    jn_end = jn[:-1] | jn[1:]
    for (a, b), name in CATS.items():
        sel = ch & (prev == a) & (nxt == b)
        c[f"raw_{name}"] += int(sel.sum())
        c[f"raw_{name}_junction_end"] += int((sel & jn_end).sum())
    other = ch & ~np.isin(prev * 3 + nxt, [3, 1, 7])
    c["raw_other"] += int(other.sum())
    c["raw_other_junction_end"] += int((other & jn_end).sum())
    c["raw_changes"] += int(ch.sum())
    c["raw_changes_junction_end"] += int((ch & jn_end).sum())
    chv, pv, nv = _changes(lanes, hdv_v)
    for (a, b), name in CATS.items():
        c[f"lossvalid_{name}"] += int((chv & (pv == a) & (nv == b)).sum())
    c["lossvalid_changes"] += int(chv.sum())
    c["lossvalid_hdv_rows"] += int(hdv_v[1:].sum())
    c["present_hdv_rows"] += int(hdv_p[1:].sum())
    c["junction_rows_hdv"] += int((jn & ~cav).sum())
    c["junction_rows_cav"] += int((jn & cav).sum())
    c["junction_rows_lossvalid"] += int((jn & lv).sum())
    frozen = np.zeros_like(jn)
    frozen[1:] = np.abs(st[1:, :, 0] - st[:-1, :, 0]) < 1e-4
    c["junction_rows_frozen_s"] += int((jn & frozen).sum())
    c["junction_rows_lane0_onehot"] += int((jn & (lanes == 0)).sum())
    c["junction_rows_lane_minus1_teacher"] += int((jn & (tlane < 0)).sum())
    c["cav_teacher_rows_lane_minus1"] += int((pr & cav & (tlane < 0)).sum())
    j0 = jn[0]
    c["t0_junction_hdv"] += int((j0 & ~cav[0]).sum())
    c["t0_junction_cav"] += int((j0 & cav[0]).sum())
    c["windows_with_t0_junction"] += int(j0.any())
    # loss_valid HDV targets of slots that start the rollout on a junction row
    c["lossvalid_targets_of_t0_junction_slots"] += int((hdv_v[1:] & j0[None]).sum())
    # fake-lane initial states: t0 slot whose t0 lane differs from its first loss_valid lane
    first = np.where(hdv_v.any(0), hdv_v.argmax(0), -1)
    k_idx = np.nonzero(j0 & (first > 0))[0]
    c["t0_junction_slots_start_lane_mismatch"] += int(
        sum(lanes[0, k] != lanes[first[k], k] for k in k_idx))
    # non-junction loss_valid rows with a frozen s (float16 spacing 2 m beyond |s| > 2048 m)
    fz_nj = frozen & lv & ~jn & (st[..., 4] > 0.2)
    c["nonjunction_lossvalid_frozen_s"] += int(fz_nj.sum())
    c["nonjunction_lossvalid_frozen_s_abs_s_gt_2048"] += int(
        (fz_nj & (np.abs(st[..., 0]) > 2048)).sum())
    if "road_present" in rec:
        rp = rec["road_present"].astype(bool)
        n = min(rp.shape[0], t_n)
        c["road_present_rows"] += int(rp[:n].sum())
        c["road_present_rows_lane_minus1"] += int((rp[:n] & (tlane[:n] < 0)).sum())
    c["windows"] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--config", default="configs/c4_wm_v3.toml")
    ap.add_argument("--data-root", default="E:/research-dense")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from wmlane_smoke import point_data

    if args.data_root:
        point_data(Path(args.data_root))
    from wms.cli.assemble import build_datasets
    from wms.config.load import load_train_config

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    cfg = load_train_config(args.config, ["runtime.device=cpu", 'data.junction_fix="off"'])
    scratch = Path(args.out).with_suffix("").as_posix() + "_ds"
    train, _, _ = build_datasets(cfg, Path(scratch))
    loaders = _loaders(train)
    n = len(train) if args.limit is None else min(len(train), int(args.limit))
    totals = {"off": Counter(), "ffill": Counter()}
    by_source = {}
    t_start = time.time()
    for w in range(int(args.offset), n, int(args.stride)):
        recs = {}
        for mode in ("off", "ffill"):
            for loader in loaders:
                loader.junction_fix = mode
            torch.manual_seed(int(args.seed) * 100003 + w)
            recs[mode] = train[w]
        off = recs["off"]
        jn_off = off["present"].astype(bool) & ~off["loss_valid"].astype(bool)
        src = str(off.get("mix_source", "original"))
        for mode in ("off", "ffill"):
            c = window_stats(recs[mode], mode, junction_off=jn_off)
            if mode == "ffill":
                c["t0_slots_dropped"] += int((off["present"][0] & ~recs[mode]["present"][0]).sum())
                c["lossvalid_rows_removed"] += int(
                    (off["loss_valid"] & ~recs[mode]["loss_valid"]).sum())
                c["lossvalid_rows_added"] += int(
                    (~off["loss_valid"] & recs[mode]["loss_valid"]).sum())
                same_v = np.array_equal(off["states"][..., 4], recs[mode]["states"][..., 4])
                c["windows_speed_changed"] += int(not same_v)
            totals[mode].update(c)
            by_source.setdefault(src, {"off": Counter(), "ffill": Counter()})[mode].update(c)
        if w % 100 == 0:
            print(f"window {w}/{n} {time.time() - t_start:.0f}s", flush=True)
    out = {"config": args.config, "stride": args.stride, "offset": args.offset,
           "n_windows_total": len(train), "seed": args.seed,
           "totals": {m: dict(sorted(c.items())) for m, c in totals.items()},
           "by_source": {s: {m: dict(sorted(c.items())) for m, c in d.items()}
                         for s, d in by_source.items()}}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["totals"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
