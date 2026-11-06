"""R13T recorded quantile table for loss.accel_tail (CPU, recorded arrays only, no model, no SUMO).

For every episode of the requested split role (default ``train``) in the given C4 episode roots, after the
episode warm-up (training windows start there), every recorded HDV row at time t whose slot is present on
the road at t and t+1 with the same vehicle id and whose lane at t is in ``--lanes`` (default 1) gives

    a_t = (v_{t+1} - v_t) / dt                     (the braking terms' / BRAKE_DIAG's acceleration)
    |jerk|_t = |a_t - a_{t-1}| / dt                (same vehicle also present at t-1)

and a cell id from the recorded state at t with the brake-persistence convention
(``wms.train.losses.brake_persist.leader_cells``: gap x closing x leader decel, 4 x 4 x 4, max gap 40 m).
The table stores, per variable, the pooled and per-cell row counts and quantiles at the levels
u = 1/100 .. 99/100 (``numpy.quantile``, linear = ``torch.quantile``'s default), plus provenance (roots,
manifest sha256, episode ids, split roles).  The loss refuses a table whose split roles are not
exactly ["train"]; ``--split val`` exists only to build a PROVISIONAL local table for offline checks.

    python tools/r13t_tail_table.py --roots $L/data/v5_native $L/data/genrand_plain ... --out docs/r13t_tail_table.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from wms.train.losses.accel_tail import TABLE_SCHEMA  # noqa: E402
from wms.train.losses.brake_persist import _N_CELLS, leader_cells  # noqa: E402

LEVELS = [m / 100.0 for m in range(1, 100)]
MAX_GAP_M = 40.0          # BrakePersistConfig.max_gap_m (the event-balance cell convention)
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")


def _guard(path: str) -> None:
    low = str(path).replace("\\", "/").lower()
    if any(token in low for token in FORBIDDEN):
        raise SystemExit(f"refused restricted path: {path}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_rows(epdir: Path, lanes: tuple[int, ...], warmup_s: float, chunk: int = 200) -> dict:
    kin = np.load(epdir / "veh_kin.npy").astype(np.float32)
    present = np.load(epdir / "veh_mask.npy") > 0
    vid = np.load(epdir / "vid.npy")
    hdv = np.load(epdir / "is_cav.npy") == 0
    time_s = np.load(epdir / "time_s.npy")
    dt = float(np.round(np.median(np.diff(time_s)), 6))
    on_road = kin[..., 1:4].sum(-1) > 0
    valid = present & on_road & (vid >= 0)
    lane = np.where(on_road, kin[..., 1:4].argmax(-1), -1)
    n_t = kin.shape[0]
    cells = np.full(valid.shape, -1, np.int64)
    with torch.no_grad():
        for t0 in range(0, n_t, chunk):
            sl = slice(t0, min(n_t, t0 + chunk))
            k = torch.from_numpy(kin[sl])
            cells[sl] = leader_cells(k[..., 0], k[..., 1:4], k[..., 4], k[..., 5],
                                     torch.from_numpy(valid[sl]), max_gap_m=MAX_GAP_M).numpy()
    same_next = valid[:-1] & valid[1:] & (vid[:-1] == vid[1:])            # t -> t+1
    in_lane = np.isin(lane[:-1], lanes)
    start = int(np.ceil(warmup_s / dt - 1e-9))
    after = (np.arange(n_t - 1) >= start)[:, None]
    rows = same_next & hdv[:-1] & in_lane & after                          # [T-1, K] at t
    accel = (kin[1:, :, 4] - kin[:-1, :, 4]) / dt                          # a_t
    jrows = rows[1:] & rows[:-1] & (vid[1:-1] == vid[:-2])                 # at t = 1..T-2
    jerk = np.abs(accel[1:] - accel[:-1]) / dt
    return {"dt": dt, "accel": accel[rows], "accel_cells": cells[:-1][rows],
            "abs_jerk": jerk[jrows], "abs_jerk_cells": cells[1:-1][jrows]}


def quantile_block(values: np.ndarray, cells: np.ndarray) -> dict:
    out = {"pooled": {"n": int(values.size), "q": np.quantile(values, LEVELS).tolist()}, "cells": {}}
    for c in range(_N_CELLS):
        v = values[cells == c]
        if v.size:
            out["cells"][str(c)] = {"n": int(v.size), "q": np.quantile(v, LEVELS).tolist()}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True, help="C4 episode roots (with C4_EPISODES_MANIFEST.json)")
    p.add_argument("--split", default="train", choices=("train", "val"))
    p.add_argument("--lanes", type=int, nargs="+", default=[1])
    p.add_argument("--allow-missing", action="store_true",
                   help="skip manifest episodes of the split that are not on disk (local subsets)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    lanes = tuple(int(x) for x in args.lanes)
    acc = {"accel": [], "accel_cells": [], "abs_jerk": [], "abs_jerk_cells": []}
    provenance, dts, episodes, missing = [], set(), [], 0
    for root in args.roots:
        _guard(root)
        manifest = Path(root) / "C4_EPISODES_MANIFEST.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        chosen = [e for e in data["episodes"] if e.get("split_role") == args.split]
        provenance.append({"root": str(Path(root).resolve()), "manifest_sha256": _sha(manifest),
                           "n_split_in_manifest": len(chosen)})
        for entry in chosen:
            epdir = Path(entry["dir"])
            if not epdir.is_absolute():
                epdir = Path(root) / epdir
            _guard(str(epdir))
            if not (epdir / "veh_kin.npy").is_file():
                missing += 1
                if not args.allow_missing:
                    raise SystemExit(f"missing episode {epdir} (use --allow-missing for a local subset)")
                continue
            rows = episode_rows(epdir, lanes, float(entry["warmup_s"]))
            dts.add(rows.pop("dt"))
            for key in acc:
                acc[key].append(rows[key])
            episodes.append(entry["episode"])
            print(f"{entry['episode']}: accel rows {rows['accel'].size}", flush=True)
    if len(dts) != 1:
        raise SystemExit(f"episodes disagree on dt: {sorted(dts)}")
    table = {"schema": TABLE_SCHEMA, "levels": LEVELS, "max_gap_m": MAX_GAP_M, "dt_s": dts.pop(),
             "lanes": list(lanes), "split_roles": [args.split],
             "provisional": args.split != "train" or bool(missing),
             "cell_layout": "brake_persist.leader_cells 4 gap x 4 closing x 4 leader-decel",
             "definitions": {"accel": "(v[t+1]-v[t])/dt, same vid present on the road at t and t+1, HDV, lane(t) in lanes, t >= warmup",
                             "abs_jerk": "|a[t]-a[t-1]|/dt, both accel rows valid and same vid"},
             "sources": provenance, "n_episodes": len(episodes), "n_missing_skipped": missing,
             "episodes": episodes}
    for var in ("accel", "abs_jerk"):
        table[var] = quantile_block(np.concatenate(acc[var]), np.concatenate(acc[var + "_cells"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(table, indent=1), encoding="utf-8")
    pooled = table["accel"]["pooled"]
    print(json.dumps({"n_episodes": len(episodes), "accel_rows": pooled["n"],
                      "accel_p1_p50_p99": [pooled["q"][0], pooled["q"][49], pooled["q"][98]],
                      "abs_jerk_p50_p99": [table["abs_jerk"]["pooled"]["q"][49], table["abs_jerk"]["pooled"]["q"][98]],
                      "cells_accel": len(table["accel"]["cells"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
