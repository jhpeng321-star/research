"""Extract per-driver observable 60 s histories from FIT-world episode tapes.

Input: one or more SUMO episode directories carrying ``episode.npz`` with schema
``c4_sumo_episode/1`` (SUMO_NOTES.md section 3; produced by
``traffic_shepherd.c4.collect``).  For the WM-PHYS rebuild the inputs are the
S-WMDATA FIT block 969400-969431, native arm, on cpuQ at
``/home/ubuntu/wmdata20261031/jobs/s<seed>_native/deploy/episode/``.

Output: one compact ``setdata_hdv_histories/1`` npz + JSON side manifest; the
layout is documented in tools/rbsetdata/SCHEMA.md.

Frame facts (traffic_shepherd/c4/geometry.py:9-12, SUMO_NOTES.md section 2):
lane 0 = main left, 1 = main right, 2 = ramp/acceleration lane; s is the
merge-relative coordinate (merge zone 0..75.44 m, upstream negative); junction
internal lanes have lane -1 with s carried (``junction`` flag) and
``lane_carry`` holding the last modelled lane.  The tape dt is 0.5 s
(``--step-length 0.5`` in the recorded launch command); t0 = warmup_s = 300 s;
the history window is [t0 - 60, t0].

Usage::

    python tools/rbsetdata/extract_hdv_histories.py \
        --episodes <dir-with-episode-npz> [...] --out hist.npz \
        [--manifest hist_manifest.json] [--t0 300] [--window 60]
        [--min-ticks 20] [--include-cav]

No SUMO run, numpy only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "setdata_hdv_histories/1"
EPISODE_SCHEMA = "c4_sumo_episode/1"
#: nominal speed limit of the C4 road (wms/data/c4_episodes.py:69
#: SPEED_LIMIT_MPS); the desSpeed cap of the IDM likelihood, not a fitted
#: quantity.
SPEED_LIMIT_MPS = 31.29
NEIGHBOR_MAX_GAP_M = 200.0
REQUIRED_KEYS = ("time_s", "s_m", "v_mps", "a_mps2", "length_true_m", "lane",
                 "lane_carry", "present", "is_cav", "junction", "vid_index",
                 "vehicle_ids", "meta_json")

TICK_COLS = ("seg_id", "t", "s_m", "v_mps", "a_mps2", "lane", "eff_lane",
             "junction", "length_m", "lead_gap_m", "lead_dv_mps",
             "follow_gap_m", "follow_dv_mps", "left_lead_gap_m",
             "left_lead_dv_mps", "left_follow_gap_m", "right_lead_gap_m",
             "right_lead_dv_mps", "right_follow_gap_m")
N_LANES = 3


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_episode(path: Path) -> dict:
    """Load + validate one ``episode.npz`` (fail closed on schema drift)."""
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in REQUIRED_KEYS if k not in z.files]
        if missing:
            raise ValueError(f"{path}: episode.npz missing keys {missing}")
        ep = {k: z[k] for k in z.files}
    meta = json.loads(str(ep["meta_json"]))
    if str(meta.get("schema")) != EPISODE_SCHEMA:
        raise ValueError(f"{path}: schema {meta.get('schema')!r} != {EPISODE_SCHEMA}")
    if abs(float(meta.get("dt_s", 0.5)) - 0.5) > 1e-9:
        raise ValueError(f"{path}: unsupported dt_s {meta.get('dt_s')}")
    return ep


def world_id(meta: dict) -> str:
    """``w<seed>m<demand>p<pct>t<warmup>`` (c4_episodes.py::world_id convention)."""
    pct = float(meta["penetration"]) * 100.0
    return (f"w{int(meta['seed'])}m{int(round(float(meta['demand_vph'])))}"
            f"p{int(round(pct)):02d}t{int(round(float(meta['warmup_s'])))}")


def _lane_observers(lane_sel: np.ndarray, s_row: np.ndarray, v_row: np.ndarray,
                    len_row: np.ndarray):
    """Sorted-by-s arrays of one lane's present vehicles (or None)."""
    idx = np.flatnonzero(lane_sel)
    if idx.size == 0:
        return None
    order = idx[np.argsort(s_row[idx], kind="stable")]
    return s_row[order], v_row[order], len_row[order]


def _neighbors_per_tick(eff_lane: np.ndarray, s: np.ndarray, v: np.ndarray,
                        length: np.ndarray, present: np.ndarray):
    """Leader/follower net gaps: same lane + adjacent lanes, every tick.

    Net bumper gap to the leader: ``s_lead - s_ego - length_lead`` (the SUMO
    inter-vehicle gap; cf. BELIEF_NOTES.md Detection).  The follower gap is
    ``s_ego - s_follower - length_ego``.  Neighbor search uses the effective
    lane (``lane`` if modelled else ``lane_carry``; junction vehicles keep
    their carried s).  NaN = no such neighbor within NEIGHBOR_MAX_GAP_M (or
    the adjacent lane does not exist).
    """
    t_count, k = present.shape
    nan = np.float32(np.nan)
    out = {name: np.full((t_count, k), nan, np.float32) for name in (
        "lead_gap", "lead_dv", "follow_gap", "follow_dv",
        "left_lead_gap", "left_lead_dv", "left_follow_gap",
        "right_lead_gap", "right_lead_dv", "right_follow_gap")}
    max_gap = np.float32(NEIGHBOR_MAX_GAP_M)
    for t in range(t_count):
        s_row, v_row, len_row = s[t], v[t], length[t]
        lanes = {}
        for lane in range(N_LANES):
            lanes[lane] = _lane_observers(present[t] & (eff_lane[t] == lane),
                                          s_row, v_row, len_row)
        for lane in range(N_LANES):
            me = lanes[lane]
            if me is None:
                continue
            s_me, v_me, len_me = me
            me_ids = np.flatnonzero(present[t] & (eff_lane[t] == lane))[
                np.argsort(s_row[np.flatnonzero(present[t] & (eff_lane[t] == lane))],
                           kind="stable")]
            for other, prefix in ((lane, ""), (lane - 1, "left_"),
                                  (lane + 1, "right_")):
                if other < 0 or other >= N_LANES:
                    continue
                tgt = lanes[other]
                if tgt is None:
                    continue
                s_tg, v_tg, len_tg = tgt
                pos = np.searchsorted(s_tg, s_me, side="right")
                # leader: first target-lane vehicle ahead of me
                ok = pos < s_tg.size
                idx = np.minimum(pos, s_tg.size - 1)
                gap = s_tg[idx] - s_me - len_tg[idx]
                dv = v_tg[idx] - v_me
                good = ok & (gap <= max_gap)
                if good.any():
                    dst = me_ids[good]
                    out[f"{prefix}lead_gap"][t][dst] = gap[good]
                    out[f"{prefix}lead_dv"][t][dst] = dv[good]
                # follower: last target-lane vehicle behind me (its speed is
                # only recorded for the same lane: TICK_COLS has no
                # {left,right}_follow_dv column)
                j = pos - 1
                okf = j >= 0
                jdx = np.maximum(j, 0)
                gapf = s_me - s_tg[jdx] - len_me
                goodf = okf & (gapf <= max_gap)
                if goodf.any():
                    dst = me_ids[goodf]
                    out[f"{prefix}follow_gap"][t][dst] = gapf[goodf]
                    if prefix == "":
                        out["follow_dv"][t][dst] = (v_tg[jdx] - v_me)[goodf]
    return out


def extract_world(path: Path, t0_s: float, window_s: float, min_ticks: int,
                  include_cav: bool) -> dict:
    """One episode -> flat per-segment tick arrays + events (see SCHEMA.md)."""
    ep = load_episode(path)
    meta = json.loads(str(ep["meta_json"]))
    time_s = np.asarray(ep["time_s"], np.float64)
    t_lo, t_hi = t0_s - window_s, t0_s
    sel = np.flatnonzero((time_s >= t_lo - 1e-9) & (time_s <= t_hi + 1e-9))
    if sel.size < 2:
        raise ValueError(f"{path}: no ticks in [{t_lo}, {t_hi}]")
    ticks = np.arange(sel[0], sel[-1] + 1)
    present = np.asarray(ep["present"], bool)[ticks]
    lane = np.asarray(ep["lane"], np.int64)[ticks]
    lane_carry = np.asarray(ep.get("lane_carry", lane), np.int64)[ticks]
    eff_lane = np.where(lane >= 0, lane, lane_carry)
    s = np.asarray(ep["s_m"], np.float32)[ticks]
    v = np.asarray(ep["v_mps"], np.float32)[ticks]
    a = np.asarray(ep["a_mps2"], np.float32)[ticks]
    length = np.asarray(ep["length_true_m"], np.float32)[ticks]
    junction = np.asarray(ep["junction"], bool)[ticks]
    vid = np.asarray(ep["vid_index"], np.int64)[ticks]
    is_cav = np.asarray(ep["is_cav"], bool)[ticks]
    veh_ids = np.asarray(ep["vehicle_ids"])
    nb = _neighbors_per_tick(eff_lane, s, v, length, present)

    t_rel = (time_s[ticks] - t_lo).astype(np.float32)
    tick_rows: list[np.ndarray] = []
    segs: list[dict] = []
    lc_rows: list[dict] = []
    mg_rows: list[dict] = []

    for vi, veh_id in enumerate(veh_ids):
        vsel = present & (vid == vi)
        if not include_cav and not (vsel & ~is_cav).any():
            continue
        for k in np.flatnonzero(vsel.any(axis=0)):
            tick_idx = np.flatnonzero(vsel[:, k])
            if tick_idx.size == 0:
                continue
            # contiguous runs (a present-gap = dropped tick or exit: new segment)
            breaks = np.flatnonzero(np.diff(tick_idx) > 1) + 1
            for run_full in np.split(tick_idx, breaks):
                run = run_full[(t_rel[run_full] >= -1e-6)
                               & (t_rel[run_full] <= window_s + 1e-6)]
                if run.size < min_ticks:
                    continue
                seg_id = len(segs)
                enters = bool(t_rel[run[0]] > 1e-6)
                first_lane = int(eff_lane[run[0], k])
                lc_count = 0
                n_merge = 0
                for t in run.tolist():
                    tick_rows.append(np.array(
                        [seg_id, t_rel[t], s[t, k], v[t, k], a[t, k],
                         lane[t, k], eff_lane[t, k], int(junction[t, k]),
                         length[t, k]]
                        + [nb[name][t, k] for name in (
                            "lead_gap", "lead_dv", "follow_gap", "follow_dv",
                            "left_lead_gap", "left_lead_dv", "left_follow_gap",
                            "right_lead_gap", "right_lead_dv",
                            "right_follow_gap")], np.float32))
                # lane-change events (modelled-lane moves, not junction transits)
                for pos in range(1, run.size):
                    t0p, t1p = int(run[pos - 1]), int(run[pos])
                    if junction[t0p, k] or junction[t1p, k]:
                        continue
                    if lane[t0p, k] < 0 or lane[t1p, k] < 0:
                        continue
                    if lane[t1p, k] != lane[t0p, k]:
                        lc_count += 1
                        lc_rows.append({
                            "seg": seg_id, "t": float(t_rel[t1p]),
                            "from": int(lane[t0p, k]), "to": int(lane[t1p, k]),
                            "s": float(s[t1p, k]), "v": float(v[t1p, k]),
                            "gap_lead_pre": float(nb["lead_gap"][t0p, k]),
                            "dv_lead_pre": float(nb["lead_dv"][t0p, k]),
                            "gap_lead_post": float(nb["lead_gap"][t1p, k]),
                            "gap_follow_pre": float(nb["follow_gap"][t0p, k]),
                            "is_ramp_to_main": int(lane[t0p, k] == 2 and lane[t1p, k] < 2),
                        })
                # merge / junction-exit events: back into a modelled lane (the
                # only junction is the s~0 ramp merge: ramp lane 2 ->
                # acceleration lane 2, or ramp -> mainline lanes 0/1)
                for pos in range(1, run.size):
                    t0p, t1p = int(run[pos - 1]), int(run[pos])
                    if not (junction[t0p, k] and not junction[t1p, k]
                            and lane[t1p, k] >= 0):
                        continue
                    frm = int(lane_carry[t0p, k])
                    to = int(lane[t1p, k])
                    if frm == 2:
                        n_merge += 1
                    mg_rows.append({
                        "seg": seg_id, "t": float(t_rel[t1p]),
                        "s": float(s[t1p, k]), "v": float(v[t1p, k]),
                        "from": frm, "to": to,
                        "is_ramp_merge": int(frm == 2),
                        "gap_lead": float(nb["lead_gap"][t1p, k]),
                        "dv_lead": float(nb["lead_dv"][t1p, k]),
                    })
                segs.append({
                    "seg": seg_id, "veh_id": str(veh_id),
                    "is_cav": int(bool(is_cav[run[0], k])),
                    "t0": float(t_rel[run[0]]), "t1": float(t_rel[run[-1]]),
                    "nticks": int(run.size), "enters_window": int(enters),
                    "first_lane": first_lane, "n_lc": lc_count,
                    "n_merge": n_merge,
                })

    ticks_arr = (np.stack(tick_rows) if tick_rows
                 else np.zeros((0, len(TICK_COLS)), np.float32))
    return {
        "world": world_id(meta), "seed": int(meta["seed"]),
        "demand_vph": int(meta["demand_vph"]),
        "penetration": float(meta["penetration"]), "t0_s": float(t0_s),
        "ticks": ticks_arr, "segs": segs, "lc": lc_rows, "mg": mg_rows,
        "source": str(path), "sha256": _sha256(path),
    }


def _cols(rows: list[tuple], spec) -> dict:
    names = [n for n, _ in spec]
    if not rows:
        return {n: np.zeros(0, dt) for n, dt in spec}
    cols = list(zip(*rows))
    return {n: np.asarray(col, dt) for n, col, (_, dt) in zip(names, cols, spec)}


def build_npz(extractions: list[dict], window_s: float) -> tuple[dict, dict]:
    """Concatenate per-world extractions into the flat npz payload + manifest."""
    segs, ticks, lc, mg, worlds = [], [], [], [], []
    seg_base = 0
    for w_i, ex in enumerate(extractions):
        w_idx = len(worlds)
        worlds.append((ex["world"], ex["seed"], ex["demand_vph"],
                       ex["penetration"], ex["source"], ex["sha256"]))
        for seg in ex["segs"]:
            segs.append((seg_base + seg["seg"], w_idx, seg["veh_id"],
                         seg["is_cav"], seg["t0"], seg["t1"], seg["nticks"],
                         seg["enters_window"], seg["first_lane"],
                         seg["n_lc"], seg["n_merge"]))
        tk = ex["ticks"].copy()
        if tk.size:
            tk[:, 0] += seg_base
        if tk.shape[0]:
            ticks.append(tk)
        for e in ex["lc"]:
            lc.append((seg_base + e["seg"], e["t"], e["from"], e["to"], e["s"],
                       e["v"], e["gap_lead_pre"], e["dv_lead_pre"],
                       e["gap_lead_post"], e["gap_follow_pre"],
                       e["is_ramp_to_main"]))
        for e in ex["mg"]:
            mg.append((seg_base + e["seg"], e["t"], e["s"], e["v"], e["from"],
                       e["to"], e["is_ramp_merge"], e["gap_lead"], e["dv_lead"]))
        seg_base += len(ex["segs"])

    seg_spec = [("seg_id", np.int32), ("world", np.int32), ("veh_id", "U12"),
                ("is_cav", np.uint8), ("t0", np.float32), ("t1", np.float32),
                ("nticks", np.int32), ("enters_window", np.uint8),
                ("first_lane", np.int8), ("n_lc", np.int32),
                ("n_merge", np.uint8)]
    tick_spec = [(n, np.float32) for n in TICK_COLS]
    # LC/merge event columns are prefixed lc_/mg_ so they never collide with
    # the per-tick columns (a flat npz has one namespace).
    lc_spec = [("lc_seg_id", np.int32), ("lc_t", np.float32),
               ("lc_from_lane", np.int8), ("lc_to_lane", np.int8),
               ("lc_s_m", np.float32), ("lc_v_mps", np.float32),
               ("lc_gap_lead_pre_m", np.float32),
               ("lc_dv_lead_pre_mps", np.float32),
               ("lc_gap_lead_post_m", np.float32),
               ("lc_gap_follow_pre_m", np.float32),
               ("lc_is_ramp_to_main", np.uint8)]
    mg_spec = [("mg_seg_id", np.int32), ("mg_t", np.float32),
               ("mg_s_m", np.float32), ("mg_v_mps", np.float32),
               ("mg_from_lane", np.int8), ("mg_to_lane", np.int8),
               ("mg_is_ramp_merge", np.uint8), ("mg_gap_lead_m", np.float32),
               ("mg_dv_lead_mps", np.float32)]
    world_spec = [("world_id", "U24"), ("seed", np.int64),
                  ("demand_vph", np.int64), ("penetration", np.float64),
                  ("source", "U256"), ("episode_sha256", "U64")]

    payload: dict = {"schema": np.array(SCHEMA)}
    payload.update(_cols(segs, seg_spec))
    payload.update(_cols([], tick_spec) if not ticks else dict(
        zip([n for n, _ in tick_spec],
            [np.ascontiguousarray(c) for c in np.concatenate(ticks, axis=0).T])))
    payload.update(_cols(lc, lc_spec))
    payload.update(_cols(mg, mg_spec))
    payload.update(_cols(worlds, world_spec))
    payload["window_s"] = np.array(window_s, np.float32)
    payload["speed_limit_mps"] = np.array(SPEED_LIMIT_MPS, np.float32)

    manifest = {
        "schema": SCHEMA, "n_worlds": len(worlds), "n_segments": len(segs),
        "n_ticks": int(sum(t.shape[0] for t in ticks)),
        "n_lc_events": len(lc), "n_merge_events": len(mg),
        "worlds": [{"world_id": w[0], "seed": int(w[1]), "demand_vph": int(w[2]),
                    "penetration": float(w[3]), "source": w[4], "sha256": w[5]}
                   for w in worlds],
    }
    return payload, manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="extract per-driver 60 s histories")
    ap.add_argument("--episodes", nargs="+", required=True,
                    help="episode dirs (each containing episode.npz)")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--manifest", default=None,
                    help="output manifest .json (default: <out>_manifest.json)")
    ap.add_argument("--t0", type=float, default=None,
                    help="history end time s (default: each tape's warmup_s)")
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--min-ticks", type=int, default=20,
                    help="minimum ticks in window per segment (20 = 10 s)")
    ap.add_argument("--include-cav", action="store_true",
                    help="keep CAV segments too (default HDV only)")
    args = ap.parse_args()

    extractions = []
    for ep_dir in args.episodes:
        path = Path(ep_dir)
        npz = path if path.is_file() else path / "episode.npz"
        meta = json.loads(str(load_episode(npz)["meta_json"]))
        t0 = args.t0 if args.t0 is not None else float(meta["warmup_s"])
        ex = extract_world(npz, t0, args.window, args.min_ticks, args.include_cav)
        print(f"{ex['world']}: {len(ex['segs'])} segments, "
              f"{ex['ticks'].shape[0]} ticks, {len(ex['lc'])} LC, "
              f"{len(ex['mg'])} junction-exit events")
        extractions.append(ex)

    payload, manifest = build_npz(extractions, args.window)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    manifest_path = (Path(args.manifest) if args.manifest
                     else out.with_name(out.stem + "_manifest.json"))
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes) and {manifest_path}")


if __name__ == "__main__":
    main()
