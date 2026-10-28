"""S-EXECPARITY3: effect of the SUMO lane/link leader rule (C4_SUMO_LEADER_RULE)
on ALL vehicles' WM2 dynamics, on three FIT event worlds (E3 split 'fit'; never
validation/pairs300/calibration).

Reuses the ERRSRC runtime (WM2 b8797242, entrant+ramp ON, config
c4_wm_loop14_r16_ft) fetched read-only from cpuP into ``_parity3_data``, with
exactly two files overlaid from THIS branch: ``wms/physics/neighbours.py`` and
the new ``wms/physics/lane_chain.py`` (flag-gated; OFF is bitwise the runtime's
frozen body). Per world and flag value (OFF default, ON), one 600-step rollout
with the S-CAPFIX diagnose seed recipe (torch seed 140014+7919*gi+1000003*draw,
settings seed 140014+draw*1000003, paired_random merge_pivot keys), per-tick
capture, then WM-vs-real measurements around the REAL breakdown onset:

  - hard-braking rows (|a| <= -3.0, the composer envelope lower bound; same
    constant as S-CAPFIX), total / near-merge / on rows whose LEGACY corridor
    leader is a <=2 m phantom that the chain rule pushes >2 m away;
  - merge-exit discharge at s=75.44 (errsrc corrected continuous-identity
    crossings + 60 s symmetric window): real vs WM-OFF vs WM-ON at the real
    onset (and the WM's own onset);
  - per-tick dynamics error vs the real tape, identity-matched by slot:
    one-step acceleration MAE (whole window and the first 60 s) and speed/pos
    MAE at horizons 10/30/60/120/300 s.

CPU only; zero SUMO executions; no validation/pairs300/calibration panel data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT / "_parity3_data"
SECTION = 75.44
DT = 0.5
HARD_BRAKE_MPS2 = -3.0
WM_SHA = "b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436"
OVERLAY = ("src/wms/physics/neighbours.py", "src/wms/physics/lane_chain.py")


def setup():
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", WMS_SPEED_COMPACT_SLOTS="1",
                      WMS_SKIP_PIVOT_DIAG="1", WMS_UNIT_SEED="1", LRDIAG_TRACE_OFF="1")
    sys.path[:0] = [str(HOME / "runtime"), str(HOME / "runtime/src")]


def overlay_receipt() -> dict:
    """Overlay the two S-EXECPARITY3 physics files; SHA-bind everything.

    The pristine fetched runtime copy of each target is preserved as
    ``<name>.orig_pristine`` on first overlay; a second run must be a no-op.
    """
    receipt = {}
    for rel in OVERLAY:
        src = ROOT / rel
        dst = HOME / "runtime" / rel
        assert src.is_file()
        dst.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        entry = {"sha256": digest}
        pristine = dst.with_suffix(".orig_pristine")
        if dst.exists():
            if hashlib.sha256(dst.read_bytes()).hexdigest() != digest:
                if not pristine.exists():
                    pristine.write_bytes(dst.read_bytes())
                    entry["first_overlay"] = True
                shutil.copy2(src, dst)
            if pristine.exists():
                entry["pristine_sha256"] = hashlib.sha256(pristine.read_bytes()).hexdigest()
        else:
            shutil.copy2(src, dst)
            entry["copied"] = True
        receipt[rel] = entry
    ck = HOME / "wm2" / "wm_s.pt"
    assert hashlib.sha256(ck.read_bytes()).hexdigest() == WM_SHA
    return receipt


# ------------------------------------------------------------- numpy geometry

def chain_offsets(lane, s):
    """wms.physics.lane_chain.chain_s in numpy (int lanes, float s)."""
    lane = np.asarray(lane)
    s = np.asarray(s, dtype=np.float64)
    entry = (lane != 2) & (s > 0.)
    exit_ = (lane != 2) & (s > 75.44)
    ramp = (lane == 2) & (s > 0.)
    return s + entry * 3.00 + exit_ * 8.00 + ramp * 2.93


def leader_rule(s, v, lane, present, length, physics, *, chain=False, junc=None):
    """Exact numpy replay of wms.physics.neighbours.lead_neighbour (+ index).

    ``lane`` is the WM frame's argmax lane (empty one-hot rows read as lane 0,
    the frozen legacy quirk); ``junc`` (optional) marks junction-transit rows
    (empty lane one-hot). chain=False is the frozen legacy corridor rule
    (junction rows participate as their argmax lane, == S-CAPFIX diagnose);
    chain=True adds the lane-chain connector offsets and treats junction rows
    as non-candidates/non-egos (they are on a connector).
    """
    min_gap = 0.3
    free = float(physics.free_road_gap_m if physics.open_road else 1.0e3)
    T, K = s.shape
    gap = np.full_like(s, free)
    lead_v = v.copy()
    has = np.zeros(s.shape, bool)
    index = np.full(s.shape, -1, np.int64)
    merge_lane = int(getattr(physics, "merge_lane", 2))
    s1 = float(getattr(physics, "merge_zone_s1_m", 0.0))
    for t in range(T):
        p = present[t]
        if chain and junc is not None:
            p = p & ~junc[t]
        if not p.any():
            continue
        idx = np.flatnonzero(p)
        ln = lane[t, idx]
        ss, vv, ll = s[t, idx], v[t, idx], length[t, idx]
        cc = chain_offsets(ln, ss) if chain else ss
        ds = cc[None, :] - cc[:, None]
        same = ln[None, :] == ln[:, None]
        ahead = same & (ds > 0)
        np.fill_diagonal(ahead, False)
        bumper = np.where(ahead, ds - ll[None, :], np.inf)
        j = bumper.argmin(-1)
        g = bumper[np.arange(len(idx)), j]
        h = np.isfinite(g)
        if chain:
            end_gap = np.clip(s1 + 2.93 - cc, min_gap, None)
        else:
            end_gap = np.clip(s1 - ss, min_gap, None)
        obst = (ln == merge_lane) & (end_gap < np.where(h, g, np.inf))
        gap_t = np.where(obst, end_gap, np.where(h, np.clip(g, min_gap, None), free))
        has_t = obst | h
        lv_t = np.where(obst, 0.0, np.where(h, vv[j], vv))
        sel = obst | h
        gi = np.where(obst, -2, np.where(h, idx[j], -1))
        rows = np.flatnonzero(sel)
        gap[t, idx[rows]] = gap_t[rows]
        lead_v[t, idx[rows]] = lv_t[rows]
        has[t, idx[rows]] = True
        index[t, idx[rows]] = gi[rows]
    return gap, lead_v, has, index


def crossings_corrected(pos, present, lane, lane_valid, ident, section=SECTION):
    last = {}
    events = []
    for t in range(len(pos)):
        cur = {int(ident[t, j]): int(j) for j in np.flatnonzero(present[t])}
        for identity in set(last) - set(cur):
            last.pop(identity, None)
        for identity, j in cur.items():
            if lane_valid[t, j]:
                if identity in last:
                    pt, ps = last[identity]
                    if ps < section <= pos[t, j] and lane[t, j] < 2:
                        frac = (section - ps) / (float(pos[t, j]) - ps)
                        events.append((pt + frac * (t - pt)) * DT)
                last[identity] = (t, float(pos[t, j]))
    return sorted(events)


def window_stats(times, onset, horizon=300.):
    if onset is None:
        return {"available": False}
    duration = min(60., float(onset), horizon - float(onset))
    if duration <= 0:
        return {"available": False, "reason": "zero_window"}
    t = np.asarray(times)
    pre = int(((t >= onset - duration) & (t < onset)).sum())
    post = int(((t >= onset) & (t < onset + duration)).sum())
    return {"available": True, "duration_s": duration, "onset_s": float(onset),
            "pre_count": pre, "post_count": post,
            "pre_veh_h": pre * 3600 / duration, "post_veh_h": post * 3600 / duration,
            "drop_veh_h": (pre - post) * 3600 / duration}


def rollout_group(gi, draw, steps, cohort):
    import torch
    torch.set_num_threads(1)
    from dataclasses import replace
    from tools.r10_attribution import load_model
    from tools.lrdiag import replay
    from tools.errsrc import run as errsrc_run
    from tools.errsrc.run import fit_source
    from wms.data.collate import collate
    from wms.train.inputs import prepare_inputs

    replay.HOME = HOME
    errsrc_run.HOME = HOME      # fit_source reads this module-level HOME
    config = HOME / "runtime/configs/c4_wm_loop14_r16_ft.toml"
    ck = HOME / "wm2/wm_s.pt"
    cfg, settings, model, report = load_model(str(config), str(ck),
        ("data.entries_v2=true", "loss.paired_diff.couple_randomness=true"))
    assert not report["unused"] and not report["new"], report
    settings = replace(settings, seed=140014 + draw * 1000003)
    model.emit_physics_accel = True

    source, manifest = fit_source(HOME, None)
    g = source.groups[gi]
    assert len(g["rows"]) == 1
    records = source.group_records(gi, g["rows"], base_row=0)
    batch = collate(records)
    inp = prepare_inputs(batch, model, settings, torch.device("cpu"), steps)
    assert inp.paired_random is not None

    entry = manifest["episodes"][gi]
    raw = HOME / "fitdata" / entry["dir"]
    t0 = int(entry["t0_index"])
    kin = np.load(raw / "veh_kin.npy", mmap_mode="r")
    mask_raw = np.load(raw / "veh_mask.npy", mmap_mode="r")
    ids_raw = np.load(raw / "vid.npy", mmap_mode="r")
    cav_raw = np.load(raw / "is_cav.npy", mmap_mode="r")
    full = np.asarray(kin[t0:t0 + steps + 1], dtype=np.float64)
    full_mask = np.asarray(mask_raw[t0:t0 + steps + 1]).astype(bool) \
        & (full[..., 1:4].sum(-1) > 0)
    full_ids = np.asarray(ids_raw[t0:t0 + steps + 1])
    full_cav = np.asarray(cav_raw[t0:t0 + steps + 1]).astype(bool)
    real_pos, real_lane = full[..., 0], full[..., 1:4].argmax(-1)
    real_v, real_len = full[..., 4], full[..., 6]
    real_lane_valid = full_mask & (full[..., 1:4].sum(-1) > 0)
    real_a = np.full_like(real_v, np.nan)
    real_a[1:] = (real_v[1:] - real_v[:-1]) / DT

    onset = None
    if cohort is not None:
        row = next((w for w in cohort["worlds"] if w["group_index"] == gi), None)
        if row is not None and row.get("event") and row.get("onset_s") is not None \
                and row["onset_s"] >= 0:
            onset = float(row["onset_s"])

    frames = []
    original_forward = model.forward

    def forward(state, *a, **k):
        arr = lambda x: x.detach().cpu().numpy()
        length = state.get("length_m")
        lane_np = arr(state["lane"])
        # empty lane one-hot rows are junction transits; argmax reads them as
        # lane 0 (the frozen legacy frame quirk) - keep both signals
        if lane_np.ndim == 3:
            junc_np = lane_np.sum(-1) <= 0
            lane_np = lane_np.argmax(-1)
        else:
            junc_np = np.zeros_like(lane_np, bool)
        frames.append(dict(s=arr(state["s_m"]), v=arr(state["v_mps"]),
                           lane=lane_np.astype(np.int64),
                           junc=junc_np.astype(bool),
                           present=arr(state["present"]).astype(bool),
                           cav=arr(state["is_cav"]).astype(bool),
                           length=arr(length) if length is not None else None))
        return original_forward(state, *a, **k)

    model.forward = forward
    torch.manual_seed(140014 + 7919 * gi + 1000003 * draw)
    keys = inp.paired_random._keys & np.uint64((1 << 52) - 1)
    inp.initial["merge_pivot_rng_seed"] = torch.tensor(keys.astype(np.int64))
    kwargs = {}
    if model._samples_lanes:
        kwargs["lane_uniform"] = inp.paired_random.uniform("lane", range(steps))
    start = time.monotonic()
    with torch.no_grad():
        roll = model.rollout(inp.initial, n_steps=steps, k0=0,
                             cav_teacher=None if inp.teacher is None else inp.teacher[:, :steps + 1],
                             present_seq=inp.present_seq[:, :steps + 1],
                             is_cav_seq=inp.is_cav[:, :steps + 1],
                             history=inp.hist, history_mask=inp.hist_mask,
                             controller_id=inp.controller_id,
                             action_timeline=inp.action_timeline,
                             cav_controller=inp.cav_controller,
                             entries=inp.entries, driver_history=inp.observed_hist,
                             driver_history_mask=inp.observed_hist_mask, **kwargs)
    model.forward = original_forward
    seconds = time.monotonic() - start
    assert len(frames) == steps, (len(frames), steps)

    arr = lambda x: x.detach().cpu().numpy()
    F = frames[0]
    wm_s = np.stack([f["s"][0] for f in frames] + [arr(roll["positions"])[0, -1]], 0)
    wm_v = np.stack([f["v"][0] for f in frames] + [arr(roll["speeds"])[0, -1]], 0)
    wm_lane = np.stack([f["lane"][0] for f in frames]
                       + [arr(roll["lanes"])[0, -1].astype(np.int64)], 0)
    wm_pres = np.stack([f["present"][0] for f in frames]
                       + [arr(roll["masks"])[0, -1].astype(bool)], 0)
    wm_a = np.concatenate([arr(roll["accel"])[0],
                           np.full((1, wm_lane.shape[1]), np.nan)], 0)
    wm_junc = np.stack([f["junc"][0] for f in frames]
                       + [np.zeros(wm_lane.shape[1], bool)], 0)
    return dict(entry=entry, onset=onset, seconds=seconds, wm_s=wm_s, wm_v=wm_v,
                wm_lane=wm_lane, wm_pres=wm_pres, wm_a=wm_a, wm_junc=wm_junc,
                frames=frames,
                real_pos=real_pos, real_v=real_v, real_lane=real_lane,
                real_a=real_a, full_mask=full_mask, full_ids=full_ids,
                full_cav=full_cav, real_len=real_len, real_lane_valid=real_lane_valid,
                physics=model.physics, raw=raw, t0=t0, kin=kin, mask_raw=mask_raw)


def analyze(cap):
    from tools.lrdiag.replay import speed_bins, breakdown
    out = {}
    physics = cap["physics"]
    wm_len = np.full(cap["wm_s"].shape, 5.0)
    # legacy and chain leader replays on the SAME captured WM frames
    lgap, _, lhas, _ = leader_rule(cap["wm_s"], cap["wm_v"], cap["wm_lane"],
                                   cap["wm_pres"], wm_len, physics, chain=False)
    cgap, _, chas, _ = leader_rule(cap["wm_s"], cap["wm_v"], cap["wm_lane"],
                                   cap["wm_pres"], wm_len, physics, chain=True,
                                   junc=cap["wm_junc"])
    rgap, _, rhas, _ = leader_rule(cap["real_pos"], cap["real_v"], cap["real_lane"],
                                   cap["real_lane_valid"], cap["real_len"], physics,
                                   chain=False)

    rows = cap["wm_pres"][:-1]
    a = cap["wm_a"][:-1]
    hb = rows & (a <= HARD_BRAKE_MPS2)
    phantom = rows & lhas[:-1] & (lgap[:-1] <= 2.0) & (~chas[:-1] | (cgap[:-1] > 2.0))
    near = ((cap["wm_s"][:-1] >= -20) & (cap["wm_s"][:-1] <= SECTION + 20)) & rows
    out["hard_brake"] = {
        "rows_total": int(rows.sum()),
        "events": int(hb.sum()),
        "events_near_merge": int((hb & near).sum()),
        "events_on_phantom_rows": int((hb & phantom).sum()),
        "phantom_rows": int(phantom.sum()),
        "phantom_rows_near_merge": int((phantom & near).sum()),
        "accel_on_phantom_rows_q": np.nanpercentile(a[phantom], [5, 25, 50, 75, 95]).tolist()
        if phantom.any() else None}
    rr = cap["real_lane_valid"][:-1]
    ra = cap["real_a"][:-1]
    out["hard_brake"]["real_near_merge"] = int(
        (rr & (ra <= HARD_BRAKE_MPS2)
         & ((cap["real_pos"][:-1] >= -20) & (cap["real_pos"][:-1] <= SECTION + 20))).sum())
    out["hard_brake"]["real_rows_total"] = int(rr.sum())

    # discharge
    wm_slot_ids = np.tile(np.arange(cap["wm_s"].shape[1])[None, :],
                          (cap["wm_s"].shape[0], 1))
    wm_cross = crossings_corrected(cap["wm_s"], cap["wm_pres"], cap["wm_lane"],
                                   np.ones(cap["wm_pres"].shape, bool), wm_slot_ids)
    real_cross = crossings_corrected(cap["real_pos"], cap["full_mask"], cap["real_lane"],
                                     cap["real_lane_valid"], cap["full_ids"])
    # WM own onset (registered detector, warm real prefix + wm bins)
    warm = np.asarray(cap["kin"][cap["t0"] - 120:cap["t0"] + 1], dtype=np.float64)
    warm_mask = np.asarray(cap["mask_raw"][cap["t0"] - 120:cap["t0"] + 1]).astype(bool) \
        & (warm[..., 1:4].sum(-1) > 0)
    wbins = speed_bins(warm[..., 4], warm_mask, warm[..., 0], warm[..., 1:4].argmax(-1),
                       np.zeros_like(warm_mask))
    wmbins = speed_bins(cap["wm_v"], cap["wm_pres"], cap["wm_s"], cap["wm_lane"],
                        np.zeros_like(cap["wm_pres"]))
    wm_onset = None
    for region in ("upstream", "merge"):
        det = breakdown(wbins[region + "/all"]["speed"] + wmbins[region + "/all"]["speed"])
        if det["event"] and det["onset_s"] is not None and det["onset_s"] >= 0:
            wm_onset = det["onset_s"]
            break
    out["capacity"] = {"real": window_stats(real_cross, cap["onset"]),
                       "wm_real_onset": window_stats(wm_cross, cap["onset"]),
                       "wm_own_onset": window_stats(wm_cross, wm_onset),
                       "real_crossings": len(real_cross), "wm_crossings": len(wm_cross)}
    out["wm_onset_s"] = wm_onset
    out["real_onset_s"] = cap["onset"]

    # per-tick dynamics error, identity-matched by slot: the WM initial frame IS
    # the real t0 frame (loader initial), possibly permuted - match slots by
    # exact (s, v) at t0, then compare presence-matched pairs over the window
    T = min(cap["wm_v"].shape[0], cap["real_v"].shape[0]) - 1
    lookup = {}
    for k in np.flatnonzero(cap["full_mask"][0]):
        lookup[(float(cap["real_pos"][0, k]), float(cap["real_v"][0, k]))] = k
    pairs = [(j, lookup[(float(cap["wm_s"][0, j]), float(cap["wm_v"][0, j]))])
             for j in np.flatnonzero(cap["wm_pres"][0])
             if (float(cap["wm_s"][0, j]), float(cap["wm_v"][0, j])) in lookup]
    wm_j = np.array([p[0] for p in pairs])
    re_k = np.array([p[1] for p in pairs])
    m = cap["wm_pres"][:T + 1, wm_j] & cap["full_mask"][:T + 1, re_k]
    dv_wm = (cap["wm_v"][:T + 1, wm_j][1:] - cap["wm_v"][:T + 1, wm_j][:-1]) / DT
    dv_re = (cap["real_v"][:T + 1, re_k][1:] - cap["real_v"][:T + 1, re_k][:-1]) / DT
    ok = m[1:] & m[:-1]
    err_full = np.abs(dv_wm - dv_re)
    err1 = err_full[ok]
    out["one_step_accel_mae"] = {
        "rows": int(ok.sum()),
        "matched_slots": len(pairs),
        "mae_all": float(err1.mean()),
        "mae_first60s": float(err_full[:120][ok[:120]].mean()) if ok[:120].any() else None,
        "mae_60_300s": float(err_full[120:][ok[120:]].mean()) if ok[120:].any() else None}
    hor = {}
    for h in (10, 30, 60, 120, 300):
        t = min(h * 2, T)
        mm = m[0] & m[t]
        hor[f"{h}s"] = {"v_mae": float(np.abs(cap["wm_v"][t][wm_j][mm] - cap["real_v"][t][re_k][mm]).mean()),
                        "s_mae": float(np.abs(cap["wm_s"][t][wm_j][mm] - cap["real_pos"][t][re_k][mm]).mean()),
                        "rows": int(mm.sum())}
    out["horizon_mae"] = hor
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", default="0,3,12")
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "tools/execparity/dynamics3_results.json")
    a = ap.parse_args()
    setup()
    import torch  # noqa: F401  (after setup paths)
    receipt = overlay_receipt()
    cohort = json.loads((HOME / "capfix_PREPASS.json").read_text())
    results = {"schema": "execparity3_dynamics/1", "wm_sha256": WM_SHA,
               "overlay": receipt, "groups": [], "decision": "03:40",
               "flag_env": "C4_SUMO_LEADER_RULE",
               "seed_recipe": "torch 140014+7919*gi+1000003*draw; settings 140014+draw*1000003",
               "panels": "E3 split fit only; never validation/pairs300/calibration"}
    for gi in [int(x) for x in a.groups.split(",")]:
        for flag in ("off", "on"):
            if flag == "on":
                os.environ["C4_SUMO_LEADER_RULE"] = "1"
            else:
                os.environ.pop("C4_SUMO_LEADER_RULE", None)
            cap = rollout_group(gi, a.draw, a.steps, cohort)
            res = analyze(cap)
            res.update({"group": gi, "world": cap["entry"]["world"], "flag": flag,
                        "seconds": cap["seconds"],
                        "input_sha256": {n: hashlib.sha256((cap["raw"] / n).read_bytes()).hexdigest()
                                         for n in ("veh_kin.npy", "veh_mask.npy", "vid.npy", "is_cav.npy")}})
            results["groups"].append(res)
            print(json.dumps({"group": gi, "flag": flag, "world": res["world"],
                              "seconds": round(cap["seconds"], 1),
                              "hard_brake": res["hard_brake"]["events"],
                              "drop_real_onset_veh_h": res["capacity"]["wm_real_onset"].get("drop_veh_h"),
                              "mae_all": res["one_step_accel_mae"]["mae_all"]}), flush=True)
        os.environ.pop("C4_SUMO_LEADER_RULE", None)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    results["groups"] = clean(results["groups"])
    a.out.write_text(json.dumps(clean(results), indent=1, allow_nan=False,
                                default=lambda x: float(x) if isinstance(x, np.floating)
                                else int(x) if isinstance(x, np.integer) else None) + "\n")
    print(json.dumps({"out": str(a.out)}))


if __name__ == "__main__":
    main()
