"""Round-5 pre-registered diagnosis (WM-S-R5-20261012), measurements M1-M5'.

Implements runs/active/WM-S-R5-PREP-20261012/DIAGNOSIS_PLAN.md with the round-5 code
tree (E:/agent-work/r5-score/src):

  M1  per-horizon |ds| split wrong-lane vs right-lane (10/30/60/120 s)      (wall_same 67)
  M2  survivor speed error at 60/90/120 s + section total delay at 120 s    (wall_same 67)
  M3  TRUE lane-change rates 0-10/10-60/60-120 s vs recorded               (wall_same 67)
  M4  inference-time switch-offs diag_lane_gate_shared / diag_lane_cap_off /
      diag_encoder_population / diag_recorded_lanes (R5, wall_same 67)
  M5' encoder LEARNED per-car settings (9 params) vs TRUE driver types
      (evaluation-only; never fed to the model)

Plus the same M1-M3 on real-history quick-20.  Comparators C1 and R4 run the same
code paths.  Rows cached under cache/diag_* (incremental rerun).  Writes
DIAGNOSIS.md + diagnosis.json.  CPU only; no training, no simulation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SCORE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]          # E:/research-dense
R5_SRC = Path("E:/agent-work/r5-score/src")
if str(R5_SRC) not in sys.path:
    sys.path.insert(0, str(R5_SRC))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["TEMP"] = os.environ["TMP"] = "E:/agent-work/tmp"
sys.dont_write_bytecode = True

import numpy as np

DATA = ROOT / "data_cache"
SETS = {
    "wall_same": DATA / "wallc1_states_h60" / "wall_same.json",
    "wall_same_real": DATA / "wallc1_states_h60_real" / "wall_same.json",
}
HORIZONS = (10.0, 30.0, 60.0, 120.0)
M2_HORIZONS = (60.0, 90.0, 120.0)
LANE_WINDOWS = {"0-10": (0, 20), "10-60": (20, 120), "60-120": (120, 240)}
ARM = "wall"
TRUE_RATE_PER_HDV_STEP = 0.00124327163
C1_PATH = DATA / "ckpt_c1fast" / "wm_s.pt"
R4_PATH = ROOT / "runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt"
TRUTH = ROOT / "runs/active/WM-S-R4-CHK-TYPEID-20261011/out/truth_full.json"
JAM_SPEED_MPS = 8.0
SWITCHES = ("diag_lane_gate_shared", "diag_lane_cap_off", "diag_encoder_population",
            "diag_recorded_lanes")
PARAM_NAMES = None  # filled from wms.model.driver_encoder on first use


def sha8(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:8]


def state_key(row: dict) -> tuple:
    return (str(row.get("world")), str(row.get("key")), str(row.get("source_arm")),
            round(float(row.get("t0_s", 0)), 3))


def iter_records(path: Path) -> list[dict]:
    from wms.eval.normalize import iter_windows
    return list(iter_windows(path))


def quick20(path: Path) -> list[dict]:
    rows = iter_records(path)
    chosen: dict = {}
    for r in sorted(rows, key=lambda x: (str(x.get("world")), float(x.get("t0_s", 0)), str(x.get("key")))):
        chosen.setdefault(str(r.get("world")), r)
    return [chosen[k] for k in sorted(chosen)[:20]]


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
_W: dict = {}


def _init(model_tag: str, ckpt: str, diag: tuple[str, ...], capture_settings: bool) -> None:
    import torch
    torch.set_num_threads(1)
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    flags = RolloutFlags(entries=False, entries_v2=False,
                         **{name: name in diag for name in SWITCHES})
    predictor = WmsPredictor.from_checkpoint(ckpt, flags)
    captured: list[dict] = []
    if capture_settings:
        original = predictor.model.step

        def step(self, state, **kwargs):
            out = original(state, **kwargs)
            if kwargs.get("k_steps", 0) == 0 and "driver_parameters" in out:
                present = state.get("present")
                is_cav = state.get("is_cav")
                slots = state["s_m"].shape[-1]
                captured.append({
                    "present": (present.detach().cpu().numpy()[0] if present is not None
                                else np.ones(slots, bool)),
                    "is_cav": (is_cav.detach().cpu().numpy()[0] if is_cav is not None
                               else np.zeros(slots, bool)),
                    "values": out["driver_parameters"].detach().cpu().numpy()[0],
                })
            return out

        predictor.model.step = types.MethodType(step, predictor.model)
    _W.update(predictor=predictor, captured=captured, model_tag=model_tag)


def _frame_arrays(frames: list[dict], recorded: list[dict], t0: float):
    ids = sorted({str(v["id"]) for fr in frames + recorded for v in fr.get("vehicles", [])})
    idx = {x: i for i, x in enumerate(ids)}
    n = int(round(120 / 0.5))
    arrays = {}
    for prefix, seq in (("p", frames), ("r", recorded)):
        lane = np.full((n + 1, len(ids)), -1, dtype=np.int16)
        s_arr = np.full((n + 1, len(ids)), np.nan, dtype=float)
        speed = np.full((n + 1, len(ids)), np.nan, dtype=float)
        present = np.zeros((n + 1, len(ids)), dtype=bool)
        cav = np.zeros((n + 1, len(ids)), dtype=bool)
        for fr in seq:
            k = int(round((float(fr["time_s"]) - t0) / 0.5))
            if not (0 <= k <= n):
                continue
            for v in fr.get("vehicles", []):
                if not v.get("present", True):
                    continue
                j = idx[str(v["id"])]
                lane[k, j] = int(v.get("lane", -1))
                s_arr[k, j] = float(v.get("s", np.nan))
                speed[k, j] = float(v["v"])
                present[k, j] = True
                cav[k, j] = bool(v.get("is_cav", False))
        arrays[prefix] = (lane, s_arr, speed, present, cav)
    return ids, idx, arrays


def _lane_window(p_lane, r_lane, p_pres, r_pres, p_cav, r_cav, lo: int, hi: int) -> dict:
    """TRUE lane-change counts (forward-filled over junction -1, HDVs only)."""
    changes = steps = rec_changes = rec_steps = 0
    pred_filled, rec_filled = p_lane.copy(), r_lane.copy()
    for lanes in (pred_filled, rec_filled):
        for tick in range(1, lanes.shape[0]):
            bad = lanes[tick] < 0
            lanes[tick, bad] = lanes[tick - 1, bad]
    for k in range(lo + 1, hi + 1):
        valid = p_pres[k] & p_pres[k - 1] & r_pres[k] & r_pres[k - 1] & ~r_cav[k] & ~p_cav[k]
        for lanes, is_pred in ((pred_filled, True), (rec_filled, False)):
            cur, prev = lanes[k], lanes[k - 1]
            good = valid & (cur >= 0) & (prev >= 0)
            ch, st = int((good & (cur != prev)).sum()), int(good.sum())
            if is_pred:
                changes, steps = changes + ch, steps + st
            else:
                rec_changes, rec_steps = rec_changes + ch, rec_steps + st
    return {"changes": changes, "hdv_steps": steps,
            "recorded_changes": rec_changes, "recorded_hdv_steps": rec_steps}


def _work(job: tuple) -> dict:
    model_tag, variant, record = job
    global PARAM_NAMES
    from wms.eval.normalize import recorded_for
    from wms.eval.rollout_inputs import t0_vehicles
    if PARAM_NAMES is None:
        from wms.model.driver_encoder import PARAMETER_NAMES
        PARAM_NAMES = PARAMETER_NAMES
    predictor, captured = _W["predictor"], _W["captured"]
    del captured[:]
    t0 = float(record["t0_s"])
    arm = ARM if ARM in (record.get("recorded") or {}) else next(iter(record["recorded"]))
    err = None
    out = None
    try:
        frames = predictor.predict_states(record, {**record.get("cav_controllers", {}), "arm": arm},
                                          120.0, 0)
        recorded = recorded_for(record, arm)
        ids, idx, a = _frame_arrays(frames, recorded, t0)
        pl, ps, pv, pp, pc = a["p"]
        rl, rs, rv, rp, rc = a["r"]
        # M1: per-car |ds| split by lane agreement at h (both present, HDV, recorded lane >= 0)
        m1 = {}
        for h in HORIZONS:
            k = int(h * 2)
            mask = pp[k] & rp[k] & ~rc[k] & ~pc[k] & (rl[k] >= 0) & (pl[k] >= 0)
            ds = np.abs(ps[k] - rs[k])
            wrong = mask & (pl[k] != rl[k])
            m1[str(int(h))] = {
                "wrong": ds[wrong].astype(np.float32), "right": ds[mask & ~wrong].astype(np.float32),
                "n_matched": int(mask.sum()), "n_wrong": int(wrong.sum())}
        # M2: survivors = t0 HDVs present in both at 120 s
        mask0 = pp[0] & rp[0] & ~rc[0] & ~pc[0] & (rl[0] >= 0)
        k120 = 240
        surv = mask0 & pp[k120] & rp[k120]
        m2 = {"n": int(surv.sum()), "v0": rv[0][surv].astype(np.float32)}
        for h in M2_HORIZONS:
            k = int(h * 2)
            dv = pv[k][surv] - rv[k][surv]
            m2[str(int(h))] = {"signed": dv.astype(np.float32)}
        # section total delay at 120 s (predicted vs recorded)
        from wms.eval.score import section_delays
        v0 = float(record.get("v0_mps", 13.89))
        dp = section_delays(frames, t0, v0, (120.0,)).get("120.0", {})
        dr = section_delays(recorded, t0, v0, (120.0,)).get("120.0", {})
        m2["delay_120"] = {"predicted": dp, "recorded": dr,
                           "total_pred": float(sum(dp.values())) if dp else None,
                           "total_rec": float(sum(dr.values())) if dr else None}
        # M3: TRUE lane-change rates per window
        m3 = {key: _lane_window(pl, rl, pp, rp, pc, rc, lo, hi)
              for key, (lo, hi) in LANE_WINDOWS.items()}
        # state-median |ds| per horizon (variant effect on overall error)
        from wms.eval.metrics import car_metrics
        from wms.eval.score import car_compact
        car = car_compact(car_metrics(frames, recorded, t0_s=t0, horizons=HORIZONS))
        ds_med = {str(int(h)): car[str(float(h))]["hdv"]["ds_med"] for h in HORIZONS}
        # M5' learned settings (R5 base only; captured at the first rollout step)
        settings = {}
        if captured:
            _, ids0, _, _ = t0_vehicles(record)
            cap = captured[0]
            live = np.asarray(cap["present"], bool) & ~np.asarray(cap["is_cav"], bool)
            for j, vid in enumerate(ids0):
                if j < len(live) and live[j]:
                    settings[str(vid)] = np.asarray(cap["values"][j], np.float32)
        out = {"m1": m1, "m2": m2, "m3": m3, "ds_med": ds_med, "settings": settings}
    except Exception as exc:  # keep the batch alive
        import traceback
        err = f"{type(exc).__name__}: {exc} | {traceback.format_exc()[-400:]}"
    return {"key": state_key(record), "variant": variant, "model": model_tag,
            "result": out, "error": err}


# --------------------------------------------------------------------------- #
# incremental variant runner
# --------------------------------------------------------------------------- #
def run_variant(model_tag: str, ckpt: Path, variant: str, records: list[dict], workers: int,
                capture_settings: bool = False, set_name: str = "wall_same") -> list[dict]:
    diag = () if variant == "base" else (variant,)
    cache = SCORE / "cache" / f"diag__{set_name}__{model_tag}__{variant}__{sha8(ckpt)}.json"
    if cache.is_file():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    print(f"diag {model_tag}/{variant}: {len(records)} states", flush=True)
    t0 = time.time()
    jobs = [(model_tag, variant, r) for r in records]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(model_tag, str(ckpt), diag, capture_settings)) as pool:
        for row in pool.map(_work, jobs, chunksize=1):
            if row["error"]:
                print(f"  error {row['key']}: {row['error'][:160]}", flush=True)
            rows.append({"key": list(row["key"]), "variant": variant, "model": model_tag,
                         "result": _jsonify(row["result"]), "error": row["error"]})
    rows.sort(key=lambda r: r["key"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
    tmp.replace(cache)
    print(f"diag {model_tag}/{variant}: done in {time.time() - t0:.0f}s", flush=True)
    return rows


def _jsonify(result: dict | None) -> dict | None:
    if result is None:
        return None
    out = json.loads(json.dumps(result, default=lambda o: np.asarray(o).tolist()
                                 if hasattr(o, "tolist") else str(o)))
    return out


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "median": None, "p90": None, "mean": None}
    a = np.asarray(vals, float)
    return {"n": int(a.size), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "mean": float(a.mean())}


def aggregate(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error") and r.get("result")]
    agg: dict = {"n_states": len(ok), "errors": len(rows) - len(ok), "m1": {}, "m2": {},
                 "m3": {}, "ds_med": {}}
    for h in HORIZONS:
        key = str(int(h))
        wrong: list = []
        right: list = []
        for r in ok:
            wrong.extend(float(x) for x in r["result"]["m1"][key]["wrong"])
            right.extend(float(x) for x in r["result"]["m1"][key]["right"])
        n_wrong = sum(r["result"]["m1"][key]["n_wrong"] for r in ok)
        n_matched = sum(r["result"]["m1"][key]["n_matched"] for r in ok)
        agg["m1"][key] = {"wrong_lane": _stats(wrong), "right_lane": _stats(right),
                          "wrong_share": (n_wrong / n_matched) if n_matched else None}
    m2 = {"n": 0, "signed": {}, "abs": {}, "jam_signed": {}, "free_signed": {},
          "delay_120_total_pred": 0.0, "delay_120_total_rec": 0.0, "n_delay": 0}
    for h in M2_HORIZONS:
        m2["signed"][str(int(h))] = []
        m2["abs"][str(int(h))] = []
        m2["jam_signed"][str(int(h))] = []
        m2["free_signed"][str(int(h))] = []
    for r in ok:
        res = r["result"]["m2"]
        m2["n"] += res["n"]
        jam = np.asarray(res["v0"], float) < JAM_SPEED_MPS
        for h in M2_HORIZONS:
            dv = np.asarray(res[str(int(h))]["signed"], float)
            m2["signed"][str(int(h))].extend(dv.tolist())
            m2["abs"][str(int(h))].extend(np.abs(dv).tolist())
            m2["jam_signed"][str(int(h))].extend(dv[jam].tolist())
            m2["free_signed"][str(int(h))].extend(dv[~jam].tolist())
        if res["delay_120"]["total_pred"] is not None:
            m2["delay_120_total_pred"] += res["delay_120"]["total_pred"]
            m2["delay_120_total_rec"] += res["delay_120"]["total_rec"]
            m2["n_delay"] += 1
    agg["m2"] = {"n_survivors": m2["n"],
                 "signed_dv": {h: _stats(v) for h, v in m2["signed"].items()},
                 "abs_dv": {h: _stats(v) for h, v in m2["abs"].items()},
                 "jam_signed_dv": {h: _stats(v) for h, v in m2["jam_signed"].items()},
                 "free_signed_dv": {h: _stats(v) for h, v in m2["free_signed"].items()},
                 "delay_120_total_pred": m2["delay_120_total_pred"],
                 "delay_120_total_rec": m2["delay_120_total_rec"],
                 "n_delay_states": m2["n_delay"]}
    m3 = {}
    for window in LANE_WINDOWS:
        ch = sum(r["result"]["m3"][window]["changes"] for r in ok)
        st = sum(r["result"]["m3"][window]["hdv_steps"] for r in ok)
        rch = sum(r["result"]["m3"][window]["recorded_changes"] for r in ok)
        rst = sum(r["result"]["m3"][window]["recorded_hdv_steps"] for r in ok)
        m3[window] = {"changes": ch, "hdv_steps": st,
                      "rate_per_hdv_step": (ch / st) if st else None,
                      "per_hdv_per_10s": (20.0 * ch / st) if st else None,
                      "recorded_changes": rch, "recorded_hdv_steps": rst,
                      "recorded_rate_per_hdv_step": (rch / rst) if rst else None,
                      "recorded_per_hdv_per_10s": (20.0 * rch / rst) if rst else None}
    agg["m3"] = m3
    for h in HORIZONS:
        vals = [r["result"]["ds_med"][str(int(h))] for r in ok
                if r["result"]["ds_med"][str(int(h))] is not None]
        agg["ds_med"][str(int(h))] = _stats(vals)
    return agg


# --------------------------------------------------------------------------- #
# M5' learned settings vs true types (evaluation-only, parent process)
# --------------------------------------------------------------------------- #
def m5_settings(rows: list[dict], ckpt_tag: str) -> dict:
    from wms.model.driver_encoder import PARAMETER_NAMES
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    samples: dict[tuple, dict] = {}
    n_settings = 0
    for r in rows:
        if r.get("error") or not r.get("result"):
            continue
        world = r["key"][0]
        for vid, params in (r["result"].get("settings") or {}).items():
            n_settings += 1
            samples.setdefault((world, vid), [float(x) for x in params])
    pts = []
    for (world, vid), params in sorted(samples.items()):
        block = truth.get(world) or {}
        name = ((block.get("vehicles") or {}).get(vid))
        if name is None:
            continue
        p = (block.get("vtypes") or {}).get(name, {})
        pts.append({"world": world, "vid": vid, "type": name, "params": params,
                    "tau": float(p.get("tau", float("nan"))),
                    "minGap": float(p.get("minGap", float("nan"))),
                    "accel": float(p.get("accel", float("nan"))),
                    "speedFactor": float(p.get("speedFactor", float("nan")))})
    out = {"model": ckpt_tag, "n_settings_captured": n_settings,
           "n_typed_samples": len(pts), "parameter_names": list(PARAMETER_NAMES),
           "per_type": {}, "separability": None, "correlations": {}}
    if not pts:
        return out
    types = sorted({p["type"] for p in pts})
    P = np.array([p["params"] for p in pts], float)
    for name in types:
        sel = np.array([p["type"] == name for p in pts])
        block = {}
        for i, pn in enumerate(PARAMETER_NAMES):
            s = _stats(P[sel, i].tolist())
            block[pn] = {"n": s["n"], "mean": s["mean"], "median": s["median"],
                         "p25": float(np.percentile(P[sel, i], 25)),
                         "p75": float(np.percentile(P[sel, i], 75))}
        out["per_type"][name] = {"n": int(sel.sum()), "params": block}
    # separability: nearest centroid on standardized settings, split by md5(vid) parity
    sd = P.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (P - P.mean(0)) / sd
    import hashlib
    fit = np.array([int(hashlib.md5(p["vid"].encode()).hexdigest()[:8], 16) % 2 == 0 for p in pts])
    labels = np.array([types.index(p["type"]) for p in pts])
    if fit.any() and (~fit).any() and len(types) == 3:
        centroids = np.stack([Z[fit & (labels == t)].mean(0) for t in range(len(types))])
        d = ((Z[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        pred = d.argmin(1)
        acc_test = float((pred[~fit] == labels[~fit]).mean())
        acc_train = float((pred[fit] == labels[fit]).mean())
        counts = np.array([(labels == t).sum() for t in range(len(types))])
        confusion = [[int(((pred == j) & (labels == t) & ~fit).sum()) for j in range(len(types))]
                     for t in range(len(types))]
        out["separability"] = {
            "method": "nearest-centroid on 9 standardized learned settings; fit/test split by md5(vid) parity",
            "n_fit": int(fit.sum()), "n_test": int((~fit).sum()),
            "accuracy_test": acc_test, "accuracy_train": acc_train,
            "majority_baseline_test": float(counts.max() / counts.sum()),
            "confusion_test_rows_true_cols_pred": confusion, "types": types}
    pairs = (("time_headway", "tau"), ("min_gap", "minGap"), ("accel", "accel"),
             ("v0", "speedFactor"))
    for learned, true_name in pairs:
        i = PARAMETER_NAMES.index(learned)
        x, y = P[:, i], np.array([p[true_name] for p in pts], float)
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() > 2 and np.std(x[good]) > 0 and np.std(y[good]) > 0:
            def _pearson(a, b):
                a2, b2 = a - a.mean(), b - b.mean()
                return float((a2 * b2).sum() / np.sqrt((a2 ** 2).sum() * (b2 ** 2).sum()))
            def _rank(a):
                order = np.argsort(np.argsort(a))
                return order.astype(float)
            out["correlations"][f"{learned}_vs_{true_name}"] = {
                "n": int(good.sum()), "pearson": _pearson(x[good], y[good]),
                "spearman": _pearson(_rank(x[good]), _rank(y[good]))}
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def fmt(v, d=2):
    return "n/a" if v is None else f"{v:.{d}f}"


def write_report(diag: dict, path: Path) -> None:
    L = ["# Round 5 diagnosis (WM-S-R5-20261012) — pre-registered measurements M1-M5'", "",
         f"Plan: `WM-S-R5-PREP-20261012/DIAGNOSIS_PLAN.md`. Round-5 code tree "
         f"`E:/agent-work/r5-score/src`; comparators C1/R4 same code paths; "
         f"true driver types evaluation-only (`WM-S-R4-CHK-TYPEID-20261011/out/truth_full.json`).",
         ""]
    wall = diag["runs"].get("wall_same", {})
    L.append("## M1 — |ds| split wrong-lane vs right-lane (wall_same 67)")
    for set_name in ("wall_same", "wall_same_real"):
        runs = diag["runs"].get(set_name)
        if not runs:
            continue
        L.append(f"### {set_name}")
        L.append("| model/variant | h | n | wrong share | wrong-lane med (p90) | right-lane med (p90) |")
        L.append("|---|---|---|---|---|---|")
        for run_key, agg in runs.items():
            for h in ("10", "30", "60", "120"):
                m = agg["m1"][h]
                L.append(f"| {run_key} | {h} | {m['right_lane']['n'] + m['wrong_lane']['n']} | "
                         f"{fmt(m['wrong_share'], 3)} | "
                         f"{fmt(m['wrong_lane']['median'])} ({fmt(m['wrong_lane']['p90'])}) | "
                         f"{fmt(m['right_lane']['median'])} ({fmt(m['right_lane']['p90'])}) |")
    L.append("")
    L.append("## M2 — survivor speed drift (t0-present HDVs alive at 120 s) + delay at 120 s")
    L.append("| model/variant | set | n surv | dv60 signed (jam) | dv90 | dv120 | |dv|120 med | delay120 pred/rec (veh-s) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for set_name, runs in diag["runs"].items():
        for run_key, agg in runs.items():
            m2 = agg["m2"]
            def sv(h, k="signed_dv"):
                return fmt(m2[k][h]["mean"])
            L.append(f"| {run_key} | {set_name} | {m2['n_survivors']} | "
                     f"{sv('60')} ({sv('60', 'jam_signed_dv')}) | {sv('90')} | {sv('120')} | "
                     f"{fmt(m2['abs_dv']['120']['median'])} | "
                     f"{fmt(m2['delay_120_total_pred'], 0)} / {fmt(m2['delay_120_total_rec'], 0)} |")
    L.append("")
    L.append("## M3 — TRUE lane-change rates (changes per HDV-step; per HDV per 10 s in parens)")
    L.append("| model/variant | set | 0-10 s | 10-60 s | 60-120 s | recorded 0-120 |")
    L.append("|---|---|---|---|---|---|")
    for set_name, runs in diag["runs"].items():
        rec120 = None
        for run_key, agg in runs.items():
            m3 = agg["m3"]
            def rate(w):
                r = m3[w]["rate_per_hdv_step"]
                return "n/a" if r is None else f"{r:.5f} ({m3[w]['per_hdv_per_10s']:.3f})"
            rr = m3["60-120"].get("recorded_rate_per_hdv_step")
            rec120 = f"{rr:.5f}" if rr is not None else rec120
            L.append(f"| {run_key} | {set_name} | {rate('0-10')} | {rate('10-60')} | "
                     f"{rate('60-120')} | {rec120 or 'n/a'} |")
    L.append("")
    L.append(f"Training flip-rate allowance (TRUE rate) = {TRUE_RATE_PER_HDV_STEP} per HDV-step.")
    L.append("")
    L.append("## M4 — inference-time switch-offs (R5, wall_same 67)")
    base = diag["runs"]["wall_same"].get("R5/base")
    if base:
        L.append("| variant | 60 s ds med | 120 s ds med | wrong-lane share 120 | lane rate 60-120/step | dv120 signed |")
        L.append("|---|---|---|---|---|---|")
        for run_key, agg in diag["runs"]["wall_same"].items():
            if not run_key.startswith("R5/"):
                continue
            L.append(f"| {run_key} | {fmt(agg['ds_med']['60']['median'])} | "
                     f"{fmt(agg['ds_med']['120']['median'])} | "
                     f"{fmt(agg['m1']['120']['wrong_share'], 3)} | "
                     f"{fmt(agg['m3']['60-120']['rate_per_hdv_step'], 5)} | "
                     f"{fmt(agg['m2']['signed_dv']['120']['mean'])} |")
        for run_key, agg in diag["runs"]["wall_same"].items():
            if run_key.startswith("R5/diag_"):
                d60 = agg["ds_med"]["60"]["median"] - base["ds_med"]["60"]["median"]
                d120 = agg["ds_med"]["120"]["median"] - base["ds_med"]["120"]["median"]
                b120 = base["ds_med"]["120"]["median"]
                pct = (100 * d120 / b120) if b120 else None
                L.append(f"- {run_key}: 60 s delta {d60:+.2f} m, 120 s delta {d120:+.2f} m "
                         + (f"({pct:+.1f}% vs R5 base)" if pct is not None else ""))
    L.append("")
    L.append("## M5' — encoder's learned per-car settings vs TRUE driver types (evaluation-only)")
    m5 = diag.get("m5_settings") or {}
    if m5.get("per_type"):
        L.append("| parameter | " + " | ".join(f"{t} (n={m5['per_type'][t]['n']})"
                                               for t in m5["per_type"]) + " |")
        L.append("|---" * (len(m5["per_type"]) + 1) + "|")
        for pn in m5["parameter_names"]:
            cells = []
            for t in m5["per_type"]:
                blk = m5["per_type"][t]["params"][pn]
                cells.append(f"{fmt(blk['median'])} [{fmt(blk['p25'])}, {fmt(blk['p75'])}]")
            L.append(f"| {pn} | " + " | ".join(cells) + " |")
        sep = m5.get("separability") or {}
        if sep:
            L.append("")
            L.append(f"Separability (nearest centroid, half/half split): test accuracy "
                     f"**{fmt(sep.get('accuracy_test'), 3)}** vs majority "
                     f"{fmt(sep.get('majority_baseline_test'), 3)} "
                     f"(fit {sep.get('n_fit')}/test {sep.get('n_test')}).")
        corr = m5.get("correlations") or {}
        if corr:
            L.append("")
            L.append("Correlations with true parameters: " + "; ".join(
                f"{k}: r={fmt(v['pearson'])} (rho={fmt(v['spearman'])}, n={v['n']})"
                for k, v in corr.items()))
    else:
        L.append("(no settings captured — model not hybrid or no rows)")
    L.append("")
    L.append("## Pre-registered decision rules (auto-evaluated)")
    L.append("```")
    L.append(json.dumps(diag.get("decision_rules", {}), indent=1))
    L.append("```")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def evaluate_rules(diag: dict) -> dict:
    rules: dict = {}
    runs = diag["runs"].get("wall_same", {})
    base = runs.get("R5/base")
    if base:
        def med(run, h="120"):
            r = runs.get(run)
            return None if r is None else r["ds_med"][h]["median"]
        m = {name: med(f"R5/{name}") for name in ("base", "diag_recorded_lanes",
                                                  "diag_encoder_population")}
        b = m["base"]
        if b and all(v is not None for v in m.values()):
            rules["R1_lane_vs_encoder"] = {
                "recorded_lanes_cut_120s_pct": 100 * (m["diag_recorded_lanes"] - b) / b,
                "encoder_population_change_120s_pct": 100 * (m["diag_encoder_population"] - b) / b,
                "verdict": ("lane decisions" if (b - m["diag_recorded_lanes"]) > 0.25 * b
                            and abs(m["diag_encoder_population"] - b) < 0.10 * b else
                            "encoder settings overfit" if (b - m["diag_encoder_population"]) > 0.10 * b
                            else "mixed / inconclusive")}
        rate = base["m3"]["10-60"]["rate_per_hdv_step"] or base["m3"]["0-10"]["rate_per_hdv_step"]
        rec = base["m3"]["10-60"]["recorded_rate_per_hdv_step"]
        if rate is not None and rec:
            ratio = rate / rec if rec else None
            rules["R3_lane_rate"] = {"pred_over_recorded": ratio,
                                     "verdict": ("hard lane budget" if ratio and ratio > 2
                                                 else "lane-rate not binding if wrong-lane share unchanged"
                                                 if ratio and ratio < 1.2 else "middle ground")}
    m2 = (runs.get("R5/base") or {}).get("m2")
    if m2:
        seq = [m2["jam_signed_dv"][h]["mean"] for h in ("60", "90", "120")]
        free = [m2["free_signed_dv"][h]["mean"] for h in ("60", "90", "120")]
        grow = all(seq[i] is not None and seq[i + 1] is not None and abs(seq[i + 1]) > abs(seq[i])
                   for i in range(len(seq) - 1))
        rules["R2_jam_discharge"] = {"jam_signed_dv_60_90_120": seq, "free_signed_dv_60_90_120": free,
                                     "monotone_growth_in_abs": bool(grow),
                                     "verdict": ("keep jam-discharge hypothesis" if grow
                                                 else "no monotone jam growth")}
    return rules


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="round-5 wm_s.pt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N states per set")
    ap.add_argument("--quick-real-n", type=int, default=20)
    ap.add_argument("--skip-c1r4", action="store_true", help="skip comparator runs")
    args = ap.parse_args()
    ckpt = Path(args.ckpt).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be in 1..8")
    t_start = time.time()
    diag = {"schema": "wm_s_round5_diagnosis/1", "runs": {}, "runtimes_s": {},
            "notes": {"plan": "WM-S-R5-PREP-20261012/DIAGNOSIS_PLAN.md",
                      "scorer_src": str(R5_SRC), "jam_speed_mps": JAM_SPEED_MPS,
                      "true_rate_per_hdv_step": TRUE_RATE_PER_HDV_STEP}}

    wall_records = iter_records(SETS["wall_same"])
    real_records = quick20(SETS["wall_same_real"])
    if args.limit:
        wall_records = wall_records[:args.limit]
        real_records = real_records[:args.limit]

    jobs: list[tuple[str, Path, str, list[dict], bool]] = []
    for tag, path in (("R5", ckpt), ("R4", R4_PATH), ("C1", C1_PATH)):
        jobs.append((tag, path, "base", wall_records, tag == "R5"))
    for sw in SWITCHES:
        jobs.append(("R5", ckpt, sw, wall_records, False))
    for tag, path in (("R5", ckpt), ("R4", R4_PATH), ("C1", C1_PATH)):
        jobs.append((tag, path, "base", real_records, tag == "R5"))
    jobs.append(("R5", ckpt, "diag_encoder_population", real_records, False))
    jobs.append(("R5", ckpt, "diag_recorded_lanes", real_records, False))
    if args.skip_c1r4:
        jobs = [j for j in jobs if j[0] == "R5"]

    for tag, path, variant, records, capture in jobs:
        set_name = "wall_same" if records is wall_records else "wall_same_real"
        t0 = time.time()
        rows = run_variant(tag, path, variant, records, args.workers, capture, set_name)
        diag["runs"].setdefault(set_name, {})[f"{tag}/{variant}"] = aggregate(rows)
        diag["runtimes_s"][f"{set_name}__{tag}_{variant}"] = round(time.time() - t0, 1)
        if capture and variant == "base":
            diag["m5_settings"] = m5_settings(rows, tag)
            diag["m5_settings"]["set"] = "wall_same"
        _dump(diag)

    diag["decision_rules"] = evaluate_rules(diag)
    diag["runtimes_s"]["total"] = round(time.time() - t_start, 1)
    _dump(diag)
    write_report(diag, SCORE / "DIAGNOSIS.md")
    print(f"wrote {SCORE / 'DIAGNOSIS.md'} and diagnosis.json", flush=True)
    return 0


def _dump(diag: dict) -> None:
    (SCORE / "diagnosis.json").write_text(json.dumps(diag, indent=1), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
