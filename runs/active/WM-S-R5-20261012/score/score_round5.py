"""Round-5 one-command scoring (WM-S-R5-20261012): R5 vs C1 vs R4, CPU, clean scorer.

Usage (round-5 code tree on PYTHONPATH; this script re-adds it itself):
    python score_round5.py --ckpt <wm_s.pt> [--ckpt-best <wm_s_best.pt>] [--workers 8]

Sets: wall_same 67 + wall_pilot 53 (data_cache/wallc1_states_h60) and the FULL
real-history 120 (data_cache/wallc1_states_h60_real).  C1 / R4 comparators come from
the round-4 caches (score cache for the wall sets, CHK-FADE confirmation rows for the
full real-history set; the baseline gate re-verifies C1 8.71/99.84, R4 9.96/110.69).
Per-state rows are cached under cache/ (keyed by checkpoint sha) so a rerun only
scores missing states.  Fingerprints as in ROUND4_SCORE (gate curves incl. the new
separate lane gate, late |g*da|, signed speed error, TRUE lane-change rates, wrong-lane
share, hard-brake); C1/R4 rollout fingerprints are copied from the R4 results.json.
Writes ROUND5_SCORE.md + results.json.  No training, no simulation, no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

SCORE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]          # E:/research-dense
R5_SRC = Path("E:/agent-work/r5-score/src")          # round-5 code tree (commit 46b10d5)
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
    "wall_pilot": DATA / "wallc1_states_h60" / "wall_pilot.json",
    "wall_same_real": DATA / "wallc1_states_h60_real" / "wall_same.json",
}
HORIZONS = (10.0, 30.0, 60.0, 120.0)
ARM = "wall"
C1_PATH = DATA / "ckpt_c1fast" / "wm_s.pt"
R4_PATH = ROOT / "runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt"
R4_SCORE = ROOT / "runs/active/WM-S-R4-20261011/score"
FADE_ROWS = ROOT / "runs/active/WM-S-R4-CHK-FADE-20261011/confirmation_rows"
BASELINE_GATE = {"C1": (8.71, 99.84), "R4": (9.96, 110.69)}
STEP0_EXPECTED = 11.9576


def iter_records(path: Path) -> list[dict]:
    from wms.eval.normalize import iter_windows
    return list(iter_windows(path))


def quick20(path: Path) -> list[dict]:
    rows = iter_records(path)
    chosen: dict = {}
    for r in sorted(rows, key=lambda x: (str(x.get("world")), float(x.get("t0_s", 0)), str(x.get("key")))):
        chosen.setdefault(str(r.get("world")), r)
    return [chosen[k] for k in sorted(chosen)[:20]]


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


def tested_arm(row: dict) -> str:
    arms = row.get("arms") or {}
    return ARM if ARM in arms else next(iter(arms))


# --------------------------------------------------------------------------- #
# comparators (round-4 caches; no recomputation)
# --------------------------------------------------------------------------- #
def load_comparator(set_name: str) -> tuple[list[dict], list[dict]]:
    """(C1 rows, R4 rows) in scorer format for one set."""
    if set_name in ("wall_same", "wall_pilot"):
        def load(name: str) -> list[dict]:
            return json.loads((R4_SCORE / "cache" / f"{name}__{set_name}.json")
                              .read_text(encoding="utf-8"))["states"]
        return load("C1"), load("R4-last")
    rows = json.loads((FADE_ROWS / set_name / "R4" / "base.json").read_text(encoding="utf-8"))
    out = {"C1": [], "R4": []}
    for model, fname in (("C1", "C1"), ("R4", "R4")):
        for r in json.loads((FADE_ROWS / set_name / fname / "base.json").read_text(encoding="utf-8")):
            if r.get("error"):
                continue
            car = {str(float(h)): {"hdv": {"ds_med": r["horizons"][h]["ds_med"],
                                           "dv_med": r["horizons"][h]["dv"]}}
                   for h in ("10", "30", "60", "120") if r["horizons"].get(h)}
            out[model].append({"world": r["world"], "key": r["key"],
                               "source_arm": r["source_arm"], "t0_s": r["t0_s"],
                               "arms": {ARM: {"error": None, "car": car}}})
    return out["C1"], out["R4"]


def baseline_gate() -> dict:
    checks = {}
    for name, cache_name in (("C1", "C1"), ("R4", "R4-last")):
        summary = json.loads((R4_SCORE / "cache" / f"{cache_name}__wall_same.json")
                             .read_text(encoding="utf-8"))["summary"]
        checks[name] = {"60": summary["60"]["ds"]["median"], "120": summary["120"]["ds"]["median"]}
    passed = all(abs(checks[m][h] - BASELINE_GATE[m][i]) <= 0.01
                 for m in BASELINE_GATE for i, h in enumerate(("60", "120")))
    return {"values": checks, "expected": {m: {"60": BASELINE_GATE[m][0],
                                               "120": BASELINE_GATE[m][1]} for m in BASELINE_GATE},
            "pass": passed}


def step0(workers: int) -> dict:
    """Scorer-integrity check: C1 wall_same quick-20 60 s state-median must be 11.9576 m."""
    from wms.eval.run import CheckpointPredictorFactory
    from wms.eval.score import ScoreSpec, score_states
    path = SCORE / "step0.json"
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("pass"):
            return cached
    records = quick20(SETS["wall_same"])
    factory = CheckpointPredictorFactory(str(C1_PATH), entries=False, entries_v2=False)
    rows = score_states(records, factory, ScoreSpec(horizons_s=HORIZONS, predict_horizon_s=120.0,
                                                    seed=0), workers=workers, inflight=workers * 2)
    vals = [float(r["arms"][tested_arm(r)]["car"]["60.0"]["hdv"]["ds_med"]) for r in rows
            if not (r["arms"].get(tested_arm(r)) or {}).get("error")]
    observed = statistics.median(vals) if vals else None
    out = {"scorer": "E:/agent-work/r5-score/src wms.eval.score.score_states",
           "set": "wall_same h60 quick-20", "n_states": len(records),
           "c1_median_hdv_abs_ds_60m": observed, "expected_m": STEP0_EXPECTED,
           "absolute_tolerance_m": 0.001,
           "pass": observed is not None and abs(observed - STEP0_EXPECTED) <= 0.001,
           "errors": [r for r in rows if any(v.get("error") for v in r.get("arms", {}).values())]}
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# incremental per-state scoring
# --------------------------------------------------------------------------- #
def score_states_cached(tag: str, ckpt: Path, set_name: str, records: list[dict],
                        workers: int, limit: int | None = None) -> list[dict]:
    """Score missing states, merge with the cache, persist incrementally."""
    from wms.eval.run import CheckpointPredictorFactory
    from wms.eval.score import ScoreSpec, score_states
    cache = SCORE / "cache" / f"{tag}__{set_name}__{sha8(ckpt)}.json"
    done: dict[tuple, dict] = {}
    if cache.is_file():
        try:
            for row in json.loads(cache.read_text(encoding="utf-8"))["states"]:
                done[state_key(row)] = row
        except (OSError, json.JSONDecodeError):
            done = {}
    todo = [r for r in records if state_key(r) not in done]
    if limit is not None:
        todo = todo[:limit]
    if todo:
        print(f"scoring {tag} on {set_name}: {len(todo)} states "
              f"({len(done)} cached)", flush=True)
        factory = CheckpointPredictorFactory(str(ckpt), entries=False, entries_v2=False)
        spec = ScoreSpec(horizons_s=HORIZONS, predict_horizon_s=120.0, seed=0)
        rows = score_states(todo, factory, spec, workers=workers, inflight=workers * 2)
        for row in rows:
            done[state_key(row)] = row
        cache.parent.mkdir(parents=True, exist_ok=True)
        ordered = [done[state_key(r)] for r in records if state_key(r) in done]
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps({"summary": None, "states": ordered}, separators=(",", ":")),
                       encoding="utf-8")
        tmp.replace(cache)
    else:
        print(f"scoring {tag} on {set_name}: fully cached ({len(done)} states)", flush=True)
    return [done[state_key(r)] for r in records if state_key(r) in done]


def stats_for(rows: list[dict]) -> dict:
    from wms.eval.stats import summarize
    result = {}
    for h in HORIZONS:
        result[str(int(h))] = {}
        for metric, field in (("ds", "ds_med"), ("dv", "dv_med")):
            vals = []
            for row in rows:
                arm = tested_arm(row)
                block = ((row.get("arms") or {}).get(arm) or {}).get("car", {}).get(str(h), {}).get("hdv", {})
                v = block.get(field)
                if v is not None and np.isfinite(v):
                    vals.append(float(v))
            result[str(int(h))][metric] = summarize(vals)
    blows = n_blow = 0
    for row in rows:
        arm = tested_arm(row)
        car = (row.get("arms") or {}).get(arm, {}).get("car", {})
        a = car.get("60.0", {}).get("hdv", {}).get("ds_med")
        b = car.get("120.0", {}).get("hdv", {}).get("ds_med")
        if a is not None and b is not None:
            n_blow += 1
            blows += int(a > 50 and b > 100)
    result["blowups"] = {"n": blows, "denominator": n_blow,
                         "rule": "HDV median |ds| >50m at 60s AND >100m at 120s"}
    return result


# --------------------------------------------------------------------------- #
# fingerprints (parallel; one pool per model, permanent step tracer per worker)
# --------------------------------------------------------------------------- #
_FW: dict = {}


def _fp_init(tag: str, ckpt: str, trace: bool) -> None:
    import torch
    torch.set_num_threads(1)
    import types as _types
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    predictor = WmsPredictor.from_checkpoint(ckpt, RolloutFlags(entries=False, entries_v2=False))
    records: list[dict] = []

    if trace:
        original = predictor.model.step

        def step(self, state, **kwargs):
            out = original(state, **kwargs)
            present = state.get("present")
            is_cav = state.get("is_cav")
            slots = state["s_m"].shape[-1]
            raw = out.get("raw_residual_mps2")
            records.append({
                "present": (present.detach().cpu().numpy()[0] if present is not None
                            else np.ones(slots, bool)),
                "is_cav": (is_cav.detach().cpu().numpy()[0] if is_cav is not None
                           else np.zeros(slots, bool)),
                "raw": (raw.detach().cpu().numpy()[0] if raw is not None
                        else np.full(slots, np.nan, np.float32)),
            })
            return out

        predictor.model.step = _types.MethodType(step, predictor.model)
    _FW["predictor"] = predictor
    _FW["trace"] = records
    _FW["tag"] = tag


def _fp_work(record: dict) -> dict:
    from wms.eval.normalize import recorded_for
    predictor, trace = _FW["predictor"], _FW["trace"]
    del trace[:]
    t0 = float(record["t0_s"])
    arm = ARM if ARM in (record.get("recorded") or {}) else next(iter(record["recorded"]))
    err = None
    try:
        pred = predictor.predict_states(record, {**record.get("cav_controllers", {}), "arm": arm},
                                        120.0, 0)
        rec = recorded_for(record, arm)
        payload = _fp_state(pred, rec, t0, trace)
    except Exception as exc:  # keep the batch alive
        import traceback
        err = f"{type(exc).__name__}: {exc}: {traceback.format_exc()[-400:]}"
        payload = None
    return {"key": state_key(record), "payload": payload, "error": err}


def _frame_arrays(frames: list[dict], recorded: list[dict], t0: float):
    ids = sorted({str(v["id"]) for fr in frames + recorded for v in fr.get("vehicles", [])})
    idx = {x: i for i, x in enumerate(ids)}
    n = int(round(120 / 0.5))
    arrays = {}
    for prefix, seq in (("p", frames), ("r", recorded)):
        lane = np.full((n + 1, len(ids)), -1, dtype=np.int16)
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
                speed[k, j] = float(v["v"])
                present[k, j] = True
                cav[k, j] = bool(v.get("is_cav", False))
        arrays[prefix] = (lane, speed, present, cav)
    return ids, arrays


def _runs(flag: np.ndarray) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(flag):
        if flag[i]:
            j = i
            while j + 1 < len(flag) and flag[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _lane_rate(p_lane, r_lane, p_pres, r_pres, p_cav, r_cav, lo: int, hi: int) -> dict:
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
    return {"pred_changes": changes, "recorded_changes": rec_changes,
            "pred_hdv_steps": steps, "recorded_hdv_steps": rec_steps}


def _fp_state(pred: list[dict], rec: list[dict], t0: float, trace: list[dict]) -> dict:
    ids, a = _frame_arrays(pred, rec, t0)
    pl, pv, pp, pc = a["p"]
    rl, rv, rp, rc = a["r"]
    corr = {"0-60": [], "60-120": []}
    for bucket, bounds in (("0-60", (0, 120)), ("60-120", (120, 240))):
        for st in trace[bounds[0]:bounds[1]]:
            live = np.asarray(st["present"], bool) & ~np.asarray(st["is_cav"], bool)
            corr[bucket].append(np.abs(np.asarray(st["raw"], float))[live])
    mask_steps = pp[1:121] & rp[1:121] & ~rc[1:121] & ~pc[1:121]
    signed_dv = ((pv[1:121] - rv[1:121])[mask_steps].reshape(-1)
                 if mask_steps.any() else np.zeros(0))
    wrong = {}
    for h in (60, 120):
        k = h * 2
        mask = pp[k] & rp[k] & ~rc[k] & (rl[k] >= 0)
        wrong[str(h)] = [int((mask & (pl[k] != rl[k])).sum()), int(mask.sum())]
    rates = {key: _lane_rate(pl, rl, pp, rp, pc, rc, lo, hi)
             for key, (lo, hi) in (("0-60", (0, 120)), ("60-120", (120, 240)))}
    valid = pp[1:121] & pp[:120] & rp[1:121] & rp[:120] & ~rc[1:121] & ~rc[:120] & ~pc[1:121] & ~pc[:120]
    ap, ar = np.diff(pv[:121], axis=0) / 0.5, np.diff(rv[:121], axis=0) / 0.5
    brakes = {"recorded_events": 0, "recalled_events": 0, "pred_events": 0,
              "false_events": 0, "vehicle_minutes": 0.0}
    for j in range(valid.shape[1]):
        v = valid[:, j]
        if not v.any():
            continue
        bp, br = (ap[:, j] < -3.0) & v, (ar[:, j] < -3.0) & v
        brakes["vehicle_minutes"] += float(v.sum() * 0.5 / 60)
        pruns, rruns = _runs(bp), _runs(br)
        brakes["pred_events"] += len(pruns)
        brakes["recorded_events"] += len(rruns)
        for s0, s1 in rruns:
            if bp[max(0, s0 - 4):min(120, s1 + 5)].any():
                brakes["recalled_events"] += 1
        for s0, s1 in pruns:
            if not br[max(0, s0 - 4):min(120, s1 + 5)].any():
                brakes["false_events"] += 1
    return {"corr": {k: np.concatenate(v) if v else np.zeros(0, np.float32)
                     for k, v in corr.items()},
            "signed_dv": signed_dv.astype(np.float32),
            "lane": rates, "wrong": wrong, "brakes": brakes}


def fingerprints(ckpt: Path, records: list[dict], workers: int, tag: str) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    payloads: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_fp_init,
                             initargs=(tag, str(ckpt), True)) as pool:
        for row in pool.map(_fp_work, records, chunksize=1):
            if row["error"]:
                print(f"  fingerprint error {row['key']}: {row['error'][:160]}", flush=True)
            payloads.append(row)
    corr = {"0-60": [], "60-120": []}
    signed, lane, wrong, brakes = [], {}, {60: [0, 0], 120: [0, 0]}, {
        "recorded_events": 0, "recalled_events": 0, "pred_events": 0,
        "false_events": 0, "vehicle_minutes": 0.0}
    for row in payloads:
        p = row.get("payload")
        if p is None:
            continue
        for k in corr:
            corr[k].append(p["corr"][k])
        signed.append(p["signed_dv"])
        for key, r in p["lane"].items():
            blk = lane.setdefault(key, {"pred_changes": 0, "recorded_changes": 0,
                                        "pred_hdv_steps": 0, "recorded_hdv_steps": 0})
            for f in blk:
                blk[f] += r[f]
        for h in (60, 120):
            wrong[h][0] += p["wrong"][str(h)][0]
            wrong[h][1] += p["wrong"][str(h)][1]
        for f in brakes:
            brakes[f] += p["brakes"][f]
    residual = {}
    for period, vals in corr.items():
        flat = np.concatenate(vals) if vals else np.zeros(0)
        residual[period] = {"mean_abs_g_da": float(flat.mean()) if flat.size else None,
                            "p95_abs_g_da": float(np.percentile(flat, 95)) if flat.size else None,
                            "n_hdv_steps": int(flat.size)}
    flat_signed = np.concatenate(signed) if signed else np.zeros(0)
    lane_summary = {}
    for key, blk in lane.items():
        lane_summary[key] = {
            "pred_changes": blk["pred_changes"], "recorded_changes": blk["recorded_changes"],
            "pred_per_hdv_per_10s": (blk["pred_changes"] / (blk["pred_hdv_steps"] / 20.0)
                                     if blk["pred_hdv_steps"] else None),
            "recorded_per_hdv_per_10s": (blk["recorded_changes"] / (blk["recorded_hdv_steps"] / 20.0)
                                         if blk["recorded_hdv_steps"] else None),
            "pred_hdv_steps": blk["pred_hdv_steps"], "recorded_hdv_steps": blk["recorded_hdv_steps"]}
    vm = brakes["vehicle_minutes"]
    brakes["event_recall"] = brakes["recalled_events"] / brakes["recorded_events"] \
        if brakes["recorded_events"] else None
    brakes["false_brake_events_per_vehicle_min"] = brakes["false_events"] / vm if vm else None
    brakes["n_states"] = len(payloads)
    return {"residual": residual,
            "mean_signed_speed_error_0_60_mps": float(flat_signed.mean()) if flat_signed.size else None,
            "lane_changes_per_hdv_per_10s": lane_summary,
            "hard_brake": brakes,
            "wrong_lane_share": {str(h): (wrong[h][0] / wrong[h][1] if wrong[h][1] else None)
                                 for h in wrong}}


def gate_fingerprint(ckpt: Path) -> dict:
    import torch
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    predictor = WmsPredictor.from_checkpoint(ckpt, RolloutFlags(entries=False, entries_v2=False))
    model = predictor.model
    gate_curve = {}
    for k in (10, 20, 60, 120, 240):
        try:
            val = model.gate(k)
            gate_curve[str(k)] = float(np.asarray(val.detach().cpu() if hasattr(val, "detach")
                                                  else val).reshape(-1)[0])
        except Exception:
            gate_curve[str(k)] = None
    params = {"kind": getattr(model, "gate_kind", None)}
    if hasattr(model, "gate_g_max_logit"):
        params.update({"g_max": float(torch.sigmoid(model.gate_g_max_logit).detach().cpu()),
                       "tau_steps": float(torch.nn.functional.softplus(model.gate_tau_raw).detach().cpu())})
    lane_curve = {}
    for k in (10, 20, 60, 120, 240):
        try:
            val = model.lane_gate(k)
            lane_curve[str(k)] = float(np.asarray(val.detach().cpu() if hasattr(val, "detach")
                                                  else val).reshape(-1)[0])
        except Exception:
            lane_curve[str(k)] = None
    lane_params = {"kind": getattr(model, "lane_gate_kind", None)}
    if hasattr(model, "lane_gate_g_max_logit"):
        lane_params.update({
            "g_max": float(torch.sigmoid(model.lane_gate_g_max_logit).detach().cpu()),
            "tau_steps": float(torch.nn.functional.softplus(model.lane_gate_tau_raw).detach().cpu())})
    return {"gate_curve": gate_curve, "gate_parameter_state": params,
            "lane_gate_curve": lane_curve, "lane_gate_parameter_state": lane_params,
            "lane_logit_cap": getattr(model, "lane_logit_cap", None),
            "rollout_mode": getattr(model, "rollout_mode", None)}


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def fmt(v, digits=2):
    return "n/a" if v is None else f"{v:.{digits}f}"


def fmt_ci(block, digits=2):
    if not block or block.get("point") is None:
        return "n/a"
    return f"{fmt(block['point'], digits)} [{fmt(block.get('lo'), digits)}, {fmt(block.get('hi'), digits)}]"


def write_report(results: dict, out_path: Path) -> None:
    lines = ["# Round 5 (WM-S-R5-20261012) vs C1 / R4 — scoring summary", ""]
    lines.append("Clean scorer `E:/agent-work/r5-score/src` (`wms.eval.score.score_states`), pairing "
                 "world+key+source_arm+t0, CPU, teacher-forced CAVs, entries off. "
                 "C1/R4 rows from the round-4 caches (real-history 120: CHK-FADE confirmation rows); "
                 f"baseline gate {'PASS' if results['baseline_gate']['pass'] else 'FAIL'} "
                 "(C1 8.71/99.84, R4 9.96/110.69); "
                 f"step0 C1 quick-20 60 s = {fmt(results['step0']['c1_median_hdv_abs_ds_60m'], 4)} m "
                 f"({'pass' if results['step0']['pass'] else 'FAIL'}).")
    lines.append("")
    lines.append("## Accuracy (median HDV |ds|, m) and paired differences (95% CI over worlds)")
    lines.append("| set | model | 10 s | 30 s | 60 s | 120 s | blow-ups |")
    lines.append("|---|---|---|---|---|---|---|")
    set_labels = {"wall_same": "wall_same (67)", "wall_pilot": "wall_pilot (53)",
                  "wall_same_real": "real-history (120)"}
    for set_name, models in results["evaluation"]["sets"].items():
        first = True
        for tag in results["model_order"]:
            block = results["models"].get(tag, {}).get(set_name)
            if not block:
                continue
            s = block["summary"]
            def g(h, m="ds"):
                v = s[str(h)][m]["median"]
                return fmt(v, 2) if v is not None else "n/a"
            label = set_labels.get(set_name, set_name) if first else ""
            first = False
            lines.append(f"| {label} | {tag} | {g(10)} | {g(30)} | {g(60)} | {g(120)} | "
                         f"{s['blowups']['n']}/{s['blowups']['denominator']} |")
        for tag in results["model_order"]:
            if not tag.startswith("R5"):
                continue
            for ref in ("C1", "R4"):
                blk = results["paired"].get(set_name, {}).get(f"{tag}_minus_{ref}")
                if not blk:
                    continue
                cells = [fmt_ci(blk[str(int(h))]) for h in HORIZONS]
                lines.append(f"| | {tag} − {ref} | {cells[0]} | {cells[1]} | {cells[2]} | "
                             f"{cells[3]} | |")
    lines.append("")
    fp = results.get("fingerprints_wall_same") or {}
    have = [t for t in results["model_order"] if t in fp]
    if not have:
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return
    lines.append("## Fingerprints (wall_same 67)")
    lines.append("| | " + " | ".join(have) + " | recorded |")
    lines.append("|---" * (len(have) + 2) + "|")

    def row(label, extract, rec=""):
        cells = []
        for t in have:
            try:
                cells.append(extract(fp[t]) or "n/a")
            except Exception:
                cells.append("n/a")
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {rec} |")

    row("gate g(k) 10/20/60/120/240",
        lambda f: "/".join(fmt(f["gate_curve"][str(k)], 3) for k in (10, 20, 60, 120, 240)))
    row("learned g_max / tau (accel gate)",
        lambda f: (f"{fmt(f['gate_parameter_state'].get('g_max'), 3)} / "
                   f"{fmt(f['gate_parameter_state'].get('tau_steps'), 1)}"
                   if f["gate_parameter_state"].get("g_max") is not None
                   else f"({f['gate_parameter_state'].get('kind')})"))
    row("lane gate g(k) 10/20/60/120/240",
        lambda f: "/".join(fmt(f.get("lane_gate_curve", {}).get(str(k)), 3)
                           for k in (10, 20, 60, 120, 240)))
    row("learned lane g_max / tau",
        lambda f: (f"{fmt(f['lane_gate_parameter_state'].get('g_max'), 3)} / "
                   f"{fmt(f['lane_gate_parameter_state'].get('tau_steps'), 1)}"
                   if f.get("lane_gate_parameter_state", {}).get("g_max") is not None
                   else f"({f.get('lane_gate_parameter_state', {}).get('kind')})"))
    row("mean (p95) |g*da| 0-60 s, m/s^2",
        lambda f: f"{fmt(f['residual']['0-60']['mean_abs_g_da'], 2)} "
                  f"({fmt(f['residual']['0-60']['p95_abs_g_da'], 2)})")
    row("mean (p95) |g*da| 60-120 s, m/s^2",
        lambda f: f"{fmt(f['residual']['60-120']['mean_abs_g_da'], 2)} "
                  f"({fmt(f['residual']['60-120']['p95_abs_g_da'], 2)})")
    row("mean signed speed error 0-60 s, m/s",
        lambda f: fmt(f["mean_signed_speed_error_0_60_mps"], 2))
    row("lane changes /HDV/10 s, 0-60 s (TRUE def.)",
        lambda f: fmt(f["lane_changes_per_hdv_per_10s"]["0-60"]["pred_per_hdv_per_10s"], 3),
        rec=fmt(next((fp[t]["lane_changes_per_hdv_per_10s"]["0-60"]["recorded_per_hdv_per_10s"]
                      for t in have if fp[t].get("lane_changes_per_hdv_per_10s")), None), 3))
    row("lane changes /HDV/10 s, 60-120 s",
        lambda f: fmt(f["lane_changes_per_hdv_per_10s"]["60-120"]["pred_per_hdv_per_10s"], 3),
        rec=fmt(next((fp[t]["lane_changes_per_hdv_per_10s"]["60-120"]["recorded_per_hdv_per_10s"]
                      for t in have if fp[t].get("lane_changes_per_hdv_per_10s")), None), 3))
    row("wrong-lane share 60 / 120 s",
        lambda f: f"{fmt(f['wrong_lane_share']['60'], 3)} / {fmt(f['wrong_lane_share']['120'], 3)}")
    row("hard-brake recall / false brakes per veh-min",
        lambda f: (f"{fmt(f['hard_brake']['event_recall'], 3)} / "
                   f"{fmt(f['hard_brake']['false_brake_events_per_vehicle_min'], 2)}"))
    lines.append("")
    lines.append("Provenance: C1/R4 rollout fingerprints copied from "
                 "`WM-S-R4-20261011/score/results.json` (identical models/sets); gate curves "
                 "recomputed. R5 fingerprints computed here.")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="round-5 wm_s.pt (last)")
    ap.add_argument("--ckpt-best", default=None, help="round-5 wm_s_best.pt (optional)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--step0-only", action="store_true")
    ap.add_argument("--no-fingerprints", action="store_true")
    ap.add_argument("--fp-limit", type=int, default=None,
                    help="smoke: fingerprint only the first N wall_same states")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke: score only the first N states per set")
    ap.add_argument("--sets", nargs="+", default=["wall_same", "wall_pilot", "wall_same_real"])
    args = ap.parse_args()
    ckpt = Path(args.ckpt).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be in 1..8")

    results = {"schema": "wm_s_round5_score/1",
               "evaluation": {"workers": args.workers, "entries": False, "cav_mode": "teacher",
                              "horizons_s": list(HORIZONS),
                              "paired_key": ["world", "key", "source_arm", "t0_s"],
                              "scorer_src": str(R5_SRC), "sets": {}},
               "runtimes_s": {}}
    t_start = time.time()
    r0 = step0(args.workers)
    print(json.dumps({"step0": {k: r0[k] for k in ("pass", "c1_median_hdv_abs_ds_60m")}}), flush=True)
    results["step0"] = r0
    if not r0["pass"] or args.step0_only:
        (SCORE / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        return 0 if r0["pass"] else 2
    results["baseline_gate"] = baseline_gate()
    if not results["baseline_gate"]["pass"]:
        (SCORE / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        return 3
    print(json.dumps({"baseline_gate": "pass"}), flush=True)

    models = {"R5-last": ckpt}
    if args.ckpt_best:
        best = Path(args.ckpt_best).resolve()
        if best.is_file():
            models["R5-best"] = best
    results["model_order"] = ["C1", "R4"] + list(models)
    results["models"] = {"C1": {}, "R4": {}}
    results["paired"] = {}
    comparators = {}
    for set_name in args.sets:
        records = iter_records(SETS[set_name])
        if set_name == "wall_same_real" and args.limit:
            records = quick20(SETS[set_name])[:args.limit]
        elif args.limit:
            records = records[:args.limit]
        results["evaluation"]["sets"][set_name] = {
            "n_states": len(records),
            "n_worlds": len({str(r.get("world")) for r in records})}
        c1_rows, r4_rows = comparators.setdefault(set_name, load_comparator(set_name))
        if args.limit:
            keep = {state_key(r) for r in records}
            c1_rows = [r for r in c1_rows if state_key(r) in keep]
            r4_rows = [r for r in r4_rows if state_key(r) in keep]
        results["models"]["C1"][set_name] = {"summary": stats_for(c1_rows), "n_states": len(c1_rows)}
        results["models"]["R4"][set_name] = {"summary": stats_for(r4_rows), "n_states": len(r4_rows)}
        for tag, path in models.items():
            t_set = time.time()
            rows = score_states_cached(tag, path, set_name, records, args.workers)
            results["models"].setdefault(tag, {})[set_name] = {"summary": stats_for(rows),
                                                               "n_states": len(rows),
                                                               "states": rows,
                                                               "checkpoint": str(path),
                                                               "sha8": sha8(path)}
            results["runtimes_s"][f"{tag}__{set_name}"] = round(time.time() - t_set, 1)
        refs = {"C1": c1_rows, "R4": r4_rows}
        from wms.eval.report import paired_bootstrap
        for tag in models:
            mine = results["models"][tag][set_name]["states"]
            for ref_name, ref_rows in refs.items():
                results["paired"].setdefault(set_name, {})[f"{tag}_minus_{ref_name}"] = {
                    str(int(h)): paired_bootstrap(mine, ref_rows, ARM, h, n_boot=2000, seed=0)
                    for h in HORIZONS}

    out_path = SCORE / "results.json"
    if not args.no_fingerprints:
        t_fp = time.time()
        wall = iter_records(SETS["wall_same"])
        results["fingerprints_wall_same"] = {}
        r4_results = json.loads((R4_SCORE / "results.json").read_text(encoding="utf-8"))
        for name, key in (("C1", "C1"), ("R4", "R4-last")):
            if key in r4_results.get("fingerprints_wall_same", {}):
                results["fingerprints_wall_same"][name] = dict(
                    r4_results["fingerprints_wall_same"][key])
        for tag, path in models.items():
            wall_fp = wall[:args.fp_limit] if args.fp_limit else wall
            print(f"fingerprinting {tag} on wall_same: {len(wall_fp)} states", flush=True)
            results["fingerprints_wall_same"][tag] = fingerprints(path, wall_fp, args.workers, tag)
            _dump(out_path, results)
        for name, path in [("C1", C1_PATH), ("R4", R4_PATH)] + list(models.items()):
            if name in results["fingerprints_wall_same"]:
                results["fingerprints_wall_same"][name].update(gate_fingerprint(path))
        results["runtimes_s"]["fingerprints"] = round(time.time() - t_fp, 1)
    results["runtimes_s"]["total"] = round(time.time() - t_start, 1)
    _dump(out_path, results)
    write_report(results, SCORE / "ROUND5_SCORE.md")
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path} ({size_mb:.2f} MB) and ROUND5_SCORE.md", flush=True)
    return 0


def _dump(path: Path, results: dict) -> None:
    path.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
