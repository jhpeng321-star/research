"""CPU-only, clean-package WM-S scoring and diagnostics for R4.

All generated files stay beside this script. No training or simulation is performed.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import types
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCORE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np
from wms.eval.normalize import iter_windows
from wms.eval.predictor import RolloutFlags, WmsPredictor
from wms.eval.report import paired_bootstrap
from wms.eval.score import ScoreSpec, score_states
from wms.eval.stats import summarize

DATA = ROOT / "data_cache"
MODELS = {
    "C1": DATA / "ckpt_c1fast" / "wm_s.pt",
    "R4-last": ROOT / "runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt",
    "R4-best": ROOT / "runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s_best.pt",
    "R4-stage10": ROOT / "runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s_stage10.pt",
    "R4-stage30": ROOT / "runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s_stage30.pt",
    "R4-stage60": ROOT / "runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s_stage60.pt",
    "R4-stage120": ROOT / "runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s_stage120.pt",
}
SETS = {
    "wall_same": DATA / "wallc1_states_h60" / "wall_same.json",
    "wall_pilot": DATA / "wallc1_states_h60" / "wall_pilot.json",
    "wall_same_real": DATA / "wallc1_states_h60_real" / "wall_same.json",
}
HORIZONS = (10.0, 30.0, 60.0, 120.0)


def quick20(path: Path) -> list[dict]:
    rows = list(iter_windows(path))
    chosen = {}
    for r in sorted(rows, key=lambda x: (str(x.get("world")), float(x.get("t0_s", 0)), str(x.get("key")))):
        chosen.setdefault(str(r.get("world")), r)
    return [chosen[k] for k in sorted(chosen)[:20]]


def make_predictor(path: Path) -> WmsPredictor:
    return WmsPredictor.from_checkpoint(path, RolloutFlags(entries=False, entries_v2=False))


def score_one(name: str, records: list[dict], workers: int) -> list[dict]:
    from wms.eval.run import CheckpointPredictorFactory
    factory = CheckpointPredictorFactory(str(MODELS[name]), entries=False, entries_v2=False)
    spec = ScoreSpec(horizons_s=HORIZONS, predict_horizon_s=120.0, seed=0)
    return score_states(records, factory, spec, workers=workers, inflight=workers * 2)


def score_cached(name: str, set_name: str, records: list[dict], workers: int) -> dict:
    path = SCORE / "cache" / f"{name}__{set_name}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = score_one(name, records, workers)
    value = {"summary": stats_for(rows), "states": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    return value


def tested_arm(row: dict) -> str:
    arms = row.get("arms") or {}
    return "wall" if "wall" in arms else next(iter(arms))


def stats_for(rows: list[dict]) -> dict:
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
    blows = 0
    n_blow = 0
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


def true_lane_rate(p_lane, r_lane, p_pres, r_pres, p_cav, r_cav, lo: int, hi: int) -> dict:
    changes = steps = rec_changes = rec_steps = 0
    pred_filled, rec_filled = p_lane.copy(), r_lane.copy()
    for lanes in (pred_filled, rec_filled):
        for tick in range(1, lanes.shape[0]):
            bad = lanes[tick] < 0
            lanes[tick, bad] = lanes[tick - 1, bad]
    for k in range(lo + 1, hi + 1):
        valid = p_pres[k] & p_pres[k - 1] & r_pres[k] & r_pres[k - 1] & ~r_cav[k] & ~p_cav[k]
        for lanes, counts in ((pred_filled, "pred"), (rec_filled, "rec")):
            cur, prev = lanes[k], lanes[k - 1]
            good = valid & (cur >= 0) & (prev >= 0)
            ch = int((good & (cur != prev)).sum())
            st = int(good.sum())
            if counts == "pred":
                changes += ch
                steps += st
            else:
                rec_changes += ch
                rec_steps += st
    duration = (hi - lo) * 0.5
    # HDV-step count divided by 20 steps per 10 s is the observed-vehicle-bucket denominator.
    pred_rate = changes / (steps * 0.5 / 10) if steps else None
    rec_rate = rec_changes / (rec_steps * 0.5 / 10) if rec_steps else None
    return {"pred_changes": changes, "recorded_changes": rec_changes,
            "pred_hdv_steps": steps, "recorded_hdv_steps": rec_steps,
            "pred_per_hdv_per_10s": pred_rate, "recorded_per_hdv_per_10s": rec_rate,
            "duration_s": duration}


def _runs(flag: np.ndarray) -> list[tuple[int, int]]:
    out = []
    i = 0
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


@contextmanager
def trace_raw_residual(model):
    """Trace raw ``g*da`` before the bounded plant correction, without changing rollout outputs."""
    records = []
    original = model.step

    def step(self, state, **kwargs):
        out = original(state, **kwargs)
        def first(name, fallback=None):
            value = out.get(name, fallback)
            if value is None:
                return None
            return value.detach().cpu().numpy()[0]
        present = state.get("present")
        is_cav = state.get("is_cav")
        slots = state["s_m"].shape[-1]
        records.append({
            "present": (present.detach().cpu().numpy()[0] if present is not None else np.ones(slots, bool)),
            "is_cav": (is_cav.detach().cpu().numpy()[0] if is_cav is not None else np.zeros(slots, bool)),
            "raw_correction": first("raw_residual_mps2"),
        })
        return out

    model.step = types.MethodType(step, model)
    try:
        yield records
    finally:
        del model.step


def brake_diag(model: str, records: list[dict], workers: int) -> dict:
    """Hard-brake recall and false events/minute over 0-60 s, clean scorer predictions."""
    from concurrent.futures import ProcessPoolExecutor
    # Rollout and trace in the parent process per worker group to retain exact state arrays.
    # Scoring is CPU-limited by the caller; each model is loaded once.
    predictor = make_predictor(MODELS[model])
    result = {"recorded_events": 0, "recalled_events": 0, "pred_events": 0,
              "false_events": 0, "vehicle_minutes": 0.0}
    for record in records:
        t0 = float(record["t0_s"])
        arm = "wall" if "wall" in (record.get("recorded") or {}) else next(iter(record["recorded"]))
        with trace_raw_residual(predictor.model) as trace:
            pred = predictor.predict_states(record, {**record.get("cav_controllers", {}), "arm": arm}, 120.0, 0)
        rec = __import__("wms.eval.normalize", fromlist=["recorded_for"]).recorded_for(record, arm)
        _, a = _frame_arrays(pred, rec, t0)
        pv, pp, pc = a["p"][1], a["p"][2], a["p"][3]
        rv, rp, rc = a["r"][1], a["r"][2], a["r"][3]
        valid = pp[1:121] & pp[:120] & rp[1:121] & rp[:120] & ~rc[1:121] & ~rc[:120] & ~pc[1:121] & ~pc[:120]
        ap = np.diff(pv[:121], axis=0) / 0.5
        ar = np.diff(rv[:121], axis=0) / 0.5
        for j in range(valid.shape[1]):
            v = valid[:, j]
            if not v[:120].any():
                continue
            bp, br = (ap[:, j] < -3.0) & v, (ar[:, j] < -3.0) & v
            result["vehicle_minutes"] += float(v[:120].sum() * 0.5 / 60)
            pred_runs = [r for r in _runs(bp[:120])]
            rec_runs = [r for r in _runs(br[:120])]
            result["pred_events"] += len(pred_runs)
            result["recorded_events"] += len(rec_runs)
            for s0, s1 in rec_runs:
                if bp[max(0, s0 - 4):min(240, s1 + 5)].any():
                    result["recalled_events"] += 1
            for s0, s1 in pred_runs:
                if not br[max(0, s0 - 4):min(240, s1 + 5)].any():
                    result["false_events"] += 1
    vm = result["vehicle_minutes"]
    result["event_recall"] = result["recalled_events"] / result["recorded_events"] if result["recorded_events"] else None
    result["false_brake_events_per_vehicle_min"] = result["false_events"] / vm if vm else None
    result["n_states"] = len(records)
    return result


def fingerprints(model: str, records: list[dict]) -> dict:
    predictor = make_predictor(MODELS[model])
    gate_curve = {}
    # Read learned saturating gate parameters from the backend's model tensors/config.
    gate_obj = predictor.model
    params = {}
    if hasattr(gate_obj, "gate_g_max_logit"):
        import torch
        params = {"g_max": float(torch.sigmoid(gate_obj.gate_g_max_logit).detach().cpu()),
                  "tau_steps": float(torch.nn.functional.softplus(gate_obj.gate_tau_raw).detach().cpu())}
    for k in (10, 20, 60, 120, 240):
        try:
            val = predictor.model.gate(k)
            gate_curve[str(k)] = float(np.asarray(val.detach().cpu() if hasattr(val, "detach") else val).mean())
        except Exception:
            gate_curve[str(k)] = None
    total_corr = {"0-60": [], "60-120": []}
    signed_dv, lc = [], {"0-60": [], "60-120": []}
    wrong = {60: [0, 0], 120: [0, 0]}
    brakes = {"recorded_events": 0, "recalled_events": 0, "pred_events": 0,
              "false_events": 0, "vehicle_minutes": 0.0}
    for record in records:
        t0 = float(record["t0_s"])
        arm = "wall" if "wall" in (record.get("recorded") or {}) else next(iter(record["recorded"]))
        with trace_raw_residual(predictor.model) as trace:
            pred = predictor.predict_states(record, {**record.get("cav_controllers", {}), "arm": arm}, 120.0, 0)
        rec = __import__("wms.eval.normalize", fromlist=["recorded_for"]).recorded_for(record, arm)
        ids, a = _frame_arrays(pred, rec, t0)
        pl, pv, pp, pc = a["p"]
        rl, rv, rp, rc = a["r"]
        for bucket, bounds in (("0-60", (0, 120)), ("60-120", (120, 240))):
            for st in trace[bounds[0]:bounds[1]]:
                live = np.asarray(st["present"], bool) & ~np.asarray(st["is_cav"], bool)
                corr = np.abs(np.asarray(st["raw_correction"], float)[live])
                total_corr[bucket].extend(corr.tolist())
        for lo, hi, key in ((0, 120, "0-60"), (120, 240, "60-120")):
            rate = true_lane_rate(pl, rl, pp, rp, pc, rc, lo, hi)
            lc[key].append(rate)
        for h in (60, 120):
            k = h * 2
            mask = pp[k] & rp[k] & ~rc[k] & (rl[k] >= 0)
            wrong[h][1] += int(mask.sum())
            wrong[h][0] += int((mask & (pl[k] != rl[k])).sum())
        mask_steps = pp[1:121] & rp[1:121] & ~rc[1:121] & ~pc[1:121]
        if mask_steps.any():
            signed_dv.extend((pv[1:121] - rv[1:121])[mask_steps].reshape(-1).tolist())
        valid = pp[1:121] & pp[:120] & rp[1:121] & rp[:120] & ~rc[1:121] & ~rc[:120] & ~pc[1:121] & ~pc[:120]
        ap, ar = np.diff(pv[:121], axis=0) / 0.5, np.diff(rv[:121], axis=0) / 0.5
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
    residual = {}
    for period, vals in total_corr.items():
        residual[period] = {"mean_abs_g_da": float(np.mean(vals)) if vals else None,
                            "p95_abs_g_da": float(np.percentile(vals, 95)) if vals else None,
                            "n_hdv_steps": len(vals)}
    lane_summary = {}
    for key, vals in lc.items():
        pred_changes = sum(v["pred_changes"] for v in vals)
        rec_changes = sum(v["recorded_changes"] for v in vals)
        pred_steps = sum(v["pred_hdv_steps"] for v in vals)
        rec_steps = sum(v["recorded_hdv_steps"] for v in vals)
        scale = 20.0
        lane_summary[key] = {
            "pred_changes": pred_changes, "recorded_changes": rec_changes,
            "pred_per_hdv_per_10s": pred_changes / (pred_steps / scale) if pred_steps else None,
            "recorded_per_hdv_per_10s": rec_changes / (rec_steps / scale) if rec_steps else None,
            "pred_hdv_steps": pred_steps, "recorded_hdv_steps": rec_steps}
    vm = brakes["vehicle_minutes"]
    brakes["event_recall"] = brakes["recalled_events"] / brakes["recorded_events"] if brakes["recorded_events"] else None
    brakes["false_brake_events_per_vehicle_min"] = brakes["false_events"] / vm if vm else None
    brakes["n_states"] = len(records)
    return {"gate_curve": gate_curve, "gate_parameter_state": params,
            "residual": residual,
            "mean_signed_speed_error_0_60_mps": float(np.mean(signed_dv)) if signed_dv else None,
            "lane_changes_per_hdv_per_10s": lane_summary,
            "hard_brake": brakes,
            "wrong_lane_share": {str(h): (wrong[h][0] / wrong[h][1] if wrong[h][1] else None)
                                 for h in wrong}}


def gate_fingerprint(model: str) -> tuple[dict, dict]:
    import torch
    predictor = make_predictor(MODELS[model])
    gate_curve = {str(k): float(predictor.model.gate(k).detach().cpu())
                  for k in (10, 20, 60, 120, 240)}
    params = {}
    if hasattr(predictor.model, "gate_g_max_logit"):
        params = {"g_max": float(torch.sigmoid(predictor.model.gate_g_max_logit).detach().cpu()),
                  "tau_steps": float(torch.nn.functional.softplus(predictor.model.gate_tau_raw).detach().cpu())}
    else:
        params = {"kind": "cumsum", "g_max": None, "tau_steps": None}
    return gate_curve, params


def step0(workers: int) -> dict:
    records = quick20(SETS["wall_same"])
    rows = score_one("C1", records, workers)
    vals = []
    for row in rows:
        arm = tested_arm(row)
        val = row["arms"][arm]["car"]["60.0"]["hdv"]["ds_med"]
        if val is not None:
            vals.append(float(val))
    observed = statistics.median(vals) if vals else None
    out = {"scorer": "src/wms wms.eval.score.score_states", "set": "wall_same h60 quick-20",
           "n_states": len(records), "c1_median_hdv_abs_ds_60m": observed,
           "expected_m": 11.9576, "absolute_tolerance_m": 0.001,
           "pass": observed is not None and abs(observed - 11.9576) <= 0.001,
           "errors": [r for r in rows if any(v.get("error") for v in r.get("arms", {}).values())]}
    (SCORE / "step0.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step0-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-fingerprints", action="store_true")
    ap.add_argument("--fingerprint-models", nargs="+", default=["C1", "R4-last", "R4-best"])
    ap.add_argument("--force-fingerprints", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be in 1..8")
    r0 = step0(args.workers)
    print(json.dumps({"step0": r0}, indent=2), flush=True)
    if not r0["pass"] or args.step0_only:
        return 0 if r0["pass"] else 2
    records_by_set = {name: list(iter_windows(path)) for name, path in SETS.items()}
    records_by_set["wall_same_real"] = quick20(SETS["wall_same_real"])
    results = {"schema": "wm_s_round4_diagnostic/1", "step0": r0,
               "evaluation": {"workers": args.workers, "entries": False,
                              "cav_mode": "teacher", "horizons_s": list(HORIZONS),
                              "paired_key": ["world", "key", "source_arm", "t0_s"],
                              "sets": {}},
               "models": {}, "paired_r4_minus_c1": {}, "fingerprints_wall_same": {}}
    for set_name, records in records_by_set.items():
        results["evaluation"]["sets"][set_name] = {"n_states": len(records),
                                                     "n_worlds": len({str(r.get("world")) for r in records})}
        names = ["C1", "R4-last", "R4-best"]
        if set_name == "wall_same_real":
            names = ["C1", "R4-last", "R4-best"]
        for name in names:
            print(f"scoring {name} on {set_name}: {len(records)} states", flush=True)
            results["models"].setdefault(name, {})[set_name] = score_cached(name, set_name, records, args.workers)
    # Stage snapshots are diagnostic 20-state scores at 60 s; all available horizons retained.
    qrecords = quick20(SETS["wall_same"])
    for name in ("R4-stage10", "R4-stage30", "R4-stage60", "R4-stage120"):
        print(f"scoring {name} on wall_same quick-20", flush=True)
        results["models"].setdefault(name, {})["wall_same_quick20"] = score_cached(
            name, "wall_same_quick20", qrecords, args.workers)
    from wms.eval.report import paired_bootstrap
    for set_name in ("wall_same", "wall_pilot", "wall_same_real"):
        c1 = results["models"]["C1"][set_name]["states"]
        for candidate in ("R4-last", "R4-best"):
            mine = results["models"][candidate][set_name]["states"]
            results["paired_r4_minus_c1"].setdefault(set_name, {})[candidate] = {
                str(int(h)): paired_bootstrap(mine, c1, "wall", h, n_boot=2000, seed=0)
                for h in HORIZONS}
    out_path = SCORE / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.no_fingerprints:
        print(f"saved scores to {out_path}", flush=True)
        return 0
    wall_full = list(iter_windows(SETS["wall_same"]))
    existing_path = SCORE / "results.json"
    if existing_path.is_file():
        try:
            old = json.loads(existing_path.read_text(encoding="utf-8"))
            results["fingerprints_wall_same"].update(old.get("fingerprints_wall_same", {}))
        except (OSError, json.JSONDecodeError):
            pass
    for name in args.fingerprint_models:
        if name not in ("C1", "R4-last", "R4-best"):
            raise SystemExit(f"unsupported fingerprint model: {name}")
        if name in results["fingerprints_wall_same"] and not args.force_fingerprints:
            continue
        print(f"fingerprinting {name} on wall_same: {len(wall_full)} states", flush=True)
        results["fingerprints_wall_same"][name] = fingerprints(name, wall_full)
        out_path = SCORE / "results.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, block in results["fingerprints_wall_same"].items():
        block["gate_curve"], block["gate_parameter_state"] = gate_fingerprint(name)
    out_path = SCORE / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
