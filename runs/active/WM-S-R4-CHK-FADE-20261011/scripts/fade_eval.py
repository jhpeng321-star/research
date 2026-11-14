"""Registered CPU-only residual-gate fade screen and one-time confirmation."""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_cache"
R4 = ROOT / "runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt"
C1 = DATA / "ckpt_c1fast/wm_s.pt"
HORIZONS = (10.0, 30.0, 60.0, 120.0)
SPECS = [(f"{scope}_s{start}_l{length}", scope, start, length)
         for scope in ("accel", "lane", "both")
         for start in (60, 120, 160) for length in (0, 40)]
SPECS += [("lane_cap_g20", "lane_cap", 20, 0)]
_W = {}


def ident(record):
    return (str(record.get("world")), round(float(record.get("t0_s", 0)), 3),
            str(record.get("key")), str(record.get("source_arm")))


def arm_for(record):
    rec = record.get("recorded") or {}
    return "wall" if "wall" in rec else next(iter(rec))


def factor(k, start, length):
    if hasattr(k, "detach") and k.ndim > 0:
        torch = _W["torch"]
        if length == 0:
            return (k <= start).to(dtype=torch.float32)
        return torch.where(k <= start, torch.ones_like(k, dtype=torch.float32),
                           torch.clamp(1.0 - (k.to(dtype=torch.float32) - start) / length,
                                       min=0.0, max=1.0))
    k = int(k.detach().cpu().item()) if hasattr(k, "detach") else int(k)
    if k <= start:
        return 1.0
    if length == 0:
        return 0.0
    return max(0.0, 1.0 - (k - start) / length)


def _init(model_name):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    _W["torch"] = torch
    _W["predictor"] = WmsPredictor.from_checkpoint(R4 if model_name == "R4" else C1,
                                                    RolloutFlags(entries=False, entries_v2=False),
                                                    device="cpu")
    _W["model_name"] = model_name


def _score_one(job):
    model_name, name, scope, start, length, record = job
    import numpy as np
    import torch
    from wms.eval.metrics import car_metrics, frame_at, present_by_id
    from wms.eval.normalize import recorded_for
    from wms.eval.score import car_compact
    predictor = _W["predictor"]
    model = predictor.model
    original_step = model.step
    original_residual = model._residual_prediction
    active = {"k": 0}
    original_gate = model.gate

    def step_hook(self, state, **kwargs):
        active["k"] = kwargs.get("k_steps", 0)
        return original_step(state, **kwargs)

    def residual_hook(self, *args, **kwargs):
        base = original_residual(*args, **kwargs)
        f = factor(active["k"], start, length) if scope != "lane_cap" else 1.0
        if scope == "accel":
            base = dict(base)
            base["accel_delta_mps2"] = base["accel_delta_mps2"] * f
        if scope == "lane":
            base = dict(base)
            base["lane_logit_delta"] = base["lane_logit_delta"] * f
        if scope == "lane_cap":
            g = original_gate(active["k"])
            cap = original_gate(20)
            ratio = torch.where(g.abs() > 1e-12, torch.minimum(g, cap) / g, torch.ones_like(g))
            base = dict(base)
            base["lane_logit_delta"] = base["lane_logit_delta"] * ratio
        return base

    if scope == "both":
        def gate_hook(k):
            return original_gate(k) * factor(k, start, length)
        model.gate = gate_hook
    elif scope == "lane_cap":
        # Acceleration retains the original gate; only lane delta is scaled in residual_hook.
        pass
    model.step = types.MethodType(step_hook, model)
    model._residual_prediction = types.MethodType(residual_hook, model)
    try:
        a = arm_for(record)
        frames = predictor.predict_states(record, {**record.get("cav_controllers", {}), "arm": a}, 120.0, 0)
        recorded = recorded_for(record, a)
        car = car_compact(car_metrics(frames, recorded, t0_s=float(record["t0_s"]), horizons=HORIZONS))
        # Return only bounded state-level aggregates; never send trajectory arrays to parent.
        t0 = float(record["t0_s"])
        horizons, wrong_lane = {}, {}
        for h in HORIZONS:
            pred = present_by_id(frame_at(frames, t0 + h) or {})
            truth = present_by_id(frame_at(recorded, t0 + h) or {})
            ids = [v for v in pred.keys() & truth.keys() if not bool(truth[v].get("is_cav"))]
            horizons[str(int(h))] = {"ds_med": car[str(float(h))]["hdv"]["ds_med"],
                                     "dv": car[str(float(h))]["hdv"]["dv_med"], "n": len(ids)}
            comparable = [v for v in ids if truth[v].get("lane") is not None and int(truth[v]["lane"]) >= 0]
            wrong_lane[str(int(h))] = {"wrong": sum(pred[v].get("lane") != truth[v].get("lane") for v in comparable),
                                       "n": len(comparable)}
        lane_changes = {"0-60": {"changes": 0, "hdv_steps": 0},
                        "60-120": {"changes": 0, "hdv_steps": 0}}
        previous_lane, previous_valid, filled_lane = {}, set(), {}
        for fr in frames:
            now = float(fr["time_s"])
            rec_fr = frame_at(recorded, now)
            rec = present_by_id(rec_fr or {})
            cur = present_by_id(fr)
            valid = {vid for vid in cur.keys() & rec.keys() if not bool(rec[vid].get("is_cav"))}
            current_lane = {}
            for vid, vehicle in cur.items():
                raw = vehicle.get("lane", -1)
                if raw is not None and int(raw) >= 0:
                    filled_lane[vid] = int(raw)
                current_lane[vid] = filled_lane.get(vid, -1)
            if now > t0:
                for vid in valid & previous_valid:
                    old, value = previous_lane.get(vid, -1), current_lane.get(vid, -1)
                    if old >= 0 and value >= 0:
                        bucket = "0-60" if now <= t0 + 60 else "60-120"
                        lane_changes[bucket]["hdv_steps"] += 1
                        lane_changes[bucket]["changes"] += int(value != old)
            previous_lane, previous_valid = current_lane, valid
        p120 = present_by_id(frame_at(frames, t0 + 120.0) or {})
        r120 = present_by_id(frame_at(recorded, t0 + 120.0) or {})
        survivors = [v for v in p120.keys() & r120.keys() if not bool(r120[v].get("is_cav"))]
        blowup = (horizons["60"]["ds_med"] is not None and horizons["120"]["ds_med"] is not None
                  and horizons["60"]["ds_med"] > 50 and horizons["120"]["ds_med"] > 100)
        return {"variant": name, "world": str(record.get("world")), "key": record.get("key"),
                "source_arm": record.get("source_arm"), "t0_s": t0, "horizons": horizons,
                "blowup": blowup, "wrong_lane": wrong_lane, "lane_changes": lane_changes,
                "survivors": {"n": len(survivors),
                    "signed_ds_sum": sum(float(p120[v]["s"]) - float(r120[v]["s"]) for v in survivors),
                    "signed_dv_sum": sum(float(p120[v]["v"]) - float(r120[v]["v"]) for v in survivors)},
                "error": None}
    except Exception as exc:
        return {"variant": name, "world": str(record.get("world")), "key": record.get("key"),
                "source_arm": record.get("source_arm"), "t0_s": float(record["t0_s"]),
                "error": f"{type(exc).__name__}: {exc}"}
    finally:
        model.step = original_step
        model._residual_prediction = original_residual
        model.gate = original_gate


def run_variant(model, variant, records, workers):
    name, scope, start, length = variant
    jobs = [(model, name, scope, start, length, r) for r in records]
    rows = []
    peak = {"parent_mb": 0.0, "parent_plus_workers_mb": 0.0}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(model,)) as pool:
        futures = [pool.submit(_score_one, job) for job in jobs]
        for f in as_completed(futures):
            rows.append(f.result())
            try:
                sample = memory_snapshot()
                for key in peak:
                    peak[key] = max(peak[key], sample[key])
            except Exception:
                pass
    rows.sort(key=lambda r: (r["world"], r["t0_s"], str(r["key"])))
    run_variant.last_memory = peak
    return rows


def memory_snapshot():
    import psutil
    parent = psutil.Process()
    children = parent.children(recursive=True)
    parent_mb = parent.memory_info().rss / (1024 * 1024)
    total_mb = parent_mb + sum(p.memory_info().rss for p in children if p.is_running()) / (1024 * 1024)
    return {"parent_mb": round(parent_mb, 1), "parent_plus_workers_mb": round(total_mb, 1)}


def records(path):
    from wms.eval.normalize import iter_windows
    return list(iter_windows(path))


def vals(rows, h, metric="ds_med"):
    key = str(float(h))
    return [float(r["car"][key]["hdv"][metric]) for r in rows
            if not r.get("error") and r["car"][key]["hdv"][metric] is not None]


def summarize(rows):
    result = {}
    for h in HORIZONS:
        values = [float(r["horizons"][str(int(h))]["ds_med"]) for r in rows
                  if not r.get("error") and r["horizons"][str(int(h))]["ds_med"] is not None]
        result[str(int(h))] = {"n": len(values), "median": statistics.median(values) if values else None}
    blows = 0
    denom = 0
    for r in rows:
        if r.get("error"):
            continue
        a = r["horizons"]["60"]["ds_med"]
        b = r["horizons"]["120"]["ds_med"]
        if a is not None and b is not None:
            denom += 1
            blows += int(a > 50 and b > 100)
    result["blowups"] = {"n": blows, "denominator": denom,
                         "rule": "HDV median |ds| >50m at 60s AND >100m at 120s"}
    result["errors"] = sum(bool(r.get("error")) for r in rows)
    result["wrong_lane"] = {str(int(h)): {"wrong": sum(r.get("wrong_lane", {}).get(str(int(h)), {}).get("wrong", 0) for r in rows),
                                                    "n": sum(r.get("wrong_lane", {}).get(str(int(h)), {}).get("n", 0) for r in rows)} for h in HORIZONS}
    for values in result["wrong_lane"].values():
        values["share"] = values["wrong"] / values["n"] if values["n"] else None
    result["lane_changes"] = {w: {"changes": sum(r.get("lane_changes", {}).get(w, {}).get("changes", 0) for r in rows),
                                  "hdv_steps": sum(r.get("lane_changes", {}).get(w, {}).get("hdv_steps", 0) for r in rows)}
                              for w in ("0-60", "60-120")}
    for values in result["lane_changes"].values():
        values["per_hdv_per_10s"] = 20 * values["changes"] / values["hdv_steps"] if values["hdv_steps"] else None
    survivor_n = sum(r.get("survivors", {}).get("n", 0) for r in rows)
    ds_sum = sum(r.get("survivors", {}).get("signed_ds_sum", 0) for r in rows)
    dv_sum = sum(r.get("survivors", {}).get("signed_dv_sum", 0) for r in rows)
    result["survivors"] = {"n": survivor_n, "signed_ds_sum": ds_sum, "signed_dv_sum": dv_sum,
                            "signed_ds_mean": ds_sum / survivor_n if survivor_n else None,
                            "signed_dv_mean": dv_sum / survivor_n if survivor_n else None}
    return result


def paired(candidate, baseline, h):
    from wms.eval.stats import world_bootstrap_median
    field = str(float(h))
    def keyed(rows):
        out = {}
        for r in rows:
            if not r.get("error") and r["horizons"][str(int(h))]["ds_med"] is not None:
                out[(r["world"], r["key"], r["source_arm"], round(r["t0_s"], 3))] = float(r["horizons"][str(int(h))]["ds_med"])
        return out
    aa, bb = keyed(candidate), keyed(baseline)
    grouped = {}
    for key in set(aa) & set(bb):
        grouped.setdefault(key[0], []).append(aa[key] - bb[key])
    return world_bootstrap_median(grouped, n_boot=2000, seed=0)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def run_confirmation_once(setname, model, variant, records, workers):
    cache = OUT / "confirmation_rows" / setname / model / f"{variant[0]}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    rows = run_variant(model, variant, records, workers)
    peak = run_variant.last_memory
    if peak["parent_mb"] >= 3072 or peak["parent_plus_workers_mb"] >= 10240:
        raise SystemExit(f"memory safety cap exceeded during confirmation {setname}/{model}/{variant[0]}: {peak}")
    write_json(cache, rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("screen", "confirm"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if not 1 <= args.workers <= 6:
        raise SystemExit("workers must be in 1..6")
    path = OUT / "results.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "schema": "WM-S-R4-CHK-FADE-20261011/1", "models": {}, "selection": {}, "confirmation": {}}
    from wms.eval.normalize import iter_windows
    from wms.eval.report import state_key
    wall_cache = ROOT / "runs/active/WM-S-R4-20261011/score/cache"
    # Baseline gate is based on the saved, clean scorer cache and runs before any new screen.
    if "baseline_gate" not in result:
        checks = {}
        for m, cache_name in (("R4", "R4-last"), ("C1", "C1")):
            cached = json.loads((wall_cache / f"{cache_name}__wall_same.json").read_text(encoding="utf-8"))["summary"]
            checks[m] = {"60": cached["60"]["ds"]["median"], "120": cached["120"]["ds"]["median"]}
        passed = (abs(checks["C1"]["60"] - 8.71) <= .01 and abs(checks["C1"]["120"] - 99.84) <= .01
                  and abs(checks["R4"]["60"] - 9.96) <= .01 and abs(checks["R4"]["120"] - 110.69) <= .01)
        result["baseline_gate"] = {"values": checks, "pass": passed}
        write_json(path, result)
        if not passed:
            raise SystemExit("baseline cache reproduction failed; stopped before screening")
    if args.mode == "screen":
        all_seen = records(DATA / "wallc1_states_h60/seen_controls.json")
        idents = sorted({(str(r.get("world")), round(float(r.get("t0_s", 0)), 3), str(r.get("key")), str(r.get("source_arm"))) for r in all_seen})
        selected_ids = set(random.Random(0).sample(idents, 40))
        seen = [r for r in all_seen if (str(r.get("world")), round(float(r.get("t0_s", 0)), 3), str(r.get("key")), str(r.get("source_arm"))) in selected_ids]
        result["screen_protocol"] = {"n_seen": len(all_seen), "n_selected": len(seen), "subset_rule": "random.Random(0).sample(sorted(idents),40)",
                                     "schedule_count": len(SPECS) - 1, "reference_count": 1, "models": ["R4", "C1"]}
        for model in ("R4", "C1"):
            if model in result.get("selection", {}) and model in result.get("models", {}):
                print(f"screen {model} already complete; reusing saved screen summaries", flush=True)
                continue
            baseline = run_variant(model, ("base", "none", 0, 0), seen, args.workers)
            basestats = summarize(baseline)
            variant_stats = {"base": basestats}
            result.setdefault("screen_progress", {}).setdefault(model, {})["base"] = basestats
            result.setdefault("memory_checks", []).append({"stage": "screen", "model": model,
                "variant": "base", **run_variant.last_memory})
            write_json(path, result)
            del baseline
            for spec in SPECS:
                print(f"screen {model} {spec[0]}", flush=True)
                prior = result.get("screen_progress", {}).get(model, {}).get(spec[0])
                if prior is not None:
                    variant_stats[spec[0]] = prior
                    print(f"reusing persisted {spec[0]} summary", flush=True)
                    continue
                rows = run_variant(model, spec, seen, args.workers)
                variant_stats[spec[0]] = summarize(rows)
                del rows
                mem = run_variant.last_memory
                result.setdefault("memory_checks", []).append({"stage": "screen", "model": model,
                    "variant": spec[0], **mem})
                if mem["parent_mb"] >= 3072 or mem["parent_plus_workers_mb"] >= 10240:
                    result["memory_stop"] = {"model": model, "variant": spec[0], **mem}
                    write_json(path, result)
                    raise SystemExit("memory safety cap exceeded; stopped at completed variant")
                if not result.get("first_variant_memory"):
                    result["first_variant_memory"] = mem
                    print(f"first-variant memory: {mem}", flush=True)
                result.setdefault("screen_progress", {}).setdefault(model, {})[spec[0]] = variant_stats[spec[0]]
                write_json(path, result)
            eligible = []
            for spec in SPECS:
                name = spec[0]
                stats = variant_stats[name]
                if (spec[1] != "lane_cap"
                        and stats["60"]["median"] <= basestats["60"]["median"] + 0.5
                        and stats["blowups"]["n"] <= basestats["blowups"]["n"]):
                    eligible.append((stats["120"]["median"], spec[2], spec[3], ("accel", "lane", "both").index(spec[1]) if spec[1] in ("accel", "lane", "both") else 0, spec))
            chosen = min(eligible)[-1] if eligible else None
            result["models"][model] = {name: {"summary": stats} for name, stats in variant_stats.items()}
            result["selection"][model] = {"baseline": basestats, "eligible_count": len(eligible),
                                            "selected": chosen[0] if chosen else "base",
                                            "selected_schedule": list(chosen[1:]) if chosen else None,
                                            "selected_screen": variant_stats[chosen[0]] if chosen else basestats,
                                           "rule": "min screen median 120s subject to 60s <= baseline+0.5m and blowups <= baseline"}
            write_json(path, result)
        # Save baseline/selected identity artifact before any unseen confirmation.
        write_json(OUT / "selection.json", result["selection"])
    else:
        selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
        sets = {"wall_same": DATA / "wallc1_states_h60/wall_same.json",
                "wall_pilot": DATA / "wallc1_states_h60/wall_pilot.json",
                "wall_same_real": DATA / "wallc1_states_h60_real/wall_same.json"}
        for setname, setpath in sets.items():
            recs = records(setpath)
            result["confirmation"].setdefault(setname, {})["n_states"] = len(recs)
            c1base = run_confirmation_once(setname, "C1", ("base", "none", 0, 0), recs, args.workers)
            result["confirmation"][setname]["roster"] = sorted({r["world"] for r in recs})
            for model in ("C1", "R4"):
                chosen_name = selection[model]["selected"]
                chosen_spec = (("base", "none", 0, 0) if chosen_name == "base"
                               else next(s for s in SPECS if s[0] == chosen_name))
                baseline = c1base if model == "C1" else run_confirmation_once(
                    setname, model, ("base", "none", 0, 0), recs, args.workers)
                selected = baseline if chosen_name == "base" else run_confirmation_once(
                    setname, model, chosen_spec, recs, args.workers)
                result["confirmation"][setname].setdefault(model, {})
                result["confirmation"][setname][model]["base"] = {"summary": summarize(baseline)}
                result["confirmation"][setname][model][chosen_name] = {
                    "summary": summarize(selected),
                    "paired_vs_model_base": {str(int(h)): paired(selected, baseline, h) for h in HORIZONS},
                    "paired_vs_C1": ({str(int(h)): paired(selected, c1base, h) for h in HORIZONS}
                                      if model == "R4" else None)}
                del selected
                if model != "C1":
                    del baseline
                mem = memory_snapshot()
                result["memory_checks"].append({"stage": "confirm", "set": setname, "model": model, **mem})
                if mem["parent_mb"] >= 3072 or mem["parent_plus_workers_mb"] >= 10240:
                    result["memory_stop"] = {"stage": "confirm", "set": setname, "model": model, **mem}
                    write_json(path, result)
                    raise SystemExit("memory safety cap exceeded; stopped at completed confirmation")
                write_json(path, result)
            del c1base
    write_json(path, result)


if __name__ == "__main__":
    main()
