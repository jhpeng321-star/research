"""C6/C2 wave-fidelity mechanism diagnosis on fixed TRAIN-val starts.

Decomposes the WM acceleration at leader-deceleration onsets (``a < -1.5 m/s^2``) into

* the IDM physics core (free-road + interaction term) with the learned driver settings,
* the learned residual (``a - a_physics``) and the gate that scales it,

and compares onset counts / duration / depth / speed drop with the recorded SUMO stream.
The WM rollout is instrumented by wrapping ``model.step``; no model code is changed.

    python scripts/c4/wm_wave_diag.py \\
        --wm E:/Temp/dc_laptop/ck_round2/wm_s_stage240.pt \\
        --states .../v10delaycheck/inputs/delay_states.json --select 4 \\
        --horizon 120 --device cpu --out wave_diag.json --md wave_diag.md

TRAIN-val only; refuses restricted path tokens like the other C4 tools.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ON_FLAG = -1.5
WINDOW_S = 10.0
DT_S = 0.5
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")


def _delay_module():
    path = Path(__file__).resolve().parent / "wm_delay_check.py"
    spec = importlib.util.spec_from_file_location("wm_wave_delay_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_src() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _refuse(path) -> str:
    text = str(path).replace("\\", "/").lower()
    if any(token in text for token in FORBIDDEN):
        raise SystemExit(f"restricted path token refused: {path}")
    return str(path)


def _finite(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and math.isfinite(float(x))


def _mean(vals):
    vals = [float(v) for v in vals if _finite(v)]
    return sum(vals) / len(vals) if vals else None


def _median(vals):
    vals = sorted(float(v) for v in vals if _finite(v))
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def _pct(vals, q):
    vals = sorted(float(v) for v in vals if _finite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


# --------------------------------------------------------------------------- instrument
class StepRecorder:
    """Wrap ``model.step`` and keep the per-step acceleration decomposition."""

    def __init__(self, model) -> None:
        self.model = model
        self.rows: list[dict] = []
        self.base_rows: list[np.ndarray] = []
        self._orig = model.step
        self._orig_residual = model.residual_net.forward
        model.step = self._hook
        model.residual_net.forward = self._residual_hook

    def _residual_hook(self, *args, **kwargs):
        out = self._orig_residual(*args, **kwargs)
        delta = out["accel_delta_mps2"]
        self.base_rows.append(delta.detach().cpu().numpy()[0])
        return out

    def _hook(self, state, **kwargs):
        out = self._orig(state, **kwargs)
        row = {}
        for name in ("v_mps", "s_m", "lane", "present", "is_cav", "a_mps2", "length_m"):
            value = state.get(name)
            row[name] = (value.detach().cpu().numpy()[0] if hasattr(value, "detach")
                         else np.asarray(value)[0])
        for name in ("accel_mps2", "physics_accel_mps2", "raw_residual_mps2",
                     "accel_delta_mps2", "gate"):
            value = out.get(name)
            row[name] = (value.detach().cpu().numpy()[0] if hasattr(value, "detach")
                         else np.asarray(value)[0])
        params = out.get("driver_parameters")
        row["driver_parameters"] = (params.detach().cpu().numpy()[0]
                                    if hasattr(params, "detach") else None)
        emb = out.get("driver_embedding")
        row["driver_embedding"] = (emb.detach().cpu().numpy()[0]
                                   if hasattr(emb, "detach") else None)
        row["k"] = int(np.asarray(kwargs.get("k_steps", 0)).reshape(-1)[0])
        self.rows.append(row)
        return out

    def restore(self) -> None:
        self.model.step = self._orig
        self.model.residual_net.forward = self._orig_residual

    def arrays(self) -> dict[str, np.ndarray]:
        keys = ("v_mps", "s_m", "lane", "present", "is_cav", "a_mps2", "length_m",
                "accel_mps2", "physics_accel_mps2", "raw_residual_mps2", "gate")
        out = {key: np.stack([row[key] for row in self.rows]) for key in keys}
        out["base_delta"] = np.stack(self.base_rows) if self.base_rows else np.zeros_like(
            out["accel_mps2"])
        return out


def _idm_terms(v, lead_v, gap, params):
    """IDM free-road and interaction terms with the learned driver settings."""
    v0 = max(float(params["v0"]), 1e-3)
    tau = float(params["time_headway"])
    s0 = float(params["min_gap"])
    a_max = float(params["accel"])
    b = max(float(params["decel"]), 1e-3)
    closing = v - lead_v
    desired = s0 + max(0.0, v * tau + v * closing / (2.0 * math.sqrt(a_max * b)))
    free = a_max * (1.0 - (v / v0) ** 4)
    inter = -a_max * (desired / max(gap, 0.3)) ** 2
    return free, inter


def _event_series(arrays, steps):
    """Per-slot accel/state series over the captured steps."""
    out = []
    for i in range(steps):
        out.append({key: arrays[key][i] for key in arrays})
    return out


def _find_events(wm: dict, n_slots: int) -> list[dict]:
    """Leader-decel onsets on learned (HDV) slots; event step index."""
    events = []
    prev_a = np.full(n_slots, np.nan)
    prev_present = np.zeros(n_slots, dtype=bool)
    for i in range(len(wm["accel_mps2"])):
        a = wm["accel_mps2"][i]
        present = wm["present"][i]
        hdv = present & ~wm["is_cav"][i]
        for slot in range(n_slots):
            if not hdv[slot]:
                prev_a[slot] = np.nan
                prev_present[slot] = False
                continue
            if a[slot] < ON_FLAG and (not prev_present[slot] or not (prev_a[slot] < ON_FLAG)):
                events.append({"step": i, "slot": slot, "a": float(a[slot])})
            prev_a[slot] = float(a[slot])
            prev_present[slot] = True
    return events


def _duration_depth(wm: dict, step: int, slot: int, max_steps: int = 40) -> tuple[int, float]:
    count, depth = 0, 0.0
    for j in range(step, min(step + max_steps, len(wm["accel_mps2"]))):
        if not wm["present"][j][slot]:
            break
        value = float(wm["accel_mps2"][j][slot])
        if value >= ON_FLAG:
            break
        count += 1
        depth = min(depth, value)
    return count, depth


def _speed_drop(wm: dict, step: int, slot: int, window_steps: int) -> float | None:
    if step >= len(wm["v_mps"]):
        return None
    start = float(wm["v_mps"][step][slot])
    vals = [float(wm["v_mps"][j][slot]) for j in range(step, min(step + window_steps + 1,
                                                               len(wm["v_mps"])))
            if wm["present"][j][slot]]
    return max(0.0, start - min(vals)) if vals else None


def _rank_drops(wm: dict, step: int, slot: int, window_steps: int, max_rank: int = 3):
    """Same-lane followers' speed drops at the event (rank 1 = nearest behind)."""
    lane = int(wm["lane"][step][slot])
    s = float(wm["s_m"][step][slot])
    candidates = []
    for other in range(wm["accel_mps2"].shape[1]):
        if other == slot or not wm["present"][step][other]:
            continue
        if int(wm["lane"][step][other]) != lane:
            continue
        if float(wm["s_m"][step][other]) < s:
            candidates.append((float(wm["s_m"][step][other]), other))
    candidates.sort(reverse=True)
    drops = []
    for _, other in candidates[:max_rank]:
        drops.append(_speed_drop(wm, step, other, window_steps))
    return drops


def _sumo_series(record, arm: str, horizon: float = 300.0):
    """Recorded SUMO per-vehicle arrays at 0.5 s and the t0 roster."""
    _ensure_src()
    from wms.eval.normalize import recorded_for

    frames = recorded_for(record, arm)
    t0 = float(record.get("t0_s", 0.0))
    ids = [str(v["id"]) for v in (record.get("initial_state") or record.get("state")
                                  or {}).get("vehicles", []) if v.get("present", True)]
    index = {vid: k for k, vid in enumerate(ids)}
    n = max(1, int(round(float(horizon) / DT_S)))
    arrays = {key: np.zeros((n, len(ids)), dtype=np.float64)
              for key in ("v_mps", "s_m", "lane", "accel_mps2")}
    arrays["present"] = np.zeros((n, len(ids)), dtype=bool)
    arrays["is_cav"] = np.zeros((n, len(ids)), dtype=bool)
    for frame in frames:
        i = int(round((float(frame.get("time_s", 0.0)) - t0) / DT_S))
        if i <= 0 or i > n:
            continue
        for row in frame.get("vehicles", []):
            slot = index.get(str(row.get("id")))
            if slot is None or not row.get("present", True):
                continue
            arrays["v_mps"][i - 1, slot] = float(row.get("v", 0.0))
            arrays["s_m"][i - 1, slot] = float(row.get("s", 0.0))
            arrays["lane"][i - 1, slot] = int(row.get("lane", -1))
            arrays["present"][i - 1, slot] = True
            arrays["is_cav"][i - 1, slot] = bool(row.get("is_cav"))
    dt = DT_S
    for i in range(1, n):
        good = arrays["present"][i] & arrays["present"][i - 1]
        arrays["accel_mps2"][i] = np.where(
            good, (arrays["v_mps"][i] - arrays["v_mps"][i - 1]) / dt, 0.0)
    arrays["accel_mps2"][0] = 0.0
    return arrays, ids, t0


def _sumo_events(arrays, ids):
    """SUMO HDV onsets with duration / depth / drop, over the same 0.5 s grid."""
    events = _find_events(arrays, len(ids))
    labels = np.array(ids)
    for event in events:
        event["id"] = str(labels[event["slot"]])
        event["duration"], event["depth"] = _duration_depth(arrays, event["step"],
                                                            event["slot"])
        event["drop"] = _speed_drop(arrays, event["step"], event["slot"], 20)
    return events


def _event_decomposition(wm, events, driver_params, ids, window_steps):
    """Attach physics/residual/gate/driver settings to every WM onset."""
    names = ("v0", "time_headway", "min_gap", "accel", "decel")
    last = {}
    rows = []
    for event in events:
        step, slot = event["step"], event["slot"]
        params = dict(zip(names, driver_params[slot, :5]))
        physics = float(wm["physics_accel_mps2"][step][slot])
        total = float(wm["accel_mps2"][step][slot])
        free, inter = _idm_terms(float(wm["v_mps"][step][slot]), 0.0, 1e9, params)
        nxt = step + 1 if step + 1 < len(wm["accel_mps2"]) else None
        a_next = float(wm["accel_mps2"][nxt][slot]) if nxt is not None else None
        res_next = (float(wm["accel_mps2"][nxt][slot]
                          - wm["physics_accel_mps2"][nxt][slot])
                    if nxt is not None else None)
        prev_step = last.get(slot)
        last[slot] = step
        gate = float(wm["gate"][step][slot])
        base = float(wm["base_delta"][step][slot])
        raw = float(wm["raw_residual_mps2"][step][slot])
        temporal = (raw / gate - base) if gate > 1e-6 else None
        rows.append({
            **event,
            "a_total": total,
            "a_physics": physics,
            "a_residual": total - physics,
            "raw_residual": raw,
            "gate": gate,
            "base_delta": base,
            "temporal_delta": temporal,
            "a_free": free,
            "driver": params,
            "duration": _duration_depth(wm, step, slot)[0],
            "depth": _duration_depth(wm, step, slot)[1],
            "drop": _speed_drop(wm, step, slot, window_steps),
            "a_next": a_next,
            "residual_next": res_next,
            "onset_gap_steps": (step - prev_step) if prev_step is not None else None,
            "rank_drops": _rank_drops(wm, step, slot, window_steps),
        })
    return rows


def _summary(rows) -> dict:
    def col(key):
        return [row.get(key) for row in rows]

    return {
        "n_events": len(rows),
        "n_unique_vehicles": len({(row.get("id") or row.get("slot")) for row in rows}),
        "a_total_mean": _mean(col("a_total")),
        "a_physics_mean": _mean(col("a_physics")),
        "a_residual_mean": _mean(col("a_residual")),
        "residual_positive_share": _mean([1.0 if (r.get("a_residual") or 0) > 0 else 0.0
                                          for r in rows]),
        "gate_mean": _mean(col("gate")),
        "base_delta_mean": _mean(col("base_delta")),
        "base_saturated_share": _mean(
            [1.0 if abs(r.get("base_delta") or 0.0) >= 1.45 else 0.0 for r in rows]),
        "temporal_delta_mean": _mean(col("temporal_delta")),
        "temporal_over_base_share": _mean(
            [1.0 if abs(r.get("temporal_delta") or 0.0) > abs(r.get("base_delta") or 0.0)
             else 0.0 for r in rows]),
        "duration_mean": _mean(col("duration")),
        "duration_median": _median(col("duration")),
        "one_step_share": _mean([1.0 if r.get("duration") == 1 else 0.0 for r in rows]),
        "depth_mean": _mean(col("depth")),
        "drop_median": _median(col("drop")),
        "a_next_mean": _mean(col("a_next")),
        "residual_next_mean": _mean(col("residual_next")),
        "residual_next_positive_share": _mean(
            [1.0 if (r.get("residual_next") or 0.0) > 0.0 else 0.0 for r in rows]),
        "onset_gap_steps_median": _median(col("onset_gap_steps")),
        "b_mean": _mean([(r.get("driver") or {}).get("decel") for r in rows]),
        "b_p10": _pct([(r.get("driver") or {}).get("decel") for r in rows], 10),
        "b_p50": _pct([(r.get("driver") or {}).get("decel") for r in rows], 50),
        "b_p90": _pct([(r.get("driver") or {}).get("decel") for r in rows], 90),
        "T_mean": _mean([(r.get("driver") or {}).get("time_headway") for r in rows]),
        "s0_mean": _mean([(r.get("driver") or {}).get("min_gap") for r in rows]),
        "amax_mean": _mean([(r.get("driver") or {}).get("accel") for r in rows]),
        "rank1_drop_median": _median([d[0] for r in rows for d in [r.get("rank_drops") or []]
                                      if d]),
    }


def _physics_only(wm, driver_params, window_steps):
    """Counterfactual: would the IDM core alone have produced the onset?  Not a rollout."""
    # Every learned slot step with a would-be IDM braking (physics < -1.5) and the
    # residual that follows it.
    residuals, phys, totals = [], [], []
    for i in range(len(wm["physics_accel_mps2"])):
        present = wm["present"][i] & ~wm["is_cav"][i]
        mask = present & (wm["physics_accel_mps2"][i] < ON_FLAG)
        for slot in np.nonzero(mask)[0]:
            phys.append(float(wm["physics_accel_mps2"][i][slot]))
            residuals.append(float(wm["accel_mps2"][i][slot] - wm["physics_accel_mps2"][i][slot]))
            totals.append(float(wm["accel_mps2"][i][slot]))
    return {
        "n_physics_braking_steps": len(phys),
        "physics_braking_a_mean": _mean(phys),
        "residual_on_physics_braking_mean": _mean(residuals),
        "residual_positive_share": _mean([1.0 if r > 0 else 0.0 for r in residuals]),
        "total_a_mean": _mean(totals),
        "residual_cancels_below_threshold_share": _mean(
            [1.0 if t >= ON_FLAG else 0.0 for t in totals]),
    }


def _residual_head_probe(recorder, wm, events) -> dict:
    """Is the residual attention head saturated?  Gradient of |base delta| at the onsets."""
    import torch

    model = recorder.model
    embedding = next((row["driver_embedding"] for row in recorder.rows
                      if row["driver_embedding"] is not None), None)
    if embedding is None or not events:
        return {"n": 0, "grad_norm": None, "base_abs_mean": None,
                "saturated_share": None}
    kin_rows, emb_rows = [], []
    for event in events:
        step, slot = event["step"], event["slot"]
        lane = int(wm["lane"][step][slot])
        onehot = [0.0, 0.0, 0.0]
        onehot[max(0, min(2, lane))] = 1.0
        kin_rows.append([float(wm["s_m"][step][slot]), *onehot,
                         float(wm["v_mps"][step][slot]),
                         float(wm["a_mps2"][step][slot]),
                         float(wm["length_m"][step][slot])])
        emb_rows.append(embedding[slot])
    kin = torch.tensor(kin_rows, dtype=torch.float32).unsqueeze(1)
    present = torch.ones((len(kin_rows), 1), dtype=torch.bool)
    emb = torch.tensor(emb_rows, dtype=torch.float32).unsqueeze(1)
    model.zero_grad(set_to_none=True)
    out = model.residual_net(kin, present, conditioning=emb)
    delta = out["accel_delta_mps2"]
    delta.abs().sum().backward()
    grad = torch.sqrt(sum((p.grad.detach() ** 2).sum()
                          for p in model.residual_net.parameters()
                          if p.grad is not None))
    result = {"n": len(kin_rows), "grad_norm": float(grad),
              "base_abs_mean": float(delta.detach().abs().mean()),
              "saturated_share": float((delta.detach().abs() >= 1.45).float().mean())}
    model.zero_grad(set_to_none=True)
    return result


def _run_start(predictor, record, horizon, seed, driver_names, window_steps):
    arm = str(record.get("source_arm") or record.get("arm") or "native")
    recorded = record.get("recorded") or {}
    if isinstance(recorded, dict) and recorded and arm not in recorded:
        arm = sorted(recorded)[0]
    recorder = StepRecorder(predictor.model)
    try:
        predictor.predict_states(record, {"arm": arm, "cav_mode": "teacher"},
                                 horizon, seed)
    finally:
        recorder.restore()
    wm = recorder.arrays()
    steps = len(recorder.rows)
    n_slots = wm["accel_mps2"].shape[1]
    ids = [str(v["id"]) for v in (record.get("initial_state") or record.get("state")
                                  or {}).get("vehicles", []) if v.get("present", True)]
    # C6 uses the t0 roster only (the recorded stream carries no post-t0 entrants).
    n_slots = min(n_slots, len(ids))
    wm = {key: value[:, :n_slots] for key, value in wm.items()}
    ids = ids[:n_slots]
    driver_params = None
    for row in recorder.rows:
        if row["driver_parameters"] is not None:
            driver_params = row["driver_parameters"]
    if driver_params is None:
        driver_params = np.zeros((n_slots, 9))
    driver_params = driver_params[:n_slots]
    events = _find_events(wm, n_slots)
    wm_rows = _event_decomposition(wm, events, driver_params, ids, window_steps)
    for row in wm_rows:
        row["id"] = ids[row["slot"]]
    sumo_arrays, sumo_ids, t0 = _sumo_series(record, arm, horizon)
    sumo_events = _sumo_events(sumo_arrays, sumo_ids)
    physics_only = _physics_only(wm, driver_params, window_steps)
    residual_probe = _residual_head_probe(recorder, wm, events)

    def _rate(arrays, n_events):
        hdv_steps = int(np.sum(arrays["present"] & ~arrays["is_cav"]))
        return {"hdv_steps": hdv_steps,
                "onsets_per_1000_hdv_steps": (1000.0 * n_events / hdv_steps
                                              if hdv_steps else None)}

    return {
        "world": record.get("world"),
        "key": record.get("key"),
        "t0_s": float(record.get("t0_s", 0.0)),
        "steps": steps,
        "n_slots": n_slots,
        "wm_events": wm_rows,
        "wm_summary": {**_summary(wm_rows), **_rate(wm, len(wm_rows))},
        "sumo_summary": {**_summary(sumo_events), **_rate(sumo_arrays, len(sumo_events))},
        "wm_physics_only": physics_only,
        "residual_head_probe": residual_probe,
        "driver_b_all": {
            "b_p10": _pct(driver_params[:, 4], 10),
            "b_p50": _pct(driver_params[:, 4], 50),
            "b_p90": _pct(driver_params[:, 4], 90),
            "T_p50": _pct(driver_params[:, 1], 50),
            "s0_p50": _pct(driver_params[:, 2], 50),
            "amax_p50": _pct(driver_params[:, 3], 50),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wm", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--select", type=int, default=4)
    ap.add_argument("--horizon", type=float, default=120.0)
    ap.add_argument("--window-s", type=float, default=WINDOW_S)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    args = ap.parse_args(argv)
    _ensure_src()
    for path in (args.wm, args.states, args.out, args.md):
        if path:
            _refuse(path)
    helpers = _delay_module()
    starts = helpers.select_evenly(helpers.load_records(args.states), int(args.select))
    if not starts:
        raise SystemExit("no starts selected")
    from wms.eval.predictor import RolloutFlags, WmsPredictor

    predictor = WmsPredictor.from_checkpoint(
        args.wm, RolloutFlags(entries=True, entries_v2=True, entry_queue_state=True),
        device=args.device)
    window_steps = max(1, int(round(float(args.window_s) / DT_S)))
    results = []
    for index, record in enumerate(starts):
        results.append(_run_start(predictor, record, float(args.horizon), index,
                                  None, window_steps))
    pooled_wm = {key: _mean([row["wm_summary"][key] for row in results])
                 for key in results[0]["wm_summary"]}
    pooled_sumo = {key: _mean([row["sumo_summary"][key] for row in results])
                   for key in results[0]["sumo_summary"]}
    phys = {key: _mean([row["wm_physics_only"][key] for row in results])
            for key in results[0]["wm_physics_only"]}
    head = {key: _mean([row["residual_head_probe"][key] for row in results])
            for key in results[0]["residual_head_probe"]}
    doc = {"schema": "c4_wm_wave_diag/1", "wm": args.wm, "states": args.states,
           "select": int(args.select), "horizon_s": float(args.horizon),
           "onset_decel_mps2": ON_FLAG, "window_s": float(args.window_s),
           "starts": results, "wm_pooled": pooled_wm, "sumo_pooled": pooled_sumo,
           "physics_only_pooled": phys, "residual_head_pooled": head}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float), encoding="utf-8")
    if args.md:
        Path(args.md).write_text(render_markdown(doc), encoding="utf-8")
    print(json.dumps({"out": str(out), "n_starts": len(results),
                      "wm": pooled_wm, "sumo": pooled_sumo, "physics": phys,
                      "residual_head": head},
                     indent=2, sort_keys=True, default=float))
    return 0


def _f(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def render_markdown(doc: dict) -> str:
    wm, sumo, phys = doc["wm_pooled"], doc["sumo_pooled"], doc["physics_only_pooled"]
    lines = ["# WM wave mechanism diagnosis", "",
             f"- WM: `{doc['wm']}`", f"- starts: {doc['select']} TRAIN-val, "
             f"{doc['horizon_s']:.0f} s, onset a < {doc['onset_decel_mps2']} m/s^2, "
             f"window {doc['window_s']:.0f} s", "",
             "## Onset events (HDV, learned slots)", "",
             "| metric | WM | SUMO |", "|---|---|---|"]
    for key in ("n_events", "n_unique_vehicles", "hdv_steps", "onsets_per_1000_hdv_steps",
                "a_total_mean", "a_physics_mean", "a_residual_mean",
                "residual_positive_share", "gate_mean", "duration_mean",
                "duration_median", "one_step_share", "depth_mean", "drop_median",
                "a_next_mean", "residual_next_mean", "residual_next_positive_share",
                "onset_gap_steps_median", "b_mean", "b_p50", "T_mean", "s0_mean",
                "amax_mean", "rank1_drop_median"):
        lines.append(f"| {key} | {_f(wm.get(key))} | {_f(sumo.get(key))} |")
    lines += ["", "## Residual attention head vs temporal MLP at onsets", "",
              "| metric | value |", "|---|---|"]
    for key, value in doc.get("residual_head_pooled", {}).items():
        lines.append(f"| {key} | {_f(value)} |")
    lines += ["", "## IDM core vs residual on every would-be IDM braking step", "",
              "| metric | value |", "|---|---|"]
    for key, value in phys.items():
        lines.append(f"| {key} | {_f(value)} |")
    lines += ["", "## Per start", "",
              "| world | key | WM events | SUMO events | WM drop | SUMO drop | "
              "WM duration | SUMO duration |", "|---|---|---|---|---|---|---|---|"]
    for row in doc["starts"]:
        a, b = row["wm_summary"], row["sumo_summary"]
        lines.append(f"| {row['world']} | {row['key']} | {_f(a['n_events'])} | "
                     f"{_f(b['n_events'])} | {_f(a['drop_median'])} | "
                     f"{_f(b['drop_median'])} | {_f(a['duration_mean'])} | "
                     f"{_f(b['duration_mean'])} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
