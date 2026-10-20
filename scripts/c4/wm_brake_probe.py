"""V10BRKPROBE: baseline vs inference-only braking ablations on fixed TRAIN-val starts.

Runs the round-2 WM-S checkpoint on

* 4 TRAIN-val starts (120 s, entries queue) with the wave mechanism metrics, and
* 8 TRAIN-val starts (300 s, entries queue) with the C1-C6 suite metrics,

for the baseline and each switch combination (zero temporal output, combined-residual bound
0.5/1.0/1.5 m/s^2, temporal low-pass 1/2/4 s).  A paired teacher-history probe reruns the same
starts with the true HDV states revealed every step (recorded history) and reports whether the
one-step pulses survive.

CPU only, <= 3 workers, no training, no SUMO.  Restrict to TRAIN-val state files; restricted
path tokens are refused like the other C4 tools.

    python scripts/c4/wm_brake_probe.py \
      --wm E:/Temp/dc_laptop/ck_round2/wm_s_stage240.pt \
      --states .../v10delaycheck/inputs/delay_states.json \
      --out .../v10wave/brake_probe.json --md .../v10wave/BRAKE_PROBES.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

DT_S = 0.5
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")

VARIANTS = (
    ("baseline", {}),
    ("zero_temporal", {"diag_zero_temporal": True}),
    ("bound_0.5", {"diag_residual_bound_mps2": 0.5}),
    ("bound_1.0", {"diag_residual_bound_mps2": 1.0}),
    ("bound_1.5", {"diag_residual_bound_mps2": 1.5}),
    ("tau_1s", {"diag_temporal_tau_s": 1.0}),
    ("tau_2s", {"diag_temporal_tau_s": 2.0}),
    ("tau_4s", {"diag_temporal_tau_s": 4.0}),
)


def _module(name: str):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"brkprobe_{name}", path)
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


def _finite(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))


def _mean(values):
    vals = [float(v) for v in values if _finite(v)]
    return sum(vals) / len(vals) if vals else None


def _frame_values(frame):
    return {str(row.get("id")): row for row in frame.get("vehicles", [])
            if row.get("present", True)}


def _recorded_arrays(record, arm, ids, t0_s, n_steps):
    """Recorded per-slot arrays in the caller's (sorted) id order, 0.5 s grid."""
    from wms.eval.normalize import recorded_for

    ids = [str(vid) for vid in ids]
    index = {vid: k for k, vid in enumerate(ids)}
    arrays = {key: np.zeros((n_steps, len(ids)), dtype=float)
              for key in ("s_m", "v_mps", "lane", "accel_mps2")}
    arrays["present"] = np.zeros((n_steps, len(ids)), dtype=bool)
    prev_v = None
    for frame in recorded_for(record, arm):
        i = int(round((float(frame.get("time_s", 0.0)) - float(t0_s)) / DT_S))
        if i <= 0 or i > n_steps:
            continue
        for vid, row in _frame_values(frame).items():
            slot = index.get(vid)
            if slot is None:
                continue
            arrays["v_mps"][i - 1, slot] = float(row.get("v", 0.0))
            arrays["s_m"][i - 1, slot] = float(row.get("s", 0.0))
            arrays["lane"][i - 1, slot] = int(row.get("lane", -1))
            arrays["present"][i - 1, slot] = True
    for i in range(n_steps):
        if i == 0:
            continue
        good = arrays["present"][i] & arrays["present"][i - 1]
        arrays["accel_mps2"][i] = np.where(
            good, (arrays["v_mps"][i] - arrays["v_mps"][i - 1]) / DT_S, 0.0)
    return arrays


def _event_summary(arrays, n_slots, window_steps):
    wave = _module("wm_wave_diag")
    events = wave._find_events(arrays, n_slots)
    hdv_steps = int(np.sum(arrays["present"] & ~arrays["is_cav"]))
    durations, drops, rank1 = [], [], []
    for event in events:
        step, slot = event["step"], event["slot"]
        duration, _ = wave._duration_depth(arrays, step, slot)
        durations.append(duration)
        drops.append(wave._speed_drop(arrays, step, slot, window_steps))
        ranks = wave._rank_drops(arrays, step, slot, window_steps)
        if ranks:
            rank1.append(ranks[0])
    return {
        "n_events": len(events),
        "hdv_steps": hdv_steps,
        "onsets_per_1000_hdv_steps": (1000.0 * len(events) / hdv_steps
                                      if hdv_steps else None),
        "one_step_share": _mean([1.0 if d == 1 else 0.0 for d in durations]),
        "duration_mean": _mean(durations),
        "drop_median": (float(np.median([d for d in drops if _finite(d)]))
                        if any(_finite(d) for d in drops) else None),
        "rank1_drop_median": (float(np.median([d for d in rank1 if _finite(d)]))
                              if any(_finite(d) for d in rank1) else None),
    }


def _rollout_arrays(roll, ids, is_cav):
    """Model rollout to the per-slot arrays used by the wave event helpers."""
    accel = roll["accel"][0].numpy()
    masks = roll["masks"][0].numpy().astype(bool)
    speeds = roll["speeds"][0].numpy()
    n_steps, n_slots = accel.shape
    arrays = {
        "accel_mps2": accel,
        "v_mps": np.concatenate((np.zeros((1, n_slots)), speeds), 0)[:n_steps],
        "s_m": roll["positions"][0].numpy(),
        "lane": roll["lanes"][0].numpy(),
        "present": masks,
        "is_cav": np.broadcast_to(np.asarray(is_cav, dtype=bool), (n_steps, n_slots)),
    }
    return arrays


def _history_probe(predictor, record, horizon, arm):
    """Paired free vs recorded-history (one-step) rollouts on the t0 roster, no entries."""
    import torch
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles
    from wms.eval.normalize import DT_S as GRID

    model = predictor.model
    t0_s, ids, kin, is_cav = t0_vehicles(record)
    n_steps = max(1, int(round(float(horizon) / GRID)))
    driver_history, driver_mask = predictor._history.build(record, ids)
    initial = {
        "s_m": torch.as_tensor(kin[None, :, 0]),
        "v_mps": torch.as_tensor(kin[None, :, 4]),
        "a_mps2": torch.zeros((1, len(ids))),
        "lane": torch.as_tensor(kin[None, :, 1:4].argmax(-1)),
        "length_m": torch.as_tensor(kin[None, :, 6]),
        "present": torch.ones((1, len(ids)), dtype=torch.bool),
        "is_cav": torch.as_tensor(is_cav[None]),
    }
    teacher, present_seq, junction_seq = recorded_teacher(
        record, t0_s, ids, kin, n_steps, arm, predictor.backend.tools)
    from wms.physics.batched import road_end_exit_bound
    exit_s = road_end_exit_bound(getattr(model, "physics", None))
    if exit_s is not None:
        initial["exit_s_m"] = torch.full((1, len(ids)), float(exit_s))
    controller = None if teacher is not None else (
        model.hold_teacher if bool(np.any(is_cav)) else None)

    def run(observe=None, mask=None):
        model.reset_diag_state()
        with torch.no_grad():
            roll = model.rollout(
                dict(initial), n_steps=n_steps, cav_teacher=teacher,
                present_seq=present_seq, junction_seq=junction_seq,
                cav_controller=controller, driver_history=driver_history,
                driver_history_mask=driver_mask, observe_states=observe,
                observe_mask=mask)
        arrays = _rollout_arrays(roll, ids, is_cav)
        return _event_summary(arrays, len(ids), int(round(10.0 / GRID)))

    free = run()
    recorded = _recorded_arrays(record, arm, ids, t0_s, n_steps)
    observe = np.repeat(kin[None, None], n_steps + 1, axis=1)
    observe_mask = np.zeros((1, n_steps + 1, len(ids)), dtype=bool)
    observe_mask[0, 0] = True
    for i in range(n_steps):
        row = np.zeros((len(ids), 7), dtype=np.float32)
        row[:, 0] = recorded["s_m"][i]
        lane = recorded["lane"][i].astype(int)
        for slot in range(len(ids)):
            if 0 <= lane[slot] < 3:
                row[slot, 1 + lane[slot]] = 1.0
        row[:, 4] = recorded["v_mps"][i]
        row[:, 5] = recorded["accel_mps2"][i]
        row[:, 6] = kin[:, 6]
        observe[0, i + 1] = row
        observe_mask[0, i + 1] = recorded["present"][i]
    teacher_hist = run(torch.as_tensor(observe), torch.as_tensor(observe_mask))
    return {"world": record.get("world"), "key": record.get("key"),
            "free": free, "teacher_history": teacher_hist}


# --------------------------------------------------------------------------- worker
_WORKER: dict = {}


def _init_worker(wm, wave_starts, suite_starts, wave_horizon, suite_horizon, window_steps):
    import torch
    torch.set_num_threads(1)
    _WORKER.update(wm=str(wm), wave=wave_starts, suite=suite_starts,
                   wave_horizon=float(wave_horizon), suite_horizon=float(suite_horizon),
                   window_steps=int(window_steps), predictors={})


def _predictor(flags):
    key = json.dumps(flags, sort_keys=True)
    if key not in _WORKER["predictors"]:
        from wms.eval.predictor import RolloutFlags, WmsPredictor
        _WORKER["predictors"][key] = WmsPredictor.from_checkpoint(
            _WORKER["wm"], RolloutFlags(entries=True, entries_v2=True,
                                        entry_queue_state=True, **flags), device="cpu")
    return _WORKER["predictors"][key]


def _wave_work(payload):
    variant, flags, index = payload
    predictor = _predictor(flags)
    record = _WORKER["wave"][index]
    wave = _module("wm_wave_diag")
    arm = str(record.get("source_arm") or record.get("arm") or "native")
    recorder = wave.StepRecorder(predictor.model)
    try:
        predictor.predict_states(record, {"arm": arm, "cav_mode": "teacher"},
                                 _WORKER["wave_horizon"], index)
    finally:
        recorder.restore()
    arrays = recorder.arrays()
    ids = [str(v["id"]) for v in (record.get("initial_state") or record.get("state")
                                  or {}).get("vehicles", []) if v.get("present", True)]
    n_slots = min(arrays["accel_mps2"].shape[1], len(ids))
    arrays = {key: value[:, :n_slots] for key, value in arrays.items()}
    summary = _event_summary(arrays, n_slots, _WORKER["window_steps"])
    return {"kind": "wave", "variant": variant, "index": index,
            "world": record.get("world"), "key": record.get("key"), "metrics": summary}


def _fast_c6(frames, t0, horizon, roster, window_s):
    """C6 leader/follower drops with a speed matrix instead of per-event frame scans."""
    roster = [str(vid) for vid in roster]
    by_time = {round(float(f.get("time_s", 0.0)), 6): f for f in frames
               if t0 - 1e-6 <= float(f.get("time_s", 0.0)) <= t0 + horizon + 1e-6}
    times = sorted(by_time)
    if len(times) < 3 or not roster:
        return {"n_events": 0, "leader_drop_median": None, "drops": [None] * 5}
    slot = {vid: k for k, vid in enumerate(roster)}
    v = np.full((len(times), len(roster)), np.nan)
    s = np.full_like(v, np.nan)
    lane = np.full_like(v, np.nan)
    for i, t in enumerate(times):
        for veh in by_time[t].get("vehicles", []):
            k = slot.get(str(veh.get("id")))
            if k is None or not veh.get("present", True):
                continue
            v[i, k] = float(veh.get("v", 0.0))
            s[i, k] = float(veh.get("s", 0.0))
            lane[i, k] = int(veh.get("lane", -1))
    w = max(1, int(round(float(window_s) / DT_S)))
    horizon_steps = max(0, int(round(float(horizon) / DT_S)) - w)
    drops = [[] for _ in range(5)]
    leader_amps = []
    n_events = 0
    for k in range(len(roster)):
        for i in range(1, min(len(times), horizon_steps + 1)):
            a = (v[i, k] - v[i - 1, k]) / DT_S if _finite(v[i, k]) and _finite(v[i - 1, k]) else np.nan
            ap = ((v[i - 1, k] - v[i - 2, k]) / DT_S
                  if i >= 2 and _finite(v[i - 1, k]) and _finite(v[i - 2, k]) else np.nan)
            if not _finite(a) or a >= -1.5 or (_finite(ap) and ap < -1.5):
                continue
            n_events += 1
            if not _finite(lane[i, k]) or lane[i, k] < 0 or not _finite(s[i, k]):
                continue
            same = [j for j in range(len(roster))
                    if j != k and _finite(lane[i, j]) and lane[i, j] == lane[i, k]
                    and _finite(s[i, j]) and s[i, j] < s[i, k]]
            if not same:
                continue
            same.sort(key=lambda j: s[i, j], reverse=True)
            future = v[i:i + w + 1, k]
            future = future[np.isfinite(future)]
            if not future.size:
                continue
            lead_drop = max(0.0, v[i, k] - float(future.min()))
            leader_amps.append(lead_drop)
            for rank, j in enumerate(same[:5]):
                fut = v[i:i + w + 1, j]
                fut = fut[np.isfinite(fut)]
                if fut.size:
                    drops[rank].append(max(0.0, v[i, j] - float(fut.min())))
    return {
        "n_events": n_events,
        "leader_drop_median": (float(np.median(leader_amps)) if leader_amps else None),
        "drops": [(float(np.median(vals)) if vals else None) for vals in drops],
        "follower_n": [len(vals) for vals in drops],
    }


def _suite_metrics(frames, record, horizon):
    suite = _module("wm_fidelity_suite")
    t0_frame, roster, cavs = suite._t0_roster(record)
    t0 = float(t0_frame["time_s"])
    c2 = suite.ttc_distribution(frames, t0, horizon, roster, cavs)["all_pairs"]
    c6 = _fast_c6(frames, t0, horizon, roster, 10.0)
    return {"C2": c2, "C6": c6}


def _suite_work(payload):
    variant, flags, index = payload
    predictor = _predictor(flags)
    record = _WORKER["suite"][index]
    horizon = _WORKER["suite_horizon"]
    frames = predictor.predict_states(record, {"arm": "native"}, horizon, index)
    wm = _suite_metrics(frames, record, horizon)
    if index not in _WORKER.setdefault("sumo", {}):
        from wms.eval.normalize import recorded_for
        _WORKER["sumo"][index] = _suite_metrics(
            recorded_for(record, "native"), record, horizon)
    return {"kind": "suite", "variant": variant, "index": index,
            "state": {"wm": wm, "sumo": _WORKER["sumo"][index]}}


def _history_work(payload):
    variant, flags, index = payload
    predictor = _predictor(flags)
    record = _WORKER["wave"][index]
    arm = str(record.get("source_arm") or record.get("arm") or "native")
    return {"kind": "history", "variant": variant, "index": index,
            **_history_probe(predictor, record, _WORKER["wave_horizon"], arm)}


# --------------------------------------------------------------------------- report
def _pool_map(payloads, workers, initargs):
    workers = max(1, min(3, int(workers)))
    if workers <= 1:
        _init_worker(*initargs)
        return [_dispatch(p) for p in payloads]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=initargs) as pool:
        return list(pool.map(_dispatch, payloads))


def _dispatch(payload):
    kind = payload[0]
    if kind == "wave":
        return _wave_work(payload[1:])
    if kind == "suite":
        return _suite_work(payload[1:])
    return _history_work(payload[1:])


def _suite_pooled(states):
    suite = _module("wm_fidelity_suite")
    wm = suite._avg_numeric([s["wm"] for s in states])
    sumo = suite._avg_numeric([s["sumo"] for s in states])
    c6 = wm.get("C6") or {}
    s6 = sumo.get("C6") or {}
    c2 = (wm.get("C2") or {}).get("all_pairs") or {}
    s2 = (sumo.get("C2") or {}).get("all_pairs") or {}
    return {
        "n_starts": len(states),
        "C6_leader_drop_median": c6.get("leader_drop_median"),
        "C6_rank_drops": list(c6.get("drops") or []),
        "C6_n_events": c6.get("n_events"),
        "C2_near_miss_3_rate": c2.get("near_miss_3_rate"),
        "C2_near_miss_15_rate": c2.get("near_miss_15_rate"),
        "C2_ttc_p05": c2.get("ttc_p05"),
        "sumo_C6_leader_drop_median": s6.get("leader_drop_median"),
        "sumo_C6_rank_drops": list(s6.get("drops") or []),
        "sumo_C2_near_miss_3_rate": s2.get("near_miss_3_rate"),
        "sumo_C2_near_miss_15_rate": s2.get("near_miss_15_rate"),
    }


def _mean_blocks(blocks):
    """Average a list of metric dicts key by key."""
    if not blocks:
        return {}
    return {key: _mean([block.get(key) for block in blocks])
            for key in blocks[0]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wm", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--wave-select", type=int, default=4)
    ap.add_argument("--wave-horizon", type=float, default=120.0)
    ap.add_argument("--suite-select", type=int, default=8)
    ap.add_argument("--suite-horizon", type=float, default=300.0)
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--variants", default="",
                    help="comma-separated subset of variant names (default: all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    args = ap.parse_args(argv)
    _ensure_src()
    for token in (args.wm, args.states, args.out, args.md):
        if token:
            _refuse(token)
    delay = _module("wm_delay_check")
    records = delay.load_records(args.states)
    wave_starts = delay.select_evenly(records, int(args.wave_select))
    suite_starts = delay.select_evenly(records, int(args.suite_select))
    if not wave_starts or not suite_starts:
        raise SystemExit("no starts selected")
    window_steps = max(1, int(round(float(args.window_s) / DT_S)))
    wanted = {part.strip() for part in str(args.variants).split(",") if part.strip()}
    variants = tuple(v for v in VARIANTS if not wanted or v[0] in wanted)
    if not variants:
        raise SystemExit(f"no known variants in {args.variants!r}")
    initargs = (args.wm, wave_starts, suite_starts, args.wave_horizon,
                args.suite_horizon, window_steps)
    payloads = []
    for name, flags in variants:
        for index in range(len(wave_starts)):
            payloads.append(("wave", name, flags, index))
        for index in range(len(suite_starts)):
            payloads.append(("suite", name, flags, index))
    results = _pool_map(payloads, args.workers, initargs)
    history = _pool_map([("history", "baseline", {}, index)
                         for index in range(len(wave_starts))],
                        args.workers, initargs)
    by_variant: dict[str, dict] = {}
    for row in results:
        block = by_variant.setdefault(row["variant"], {"wave": [], "suite": []})
        block[row["kind"]].append(row)
    doc = {"schema": "c4_wm_brake_probe/1", "wm": args.wm, "states": args.states,
           "wave_select": int(args.wave_select), "wave_horizon_s": float(args.wave_horizon),
           "suite_select": int(args.suite_select),
           "suite_horizon_s": float(args.suite_horizon),
           "wave_starts": [{"world": r.get("world"), "key": r.get("key")} for r in wave_starts],
           "suite_starts": [{"world": r.get("world"), "key": r.get("key")}
                            for r in suite_starts],
           "variants": {}, "history": history}
    for name, flags in variants:
        block = by_variant.get(name, {"wave": [], "suite": []})
        wave_rows = sorted(block["wave"], key=lambda r: r["index"])
        suite_rows = sorted(block["suite"], key=lambda r: r["index"])
        doc["variants"][name] = {
            "flags": flags,
            "wave": _mean_blocks([r["metrics"] for r in wave_rows]),
            "suite": _suite_pooled([r["state"] for r in suite_rows]),
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float), encoding="utf-8")
    if args.md:
        Path(args.md).write_text(render_markdown(doc), encoding="utf-8")
    print(json.dumps({"out": str(out), "md": args.md,
                      "variants": list(doc["variants"]),
                      "history_free": _mean([r["free"]["onsets_per_1000_hdv_steps"]
                                             for r in history]),
                      "history_teacher": _mean(
                          [r["teacher_history"]["onsets_per_1000_hdv_steps"]
                           for r in history])}, indent=2, default=float))
    return 0


def _f(value, digits=3):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render_markdown(doc: dict) -> str:
    base = doc["variants"].get("baseline", {})
    lines = [
        "# Braking diagnostic probes (inference only)", "",
        f"- WM: `{doc['wm']}`",
        f"- wave sample: {doc['wave_select']} TRAIN-val starts, "
        f"{doc['wave_horizon_s']:.0f} s, entries queue",
        f"- suite sample: {doc['suite_select']} TRAIN-val starts, "
        f"{doc['suite_horizon_s']:.0f} s, entries queue",
        "", "## Wave mechanism metrics", "",
        "| variant | onsets/1000 HDV steps | one-step share | mean duration s | drop median | rank1 drop |",
        "|---|---|---|---|---|---|",
    ]
    for name, block in doc["variants"].items():
        w = block["wave"]
        lines.append(f"| {name} | {_f(w.get('onsets_per_1000_hdv_steps'))} | "
                     f"{_f(w.get('one_step_share'))} | {_f(w.get('duration_mean'))} | "
                     f"{_f(w.get('drop_median'))} | {_f(w.get('rank1_drop_median'))} |")
    lines += ["", "## C6 / C2 (300 s, 8 starts)", "",
              "| variant | C6 leader drop | C6 rank1-3 drops | C6 events | C2 near<3 s | C2 near<1.5 s |",
              "|---|---|---|---|---|---|"]
    for name, block in doc["variants"].items():
        s = block["suite"]
        ranks = ", ".join(_f(v) for v in (s.get("C6_rank_drops") or [])[:3])
        lines.append(f"| {name} | {_f(s.get('C6_leader_drop_median'))} | {ranks} | "
                     f"{_f(s.get('C6_n_events'))} | {_f(s.get('C2_near_miss_3_rate'))} | "
                     f"{_f(s.get('C2_near_miss_15_rate'))} |")
    if base:
        s = base["suite"]
        lines += ["", f"SUMO reference: C6 leader drop {_f(s.get('sumo_C6_leader_drop_median'))}, "
                  f"rank1-3 {', '.join(_f(v) for v in (s.get('sumo_C6_rank_drops') or [])[:3])}, "
                  f"C2 near<3 s {_f(s.get('sumo_C2_near_miss_3_rate'))}, "
                  f"near<1.5 s {_f(s.get('sumo_C2_near_miss_15_rate'))}."]
    lines += ["", "## Teacher-history vs free-history (baseline, 4 starts, 120 s)", "",
              "| history | onsets/1000 HDV steps | one-step share | mean duration s | drop median |",
              "|---|---|---|---|---|"]
    free = _mean_blocks([r["free"] for r in doc["history"]]) if doc["history"] else {}
    teacher = (_mean_blocks([r["teacher_history"] for r in doc["history"]])
               if doc["history"] else {})
    for label, block in (("free rollout", free), ("recorded one-step", teacher)):
        lines.append(f"| {label} | {_f(block.get('onsets_per_1000_hdv_steps'))} | "
                     f"{_f(block.get('one_step_share'))} | {_f(block.get('duration_mean'))} | "
                     f"{_f(block.get('drop_median'))} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
