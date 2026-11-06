"""SLOWRES CPU proof: fixed-tau sweep of the learned low-pass temporal state.

Runs the round-2 WM-S checkpoint on the same four TRAIN-val starts as the braking
probes (120 s, entries queue) and reports the wave mechanism metrics and the C6
leader drop for ``model.temporal_residual_mode='direct'`` and ``'slow'`` at a few
fixed time constants (no training).  The tau values are initialised through the
softplus parameter, exactly as a fine-tune would start.

    OMP_NUM_THREADS=1 python scripts/c4/slowres_probe.py \\
      --wm E:/Temp/dc_laptop/ck_round2/wm_s_stage240.pt \\
      --states E:/research-dense/runs/active/C4-V7FT-20261017/v10delaycheck/inputs/delay_states.json \\
      --select 4 --horizon 120 --taus 0.1,0.25,0.5,1.0 \\
      --workers 3 --out .../v10wave/slowres_probe.json --md .../v10wave/slowres_probe.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")


def _ensure_src() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _module(name: str):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"slowres_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _refuse(path) -> str:
    text = str(path).replace("\\", "/").lower()
    if any(token in text for token in FORBIDDEN):
        raise SystemExit(f"restricted path token refused: {path}")
    return str(path)


def _mean(values):
    vals = [float(v) for v in values if isinstance(v, (int, float, np.integer, np.floating))
            and np.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def _median(values):
    vals = sorted(float(v) for v in values if isinstance(
        v, (int, float, np.integer, np.floating)) and np.isfinite(float(v)))
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def _summary(rows: list[dict], key: str) -> dict:
    return {name: _mean([row[key].get(name) for row in rows])
            for name in ("onsets_per_1000_hdv_steps", "one_step_share", "duration_mean",
                         "drop_median", "rank1_drop_median", "n_events")}


_WORKER: dict = {}


def _init_worker(wm, starts, variants, horizon, device):
    import torch
    torch.set_num_threads(1)
    _ensure_src()
    from wms.eval.predictor import RolloutFlags, WmsPredictor

    flags = RolloutFlags(entries=True, entries_v2=True, entry_queue_state=True)
    predictors = {}
    for name, tau in variants:
        config = (None if tau is None else {
            "temporal_residual_mode": "slow", "temporal_tau_init_s": float(tau),
            "temporal_tau_conditioned": False})
        predictors[name] = WmsPredictor.from_checkpoint(
            wm, flags, device=device, model_config=config)
    _WORKER.update(starts=starts, variants=variants, horizon=float(horizon),
                   predictors=predictors)


def _work(index: int) -> dict:
    from wms.eval.normalize import recorded_for

    wave = _module("wm_wave_diag")
    brake = _module("wm_brake_probe")
    suite = _module("wm_fidelity_suite")
    record = _WORKER["starts"][index]
    horizon = _WORKER["horizon"]
    window_steps = int(round(10.0 / 0.5))
    t0_frame, roster, _ = suite._t0_roster(record)
    t0 = float(t0_frame["time_s"])
    arm = str(record.get("source_arm") or record.get("arm") or "native")
    rec_arrays = brake._recorded_arrays(record, arm, roster, t0,
                                        int(round(horizon / 0.5)))
    rec_arrays["is_cav"] = np.zeros_like(rec_arrays["present"])
    rec_arrays["accel_mps2"] = np.where(rec_arrays["present"],
                                        rec_arrays["accel_mps2"], 0.0)
    recorded = brake._event_summary(rec_arrays, len(roster), window_steps)
    recorded_c6 = brake._fast_c6(recorded_for(record, arm), t0, horizon, roster, 10.0)
    rows = []
    for name, tau in _WORKER["variants"]:
        predictor = _WORKER["predictors"][name]
        recorder = wave.StepRecorder(predictor.model)
        try:
            frames = predictor.predict_states(
                record, {"arm": arm, "cav_mode": "teacher"}, horizon, index)
        finally:
            recorder.restore()
        arrays = recorder.arrays()
        n_slots = min(arrays["accel_mps2"].shape[1], len(roster))
        arrays = {key: value[:, :n_slots] for key, value in arrays.items()}
        metrics = brake._event_summary(arrays, n_slots, window_steps)
        c6 = brake._fast_c6(frames, t0, horizon, roster, 10.0)
        rows.append({"variant": name, "tau_s": tau, "index": index,
                     "world": record.get("world"), "metrics": metrics,
                     "c6_leader_drop_median": c6.get("leader_drop_median"),
                     "c6_rank_drops": c6.get("drops"),
                     "recorded_c6_leader_drop_median": recorded_c6.get("leader_drop_median")})
        print(f"{name:>16} start {index + 1}: "
              f"onsets={metrics['onsets_per_1000_hdv_steps']:.1f} "
              f"one_step={metrics['one_step_share']:.3f} "
              f"dur={metrics['duration_mean']:.2f} "
              f"C6={c6.get('leader_drop_median')}", flush=True)
    return {"recorded": recorded, "recorded_c6": recorded_c6, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wm", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--select", type=int, default=4)
    ap.add_argument("--horizon", type=float, default=120.0)
    ap.add_argument("--taus", default="0.1,0.25,0.5,1.0")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    args = ap.parse_args(argv)
    _ensure_src()
    for path in (args.wm, args.states, args.out, args.md):
        if path:
            _refuse(path)

    wave = _module("wm_wave_diag")
    delay = wave._delay_module()
    starts = delay.select_evenly(delay.load_records(args.states), int(args.select))
    if not starts:
        raise SystemExit("no starts selected")
    taus = [float(x) for x in str(args.taus).split(",") if x.strip()]
    variants = [("direct", None)] + [(f"slow_tau_{tau:g}", tau) for tau in taus]
    workers = max(1, min(3, int(args.workers), len(starts)))

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(args.wm, starts, variants, args.horizon,
                                       args.device)) as pool:
        parts = list(pool.map(_work, range(len(starts))))

    results = [row for part in parts for row in part["rows"]]
    recorded_rows = [part["recorded"] for part in parts]
    pooled = []
    for name, tau in variants:
        rows = [row for row in results if row["variant"] == name]
        pooled.append({
            "variant": name, "tau_s": tau,
            "wave": _summary(rows, "metrics"),
            "c6_leader_drop_median": _median([r["c6_leader_drop_median"] for r in rows]),
            "c6_rank_drops": [_median([(r["c6_rank_drops"] or [None] * 5)[rank]
                                       for r in rows]) for rank in range(5)],
            "per_start": rows,
        })
    doc = {"schema": "c4_slowres_probe/1", "wm": args.wm, "states": args.states,
           "select": int(args.select), "horizon_s": float(args.horizon),
           "workers": int(workers), "taus_s": taus,
           "recorded_wave": _summary([{"metrics": m} for m in recorded_rows], "metrics"),
           "variants": pooled}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float), encoding="utf-8")
    if args.md:
        lines = ["# SLOWRES tau sweep (round-2 checkpoint, CPU)", "",
                 f"- WM: `{args.wm}`", f"- {len(starts)} TRAIN-val starts, "
                 f"{float(args.horizon):g} s, entries queue", "",
                 "| variant | tau s | onsets/1000 | one-step | duration s | C6 leader drop |",
                 "|---|---|---|---|---|---|"]
        for row in pooled:
            w = row["wave"]
            tau_txt = "" if row["tau_s"] is None else format(row["tau_s"], "g")
            lines.append(
                f"| {row['variant']} | {tau_txt} "
                f"| {w['onsets_per_1000_hdv_steps']:.1f} | {w['one_step_share']:.3f} "
                f"| {w['duration_mean']:.2f} | {row['c6_leader_drop_median']:.3f} |")
        rec = doc["recorded_wave"]
        lines.append(f"| recorded | - | {rec['onsets_per_1000_hdv_steps']:.1f} "
                     f"| {rec['one_step_share']:.3f} | {rec['duration_mean']:.2f} | - |")
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "variants": pooled}, indent=2,
                     sort_keys=True, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
