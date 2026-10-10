"""ENTRY CPU check: run-1 WM-S checkpoint on ws-eval dev states, with/without entries.

TRAIN-only, CPU, one process, no SUMO/GPU.  Reads the run-1 checkpoint and dev
states READ-ONLY from the ws-eval worktree and scores merge queue/stopped/delay
against the recorded native run.  Writes ``_out/entry_cpu.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"E:/rd-wt/ws-entry")
WM = ROOT / "output" / "wd01" / "wm_n1"
WS_EVAL = Path(r"E:/rd-wt/ws-eval")
CKPT = WS_EVAL / "_out" / "wms_ckpt" / "wm_s.pt"
PHYSICS = WS_EVAL / "_out" / "wms_ckpt" / "physics.json"
STATES = WS_EVAL / "_out" / "states_dev.json"

sys.path.insert(0, str(WM))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

import wms_eval.data as data  # noqa: E402
import wms_eval.wms_plugin as plugin  # noqa: E402

# resolve the ws-eval dev-state relative paths / checkpoint read-only
data.WORKSPACE = WS_EVAL
plugin.WORKSPACE = WS_EVAL

from wms_eval.data import load_states, recorded_for  # noqa: E402
from wms_eval.metrics import traffic_metrics  # noqa: E402
from wms_eval.models import RolloutFlags  # noqa: E402

BIN_S = 30.0


def _merge_bins(traffic: dict) -> list:
    out = []
    for key in sorted(traffic, key=lambda k: int(k)):
        block = traffic[key]["sections"]["merge"]
        out.append({"bin": int(key), "start_s": traffic[key]["start_s"],
                    "queue": block["merge_queue_length"],
                    "stopped": block["stopped_vehicles"],
                    "delay": block["total_delay"], "mean_speed": block["mean_speed"],
                    "flow": block["merge_flow"]})
    return out


def _agg(bins: list) -> dict:
    def mean(key):
        vals = [b[key] for b in bins if b[key] is not None]
        return float(np.mean(vals)) if vals else None
    return {"queue": mean("queue"), "stopped": mean("stopped"),
            "delay": float(sum(b["delay"] or 0.0 for b in bins)),
            "mean_speed": mean("mean_speed"), "flow": mean("flow")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=5)
    ap.add_argument("--horizon", type=float, default=300.0)
    args = ap.parse_args()

    physics = json.loads(PHYSICS.read_text(encoding="utf-8"))
    if isinstance(physics, dict) and isinstance(physics.get("physics"), dict):
        physics = physics["physics"]
    states = load_states(STATES)[: int(args.states)]
    off = plugin.WmsCheckpointModel(str(CKPT), RolloutFlags(), physics=physics)
    on = plugin.WmsCheckpointModel(str(CKPT), RolloutFlags(open_entries=True),
                                   physics=physics)

    per_state, agg = {}, {}
    for i, rec in enumerate(states):
        started = time.time()
        t0 = float(rec.get("t0_s", 0.0))
        seed = int(rec.get("seed", 0))
        v0 = float(rec.get("v0_mps", 13.89))
        horizon = min(float(args.horizon), float(rec.get("horizon_s", args.horizon)))
        recorded = [f for f in recorded_for(rec, "native")
                    if float(f["time_s"]) <= t0 + horizon + 1e-9]
        pred_off = off.predict_states(rec, {"arm": "native"}, horizon, seed)
        pred_on = on.predict_states(rec, {"arm": "native"}, horizon, seed)
        res = {}
        for name, frames in (("recorded", recorded), ("off", pred_off), ("on", pred_on)):
            traffic = traffic_metrics(frames, t0_s=t0, free_speed_mps=v0, bin_s=BIN_S)
            bins = _merge_bins(traffic)
            res[name] = {"bins": bins, "agg": _agg(bins),
                         "last_bin": bins[-1] if bins else None}
        res["entries"] = getattr(on, "entries_summary", None)
        res["seconds"] = round(time.time() - started, 1)
        per_state[rec.get("world", f"s{i}")] = res
        for name in ("recorded", "off", "on"):
            a = agg.setdefault(name, {"queue": [], "stopped": [], "delay": [],
                                      "mean_speed": [], "flow": []})
            for key in a:
                if res[name]["agg"][key] is not None:
                    a[key].append(res[name]["agg"][key])
        def fmt(x):
            return "n/a" if x is None else f"{x:.2f}"
        print(f"[{i+1}/{len(states)}] {rec.get('world')} "
              f"off q={fmt(res['off']['agg']['queue'])} "
              f"on q={fmt(res['on']['agg']['queue'])} "
              f"rec q={fmt(res['recorded']['agg']['queue'])} {res['seconds']}s", flush=True)

    summary = {}
    for name, a in agg.items():
        summary[name] = {k: (float(np.mean(v)) if v else None) for k, v in a.items()}
    out = {"checkpoint": str(CKPT), "physics": physics, "n_states": len(states),
           "horizon_s": args.horizon, "summary": summary, "per_state": per_state}
    (ROOT / "_out" / "entry_cpu.json").write_text(json.dumps(out, indent=2),
                                                  encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
