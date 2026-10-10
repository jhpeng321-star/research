"""ENTRY2 CPU check: run-1 WM-S checkpoint on ws-eval dev states, v1 vs v2 entries.

TRAIN-only, CPU, one heavy process, no SUMO/GPU.  Reads the run-1 checkpoint and
dev states READ-ONLY from the ws-eval worktree and scores merge flow/queue/
stopped/delay plus entrant counts per entrance (v1 vs v2 vs recorded) against the
recorded native run.  Memory rule: states are processed one at a time and only
the 0.9 MB JSON state list is read.  Writes ``_out/entry2_cpu.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"E:/rd-wt/ws-entry2")
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

data.WORKSPACE = WS_EVAL
plugin.WORKSPACE = WS_EVAL

from wms_eval.data import load_states, recorded_for  # noqa: E402
from wms_eval.metrics import traffic_metrics  # noqa: E402
from wms_eval.models import RolloutFlags  # noqa: E402

BIN_S = 30.0
_ENT = {"mainline": 0, "ramp": 1}


def _merge_bins(traffic: dict) -> list:
    out = []
    for key in sorted(traffic, key=lambda k: int(k)):
        block = traffic[key]["sections"]["merge"]
        out.append({"bin": int(key), "queue": block["merge_queue_length"],
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


def _recorded_entrants(rec: dict, horizon: float) -> dict:
    """Recorded insertions in (t0, t0+horizon] per entrance (read-only labels)."""
    t0 = float(rec.get("t0_s", 0.0))
    base = WS_EVAL / str(rec.get("base_run_dir", ""))
    path = base / "run" / "insertions.jsonl"
    counts = [0, 0]
    if not path.is_file():
        return {"per_entrance": counts, "total": 0}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = float(row.get("inserted_s", row.get("scheduled_depart_s", 0.0)))
            if t0 < t <= t0 + float(horizon) + 1e-9:
                counts[_ENT.get(str(row.get("entrance", "mainline")), 0)] += 1
    return {"per_entrance": counts, "total": int(sum(counts))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=5)
    ap.add_argument("--horizon", type=float, default=300.0)
    args = ap.parse_args()

    physics = json.loads(PHYSICS.read_text(encoding="utf-8"))
    if isinstance(physics, dict) and isinstance(physics.get("physics"), dict):
        physics = physics["physics"]
    states = load_states(STATES)[: int(args.states)]
    models = {
        "v1": plugin.WmsCheckpointModel(str(CKPT), RolloutFlags(open_entries=True),
                                        physics=physics),
        "v2": plugin.WmsCheckpointModel(str(CKPT),
                                        RolloutFlags(open_entries=True, entries_v2=True),
                                        physics=physics),
    }

    per_state, agg = {}, {}
    for i, rec in enumerate(states):
        started = time.time()
        t0 = float(rec.get("t0_s", 0.0))
        seed = int(rec.get("seed", 0))
        v0 = float(rec.get("v0_mps", 13.89))
        horizon = min(float(args.horizon), float(rec.get("horizon_s", args.horizon)))
        recorded = [f for f in recorded_for(rec, "native")
                    if float(f["time_s"]) <= t0 + horizon + 1e-9]
        res = {}
        for name, model in models.items():
            frames = model.predict_states(rec, {"arm": "native"}, horizon, seed)
            traffic = traffic_metrics(frames, t0_s=t0, free_speed_mps=v0, bin_s=BIN_S)
            bins = _merge_bins(traffic)
            summary = getattr(model, "entries_summary", {}) or {}
            res[name] = {"bins": bins, "agg": _agg(bins),
                         "entrants_per_entrance": summary.get("per_entrance"),
                         "inserted": summary.get("inserted"),
                         "no_slot": summary.get("no_slot"),
                         "rate_veh_s": summary.get("rate_veh_s"),
                         "n_obs": summary.get("n_obs")}
        traffic = traffic_metrics(recorded, t0_s=t0, free_speed_mps=v0, bin_s=BIN_S)
        bins = _merge_bins(traffic)
        res["recorded"] = {"bins": bins, "agg": _agg(bins)}
        res["recorded_entrants"] = _recorded_entrants(rec, horizon)
        res["seconds"] = round(time.time() - started, 1)
        per_state[rec.get("world", f"s{i}")] = res
        for name in ("recorded", "v1", "v2"):
            a = agg.setdefault(name, {"queue": [], "stopped": [], "delay": [],
                                      "mean_speed": [], "flow": []})
            for key in a:
                if res[name]["agg"][key] is not None:
                    a[key].append(res[name]["agg"][key])
        fmt = lambda x: "n/a" if x is None else f"{x:.2f}"  # noqa: E731
        print(f"[{i+1}/{len(states)}] {rec.get('world')} "
              f"v1 q={fmt(res['v1']['agg']['queue'])} "
              f"v2 q={fmt(res['v2']['agg']['queue'])} "
              f"rec q={fmt(res['recorded']['agg']['queue'])} "
              f"e1/v1={res['v1']['entrants_per_entrance']} "
              f"e2/v2={res['v2']['entrants_per_entrance']} "
              f"rec={res['recorded_entrants']['per_entrance']} {res['seconds']}s",
              flush=True)

    summary = {}
    for name, a in agg.items():
        summary[name] = {k: (float(np.mean(v)) if v else None) for k, v in a.items()}
    out = {"checkpoint": str(CKPT), "physics": physics, "n_states": len(states),
           "horizon_s": args.horizon, "summary": summary, "per_state": per_state}
    (ROOT / "_out" / "entry2_cpu.json").write_text(json.dumps(out, indent=2),
                                                   encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
