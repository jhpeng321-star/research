"""C4 physics-only lane-rule check on the 67 wall_same states (CPU, no GPU/SUMO).

Compares three physics-only lane rules with DEFAULT driver settings (residual gate
forced to zero, entries OFF, CAVs teacher-forced, seed 0, dt 0.5 s — the R5 scoring
entries; harness identical to WM-S-R4-CHK-TYPEID run_rollouts.py):

  phys_mobil         MOBIL + mandatory merge (lane_rule='mobil')   [verify 3.72/10.17/15.68/53.12]
  phys_forced        lane_rule='forced_merge_only' (no discretionary changes)
  phys_recorded_lanes lane_teacher = recorded lanes (perfect lanes reference)

plus, when the R5b best checkpoint is available, two DIAGNOSTIC variants with the
learned residual/gate intact and the lane rule switched at inference only:

  r5b_mobil          R5b best checkpoint as trained (MOBIL)
  r5b_forced         R5b best checkpoint with lane_rule='forced_merge_only'

Reports the median HDV |ds| at 10/30/60/120 s (state-median, then median over the
67 states, as in score_round5.stats_for), blow-ups (HDV ds_med > 50 m at 60 s AND
> 100 m at 120 s) and the wrong-lane share at 60/120 s (pooled HDV slots, recorded
non-CAV, predicted present, recorded lane valid — as in the R5 fingerprints).

Writes a small summary JSON (summaries only) and LANERULE_CHECK.md at the repo
root.  Reads only read-only data: data_cache eval states and run checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["TEMP"] = os.environ["TMP"] = "E:/agent-work/tmp"

import numpy as np  # noqa: E402
import torch  # noqa: E402

DATA = Path("E:/research-dense/data_cache")
WALL_SAME = DATA / "wallc1_states_h60" / "wall_same.json"
R4_CKPT = Path("E:/research-dense/runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt")
R5B_BEST = Path("E:/research-dense/runs/active/WM-S-R5B-20261012/box/gpuA/r5b_main_a1/wm_s_best.pt")
OUT_DIR = ROOT / "runs" / "active" / "LANERULE-CHECK-20261012"
HORIZONS = (10.0, 30.0, 60.0, 120.0)
N_STEPS = 240

VARIANTS = {
    "phys_mobil": dict(ckpt=str(R4_CKPT), zero_gate=True, lane_rule="mobil",
                       lane_teacher=False),
    "phys_forced": dict(ckpt=str(R4_CKPT), zero_gate=True, lane_rule="forced_merge_only",
                        lane_teacher=False),
    "phys_recorded_lanes": dict(ckpt=str(R4_CKPT), zero_gate=True, lane_rule="mobil",
                                lane_teacher=True),
    "r5b_mobil": dict(ckpt=str(R5B_BEST), zero_gate=False, lane_rule="mobil",
                      lane_teacher=False),
    "r5b_forced": dict(ckpt=str(R5B_BEST), zero_gate=False, lane_rule="forced_merge_only",
                       lane_teacher=False),
}

_WORKER: dict = {}


def iter_records(path: Path):
    from wms.eval.normalize import iter_windows
    return list(iter_windows(path))


def _init(variant: str) -> None:
    torch.set_num_threads(1)
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    spec = VARIANTS[variant]
    predictor = WmsPredictor.from_checkpoint(spec["ckpt"], RolloutFlags(
        entries=False, entries_v2=False))
    model = predictor.model
    if spec["zero_gate"]:
        model.gate = lambda k: torch.zeros((), dtype=torch.float32)
    model.physics.lane_rule = spec["lane_rule"]
    _WORKER.update(predictor=predictor, variant=variant, spec=spec)


def _arrays(frames: list[dict], ids: list[str], t0: float):
    idx = {x: i for i, x in enumerate(ids)}
    n = N_STEPS
    lane = np.full((n + 1, len(ids)), -2, dtype=np.int16)
    present = np.zeros((n + 1, len(ids)), dtype=bool)
    cav = np.zeros((n + 1, len(ids)), dtype=bool)
    for fr in frames:
        k = int(round((float(fr["time_s"]) - t0) / 0.5))
        if not (0 <= k <= n):
            continue
        for veh in fr.get("vehicles", []):
            if not veh.get("present", True):
                continue
            vid = str(veh["id"])
            if vid not in idx:        # post-t0 entrants: entries are OFF in this check
                continue
            j = idx[vid]
            lane[k, j] = int(veh.get("lane", -1))
            present[k, j] = True
            cav[k, j] = bool(veh.get("is_cav", False))
    return lane, present, cav


def _work(task):
    variant, lo, hi = task
    from wms.eval.metrics import car_metrics
    from wms.eval.normalize import recorded_for
    from wms.eval.predictor import _frames_from_rollout
    from wms.eval.rollout_inputs import action_timeline_inputs, recorded_teacher, t0_vehicles
    from wms.eval.score import car_compact

    predictor = _WORKER["predictor"]
    spec = _WORKER["spec"]
    model = predictor.model
    rows = []
    t_start = time.time()
    for i, record in enumerate(iter_records(WALL_SAME)):
        if i < lo or i >= hi:
            continue
        t0 = float(record["t0_s"])
        arm = "wall" if "wall" in (record.get("recorded") or {}) else next(iter(record["recorded"]))
        controllers = {**record.get("cav_controllers", {}), "arm": arm}
        try:
            tools = predictor.backend.tools
            t0_s, ids, kin, is_cav = t0_vehicles(record)
            driver_history, driver_history_mask = predictor._history.build(record, ids)
            action_timeline = action_timeline_inputs(record, controllers, ids, 0, tools)
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
                record, t0_s, ids, kin, N_STEPS, arm, tools)
            lane_teacher = teacher[..., 1].long() if spec["lane_teacher"] else None
            with torch.no_grad():
                roll = model.rollout(
                    initial, n_steps=N_STEPS, k0=0, cav_teacher=teacher,
                    present_seq=present_seq, junction_seq=junction_seq,
                    cav_controller=None,
                    driver_history=driver_history,
                    driver_history_mask=driver_history_mask,
                    driver_params=None, controller_id=None,
                    action_timeline=action_timeline,
                    projection=predictor.backend.projection,
                    lane_teacher=lane_teacher)
            pred = _frames_from_rollout(roll, t0_s, ids, kin, is_cav, 0, N_STEPS)
            rec = recorded_for(record, arm)
            car = car_compact(car_metrics(pred, rec, t0_s=t0, horizons=HORIZONS))
            pl, pp, pc = _arrays(pred, ids, t0)
            rl, rp, rc = _arrays(rec, ids, t0)
            wrong = {}
            for h in (60, 120):
                k = h * 2
                mask = pp[k] & rp[k] & ~rc[k] & (rl[k] >= 0)
                wrong[str(h)] = [int((mask & (pl[k] != rl[k])).sum()), int(mask.sum())]
            changes = steps = 0
            for k in range(1, N_STEPS + 1):
                valid = pp[k] & pp[k - 1] & rp[k] & rp[k - 1] & ~rc[k] & ~pc[k] \
                        & (pl[k] >= 0) & (pl[k - 1] >= 0)
                changes += int((valid & (pl[k] != pl[k - 1])).sum())
                steps += int(valid.sum())
            rows.append({"i": i, "key": record.get("key"), "world": record.get("world"),
                         "error": None, "car": car, "wrong": wrong,
                         "changes": changes, "steps": steps})
        except Exception as exc:  # noqa: BLE001 - per state, never abort the pass
            import traceback
            rows.append({"i": i, "key": record.get("key"), "world": record.get("world"),
                         "error": f"{type(exc).__name__}: {exc}",
                         "trace": traceback.format_exc(limit=3)})
        print(f"[{variant}] state {i} done ({time.time() - t_start:.0f}s)", flush=True)
    return variant, rows


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    out = {"n_states": len(rows), "n_errors": len(rows) - len(ok), "ds_med": {},
           "blowups": 0, "wrong_lane": {}, "lane_changes_per_hdv_step": None}
    for h in HORIZONS:
        vals = [float(r["car"][str(h)]["hdv"]["ds_med"]) for r in ok
                if r["car"][str(h)]["hdv"].get("ds_med") is not None]
        out["ds_med"][str(int(h))] = statistics.median(vals) if vals else None
    for r in ok:
        a = r["car"].get("60.0", {}).get("hdv", {}).get("ds_med")
        b = r["car"].get("120.0", {}).get("hdv", {}).get("ds_med")
        if a is not None and b is not None and a > 50 and b > 100:
            out["blowups"] += 1
    for h in (60, 120):
        wrong = sum(r["wrong"][str(h)][0] for r in ok)
        total = sum(r["wrong"][str(h)][1] for r in ok)
        out["wrong_lane"][str(h)] = {"wrong": wrong, "n": total,
                                     "share": (wrong / total) if total else None}
    changes = sum(r["changes"] for r in ok)
    steps = sum(r["steps"] for r in ok)
    out["lane_changes_per_hdv_step"] = changes / steps if steps else None
    if ok and ok[0].get("error") is None:
        pass
    errors = [f"{r.get('key')}: {r['error']}" for r in rows if r.get("error")]
    if errors:
        out["errors"] = errors[:10]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N states")
    args = ap.parse_args()
    wanted = list(VARIANTS) if args.variants is None else args.variants
    for name in wanted:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name!r}; choose from {sorted(VARIANTS)}")
        if not Path(VARIANTS[name]["ckpt"]).is_file():
            print(f"skip {name}: checkpoint missing ({VARIANTS[name]['ckpt']})", flush=True)
            wanted = [v for v in wanted if v != name]
    n_states = len(iter_records(WALL_SAME))
    hi = min(n_states, args.limit) if args.limit else n_states
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"schema": "c4_lanerule_check/1",
               "set": str(WALL_SAME), "n_states": hi,
               "entries": "R5 scoring (entries off, CAV teacher, seed 0, dt 0.5 s)",
               "driver_settings": "default (population prior; physics-only gate zeroed)",
               "variants": {}}
    for variant in wanted:
        t0 = time.time()
        rows_all: list[dict] = [None] * hi
        chunk = 12
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=(variant,)) as pool:
            for _, rows in pool.map(_work, [(variant, lo, min(lo + chunk, hi))
                                            for lo in range(0, hi, chunk)]):
                for row in rows:
                    rows_all[row["i"]] = row
        summary = summarize(rows_all)
        summary["seconds"] = round(time.time() - t0, 1)
        results["variants"][variant] = summary
        print(json.dumps({variant: {k: summary[k] for k in ("ds_med", "blowups",
                                                            "wrong_lane", "n_errors")}}),
              flush=True)
        (OUT_DIR / f"{variant}__rows.json").write_text(
            json.dumps([{k: v for k, v in r.items() if k != "trace"} for r in rows_all],
                       separators=(",", ":")), encoding="utf-8")
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
