"""Score the C1 models (+ pure_physics / keep_speed references) on the unseen-wall sets.

CPU only; no training, no SUMO.  One model x one state at a time, bounded
process workers (<= 4), each worker holding a single model and a single state.
Packs / runs are never read here: the inputs are the materialised state sets in
_out/wallc1_states_v2.  Checkpoints are read-only staged copies of the C1FAST
box checkpoints; each is scored with ITS OWN TRAIN_RECEIPT physics flags.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
WM_N1 = WORKSPACE / "output" / "wd01" / "wm_n1"
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))

from wms_eval.c1v2 import _car_compact  # noqa: E402
from wms_eval.data import recorded_for  # noqa: E402
from wms_eval.metrics import SECTIONS, car_metrics, traffic_metrics  # noqa: E402
from wms_eval.models import flags_for, load_model, teacher_force_cavs  # noqa: E402
from wms_eval.wms_plugin import WmsCheckpointModel  # noqa: E402

HORIZONS = (10.0, 30.0, 60.0, 120.0)
DELAY_HORIZONS = (60.0, 120.0)
MODELS = ("struct", "blackbox_traj", "blackbox_id", "pure_physics", "keep_speed")
BOX = Path("E:/research-dense/runs/active/WM-S-C1FAST-20261006/box")
CKPT = WORKSPACE / "_out" / "wall_scores" / "_ckpt"
SETS = {
    "wall_same": WORKSPACE / "_out" / "wallc1_states_v2" / "wall_same.json",
    "wall_pilot": WORKSPACE / "_out" / "wallc1_states_v2" / "wall_pilot.json",
    "seen_controls": WORKSPACE / "_out" / "wallc1_states_v2" / "seen_controls.json",
}
ADAPTER = {
    "struct": ("wms", CKPT / "struct" / "wm_s.pt", "structured", "full_struct"),
    "blackbox_traj": ("blackbox_traj", CKPT / "bb_traj" / "wm_s.pt", "blackbox_traj", "full_bb_traj"),
    "blackbox_id": ("blackbox_id", CKPT / "bb_id" / "wm_s.pt", "blackbox_id", "full_bb_id"),
}


def receipt(dirname: str) -> dict:
    return json.loads((BOX / dirname / "TRAIN_RECEIPT.json").read_text(encoding="utf-8"))


def receipt_physics(dirname: str) -> dict:
    return receipt(dirname)["physics"]


def _tf(base, base_arm="native"):
    def pred(state, controllers, horizon, seed):
        arm = str((controllers or {}).get("arm", "native"))
        frames = base(state, {**(controllers or {}), "arm": base_arm}, horizon, seed)
        return teacher_force_cavs(frames, state, arm)
    return pred


def build_predictor(name: str):
    if name in ADAPTER:
        spec, ckpt, kind, dirname = ADAPTER[name]
        return WmsCheckpointModel(ckpt, flags_for(spec), physics=receipt_physics(dirname),
                                  model_kind=kind).predict_states
    if name == "pure_physics":
        return _tf(load_model("pure_physics")[0])
    if name == "keep_speed":
        return _tf(load_model("keep_speed")[0])
    raise KeyError(name)


def _section_delays(frames, t0_s, v0, horizons):
    out = {}
    for h in horizons:
        selected = [f for f in frames if t0_s <= float(f["time_s"]) < t0_s + h]
        if not selected:
            continue
        bins = traffic_metrics(selected, t0_s=t0_s, bin_s=h, free_speed_mps=v0)
        block = {}
        for section in SECTIONS:
            vals = [bins[b]["sections"][section].get("total_delay") for b in bins]
            vals = [v for v in vals if v is not None and math.isfinite(float(v))]
            if vals:
                block[section] = float(sum(vals))
        out[str(h)] = block
    return out


def evaluate(record: dict, predictor, horizon=120.0, seed=0) -> dict:
    t0_s = float(record["t0_s"])
    v0 = float(record.get("v0_mps", 13.89))
    controllers = record.get("cav_controllers", {})
    arms = {}
    for arm in sorted((record.get("recorded") or {}).keys()):
        try:
            predicted = predictor(record, {**controllers, "arm": arm}, horizon, seed)
            recorded = recorded_for(record, arm)
            car = _car_compact(car_metrics(predicted, recorded, t0_s=t0_s,
                                           horizons=tuple(HORIZONS)))
            delays = {"predicted": _section_delays(predicted, t0_s, v0, DELAY_HORIZONS),
                      "recorded": _section_delays(recorded, t0_s, v0, DELAY_HORIZONS)}
            arms[arm] = {"error": None, "car": car, "delay": delays}
        except Exception as exc:  # one arm must not abort the rest
            arms[arm] = {"error": f"{type(exc).__name__}: {exc}"}
    tl = record.get("wall_timeline") or {}
    return {
        "world": str(record.get("world")), "t0_s": t0_s, "key": record.get("key"),
        "run": record.get("run"), "source_arm": record.get("source_arm"),
        "source_set": record.get("source_set"), "own_arm": record.get("arm"),
        "pack": record.get("pack"), "paired": record.get("paired"),
        "native_pair_status": record.get("native_pair_status"),
        "wall_exposure_status": record.get("wall_exposure_status"),
        "wall_phase_at_t0": record.get("wall_phase_at_t0"),
        "wall_engaged": record.get("wall_engaged"),
        "assembled_rel_s": tl.get("assembled_rel_s"),
        "t0_shift_s": record.get("t0_shift_s"),
        "arms": arms,
    }


_W: dict = {}


def _init_worker(name: str, _physics: str | None):
    os.environ["WMS_EVAL_THREADS"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    _W["predictor"] = build_predictor(name)


def _work(payload):
    index, record = payload
    return index, evaluate(record, _W["predictor"])


def iter_windows(path: Path, limit=None):
    from wms_eval.c1v2 import iter_windows as _iw
    return _iw(path, limit=limit)


def score_model(name: str, set_name: str, path: Path, workers: int, inflight: int,
                limit=None) -> list[dict]:
    records = iter_windows(path, limit=limit)
    if workers <= 1:
        _init_worker(name, None)
        return [res for _, res in (_work((i, rec)) for i, rec in enumerate(records))]
    results = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(name, None)) as pool:
        pending = {}
        it = enumerate(records)
        exhausted = False

        def submit():
            nonlocal exhausted
            if exhausted:
                return
            try:
                i, rec = next(it)
            except StopIteration:
                exhausted = True
                return
            pending[pool.submit(_work, (i, rec))] = i

        for _ in range(max(1, inflight)):
            submit()
        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                i, res = fut.result()
                results[i] = res
                del pending[fut]
                submit()
    return [results[i] for i in sorted(results)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--inflight", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    out_root = WORKSPACE / "_out" / "wall_scores"
    receipt_out = {"schema": "wall_c1_score_receipt/1",
                   "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "workers": args.workers, "entries": False, "horizon_s": 120.0,
                   "models": {}}
    for name in args.models.split(","):
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if name in ADAPTER:
            _, _, _, dirname = ADAPTER[name]
            receipt_out["models"][name] = {
                "spec": ADAPTER[name][0], "receipt": str(BOX / dirname / "TRAIN_RECEIPT.json"),
                "physics": receipt_physics(dirname),
                "checkpoint_sha256": _sha256(ADAPTER[name][1])}
        else:
            receipt_out["models"][name] = {"spec": name, "checkpoint": None,
                                           "forcings": "teacher-forced recorded arm"}
        for set_name in args.sets.split(","):
            dest = out_dir / f"states__{name}__{set_name}.json"
            if dest.exists():
                print(f"skip existing {dest}", flush=True)
                continue
            rows = score_model(name, set_name, SETS[set_name], args.workers,
                               args.inflight, limit=args.limit)
            dest.write_text(json.dumps({"model": name, "set": set_name, "states": rows}),
                            encoding="utf-8")
            print(f"scored {name} [{set_name}]: {len(rows)} states", flush=True)
    (out_root / "SCORE_RECEIPT.json").write_text(
        json.dumps(receipt_out, indent=1), encoding="utf-8")
    return 0


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
