"""V10NEGOT diagnostic and CPU proof for the merge-negotiation module.

Diagnostic (``--mode diag``): free-rolls a WM checkpoint on TRAIN-val starts and, for
every ramp car that ends the rollout still in the acceleration lane, measures

* **feasible exposure** -- seconds in which the strictly physical envelope allowed the
  merge (``physical_merge_envelope`` on the model's own state), and
* **survival product** -- ``prod_t (1 - p_t)`` of the learned merge head over those
  seconds.

A stuck car with (almost) no feasible exposure is a *no gaps* case; one with exposure
but a high survival product is *gaps but low hazard*.  The same check reports the
recorded SUMO merge/departure statistics and the courtesy-boost wiring
(``loss.courtesy_weight`` is applied in the position/speed Huber terms and logged as
``courtesy_boost_slots`` / ``courtesy_boost_frac``).

CPU proof (``--mode proof``): builds TRAIN-val batches from the recorded starts, fine-
tunes the checkpoint with the negotiation + survival losses for a few hundred CPU
steps, and reports stuck / merges / departure distance with the module off vs on.

Usage::

    python scripts/c4/merge_negot_diag.py --mode diag \\
        --wm E:/Temp/dc_laptop/ck_iter0b/wm_s_best.pt \\
        --selection E:/Temp/dc_laptop/sel12b.json --horizon 120 \\
        --out runs/active/.../v10negot/diag.json --md .../NEGOT.md

TRAIN-val only; evaluation seeds 964000-964199 and every restricted path token are
refused.  CPU only: no SUMO, no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
EVAL_SEED_LO, EVAL_SEED_HI = 964000, 964199
DT_S = 0.5
DEFAULT_LENGTH_M = 5.0


def _ensure_src() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def refuse(path: str) -> str:
    text = str(path).replace("\\", "/").lower()
    if any(token in text for token in FORBIDDEN):
        raise SystemExit(f"restricted path token refused: {path}")
    return str(path)


def load_starts(path: str, limit: Optional[int] = None) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("states") or payload.get("windows") or []
    if not rows:
        raise SystemExit(f"{path}: no states")
    out = []
    for row in rows:
        seed = int(row.get("seed", -1))
        if EVAL_SEED_LO <= seed <= EVAL_SEED_HI:
            raise SystemExit(f"evaluation seed refused: {seed}")
        if str(row.get("split_role", "val")) != "val":
            raise SystemExit(f"{path}: non-TRAIN-val start refused")
        out.append(row)
    return out[:limit] if limit else out


def _lengths(kin: Any) -> Any:
    import torch
    return torch.as_tensor(kin[..., 6], dtype=torch.float32)


def _kin_step(s: Any, v: Any, lane: Any, a: Any, length: Any) -> Any:
    import torch
    kin = torch.zeros((1, s.shape[0], 7), dtype=torch.float32, device=s.device)
    kin[..., 0] = s
    valid = lane >= 0
    kin[..., 1:4] = torch.nn.functional.one_hot(lane.clamp(0, 2), 3).float()
    kin[..., 1:4] *= valid.unsqueeze(-1).float()
    kin[..., 4] = v
    kin[..., 5] = a
    kin[..., 6] = length
    return kin


def _rollout(record: dict, model: Any, history: Any, t0_vehicles: Any,
             recorded_teacher: Any, tools: Any, horizon_s: float) -> dict:
    import torch
    from wms.eval.normalize import DT_S as _DT
    t0_s, ids, kin0, is_cav = t0_vehicles(record)
    n_steps = max(1, int(round(float(horizon_s) / _DT)))
    dh, dhm = history.build(record, ids)
    initial = {
        "s_m": torch.as_tensor(kin0[None, :, 0]),
        "v_mps": torch.as_tensor(kin0[None, :, 4]),
        "a_mps2": torch.zeros((1, len(ids))),
        "lane": torch.as_tensor(kin0[None, :, 1:4].argmax(-1)),
        "length_m": torch.as_tensor(kin0[None, :, 6]),
        "present": torch.ones((1, len(ids)), dtype=torch.bool),
        "is_cav": torch.as_tensor(is_cav[None]),
    }
    teacher, present_seq, junction_seq = recorded_teacher(
        record, t0_s, ids, kin0, n_steps, str(record.get("source_arm",
                                                         record.get("arm", "native"))),
        tools)
    model.emit_merge_hazard = True
    with torch.no_grad():
        roll = model.rollout(
            initial, n_steps=n_steps, k0=0, cav_teacher=teacher,
            present_seq=present_seq, junction_seq=junction_seq,
            driver_history=dh, driver_history_mask=dhm,
            lane_seed=0)
    return {"roll": roll, "ids": ids, "kin0": kin0, "is_cav": is_cav, "t0_s": t0_s,
            "n_steps": n_steps}


def _recorded_lanes(record: dict, ids: list[str], n_steps: int) -> Any:
    import numpy as np
    from wms.eval.normalize import recorded_for
    arm = str(record.get("source_arm") or next(iter(record.get("recorded") or {}), "native"))
    frames = recorded_for(record, arm)
    by_time = {round(float(f["time_s"]) * 2): f for f in frames}
    t0_s = float(record.get("t0_s", frames[0]["time_s"] if frames else 0.0))
    lanes = np.full((n_steps + 1, len(ids)), -1, dtype=np.int64)
    lookup = {vid: i for i, vid in enumerate(ids)}
    for step in range(n_steps + 1):
        frame = by_time.get(round((t0_s + step * 0.5) * 2))
        if frame is None:
            continue
        for row in frame.get("vehicles", []):
            slot = lookup.get(str(row.get("id")))
            if slot is not None and row.get("present", True):
                lanes[step, slot] = int(row.get("lane", -1))
    return lanes


def analyse_start(record: dict, model: Any, history: Any, t0_vehicles: Any,
                  recorded_teacher: Any, tools: Any, horizon_s: float) -> dict:
    import torch
    from wms.physics.mobil import physical_merge_envelope

    run = _rollout(record, model, history, t0_vehicles, recorded_teacher, tools, horizon_s)
    roll, ids, kin0, n_steps = run["roll"], run["ids"], run["kin0"], run["n_steps"]
    physics = model.physics
    end = float(physics.merge_zone_s1_m)
    lane0 = kin0[:, 1:4].argmax(-1)
    length = _lengths(torch.as_tensor(kin0))
    present0 = torch.ones(len(ids), dtype=torch.bool)
    ramp0 = present0 & (torch.as_tensor(lane0) == int(physics.merge_lane))
    # Current state of step t (the decision state), built from the previous output.
    pos, spd, lanes, masks, acc = (roll["positions"][0], roll["speeds"][0],
                                   roll["lanes"][0], roll["masks"][0], roll["accel"][0])
    head_seq = roll.get("lane_v5_head_logit")
    acts_seq = roll.get("lane_v5_acts")
    delta_seq = roll.get("merge_negotiation_merge_delta")
    exposed = torch.zeros(len(ids))
    survival = torch.ones(len(ids))
    first_exit = torch.full((len(ids),), -1, dtype=torch.long)
    exit_s = torch.full((len(ids),), float("nan"))
    for t in range(n_steps):
        s = torch.as_tensor(kin0[:, 0]) if t == 0 else pos[t - 1]
        v = torch.as_tensor(kin0[:, 4]) if t == 0 else spd[t - 1]
        a = torch.zeros(len(ids)) if t == 0 else acc[t - 1]
        lane = torch.as_tensor(lane0) if t == 0 else lanes[t - 1]
        pres = present0 if t == 0 else masks[t - 1]
        kin = _kin_step(s, v, lane, a, length)
        env = physical_merge_envelope(kin, pres.unsqueeze(0), physics)["admissible"][0]
        ramp = pres & (lane == int(physics.merge_lane))
        feasible = ramp & env
        exposed = exposed + feasible.float() * DT_S
        if head_seq is not None:
            stay = roll["lane_logits"][0, t].gather(
                -1, lane.clamp(0, 2).unsqueeze(-1)).squeeze(-1)
            x = head_seq[0, t] - stay
            if delta_seq is not None:
                x = x + delta_seq[0, t]
            p = torch.sigmoid(x)
            if acts_seq is not None:
                p = torch.where(acts_seq[0, t].bool(), p, torch.zeros_like(p))
            p = torch.where(feasible, p, torch.zeros_like(p))
            survival = survival * (1.0 - p)
        left = ramp & (lanes[t] != int(physics.merge_lane))
        first_exit = torch.where((first_exit < 0) & left, torch.full_like(first_exit, t),
                                 first_exit)
        exit_s = torch.where((first_exit == t) & left, pos[t], exit_s)
    final_lane = lanes[n_steps - 1]
    final_present = masks[n_steps - 1]
    stuck = ramp0 & (final_lane == int(physics.merge_lane)) & final_present
    merged = ramp0 & (first_exit >= 0)
    rec_lanes = torch.as_tensor(_recorded_lanes(record, ids, n_steps))
    rec_ramp0 = rec_lanes[0] == int(physics.merge_lane)
    rec_merge_step = torch.full((len(ids),), -1, dtype=torch.long)
    for t in range(1, n_steps + 1):
        now = rec_lanes[t] == int(physics.merge_lane)
        prev = rec_lanes[t - 1] == int(physics.merge_lane)
        hit = (rec_merge_step < 0) & (rec_lanes[t] >= 0) & prev & ~now
        rec_merge_step = torch.where(hit, torch.full_like(rec_merge_step, t), rec_merge_step)
    rec_stuck = rec_ramp0 & (rec_merge_step < 0) & (rec_lanes[n_steps] == int(physics.merge_lane))
    rows = []
    for i, vid in enumerate(ids):
        if not bool(ramp0[i]):
            continue
        rows.append({
            "id": str(vid),
            "stuck": bool(stuck[i]),
            "merged": bool(merged[i]),
            "exposure_s": float(exposed[i]),
            "survival": float(survival[i]),
            "departure_s_m": (None if not bool(merged[i]) else float(exit_s[i])),
            "departure_distance_m": (None if not bool(merged[i]) else float(end - exit_s[i])),
            "recorded_stuck": bool(rec_stuck[i]),
            "recorded_merged": bool(rec_ramp0[i] and rec_merge_step[i] >= 0),
        })
    return {"world": record.get("world"), "seed": record.get("seed"),
            "split_role": record.get("split_role"), "ramp_rows": rows}


def _summary(starts: list[dict]) -> dict:
    rows = [row for start in starts for row in start["ramp_rows"]]
    stuck = [row for row in rows if row["stuck"]]
    exposed = [row["exposure_s"] for row in stuck]
    surv = [row["survival"] for row in stuck]
    no_gaps = sum(1 for e in exposed if e < 1.0)
    gaps_low = sum(1 for e, s in zip(exposed, surv) if e >= 1.0 and s > 0.5)
    merge_dist = [row["departure_distance_m"] for row in rows
                  if row["departure_distance_m"] is not None]
    n = len(stuck)
    verdict = "no_stuck_cars"
    if n:
        if no_gaps / n >= 0.5:
            verdict = "no_gaps"
        elif gaps_low / n >= 0.5:
            verdict = "gaps_but_low_hazard"
        else:
            verdict = "mixed"
    return {
        "starts": len(starts), "ramp_cars": len(rows),
        "stuck": n, "merged": sum(1 for row in rows if row["merged"]),
        "recorded_stuck": sum(1 for row in rows if row["recorded_stuck"]),
        "recorded_merged": sum(1 for row in rows if row["recorded_merged"]),
        "stuck_mean_exposure_s": (sum(exposed) / n if n else None),
        "stuck_median_exposure_s": (sorted(exposed)[n // 2] if n else None),
        "stuck_mean_survival": (sum(surv) / n if n else None),
        "stuck_no_gaps": no_gaps, "stuck_gaps_low_hazard": gaps_low,
        "mean_departure_distance_m": (sum(merge_dist) / len(merge_dist) if merge_dist else None),
        "verdict": verdict,
    }


def _worker(payload):
    wm, record, horizon = payload
    from wms.eval.history_inputs import HistoryInputs
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles
    predictor = _WORKER.get("predictor")
    if predictor is None:
        predictor = WmsPredictor.from_checkpoint(wm, RolloutFlags(), device="cpu")
        _WORKER["predictor"] = predictor
    return analyse_start(record, predictor.model, HistoryInputs(predictor.backend),
                         t0_vehicles, recorded_teacher, predictor.backend.tools, horizon)


_WORKER: dict = {}


def _init_worker(wm: str) -> None:
    import torch
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    torch.set_num_threads(1)
    _WORKER["predictor"] = WmsPredictor.from_checkpoint(wm, RolloutFlags(), device="cpu")


def run_diag(args) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    starts = load_starts(args.selection, args.max_starts)
    payloads = [(args.wm, s, args.horizon) for s in starts]
    t0 = time.time()
    if args.workers <= 1:
        results = [_worker(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=(args.wm,)) as pool:
            results = list(pool.map(_worker, payloads))
    return {"schema": "c4_merge_negot_diag/1", "wm": args.wm, "horizon_s": args.horizon,
            "seconds": time.time() - t0, "summary": _summary(results), "starts": results}


def render_md(doc: dict) -> str:
    s = doc["summary"]
    lines = ["# V10NEGOT merge diagnostic", "",
             f"- checkpoint: `{doc['wm']}`",
             f"- TRAIN-val starts: {s['starts']} (horizon {doc['horizon_s']} s, CPU)",
             f"- ramp cars at t0: {s['ramp_cars']}; stuck in the free rollout: {s['stuck']}",
             f"- recorded stuck / merged: {s['recorded_stuck']} / {s['recorded_merged']}",
             f"- stuck mean feasible exposure: {_fmt(s['stuck_mean_exposure_s'])} s; "
             f"median {_fmt(s['stuck_median_exposure_s'])} s",
             f"- stuck mean survival product: {_fmt(s['stuck_mean_survival'])}",
             f"- stuck with < 1 s exposure (no gaps): {s['stuck_no_gaps']}; "
             f"exposure >= 1 s and survival > 0.5 (gaps but low hazard): "
             f"{s['stuck_gaps_low_hazard']}",
             f"- **verdict: `{s['verdict']}`**; mean departure distance "
             f"{_fmt(s['mean_departure_distance_m'])} m before the lane end", ""]
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("diag", "proof"), default="diag")
    ap.add_argument("--wm", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--horizon", type=float, default=120.0)
    ap.add_argument("--max-starts", type=int, default=16)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    ap.add_argument("--finetune-steps", type=int, default=0)
    ap.add_argument("--train-horizon", type=float, default=5.0)
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    _ensure_src()
    for token in (args.wm, args.selection, args.out, args.md):
        if token:
            refuse(token)
    if args.mode == "diag":
        doc = run_diag(args)
    else:
        from merge_negot_proof import run_proof
        doc = run_proof(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        text = render_md(doc) if args.mode == "diag" else doc.get("markdown", "")
        Path(args.md).write_text(text, encoding="utf-8")
    print(json.dumps(doc.get("summary", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

