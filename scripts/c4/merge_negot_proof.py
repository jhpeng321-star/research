"""V10NEGOT CPU proof: fine-tune the merge-negotiation module and compare rollouts.

The checkpoint is loaded as the evaluator loads it (physics.json included), the
negotiation block is enabled zero-initialised, and a few hundred CPU steps are taken on
TRAIN-val windows built from the recorded start records with the survival loss and the
matched non-yield negatives.  The same starts are then rolled out with the block off
and on: stuck ramp cars, merges and departure distance (metres before the lane end).
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import torch

from wms.config.train import TrainConfig
from wms.eval.history_inputs import HistoryInputs
from wms.eval.normalize import DT_S, recorded_for
from wms.eval.predictor import RolloutFlags, WmsPredictor
from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles
from wms.train.losses.compose import build_loss
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

from merge_negot_diag import analyse_start

TRAIN_STEPS_S = 5.0
DEFAULT_FINETUNE_STEPS = 200


def _frame_at(frames_by_key: dict, t0_s: float, step: int):
    return frames_by_key.get(round((t0_s + step * DT_S) * 2))


def _window(record: dict, history: HistoryInputs, n_steps: int) -> dict:
    t0_s, ids, kin0, is_cav = t0_vehicles(record)
    arm = str(record.get("source_arm") or next(iter(record.get("recorded") or {}), "native"))
    frames = recorded_for(record, arm)
    by_key = {round(float(f["time_s"]) * 2): f for f in frames}
    k = len(ids)
    states = np.zeros((n_steps + 1, k, 7), dtype=np.float32)
    present = np.zeros((n_steps + 1, k), dtype=bool)
    cav = np.zeros((n_steps + 1, k), dtype=bool)
    last = {vid: (float(kin0[i, 0]), int(kin0[i, 1:4].argmax()), float(kin0[i, 4]))
            for i, vid in enumerate(ids)}
    prev_v = np.array([last[vid][2] for vid in ids], dtype=np.float32)
    for step in range(n_steps + 1):
        frame = _frame_at(by_key, t0_s, step) if step else None
        rows = {str(r["id"]): r for r in (frame or {}).get("vehicles", [])
                if r.get("present", True)}
        for i, vid in enumerate(ids):
            row = rows.get(vid)
            if step == 0 or row is None:
                s, lane, v = last[vid]
                if step == 0:
                    present[step, i] = True
                else:
                    present[step, i] = False
                    s, lane, v = states[step - 1, i, 0], int(states[step - 1, i, 1:4].argmax()), \
                        states[step - 1, i, 4]
            else:
                s, lane, v = float(row["s"]), int(row.get("lane", -1)), float(row["v"])
                present[step, i] = lane >= 0
                last[vid] = (s, lane, v)
            states[step, i, 0] = s
            if lane >= 0:
                states[step, i, 1 + lane] = 1.0
            states[step, i, 4] = v
            states[step, i, 5] = (v - prev_v[i]) / DT_S if step else 0.0
            states[step, i, 6] = 5.0
            cav[step, i] = bool(is_cav[i])
        prev_v = states[step, :, 4].copy()
    dh, dhm = history.build(record, ids)
    teacher = np.zeros((n_steps + 1, k, 4), dtype=np.float32)
    teacher[..., 0], teacher[..., 1], teacher[..., 2], teacher[..., 3] = (
        states[..., 0], states[..., 1:4].argmax(-1), states[..., 4], states[..., 5])
    return {"states": states, "present": present, "is_cav": cav,
            "loss_valid": present & (states[..., 1:4].sum(-1) > 0),
            "cav_teacher": teacher,
            "driver_hist": dh[0].numpy(), "driver_hist_mask": dhm[0].numpy()}


def _batch(windows: list[dict], indices: list[int]) -> dict:
    rows = [windows[i] for i in indices]
    kmax = max(w["states"].shape[1] for w in rows)
    hmax = max(w["driver_hist"].shape[0] for w in rows)
    out = {}
    for key in ("states", "present", "is_cav", "loss_valid", "cav_teacher",
                "driver_hist", "driver_hist_mask"):
        padded = []
        for w in rows:
            x = w[key]
            k = w["states"].shape[1]
            if x.ndim == 4 or (x.ndim == 3 and key in ("states", "cav_teacher")):
                pad = np.zeros((x.shape[0], kmax - k, x.shape[2]), dtype=x.dtype)
                x = np.concatenate((x, pad), axis=1)
            elif key.startswith("driver_hist") and key.endswith("mask"):
                pad = np.zeros((hmax - x.shape[0], x.shape[1]), dtype=x.dtype)
                x = np.concatenate((x, pad), axis=0)
            elif key.startswith("driver_hist"):
                pad = np.zeros((hmax - x.shape[0], x.shape[1], x.shape[2]), dtype=x.dtype)
                x = np.concatenate((x, pad), axis=0)
            else:
                pad = np.zeros((x.shape[0], kmax - k), dtype=x.dtype)
                x = np.concatenate((x, pad), axis=1)
            padded.append(x)
        out[key] = torch.as_tensor(np.stack(padded))
    return out


def _finetune(model, predictor, records, steps: int, horizon_s: float):
    settings = TrainSettings.from_config(TrainConfig.from_legacy_flat(predictor.backend.config))
    settings = replace(settings, merge_negotiation=True, merge_survival_weight=1.0,
                       merge_survival_negative_weight=0.5,
                       merge_teacher_frac_schedule=(0.5, 0.25, 0.0),
                       aggregate_weight=0.0, agg_delay_weight=0.0,
                       agg_delay_paired_weight=0.0, paired_diff_weight=0.0,
                       lane_flow_weight=0.0, lane_event_weight=0.0,
                       merge_loss_speed_weight=0.0, merge_loss_discharge_weight=0.0,
                       traffic_loss=False)
    composer = build_loss(settings, physics=model.physics)
    runner = UnrollRunner(model, settings, composer, projection=None, device=torch.device("cpu"))
    # Fine-tune ONLY the negotiation block: the off arm is then the unchanged checkpoint.
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.merge_negotiation.parameters():
        param.requires_grad_(True)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3.0e-4)
    n_steps = max(1, int(round(float(horizon_s) / DT_S)))
    windows = [_window(r, predictor._history, n_steps) for r in records]
    model.train()
    losses = []
    t0 = time.time()
    order = list(range(len(windows)))
    for step in range(int(steps)):
        if step % len(order) == 0:
            np.random.default_rng(1000 + step).shuffle(order)
        idx = order[step % len(order): step % len(order) + 2]
        if len(idx) < 2:
            idx = order[:2]
        batch = _batch(windows, idx)
        model.merge_teacher_frac = 0.5 * (1.0 - step / max(1, steps))
        optimizer.zero_grad(set_to_none=True)
        rec = runner.run(batch, unroll_steps=n_steps)
        if not torch.isfinite(torch.tensor(rec["loss"])):
            raise RuntimeError("non-finite proof loss")
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(rec["loss"]))
    model.eval()
    model.merge_teacher_frac = 0.0
    return {"steps": int(steps), "seconds": time.time() - t0,
            "loss_first": losses[:5], "loss_last": losses[-5:],
            "loss_mean_last20": float(np.mean(losses[-20:])) if losses else None}


def _evaluate(model, predictor, records, horizon_s: float, enabled: bool) -> dict:
    module = model.merge_negotiation
    model.merge_negotiation = module if enabled else None
    model.emit_merge_hazard = True
    model.eval()
    model.merge_teacher_frac = 0.0
    try:
        starts = [analyse_start(r, model, predictor._history, t0_vehicles,
                                recorded_teacher, predictor.backend.tools, horizon_s)
                  for r in records]
    finally:
        model.merge_negotiation = module
    rows = [row for s in starts for row in s["ramp_rows"]]
    dist = [row["departure_distance_m"] for row in rows
            if row["departure_distance_m"] is not None]
    merged = sum(1 for row in rows if row["merged"])
    stuck = sum(1 for row in rows if row["stuck"])
    return {"stuck": stuck, "merged": merged, "ramp_cars": len(rows),
            "mean_departure_distance_m": (sum(dist) / len(dist) if dist else None),
            "starts": starts}


def run_proof(args) -> dict:
    from merge_negot_diag import load_starts

    records = load_starts(args.selection, args.max_starts)
    predictor = WmsPredictor.from_checkpoint(args.wm, RolloutFlags(), device="cpu")
    model = predictor.model
    model.set_merge_negotiation(True)
    before = _evaluate(model, predictor, records, args.horizon, enabled=False)
    train = (_finetune(model, predictor, records, args.finetune_steps or DEFAULT_FINETUNE_STEPS,
                       float(getattr(args, "train_horizon", TRAIN_STEPS_S)))
             if (args.finetune_steps or DEFAULT_FINETUNE_STEPS) > 0 else {})
    after = _evaluate(model, predictor, records, args.horizon, enabled=True)
    doc = {"schema": "c4_merge_negot_proof/1", "wm": args.wm,
           "starts": len(records), "horizon_s": args.horizon, "finetune": train,
           "off": {k: v for k, v in before.items() if k != "starts"},
           "on": {k: v for k, v in after.items() if k != "starts"}}
    doc["summary"] = {
        "starts": len(records),
        "off_stuck": before["stuck"], "on_stuck": after["stuck"],
        "off_merged": before["merged"], "on_merged": after["merged"],
        "off_departure_distance_m": before["mean_departure_distance_m"],
        "on_departure_distance_m": after["mean_departure_distance_m"]}
    doc["markdown"] = _markdown(doc)
    return doc


def _markdown(doc: dict) -> str:
    def f(x):
        return "n/a" if x is None else f"{float(x):.3f}"
    s = doc["summary"]
    t = doc.get("finetune") or {}
    return "\n".join([
        "# V10NEGOT CPU proof", "",
        f"- checkpoint: `{doc['wm']}`; {doc['starts']} TRAIN-val starts; "
        f"rollout {doc['horizon_s']} s; CPU only",
        f"- fine-tune: {t.get('steps', 0)} steps in {f(t.get('seconds'))} s "
        f"(last-20 mean loss {f(t.get('loss_mean_last20'))})", "",
        "| arm | stuck | merged | mean departure distance (m before end) |",
        "|---|---|---|---|",
        f"| negotiation off | {s['off_stuck']} | {s['off_merged']} | "
        f"{f(s['off_departure_distance_m'])} |",
        f"| negotiation on (fine-tuned) | {s['on_stuck']} | {s['on_merged']} | "
        f"{f(s['on_departure_distance_m'])} |", "",
        "Note: only the locally available TRAIN-val starts are used; the full "
        "`merge_calib/data` tree is not in this workspace."])

