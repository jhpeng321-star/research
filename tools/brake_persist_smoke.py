"""CPU smoke of the V10PERSIST braking-persistence terms on TRAIN-val windows.

Reads only ``merge_calib/data`` TRAIN-val episodes (never evaluation seeds, sealed or
holdout paths), selects four windows with the most recorded HDV braking onsets, and runs
a few optimizer steps from the v5m checkpoint.  Reports onset counts, the persistence
loss terms, whether gradients reach ``residual_net`` and the loss trajectory.

    python tools/brake_persist_smoke.py [--steps 8] [--out smoke.json]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from wms.data.c4_episodes import _one_split, read_source_manifest
from wms.data.collate import collate
from wms.model.checkpoint import load_checkpoint
from wms.train.losses.compose import build_loss
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

DATA_ROOT = Path("E:/research-dense/runs/active/C4-V7FT-20261017/merge_calib/data/v5_native")
CKPT = Path("E:/research-dense/runs/active/C4-V7FT-20261017/merge_calib/ckpt/v5m_mcal/"
            "wm_s_best_ap_mcal.pt")
OUT_ROOT = Path("E:/research-dense/runs/active/C4-V7FT-20261017/v10persist")
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
EVAL_SEEDS = range(964000, 964200)
DT = 0.5
STEPS = 120                     # 60 s windows
TRUNCATION = 40                 # 20 s BPTT chunks (> the 4 s persistence window)


def guard(path: Path) -> Path:
    text = str(path).replace("\\", "/").lower()
    if any(token in text for token in FORBIDDEN):
        raise SystemExit(f"refusing restricted path {path}")
    return path


def _slice_item(item: dict, max_slots: int) -> dict:
    """Keep the first ``max_slots`` slots of a window (CPU smoke memory bound)."""
    out = dict(item)
    for key in ("states", "present", "loss_valid", "slot_index", "is_cav", "cav_teacher",
                "road_present", "road_continue"):
        out[key] = item[key][:, :max_slots]
    for key in ("hist", "hist_mask", "vehicle_ids", "driver_hist", "driver_hist_mask"):
        out[key] = item[key][:max_slots]
    timeline = item.get("action_timeline")
    if timeline is not None:
        out["action_timeline"] = {
            key: (value[:, :max_slots] if hasattr(value, "shape") and value.ndim == 2
                  else value) for key, value in timeline.items()}
    return out


def fresh_onsets(item: dict) -> int:
    """Recorded fresh HDV onsets: a < -1 m/s^2 after at least 1 s without braking."""
    a = item["states"][1:, :, 5]
    valid = item.get("loss_valid", item["present"])[1:] & ~item["is_cav"][1:]
    braking = valid & (a < -1.0)
    clear = np.ones_like(braking)
    for lag in (1, 2):
        prev = np.zeros_like(braking)
        prev[:, lag:] = braking[:, :-lag]
        clear &= ~prev
    enough = np.arange(braking.shape[0]) >= 2
    return int((braking & clear & enough[:, None]).sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--window-steps", type=int, default=60)
    parser.add_argument("--max-slots", type=int, default=64)
    parser.add_argument("--scan", type=int, default=64)
    parser.add_argument("--out", default=str(OUT_ROOT / "smoke.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = guard(DATA_ROOT)
    doc = read_source_manifest(root)
    seeds = [int(e["seed"]) for e in doc["episodes"] if e.get("split_role") == "val"]
    bad = [s for s in seeds if s in EVAL_SEEDS]
    if bad:
        raise SystemExit(f"evaluation seeds refused: {bad[:3]}")
    loaded = load_checkpoint(guard(CKPT), device="cpu")
    settings = TrainSettings.from_flat(loaded.legacy_config)
    settings = replace(settings, position_weight=0.0, speed_weight=0.0, lane_weight=0.0,
                       traffic_loss=False, brake_persist_weight=1.0,
                       brake_persist_quantile_weight=0.5, brake_persist_event_balance=1.0,
                       brake_persist_window_s=4.0, truncation_steps=TRUNCATION)
    window_steps = int(args.window_steps)
    val, _ = _one_split(root, doc, "val", horizon_steps=window_steps,
                        history_s=settings.history_s,
                        history_stride_s=settings.history_stride_s,
                        entries_v2=settings.entries_v2, stride_s=10.0,
                        start_s=None, refuse_worlds=None)
    print(f"TRAIN-val windows: {len(val)}", flush=True)
    scored = []
    for i in range(min(int(args.scan), len(val))):
        item = val[i]
        scored.append((fresh_onsets(item), i))
    scored.sort(reverse=True)
    selected = [i for count, i in scored[:4] if count > 0]
    if len(selected) < 4:
        raise SystemExit("fewer than four TRAIN-val windows with recorded onsets")
    items = [_slice_item(val[i], int(args.max_slots)) for i in selected]
    batch = collate(items)
    counts = [fresh_onsets(item) for item in items]
    keys = [item.get("meta", {}).get("key", str(i)) for item, i in zip(items, selected)]
    print(json.dumps({"selected": keys, "onsets": counts}), flush=True)
    composer = build_loss(settings, physics=loaded.model.physics)
    runner = UnrollRunner(loaded.model, settings, composer, projection=None,
                          device=torch.device("cpu"))
    model = loaded.model
    model.train()
    # The v5m residual head is tanh-saturated at its +-1.5 m/s^2 scale on these windows
    # (zero derivative), so the smoke optimises the whole learned dynamics path
    # (temporal / history / k-embedding / driver encoder as well as the residual head).
    dynamics = [(name, p) for name, p in model.named_parameters()
                if not name.startswith("lane_v5.")]
    optimizer = torch.optim.Adam([p for _, p in dynamics], lr=float(args.lr))
    history = []
    for step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(1234)   # same lane draws every step: compare under one draw
        record = runner.run(batch, unroll_steps=window_steps, backward=True,
                            truncation_steps=TRUNCATION)
        grad_norm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum()
                                         for _, p in dynamics if p.grad is not None)))
        dyn_grad = float(sum(p.grad.detach().abs().sum() for _, p in dynamics
                             if p.grad is not None))
        nonzero = [(name, float(p.grad.detach().abs().sum()))
                   for name, p in dynamics
                   if p.grad is not None and float(p.grad.detach().abs().sum()) > 0.0]
        temporal_grad = float(sum(p.grad.detach().abs().sum() for name, p in dynamics
                                  if name.startswith("temporal.") and p.grad is not None))
        residual_grad = float(sum(p.grad.detach().abs().sum() for name, p in dynamics
                                  if name.startswith("residual_net.") and p.grad is not None))
        history.append({"step": step, "loss": float(record["loss"]),
                        "nonzero_grad_params": len(nonzero),
                        "sample_grad_params": nonzero[:8],
                        "brake_persist": float(record.get("brake_persist", 0.0)),
                        "brake_persist_quantile":
                            float(record.get("brake_persist_quantile", 0.0)),
                        "onsets": int(record.get("brake_persist_onsets", 0)),
                        "dynamics_grad_norm": grad_norm,
                        "dynamics_grad_abs_sum": dyn_grad,
                        "temporal_grad_abs_sum": temporal_grad,
                        "residual_grad_abs_sum": residual_grad,
                        "finite": bool(np.isfinite(float(record["loss"])))})
        print(json.dumps(history[-1]), flush=True)
        optimizer.step()
    out = {"data_root": str(root), "checkpoint": str(CKPT), "windows": keys,
           "window_onset_counts": counts, "total_onsets": int(sum(counts)),
           "settings": {"weight": settings.brake_persist_weight,
                        "quantile_weight": settings.brake_persist_quantile_weight,
                        "event_balance": settings.brake_persist_event_balance,
                        "window_s": settings.brake_persist_window_s,
                        "truncation_steps": settings.truncation_steps},
           "history": history,
           "loss_decreased": bool(history[-1]["loss"] < history[0]["loss"]),
           "gradients_reach_dynamics": bool(all(h["dynamics_grad_abs_sum"] > 0.0
                                                for h in history)),
           "all_finite": bool(all(h["finite"] for h in history))}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("total_onsets", "loss_decreased",
                                          "gradients_reach_dynamics", "all_finite")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

