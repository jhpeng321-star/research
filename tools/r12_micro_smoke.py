"""R12 local CPU micro-smoke: real optimizer steps of the round-12 config on the local TRAIN slice.

(Derived from tools/r11_micro_smoke.py.)  Round 12 = round 9 + base calibration (physical driver params from the
learned centre, re-initialised residual acceleration read-out) + the round-11 contrast courtesy head (zero init) + the
filled replay presence; fix B is off, the gate scalars are frozen.  Uses the real ``WmSTrainer`` (init from ``--ckpt``
with the R12 init switches, rollout + every round-9 motion loss + the courtesy term + Adam + non-finite guards) on
windows that contain a recorded ramp->lane-1 insertion, then checks: finite losses; no cut-in term; courtesy rows > 0
with contrast controls; the fix-D telemetry; only trainable tensors changed (merge_pivot, gate scalars and every frozen
tensor bitwise identical); the learned centre, the driver head's longitudinal rows, the residual read-out row and the
courtesy head moved.  After the steps it measures each loss term's share of the gradient norm over the trainable set
on one probe chunk (``UnrollRunner`` share mode).  Not a training recipe.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from wms.cli.assemble import build_model, build_physics  # noqa: E402
from wms.config.load import load_train_config  # noqa: E402
from wms.data.c4_episodes import build_c4_datasets  # noqa: E402
from wms.data.collate import collate  # noqa: E402
from wms.train.logging import JsonlLog  # noqa: E402
from wms.train.trainer import WmSTrainer  # noqa: E402


def event_ticks(episode_dir: Path, warmup_ticks: int) -> list[int]:
    kin = np.load(episode_dir / "veh_kin.npy", mmap_mode="r")
    vid = np.load(episode_dir / "vid.npy", mmap_mode="r")
    lane = np.where(kin[..., 1:4].sum(-1) > 0, kin[..., 1:4].argmax(-1), -1)
    ins = (lane[:-1] == 2) & (lane[1:] == 1) & (vid[:-1] == vid[1:]) & (vid[1:] >= 0)
    ticks = np.nonzero(ins.any(-1))[0] + 1
    return [int(t) for t in ticks if t > warmup_ticks]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/c4_wm_v10round12.toml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--horizon-steps", type=int, default=40)
    p.add_argument("--window-steps", type=int, default=0, help="dataset window length (0 = horizon); unroll uses --horizon-steps")
    p.add_argument("--pre-steps", type=int, default=24)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--split", default="train", choices=("train", "val"))
    args = p.parse_args(argv)
    cfg = load_train_config(ROOT / args.config, [
        f"model.init_from={args.ckpt}", "runtime.device=cpu",
        f"runtime.threads={args.threads}", *args.set])
    torch.set_num_threads(args.threads)
    torch.manual_seed(cfg.runtime.seed)
    model = build_model(cfg, build_physics(cfg))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    picked, dataset = [], None
    for root in args.roots:
        built = build_c4_datasets(
            root, horizon_steps=(args.window_steps or args.horizon_steps), history_s=cfg.data.history_s,
            history_stride_s=cfg.data.history_stride_s, entries_v2=bool(cfg.data.entries_v2),
            stride_s=1.0, k_slots=None, role_of=None, junction_fix=cfg.data.junction_fix,
            command_known=True, merge_pivot_history=True)
        ds = built[args.split]   # local roots carry the TRAIN corpus's val-role episodes only
        for index, (ep, tick) in enumerate(ds.positions):
            epdir = Path(ds._loader.root) / ds._loader.entries[ep]["dir"]
            if (epdir / "veh_kin.npy").is_file():
                if not hasattr(ds, "_ev"):
                    ds._ev = {}
                if ep not in ds._ev:
                    ds._ev[ep] = event_ticks(epdir, 600)
                if any(0 <= e - tick - args.pre_steps <= 1 for e in ds._ev[ep]):
                    picked.append((ds, index))
        dataset = ds if dataset is None else dataset
    print("windows with an insertion at step ~%d: %d" % (args.pre_steps, len(picked)), flush=True)
    if args.steps < 1 or len(picked) < args.batch:
        raise ValueError("smoke requires positive steps and enough insertion windows for a batch")
    trainer = WmSTrainer(cfg, model, picked[0][0], val_dataset=None, out_dir=out)
    model = trainer.model
    model.train()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    head_grads = []
    trainable = {n for n, prm in model.named_parameters() if prm.requires_grad}
    log = JsonlLog(out / "train_log.jsonl")
    log.reset()
    rng = np.random.default_rng(0)
    results = []
    for step in range(args.steps):
        chosen = rng.choice(len(picked), size=args.batch, replace=False)
        t_data = time.time()
        batch = collate([picked[i][0][picked[i][1]] for i in chosen])
        print(f"batch built in {time.time() - t_data:.0f}s", flush=True)
        record = trainer.train_batch(batch, unroll_steps=args.horizon_steps,
                                     truncation_steps=args.horizon_steps)
        head_grads.append(float(sum(p.grad.abs().sum() for n, p in model.named_parameters()
                                    if n.startswith("follower_courtesy.") and p.grad is not None)))
        record.update(stage_s=args.horizon_steps * 0.5, step=step + 1)
        log.append(record)
        results.append(record)
        print(f"step {step + 1}: loss {record['loss']:.4f} grad_norm {record.get('grad_norm')} "
              f"prior {record.get('driver_prior_means')} courtesy {record.get('follower_courtesy')} "
              f"rows {record.get('follower_courtesy_n_rows')} extra {record.get('follower_courtesy_replay_filled_extra_rows')} "
              f"seconds {record['seconds']:.0f}", flush=True)
    after = model.state_dict()
    changed = sorted(k for k in after if not torch.equal(after[k], before[k]))
    # Gradient shares of every loss term over the trainable set on one probe chunk (forward-only diagnostic).
    t_share = time.time()
    batch = collate([picked[i][0][picked[i][1]] for i in rng.choice(len(picked), size=args.batch, replace=False)])
    shares = trainer.forward_backward(batch, unroll_steps=args.horizon_steps, backward=False, share=True,
                                      truncation_steps=args.horizon_steps)
    share_seconds = time.time() - t_share
    enc = model.driver_encoder
    lo, hi = enc.physical_lower, enc.physical_upper
    centre_now = (lo + (hi - lo) * torch.sigmoid(enc.centre_logits.detach())).tolist()
    last = enc.head[-1]
    summary = {
        "steps": len(results),
        "losses": [r["loss"] for r in results],
        "all_finite": all(math.isfinite(r["loss"]) for r in results),
        "seconds": [r.get("seconds") for r in results],
        "grad_norm_preclip": [r.get("grad_norm") for r in results],
        "cutin_keys_present": sorted({k for r in results for k in r if k.startswith("cutin_reaction")}),
        "trainable_parameter_names": sorted(trainable),
        "changed_tensors": changed,
        "changed_subset_of_trainable": set(changed) <= trainable,
        "frozen_changed": [k for k in changed if k not in trainable],
        "merge_pivot_changed": [k for k in changed if k.startswith("merge_pivot")],
        "gate_changed": [k for k in changed if k.startswith("gate_")],
        "centre_moved": "driver_encoder.centre_logits" in changed,
        "centre_init_logits": trainer.init_report.get("driver_param_init", {}).get("centre_logits"),
        "centre_values_now": centre_now,
        "driver_head_rows_nonzero": [bool(last.weight[j].abs().sum() + last.bias[j].abs() > 0) for j in range(5)],
        "residual_accel_row0_abs": float(model.residual_net.accel[-1].weight[0].abs().sum()
                                         + model.residual_net.accel[-1].bias[0].abs()),
        "driver_prior_means": [r.get("driver_prior_means") for r in results],
        "courtesy_values": [r.get("follower_courtesy") for r in results],
        "courtesy_rows": [r.get("follower_courtesy_n_rows") for r in results],
        "courtesy_head_mean": [r.get("follower_courtesy_head_mean") for r in results],
        "courtesy_head_min_max": [(r.get("follower_courtesy_head_min"), r.get("follower_courtesy_head_max"))
                                  for r in results],
        "courtesy_head_grad_abs_sum": head_grads,
        "courtesy_head_changed": [k for k in changed if k.startswith("follower_courtesy.")],
        "contrast_n_controls": [r.get("follower_courtesy_contrast_n_controls") for r in results],
        "contrast_shift_mean": [r.get("follower_courtesy_contrast_shift_mean") for r in results],
        "contrast_control_resid_mean": [r.get("follower_courtesy_contrast_control_resid_mean") for r in results],
        "replay_filled_extra_rows": [r.get("follower_courtesy_replay_filled_extra_rows") for r in results],
        "merge_tf_excluded_motion": [r.get("tf_excluded_motion") for r in results],
        "merge_tf_scored_cars": [r.get("tf_scored_cars") for r in results],
        "loss_terms_step1": {k: v for k, v in results[0].items()
                             if isinstance(v, (int, float)) and not isinstance(v, bool)
                             and not k.startswith(("follower_", "support_"))
                             and k not in ("step", "stage_s", "seconds")},
        "loss_grad_share": shares.get("loss_share"),
        "loss_grad_norm": shares.get("loss_grad_norm"),
        "share_seconds": share_seconds,
        "init_report": {k: trainer.init_report.get(k) for k in (
            "expanded", "new_in_model", "shape_mismatch", "unused_from_checkpoint", "follower_courtesy_init",
            "driver_param_init", "residual_accel_reinit")},
        "head_version": int(model.follower_courtesy.version),
    }
    (out / "micro_smoke_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    ok = (summary["all_finite"] and changed and summary["changed_subset_of_trainable"]
          and not summary["merge_pivot_changed"] and not summary["gate_changed"]
          and not summary["cutin_keys_present"] and summary["centre_moved"]
          and all(summary["driver_head_rows_nonzero"]) and summary["residual_accel_row0_abs"] > 0
          and all(v is not None and math.isfinite(v) and v > 0
                  for key in ("courtesy_values", "courtesy_rows", "courtesy_head_grad_abs_sum")
                  for v in summary[key])
          and summary["courtesy_head_changed"]
          and all(v is not None and v > 0 for v in summary["contrast_n_controls"])
          and all(v is not None and v >= 0 for v in summary["replay_filled_extra_rows"])
          and summary["head_version"] == 2)
    if not ok:
        raise RuntimeError("R12 micro-smoke failed (finite loss, coverage or parameter-change checks)")


if __name__ == "__main__":
    main()
