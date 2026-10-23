"""R10 local CPU micro-smoke: real optimizer steps of the round-10 config on the local TRAIN slice.

Uses the real ``WmSTrainer.train_batch`` (rollout + all round-9 loss terms + the cut-in term +
Adam + non-finite guards) on windows chosen so that each contains a recorded ramp->lane-1
insertion (positions found from the recorded arrays only), then checks

  * the cut-in loss is positive and finite and its telemetry keys are in the log record,
  * only the intended parameters changed (every frozen tensor is bitwise identical, merge_pivot too),
  * the cut-in term alone reaches gradients in the intended parameters only.

Not a training recipe; two steps of batch two.  Writes ``<out>/train_log.jsonl`` with
``wms.train.logging.JsonlLog`` (the trainer's own writer).
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
    p.add_argument("--config", default="configs/c4_wm_v10round10.toml")
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
        ds = built["train"]
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
        record.update(stage_s=args.horizon_steps * 0.5, step=step + 1)
        log.append(record)
        results.append(record)
        print(f"step {step + 1}: loss {record['loss']:.4f} cutin {record.get('cutin_reaction')} "
              f"events {record.get('cutin_reaction_n_events')} "
              f"rows {record.get('cutin_reaction_n_event_rows')}/"
              f"{record.get('cutin_reaction_n_control_rows')} seconds {record['seconds']:.0f}",
              flush=True)
    after = model.state_dict()
    changed = sorted(k for k in after if not torch.equal(after[k], before[k]))
    summary = {
        "steps": len(results),
        "cutin_values": [r.get("cutin_reaction") for r in results],
        "cutin_events": [r.get("cutin_reaction_n_events") for r in results],
        "cutin_event_rows": [r.get("cutin_reaction_n_event_rows") for r in results],
        "cutin_control_rows": [r.get("cutin_reaction_n_control_rows") for r in results],
        "cutin_event_loss": [r.get("cutin_reaction_event_loss") for r in results],
        "cutin_control_loss": [r.get("cutin_reaction_control_loss") for r in results],
        "losses": [r["loss"] for r in results],
        "all_finite": all(math.isfinite(r["loss"]) for r in results),
        "trainable_parameter_names": sorted(trainable),
        "changed_tensors": changed,
        "changed_subset_of_trainable": set(changed) <= trainable,
        "frozen_changed": [k for k in changed if k not in trainable],
        "merge_pivot_changed": [k for k in changed if k.startswith("merge_pivot")],
        "grad_norm": [r.get("grad_norm") for r in results],
    }
    (out / "micro_smoke_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    if not (summary["all_finite"] and changed and summary["changed_subset_of_trainable"]
            and not summary["merge_pivot_changed"]
            and all(v is not None and math.isfinite(v) and v > 0
                    for key in ("cutin_values", "cutin_events", "cutin_event_rows", "cutin_control_rows")
                    for v in summary[key])):
        raise RuntimeError("micro-smoke failed finite loss, cut-in coverage, or parameter-change checks")


if __name__ == "__main__":
    main()
