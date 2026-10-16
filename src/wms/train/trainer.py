"""WM-S multi-stage trainer: curriculum loop, validation, selection, checkpoints, receipt.

One trainer instance owns one model.  No CAL/HOLDOUT access exists in this class:
data arrive as datasets built (and source-checked) by ``wms.data``.  The loss and the
chunked unroll live in ``wms.train.losses`` / ``wms.train.unroll``.
"""
from __future__ import annotations

import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from wms.core.errors import WmsError
from wms.data.collate import collate
from wms.model.checkpoint import save_checkpoint, sha256_file
from wms.physics.projection import FeasibilityProjection, ProjectionConfig

from .curriculum import StagePlan, stage_plan
from .logging import JsonlLog, write_json
from .losses.compose import build_loss
from .receipt import write_train_receipt
from .selection import (MONITORS, LongHorizonSelection, MixedHorizonSchedule,
                        MultiHorizonSelection, SameHorizonSelection, eligible_stage_s,
                        validation_horizons)
from .settings import TrainSettings, validate_bounds
from .unroll import UnrollRunner

#: Env hook kept for the box scripts: copy every stage checkpoint here with a timestamp.
STAGE_SNAPSHOT_ENV = "WM_S_STAGE_CKPT_COPY_DIR"


class HardCapRefused(WmsError, ValueError):
    """A hard residual clamp was configured for a new run (REJECTED, change-log #3)."""


class CheckpointInitError(WmsError, RuntimeError):
    """``model.init_from`` did not point at a loadable ``wm_s/1`` checkpoint."""


@dataclass
class TrainResult:
    """Outcome of :meth:`WmSTrainer.fit`."""

    receipt: dict
    checkpoint: Path
    history: list[dict]
    val_by_stage: list[dict]
    selection: Optional[dict] = None
    """SCHED / INFRAV5 selection table (None for the default long-horizon policy)."""


def switch_window_sampler(dataset: Dataset, fraction: float) -> WeightedRandomSampler:
    """Oversample windows whose action timeline switches controller to ``fraction`` of draws."""
    switched = []
    # Lazy indexed inputs are inspected one window at a time; only switch membership is kept.
    records = (dataset.records if getattr(dataset, "_loader", None) is None
               else (dataset[i] for i in range(len(dataset))))
    for record in records:
        action = record.get("action_timeline")
        switched.append(bool(action is not None
                             and np.any(action["controller"][1:] != action["controller"][:-1])))
    n_switch, n_fixed = sum(switched), len(switched) - sum(switched)
    if not n_switch:
        raise ValueError("switch_window_fraction requested but dataset has no switches")
    if fraction < 1.0 and not n_fixed:
        raise ValueError("switch_window_fraction requires fixed windows too")
    weights = [((fraction / n_switch) if flag else ((1.0 - fraction) / n_fixed if n_fixed else 0.0))
               for flag in switched]
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


class WmSTrainer:
    """Train one WM-S member through the 10/30/60/120 s curriculum."""

    def __init__(self, cfg: Any, model: Any, dataset: Dataset, *,
                 val_dataset: Optional[Dataset] = None, out_dir: Path,
                 allow_rejected_hard_cap: bool = False) -> None:
        self.cfg = cfg
        self.settings = TrainSettings.from_config(cfg)
        s = self.settings
        if s.uses_hard_cap and not allow_rejected_hard_cap:
            raise HardCapRefused(
                "hard residual clamp (residual_cap_mps2 / residual_cap_neg_mps2) is REJECTED for "
                "training (change-log #3); use residual_soft_bound_mps2, or pass "
                "allow_rejected_hard_cap=True only to replay/score a legacy configuration")
        if s.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA is unavailable; WM-S fails closed")
        torch.set_num_threads(int(s.threads))
        torch.manual_seed(int(s.seed))
        random.seed(int(s.seed))
        self.device = torch.device(s.device)
        self.model = model.to(self.device)
        if (self.model.history_s != s.history_s or (
                s.history_s is not None and self.model.history_stride_s != s.history_stride_s)):
            raise ValueError("model and trainer history configuration must match")
        self._apply_model_overrides()
        projection = (FeasibilityProjection(ProjectionConfig(min_gap_m=float(s.projection_min_gap_m)))
                      if s.projection_enabled else None)
        self.composer = build_loss(s, physics=self.model.physics)
        self.runner = UnrollRunner(self.model, s, self.composer, projection=projection,
                                   device=self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=s.lr,
                                          weight_decay=s.weight_decay)
        self.dataset, self.val_dataset = dataset, val_dataset
        self.out_dir = Path(out_dir)
        self.init_report = self._init_from_checkpoint(s.init_from)
        self.history: list[dict] = []
        self.stages: list[StagePlan] = stage_plan(s, float(self.model.physics.dt_s))
        self.capture_loss_shares = False

    def _apply_model_overrides(self) -> None:
        """Mirror the flag-gated model bounds onto the model (no-ops when unset)."""
        s, model = self.settings, self.model
        model.apply_direct_commands = False  # recorded teacher or the full dispatcher owns CAVs
        if getattr(s, "lane_hazard_decouple", False):
            # LANEFIX: rollout lane CE / flip-rate do not train the hazard (event loss does).
            model.lane_hazard_detach = True
        if getattr(s, "lane_v5_pretrain", None):
            # LANEV5: offline pre-fit of the lane network (init_from, if any, overrides it).
            from wms.model.lane_v5 import load_pretrain

            model.lane_v5_pretrain_report = load_pretrain(model.lane_v5, s.lane_v5_pretrain)
        if s.residual_cap_mps2 is not None:
            model.residual_cap_mps2 = float(s.residual_cap_mps2)
        if s.lane_logit_cap is not None:
            model.lane_logit_cap = float(s.lane_logit_cap)
        if s.residual_cap_neg_mps2 is not None:
            model.residual_cap_neg_mps2 = float(s.residual_cap_neg_mps2)
        if s.residual_soft_bound_mps2 is not None:
            model.residual_soft_bound_mps2 = s.residual_soft_bound_mps2
            model.residual_soft_bound_neg_mps2 = s.residual_soft_bound_neg_mps2
        validate_bounds(model.residual_soft_bound_mps2, model.residual_soft_bound_neg_mps2,
                        model.residual_cap_mps2, model.residual_cap_neg_mps2)

    def _init_from_checkpoint(self, name: Optional[str]) -> Optional[dict]:
        """Initialise matching weights from a ``wm_s/1`` checkpoint (``model.init_from``).

        Only same-name/same-shape tensors are copied (strict=False); the report lists
        what transferred, what was new in this model and what the checkpoint had that
        this model lacks.  The name resolves through the data registry (an absolute
        path is taken as already resolved), so restricted material is refused.
        """
        if name is None:
            return None
        import torch as _torch

        from ..data.registry import resolve

        path = Path(str(resolve(name)))
        if not path.is_file():
            raise FileNotFoundError(f"model.init_from: {name!r} resolves to {path}, not a file")
        payload = _torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != "wm_s/1":
            raise CheckpointInitError(f"model.init_from: {path} is not a wm_s/1 checkpoint")
        own = self.model.state_dict()
        source = payload.get("model") or {}
        matched = {key: value for key, value in source.items()
                   if key in own and own[key].shape == value.shape}
        self.model.load_state_dict(matched, strict=False)
        report = {
            "source": str(path), "sha256": sha256_file(path),
            "loaded": sorted(matched),
            "new_in_model": sorted(set(own) - set(source)),
            "shape_mismatch": sorted(key for key in set(source) & set(own)
                                     if own[key].shape != source[key].shape),
            "unused_from_checkpoint": sorted(set(source) - set(own)),
        }
        print(f"init_from {path.name}: loaded {len(report['loaded'])} tensors, "
              f"new {len(report['new_in_model'])}, shape-mismatch "
              f"{len(report['shape_mismatch'])}, unused {len(report['unused_from_checkpoint'])}",
              file=sys.stderr)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.out_dir / "INIT_FROM.json", report)
        return report

    # ------------------------------------------------------------------ steps
    def forward_backward(self, batch: dict, *, unroll_steps: int, backward: bool = True,
                         share: bool = False, truncation_steps: Optional[int] = None) -> dict:
        """One unroll pass (see :meth:`UnrollRunner.run`)."""
        return self.runner.run(batch, unroll_steps=unroll_steps, backward=backward, share=share,
                               capture_shares=self.capture_loss_shares,
                               truncation_steps=truncation_steps)

    def train_batch(self, batch: dict, *, unroll_steps: int,
                    truncation_steps: Optional[int] = None) -> dict:
        """One optimizer step; fails closed on non-finite gradients or parameters."""
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        parts = self.forward_backward(batch, unroll_steps=unroll_steps,
                                      truncation_steps=truncation_steps)
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
               for p in self.model.parameters()):
            raise RuntimeError("non-finite WM-S gradient")
        grad_norm = None
        if self.settings.grad_clip:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                             self.settings.grad_clip))
        self.optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in self.model.parameters()):
            raise RuntimeError("non-finite WM-S parameter")
        record = {**parts, "seconds": time.perf_counter() - started, "finite": True}
        if grad_norm is not None:
            # pre-clip global gradient norm and whether the clip was active (round 5b logging)
            record["grad_norm"] = grad_norm
            record["grad_clipped"] = bool(grad_norm > float(self.settings.grad_clip))
        record["lr"] = float(self.optimizer.param_groups[0]["lr"])
        if self.device.type == "cuda":
            record["cuda_peak_mb"] = torch.cuda.max_memory_allocated(self.device) / 2 ** 20
        return record

    def _eval_loader(self, dataset: Dataset) -> DataLoader:
        cap = int(getattr(self.settings, "val_max_windows", 0) or 0)
        if cap and len(dataset) > cap:
            from torch.utils.data import Subset
            step = len(dataset) / cap      # fixed, evenly spaced subset (same windows every time)
            dataset = Subset(dataset, sorted({int(i * step) for i in range(cap)}))
        return DataLoader(dataset, batch_size=self.settings.batch_size, shuffle=False,
                          collate_fn=collate, num_workers=0, drop_last=False)

    @torch.no_grad()
    def evaluate(self, val_dataset: Dataset, *, unroll_steps: int,
                 require_full_horizon: bool = False) -> dict:
        """Forward-only batch means; strict horizons cannot silently truncate.

        LANEFIX: with sampled lane decisions every evaluation replays the same lane draws (a
        forked, fixed-seed RNG), so validation/selection is reproducible and the training RNG
        stream is untouched.  Argmax models take the unchanged path.
        """
        if not getattr(self.model, "_samples_lanes", False):
            return self._evaluate(val_dataset, unroll_steps=unroll_steps,
                                  require_full_horizon=require_full_horizon)
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(self.settings.seed) + 7919)
            return self._evaluate(val_dataset, unroll_steps=unroll_steps,
                                  require_full_horizon=require_full_horizon)

    def _evaluate(self, val_dataset: Dataset, *, unroll_steps: int,
                  require_full_horizon: bool = False) -> dict:
        was_training = self.model.training
        self.model.eval()
        total = {"loss": 0.0, "position": 0.0, "speed": 0.0, "lane": 0.0}
        optional = ("traffic", "residual_mag", "lane_rate", "flip_rate",
                    *self.composer.braking.log_terms, "agg_field", "agg_reward", "agg_flow")
        total.update({name: 0.0 for name in self.composer.names if name in optional})
        total.update({name: 0.0 for name in MONITORS})
        n_batches = valid_targets = available_steps = 0
        reached_steps = None
        for batch in self._eval_loader(val_dataset):
            available_steps = max(available_steps, min(int(unroll_steps), batch["states"].shape[1] - 1))
            if require_full_horizon and batch["states"].shape[1] <= unroll_steps:
                continue  # an unavailable horizon is not selection evidence
            parts = self.forward_backward(batch, unroll_steps=int(unroll_steps), backward=False)
            reached_steps = (parts["unroll_steps"] if reached_steps is None
                             else min(reached_steps, parts["unroll_steps"]))
            for key in total:
                total[key] += float(parts.get(key, 0.0))
            valid_targets += int(parts["valid_targets"])
            n_batches += 1
        if was_training:
            self.model.train()
        mean = {key: (value / n_batches if n_batches else float("nan"))
                for key, value in total.items()}
        reached = ({"reached_steps": reached_steps, "available_steps": available_steps}
                   if require_full_horizon else {})
        return {"n_batches": n_batches, "valid_targets": valid_targets, **reached,
                "unroll_steps": int(unroll_steps), **{f"mean_{k}": v for k, v in mean.items()}}

    @torch.no_grad()
    def monitor(self, dataset: Dataset, *, unroll_steps: int,
                max_batches: Optional[int] = None) -> Optional[float]:
        """Mean forward-only loss over up to ``max_batches`` batches (monitoring only)."""
        was_training = self.model.training
        self.model.eval()
        losses = []
        for i, batch in enumerate(self._eval_loader(dataset)):
            if max_batches is not None and i >= int(max_batches):
                break
            losses.append(float(self.forward_backward(batch, unroll_steps=unroll_steps,
                                                      backward=False)["loss"]))
        if was_training:
            self.model.train()
        return float(np.mean(losses)) if losses else None

    # -------------------------------------------------------------------- fit
    def _make_loader(self, sampler: Optional[WeightedRandomSampler]) -> DataLoader:
        s = self.settings
        workers = int(s.num_workers)
        return DataLoader(self.dataset, batch_size=s.batch_size, shuffle=sampler is None,
                          sampler=sampler, collate_fn=collate, num_workers=workers,
                          persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None,
                          drop_last=False)

    def _save(self, path: Path, **extra) -> None:
        save_checkpoint(path, self.model, self.cfg, extra=extra)

    def fit(self) -> TrainResult:
        """Run every curriculum stage and write checkpoints, logs and the TRAIN receipt."""
        s = self.settings
        if s.select_by_long_horizon and self.val_dataset is None:
            raise ValueError("select-by-long-horizon requires a validation dataset")
        flat = self.cfg.to_legacy_flat()
        out = self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        # Checkpoints do not embed the physics switches (merge_cal, sumo_fidelity ...); the loader
        # (`wms.model.checkpoint`) finds them here, so a mid-run stage checkpoint is scoreable.
        write_json(out / "physics.json", asdict(self.model.physics))
        if s.long_horizon_on:
            write_json(out / "train_config.json", flat)
        if self.composer.braking.any_on:
            write_json(out / "BRAKEFIX_CONFIG.json", flat)
        train_log, val_log, share_log = (JsonlLog(out / n) for n in
                                         ("train_log.jsonl", "val_log.jsonl", "loss_share.jsonl"))
        train_log.reset()
        if self.val_dataset is not None:
            val_log.reset()
        if s.loss_share_logging:
            share_log.reset()
        sampler = (switch_window_sampler(self.dataset, float(s.switch_window_fraction))
                   if s.switch_window_fraction > 0.0 else None)
        # A mixed dataset supplies per-stage sampling weights (source mix ratio and
        # over-sampling of switch/direct windows visible at that stage's horizon).
        stage_weights = getattr(self.dataset, "stage_weights", None)
        set_paired_horizon = (getattr(self.dataset, "set_paired_horizon", None)
                              if s.paired_diff_horizon_mode == "stage" else None)
        loader = (self._make_loader(sampler)
                  if stage_weights is None and set_paired_horizon is None else None)
        selector = LongHorizonSelection()
        # SCHED (default off): multi-horizon selection and short mixed-horizon batches.
        multi = (MultiHorizonSelection(
            horizons=s.selection_horizons_s, references=s.selection_reference,
            eligible_stage_s=eligible_stage_s([p.stage_s for p in self.stages],
                                              s.selection_min_stage_s),
            agg_reward_weight=s.selection_agg_reward_weight)
            if s.selection_mode == "multi_horizon" else None)
        # INFRAV5 (default off): same-horizon selection + end-of-run candidate comparison.
        same = (SameHorizonSelection(
            horizons=s.selection_horizons_s, references=s.selection_reference,
            eligible_stage_s=eligible_stage_s([p.stage_s for p in self.stages],
                                              s.selection_min_stage_s))
            if s.selection_mode == "multi_horizon_v2" else None)
        snapshots: dict[int, dict] = {}   # step -> CPU weights of the v2 candidates
        mixer = (MixedHorizonSchedule(s.mixed_horizon_frac, s.mixed_horizon_s,
                                      s.mixed_horizon_min_stage_s, s.seed)
                 if s.mixed_horizon_frac > 0.0 else None)
        dt_s = float(self.model.physics.dt_s)
        val_by_stage: list[dict] = []
        global_step = 0
        last_batch = None

        def evaluate_horizon(horizon_s: float) -> tuple[dict, float]:
            horizon_steps = max(1, round(horizon_s / self.model.physics.dt_s))
            row = self.evaluate(self.val_dataset, unroll_steps=horizon_steps,
                                require_full_horizon=True)
            if row["n_batches"] == 0 and row.get("available_steps", 0) > 0:
                horizon_steps = row["available_steps"]
                horizon_s = horizon_steps * self.model.physics.dt_s
                row = self.evaluate(self.val_dataset, unroll_steps=horizon_steps,
                                    require_full_horizon=True)
            return row, horizon_s

        def validate_long(stage_s: float, stage_end: bool = False) -> dict:
            rows: list[dict] = []
            for horizon_s in validation_horizons(stage_s, all_horizons=s.val_all_horizons,
                                                 horizons=s.selection_horizons_s):
                row, horizon_s = evaluate_horizon(horizon_s)
                if any(r["horizon_s"] == horizon_s for r in rows):
                    continue
                row.update(stage_s=float(stage_s), step=global_step, horizon_s=horizon_s,
                           selection=False, early_stop=False)
                if same is not None:
                    val_by_stage.append(row)
                    rows.append(row)
                    continue
                if multi is None:
                    row["selection"] = selector.consider(row)
                    if row["selection"]:
                        self._save(out / "wm_s_best.pt", stage_s=float(stage_s),
                                   step=global_step, val=row)
                val_by_stage.append(row)
                rows.append(row)
                if multi is None:
                    val_log.append(row)
            if multi is not None:
                # One decision per validation event (all horizons), then log the rows.
                chosen = multi.consider_event(rows, stage_s=float(stage_s), step=global_step)
                score = multi.table[-1]["score"]
                for row in rows:
                    row.update(selection=chosen, selection_score=score)
                    val_log.append(row)
                if chosen:
                    self._save(out / "wm_s_best.pt", stage_s=float(stage_s), step=global_step,
                               val=rows[-1], selection=multi.table[-1])
            if same is not None:
                flags = same.consider_event(rows, stage_s=float(stage_s), step=global_step,
                                            stage_end=stage_end)
                score = same.table[-1]["score"]
                for row in rows:
                    row.update(selection=flags["global_best"], selection_score=score)
                    val_log.append(row)
                if flags["stage_best"] or flags["stage_last"]:
                    snapshots[global_step] = self._cpu_weights()
                    keep = {c["step"] for c in same.candidates()}
                    for step in [k for k in snapshots if k not in keep]:
                        del snapshots[step]
                if flags["global_best"]:
                    # provisional best (overwritten by the end-of-run comparison)
                    self._save(out / "wm_s_best.pt", stage_s=float(stage_s), step=global_step,
                               val=rows[-1], selection=same.table[-1], provisional=True)
            return rows[-1]

        def revalidate(step: int, horizons: list) -> list[dict]:
            """INFRAV5: missing horizons of a v2 candidate, on the same fixed val subset."""
            self.model.load_state_dict(snapshots[step])
            stage = next(c["stage_s"] for c in same.candidates() if c["step"] == step)
            rows: list[dict] = []
            for horizon_s in horizons:
                row, horizon_s = evaluate_horizon(float(horizon_s))
                row.update(stage_s=float(stage), step=int(step), horizon_s=horizon_s,
                           selection=False, early_stop=False, revalidated=True)
                rows.append(row)
                val_log.append(row)
            return rows

        for stage_i, plan in enumerate(self.stages):
            stage_s = plan.stage_s
            if s.lr_per_stage is not None:
                for group in self.optimizer.param_groups:
                    group["lr"] = float(s.lr_per_stage[stage_i])
            done = 0
            steps_before = global_step
            last_validation = None
            if set_paired_horizon is not None:
                # Retire persistent workers before changing the parent dataset. Each stage's
                # new loader must copy the updated eligible groups and window horizon.
                loader = None
                set_paired_horizon(plan.unroll_steps)
            if stage_weights is not None:
                gen = torch.Generator().manual_seed(int(s.seed) + 1000 * stage_i)
                loader = self._make_loader(WeightedRandomSampler(
                    torch.as_tensor(stage_weights(float(stage_s)), dtype=torch.double),
                    num_samples=len(self.dataset), replacement=True, generator=gen))
            elif set_paired_horizon is not None:
                loader = self._make_loader(sampler)
            while done < plan.steps:
                for batch in loader:
                    last_batch = batch
                    # Closed-loop shares are measured in the real pass so stateful
                    # controllers are not replayed a second time.
                    self.capture_loss_shares = (s.loss_share_logging
                                                and s.cav_drive == "closed_loop" and done == 0)
                    stage_pair = (s.paired_diff_horizon_mode == "stage"
                                  and s.paired_diff_weight > 0.0 and "pair_ref" in batch)
                    short_s = None if mixer is None or stage_pair else mixer.draw(stage_s)
                    # A mixed (short) batch reuses the stage's loss composer, lr and BPTT
                    # chunk length; a chunk longer than the unroll is one chunk, and the
                    # aggregate ramp/normalisers are computed over the unrolled steps.
                    unroll_steps = (plan.unroll_steps if short_s is None
                                    else max(1, min(plan.unroll_steps, round(short_s / dt_s))))
                    record = self.train_batch(batch, unroll_steps=unroll_steps,
                                              truncation_steps=plan.truncation_steps
                                              or None)
                    global_step += 1
                    record.update(stage_s=float(stage_s), epoch=stage_i + 1, step=global_step)
                    if short_s is not None:
                        record["mixed_horizon_s"] = float(short_s)
                    self.history.append(record)
                    train_log.append(record)
                    if "loss_share" in record:
                        share_log.append({k: record[k] for k in
                                          ("stage_s", "step", "loss_share", "loss_grad_norm")})
                    done += 1
                    if s.select_by_long_horizon and (
                            global_step % s.val_every == 0 or done >= plan.steps):
                        last_validation = (validate_long(stage_s) if same is None else
                                           validate_long(stage_s, stage_end=done >= plan.steps))
                    latest_every = int(os.environ.get("WMS_LATEST_EVERY", "0") or 0)
                    if latest_every and global_step % latest_every == 0:
                        # crash insurance: weights only, overwritten in place (2026-10-16)
                        self._save(out / "wm_s_latest.pt", stage_s=float(stage_s), step=global_step,
                                   val=None)
                    if done >= plan.steps:
                        break
            if global_step == steps_before:
                raise RuntimeError("training stage completed zero optimizer steps")
            stage_record = None
            if s.select_by_long_horizon:
                stage_record = last_validation
            elif self.val_dataset is not None:
                # Monitoring only: no early stopping, no checkpoint selection.
                stage_record = self.evaluate(self.val_dataset, unroll_steps=plan.unroll_steps)
                stage_record.update(stage_s=float(stage_s), step=global_step, selection=False,
                                    early_stop=False)
            if s.loss_share_logging and last_batch is not None and s.cav_drive != "closed_loop":
                # Diagnostic-only per-stage gradient-norm shares on the last batch.
                parts = self.forward_backward(last_batch, unroll_steps=plan.unroll_steps,
                                              backward=False, share=True)
                share_record = {"stage_s": float(stage_s), "step": global_step,
                                "unroll_steps": int(plan.unroll_steps),
                                "loss_share": parts.get("loss_share")}
                share_log.append(share_record)
                if stage_record is not None:
                    stage_record["loss_share"] = share_record["loss_share"]
            if stage_record is not None and not s.select_by_long_horizon:
                val_by_stage.append(stage_record)
                val_log.append(stage_record)
            stage_path = out / f"wm_s_stage{int(stage_s)}.pt"
            self._save(stage_path, stage_s=float(stage_s), step=global_step, val=stage_record)
            self._snapshot_stage(stage_path)
        checkpoint = out / "wm_s.pt"
        save_checkpoint(checkpoint, self.model, self.cfg, history=self.history,
                        val_by_stage=val_by_stage)
        if same is not None:
            self._final_same_horizon(same, snapshots, revalidate, out)
        receipt = write_train_receipt(
            out, settings=s, flat_config=flat, physics=self.model.physics,
            device=str(self.device), checkpoint=checkpoint, log_path=train_log.path,
            steps=len(self.history), val_by_stage=val_by_stage, val_log_path=val_log.path,
            has_val=self.val_dataset is not None,
            data_receipt=getattr(self.dataset, "receipt", {}),
            val_data_receipt=getattr(self.val_dataset, "receipt", {}),
            selection=selector if multi is None and same is None else (multi or same),
            best_checkpoint=out / "wm_s_best.pt", config_fingerprint=self._fingerprint())
        if multi is None and same is None:
            return TrainResult(receipt, checkpoint, self.history, val_by_stage)
        chosen = multi or same
        table = {**chosen.summary(),
                 "checkpoint": str(out / "wm_s_best.pt") if chosen.best else None}
        return TrainResult(receipt, checkpoint, self.history, val_by_stage, selection=table)

    def _cpu_weights(self) -> dict:
        return {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}

    def _final_same_horizon(self, same: SameHorizonSelection, snapshots: dict, revalidate,
                            out: Path) -> None:
        """INFRAV5: complete every candidate on the full horizon set, pick, write wm_s_best.pt.

        The model is restored to its final (last-optimizer-step) weights afterwards."""
        final_weights = self._cpu_weights()
        try:
            final = same.finalize(revalidate)
            if final is not None and same.best is not None:
                row = next(r for r in final["table"] if r["selected"])
                self.model.load_state_dict(snapshots[same.best["step"]])
                self._save(out / "wm_s_best.pt", stage_s=float(same.best["stage_s"]),
                           step=int(same.best["step"]), val=same.best["rows"][-1],
                           selection=row, final_comparison=final)
        finally:
            self.model.load_state_dict(final_weights)

    def _fingerprint(self) -> Optional[str]:
        fingerprint = getattr(self.cfg, "fingerprint", None)
        return fingerprint() if callable(fingerprint) else None

    @staticmethod
    def _snapshot_stage(stage_path: Path) -> None:
        """Optional after-each-stage copy (env hook) so stages can be scored while training."""
        snapshot_dir = os.environ.get(STAGE_SNAPSHOT_ENV)
        if snapshot_dir:
            dest = Path(snapshot_dir)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stage_path, dest / f"{stage_path.stem}_{time.strftime('%Y%m%dT%H%M%S')}.pt")
