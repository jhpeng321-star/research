"""WM-S multi-stage trainer: curriculum loop, validation, selection, checkpoints, receipt.

One trainer instance owns one model.  No CAL/HOLDOUT access exists in this class:
data arrive as datasets built (and source-checked) by ``wms.data``.  The loss and the
chunked unroll live in ``wms.train.losses`` / ``wms.train.unroll``.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from wms.core.errors import WmsError
from wms.data.collate import collate
from wms.model.checkpoint import _expand_lane_v5_inputs, save_checkpoint, sha256_file
from wms.model.follower_courtesy import expand_head_inputs
from wms.physics.projection import FeasibilityProjection, ProjectionConfig

from .curriculum import PlateauRule, StagePlan, stage_plan
from .logging import JsonlLog, write_json
from .losses.compose import build_loss
from .receipt import write_train_receipt
from .losses.merge import METRIC_NAMES as MERGE_METRIC_NAMES
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


def build_optimizer(model, *, lr: float, weight_decay: float,
                    freeze_prefixes: tuple[str, ...] = (),
                    lr_scale: Optional[dict[str, float]] = None):
    """R5TOOLS: Adam over trainable parameter groups with optional freezing / LR scales.

    ``freeze_prefixes`` match by ``name.startswith(prefix)`` and set ``requires_grad=False``
    (the tensors are not passed to the optimizer).  ``lr_scale`` maps a prefix to a multiplier
    of the stage learning rate; when prefixes overlap the longest one wins.  With both options
    empty this returns exactly the single-group optimizer over ``model.parameters()``.
    """
    named = list(model.named_parameters())
    scale = dict(lr_scale or {})
    frozen = tuple(freeze_prefixes)
    buckets: dict[str, list] = {"": []}
    frozen_tensors = frozen_parameters = 0
    matched_freeze: set[str] = set()
    matched_scale: set[str] = set()
    for name, param in named:
        hit = next((prefix for prefix in frozen if name.startswith(prefix)), None)
        if hit is not None:
            param.requires_grad_(False)
            frozen_tensors += 1
            frozen_parameters += int(param.numel())
            matched_freeze.add(hit)
            continue
        if not param.requires_grad:
            continue
        keys = [prefix for prefix in scale if name.startswith(prefix)]
        key = max(keys, key=len) if keys else ""
        matched_scale.update(keys)
        buckets.setdefault(key, []).append(param)
    groups = []
    report_groups = []
    for key in [""] + sorted(k for k in buckets if k):
        params = buckets.get(key) or []
        if not params:
            continue
        factor = float(scale.get(key, 1.0))
        groups.append({"params": params, "lr": float(lr) * factor,
                       "weight_decay": float(weight_decay), "lr_scale": factor})
        report_groups.append({"prefix": key or None, "lr_scale": factor,
                              "lr": float(lr) * factor, "tensors": len(params),
                              "parameters": sum(int(p.numel()) for p in params)})
    if not groups:
        raise ValueError("optimizer has no trainable parameters after freeze_prefixes")
    report = {
        "groups": report_groups,
        "freeze_prefixes": list(frozen),
        "frozen_tensors": frozen_tensors,
        "frozen_parameters": frozen_parameters,
        "trainable_tensors": sum(g["tensors"] for g in report_groups),
        "trainable_parameters": sum(g["parameters"] for g in report_groups),
        "total_parameters": sum(int(p.numel()) for _, p in named),
        "unmatched_freeze_prefixes": sorted(set(frozen) - matched_freeze),
        "unmatched_lr_scale_prefixes": sorted(set(scale) - matched_scale),
    }
    return (torch.optim.Adam(groups, lr=lr, weight_decay=weight_decay), report)


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
        self.optimizer, self.optimizer_groups = build_optimizer(
            self.model, lr=s.lr, weight_decay=s.weight_decay,
            freeze_prefixes=tuple(s.freeze_prefixes), lr_scale=s.lr_scale)
        print("WM-S optimizer groups: " + json.dumps(self.optimizer_groups, sort_keys=True),
              file=sys.stderr)
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
        model.checkpoint_forward = bool(getattr(s, "checkpoint_forward", False))
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
        this model lacks.  A narrower lane-v5 first layer is ZERO-expanded to the model's
        ``lane_v5_extra_features`` width first (exact identity), exactly as
        :func:`wms.model.checkpoint.load_checkpoint` does; those tensors are reported as
        ``expanded`` instead of shape mismatches.  The name resolves through the data
        registry (an absolute path is taken as already resolved), so restricted material
        is refused.
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
        raw_source = payload.get("model") or {}
        courtesy_init = None
        if getattr(self.model, "follower_courtesy", None) is not None:
            if bool(getattr(self.settings, "follower_courtesy_init_zero", False)):
                # R11 switch: a fresh zero-initialised head even if the checkpoint has one.
                raw_source = {key: value for key, value in raw_source.items()
                              if not key.startswith("follower_courtesy.")}
                courtesy_init = "zero"
            else:
                courtesy_init = ("checkpoint" if any(key.startswith("follower_courtesy.")
                                                     for key in raw_source) else "zero")
        source = _expand_lane_v5_inputs(raw_source, self.model)
        # R11: a round-10b head (10 pair inputs) zero-expands exactly to the version-2 head.
        source = expand_head_inputs(source, own)
        expanded = sorted(key for key in set(source) & set(raw_source)
                          if source[key].shape != raw_source[key].shape)
        matched = {key: value for key, value in source.items()
                   if key in own and own[key].shape == value.shape}
        self.model.load_state_dict(matched, strict=False)
        report = {
            "source": str(path), "sha256": sha256_file(path),
            "loaded": sorted(matched),
            "expanded": expanded,
            "new_in_model": sorted(set(own) - set(source)),
            "shape_mismatch": sorted(key for key in set(source) & set(own)
                                     if own[key].shape != source[key].shape),
            "unused_from_checkpoint": sorted(set(source) - set(own)),
        }
        if courtesy_init is not None:
            report["follower_courtesy_init"] = courtesy_init
        print(f"init_from {path.name}: loaded {len(report['loaded'])} tensors, "
              f"expanded {len(report['expanded'])}, "
              f"new {len(report['new_in_model'])}, shape-mismatch "
              f"{len(report['shape_mismatch'])}, unused {len(report['unused_from_checkpoint'])}",
              file=sys.stderr)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.out_dir / "INIT_FROM.json", report)
        return report

    def set_stage_lr(self, lr: float) -> None:
        """R5TOOLS: apply a stage learning rate while keeping each group's ``lr_scale``."""
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr) * float(group.get("lr_scale", 1.0))

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
        # Diagnostic only: detached pre-clip L2 norm; never modify the gradients.
        pivot = getattr(self.model, "merge_pivot", None)
        pivot_grad_norm = (math.hypot(*(float(p.grad.detach().double().norm())
                                      for p in pivot.parameters() if p.grad is not None))
                           if pivot is not None else None)
        grad_norm = None
        if self.settings.grad_clip:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                             self.settings.grad_clip))
        self.optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in self.model.parameters()):
            raise RuntimeError("non-finite WM-S parameter")
        record = {**parts, "seconds": time.perf_counter() - started, "finite": True}
        if pivot_grad_norm is not None:
            record["merge_pivot_grad_norm"] = pivot_grad_norm
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
            teacher = getattr(self.model, "merge_teacher_frac", 0.0)
            self.model.merge_teacher_frac = 0.0
            try:
                return self._evaluate(val_dataset, unroll_steps=unroll_steps,
                                      require_full_horizon=require_full_horizon)
            finally:
                self.model.merge_teacher_frac = teacher
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        teacher = getattr(self.model, "merge_teacher_frac", 0.0)
        self.model.merge_teacher_frac = 0.0
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(self.settings.seed) + 7919)
                return self._evaluate(val_dataset, unroll_steps=unroll_steps,
                                      require_full_horizon=require_full_horizon)
        finally:
            self.model.merge_teacher_frac = teacher

    def _evaluate(self, val_dataset: Dataset, *, unroll_steps: int,
                  require_full_horizon: bool = False) -> dict:
        was_training = self.model.training
        self.model.eval()
        total = {"loss": 0.0, "position": 0.0, "speed": 0.0, "lane": 0.0}
        optional = ("traffic", "residual_mag", "lane_rate", "flip_rate",
                    *self.composer.braking.log_terms, "brake_persist",
                    "brake_persist_quantile", "brake_wave", "brake_wave_residual",
                    "brake_wave_spurious", "agg_field", "agg_reward", "agg_flow",
                    "merge_speed", "merge_discharge", "merge_timing")
        total.update({name: 0.0 for name in self.composer.names if name in optional})
        if any(name in ("merge_speed", "merge_discharge") for name in self.composer.names):
            # V7FT merge-zone calibration metrics (whole-unroll, per batch; mean over batches)
            total.update({name: 0.0 for name in MERGE_METRIC_NAMES})
        total.update({name: 0.0 for name in MONITORS})
        timing = "merge_timing" in self.composer.names
        if timing:
            # V8WM merge-timing metrics: pooled sums (ratios recomputed after the loop).
            from .losses.merge_timing import ERRORS_KEY, SUM_KEYS, pooled_metrics
            total.update({name: 0.0 for name in SUM_KEYS})
        timing_errors: list = []
        n_batches = valid_targets = available_steps = 0
        reached_steps = None
        for batch in self._eval_loader(val_dataset):
            available_steps = max(available_steps, min(int(unroll_steps), batch["states"].shape[1] - 1))
            if require_full_horizon and batch["states"].shape[1] <= unroll_steps:
                continue  # an unavailable horizon is not selection evidence
            parts = self.forward_backward(batch, unroll_steps=int(unroll_steps), backward=False)
            reached_steps = (parts["unroll_steps"] if reached_steps is None
                             else min(reached_steps, parts["unroll_steps"]))
            if n_batches == 0 and MERGE_METRIC_NAMES[0] in total:
                # per-cell merge metrics (their number follows loss.merge.cell_edges_m)
                total.update({key: 0.0 for key, value in parts.items()
                              if key.startswith("merge_") and key not in total
                              and isinstance(value, (int, float))
                              and not isinstance(value, bool)})
            for key in total:
                total[key] += float(parts.get(key, 0.0))
            if timing:
                timing_errors.extend(parts.get(ERRORS_KEY, []))
            valid_targets += int(parts["valid_targets"])
            n_batches += 1
        if was_training:
            self.model.train()
        mean = {key: (value / n_batches if n_batches else float("nan"))
                for key, value in total.items()}
        reached = ({"reached_steps": reached_steps, "available_steps": available_steps}
                   if require_full_horizon else {})
        result = {"n_batches": n_batches, "valid_targets": valid_targets, **reached,
                  "unroll_steps": int(unroll_steps), **{f"mean_{k}": v for k, v in mean.items()}}
        if timing and n_batches:
            # pooled over all validation vehicles (ramp |ds|, stuck fractions, merge-time error)
            result.update(pooled_metrics(result, timing_errors))
        return result

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

    @torch.no_grad()
    def _replay_probe(self, replay: Any, *, unroll_steps: int) -> Optional[dict]:
        """V10HORIZON: forward-probe per-window error and update replay high-error priorities.

        Probes an evenly spaced subset of ``replay_probe_windows`` training windows at the
        stage horizon; returns the priority snapshot (or None when probing is off/empty).
        """
        k = int(self.settings.replay_probe_windows)
        n = len(replay)
        if k <= 0 or n <= 0:
            return None
        indices = (list(range(n)) if k >= n
                   else sorted({int(i * (n / k)) for i in range(k)}))
        batch = collate([replay[i] for i in indices])
        errors = self.runner.window_errors(batch, unroll_steps=int(unroll_steps))
        replay.update_errors(indices, errors, decay=float(self.settings.replay_error_decay))
        return replay.error_snapshot()

    # -------------------------------------------------------------------- fit
    def _make_loader(self, sampler: Optional[WeightedRandomSampler], *,
                     batch_size: Optional[int] = None) -> DataLoader:
        s = self.settings
        workers = int(s.num_workers)
        return DataLoader(self.dataset, batch_size=int(batch_size or s.batch_size),
                          shuffle=sampler is None,
                          sampler=sampler, collate_fn=collate, num_workers=workers,
                          persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None,
                          drop_last=False)

    def _save(self, path: Path, **extra) -> None:
        save_checkpoint(path, self.model, self.cfg, extra=extra)

    def _merge_teacher_fraction(self, stage_i: int, done: int, steps: int) -> float:
        """V10NEGOT: per-stage ramp-only merge teacher fraction, decaying over the stage."""
        schedule = tuple(float(x) for x in getattr(
            self.settings, "merge_teacher_frac_schedule", ()))
        if not schedule:
            return 0.0
        base = schedule[min(int(stage_i), len(schedule) - 1)]
        return max(0.0, base * (1.0 - float(done) / max(int(steps), 1)))

    def fit(self) -> TrainResult:
        """Run every curriculum stage and write checkpoints, logs and the TRAIN receipt."""
        s = self.settings
        if s.select_by_long_horizon and self.val_dataset is None:
            raise ValueError("select-by-long-horizon requires a validation dataset")
        if s.plateau_advance and self.val_dataset is None:
            raise ValueError("plateau curriculum advancement requires a validation dataset "
                             "(set selection.select_by_long_horizon or provide val windows)")
        flat = self.cfg.to_legacy_flat()
        out = self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        # Checkpoints do not embed the physics switches (merge_cal, sumo_fidelity ...); the loader
        # (`wms.model.checkpoint`) finds them here, so a mid-run stage checkpoint is scoreable.
        inference_physics = self.inference_physics()
        write_json(out / "physics.json", asdict(inference_physics))
        if inference_physics != self.model.physics:
            # V8WM: trained with these physics; physics.json (what every loader reads) has the
            # physics.inference values applied.
            write_json(out / "physics_train.json", asdict(self.model.physics))
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
        # V10HORIZON: the priority-replay wrapper, when the assembly built one.
        replay = (self.dataset if (s.replay_enabled and hasattr(self.dataset, "update_errors")
                                   and hasattr(self.dataset, "error_snapshot")) else None)
        set_paired_horizon = (getattr(self.dataset, "set_paired_horizon", None)
                              if s.paired_diff_horizon_mode == "stage" else None)
        per_stage_batch = self.settings.batch_size_per_stage
        first_batch = per_stage_batch[0] if per_stage_batch else None
        loader = (self._make_loader(sampler, batch_size=first_batch)
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
                                              s.selection_min_stage_s),
            ramp_weight=float(getattr(s, "selection_ramp_weight", 0.0)),
            stuck_weight=float(getattr(s, "selection_stuck_weight", 0.0)))
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
                self.set_stage_lr(float(s.lr_per_stage[stage_i]))
            done = 0
            steps_before = global_step
            last_validation = None
            # V10HORIZON (default off): advance the stage when its own-horizon validation
            # error plateaus; the rule reads only validation values (no numerics change).
            plateau = (PlateauRule(patience=s.plateau_patience, min_delta=s.plateau_min_delta,
                                   min_steps=s.plateau_min_steps, max_steps=s.plateau_max_steps)
                       if s.plateau_advance else None)
            advanced = False
            # V10HORIZON: forward-probe the per-window error at the stage horizon and refresh
            # the replay priorities before this stage's loader is built.
            replay_snapshot = (self._replay_probe(replay, unroll_steps=plan.unroll_steps)
                               if replay is not None else None)
            stage_bs = per_stage_batch[stage_i] if per_stage_batch else None
            if set_paired_horizon is not None:
                # Retire persistent workers before changing the parent dataset. Each stage's
                # new loader must copy the updated eligible groups and window horizon.
                loader = None
                set_paired_horizon(plan.unroll_steps)
            if stage_weights is not None:
                gen = torch.Generator().manual_seed(int(s.seed) + 1000 * stage_i)
                loader = self._make_loader(WeightedRandomSampler(
                    torch.as_tensor(stage_weights(float(stage_s)), dtype=torch.double),
                    num_samples=len(self.dataset), replacement=True, generator=gen),
                    batch_size=stage_bs)
            elif set_paired_horizon is not None:
                loader = self._make_loader(sampler, batch_size=stage_bs)
            elif per_stage_batch is not None:
                # V10HORIZON: per-horizon batch size (long-horizon memory control).
                loader = self._make_loader(sampler, batch_size=stage_bs)
            while done < plan.steps and not advanced:
                if (replay is not None and s.replay_refresh_every > 0 and done > 0
                        and done % s.replay_refresh_every == 0 and stage_weights is not None):
                    # V10HORIZON online refresh: re-probe and rebuild the loader so the updated
                    # high-error priorities take effect without waiting for the next stage.
                    refreshed = self._replay_probe(replay, unroll_steps=plan.unroll_steps)
                    if refreshed is not None:
                        replay_snapshot = refreshed
                    gen = torch.Generator().manual_seed(int(s.seed) + 1000 * stage_i + done)
                    loader = self._make_loader(WeightedRandomSampler(
                        torch.as_tensor(stage_weights(float(stage_s)), dtype=torch.double),
                        num_samples=len(self.dataset), replacement=True, generator=gen),
                        batch_size=stage_bs)
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
                    self.model.merge_teacher_frac = self._merge_teacher_fraction(
                        stage_i, done, plan.steps)
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
                    if plateau is not None and (
                            global_step % s.val_every == 0 or done >= plan.steps):
                        # V10HORIZON: validate the stage's own (shorter) horizon and advance
                        # the curriculum once its error plateaus.  Forward-only, no grad.
                        prow, p_horizon = evaluate_horizon(stage_s)
                        advanced = plateau.observe(global_step, prow["mean_position"])
                        prow.update(stage_s=float(stage_s), step=global_step, horizon_s=p_horizon,
                                    selection=False, early_stop=False,
                                    plateau_best=(None if not math.isfinite(plateau.best)
                                                  else float(plateau.best)),
                                    plateau_stale=int(plateau.stale),
                                    plateau_observations=int(plateau.observations),
                                    plateau_advanced=bool(advanced),
                                    plateau_reason=(plateau.reason or None))
                        val_by_stage.append(prow)
                        val_log.append(prow)
                        if advanced:
                            last_validation = prow
                            break
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
            if replay is not None and replay_snapshot is not None:
                # V10HORIZON: priority-replay bookkeeping of the stage (error/boost stats).
                if stage_record is not None:
                    stage_record["replay"] = dict(replay_snapshot)
                val_log.append({"stage_s": float(stage_s), "step": global_step,
                                "replay": dict(replay_snapshot)})
            if plateau is not None:
                # V10HORIZON: the stage's plateau decision (best/stale/advanced/reason/history).
                summary = plateau.summary()
                if stage_record is not None:
                    stage_record["plateau"] = dict(summary)
                    stage_record["curriculum_advance"] = bool(plateau.advanced)
                val_log.append({"stage_s": float(stage_s), "step": global_step,
                                "plateau": summary,
                                "curriculum_advance": bool(plateau.advanced),
                                "curriculum_advance_reason": plateau.reason or None})
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
            out, settings=s, flat_config=flat, physics=inference_physics,
            physics_train=(None if inference_physics == self.model.physics
                           else self.model.physics),
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

    def inference_physics(self):
        """V8WM: the physics written to ``physics.json`` (training physics + physics.inference)."""
        overrides = dict(getattr(self.settings, "physics_inference", None) or {})
        if not overrides:
            return self.model.physics
        return replace(self.model.physics, **overrides)

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
