"""Differentiable traffic-level losses for the WM-S rollout (Run 2 / B1).

The per-car Huber/CE objective of the trainer matches individual HDV
targets, but it collapses past ~60 s because the identity-matched vehicle set
empties out (RUN1_EVAL_FAIR).  This module scores *sections* instead: for each
of upstream / merge / downstream (the s_down-frame geometry of
``wms.physics`` and ``wms.eval.metrics``: merge zone ``[0, 75.44]``) it compares
predicted and recorded traffic summaries aggregated over 30 s bins.

Predicted summaries are computed differentiably from the rollout states; counts
(stopped vehicles, merge queue length) use smooth/sigmoid surrogates so the
gradient flows to every vehicle.  Recorded summaries use the same functions on
the recorded target states.  CAV and HDV slots are treated identically: traffic
metrics carry no controller label.  An optional per-step ``entrant`` mask is
excluded from the per-car loss but counted in the predicted traffic aggregate.

All quantities are established for a single ``[B, T, K]`` chunk; the loss groups
steps by their absolute index so a 30 s bin is consistent across the chunked
truncated-BPTT loop.  Flag-gated and OFF by default: when disabled the trainer
never calls this module and its numerics are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import LossContext, LossTerm, TermValue

#: Section order shared with ``wms.eval.metrics.SECTIONS``.
SECTIONS = ("upstream", "merge", "downstream")
#: Metric order used by the loss and the diagnostics.
METRICS = ("mean_speed", "density", "stopped_vehicles", "merge_queue_length",
           "normalized_delay")


@dataclass
class TrafficLossConfig:
    """Smoothing and aggregation settings for the traffic-level loss.

    ``section_bounds_m`` is ``(merge_start_s, merge_end_s)`` in the model's
    s_down frame and is set from :class:`wms.model.PhysicsConfig`
    (``merge_zone_s0_m``/``merge_zone_s1_m``) by the trainer so the loss and the
    eval share one geometry contract.
    """

    dt_s: float = 0.5
    bin_s: float = 30.0
    free_speed_mps: float = 13.89
    section_bounds_m: tuple[float, float] = (0.0, 75.44)
    section_lengths_km: tuple[float, float, float] = (1.0, 0.07544, 1.0)
    section_temp_m: float = 4.0
    stop_speed_mps: float = 0.5
    stop_temp_mps: float = 0.25
    queue_temp_m: float = 2.0
    queue_gate_temp: float = 0.25
    smooth_relu_beta: float = 20.0
    #: Per-metric denominators keeping the MSE terms O(1).  Each value must be a
    #: *typical magnitude* of that metric, not an inverse weight: the term is
    #: ``((pred - target) / scale) ** 2``.  Run 2 shipped scales 50-100x below
    #: the metric ranges (e.g. merge queue length is metres, O(10-300), but was
    #: divided by 0.02), which inflated the term to 1e7-1e9 and drowned the
    #: per-car loss; see RUN2_TRAFFIC_LOSS_DIAG.md.
    #:
    #: The values are per-*bin* typical magnitudes under the per-step-mean
    #: aggregation below (see :func:`_aggregate_bins`): ``mean_speed`` is a speed
    #: (0-14 m/s), ``density`` vehicles/km (O(10-30) upstream, O(100) in the 75 m
    #: merge), ``stopped_vehicles`` a count (0-30), ``merge_queue_length`` metres
    #: (0-300), and ``normalized_delay`` is now a *per-step* delay (O(1-15),
    #: free-speed-normalised seconds per 0.5 s step) rather than a 30 s
    #: accumulation.
    metric_scales: Mapping[str, float] = None  # type: ignore[assignment]
    #: Metric-level multipliers (queue only applies to the merge section).
    metric_weights: Mapping[str, float] = None  # type: ignore[assignment]
    #: Hard cap (in normalized units) on each metric residual before squaring.
    #: Bounds a diverging truncated-BPTT rollout's contribution so one blow-up
    #: cannot dominate the gradient; ``None`` restores the uncapped behaviour.
    residual_cap: float | None = 5.0
    #: Horizon schedule for the loss weight: 0 before ``ramp_start_s``, 1 after
    #: ``ramp_end_s`` (per-car loss dominates the short horizons).
    ramp_start_s: float = 30.0
    ramp_end_s: float = 120.0
    weight: float = 0.1

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0 or self.bin_s <= 0.0:
            raise ValueError("dt_s and bin_s must be positive")
        if self.section_temp_m <= 0.0 or self.stop_temp_mps <= 0.0:
            raise ValueError("section/stop temperatures must be positive")
        if self.ramp_end_s <= self.ramp_start_s:
            raise ValueError("ramp_end_s must exceed ramp_start_s")
        if self.metric_scales is None:
            self.metric_scales = {"mean_speed": 10.0, "density": 20.0,
                                  "stopped_vehicles": 10.0, "merge_queue_length": 100.0,
                                  "normalized_delay": 10.0}
        if self.metric_weights is None:
            self.metric_weights = {name: 1.0 for name in METRICS}
        if self.residual_cap is not None and float(self.residual_cap) <= 0.0:
            raise ValueError("residual_cap must be positive or None")

    @property
    def bin_steps(self) -> int:
        return max(1, int(round(self.bin_s / self.dt_s)))

    def schedule(self, step_indices: Tensor) -> Tensor:
        """Per-step horizon weight in ``[0, 1]`` (0 early, 1 past the ramp)."""
        t = step_indices.to(torch.float64) * self.dt_s
        frac = (t - float(self.ramp_start_s)) / float(self.ramp_end_s - self.ramp_start_s)
        return frac.clamp(0.0, 1.0).to(torch.float32)


def soft_stopped(v_mps: Tensor, *, cfg: TrafficLossConfig) -> Tensor:
    """Smooth indicator of a stopped vehicle (``v < stop_speed``)."""
    return torch.sigmoid((cfg.stop_speed_mps - v_mps) / cfg.stop_temp_mps)


def section_weights(s_m: Tensor, *, bounds: tuple[float, float],
                    temp_m: float) -> Tensor:
    """Soft section membership ``[..., 3]`` (upstream/merge/downstream).

    Uses two sigmoid steps so membership is a smooth partition of unity: the
    hard assignment ``argmax`` recovers ``wms.eval.metrics._section``.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    if temp_m <= 0.0:
        raise ValueError("temp_m must be positive")
    above_lo = torch.sigmoid((s_m - lo) / temp_m)
    above_hi = torch.sigmoid((s_m - hi) / temp_m)
    return torch.stack((1.0 - above_lo, above_lo - above_hi, above_hi), dim=-1)


def _smooth_relu(x: Tensor, beta: float) -> Tensor:
    return F.softplus(x * beta) / beta


def _merge_queue(s_m: Tensor, weight: Tensor, stopped: Tensor, *,
                 cfg: TrafficLossConfig) -> Tensor:
    """Differentiable span of stopped merge vehicles ``[..., 1]``."""
    stop_w = weight * stopped
    tau = cfg.queue_temp_m
    logw = torch.log(stop_w.clamp_min(1.0e-6))
    soft_max = tau * torch.logsumexp(s_m / tau + logw, dim=-1)
    soft_min = -tau * torch.logsumexp(-s_m / tau + logw, dim=-1)
    span = (soft_max - soft_min).clamp_min(0.0)
    total = stop_w.sum(-1)
    gate = torch.sigmoid((total - 0.5) / cfg.queue_gate_temp)
    return gate * span


def section_metrics(s_m: Tensor, v_mps: Tensor, present: Tensor, *,
                    cfg: TrafficLossConfig) -> dict[str, dict[str, Tensor]]:
    """Per-step, per-section traffic summaries for a ``[B, T, K]`` chunk.

    Returns ``{section: {metric: [B, T]}}``.  Mean-like metrics average over
    present vehicles; ``normalized_delay`` is a per-step ``dt_s``
    free-speed-normalised accumulation over present vehicles, and the bin layer
    averages it over the bin's steps like every other metric.
    """
    if s_m.shape != v_mps.shape or s_m.shape != present.shape:
        raise ValueError("s_m, v_mps and present must share [B,T,K]")
    mask = present.to(s_m.dtype)
    sw = section_weights(s_m, bounds=cfg.section_bounds_m,
                         temp_m=cfg.section_temp_m)          # [B,T,K,3]
    w = sw * mask.unsqueeze(-1)
    count = w.sum(-2)                                        # [B,T,3]
    speed_sum = (w * v_mps.unsqueeze(-1)).sum(-2)
    stop = soft_stopped(v_mps, cfg=cfg)                      # [B,T,K]
    stopped = (w * stop.unsqueeze(-1)).sum(-2)
    lengths = torch.as_tensor(cfg.section_lengths_km, dtype=s_m.dtype,
                              device=s_m.device).clamp_min(1.0e-6)
    delay_per_v = _smooth_relu(cfg.free_speed_mps - v_mps, cfg.smooth_relu_beta) \
        / cfg.free_speed_mps
    delay = (w * delay_per_v.unsqueeze(-1)).sum(-2) * cfg.dt_s
    queue = _merge_queue(s_m, sw[..., 1] * mask, stop, cfg=cfg)
    out: dict[str, dict[str, Tensor]] = {}
    for idx, name in enumerate(SECTIONS):
        out[name] = {
            "mean_speed": speed_sum[..., idx] / count[..., idx].clamp_min(1.0e-6),
            "density": count[..., idx] / lengths[idx],
            "stopped_vehicles": stopped[..., idx],
            "merge_queue_length": queue if name == "merge" else torch.zeros_like(
                count[..., idx]),
            "normalized_delay": delay[..., idx],
        }
    return out


def _bin_index(step_indices: Tensor, *, cfg: TrafficLossConfig) -> Tensor:
    return torch.div(step_indices.to(torch.int64), cfg.bin_steps, rounding_mode="floor")


def _aggregate_bins(metrics: dict[str, dict[str, Tensor]], step_indices: Tensor, *,
                    cfg: TrafficLossConfig) -> dict[str, dict[str, Tensor]]:
    """Group per-step metrics into 30 s bins (mean over the bin's steps).

    Returns ``{section: {metric: [n_bins]}}`` (no batch dimension; the loss is
    averaged over batch and bins) with only bins that occur in ``step_indices``.
    Steps arriving in separate chunks land in the same bin by absolute index.

    Every metric, including ``normalized_delay``, is a per-step mean over the
    bin.  The previous code *summed* delay to match ``wms.eval.metrics``
    ``total_delay`` (a window accumulation), but that made the term scale with
    bin length, so it was not comparable to the other per-step summaries and
    grew with horizon; the mean is the same objective up to the fixed
    ``bin_steps`` factor for complete bins, and is horizon-invariant.  The
    per-step ``normalized_delay`` scale was adjusted to match (10.0, not 100.0).
    """
    bins = _bin_index(step_indices, cfg=cfg)
    unique = torch.unique(bins)
    out: dict[str, dict[str, Tensor]] = {name: {} for name in SECTIONS}
    for name in SECTIONS:
        for metric in METRICS:
            values = metrics[name][metric]           # [B, T]
            cols = []
            for bin_id in unique.tolist():
                sel = bins == bin_id
                block = values[:, sel]               # [B, n_sel]
                if block.numel() == 0:
                    continue
                cols.append(block.mean(dim=1))
            out[name][metric] = (torch.stack(cols, dim=1) if cols
                                 else values.new_zeros(values.shape[0], 0))
    return out


def traffic_bin_loss(pred_s: Tensor, pred_v: Tensor, pred_present: Tensor,
                     tgt_s: Tensor, tgt_v: Tensor, tgt_present: Tensor, *,
                     step_indices: Tensor, cfg: TrafficLossConfig,
                     ) -> tuple[Tensor, dict[str, float]]:
    """Section / 30 s-bin traffic MSE between predicted and recorded states.

    ``pred_*`` and ``tgt_*`` are ``[B, T, K]`` on the same time grid.  Returns
    the scalar loss and detached diagnostics (per-metric absolute error means).
    A prediction identical to the target yields exactly 0.
    """
    pred = _aggregate_bins(section_metrics(pred_s, pred_v, pred_present, cfg=cfg),
                           step_indices, cfg=cfg)
    tgt = _aggregate_bins(section_metrics(tgt_s, tgt_v, tgt_present, cfg=cfg),
                          step_indices, cfg=cfg)
    total = pred_s.new_zeros(())
    diag: dict[str, float] = {}
    n_terms = 0
    for name in SECTIONS:
        for metric in METRICS:
            weight = float(cfg.metric_weights.get(metric, 1.0))
            if weight == 0.0:
                continue
            scale = float(cfg.metric_scales[metric])
            p = pred[name][metric]
            t = tgt[name][metric]
            if p.numel() == 0 or p.shape != t.shape:
                continue
            diff = (p - t) / scale
            if cfg.residual_cap is not None:
                diff = diff.clamp(-float(cfg.residual_cap), float(cfg.residual_cap))
            total = total + weight * diff.pow(2).mean()
            diag[f"{name}.{metric}"] = float(diff.detach().abs().mean())
            n_terms += 1
    if n_terms:
        total = total / n_terms
    return total, diag


def scheduled_traffic_weight(step_indices: Tensor, *, cfg: TrafficLossConfig) -> float:
    """Mean horizon weight over a chunk (scales the chunk's traffic term).

    ``cfg.weight * mean(schedule)`` keeps the loss additive across chunks while
    per-car terms dominate before ``ramp_start_s``.
    """
    if step_indices.numel() == 0:
        return 0.0
    return float(cfg.weight) * float(cfg.schedule(step_indices).mean())


def entrant_steps(entrant: Optional[Tensor], n_steps: int, n_target_steps: int,
                  device) -> Optional[Tensor]:
    """Slice an optional per-step entrant mask to the ``n_steps`` rolled steps.

    Accepts ``[B, n_steps+1, K]`` (aligned with ``present``/``cav_teacher``) or
    ``[B, n_steps, K]``; returns ``None`` when absent.
    """
    if entrant is None:
        return None
    mask = entrant.to(device).bool()
    if mask.ndim != 3:
        raise ValueError("entrant mask must be [B,T,K]")
    if mask.shape[1] == n_steps + 1:
        return mask[:, 1:n_steps + 1]
    if mask.shape[1] == n_target_steps:
        return mask[:, :n_steps]
    if mask.shape[1] == n_steps:
        return mask
    raise ValueError(
        f"entrant mask has {mask.shape[1]} steps; expected {n_steps} or "
        f"{n_steps + 1}")


class TrafficLoss(LossTerm):
    """Section/bin traffic MSE, ramped in with the horizon; logged as a per-chunk mean."""

    name = "traffic"
    chunk_mean_log = True

    def __init__(self, cfg: TrafficLossConfig) -> None:
        self.cfg = cfg

    def __call__(self, ctx: LossContext) -> TermValue:
        t = ctx.targets
        idx = torch.arange(ctx.c0, ctx.c1, device=t.states.device)
        tloss, _ = traffic_bin_loss(
            ctx.pred["positions"], ctx.pred["speeds"], ctx.pred["masks"],
            ctx.rec_pos, ctx.rec_spd, t.loss_valid[:, ctx.c0 + 1:ctx.c1 + 1],
            step_indices=idx, cfg=self.cfg)
        return TermValue(scheduled_traffic_weight(idx, cfg=self.cfg) * tloss)
