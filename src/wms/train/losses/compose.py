"""Compose the enabled loss terms in the fixed legacy summation order.

The optimised chunk loss is accumulated in exactly the order position, speed,
lane, residual_mag, lane_rate, asym_speed, decel_weight, propagation, traffic,
residual_penalty (then the opt-in C4 aggregate terms agg_field/agg_reward/agg_flow), so the autograd graph (and therefore every gradient bit) is the
one of the pre-refactor trainer.  Terms are enabled by the same flags as before;
with all optional weights at their defaults only position/speed/lane run.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from ..settings import TrainSettings
from .aggregate import AggregateLossConfig, aggregate_terms
from .agg_horizon import AggHorizonConfig, agg_horizon_terms
from .base import (LossBreakdown, LossContext, LossTerm, TermValue, UnrollContext, UnrollTargets,
                   UnrollTerm)
from .braking import AsymSpeedLoss, BrakingParams, DecelWeightLoss, PropagationLoss
from .flip_rate import FlipRateLoss
from .lane import LaneLoss
from .lane_event import LaneEventLoss
from .lane_rate import LaneRateLoss
from .merge import MergeLossConfig, cap_merge_share, merge_terms
from .lane_flow import LaneFlowConfig, lane_flow_terms
from .merge_timing import MergeTimingConfig, merge_timing_terms
from .paired_diff import PairedDiffConfig, PairedDiffTerm, paired_diff_terms
from .residual import ResidualMagnitudeLoss, ResidualPenalty
from .stability import ConservationLoss, StabilityLoss
from .traffic import TrafficLoss, TrafficLossConfig
from .trajectory import PositionLoss, SpeedLoss


class LossComposer:
    """Ordered chunk terms plus once-per-unroll terms."""

    def __init__(self, chunk_terms: Sequence[LossTerm], unroll_terms: Sequence[UnrollTerm] = (),
                 *, braking: BrakingParams, merge_share_cap: Optional[float] = None) -> None:
        names = [t.name for t in (*chunk_terms, *unroll_terms)]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate loss term names: {names}")
        self.chunk_terms = tuple(chunk_terms)
        self.unroll_terms = tuple(unroll_terms)
        self.braking = braking
        #: V8WM ``loss.merge.max_share`` (None = off: the chunk total is the plain sum).
        self.merge_share_cap = merge_share_cap
        self.lane_rate = next((t for t in chunk_terms if isinstance(t, LaneRateLoss)), None)
        # Shared helpers for the runner's no-grad whole-horizon probe and gradient replay.
        self.paired_diff = next((t.shared for t in chunk_terms if isinstance(t, PairedDiffTerm)), None)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(t.name for t in (*self.chunk_terms, *self.unroll_terms))

    def prepare(self, targets: UnrollTargets) -> None:
        """Compute unroll-constant data (recorded braking masks) once."""
        for term in self.chunk_terms:
            term.prepare(targets)

    def __call__(self, ctx: LossContext) -> LossBreakdown:
        """Evaluate every chunk term; the total sums the weighted values in fixed order."""
        total: Any = 0.0
        terms: dict[str, TermValue] = {}
        for term in self.chunk_terms:
            value = term(ctx)
            if value is None:
                continue
            terms[term.name] = value
            total = total + value.value
        if self.merge_share_cap is not None and cap_merge_share(
                terms, float(self.merge_share_cap), ctx.targets.prepared.get("merge"), ctx.c0):
            # V8WM: merge terms scaled to the share cap; re-sum in the same term order.
            total = 0.0
            for value in terms.values():
                total = total + value.value
        return LossBreakdown(total=total, terms=terms)

    def evaluate_unroll(self, ctx: UnrollContext) -> dict[str, TermValue]:
        """Evaluate the once-per-unroll terms (stability, conservation)."""
        out: dict[str, TermValue] = {}
        for term in self.unroll_terms:
            value = term(ctx)
            if value is not None:
                out[term.name] = value
        return out

    def extra_total(self, unroll_values: dict[str, TermValue], like: Tensor) -> Tensor:
        """Sum of the unroll terms in term order (added to the *logged* loss only)."""
        extra = torch.zeros((), device=like.device)
        for name, value in unroll_values.items():
            extra = extra + value.value
        return extra

    def differentiable(self, name: str) -> bool:
        return any(t.name == name and t.differentiable for t in self.unroll_terms)

    def shareable_chunk(self, breakdown: LossBreakdown) -> dict[str, Tensor]:
        """Weighted chunk terms that take part in the gradient-share diagnostics."""
        shareable = {t.name for t in self.chunk_terms if t.shareable}
        return {n: v.value for n, v in breakdown.terms.items() if n in shareable}

    def chunk_mean_terms(self) -> tuple[str, ...]:
        """Terms logged as the mean over chunks instead of the sum."""
        return tuple(t.name for t in self.chunk_terms if t.chunk_mean_log)


def hybrid_residual_weight(settings: TrainSettings) -> float:
    """Resolved residual-magnitude penalty weight (modest default only in hybrid mode)."""
    weight = settings.residual_magnitude_weight
    if weight is None:
        weight = 1.0e-3 if settings.rollout_mode == "hybrid" else 0.0
    return float(weight)


def build_loss(settings: TrainSettings, *, physics: Optional[Any] = None) -> LossComposer:
    """Assemble the composer for ``settings`` (``physics`` supplies dt and merge-zone bounds)."""
    dt_s = 0.5 if physics is None else float(physics.dt_s)
    braking = BrakingParams.from_flat(
        {"asym_speed_alpha": settings.asym_speed_alpha, "decel_weight": settings.decel_weight,
         "decel_threshold_mps2": settings.decel_threshold_mps2,
         "propagation_weight": settings.propagation_weight,
         "prop_window_s": settings.prop_window_s, "prop_max_gap_m": settings.prop_max_gap_m},
        dt_s=dt_s)
    chunk: list[LossTerm] = [
        PositionLoss(settings.position_weight), SpeedLoss(settings.speed_weight),
        LaneLoss(settings.lane_weight, history_head=settings.lane_head == "history")]
    if settings.residual_mag_weight:
        chunk.append(ResidualMagnitudeLoss(settings.residual_mag_weight))
    if settings.lane_rate_weight:
        chunk.append(LaneRateLoss(settings.lane_rate_weight, settings.lane_rate_target))
    if settings.flip_rate_weight:
        chunk.append(FlipRateLoss(settings.flip_rate_weight, settings.flip_rate_allowance))
    brake_kwargs = {"position_weight": settings.position_weight,
                    "speed_weight": settings.speed_weight}
    if braking.asym_on:
        chunk.append(AsymSpeedLoss(braking, **brake_kwargs))
    if braking.decel_on:
        chunk.append(DecelWeightLoss(braking, **brake_kwargs))
    if braking.prop_on:
        chunk.append(PropagationLoss(braking, **brake_kwargs))
    if settings.traffic_loss:
        chunk.append(TrafficLoss(TrafficLossConfig(
            dt_s=dt_s, bin_s=float(settings.traffic_bin_s),
            free_speed_mps=float(settings.traffic_free_speed_mps),
            section_bounds_m=(float(physics.merge_zone_s0_m), float(physics.merge_zone_s1_m)),
            ramp_start_s=float(settings.traffic_ramp_start_s),
            ramp_end_s=float(settings.traffic_ramp_end_s), weight=float(settings.traffic_weight))))
    residual_weight = hybrid_residual_weight(settings)
    if settings.rollout_mode == "hybrid" and residual_weight > 0.0:
        chunk.append(ResidualPenalty(residual_weight))
    if settings.aggregate_weight > 0.0:
        # C4 aggregate-consistency terms go LAST so every pre-existing term keeps its
        # summation position (flag off: never built, bit-identical).
        chunk.extend(aggregate_terms(AggregateLossConfig(
            dt_s=dt_s, weight=float(settings.aggregate_weight),
            field_weight=float(settings.aggregate_field_weight),
            reward_weight=float(settings.aggregate_reward_weight),
            flow_weight=float(settings.aggregate_flow_weight),
            bin_s=float(settings.aggregate_bin_s),
            every_steps=int(settings.aggregate_every_steps),
            horizon_start_s=float(settings.aggregate_horizon_start_s),
            horizon_full_s=float(settings.aggregate_horizon_full_s),
            cell_temp_m=float(settings.aggregate_cell_temp_m))))
    if getattr(settings, "paired_diff_weight", 0.0) > 0.0:
        # PAIRLOSS paired-difference terms go after every existing term (flag off: never
        # built, bit-identical); they are zero on batches without pair groups.
        chunk.extend(paired_diff_terms(PairedDiffConfig(
            dt_s=dt_s, weight=float(settings.paired_diff_weight),
            position_weight=float(settings.paired_diff_position_weight),
            speed_weight=float(settings.paired_diff_speed_weight),
            aggregate_weight=float(settings.paired_diff_aggregate_weight),
            return_weight=float(getattr(settings, "paired_diff_return_weight", 0.0)),
            rank_weight=float(getattr(settings, "paired_diff_rank_weight", 0.0)),
            rank_margin_veh_s=float(getattr(settings, "paired_diff_rank_margin_veh_s", 5.0)),
            position_delta_m=float(settings.position_huber_delta_m),
            speed_delta_mps=float(settings.speed_huber_delta_mps),
            bin_s=float(settings.aggregate_bin_s),
            cell_temp_m=float(settings.aggregate_cell_temp_m))))
    if getattr(settings, "merge_loss_on", False):
        # V7FT merge-zone calibration terms go after every existing term (flag off: never
        # built, bit-identical).
        chunk.extend(merge_terms(MergeLossConfig(
            dt_s=dt_s, speed_weight=float(settings.merge_loss_speed_weight),
            discharge_weight=float(settings.merge_loss_discharge_weight),
            cell_edges_m=tuple(float(x) for x in settings.merge_loss_cell_edges_m),
            line_m=float(settings.merge_loss_line_m), bin_s=float(settings.merge_loss_bin_s),
            cell_temp_m=float(settings.merge_loss_cell_temp_m),
            cross_temp_m=float(settings.merge_loss_cross_temp_m),
            speed_scale_mps=float(settings.merge_loss_speed_scale_mps),
            discharge_scale_veh=float(settings.merge_loss_discharge_scale_veh),
            speed_cells=(None if getattr(settings, "merge_loss_speed_cells", None) is None
                         else tuple(int(c) for c in settings.merge_loss_speed_cells)))))
    merge_cap = (getattr(settings, "merge_loss_max_share", None)
                 if getattr(settings, "merge_loss_on", False) else None)
    if getattr(settings, "merge_timing_on", False):
        # V8WM per-vehicle merge-timing (survival) term, last chunk term (off: never built).
        chunk.extend(merge_timing_terms(MergeTimingConfig(
            weight=float(settings.merge_timing_weight),
            metrics=bool(settings.merge_timing_metrics),
            merge_lane=2 if physics is None else int(physics.merge_lane),
            zone_end_m=75.44 if physics is None else float(physics.merge_zone_s1_m),
            forced_window_hazard=(physics is not None and getattr(
                physics, "forced_window_mode", "deterministic") == "hazard"))))
    if float(getattr(settings, "lane_flow_weight", 0.0)) > 0.0:
        # V8LANE on-policy lane-flow term, after every existing term (off: never built).
        chunk.extend(lane_flow_terms(LaneFlowConfig(
            weight=float(settings.lane_flow_weight),
            balance=float(getattr(settings, "lane_flow_balance", 0.0)))))
    if getattr(settings, "agg_delay_on", False):
        # A5 long-horizon aggregate delay terms go LAST (off: never built, bit-identical).
        chunk.extend(agg_horizon_terms(AggHorizonConfig(
            dt_s=dt_s, weight=float(settings.agg_delay_weight),
            paired_weight=float(settings.agg_delay_paired_weight),
            start_s=float(settings.agg_delay_start_s),
            horizon_s=float(settings.agg_delay_horizon_s),
            fade_s=float(settings.agg_delay_fade_s),
            bin_s=float(settings.agg_delay_bin_s),
            merge_s0_m=(0.0 if physics is None else float(physics.merge_zone_s0_m)),
            merge_s1_m=(75.44 if physics is None else float(physics.merge_zone_s1_m)),
            merge_lane=(2 if physics is None else int(physics.merge_lane)),
            road_end_m=float(settings.agg_delay_road_end_m),
            sections_m=tuple(float(x) for x in settings.agg_delay_sections_m),
            queue_speed_mps=float(settings.agg_delay_queue_speed_mps),
            region_temp_m=float(settings.agg_delay_region_temp_m),
            queue_temp_mps=float(settings.agg_delay_queue_temp_mps),
            delay_scale_veh_s=float(settings.agg_delay_delay_scale_veh_s),
            queue_scale_veh=float(settings.agg_delay_queue_scale_veh),
            throughput_scale_veh=float(settings.agg_delay_throughput_scale_veh),
            huber_delta=float(settings.agg_delay_huber_delta))))
    unroll: list[UnrollTerm] = []
    if settings.conservation_weight:
        unroll.append(ConservationLoss(settings.conservation_weight))
    if getattr(settings, "lane_event_weight", 0.0):
        # WMLANE A: teacher-forced lane-change event loss of the hazard lane model.
        unroll.append(LaneEventLoss(settings.lane_event_weight,
                                    gamma=settings.lane_event_gamma,
                                    max_weight=settings.lane_event_max_weight))
    if settings.stability_weight:
        unroll.append(StabilityLoss(settings.stability_weight, margin=settings.stability_margin,
                                    probes=settings.stability_probes))
    return LossComposer(chunk, unroll, braking=braking,
                        merge_share_cap=None if merge_cap is None else float(merge_cap))
