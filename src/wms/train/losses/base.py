"""Shared contracts of the WM-S loss terms.

A *chunk term* is evaluated once per truncated-BPTT chunk from a
:class:`LossContext`; an *unroll term* is evaluated once per unroll from an
:class:`UnrollContext` (stability, conservation, whole-batch lane-rate).  Every
term returns a :class:`TermValue` with the tensor that enters the optimised loss
(``value``) and, when the logged number is different (position/speed/lane are
logged without their weight), the tensor to log (``log``).

All per-unroll constants (recorded states, step weights, masks, normaliser) live
in :class:`UnrollTargets` and are computed once; the per-chunk views and the
base Huber losses are cached on the context so the braking terms re-use them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class TermValue:
    """One term's contribution: ``value`` is optimised, ``log`` is what is reported."""

    value: Tensor
    valid_count: Optional[int] = None
    log: Optional[Tensor] = None

    @property
    def logged(self) -> Tensor:
        return self.value if self.log is None else self.log


@dataclass
class UnrollTargets:
    """Whole-unroll recorded targets and normalisers (computed once per unroll)."""

    states: Tensor                 # [B, T+1, K, 7] recorded kinematics
    n_steps: int
    weights: Tensor                # [1, n_steps, 1] normalised step weights
    hdv_mask: Tensor               # [B, n_steps, K] supervised per-car mask
    aux_mask: Tensor               # [B, n_steps, K] HDV, present at the step start
    aux_count: int
    denom: Tensor                  # global valid-count normaliser
    lane_target: Tensor            # [B, n_steps, K] recorded lane index
    loss_valid: Tensor             # [B, T+1, K]
    is_cav: Tensor                 # [B, T+1, K]
    dt_s: float
    position_delta_m: float
    speed_delta_mps: float
    history_lane_head: bool
    prepared: dict[str, Any] = field(default_factory=dict)
    road: Any = None               # aggregate loss only: losses.aggregate.RoadTargets
    pair: Any = None               # paired_diff loss only: losses.paired_diff.PairTargets


@dataclass
class LossContext:
    """One truncated-BPTT chunk ``[c0, c1)`` of an unroll."""

    pred: Mapping[str, Any]        # model.rollout output for the chunk
    current: Mapping[str, Any]     # chunk-initial state
    targets: UnrollTargets
    c0: int
    c1: int
    rate_reference: Optional[Tensor] = None
    cache: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def w(self) -> Tensor:
        return self.targets.weights[:, self.c0:self.c1]

    @cached_property
    def m(self) -> Tensor:
        return self.targets.hdv_mask[:, self.c0:self.c1].to(self.w.dtype)

    @cached_property
    def valid_aux(self) -> Tensor:
        return self.targets.aux_mask[:, self.c0:self.c1]

    @cached_property
    def rec_pos(self) -> Tensor:
        return self.targets.states[:, self.c0 + 1:self.c1 + 1, :, 0]

    @cached_property
    def rec_spd(self) -> Tensor:
        return self.targets.states[:, self.c0 + 1:self.c1 + 1, :, 4]

    @cached_property
    def pos_huber(self) -> Tensor:
        return F.huber_loss(self.pred["positions"], self.rec_pos,
                            delta=self.targets.position_delta_m, reduction="none")

    @cached_property
    def spd_huber(self) -> Tensor:
        return F.huber_loss(self.pred["speeds"], self.rec_spd,
                            delta=self.targets.speed_delta_mps, reduction="none")

    @cached_property
    def raw_residual(self) -> Optional[Tensor]:
        return self.pred.get("raw_residual_mps2")

    @cached_property
    def change_probabilities(self) -> Tensor:
        """Per-slot lane-change probability of the rollout (logits fallback)."""
        change_p = self.pred.get("lane_change_probabilities")
        if change_p is not None:
            return change_p
        # Historical/custom rollout adapters may supply logits only.
        probs = torch.softmax(self.pred["lane_logits"], dim=-1)
        if self.targets.history_lane_head:
            return 1 - probs[..., 0]
        lanes = torch.cat((self.current["lane"].unsqueeze(1), self.pred["lanes"][:, :-1]), 1)
        return 1 - probs.gather(-1, lanes.clamp(0, 2).unsqueeze(-1)).squeeze(-1)


@dataclass
class UnrollContext:
    """End-of-unroll inputs for the stability / conservation / lane-rate terms."""

    model: Any
    initial: Mapping[str, Any]
    present0: Tensor
    targets: UnrollTargets
    conservation: Any = None


class LossTerm:
    """A per-chunk loss term; subclasses return the *weighted* contribution.

    ``shareable`` terms take part in the gradient-norm share diagnostics;
    ``chunk_mean_log`` terms (traffic) are logged as the mean over chunks so the
    logged magnitude does not scale with the unroll length.
    """

    name: str = ""
    shareable: bool = True
    chunk_mean_log: bool = False

    def prepare(self, targets: UnrollTargets) -> None:
        """Attach unroll-constant data to ``targets.prepared`` (default: nothing)."""

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        raise NotImplementedError


class UnrollTerm:
    """A once-per-unroll term; ``differentiable`` terms are back-propagated at once."""

    name: str = ""
    differentiable: bool = False

    def __call__(self, ctx: UnrollContext) -> Optional[TermValue]:
        raise NotImplementedError


@dataclass
class LossBreakdown:
    """Chunk result: optimised total, logged value and weighted tensor per term."""

    total: Tensor
    terms: dict[str, TermValue]
    shares: dict[str, float] = field(default_factory=dict)

    def logged(self) -> dict[str, Tensor]:
        return {name: value.logged for name, value in self.terms.items()}
