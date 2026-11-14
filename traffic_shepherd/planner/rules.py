"""Decision rules of the PL01_PLAN loop: switch, native-return, abstain.

- Switch rule (step 4): paired saving = current.delay_vs_native_s -
  candidate.delay_vs_native_s per M*K cell. Switch only if the candidate is
  safe, median saving > margin_floor_s + k * std(saving), and the fraction of
  cells with saving > 0 >= agreement. All statistics over ALL M*K cells
  (contract section 6). Wall candidates use the higher wall_agreement /
  wall_margin_k bar (the bar is declared here; wall candidates themselves are
  PL01-B).
- M = 1 (contract v2.1, Part A v2 general): every statistic above flattens
  over the M*K cells, so M = 1 is just the K axis alone -- no special case.
  With K = 1 as well (the perfect-sim K=1 mode) the saving is a single
  cell: std(saving) = 0.0 and the margin collapses to margin_floor_s
  EXACTLY. No uncertainty is invented: no spread floor, no epsilon, no
  widened margin (a deterministic single draw carries no spread to price).
- Native-return rule: if the current plan's median delay_vs_native_s > 0
  (worse than native) or it is unsafe, the group returns to native (locked
  wall members excepted; enforced in loop.py / interface.with_actions).
- Abstain rule (step 0): keep the current plan if the fraction of state
  samples with merge_observed < abstain_min_observed. The fraction is over
  the K samples -- independent of M (v2.1 section 1), so M = 1 changes
  nothing.
- Part A scope (contract v2.1 section 3): the planner decides on TOTAL
  delay (delay_vs_native_s; totals are always comparable). A cross-
  prediction comparison of ``delay_part_a_s`` is legal ONLY between EQUAL
  ``meta.part_a_scope`` values: :func:`compare_part_a` raises the NAMED
  :class:`PartAScopeMismatchError` on a mismatch (or skips with the named
  reason via :func:`part_a_scope_mismatch_reason`). No decision rule here
  compares part A across predictors.

Zero simulation: no SUMO/traci/libsumo import anywhere in this package.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import PlannerConfig
from .interface import (
    ACTION_RANK,
    PlanCandidate,
    Prediction,
    part_a_scope,
)
from .safety import SafetyVerdict, safety_check


def median_cells(values) -> float:
    """Median over ALL M*K cells of a paired array (np.median flattens)."""
    return float(np.median(np.asarray(values, dtype=float)))


def paired_saving(current: Prediction, candidate: Prediction) -> np.ndarray:
    """Per-cell saving of ``candidate`` vs ``current`` [M, K] (positive = better).

    TOTAL delay comparison (contract v2.1 section 3): totals are always
    comparable across predictors, whatever their ``part_a_scope``."""
    return (np.asarray(current.delay_vs_native_s, dtype=float)
            - np.asarray(candidate.delay_vs_native_s, dtype=float))


class PartAScopeMismatchError(ValueError):
    """A cross-prediction ``delay_part_a_s`` comparison across UNEQUAL
    ``meta.part_a_scope`` values (contract v2.1 section 3). Part A of a
    'tracked' predictor and part A of an 'all_vehicles' predictor are
    different quantities; only their TOTALS share an accounting corridor.
    Raised BY NAME by :func:`compare_part_a`; never silently defaulted."""


def part_a_scope_mismatch_reason(current: Prediction, candidate: Prediction) -> Optional[str]:
    """The named skip reason for a cross-scope part A comparison, None when
    the scopes are equal (i.e. the comparison is legal).

    ``"part_a_scope_mismatch: <left> vs <right>"`` with each side rendered
    as its scope or ``"unreported"``. Callers that prefer skipping over
    raising read this first; the mismatch name is stable either way."""
    left, right = part_a_scope(current), part_a_scope(candidate)
    if left == right:
        return None
    return f"part_a_scope_mismatch: {left or 'unreported'} vs {right or 'unreported'}"


def compare_part_a(
    current: Prediction, candidate: Prediction, *, on_mismatch: str = "raise"
) -> Optional[np.ndarray]:
    """Paired ``delay_part_a_s`` difference (current - candidate) [M, K],
    scope-guarded per contract v2.1 section 3.

    Legal only when both predictions report the SAME ``part_a_scope``
    (two unreported scopes count as equal -- nothing distinguishes them).
    On a mismatch: ``on_mismatch="raise"`` (default) raises
    :class:`PartAScopeMismatchError` naming both scopes;
    ``on_mismatch="skip"`` returns None (the caller logs
    :func:`part_a_scope_mismatch_reason`). Any other value raises
    ValueError. This is a DIAGNOSTIC comparison: no planner decision rule
    calls it -- the planner decides on TOTAL delay."""
    if on_mismatch not in ("raise", "skip"):
        raise ValueError(f"on_mismatch must be 'raise' or 'skip', got {on_mismatch!r}")
    reason = part_a_scope_mismatch_reason(current, candidate)
    if reason is not None:
        if on_mismatch == "skip":
            return None
        raise PartAScopeMismatchError(
            "cross-scope delay_part_a_s comparison refused (contract v2.1 "
            "section 3): " + reason + "; compare TOTAL delay instead "
            "(delay_vs_native_s is always comparable)"
        )
    return (np.asarray(current.delay_part_a_s, dtype=float)
            - np.asarray(candidate.delay_part_a_s, dtype=float))


@dataclass(frozen=True)
class SwitchVerdict:
    """Outcome of the switch rule (safety is folded in when provided)."""

    median_saving_s: float
    margin_s: float
    agreement: float
    margin_ok: bool
    agreement_ok: bool
    passes: bool


def switch_check(
    current: Prediction,
    candidate: Prediction,
    config: PlannerConfig,
    *,
    wall: bool = False,
    safety: Optional[SafetyVerdict] = None,
) -> SwitchVerdict:
    """Margin + agreement switch rule over ALL M*K cells of the paired saving.

    M = 1 (contract v2.1): the statistics simply run over the K cells.
    With a single cell (M = K = 1) std(saving) is 0.0, so margin ==
    margin_floor_s exactly -- no spread floor or epsilon is added and none
    is invented for a deterministic single-member output."""
    saving = paired_saving(current, candidate)
    median_saving = median_cells(saving)
    spread = float(np.std(saving))
    multiplier = config.wall_margin_k if wall else config.margin_k
    threshold = config.wall_agreement if wall else config.agreement
    margin = config.margin_floor_s + multiplier * spread
    agreement = float(np.mean(saving > 0.0))
    margin_ok = median_saving > margin
    agreement_ok = agreement >= threshold
    safe = True if safety is None else bool(safety.ok)
    return SwitchVerdict(
        median_saving_s=median_saving,
        margin_s=margin,
        agreement=agreement,
        margin_ok=bool(margin_ok),
        agreement_ok=bool(agreement_ok),
        passes=bool(safe and margin_ok and agreement_ok),
    )


@dataclass(frozen=True)
class NativeReturnVerdict:
    """Native-return rule on the current plan (step 4 exception)."""

    should_return: bool
    reason: Optional[str]  # "unsafe" | "worse_than_native" | None
    median_delay_vs_native_s: float


def native_return_check(
    current: Prediction,
    config: PlannerConfig,
    *,
    safety: Optional[SafetyVerdict] = None,
) -> NativeReturnVerdict:
    """True when the current plan is worse than native or unsafe."""
    verdict = safety if safety is not None else safety_check(current, config)
    median_delay = median_cells(current.delay_vs_native_s)
    if not verdict.ok:
        return NativeReturnVerdict(True, "unsafe", median_delay)
    if median_delay > 0.0:
        return NativeReturnVerdict(True, "worse_than_native", median_delay)
    return NativeReturnVerdict(False, None, median_delay)


@dataclass(frozen=True)
class AbstainVerdict:
    """Abstain rule on the state samples (step 0)."""

    observed_fraction: float
    threshold: float
    should_abstain: bool


def abstain_check(state_samples: Sequence, config: PlannerConfig) -> AbstainVerdict:
    """Abstain if the fraction of samples with merge_observed < threshold.

    The fraction is over the K samples (not M*K cells). At equality the
    planner does NOT abstain (rule is strict '<').
    """
    samples = list(state_samples)
    if not samples:
        raise ValueError("state_samples must be non-empty")
    observed = sum(bool(sample.meta["merge_observed"]) for sample in samples)
    fraction = observed / len(samples)
    return AbstainVerdict(fraction, config.abstain_min_observed,
                          fraction < config.abstain_min_observed)


def candidate_sort_key(candidate: Prediction, plan: PlanCandidate) -> tuple:
    """Deterministic candidate key: (median delay_vs_native, action order, ids).

    Ties break by (median delay_vs_native_s, then action order
    native < damping < pi-external-u, then sorted CAV ids) -- the sorted-id
    tuple of ACTION_RANKs compares lexicographically, so at the first CAV
    (in sorted id order) whose action differs, the lower-ranked action wins.
    """
    ranks = tuple(ACTION_RANK[plan.per_cav[cav]] for cav in sorted(plan.per_cav))
    return (median_cells(candidate.delay_vs_native_s), ranks)
