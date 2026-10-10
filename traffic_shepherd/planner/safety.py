"""Absolute and relative safety checks on a Prediction (PL01_PLAN step 0/4),
FAIL CLOSED per contract v1.1 (§8.7, §9).

Owner rule (recorded 2026-09-25, PL01_PLAN "Declared constants"):

- ABSOLUTE: in ALL M*K cells (``abs_safety_quantile`` = 1.0) the plan is
  collision-free with ``min_ttc_s >= ttc_floor_s`` (1.5). A collision or a
  min TTC below the floor in ANY cell vetoes. RULE C (owner, PL-EXP1c; the
  default, ``PlannerConfig.absolute_ttc_rule`` "relative_to_native"): the
  per-cell floor is ``min(ttc_floor_s, NATIVE min TTC of the same cell)``
  (NaN native -> ``ttc_floor_s``), likewise for the G-1 future floor against
  the native future twin; "absolute" restores this rule A exactly. The
  verdict carries a report-only rule-A shadow.
- RELATIVE: for BOTH the config-selected TTC exposure metric
  (:func:`relative_ttc_metric`: ``relative_ttc_threshold_s`` 1.5 -> the
  primary ``ttc_lt1p5_exposure_s`` fields (T-A, PL-EXP1b addendum 2);
  3.0 -> the frozen ``ttc_lt3_exposure_s`` fields) and ``hard_brake_s``,
  the mean increase vs native must be <= ``rel_safety_tol_s`` AND the
  fraction of cells whose increase is <= tol must be >=
  ``rel_safety_quantile`` (0.90). A missing selected field raises the
  named :class:`RelativeSafetyFieldMissingError` (fail closed). With the
  1.5 s selection the frozen 3 s fields carry a report-only SHADOW
  verdict (:attr:`SafetyVerdict.shadow_relative_3s`); hard braking is
  unchanged.

FAIL CLOSED (contract §9, owner hard limit): the §9 vetoes are evaluated
BEFORE any ``min_ttc_s`` read (see :func:`fail_closed_mask`): an invalid
cell (``cell_valid``/``native_cell_valid`` False, or a v1.1 prediction that
omits the diagnostics entirely), a non-finite value in any delay
field (NaN or either infinity, PL01-E3), a collision
(True or a non-bool/NaN-carrying flag), or a wall ``safety_abort`` in a plan
cell vetoes regardless of what min_ttc says. TTC rule (clarification 7 as
amended by §9): NaN ``min_ttc_s`` passes the floor ONLY in a valid cell with
``has_closing_pair == False``; NaN with a closing pair, NaN on a v1.0
prediction (no pair diagnostics), and any other non-finite value are hard
vetoes. All fractions are over ALL M*K cells, matching contract section 6.

PL-06 (owner decision 2026-09-27, option A = the default): with
``PlannerConfig.native_collision_vetoes_plan`` False, a NATIVE collision
alone no longer vetoes the plan -- the plan's own absolute gate (no plan
collision, min TTC >= floor in all cells, the v2 future rules) and the
relative rule vs native still decide, and the native collision is recorded
as a logged 'native_collision' diagnostic (:attr:`SafetyVerdict.
native_collision_cells`, :func:`native_collision_note`, and a decision
reason). Every other native fail-closed term above stays fail-closed
regardless of the flag.

Zero simulation: no SUMO/traci/libsumo import anywhere in this package.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import RULE_TTC_ABSOLUTE, RULE_TTC_RELATIVE_TO_NATIVE, PlannerConfig
from .interface import Prediction

#: Metrics gated by the relative rule, each against its ``native_`` twin.
#: The TTC entry is chosen from the config (T-A):
#: :func:`relative_ttc_metric` maps ``relative_ttc_threshold_s`` 1.5 -> the
#: primary ``ttc_lt1p5_exposure_s`` fields and 3.0 -> the frozen
#: ``ttc_lt3_exposure_s`` fields; hard braking is always gated on
#: ``hard_brake_s``.
RELATIVE_TTC_METRIC_1P5: str = "ttc_lt1p5_exposure_s"
RELATIVE_TTC_METRIC_3: str = "ttc_lt3_exposure_s"
HARD_BRAKE_METRIC: str = "hard_brake_s"
RELATIVE_METRICS: Tuple[str, ...] = (RELATIVE_TTC_METRIC_3, HARD_BRAKE_METRIC)
#: The allowed ``PlannerConfig.relative_ttc_threshold_s`` values (owner,
#: PL-EXP1b addendum 2; no other threshold exists).
RELATIVE_TTC_THRESHOLDS: Tuple[float, ...] = (1.5, 3.0)
TTC_METRIC_BY_THRESHOLD = {
    1.5: RELATIVE_TTC_METRIC_1P5,
    3.0: RELATIVE_TTC_METRIC_3,
}


class RelativeSafetyFieldMissingError(ValueError):
    """The relative rule's config-SELECTED TTC exposure field is ABSENT on
    the prediction (T-A fail-closed rule): with
    ``relative_ttc_threshold_s`` 1.5 a predictor that does not report
    ``ttc_lt1p5_exposure_s`` (or its native_/future_ twins on a v2
    prediction -- e.g. a learned predictor under the frozen v2.1 contract)
    cannot support the primary relative metric, so deciding on it is
    REFUSED with this named error, never silently re-pointed at the 3 s
    field."""


def relative_ttc_metric(config: PlannerConfig) -> str:
    """The config-selected relative TTC exposure field name."""
    threshold = float(config.relative_ttc_threshold_s)
    metric = TTC_METRIC_BY_THRESHOLD.get(threshold)
    if metric is None:
        raise RelativeSafetyFieldMissingError(
            f"relative_ttc_threshold_s {threshold!r} is not one of "
            f"{RELATIVE_TTC_THRESHOLDS}: no TTC exposure field exists for it"
        )
    return metric


def relative_metrics(config: PlannerConfig) -> Tuple[str, ...]:
    """The relative rule's metrics under ``config``: the selected TTC
    exposure field plus the unchanged hard-brake metric."""
    return (relative_ttc_metric(config), HARD_BRAKE_METRIC)


def _require_relative_fields(prediction: Prediction, metrics: Tuple[str, ...]) -> None:
    """Fail closed: every relative metric needs its plan and native window
    fields present (not None); on a v2 prediction the future twins too."""
    is_v2 = _is_v2(prediction)
    needed: "list[str]" = []
    for metric in metrics:
        needed.extend((metric, "native_" + metric))
        if is_v2:
            needed.extend(("future_" + metric, "native_future_" + metric))
    missing = [name for name in needed if getattr(prediction, name, None) is None]
    if missing:
        raise RelativeSafetyFieldMissingError(
            "relative safety fails closed: the prediction does not report "
            f"the selected relative field(s) {sorted(missing)} "
            f"(relative_ttc_threshold_s picks the TTC field; a predictor "
            "without it cannot be decided on)"
        )

#: Delay fields where non-finite values (NaN, +-inf) are never a
#: legitimate signal (§8.7/§9; PL01-E3).
_DELAY_FIELDS: Tuple[str, ...] = (
    "plan_delay_s",
    "native_delay_s",
    "delay_vs_native_s",
    "delay_part_a_s",
    "delay_part_b_s",
)

CONTRACT_VERSION_V1_1 = "wd01_predict/1.1"
CONTRACT_VERSION_V2 = "wd01_predict/2"

#: v2 REQUIRED fields (fail closed when missing on a v2 prediction; post-
#: review contract sha db475371...0775: future_has_closing_pair is REQUIRED,
#: not optional).
_V2_REQUIRED_FIELDS: Tuple[str, ...] = (
    "future_ttc_lt3_exposure_s",
    "native_future_ttc_lt3_exposure_s",
    "future_hard_brake_s",
    "native_future_hard_brake_s",
    "p_future_collision",
    "native_p_future_collision",
    "future_min_ttc_q05_s",
    "native_future_min_ttc_q05_s",
    "future_cell_valid",
    "native_future_cell_valid",
    "future_has_closing_pair",
    "native_future_has_closing_pair",
)


class FutureSafetyRefusal(RuntimeError):
    """G-1/G-2 mode gate: the planner refuses to decide on a prediction that
    cannot support the future-safety rule (G-1) or carries unvalidated
    relative-safety verdicts (G-2) in the configured mode (v1/v1.1
    predictions, or a v2 prediction whose future_min_ttc_q05 head /
    relative-safety block is not validated, in 'live'/'closed_loop' mode;
    unvalidated heads in 'controlled_experiment' without the explicit
    override flag). PL-09: a configured mode that is not one of the four
    valid ones -- i.e. anything but 'offline' as the non-gated mode -- is a
    refusal too, never an implicit pass."""


def _gate_validated(meta: dict, key: str) -> bool:
    """PL-09 fail-closed read of a G-1/G-2 validation flag (contract v2
    declares the meta field ``bool``): ONLY the literal ``True`` counts as
    validated. A missing field, None, 0/1, or a string ('True'/'False') is
    an illegible validation claim and is treated as NOT validated."""
    return meta.get(key) is True


def _is_v2(prediction: Prediction) -> bool:
    return str((prediction.meta or {}).get("contract_version", "")) == CONTRACT_VERSION_V2


@dataclass(frozen=True)
class RelativeMetricVerdict:
    """One relative metric: mean increase and within-tolerance cell fraction."""

    metric: str
    mean_increase_s: float
    fraction_within_tol: float
    ok: bool


@dataclass(frozen=True)
class SafetyVerdict:
    """Combined safety verdict for one Prediction.

    The v2 fields (``future_ok`` / ``future_relative`` /
    ``p_future_collision_veto`` / ``g1_future_floor_veto``) are additive and
    all-True/False-safe on v1/v1.1 predictions, so v1/v1.1 verdicts are
    unchanged."""

    ok: bool  # abs_ok and rel_ok and future_ok and not window_relative_veto
    abs_ok: bool
    rel_ok: bool
    abs_safe_cells: int
    abs_total_cells: int
    abs_safe_fraction: float
    relative: Tuple[RelativeMetricVerdict, ...]
    vetoed_cells: int = 0  # cells that failed the §9 fail-closed vetoes
    # -- PL-06: cells with a truthy native_collision flag. With
    # native_collision_vetoes_plan=True these are among the vetoed cells;
    # with False (the 2026-09-27 owner default, option A) they are a logged
    # diagnostic only.
    native_collision_cells: int = 0
    # -- contract v2 future block (defaults keep v1/v1.1 verdicts identical) --
    future_ok: bool = True
    future_relative: Tuple[RelativeMetricVerdict, ...] = ()
    p_future_collision_veto: bool = False
    g1_future_floor_veto: bool = False
    # -- PL-07 (owner decision 2026-09-27): True when the 30 s window
    # difference ALONE fails the mean/quantile relative test on a v2
    # prediction (window_relative_nonincreasing=True) even if the summed
    # window+future rule passes; named 'window-only relative'. Always False
    # on v1/v1.1 (their window rule is the only relative test) and on v2
    # with the flag False. The failing window metrics are the non-ok entries
    # of ``relative``.
    window_relative_veto: bool = False
    # -- T-A: the 3 s SHADOW relative verdicts, present when the config
    # selects the 1.5 s metric (report-only: they never decide ``ok``).
    # Entries follow the same mean/quantile rule over the FROZEN
    # ttc_lt3_exposure_s fields (window, and window+future combined on v2),
    # so a run can report what the pre-T-A 3 s rule would have said. Empty
    # when the threshold already is 3.0 (the primary verdict IS the 3 s
    # one) or when the shadow fields are absent (report-only: never a
    # refusal).
    shadow_relative_3s: Tuple[RelativeMetricVerdict, ...] = ()
    # -- Rule C (owner, PL-EXP1c): which absolute-TTC rule decided
    # (``config.absolute_ttc_rule``), the smallest per-cell native floor
    # ``min(ttc_floor_s, native min TTC)`` used in the window block
    # (None under "absolute"), and the report-only RULE A SHADOW: what the
    # old absolute rule (plan min TTC >= ttc_floor_s everywhere, window and
    # G-1 future floor) would have said for the whole verdict. They never
    # decide ``ok``.
    absolute_ttc_rule: str = RULE_TTC_ABSOLUTE
    native_floor_min_s: Optional[float] = None
    shadow_rule_a_ok: Optional[bool] = None
    shadow_rule_a_abs_ok: Optional[bool] = None
    shadow_rule_a_future_ok: Optional[bool] = None


def _optional_bool(value) -> Optional[np.ndarray]:
    """bool[M, K] view of an optional v1.1 diagnostic (None = absent)."""
    if value is None:
        return None
    return np.asarray(value).astype(bool)


def _collision_veto(value) -> np.ndarray:
    """collision/native_collision vetoes: True, NaN-carrying, or non-bool."""
    arr = np.asarray(value)
    if arr.dtype == np.bool_:
        return arr.astype(bool)
    if arr.dtype.kind == "f":
        # NaN in a collision flag is an invalid signal (fail closed), as is
        # any nonzero entry.
        return np.isnan(arr) | (arr != 0)
    return np.ones(arr.shape, dtype=bool)


def fail_closed_mask(
    prediction: Prediction, config: Optional[PlannerConfig] = None
) -> np.ndarray:
    """§9 FAIL CLOSED vetoes per cell (bool[M, K]; True = the cell is unsafe).

    Evaluated BEFORE any ``min_ttc_s`` read. A v1.1 prediction that omits
    ``cell_valid``/``native_cell_valid`` vetoes EVERY cell (never defaulted
    True); a v1.0 prediction (diagnostics absent) has no invalid-cell
    concept, so only the remaining conditions apply. This mask matches the
    kit's reference ``predict.safety.safety_veto`` on predictions that carry
    the 1.1 diagnostics.

    PL-06: ``config.native_collision_vetoes_plan`` False removes ONLY the
    native-collision term from the mask -- the plan's own gates (no plan
    collision, the TTC floor, the v2 future rules) and the relative rule vs
    native still decide, and every other native fail-closed term here
    (native validity heads, NaN native delays, native NaN TTC with a
    closing pair) stays. Called WITHOUT a config the native collision
    vetoes as before (the historical/kit-reference behaviour).

    Contract v2 extension: the future validity heads are REQUIRED --
    a v2 prediction missing any ``_V2_REQUIRED_FIELDS`` entry vetoes EVERY
    cell; ``future_cell_valid``/``native_future_cell_valid`` False vetoes
    the cell (v2 §5 rule 1); and NaN ``future_min_ttc_q05_s`` (or its
    native twin) fails in a valid future cell where
    ``future_has_closing_pair`` is True (NaN q05 signals "no closing pair",
    never "unknown safety")."""
    shape = np.shape(np.asarray(prediction.collision))
    veto = np.zeros(shape, dtype=bool)
    version = str((prediction.meta or {}).get("contract_version", ""))
    cell_valid = _optional_bool(prediction.cell_valid)
    native_cell_valid = _optional_bool(prediction.native_cell_valid)
    if version == CONTRACT_VERSION_V1_1 and (cell_valid is None or native_cell_valid is None):
        # a 1.1 prediction without the diagnostics is invalid wholesale
        return np.ones(shape, dtype=bool)
    if cell_valid is not None:
        veto |= ~cell_valid
    if native_cell_valid is not None:
        veto |= ~native_cell_valid
    for name in _DELAY_FIELDS:
        # PL01-E3: +-inf delay vetoes too -- only finite delays are legible
        veto |= ~np.isfinite(np.asarray(getattr(prediction, name), dtype=float))
    veto |= _collision_veto(prediction.collision)
    if config is None or config.native_collision_vetoes_plan:
        veto |= _collision_veto(prediction.native_collision)
    for wall in prediction.walls.values():
        veto |= np.asarray(wall.safety_abort, dtype=bool)
    if version == CONTRACT_VERSION_V2:
        for name in _V2_REQUIRED_FIELDS:
            if getattr(prediction, name, None) is None:
                # a v2 prediction missing a REQUIRED field is invalid
                # wholesale (never defaulted)
                return np.ones(shape, dtype=bool)
        future_valid = _optional_bool(prediction.future_cell_valid)
        native_future_valid = _optional_bool(prediction.native_future_cell_valid)
        veto |= ~future_valid
        veto |= ~native_future_valid
        pair = _optional_bool(prediction.future_has_closing_pair)
        native_pair = _optional_bool(prediction.native_future_has_closing_pair)
        q05 = np.asarray(prediction.future_min_ttc_q05_s, dtype=float)
        veto |= np.isnan(q05) & pair
        native_q05 = np.asarray(prediction.native_future_min_ttc_q05_s, dtype=float)
        veto |= np.isnan(native_q05) & native_pair
    return veto


def _floor_cells(
    plan_shape, native_ttc, config: PlannerConfig, rule: str
) -> np.ndarray:
    """Per-cell TTC floor. "absolute": ``ttc_floor_s`` everywhere. Rule C
    ("relative_to_native"): ``min(ttc_floor_s, native TTC)`` where the native
    value is finite, else ``ttc_floor_s`` (NaN native = no closing pair)."""
    floor = np.full(plan_shape, float(config.ttc_floor_s), dtype=float)
    if rule != RULE_TTC_RELATIVE_TO_NATIVE or native_ttc is None:
        return floor
    native = np.asarray(native_ttc, dtype=float)
    return np.where(np.isfinite(native), np.minimum(floor, native), floor)


def _tol(config: PlannerConfig, rule: str) -> float:
    return float(config.ttc_rule_tol_s) if rule == RULE_TTC_RELATIVE_TO_NATIVE else 0.0


def native_floor_cells(prediction: Prediction, config: PlannerConfig) -> np.ndarray:
    """The window floor per cell under ``config.absolute_ttc_rule``."""
    shape = np.shape(np.asarray(prediction.min_ttc_s))
    return _floor_cells(
        shape, prediction.native_min_ttc_s, config, config.absolute_ttc_rule
    )


def absolute_safety_mask(
    prediction: Prediction, config: PlannerConfig, rule: Optional[str] = None
) -> np.ndarray:
    """Per-cell bool[M, K]: no §9 veto and min TTC >= the floor.

    TTC evaluation (clarification 7 as amended by §9): NaN ``min_ttc_s``
    passes ONLY in a cell that is valid AND reports ``has_closing_pair ==
    False``; NaN with a closing pair, NaN on a v1.0 prediction, and any
    other non-finite value are hard vetoes. The native ``min_ttc_s`` is
    held to the symmetric rule (NaN native TTC vetoes unless the native
    cell is valid with ``native_has_closing_pair == False``). PL-06: the
    config's ``native_collision_vetoes_plan`` flag is honoured (False keeps
    the native collision out of the §9 mask). RULE C: the floor is per cell,
    ``min(ttc_floor_s, native min TTC)`` under ``absolute_ttc_rule`` =
    "relative_to_native" (NaN native -> ``ttc_floor_s``); ``rule`` overrides
    the config (used for the rule-A shadow). Every fail-closed rule is
    identical under both rules."""
    rule = config.absolute_ttc_rule if rule is None else rule
    safe = ~fail_closed_mask(prediction, config)
    if not bool(safe.any()):
        return safe

    min_ttc = np.asarray(prediction.min_ttc_s, dtype=float)
    nan = np.isnan(min_ttc)
    cell_valid = _optional_bool(prediction.cell_valid)
    pair = _optional_bool(prediction.has_closing_pair)
    nan_ok = np.zeros(nan.shape, dtype=bool)
    if cell_valid is not None and pair is not None:
        nan_ok = nan & cell_valid & ~pair
    finite = np.isfinite(min_ttc)
    filled = np.where(finite, min_ttc, -np.inf)
    floor = _floor_cells(min_ttc.shape, prediction.native_min_ttc_s, config, rule)
    ttc_ok = (finite & (filled >= floor - _tol(config, rule))) | nan_ok

    native_ttc = np.asarray(prediction.native_min_ttc_s, dtype=float)
    native_nan = np.isnan(native_ttc)
    native_cell_valid = _optional_bool(prediction.native_cell_valid)
    native_pair = _optional_bool(prediction.native_has_closing_pair)
    native_nan_ok = np.zeros(native_nan.shape, dtype=bool)
    if native_cell_valid is not None and native_pair is not None:
        native_nan_ok = native_nan & native_cell_valid & ~native_pair
    safe &= ~(native_nan & ~native_nan_ok)
    return safe & ttc_ok


def native_collision_note(prediction: Prediction) -> str:
    """PL-06 'native_collision' diagnostic note for the decision record.

    Empty when the native reference reports no collision anywhere, else
    ``"native_collision cells=N"`` with N the number of truthy
    ``native_collision`` cells. Logged into the decision reasons when
    PlannerConfig.native_collision_vetoes_plan is False (option A: the
    native collision is a recorded diagnostic, not a veto -- the 2026-09-27
    owner default)."""
    cells = int(_collision_veto(prediction.native_collision).sum())
    return f"native_collision cells={cells}" if cells else ""


def window_relative_note(verdict: SafetyVerdict) -> str:
    """PL-07 'window-only relative' why-vetoed note for the decision record.

    Empty unless the verdict carries the PL-07 window-only veto; else
    ``"window-only relative: <metric>, ..."`` naming the window metrics
    whose plan-native difference failed the mean/quantile test."""
    if not verdict.window_relative_veto:
        return ""
    failed = ", ".join(v.metric for v in verdict.relative if not v.ok)
    return "window-only relative: " + (failed or "window rule failed")


def future_collision_veto(prediction: Prediction, config: PlannerConfig) -> bool:
    """v2 §5 rule 3: True when ``p_future_collision`` exceeds the absolute
    max in ANY cell (NaN p fails closed), or exceeds the paired native's
    probability by more than ``p_future_collision_rel_tol``."""
    p = np.asarray(prediction.p_future_collision, dtype=float)
    if not bool(np.all(np.isfinite(p))):
        return True  # NaN/inf probability is not a legible signal
    if bool(np.any(p > config.p_future_collision_abs_max)):
        return True
    native_p = np.asarray(prediction.native_p_future_collision, dtype=float)
    return bool(np.any(p > native_p + config.p_future_collision_rel_tol))


def future_relative_verdicts(
    prediction: Prediction, config: PlannerConfig
) -> Tuple[RelativeMetricVerdict, ...]:
    """v2 §5 rule 4: the relative rule over the window difference PLUS the
    future difference, per metric (mean increase <= tol AND the >=-quantile
    of cells within tol). NaN anywhere poisons the cell delta and hence the
    mean (NaN mean compares False -> veto), matching the window rule. The
    metrics follow the config (T-A): the selected TTC exposure field plus
    hard braking."""
    out = []
    for metric in relative_metrics(config):
        plan_win = np.asarray(getattr(prediction, metric), dtype=float)
        native_win = np.asarray(getattr(prediction, "native_" + metric), dtype=float)
        plan_future = np.asarray(getattr(prediction, "future_" + metric), dtype=float)
        native_future = np.asarray(getattr(prediction, "native_future_" + metric), dtype=float)
        delta = (plan_win - native_win) + (plan_future - native_future)
        mean_increase = float(np.mean(delta))
        within = float(np.mean(delta <= config.rel_safety_tol_s))
        metric_ok = (
            np.isfinite(mean_increase)
            and np.isfinite(within)
            and mean_increase <= config.rel_safety_tol_s
            and within >= config.rel_safety_quantile
        )
        out.append(
            RelativeMetricVerdict(metric + "+future", mean_increase, within, bool(metric_ok))
        )
    return tuple(out)


def g1_future_floor_veto(
    prediction: Prediction, config: PlannerConfig, rule: Optional[str] = None
) -> bool:
    """v2 §5 rule 5 (G-1): the 1.5 s min-TTC floor applies to
    ``future_min_ttc_q05_s`` ONLY when ``meta.future_min_ttc_validated`` is
    True; validated, a finite q05 below the floor in any cell vetoes (a NaN
    q05 without a closing pair passes -- it is already the NaN contract).
    RULE C: the floor is ``min(ttc_floor_s, native twin q05)`` (see
    :func:`_floor_cells`); ``rule`` overrides the config (rule-A shadow)."""
    if not bool((prediction.meta or {}).get("future_min_ttc_validated", False)):
        return False
    rule = config.absolute_ttc_rule if rule is None else rule
    q05 = np.asarray(prediction.future_min_ttc_q05_s, dtype=float)
    finite = np.isfinite(q05)
    filled = np.where(finite, q05, -np.inf)
    # RULE C: the floor is min(ttc_floor_s, native twin q05) where a native
    # twin exists and is finite; with no twin (None) or NaN native the
    # absolute floor stays.
    floor = _floor_cells(
        q05.shape, getattr(prediction, "native_future_min_ttc_q05_s", None), config, rule
    )
    return bool(np.any(finite & (filled < floor - _tol(config, rule))))


def enforce_future_gate(prediction: Prediction, config: PlannerConfig) -> Optional[str]:
    """G-1/G-2 mode gate (contract v2 §5 rule 5 + §8; PL01_PLAN gating
    items G-1 and G-2).

    Returns an override note to log in the decision record, or None. Raises
    :class:`FutureSafetyRefusal` -- BEFORE any decision is taken -- when the
    configured mode may not consume this prediction:

    - ``live`` / ``closed_loop``: only a v2 prediction is accepted whose
      ``future_min_ttc_validated`` is True with a non-None
      ``future_min_ttc_validation_id`` (G-1) AND whose
      ``relative_safety_validated`` is True with a non-None
      ``relative_safety_validation_id`` (G-2): BOTH gates must pass. A
      v1/v1.1 prediction (no future safety at all), an unvalidated v2 future
      head and an unvalidated v2 relative-safety block are each refused; a
      missing G-2 meta field is treated as False (fail closed).
    - ``controlled_experiment``: v1/v1.1 predictions are allowed as-is; an
      unvalidated v2 future head is allowed ONLY with
      ``config.allow_unvalidated_future_ttc`` and unvalidated
      relative-safety verdicts ONLY with
      ``config.allow_unvalidated_relative_safety`` (the returned note
      records every applied override).
    - ``offline``: the ONLY non-gated mode (PL-09); never refuses, loop.py
      logs the G-2 :func:`relative_safety_diagnostic` note instead.
    - any other mode: :class:`FutureSafetyRefusal` (PL-09 -- an
      unknown/typo mode must never pass the gate OPEN; normally
      unreachable because PlannerConfig validates ``mode`` at
      construction, but enforced here in case a config object is doctored
      past that validation).

    PL-09: "validated" above means the meta field is the LITERAL ``True``
    (:func:`_gate_validated`) -- ``'True'``, ``1`` etc. are treated as NOT
    validated (fail closed), not merely truthy.
    """
    mode = config.mode
    meta = prediction.meta or {}
    version = str(meta.get("contract_version", ""))
    if mode == "offline":
        return None
    if mode not in ("live", "closed_loop", "controlled_experiment"):
        raise FutureSafetyRefusal(
            f"unknown mode {mode!r}: the G-1/G-2 mode gate passes ONLY "
            "'offline' and gates 'live', 'closed_loop' and "
            "'controlled_experiment' -- refusing rather than passing OPEN "
            "(PL-09)"
        )
    if mode in ("live", "closed_loop"):
        if version != CONTRACT_VERSION_V2:
            raise FutureSafetyRefusal(
                f"{mode} mode refuses contract {version or 'unversioned'} "
                "predictions: no future-safety information beyond the commit "
                "window (G-1)"
            )
        if not _gate_validated(meta, "future_min_ttc_validated"):
            raise FutureSafetyRefusal(
                f"{mode} mode refuses a v2 prediction whose "
                "future_min_ttc_q05_s head is not validated (G-1); run a "
                "controlled experiment or validate the head first"
            )
        if meta.get("future_min_ttc_validation_id") is None:
            raise FutureSafetyRefusal(
                "future_min_ttc_validated=True requires a non-None future_min_ttc_validation_id"
            )
        # G-2 (contract v2 §8): the paired relative-safety verdicts
        # themselves (TTC<3 exposure, hard braking) must be validated too.
        if not _gate_validated(meta, "relative_safety_validated"):
            raise FutureSafetyRefusal(
                f"{mode} mode refuses a v2 prediction whose relative-safety "
                "verdicts (TTC<3 exposure, hard braking) are not validated "
                "(G-2); run a controlled experiment or validate the paired "
                "deltas first"
            )
        if meta.get("relative_safety_validation_id") is None:
            raise FutureSafetyRefusal(
                "relative_safety_validated=True requires a non-None "
                "relative_safety_validation_id (G-2)"
            )
        return None
    # controlled_experiment
    notes = []
    if version == CONTRACT_VERSION_V2:
        if not _gate_validated(meta, "future_min_ttc_validated"):
            if not config.allow_unvalidated_future_ttc:
                raise FutureSafetyRefusal(
                    "controlled_experiment mode consumes an unvalidated v2 "
                    "future_min_ttc_q05 head only with "
                    "allow_unvalidated_future_ttc=True (the override is logged "
                    "in the decision record)"
                )
            notes.append(
                "controlled_experiment override: consuming an unvalidated "
                "future_min_ttc_q05 head (allow_unvalidated_future_ttc=True; "
                "the 1.5 s floor does NOT apply beyond the commit window)"
            )
        if not _gate_validated(meta, "relative_safety_validated"):
            if not config.allow_unvalidated_relative_safety:
                raise FutureSafetyRefusal(
                    "controlled_experiment mode consumes unvalidated v2 "
                    "relative-safety verdicts (TTC<3 exposure, hard braking) "
                    "only with allow_unvalidated_relative_safety=True (the "
                    "override is logged in the decision record)"
                )
            notes.append(
                "controlled_experiment override: consuming UNVALIDATED "
                "relative-safety verdicts, TTC<3 exposure and hard braking "
                "(allow_unvalidated_relative_safety=True; G-2)"
            )
    if version != CONTRACT_VERSION_V2:
        notes.append(
            "controlled_experiment override: consuming a contract "
            f"{version or 'unversioned'} prediction with no future-safety "
            "information beyond the commit window"
        )
    return "; ".join(notes) if notes else None


def relative_safety_diagnostic(prediction: Prediction) -> str:
    """G-2 'relative_safety_validated=False' note for the decision record.

    Empty unless the prediction is contract v2 AND its meta reports
    ``relative_safety_validated`` False (or omits the field -- fail closed):
    the relative-safety verdicts (TTC<3 exposure, hard braking) of that
    prediction are UNVALIDATED. offline mode never refuses (G-2), so
    loop.py logs this note into every decision record of the tick instead;
    controlled_experiment logs its override note and live/closed_loop
    refuse before any decision is taken."""
    if not _is_v2(prediction):
        return ""
    if bool((prediction.meta or {}).get("relative_safety_validated", False)):
        return ""
    return (
        "relative_safety_validated=False (G-2): relative-safety verdicts "
        "(TTC<3 exposure, hard braking) are UNVALIDATED; offline diagnostic, "
        "not a veto"
    )


def safety_check(prediction: Prediction, config: PlannerConfig) -> SafetyVerdict:
    """Absolute + relative safety of ``prediction`` vs its shared native.

    Contract v2 (wd01_predict/2) extends the verdict with the future block:
    the p_future_collision absolute + paired-relative vetoes, the
    window+future relative rule, and the G-1 conditional future floor. The
    v2 blocks are all-True on v1/v1.1 predictions (no future heads), so
    v1/v1.1 verdicts are byte-identical to the pre-v2 behaviour. PL-06:
    ``config.native_collision_vetoes_plan`` False (the 2026-09-27 owner
    default) keeps a native collision out of the veto count (it no longer
    vetoes) and reports it in ``native_collision_cells`` instead. PL-07:
    on v2, ``config.window_relative_nonincreasing`` True (the owner
    default) additionally requires the 30 s WINDOW difference alone to
    pass the mean/quantile relative test; a window-only failure vetoes and
    is reported in ``window_relative_veto`` (named 'window-only
    relative'). T-A: the relative TTC metric follows
    ``config.relative_ttc_threshold_s`` (1.5 -> the primary
    ``ttc_lt1p5_exposure_s`` fields, 3.0 -> the frozen ``ttc_lt3_
    exposure_s`` fields); a missing selected field raises the named
    :class:`RelativeSafetyFieldMissingError` (fail closed, before any
    decision). With the 1.5 s selection the FROZEN 3 s fields additionally
    get a report-only SHADOW verdict (:attr:`SafetyVerdict.shadow_relative_
    3s`): what the pre-T-A 3 s rule would have said, never a veto."""
    metrics = relative_metrics(config)
    # T-A fail-closed FIRST: the selected TTC field (and its native/future
    # twins) must be reported, or the named error refuses the decision.
    _require_relative_fields(prediction, metrics)
    is_v2 = _is_v2(prediction)
    mask = absolute_safety_mask(prediction, config)
    veto = fail_closed_mask(prediction, config)
    native_collision_cells = int(_collision_veto(prediction.native_collision).sum())
    total = int(mask.size)
    safe = int(mask.sum())
    fraction = safe / total if total else 1.0
    abs_ok = fraction >= config.abs_safety_quantile

    def _relative_verdicts(selected: Tuple[str, ...]) -> "list[RelativeMetricVerdict]":
        out = []
        for metric in selected:
            plan_v = np.asarray(getattr(prediction, metric), dtype=float)
            native_v = np.asarray(getattr(prediction, "native_" + metric), dtype=float)
            delta = plan_v - native_v
            mean_increase = float(np.mean(delta))
            within = float(np.mean(delta <= config.rel_safety_tol_s))
            # NaN deltas poison the mean; a NaN mean compares False -> veto.
            metric_ok = (
                np.isfinite(mean_increase)
                and np.isfinite(within)
                and mean_increase <= config.rel_safety_tol_s
                and within >= config.rel_safety_quantile
            )
            out.append(RelativeMetricVerdict(metric, mean_increase, within, bool(metric_ok)))
        return out

    relative = _relative_verdicts(metrics)
    # v1/v1.1: the window rule IS the relative decision; v2 replaces it
    # with the combined window+future rule below.
    rel_ok = bool(all(v.ok for v in relative))

    # contract v2 future block (all-True on v1/v1.1).  For v2 the RELATIVE
    # rule is the COMBINED window+future difference (contract v2 §5 rule 4):
    # it REPLACES the window-only relative test as the ``rel_ok`` decision,
    # whose verdict stays in ``relative``.  PL-07 (owner decision
    # 2026-09-27) re-arms that window test as an ADDITIVE veto: with
    # window_relative_nonincreasing=True (the default) the window
    # difference alone must also pass the same mean/quantile test, else the
    # plan is vetoed with the 'window-only relative' note.  ``future_ok``
    # covers the p_future_collision vetoes and the G-1 conditional future
    # floor.
    p_veto = future_collision_veto(prediction, config) if is_v2 else False
    future_relative = future_relative_verdicts(prediction, config) if is_v2 else ()
    g1_veto = g1_future_floor_veto(prediction, config) if is_v2 else False
    # RULE A SHADOW (report-only): the same verdict under the old absolute
    # floor. Identical to the primary when the config already selects it.
    a_mask = absolute_safety_mask(prediction, config, RULE_TTC_ABSOLUTE)
    a_abs_ok = bool(
        (int(a_mask.sum()) / int(a_mask.size) if a_mask.size else 1.0)
        >= config.abs_safety_quantile
    )
    a_g1 = g1_future_floor_veto(prediction, config, RULE_TTC_ABSOLUTE) if is_v2 else False
    a_future_ok = bool(not p_veto and not a_g1)
    if config.absolute_ttc_rule == RULE_TTC_RELATIVE_TO_NATIVE:
        floors = native_floor_cells(prediction, config)
        native_floor_min = float(np.min(floors)) if floors.size else None
    else:
        native_floor_min = None
    if is_v2:
        # the COMBINED window+future rule REPLACES the window-only
        # relative test as ``rel_ok`` (contract v2 §5 rule 4)
        rel_ok = bool(all(v.ok for v in future_relative))
    # PL-07: the window-only verdicts are exactly the ``relative`` entries
    # above (plan - native over the window); a failure vetoes additively.
    # v1/v1.1 are unaffected -- their window rule is the only relative
    # test and already decides ``rel_ok``.
    window_veto = bool(
        is_v2 and config.window_relative_nonincreasing and not all(v.ok for v in relative)
    )
    future_ok = bool(not p_veto and not g1_veto)

    # T-A: the 3 s SHADOW verdict (report-only) when the primary metric is
    # the 1.5 s exposure. Best effort: absent shadow fields leave it empty
    # (the frozen v2 REQUIRED-field gate already vetoes a v2 prediction
    # without them; a v1 prediction simply has no future block).
    shadow: Tuple[RelativeMetricVerdict, ...] = ()
    if relative_ttc_metric(config) == RELATIVE_TTC_METRIC_1P5:
        window_fields = ("ttc_lt3_exposure_s", "native_ttc_lt3_exposure_s")
        if all(getattr(prediction, name, None) is not None for name in window_fields):
            shadow = tuple(
                _relative_verdicts((RELATIVE_TTC_METRIC_3,))
            ) + (
                tuple(_combined_3s_shadow(prediction, config)) if is_v2 else ()
            )

    return SafetyVerdict(
        ok=bool(abs_ok and rel_ok and future_ok and not window_veto),
        abs_ok=bool(abs_ok),
        rel_ok=bool(rel_ok),
        abs_safe_cells=safe,
        abs_total_cells=total,
        abs_safe_fraction=fraction,
        relative=tuple(relative),
        vetoed_cells=int(veto.sum()),
        native_collision_cells=native_collision_cells,
        future_ok=bool(future_ok),
        future_relative=future_relative,
        p_future_collision_veto=bool(p_veto),
        g1_future_floor_veto=bool(g1_veto),
        window_relative_veto=window_veto,
        shadow_relative_3s=shadow,
        absolute_ttc_rule=config.absolute_ttc_rule,
        native_floor_min_s=native_floor_min,
        shadow_rule_a_ok=bool(a_abs_ok and rel_ok and a_future_ok and not window_veto),
        shadow_rule_a_abs_ok=a_abs_ok,
        shadow_rule_a_future_ok=a_future_ok,
    )


def _combined_3s_shadow(prediction: Prediction, config: PlannerConfig):
    """The v2 combined window+future verdict over the FROZEN 3 s fields for
    the report-only shadow (the same mean/quantile rule as
    :func:`future_relative_verdicts`, TTC metric only)."""
    metric = RELATIVE_TTC_METRIC_3
    needed = (
        metric,
        "native_" + metric,
        "future_" + metric,
        "native_future_" + metric,
    )
    if any(getattr(prediction, name, None) is None for name in needed):
        return ()
    plan_win = np.asarray(getattr(prediction, metric), dtype=float)
    native_win = np.asarray(getattr(prediction, "native_" + metric), dtype=float)
    plan_future = np.asarray(getattr(prediction, "future_" + metric), dtype=float)
    native_future = np.asarray(getattr(prediction, "native_future_" + metric), dtype=float)
    delta = (plan_win - native_win) + (plan_future - native_future)
    mean_increase = float(np.mean(delta))
    within = float(np.mean(delta <= config.rel_safety_tol_s))
    metric_ok = (
        np.isfinite(mean_increase)
        and np.isfinite(within)
        and mean_increase <= config.rel_safety_tol_s
        and within >= config.rel_safety_quantile
    )
    return (
        RelativeMetricVerdict(metric + "+future", mean_increase, within, bool(metric_ok)),
    )
