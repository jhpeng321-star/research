"""WM-N1 CAL-tune scorer labels: zero-label census and per-horizon truth.

Contract (1) is solved upstream: :mod:`evalh.real_source` keeps refusing a
selected label whose decision time differs from its t0 snapshot and a shard
that repeats a tau, so this module never reinterprets those rows.

Owner rule (b) exclusions are a second, counted, metadata-only skip.  Training
removed outcome supervision for the ``base`` / ``native_base`` arms
(``train.dataset.OUTCOME_SUPERVISION_EXCLUDED_ARMS``,
``train/dataset.py:124``): their feature snapshot sits near the run start and
does not line up with the packed ``t0`` label rows.  The scorer applies the same
rule to CAL-tune items BEFORE any payload is opened, so a shard whose label
decision time differs from ``t0`` can never abort the pass.  Excluded items are
counted in :class:`LabelCensus` (``n_excluded_b`` / ``excluded_b_keys``) and
never appear in :func:`planned_reads`.

Zero-label shards are an intentional, counted skip.  A shard is zero-label
when the pack INDEX says ``n_outcome_rows == 0`` (equivalently: no outcomes
file is pinned) AND the SHA-pinned ``DONE.json`` receipt agrees that no
``outcomes.parquet`` was written.  This is decided from metadata alone, BEFORE
any ``outcomes.parquet`` or ``.npy`` payload is opened.  A shard whose INDEX
records rows but whose file is missing still FAILS CLOSED, never skipped.

Per-horizon survivor semantics (declared here, not inferred by a consumer):

* a vehicle on the t0 roster survives to horizon ``h`` iff, at every step from
  t0 up to and including ``h``, its target mask is live, its presence mask is
  live and its slot id equals the t0 slot id; the survivor set is monotone in
  ``h`` and a reused slot is never mistaken for the original car;
* dynamics metrics at horizon ``h`` use exactly the survivors at ``h``
  (``mask[h]``); ``positions``/``speeds`` are NaN off a survivor column;
* outcome metrics use label rows with ``tau_available > 0`` at that tau, for
  the registered taus ``(30, 60, 120, 300, 570) s``;
* short horizons are NOT conditioned on long-horizon survival;
* a horizon whose step index lies beyond the run's last tick is masked out and
  counted (``horizons_masked`` / ``survivor_counts``), never clamped into a
  following run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..cal_split import CAL_TUNE
from ..real_source import RealSourceRefusal, _DT_S, _item_key

#: Registered outcome taus (seconds) of the scorer's metric loop.
TAUS_S = (30.0, 60.0, 120.0, 300.0, 570.0)

#: The single module constant naming every outcome column a scoring pass reads.
OUTCOME_COLUMNS = (
    "world", "source", "split", "t0_s", "t", "tau", "tau_available",
    "total_delay_ref_s", "collision_any", "ttc_lt3_veh_s", "hard_brake_s",
    "n_vehicles_present",
)

#: Arrays the rollout pass needs from each selected shard.  ``hdv_base`` and
#: ``event_tok`` / ``event_mask`` are deliberately excluded: the WM-N1 rollout
#: derives its HDV residual from the loaded member bundle and the P6 event
#: tokens are off (``event_tokens=False``), so the packed channels are unused.
#: ``cav_plan_ch`` feeds BOTH the controller action schedule and the
#: teacher-mode training-parity correction ``summary_raw``
#: (``evalh.scorer.predictors.training_summary_raw``, contract 3).
ROLLOUT_ARRAYS = ("veh_kin", "veh_mask", "slot_index", "target_mask",
                  "cav_plan_ch")

#: E7 extras: the v1 arrays the real per-item predictors read (CAV roster,
#: A6 entrance counter, packed per-CAV action schedule).  Listed by
#: :func:`planned_reads` only when the release pins them, so a fixture that
#: predates E7 keeps its exact read census.
_EXTRA_INPUT_ARRAYS = ("veh_step", "counter", "plan")

class ScorerLabelRefusal(RealSourceRefusal):
    """A label/truth request that the CAL-tune scorer must fail closed on."""


@dataclass
class LabelCensus:
    """How many selected shards yielded labels, and which were skipped.

    ``n_excluded_b`` / ``excluded_b_keys`` count the owner rule-(b)
    outcome-supervision exclusions (``base`` / ``native_base``), skipped on
    metadata before any payload open.
    """

    n_items: int
    n_zero_label_skipped: int
    skipped_keys: list
    n_rows_by_tau: dict
    n_excluded_b: int = 0
    excluded_b_keys: list = field(default_factory=list)


@dataclass
class HorizonTruth:
    """Per-horizon true states over the t0 roster.

    ``positions``/``speeds``/``mask`` are ``[H, K]`` over ``horizons_s`` and
    the ``K`` t0-roster columns (``t0_slot_ids`` records their slot identity).
    ``mask[h, k]`` is True iff column ``k`` survives to ``horizons_s[h]``;
    non-survivor entries of ``positions``/``speeds`` are NaN.  ``horizons_masked[h]``
    flags a horizon past the run end (all-False mask, NaN states);
    ``survivor_counts[h]`` is ``mask[h].sum()``.
    """

    horizons_s: tuple
    positions: np.ndarray
    speeds: np.ndarray
    mask: np.ndarray
    t0_slot_ids: np.ndarray
    horizon_masked: np.ndarray = None
    survivor_counts: np.ndarray = None
    run_key: str = ""


def _selected_entry(source, item):
    """Accept a ``(pos, item, run)`` entry or an item dict of this source."""
    if isinstance(item, tuple) and len(item) == 3:
        return item
    target = _item_key(item)
    for entry in getattr(source, "selected", ()) or ():
        if _item_key(entry[1]) == target:
            return entry
    raise ScorerLabelRefusal(f"item {target!r} is not a selected shard")


def _require_tune_half(source):
    """The scorer is CAL-tune only; a CAL-cal release is refused up front."""
    half = getattr(source, "half", None)
    if half != CAL_TUNE:
        raise ScorerLabelRefusal(
            f"the scorer labels only the {CAL_TUNE!r} half; this release names "
            f"{half!r}")


def outcome_supervision_excluded(item) -> bool:
    """True iff training rule (b) removed this arm's outcome supervision.

    Imports (never copies) ``train.dataset.is_outcome_supervision_excluded``,
    whose registered basis is ``OUTCOME_SUPERVISION_EXCLUDED_ARMS``
    (``train/dataset.py:124``, ``("base", "native_base")``).  Metadata only.
    """
    from train.dataset import is_outcome_supervision_excluded

    return bool(is_outcome_supervision_excluded(item))


def _split_excluded(selected):
    """``(kept, excluded_keys)`` over selected entries, metadata only."""
    kept, excluded = [], []
    for entry in selected:
        if outcome_supervision_excluded(entry[1]):
            excluded.append(_item_key(entry[1]))
        else:
            kept.append(entry)
    return tuple(kept), tuple(excluded)


def selected_items(source):
    """The CAL-tune selected items that keep outcome supervision (rule b).

    Returns ``(kept, excluded_keys)``: ``kept`` is the tuple of
    ``(position, item, run)`` entries whose arm is NOT in the registered
    exclusion set; ``excluded_keys`` names the skipped shards in selection
    order.  This is the canonical item enumeration for the predictors, truth
    and :func:`planned_reads`; it reads metadata only, so no payload is opened
    for a skipped shard.
    """
    _require_tune_half(source)
    return _split_excluded(tuple(getattr(source, "selected", ()) or ()))


def _is_zero_label(source, item, run):
    """Metadata-only zero-label decision, before any payload is opened."""
    key = _item_key(item)
    pins = getattr(source, "pins", None)
    if not isinstance(pins, dict) or key not in pins:
        raise ScorerLabelRefusal(f"release pins no selected shard {key!r}")
    files = pins[key].get("files") or {}
    pinned = "outcomes.parquet" in files
    # The SHA-pinned receipt is read here, before any parquet/array payload.
    receipt = source.receipt_for(item)
    rec_n = int(receipt.get("n_outcome_rows", 0) or 0)
    rec_has = "outcomes.parquet" in (receipt.get("outputs") or {})
    n_index = int(run.get("n_outcome_rows", 0))
    if n_index == 0:
        if rec_has or rec_n != 0:
            raise ScorerLabelRefusal(
                f"shard {key!r}: INDEX records no outcome rows but its pinned "
                "receipt disagrees; refusing the zero-label skip")
        return True
    if not pinned:
        raise ScorerLabelRefusal(
            f"shard {key!r}: INDEX records {n_index} outcome rows but the "
            "release pins no outcomes.parquet; refusing (fail closed)")
    if not rec_has or rec_n != n_index:
        raise ScorerLabelRefusal(
            f"shard {key!r}: INDEX records {n_index} outcome rows but its "
            "pinned receipt disagrees; refusing (fail closed)")
    return False


def tune_outcome_rows(source):
    """Read the released CAL-tune label rows, skipping excluded/zero shards.

    Returns ``(rows, LabelCensus)``.  Owner rule-(b) arms (``base`` /
    ``native_base``) are skipped on metadata BEFORE any payload open, counted
    in ``n_excluded_b`` / ``excluded_b_keys``, and can therefore never reach
    ``real_source.outcome_rows_for`` (whose decision != t0 check would abort).
    Zero-label shards are counted too, not opened; a shard the INDEX says has
    rows but cannot deliver them fails closed.
    """
    _require_tune_half(source)
    selected = tuple(getattr(source, "selected", ()) or ())
    kept, excluded_keys = _split_excluded(selected)
    rows, skipped, by_tau = [], [], {}
    for position, item, run in kept:
        if _is_zero_label(source, item, run):
            skipped.append(_item_key(item))
            continue
        emitted = source.outcome_rows_for(item, run, position,
                                          columns=OUTCOME_COLUMNS)
        rows.extend(emitted)
        for row in emitted:
            by_tau[row.tau_s] = by_tau.get(row.tau_s, 0) + 1
    census = LabelCensus(
        n_items=len(selected), n_zero_label_skipped=len(skipped),
        skipped_keys=skipped, n_rows_by_tau=dict(sorted(by_tau.items())),
        n_excluded_b=len(excluded_keys), excluded_b_keys=list(excluded_keys))
    return rows, census


def _horizon_steps(horizons_s):
    horizons = tuple(float(h) for h in horizons_s)
    if not horizons:
        raise ScorerLabelRefusal("rollout horizons must be non-empty")
    steps = []
    for h in horizons:
        if not math.isfinite(h) or h <= 0.0:
            raise ScorerLabelRefusal("rollout horizons must be finite and positive")
        ratio = h / _DT_S
        if not ratio.is_integer():
            raise ScorerLabelRefusal("rollout horizons must lie on the 0.5 s grid")
        steps.append(int(ratio))
    return horizons, tuple(steps)


def masked_rollout_truth(source, item, horizons_s):
    """True per-horizon states of the t0 roster for one selected shard.

    Survivors at each horizon are computed independently, so a car present at
    a short horizon but gone before a long one still counts at the short one.
    Horizons past the run end are masked out and counted, not clamped.
    """
    _require_tune_half(source)
    _position, item, run = _selected_entry(source, item)
    horizons, steps = _horizon_steps(horizons_s)

    kin = np.load(source._checked_file(item, "veh_kin.npy"), mmap_mode="r")
    mask = np.load(source._checked_file(item, "veh_mask.npy"), mmap_mode="r")
    slot = np.load(source._checked_file(item, "slot_index.npy"), mmap_mode="r")
    target = np.load(source._checked_file(item, "target_mask.npy"), mmap_mode="r")
    if kin.ndim != 3 or kin.shape[-1] != 7:
        raise ScorerLabelRefusal("selected shard veh_kin shape is not [T, K, 7]")
    n_ticks = int(run["n_ticks"])
    if (mask.shape != kin.shape[:2] or slot.shape != mask.shape
            or target.shape != mask.shape
            or any(len(a) != n_ticks for a in (kin, mask, slot, target))):
        raise ScorerLabelRefusal(
            "selected shard truth channels disagree with the run length")
    local_tick = int(item["tick"]) - int(run["tick_start"])
    if local_tick < 0 or local_tick >= n_ticks:
        raise ScorerLabelRefusal("selected t0 tick escapes its run")

    roster = np.flatnonzero(mask[local_tick].astype(bool))
    if not len(roster):
        raise ScorerLabelRefusal("selected t0 truth roster is empty")
    t0_ids = slot[local_tick, roster].astype(np.int64)
    t0_slot = slot[local_tick, roster]

    n_h, k = len(steps), len(roster)
    out_mask = np.zeros((n_h, k), dtype=bool)
    out_pos = np.full((n_h, k), np.nan, dtype=float)
    out_spd = np.full((n_h, k), np.nan, dtype=float)
    horizon_masked = np.zeros(n_h, dtype=bool)
    step_to_h = {}
    for hi, s in enumerate(steps):
        step_to_h.setdefault(s, []).append(hi)
        if local_tick + s >= n_ticks:
            horizon_masked[hi] = True

    survivor = np.ones(k, dtype=bool)
    max_step = max(steps)
    for off in range(1, max_step + 1):
        if local_tick + off >= n_ticks:
            break
        live = (target[local_tick + off - 1, roster].astype(bool)
                & mask[local_tick + off, roster].astype(bool)
                & (slot[local_tick + off, roster] == t0_slot))
        survivor = survivor & live
        for hi in step_to_h.get(off, ()):
            out_mask[hi] = survivor
            out_pos[hi] = np.where(
                survivor, kin[local_tick + off, roster, 0].astype(float), np.nan)
            out_spd[hi] = np.where(
                survivor, kin[local_tick + off, roster, 4].astype(float), np.nan)

    counts = out_mask.sum(axis=1).astype(np.int64)
    return HorizonTruth(
        horizons_s=horizons, positions=out_pos, speeds=out_spd,
        mask=out_mask, t0_slot_ids=t0_ids, horizon_masked=horizon_masked,
        survivor_counts=counts, run_key=_item_key(item))


def planned_reads(source):
    """Exactly the shard files/columns one scoring pass opens.

    Returns ``(relative_path, columns_or_array_name)`` pairs relative to the
    release's ``shards_root``.  Rule-(b) excluded shards and zero-label shards
    contribute no read at all; every remaining selected shard contributes the
    truth arrays plus ``cav_plan_ch`` (the packed schedule input).  Outcome
    columns come from the single :data:`OUTCOME_COLUMNS` constant.
    """
    _require_tune_half(source)
    kept, _excluded_keys = _split_excluded(
        tuple(getattr(source, "selected", ()) or ()))
    reads = []
    for _position, item, run in kept:
        rel = f"{item['batch']}/{item['world']}/{item['arm']}"
        if int(run.get("n_outcome_rows", 0)) > 0:
            reads.append((f"{rel}/outcomes.parquet", tuple(OUTCOME_COLUMNS)))
        for name in ROLLOUT_ARRAYS:
            reads.append((f"{rel}/{name}.npy", name))
        files = (getattr(source, "pins", None) or {}).get(
            _item_key(item), {}).get("files", {})
        for name in _EXTRA_INPUT_ARRAYS:
            if f"{name}.npy" in files:
                reads.append((f"{rel}/{name}.npy", name))
    return reads


__all__ = [
    "TAUS_S", "OUTCOME_COLUMNS", "ROLLOUT_ARRAYS", "ScorerLabelRefusal",
    "LabelCensus", "HorizonTruth", "outcome_supervision_excluded",
    "selected_items", "tune_outcome_rows", "masked_rollout_truth",
    "planned_reads",
]
