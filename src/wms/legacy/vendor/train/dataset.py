# ruff: noqa
# VENDORED from output/wd01/wm_n1/train/dataset.py (last changed in commit 108223d; copied at baseline bf7d4cb).
# Only change: absolute imports of sibling modules re-pointed to this vendor package.
# Do not edit: behaviour is frozen evidence; WP-C only mirrors it.
"""MemmapCorpus: training dataset over packed WM-N1 shards.

Reads the packed memmap produced by :mod:`corpus` (built from the per-run shards
of ``features/FORMAT.md``) and yields one t0 snapshot per item:

``veh_step [K,16] float16``, ``veh_mask [K] uint8``, ``plan [K,19,8] float16``,
``wall [4,11] float16``, ``wall_mask [4] uint8``, ``counter [2,30] float16``,
plus step targets ``accel_next [K]``, ``target_mask [K]``, ``lane_next [K]``,
``arrivals [2]`` and the packed outcome rows.

Sibling-pair sampling groups items by ``pair_key = "<world>@t0=<t0>"`` (the
``features`` INDEX.json grouping) so the contrast loss of
``model.losses.contrast_loss`` can draw the two arms of the same world/t0 in
one batch.

Packed layout (the train-side contract for :mod:`corpus`)
---------------------------------------------------------
The corpus pack is one directory holding a ``manifest.json`` plus the
concatenated memmap arrays and the archived outcome rows::

    manifest.json
    veh_step.npy  [N_ticks, K, 16] float16
    veh_mask.npy  [N_ticks, K] uint8
    plan.npy      [N_items, K, 19, 8] float16
    wall.npy      [N_ticks, 4, 11] float16
    wall_mask.npy [N_ticks, 4] uint8
    counter.npy   [N_ticks, 2, 30] float16
    accel_next.npy/target_mask.npy/lane_next.npy [N_ticks, K]
    arrivals.npy  [N_ticks, 2] int16
    outcomes.parquet

``manifest.json``::

    {"schema": "wm_n1_pack/1", "k_max": 128,
     "arrays": {name: {"file", "dtype", "shape"}},
     "outcomes": {"file": "outcomes.parquet"},
     "items": [{"world", "arm", "chunk", "split", "t0_s", "pair_key",
                "policy", "tick", "plan_row", "outcome_rows"}]}

``k_max`` is the slot width (a FORMAT parameter: 128 for the frozen v2
format, 192 for the v3 format of amendment 5; a v3 pack additionally carries
``"format": "v3"``).  A pack whose ``k_max`` disagrees with its packed slot
dims, its ``format`` claim, or an explicitly requested ``k_max`` is refused.

``items`` is ordered; its position is the item index.  ``tick`` indexes the
tick-level arrays, ``plan_row`` the plan array, and ``outcome_rows`` are the
parquet row positions aligned to ``model.constants.TAUS``.  The CORPUS piece
owns the writer; this module is the read-only consumer.  When the packed
writer is unavailable, ``train/tests/packsynth.py`` builds a byte-identical
fixture for the unit tests (interface note D3 in ``train/README.md``).

TRAIN label isolation
---------------------
A MIXED-split corpus pack (TRAIN + CALIBRATION items, e.g. ``pack_v3``: 2153
TRAIN / 310 CALIBRATION) must never have its packed ``outcomes.parquet``
opened for ``split="TRAIN"``: the table holds other splits' labels.  The
manifest is the claim of what the packed table can hold, and a PRESENT pack
INDEX must agree with it (metadata-only
:func:`train.label_isolation.audit_pack_identity`): a WHOLLY single-split
TRAIN manifest keeps the original mixed-table read only when it carries NO
INDEX (legacy/synthetic manifest-only fixtures) or when its INDEX
reconciles 1:1 with all-TRAIN run identities.  An INDEX naming a non-TRAIN
or unmatched run contradicts the training-only claim and fails closed
BEFORE any label read -- the packed table could hold excluded outcome rows.
An empty manifest is refused outright at load: it proves no training-only
table.  Any MIXED-split manifest requires identity-bearing INDEX
metadata (``shard_dir`` + ``n_outcome_rows`` per run, as the corpus writer
emits) and, for ``split="TRAIN"``, a TRAIN-only outcome sidecar: a pack
whose INDEX is missing, empty, or carries incomplete per-run identity fails
closed BEFORE any label read -- there is no synthetic/legacy mixed-pack
exception.  When the INDEX identity is valid, ``_outcomes`` refuses before
ANY label read unless the ``WM_N1_TRAIN_OUTCOMES`` environment variable names
a TRAIN-only outcome sidecar (directory) built and receipt-bound by
:mod:`train.label_isolation` from the original per-run TRAIN shards; the
loaded sidecar frame is indexed by the original global row ids and rows are
selected via ``.loc``.  The frozen ``MemmapCorpus`` signatures are unchanged.
``split="CALIBRATION"`` is an explicit authorized mixed-table read and never
consults the sidecar.  Item split/chunk spellings are conservative: only
``TRAIN``/``CALIBRATION`` splits are recognized (unknown spellings refuse
the pack) and every ``chunk5``-variant spelling is treated as the sealed
chunk.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from wms.legacy.vendor.model import constants as C
from .label_isolation import (CORPUS_INDEX_SCHEMA, SIDECAR_SCHEMA_V3,
                              TRAIN_OUTCOMES_ENV, LabelIsolationError,
                              audit_pack_identity, load_train_outcomes,
                              load_train_overlay, sidecar_schema_of)

PACK_SCHEMA = "wm_n1_pack/1"
ALLOWED_SPLITS = ("TRAIN", "CALIBRATION")
SPLIT_ALIASES = {"CAL": "CALIBRATION", "TRAIN": "TRAIN",
                 "CALIBRATION": "CALIBRATION"}
# Split labels that must never be opened inside a pack.
FORBIDDEN_SPLITS = ("HOLDOUT", "TEST", "SEALED")
FORBIDDEN_CHUNKS = ("chunk5",)
# The all-native reference arm of the paired CONTRAST loss (design section 3).
# The reference is identified by its PACK POLICY, not its arm suffix: the
# chunk-1/chunk-3-fix27 ``__native`` arms carry policy ``switching`` and are
# NOT references, while chunk4 ``native``/``native_base`` and truearms
# chunk2/3 ``native`` carry policy ``native`` and are.  ``REFERENCE_ARMS`` is
# kept for callers/tests that still name the reference arms.
REFERENCE_POLICIES = ("native",)
REFERENCE_ARMS = ("native", "native_base")

#: Arms whose OUTCOME supervision is removed for this retrain (decision
#: (b), registered audit WM-N1-LABEL-TIME-FIX1-20261003).  The source-inferred
#: snapshot at run start (``features/featurizer.py`` origin 0, lines 207-208)
#: is misaligned with the packed label ``t0`` for these arms, so no label row
#: may enter the outcome loss or the TRAIN outcome normalization.  The
#: kinematic/step/wall/HDV terms for these items are UNCHANGED; only the
#: outcome mask/targets/collision are zeroed (fail-closed).  Do not extend this
#: tuple without a new registered audit.
OUTCOME_SUPERVISION_EXCLUDED_ARMS = ("base", "native_base")

# Shard ``outcomes.parquet`` column -> model OUTCOME_TARGETS (INTERFACES
# Mismatch 3; TRAIN owns this mapping).
OUTCOME_COLUMN_MAP = {
    "delay_s": "total_delay_ref_s",
    "ttc_lt3_s": "ttc_lt3_veh_s",
    "hard_brake_s": "hard_brake_s",
}
COLLISION_COLUMN = "collision_any"
TAU_AVAILABLE_COLUMN = "tau_available"

TENSOR_KEYS = ("veh_step", "veh_mask", "plan", "wall", "wall_mask", "counter",
               "accel", "target_mask", "lane", "arrivals",
               "outcome_quantiles", "outcome_mask", "collision")
META_KEYS = ("world", "arm", "chunk", "split", "t0_s", "pair_key", "policy",
             "plan_row", "tick", "label_t0_rows", "label_t0_violations",
             "outcome_supervision_excluded")

_ITEM_REQUIRED = ("world", "arm", "chunk", "split", "t0_s", "pair_key",
                  "tick", "plan_row", "outcome_rows")

#: Packed arrays whose axis 1 is the vehicle-slot axis (K_max).  A pack whose
#: slot dim disagrees with its manifest ``k_max`` is refused (amendment 5:
#: K_max is a format parameter, 128 for v2 and 192 for v3).
SLOT_AXIS_ARRAYS = ("veh_step", "veh_mask", "plan", "accel_next",
                    "target_mask", "lane_next")


class CorpusRefusal(PermissionError):
    """Access to a forbidden split/chunk (HOLDOUT/TEST/chunk-5) was requested."""


# --------------------------------------------------------- outcome mapping --
def _as_rows(rows):
    """Yield per-tau rows as dicts from a DataFrame or an iterable of dicts."""
    if rows is None:
        return []
    cols = getattr(rows, "columns", None)
    if cols is not None:                       # pandas DataFrame
        return [dict(zip(cols, rec)) for rec in rows.itertuples(index=False)]
    if isinstance(rows, dict):
        return [rows]
    return [dict(r) for r in rows]


def _num(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def is_outcome_supervision_excluded(item_or_arm) -> bool:
    """True iff an item (or arm) loses outcome supervision for this retrain.

    Decision (b) excludes the origin-0-snapshotted ``base`` /
    ``native_base`` arms.  A re-aligned supplementary item (registered
    damping-snap, pack marker ``damping_snap=True``) carries the SAME arm name
    but a snapshot origin at its own ``t0``, so it is NOT excluded; the
    pack_v3 origin-0 copy (no marker) stays excluded.
    """
    if isinstance(item_or_arm, dict) and item_or_arm.get("damping_snap") is True:
        return False
    arm = item_or_arm.get("arm") if isinstance(item_or_arm, dict) else item_or_arm
    return str(arm).strip().lower() in OUTCOME_SUPERVISION_EXCLUDED_ARMS


def _t0_row_pick(rows, t0_s):
    """``{tau: row}`` the outcome mapper selects, proving t0 alignment.

    Legacy (``t0_s=None``) keeps last-row-wins over every row.  With ``t0_s``
    only rows stamped at that decision time are eligible; a duplicate tau at
    t0 raises ``ValueError`` (fail closed); every selected row is then
    re-checked to carry ``t == t0_s`` -- a selected row at any other decision
    time raises, so a future refactor cannot silently reintroduce the
    misaligned label.
    """
    t0_key = None if t0_s is None else round(float(t0_s), 6)
    by_tau = {}
    for row in _as_rows(rows):
        if t0_key is not None \
                and round(_num(row.get("t"), np.nan), 6) != t0_key:
            continue
        tau = round(_num(row.get("tau"), np.nan), 6)
        if t0_key is not None and tau in by_tau:
            raise ValueError(
                f"duplicate outcome row for tau {tau} at t0 {t0_key} "
                "(fail closed)")
        by_tau[tau] = row
    if t0_key is not None:
        for tau, row in by_tau.items():
            t = round(_num(row.get("t"), np.nan), 6)
            if t != t0_key:
                raise ValueError(
                    f"selected outcome row for tau {tau} is stamped t={t}, "
                    f"not the item t0 {t0_key} (fail closed)")
    return by_tau


def label_t0_rows(item_rows, t0_s) -> int:
    """Count of outcome rows actually selected at the item's own t0.

    The per-item check for the t0 label fix: re-derives the mapper's pick and
    asserts every selected row is stamped at the item ``t0_s`` (the assert is
    inside :func:`_t0_row_pick`, so any misaligned/duplicated pick raises).
    ``t0_s=None`` selects the legacy path and reports 0 asserted t0 rows.
    """
    if t0_s is None:
        return 0
    return len(_t0_row_pick(item_rows, t0_s))


def map_outcome_targets(rows, taus=C.TAUS, *, t0_s=None):
    """Map one item's shard ``outcomes.parquet`` rows to model targets.

    Returns ``(targets [N_TAU, N_TGT], mask [N_TAU], collision [N_TAU])``.
    ``targets`` is ordered as ``model.constants.OUTCOME_TARGETS``
    (``delay_s, ttc_lt3_s, hard_brake_s``) from the v0.4a columns
    (``total_delay_ref_s, ttc_lt3_veh_s, hard_brake_s``); ``mask`` follows
    ``tau_available`` (missing tau -> False); ``collision`` is
    ``collision_any``.  Missing/NaN values map to 0.

    When ``t0_s`` is given, only rows whose ``t`` equals the item's own
    ``t0_s`` are considered before the per-tau pick, so the label comes from
    the item's decision time rather than the last row in a multi-decision-time
    cell.  A tau duplicated at that t0 fails closed (``ValueError``); a tau
    absent at that t0 leaves its mask False.  ``t0_s=None`` keeps the
    historical last-row-wins behaviour.
    """
    taus = [round(float(t), 6) for t in taus]
    targets = np.zeros((len(taus), C.N_OUTCOME_TARGETS), dtype=np.float32)
    mask = np.zeros(len(taus), dtype=bool)
    collision = np.zeros(len(taus), dtype=np.float32)
    by_tau = _t0_row_pick(rows, t0_s)
    for i, tau in enumerate(taus):
        row = by_tau.get(tau)
        if row is None:
            continue
        available = row.get(TAU_AVAILABLE_COLUMN, True)
        mask[i] = True if available is None else bool(available)
        for j, name in enumerate(C.OUTCOME_TARGETS):
            targets[i, j] = _num(row.get(OUTCOME_COLUMN_MAP[name]))
        collision[i] = _num(row.get(COLLISION_COLUMN))
    return targets, mask, collision


def _wrapped_corpus(corpus):
    """The :class:`MemmapCorpus` behind a wrapper (or ``corpus`` itself)."""
    for obj in (corpus, getattr(corpus, "corpus", None),
                getattr(corpus, "_v1", None)):
        if obj is not None and hasattr(obj, "_items") \
                and hasattr(obj, "outcome_target"):
            return obj
    return None


def outcome_supervision_exclusion(corpus) -> dict:
    """FAIL-CLOSED receipt for decision (b).

    Returns ``{"arms", "count", "items"}`` for the TRAIN source: the sorted
    ``world__arm`` keys of every item whose arm is in
    :data:`OUTCOME_SUPERVISION_EXCLUDED_ARMS` AND that actually carries a
    t0-aligned label row (so supervision is truly removed).  An excluded-arm
    item with no t0 label contributes nothing and is not counted.

    Before recording a key the function re-reads the item through the
    DATASET's ``outcome_target`` (the single accessor ``fit_outcome_norm``
    consumes) and ASSERTS the delivered supervision is all-zero -- an
    all-False mask plus zero targets and collision.  A non-zero delivery for
    an excluded arm means the exclusion silently failed, so this raises
    :class:`CorpusRefusal` (fail closed) rather than allowing a contaminating
    label into the outcome loss or the OUTCOME_NORM statistics.

    A source with no packed item list (explicit synthetic windows, or an
    unresolved wrapper) returns an empty receipt; it is never guessed.
    """
    base = _wrapped_corpus(corpus)
    parts = getattr(base, "parts", None) if base is not None else None
    if parts and not hasattr(base, "_outcomes"):
        # Composite corpus (A9 --extra-corpus): run the same fail-closed
        # check on every member pack and merge, so (b) is verified and
        # counted rather than silently skipped.
        merged = {"arms": list(OUTCOME_SUPERVISION_EXCLUDED_ARMS),
                  "count": 0, "items": []}
        for part in parts:
            sub = outcome_supervision_exclusion(part)
            merged["count"] += int(sub.get("count", 0))
            merged["items"].extend(sub.get("items", []))
        merged["items"] = sorted(merged["items"])
        return merged
    if base is None or not hasattr(base, "_outcomes"):
        return {"arms": list(OUTCOME_SUPERVISION_EXCLUDED_ARMS),
                "count": 0, "items": []}
    rows = base._outcomes()
    items = base._items
    order = getattr(base, "_order", list(range(len(items))))
    keys = []
    for pos, item_id in enumerate(order):
        item = items[item_id]
        if not is_outcome_supervision_excluded(item):
            continue
        positions = [int(p) for p in item["outcome_rows"] if int(p) >= 0]
        item_rows = base._select_outcome_rows(rows, positions)
        _, raw_mask, _ = map_outcome_targets(item_rows, t0_s=item["t0_s"])
        if not bool(np.any(raw_mask)):
            continue
        targets, mask, collision = base.outcome_target(pos)
        if bool(np.any(mask)) or bool(np.any(np.asarray(targets) != 0)) \
                or bool(np.any(np.asarray(collision) != 0)):
            raise CorpusRefusal(
                f"outcome supervision for excluded arm {item['arm']!r} "
                f"(item {item['world']}__{item['arm']}) is NOT removed by "
                "outcome_target; refusing (fail closed): a label for an "
                "excluded arm would enter the outcome loss / OUTCOME_NORM")
        keys.append(f"{item['world']}__{item['arm']}")
    return {"arms": list(OUTCOME_SUPERVISION_EXCLUDED_ARMS),
            "count": len(keys), "items": sorted(keys)}


def excluded_outcome_item_keys(corpus) -> list:
    """Keys (``world__arm``) of items whose outcome supervision is removed.

    Thin list view of :func:`outcome_supervision_exclusion` (decision
    (b)); the receipt form carries the same keys under ``items``.
    """
    return list(outcome_supervision_exclusion(corpus)["items"])


# ------------------------------------------------------------- pair logic --
def is_reference(item: dict) -> bool:
    """True iff an item is the all-native reference arm of a pair.

    The pack POLICY decides: ``policy == 'native'`` (chunk4 ``native`` /
    ``native_base``, truearms chunk2/3 ``native``) is a reference; the
    chunk1/fix27 ``__native`` arms carry policy ``switching`` and are NOT
    (their arm suffix must never promote them to a reference).  The arm
    suffix is consulted only for a pack that records no policy at all.
    """
    policy = item.get("policy")
    if policy is None:
        policy = item.get("arm")
    return str(policy).strip().lower() in REFERENCE_POLICIES


def _reference_of(group, items):
    """Pick the reference item id inside a pair group (``native`` preferred).

    A reference whose outcome supervision is removed by decision (b)
    (the origin-0 ``base``/``native_base`` copies of ``pack_v3``) is NEVER
    chosen while a NON-excluded reference exists in the same group: the
    paired-effect contrast would otherwise be anchored on an item that
    delivers all-zero supervision.  The non-excluded sibling is preferred --
    concretely the re-aligned ``damping_snap`` ``native_base`` item (marker
    ``damping_snap=True``, extra pack) or a true ``native`` arm.  Only when
    EVERY reference in the group is outcome-excluded does the historical
    preference order apply (freezing the pre-damping behaviour for groups
    that have no supervised reference at all).
    """
    def _supervised(i):
        return is_reference(items[i]) \
            and not is_outcome_supervision_excluded(items[i])

    preferred = [i for i in group
                 if items[i].get("arm") == "native" and _supervised(i)]
    if preferred:
        return preferred[0]
    refs = [i for i in group if _supervised(i)]
    if refs:
        return refs[0]
    preferred = [i for i in group
                 if items[i].get("arm") == "native"
                 and is_reference(items[i])]
    if preferred:
        return preferred[0]
    refs = [i for i in group if is_reference(items[i])]
    return refs[0] if refs else None


def contrast_batch_indices(items, *, device=None):
    """Within a batch, the ``(a, b)`` positions of treatment vs native arms.

    ``a`` is the treatment arm, ``b`` the all-native reference, so the
    comonotone contrast ``q_a - q_b`` equals ``V(P) - V(all-native)``.
    Returns ``{"a": LongTensor, "b": LongTensor}`` (on ``device`` when given,
    so the positions index the batch tensors without a host->device copy);
    empty if no pair is complete in the batch.
    """
    groups = {}
    for pos, item in enumerate(items):
        groups.setdefault(item["pair_key"], []).append(pos)
    a, b = [], []
    for group in groups.values():
        refs = [p for p in group if is_reference(items[p])]
        if not refs:
            continue
        ref = refs[0]
        for p in group:
            if p == ref or is_reference(items[p]):
                continue
            a.append(p)
            b.append(ref)
    return dict(a=torch.tensor(a, dtype=torch.long, device=device),
                b=torch.tensor(b, dtype=torch.long, device=device))


# ------------------------------------------------- pair-aware batch sampler --
def pair_groups(items) -> dict:
    """``pair_key -> [item positions]`` for a source's item metadata."""
    groups: dict = {}
    for pos, item in enumerate(items):
        groups.setdefault(item["pair_key"], []).append(pos)
    return groups


def contrast_pairs(items) -> list:
    """Every ``(treatment_pos, reference_pos)`` a pack can form.

    A group is pairable only when it contains a policy-native reference
    (:func:`is_reference`); its treatments are every non-reference item.  A
    group with no policy-native item (e.g. the chunk1/fix27 switching
    ``__native`` groups) yields no pair, so its items never enter the paired
    CONTRAST loss as a reference.  Rows are ordered deterministically by
    sorted ``pair_key`` then position.
    """
    groups = pair_groups(items)
    pairs: list = []
    for key in sorted(groups):
        group = groups[key]
        ref = _reference_of(group, items)
        if ref is None:
            continue
        for pos in group:
            if pos == ref or is_reference(items[pos]):
                continue
            pairs.append((pos, ref))
    return pairs


def _eligible_partner_map(pairs) -> dict:
    """``position -> {eligible partner positions}`` over a pair pool."""
    partners: dict = {}
    for treatment, reference in pairs:
        treatment, reference = int(treatment), int(reference)
        partners.setdefault(treatment, set()).add(reference)
        partners.setdefault(reference, set()).add(treatment)
    return partners


def batch_has_pair(positions, items, pairs=None) -> bool:
    """True iff the batch completes a treatment + policy-native reference pair.

    With an eligible ``pairs`` pool supplied, only a pair FROM that pool
    counts: a batch whose only complete sibling pair lies OUTSIDE the pool
    (a filtered/unsupported pair) is NOT complete.  Without ``pairs`` the
    check stays purely semantic (any same-``pair_key`` treatment +
    policy-native reference), the historical behavior.
    """
    if pairs is not None:
        partners = _eligible_partner_map(pairs)
        present = {int(p) for p in positions}
        return any(q in present
                   for p in present for q in partners.get(p, ()))
    groups: dict = {}
    for pos in positions:
        groups.setdefault(items[pos]["pair_key"], []).append(pos)
    for group in groups.values():
        has_ref = any(is_reference(items[p]) for p in group)
        has_treat = any(not is_reference(items[p]) for p in group)
        if has_ref and has_treat:
            return True
    return False


def ensure_pair(positions, items, pairs, cursor: int) -> tuple:
    """Return ``(positions, next_cursor)`` with >=1 complete eligible pair.

    A batch that already completes a pair FROM the supplied eligible
    ``pairs`` pool is returned unchanged; a batch holding only a semantic
    sibling pair OUTSIDE the pool is completed by appending the missing
    member(s) of the deterministically cycled ``pairs[cursor % len(pairs)]``
    (the appended reference is always a policy-native item).  ``pairs=None``
    means the caller already resolved the default pool upstream (see
    :func:`epoch_batches`); an explicitly EMPTY pool never invents a pair.
    The caller's primary item stream is unchanged, so every item is still
    seen once per epoch; only the pair partner repeats.
    """
    positions = list(positions)
    if not pairs or batch_has_pair(positions, items, pairs):
        return positions, cursor
    treatment, reference = pairs[cursor % len(pairs)]
    present = set(positions)
    if reference not in present and treatment not in present:
        positions.extend([treatment, reference])
    elif reference not in present:
        positions.append(reference)
    elif treatment not in present:
        positions.append(treatment)
    return positions, cursor + 1


def pair_dense_batches(items, pairs_per_batch, seed, epoch, pairs=None,
                       merge_coverage: bool = False,
                       group_aware: bool = False) -> list:
    """The pair-DENSE batch positions of ONE deterministic epoch.

    ``pairs_per_batch=K >= 1`` makes every batch drawn from the eligible
    ``(treatment, reference)`` pool hold EXACTLY K complete pairs (the final
    group may be short when the pool size is not a multiple of K).  The pool
    is visited in a seeded permutation, so the batch composition is a pure
    function of ``(seed, epoch, pairs)``.  A group that shares one policy-
    native reference contributes ``1 + n_treatments`` distinct POSITIONS (the
    reference recurs), never a duplicated index: each of the K pairs still
    counts once through :func:`contrast_batch_indices`.

    Items that appear in NO eligible pair (a group with no policy-native
    reference, or a treatment with no shared support) are still emitted, in a
    seeded order, as trailing coverage batches.  A coverage batch holds at
    most one item per ``pair_key`` so it can never form an unsupported pair;
    every primary item is therefore still seen exactly once per epoch.

    ``pairs=None`` resolves the full default pool (:func:`contrast_pairs`);
    an explicitly EMPTY pool (no eligible pair at all) falls back to plain
    seeded batches so no item is dropped.
    """
    n = len(items)
    if n == 0:
        return []
    k = max(1, int(pairs_per_batch))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int((int(seed) * 1000003 + int(epoch))
                              % (2 ** 63 - 1)))
    if pairs is None:
        pairs = contrast_pairs(items)
    if not pairs:
        order = torch.randperm(n, generator=generator).tolist()
        return [order[s:s + 2 * k] for s in range(0, n, 2 * k)]
    perm = torch.randperm(len(pairs), generator=generator).tolist()
    if group_aware:
        # A11 ranking loss: keep the SAME seeded pair permutation, but walk it
        # group-by-group (``pair_key`` in first-appearance order) so up to K
        # pairs of one decision land in one batch and its treatment arms
        # co-occur.  Deterministic; the pair set and coverage tail are
        # unchanged.
        seen: list = []
        buckets: dict = {}
        for pi in perm:
            key = items[int(pairs[pi][0])]["pair_key"]
            if key not in buckets:
                buckets[key] = []
                seen.append(key)
            buckets[key].append(pi)
        perm = [pi for key in seen for pi in buckets[key]]
    out: list = []
    covered: set = set()
    for start in range(0, len(perm), k):
        positions: list = []
        present: set = set()
        for pi in perm[start:start + k]:
            treatment, reference = int(pairs[pi][0]), int(pairs[pi][1])
            for pos in (treatment, reference):
                if pos not in present:
                    present.add(pos)
                    positions.append(pos)
            covered.add(treatment)
            covered.add(reference)
        out.append(positions)
    groups: dict = {}
    for pos in torch.randperm(n, generator=generator).tolist():
        if pos in covered:
            continue
        groups.setdefault(items[pos]["pair_key"], []).append(pos)
    # Coverage: every item NOT in the eligible pool is still emitted exactly
    # once, but a coverage batch takes AT MOST ONE item per ``pair_key`` (the
    # buckets are drained round-robin up to the pair-batch slot budget) so it
    # can never complete an unsupported sibling pair - a treatment plus the
    # native reference of a group the eligible pool filtered out.  This is
    # what makes the outcome-required pool guarantee hold for EVERY batch.
    buckets = list(groups.values())
    if merge_coverage and out:
        # A8 batch fix: fold every coverage item INTO the pair batches
        # (round-robin, seeded order) instead of trailing pair-less batches,
        # so every batch keeps its K outcome-supported pairs (a pair-less
        # batch cannot supply the registered response_contrast term).  An
        # item never joins a batch already holding its ``pair_key`` (so it
        # cannot complete an unsupported sibling pair) and a batch takes at
        # most one coverage item per ``pair_key``.
        keys = [{items[p]["pair_key"] for p in b} for b in out]
        flat = []
        while any(buckets):
            for bucket in buckets:
                if bucket:
                    flat.append(bucket.pop())
        cursor = 0
        nb = len(out)
        for pos in flat:
            key = items[pos]["pair_key"]
            for step in range(nb):
                j = (cursor + step) % nb
                if key not in keys[j]:
                    out[j].append(pos)
                    keys[j].add(key)
                    cursor = j + 1
                    break
            else:
                raise ValueError(
                    "merge_coverage: no pair batch can take coverage item "
                    f"{pos} (pair_key {key!r}) without completing a pair")
        return out
    while any(buckets):
        batch: list = []
        for bucket in buckets:
            if len(batch) >= 2 * k:
                break
            if bucket:
                batch.append(bucket.pop())
        if batch:
            out.append(batch)
    return out


def epoch_batches(items, batch_size, seed, epoch, pairs=None,
                  pairs_per_batch=None, merge_coverage: bool = False,
                  group_aware: bool = False) -> list:
    """The pair-completed batch positions of ONE deterministic epoch.

    The epoch order is a seeded random permutation of every position; the
    per-batch pair completion reuses :func:`ensure_pair` with a per-epoch
    cursor.  ``pairs`` may be any eligible ``(treatment, reference)`` pool:
    a filtered pool restricts which completed pairs count as complete and
    which partners are appended, while ``None`` means the full default pool
    (:func:`contrast_pairs`).  The seed/epoch derived generator makes the
    batching a pure function of ``(items, batch_size, seed, epoch)`` so the
    in-process path and a ``torch.utils.data.DataLoader`` (via
    :class:`PairAwareBatchSampler`) produce byte-identical batches.

    ``pairs_per_batch=K`` switches to :func:`pair_dense_batches` (every batch
    holds K complete eligible pairs); ``None`` keeps the historical
    one-completed-pair-per-batch behaviour byte-for-byte.
    """
    if pairs_per_batch is not None:
        return pair_dense_batches(items, pairs_per_batch, seed, epoch,
                                  pairs=pairs, merge_coverage=merge_coverage,
                                  group_aware=group_aware)
    n = len(items)
    bs = max(1, int(batch_size))
    if n == 0:
        return []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int((int(seed) * 1000003 + int(epoch))
                              % (2 ** 63 - 1)))
    order = torch.randperm(n, generator=generator).tolist()
    if pairs is None:
        pairs = contrast_pairs(items)
    out: list = []
    cursor = 0
    for start in range(0, n, bs):
        idx = order[start:start + bs]
        if not idx:
            continue
        if pairs:
            idx, cursor = ensure_pair(idx, items, pairs, cursor)
        out.append(list(idx))
    return out


class PairAwareBatchSampler:
    """A ``batch_sampler`` yielding :func:`epoch_batches` (deterministic).

    Drop-in for ``torch.utils.data.DataLoader(batch_sampler=...)``: it yields
    lists of dataset positions.  With ``epochs=None`` it re-cycles epochs
    forever (matching the inference-time in-process generator); with a count it
    yields exactly that many epochs (the parity test).
    """

    def __init__(self, items, batch_size: int, seed: int = 0, *,
                 epoch: int = 0, epochs: Optional[int] = None,
                 pairs=None, pairs_per_batch=None,
                 merge_coverage: bool = False,
                 group_aware: bool = False) -> None:
        self.items = list(items)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_epoch = int(epoch)
        self.epochs = None if epochs is None else int(epochs)
        self.pairs = pairs
        self.pairs_per_batch = (None if pairs_per_batch is None
                                else int(pairs_per_batch))
        self.merge_coverage = bool(merge_coverage)
        self.group_aware = bool(group_aware)

    def _epoch_list(self, epoch: int) -> list:
        return epoch_batches(self.items, self.batch_size, self.seed, epoch,
                             self.pairs, pairs_per_batch=self.pairs_per_batch,
                             merge_coverage=self.merge_coverage,
                             group_aware=self.group_aware)

    def __iter__(self):
        epoch = self.start_epoch
        while self.epochs is None or epoch < self.start_epoch + self.epochs:
            for batch in self._epoch_list(epoch):
                yield batch
            epoch += 1

    def __len__(self) -> int:
        if self.epochs is None:
            raise TypeError("an infinite PairAwareBatchSampler has no length")
        if self.pairs_per_batch is not None:
            return self.epochs * max(1, len(self._epoch_list(self.start_epoch)))
        n = len(self.items)
        bs = max(1, self.batch_size)
        return self.epochs * max(1, (n + bs - 1) // bs)


def collate(items):
    """Stack a list of dataset items into a batch + the per-item metadata."""
    batch = {k: torch.stack([it[k] for it in items]) for k in TENSOR_KEYS}
    batch["meta"] = [{k: it[k] for k in META_KEYS} for it in items]
    return batch


# --------------------------------------------------------------- dataset ----
def _normalize_split(split: str) -> str:
    key = str(split).strip().upper()
    if key not in SPLIT_ALIASES:
        raise CorpusRefusal(
            f"split {split!r} is not allowed; the dataset opens only "
            f"{ALLOWED_SPLITS} (pass split='CALIBRATION' explicitly for CAL)")
    return SPLIT_ALIASES[key]


class MemmapCorpus(Dataset):
    """Map-style dataset over a packed memmap corpus.

    ``split`` defaults to ``"TRAIN"``.  Only ``TRAIN`` and ``CALIBRATION`` can
    be opened; CAL requires the explicit ``split="CALIBRATION"`` argument.
    A pack that contains any HOLDOUT/TEST/chunk-5 run is refused outright.
    """

    def __init__(self, pack_dir, split: str = "TRAIN", *,
                 pair_sampling: bool = False, k_max: int | None = None):
        self.pack_dir = Path(pack_dir)
        self.split = _normalize_split(split)
        self.pair_sampling = bool(pair_sampling)
        self.manifest = self._load_manifest()
        self.format = self.manifest.get("format")
        self.k_max = self._resolve_k_max(k_max)
        self._items = self._select_items()
        self._mm = {}
        self._outcome_rows = None
        self._outcome_mode = None          # lazily resolved: mixed|sidecar
        self._sidecar_dir: Optional[str] = None
        self._overlay = None               # schema/3 zero-row-item groups
        self._overlay_excluded = set()     # schema/3 excluded-group keys
        self._pack_index_data = None
        self._pack_index_loaded = False
        self._groups = self._pair_groups()
        self._order = self._build_order()
        self._pos_of = {item: pos for pos, item in enumerate(self._order)}

    # -- pickling (DataLoader workers) --------------------------------------
    def __getstate__(self) -> dict:
        """Drop open memmaps / parquet frames before pickling to a worker.

        ``np.memmap`` pickles its whole array (gigabytes for a real pack), and
        a spawned DataLoader worker re-executes the parent, so every worker
        would otherwise serialize the corpus (and the parent would stall /
        fail with a truncated pipe).  The caches are re-opened lazily on first
        ``_array`` / ``_outcomes`` access in the worker; fork inherits them.
        The sidecar decision itself (``_outcome_mode`` / ``_sidecar_dir``)
        is two small strings; the worker re-validates the sidecar receipt
        before any label read, exactly like the parent.
        """
        state = dict(self.__dict__)
        state["_mm"] = {}
        state["_outcome_rows"] = None
        state["_overlay"] = None
        state["_overlay_excluded"] = set()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("_mm", {})
        self.__dict__.setdefault("_outcome_rows", None)
        self.__dict__.setdefault("_outcome_mode", None)
        self.__dict__.setdefault("_sidecar_dir", None)
        self.__dict__.setdefault("_overlay", None)
        self.__dict__.setdefault("_overlay_excluded", set())
        self.__dict__.setdefault("_pack_index_data", None)
        self.__dict__.setdefault("_pack_index_loaded", False)

    # -- manifest / guards --------------------------------------------------
    def _resolve_k_max(self, requested) -> int | None:
        """Resolve + validate the pack K_max (slot width).

        The manifest ``k_max`` must agree with the packed slot dims, with an
        explicit ``format`` claim (v2 -> 128, v3 -> 192, amendment 5) and with
        the caller-requested ``k_max`` -- a mismatch (e.g. a v3 pack opened
        against a K=128 model/config) raises :class:`CorpusRefusal`.
        """
        declared = self.manifest.get("k_max")
        try:
            declared = None if declared is None else int(declared)
        except (TypeError, ValueError):
            raise CorpusRefusal(
                f"pack manifest k_max {self.manifest.get('k_max')!r} is not "
                "an integer") from None
        if self.format is not None:
            if self.format not in C.FORMAT_K_MAX:
                raise CorpusRefusal(
                    f"pack manifest format {self.format!r} is not a "
                    f"registered format ({sorted(C.FORMAT_K_MAX)})")
            if declared != C.k_max_of_format(self.format):
                raise CorpusRefusal(
                    f"pack format {self.format!r} carries K_max "
                    f"{C.k_max_of_format(self.format)}, not {declared}")
        for name in SLOT_AXIS_ARRAYS:
            spec = (self.manifest.get("arrays") or {}).get(name)
            if spec is None:
                continue
            shape = spec.get("shape") or []
            if len(shape) >= 2 and int(shape[1]) != declared:
                raise CorpusRefusal(
                    f"packed array {name} slot dim {shape[1]} != manifest "
                    f"k_max {declared}")
        if requested is not None and declared is not None \
                and int(requested) != declared:
            raise CorpusRefusal(
                f"K_max mismatch: pack at {self.pack_dir} carries K_max "
                f"{declared} but the model/config expects K_max "
                f"{int(requested)}")
        return declared

    def _load_manifest(self) -> dict:
        path = self.pack_dir / "manifest.json"
        if not path.is_file():
            raise CorpusRefusal(f"no packed manifest at {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        schema = manifest.get("schema")
        if schema != PACK_SCHEMA:
            raise CorpusRefusal(
                f"pack schema {schema!r} != {PACK_SCHEMA!r} at {path}")
        items = manifest.get("items")
        if not items:
            raise CorpusRefusal(
                f"pack manifest at {path} carries no items; an empty "
                "manifest proves no training-only outcome table and is "
                "refused before any label read")
        for item in items:
            self._refuse_item(item)
        return manifest

    @staticmethod
    def _refuse_item(item: dict) -> None:
        chunk = str(item.get("chunk", ""))
        split = str(item.get("split", "")).strip().upper()
        # separator/case-insensitive spelling check (mirrors the
        # dependency-free guard in train.label_isolation)
        norm = re.sub(r"[\s\-_.]+", "", chunk).lower()
        if split in FORBIDDEN_SPLITS:
            raise CorpusRefusal(
                f"packed item {item.get('world')}/{item.get('arm')} has "
                f"forbidden split {split!r}; HOLDOUT/TEST are sealed")
        if norm.startswith("chunk5"):
            raise CorpusRefusal(
                f"packed item {item.get('world')} is under {chunk!r} "
                "(chunk-5 TEST is sealed)")
        if split not in ALLOWED_SPLITS:
            raise CorpusRefusal(
                f"packed item {item.get('world')}/{item.get('arm')} carries "
                f"unknown split spelling {item.get('split')!r}; only "
                f"{ALLOWED_SPLITS} are recognized")
        if norm.startswith("chunk") and not norm[5:6].isdigit():
            raise CorpusRefusal(
                f"packed item {item.get('world')}/{item.get('arm')} carries "
                f"unknown chunk spelling {chunk!r}")

    def _select_items(self) -> list:
        items = []
        for raw in self.manifest.get("items", []):
            missing = [k for k in _ITEM_REQUIRED if k not in raw]
            if missing:
                raise CorpusRefusal(
                    f"packed item {raw.get('world')}/{raw.get('arm')} misses "
                    f"required key(s) {missing}")
            item = dict(raw)
            item["t0_s"] = float(item["t0_s"])
            item["split"] = str(item["split"]).upper()
            item["outcome_rows"] = list(item["outcome_rows"])
            items.append(item)
        return [it for it in items if it["split"] == self.split]

    # -- array access -------------------------------------------------------
    def _array(self, name):
        if name not in self._mm:
            spec = self.manifest["arrays"][name]
            path = self.pack_dir / spec["file"]
            arr = np.load(path, mmap_mode="r")
            if name in SLOT_AXIS_ARRAYS and self.k_max is not None \
                    and arr.ndim >= 2 and int(arr.shape[1]) != self.k_max:
                raise CorpusRefusal(
                    f"packed array {name} slot dim {arr.shape[1]} != "
                    f"manifest k_max {self.k_max}")
            self._mm[name] = arr
        return self._mm[name]

    def _pack_index(self) -> Optional[dict]:
        """The corpus pack ``INDEX.json`` (``None`` when the pack has none).

        Manifest-only synthetic/legacy fixtures carry no INDEX; a present
        INDEX with a foreign schema on a pack is refused (fail closed).
        """
        if not self._pack_index_loaded:
            path = self.pack_dir / "INDEX.json"
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except ValueError as e:
                    raise CorpusRefusal(
                        f"invalid pack INDEX.json at {path}: {e}") from e
                if data.get("schema") != CORPUS_INDEX_SCHEMA:
                    raise CorpusRefusal(
                        f"pack INDEX schema {data.get('schema')!r} != "
                        f"{CORPUS_INDEX_SCHEMA!r} at {path}")
                self._pack_index_data = data
            self._pack_index_loaded = True
        return self._pack_index_data

    def _mixed_split_identity_pack(self) -> bool:
        """True iff this pack mixes item splits (TRAIN then needs a sidecar).

        The manifest is the CLAIM of what the packed outcome table can hold,
        and a PRESENT pack INDEX must agree with it (metadata-only
        :func:`train.label_isolation.audit_pack_identity`): a wholly
        single-split TRAIN manifest keeps the original mixed-table read only
        when it carries NO INDEX (legacy/synthetic manifest-only fixtures) or
        when its INDEX reconciles 1:1 with all-TRAIN run identities.  An
        INDEX naming a non-TRAIN or unmatched run contradicts the
        training-only claim and fails closed with :class:`CorpusRefusal`
        BEFORE any label read -- the packed table could hold excluded rows.
        An empty manifest was already refused at load (it proves no
        training-only table).  Any MIXED-split manifest returns ``True`` only
        AFTER its INDEX carries a usable per-run identity (``shard_dir`` +
        ``n_outcome_rows`` for every run, as the corpus writer emits); a
        missing/empty INDEX or one incomplete identity run fails closed with
        :class:`CorpusRefusal` BEFORE any label read -- there is no
        synthetic/legacy mixed-pack fallback.
        """
        index = self._pack_index()
        splits = {str(it.get("split", "")).strip().upper()
                  for it in self.manifest.get("items") or []}
        if splits <= {"TRAIN"}:
            if index is not None:
                try:
                    audit_pack_identity(self.manifest, index)
                except LabelIsolationError as e:
                    raise CorpusRefusal(
                        f"corpus pack {self.pack_dir} claims a wholly TRAIN "
                        "manifest, but its INDEX contradicts it; refusing "
                        f"before any label read: {e}") from e
            return False
        if index is None:
            raise CorpusRefusal(
                f"mixed-split corpus pack {self.pack_dir} carries no "
                "INDEX.json run identity; split='TRAIN' refuses before any "
                "label read (no manifest-only mixed fallback)")
        runs = index.get("runs")
        if not isinstance(runs, list) or not runs:
            raise CorpusRefusal(
                f"mixed-split corpus pack {self.pack_dir} has an INDEX.json "
                "without a usable runs list; split='TRAIN' refuses before "
                "any label read")
        for rec in runs:
            if not isinstance(rec, dict) \
                    or rec.get("shard_dir") in (None, "") \
                    or rec.get("n_outcome_rows") in (None, ""):
                raise CorpusRefusal(
                    f"mixed-split corpus pack {self.pack_dir} has an INDEX "
                    "run with incomplete per-run identity (shard_dir / "
                    "n_outcome_rows); split='TRAIN' refuses before any "
                    "label read")
        return True

    def _resolve_outcome_source(self) -> None:
        """Decide the outcome source BEFORE any label bytes are read.

        On the refusal path the mode stays unresolved, so EVERY outcome
        access keeps refusing until a sidecar is supplied (no fallback to
        the mixed table once a mixed-split pack was detected).
        """
        if self.split == "TRAIN" and self._mixed_split_identity_pack():
            sidecar = os.environ.get(TRAIN_OUTCOMES_ENV, "").strip()
            if not sidecar:
                raise CorpusRefusal(
                    f"pack {self.pack_dir} is a mixed-split corpus pack; "
                    f"split={self.split!r} must not open its packed "
                    "outcomes table (it holds other splits' labels). "
                    f"Refusing before any label read: set "
                    f"{TRAIN_OUTCOMES_ENV} to a TRAIN-only outcome sidecar "
                    "directory built by train.label_isolation.")
            self._outcome_mode = "sidecar"
            self._sidecar_dir = sidecar
        else:
            self._outcome_mode = "mixed"
            self._sidecar_dir = None

    def _outcomes(self):
        if self._outcome_rows is None:
            if self._outcome_mode is None:
                self._resolve_outcome_source()
            if self._outcome_mode == "sidecar":
                try:
                    frame = load_train_outcomes(self.pack_dir,
                                                self._sidecar_dir)
                except LabelIsolationError as e:
                    raise CorpusRefusal(
                        f"TRAIN outcome sidecar refused: {e}") from e
                self._outcome_rows = frame
                if sidecar_schema_of(self._sidecar_dir) \
                        == SIDECAR_SCHEMA_V3:
                    try:
                        self._overlay, _rec = load_train_overlay(
                            self.pack_dir, self._sidecar_dir)
                    except LabelIsolationError as e:
                        raise CorpusRefusal(
                            f"TRAIN outcome overlay refused: {e}") from e
                    self._overlay_excluded = {
                        (str(g["batch"]), str(g["chunk"]), str(g["world"]),
                         str(g["arm"]), str(g["split"]),
                         round(float(g["t0_s"]), 6))
                        for g in (_rec.get("overlay") or {})
                        .get("excluded") or []}
            else:
                import pandas as pd

                spec = self.manifest.get("outcomes",
                                         {"file": "outcomes.parquet"})
                path = self.pack_dir / spec["file"]
                self._outcome_rows = pd.read_parquet(path)
        return self._outcome_rows

    def _select_outcome_rows(self, rows, positions, item=None):
        """Slice outcome rows for one item (``_outcomes`` called first).

        Mixed mode uses positional ``.iloc`` over the packed table; sidecar
        mode addresses the ORIGINAL global row ids through ``.loc`` over the
        isolated TRAIN frame, preserving exact values, dtypes and order.  A
        schema/2 sidecar may EXCLUDE a run's rows (its rebind failed); those
        ids are simply absent, so they are filtered out and the item trains
        with an all-False outcome mask (non-outcome losses unchanged).

        A schema/3 OVERLAY sidecar additionally carries rows for zero-row
        TRAIN items keyed by full item identity: an item whose packed
        ``outcome_rows`` is empty reads its overlay group (validated at
        load: identity, t0 rows, rule-(b) refusal, coverage).  A missing
        group refuses (fail closed) unless the arm is rule-(b) excluded,
        whose supervision stays removed by the dataset zeroing.
        """
        if not positions:
            if self._overlay is not None and item is not None:
                key = (str(item.get("batch") or item["chunk"]),
                       str(item["chunk"]), str(item["world"]),
                       str(item["arm"]), str(item["split"]),
                       round(float(item["t0_s"]), 6))
                group = self._overlay.get(key)
                if group is not None:
                    return group
                if key in getattr(self, "_overlay_excluded", ()) or \
                        is_outcome_supervision_excluded(item):
                    # receipt-excluded (e.g. alignment refusal) or rule-(b):
                    # the item legitimately trains with no outcome labels.
                    return rows.iloc[0:0]
                raise CorpusRefusal(
                    "TRAIN outcome overlay has no group for zero-row item "
                    f"{item.get('world')}__{item.get('arm')} at t0="
                    f"{item.get('t0_s')}; refusing (coverage gap)")
            return rows.iloc[0:0]
        if self._outcome_mode == "sidecar":
            present = [p for p in positions if p in rows.index]
            if not present:
                return rows.iloc[0:0]
            if len(present) != len(positions):
                # Exclusion is whole-run only; a partially present item
                # means the sidecar and pack disagree: fail closed.
                raise CorpusRefusal(
                    "TRAIN sidecar holds only part of an item's outcome "
                    f"rows ({len(present)}/{len(positions)}); refusing")
            return rows.loc[present]
        return rows.iloc[positions]

    # -- ordering / pairs ---------------------------------------------------
    def _pair_groups(self) -> dict:
        groups = {}
        for i, item in enumerate(self._items):
            groups.setdefault(item["pair_key"], []).append(i)
        return groups

    def _build_order(self) -> list:
        if not self.pair_sampling:
            return list(range(len(self._items)))
        order = []
        for pair_key in sorted(self._groups):
            group = self._groups[pair_key]
            ref = _reference_of(group, self._items)
            treatments = [i for i in group if i != ref]
            order.extend(treatments)
            if ref is not None:
                order.append(ref)
        return order

    def sibling_pairs(self, i: int) -> tuple:
        """Return two item positions of the same ``(world, t0)`` with different
        arms ``(a, b)`` for the paired CONTRAST loss, where ``a`` is the
        treatment arm and ``b`` the all-native reference.
        """
        if not 0 <= i < len(self._order):
            raise IndexError(f"item position {i} out of range")
        item_id = self._order[i]
        item = self._items[item_id]
        group = self._groups.get(item["pair_key"], [item_id])
        ref = _reference_of(group, self._items)
        if ref is None:
            raise ValueError(
                f"pair {item['pair_key']!r} has no native reference arm")
        if item_id == ref:
            others = [j for j in group if j != ref]
            if not others:
                raise ValueError(
                    f"pair {item['pair_key']!r} has no treatment arm")
            a, b = others[0], ref
        else:
            a, b = item_id, ref
        return self._pos_of[a], self._pos_of[b]

    # -- Dataset protocol ---------------------------------------------------
    def __len__(self) -> int:
        return len(self._order)

    def __getitem__(self, i: int) -> dict:
        """Return the six frozen tensors + step targets + outcome rows."""
        if i < 0:
            i += len(self._order)
        if not 0 <= i < len(self._order):
            raise IndexError(f"item position {i} out of range")
        item = self._items[self._order[i]]
        tick = int(item["tick"])
        plan_row = int(item["plan_row"])

        veh = self._array("veh_step")[tick]
        mask = self._array("veh_mask")[tick]
        plan = self._array("plan")[plan_row]
        wall = self._array("wall")[tick]
        wall_mask = self._array("wall_mask")[tick]
        counter = self._array("counter")[tick]
        accel = self._array("accel_next")[tick]
        target_mask = self._array("target_mask")[tick]
        lane = self._array("lane_next")[tick]
        arrivals = self._array("arrivals")[tick]

        rows = self._outcomes()
        positions = [int(p) for p in item["outcome_rows"] if int(p) >= 0]
        item_rows = self._select_outcome_rows(rows, positions, item=item)
        label_t0 = label_t0_rows(item_rows, item["t0_s"])
        targets, omask, collision = map_outcome_targets(
            item_rows, t0_s=item["t0_s"])
        excluded = is_outcome_supervision_excluded(item)
        if excluded:
            # Decision (b): no outcome supervision for these arms.  The
            # mask/targets/collision are zeroed here (the single place labels
            # enter); every other loss term is unchanged.
            targets = np.zeros_like(targets)
            omask = np.zeros_like(omask)
            collision = np.zeros_like(collision)

        return {
            "veh_step": torch.as_tensor(veh.astype(np.float32)),
            "veh_mask": torch.as_tensor(mask.astype(bool)),
            "plan": torch.as_tensor(plan.astype(np.float32)),
            "wall": torch.as_tensor(wall.astype(np.float32)),
            "wall_mask": torch.as_tensor(wall_mask.astype(bool)),
            "counter": torch.as_tensor(counter.astype(np.float32)),
            "accel": torch.as_tensor(accel.astype(np.float32)),
            "target_mask": torch.as_tensor(target_mask.astype(bool)),
            "lane": torch.as_tensor(lane.astype(np.int64)),
            "arrivals": torch.as_tensor(arrivals.astype(np.float32)),
            "outcome_quantiles": torch.as_tensor(targets),
            "outcome_mask": torch.as_tensor(omask),
            "collision": torch.as_tensor(collision),
            "world": item["world"],
            "arm": item["arm"],
            "chunk": item["chunk"],
            "split": item["split"],
            "t0_s": item["t0_s"],
            "pair_key": item["pair_key"],
            "policy": item.get("policy", item["arm"]),
            "plan_row": plan_row,
            "tick": tick,
            "label_t0_rows": int(label_t0),
            "label_t0_violations": 0,
            "outcome_supervision_excluded": bool(excluded),
        }

    def outcome_target(self, i: int) -> tuple:
        """``(targets, mask, collision)`` for item position ``i``.

        Reads only the packed outcome rows (never the vehicle/plan arrays), so
        the TRAIN-only normalization statistics can be fitted without building
        a full window per item.  ``i`` indexes the dataset order, exactly like
        :meth:`__getitem__`.
        """
        if i < 0:
            i += len(self._order)
        if not 0 <= i < len(self._order):
            raise IndexError(f"item position {i} out of range")
        item = self._items[self._order[i]]
        rows = self._outcomes()
        positions = [int(p) for p in item["outcome_rows"] if int(p) >= 0]
        item_rows = self._select_outcome_rows(rows, positions, item=item)
        if is_outcome_supervision_excluded(item):
            # (b): excluded arms never contribute to the outcome normalization
            # statistics either (all-False mask, zero targets/collision).
            return (np.zeros((len(C.TAUS), C.N_OUTCOME_TARGETS),
                             dtype=np.float32),
                    np.zeros(len(C.TAUS), dtype=bool),
                    np.zeros(len(C.TAUS), dtype=np.float32))
        return map_outcome_targets(item_rows, t0_s=item["t0_s"])

    # -- metadata -----------------------------------------------------------
    def worlds(self) -> list:
        """World ids in the pack (used for world-grouped early stopping)."""
        seen = []
        for item in self._items:
            if item["world"] not in seen:
                seen.append(item["world"])
        return sorted(seen)

    def pair_keys(self) -> list:
        """Sorted sibling-group keys present in the selected split."""
        return sorted(self._groups)
