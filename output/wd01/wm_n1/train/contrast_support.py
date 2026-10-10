"""Eligibility support for the paired CONTRAST loss (contrastfix recovery).

Why this module exists
----------------------
The registered A3.2 CONTRAST loss pairs sibling arms at the same
``(world, t0)`` (``pair_key``).  Sharing a world and a t0 does **not** imply
that a sibling pair shares a realised outcome horizon (the per-tau
``outcome_mask`` may cover disjoint taus) or a valid HDV slot (the two arms'
``target_mask & ~is_cav`` may be disjoint).  The previous loss formed every
policy-native pair, fed the mismatched/dead sub-term an all-false mask, and the
frozen ``contrast_loss``/``pinball`` denominators clamped a zero-support mask to
``1`` -- silently returning ``0.0``.  That is a fake positive loss: 11 of the
40 laptop smoke steps reported ``response_contrast`` exactly zero.

This module makes the support explicit and testable:

* :func:`item_support_masks` extracts the two per-item support vectors
  (``outcome`` over taus, ``hdv`` over slots) from either a v1
  ``MemmapCorpus`` item or a canonical Idea A window.
* :func:`shared_support` / :func:`build_eligible_pair_pool` keep only pairs
  that share at least one outcome tau or one valid HDV slot.  The pool feeds
  ``train.dataset.epoch_batches`` / ``PairAwareBatchSampler`` so the seeded
  pair-aware order never forces an unsupported pair into a batch -- while the
  epoch still visits **every** item, so unpaired examples keep training the
  other losses.
* :func:`shared_outcome_mask` / :func:`shared_hdv_mask` re-derive the same
  pair masks from the canonical batch at loss time (the sampling pool is a
  hint; the loss is the authority).  No epsilon floors, no fabricated labels.

Verified identity alignment (contrastfix identity recovery)
-----------------------------------------------------------
Slot numbers are NOT identities: FORMAT slots are persistent/reused
run-local positions, so two sibling arms may hold the same vehicle at
DIFFERENT slot indices and the same index at DIFFERENT vehicles.  When a
batch or item support carries verified canonical ``vehicle_ids`` (from the
TRAIN identity sidecar; ``-1`` = absent slot), HDV eligibility and the
loss-time HDV alignment match by COMMON CANONICAL ID instead of slot
intersection:

* :func:`common_verified_ids` / :func:`align_hdv_ids` compute the
  per-common-ID A/B slot pairs (each arm at ITS own index);
* :func:`shared_support` reports ``common_ids`` and :func:`has_support`
  requires them for the HDV sub-term whenever both sides carry IDs;
* batches without IDs (legacy synthetic windows) keep the documented
  slot-intersection fallback.  An actual packed fit never reaches that
  fallback: the trainer's verified-identity gate refuses it first.
  IDs never enter the network; the loss math is unchanged.

Everything here is pure tensor/array inspection: no network, no SUMO, no
writes.  Real packs are read only through the source the trainer was handed
and only for the mask arrays.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor


def _as_bool_vec(value) -> Tensor:
    """Flatten any mask-like value to a 1-D bool CPU tensor."""
    if isinstance(value, Tensor):
        return value.to(torch.bool).reshape(-1)
    return torch.as_tensor(np.asarray(value)).to(torch.bool).reshape(-1)


def item_support_masks(item: dict) -> dict:
    """``{"outcome": bool[N_TAU], "hdv": bool[K], "ids": int64[K]|None}``.

    ``outcome`` is the item's ``outcome_mask`` (per-tau availability); ``hdv``
    is ``target_mask & ~is_cav`` (the slots whose HDV response is realised and
    is not a CAV command slot).  ``ids`` is the item's verified canonical
    ``vehicle_ids`` (``-1`` absent) when it carries one, else ``None``.
    Accepts a v1 ``MemmapCorpus`` item (``target_mask`` ``[K]`` +
    ``veh_step`` for the CAV flag) or a canonical Idea A window
    (``target_mask``/``is_cav`` ``[1,K]``).
    """
    outcome = item.get("outcome_mask")
    target = item.get("target_mask")
    if outcome is None or target is None:
        raise ValueError(
            "contrast support needs 'outcome_mask' and 'target_mask' on the "
            "item; a canonical Idea A batch must not drop them")
    cav = item.get("is_cav")
    if cav is None and "veh_step" in item:
        from model import constants as C

        veh = item["veh_step"]
        cav = (torch.as_tensor(np.asarray(veh))[..., C.IS_CAV_IDX] > 0.5)
    hdv = _as_bool_vec(target)
    if cav is not None:
        hdv = hdv & ~_as_bool_vec(cav)
    return {"outcome": _as_bool_vec(outcome), "hdv": hdv,
            "ids": support_ids(item)}


def support_ids(item: dict) -> Optional[Tensor]:
    """The item's verified canonical ``vehicle_ids`` int64 ``[K]`` or ``None``."""
    ids = item.get("vehicle_ids", item.get("ids"))
    if ids is None:
        return None
    return torch.as_tensor(np.asarray(ids)).to(torch.long).reshape(-1)


def _long_vec(value) -> Tensor:
    return torch.as_tensor(np.asarray(value)).to(torch.long).reshape(-1)


def common_verified_ids(ids_a, valid_a, ids_b, valid_b) -> list:
    """Sorted canonical vehicle IDs HDV-verified (valid slot, ID >= 0) in both arms."""
    def _eligible(ids, valid):
        ids = _long_vec(ids)
        valid = _as_bool_vec(valid)
        return {int(v) for v, ok in zip(ids.tolist(), valid.tolist())
                if bool(ok) and int(v) >= 0}

    return sorted(_eligible(ids_a, valid_a) & _eligible(ids_b, valid_b))


def align_hdv_ids(ids_a, valid_a, ids_b, valid_b) -> tuple:
    """``(ja [M], jb [M])`` slot index pairs holding the SAME canonical ID.

    A shared slot number is not an identity: each matched pair ``(ja[m],
    jb[m])`` gathers arm A's response at ITS slot and arm B's at ITS OWN
    (possibly different) slot.  IDs are unique per row, so every slot index
    appears at most once and the matching is a bijection on the common IDs.
    """
    ids_a = _long_vec(ids_a)
    ids_b = _long_vec(ids_b)
    valid_a = _as_bool_vec(valid_a)
    valid_b = _as_bool_vec(valid_b)
    match = ((ids_a.unsqueeze(1) == ids_b.unsqueeze(0))
             & (ids_a >= 0).unsqueeze(1) & (ids_b >= 0).unsqueeze(0)
             & valid_a.unsqueeze(1) & valid_b.unsqueeze(0))
    return torch.nonzero(match, as_tuple=True)


def shared_support(support_a: dict, support_b: dict) -> dict:
    """Element-wise AND plus the common verified IDs of two support masks.

    ``common_ids`` is ``None`` when either side carries no verified IDs
    (legacy slot-intersection behaviour applies); otherwise it is the sorted
    list of canonical IDs that are HDV-verified in BOTH arms.
    """
    out = {"outcome": support_a["outcome"] & support_b["outcome"],
           "hdv": support_a["hdv"] & support_b["hdv"]}
    ids_a, ids_b = support_a.get("ids"), support_b.get("ids")
    out["common_ids"] = (
        common_verified_ids(ids_a, support_a["hdv"], ids_b, support_b["hdv"])
        if ids_a is not None and ids_b is not None else None)
    return out


def has_support(shared: dict) -> bool:
    """True iff the pair shares an outcome tau or a verified common HDV ID.

    Without verified IDs on both sides this falls back to the legacy slot
    intersection (synthetic windows only; never an actual packed fit).
    """
    ids = shared.get("common_ids")
    if ids is not None:
        return bool(shared["outcome"].any()) or len(ids) > 0
    return bool(shared["outcome"].any()) or bool(shared["hdv"].any())


def shared_outcome_mask(outcome_mask: Optional[Tensor], a: Tensor,
                        b: Tensor) -> Optional[Tensor]:
    """Loss-time pair outcome mask ``omask[a] & omask[b]`` (``None`` if absent)."""
    if outcome_mask is None:
        return None
    return outcome_mask[a].to(torch.bool) & outcome_mask[b].to(torch.bool)


def shared_hdv_mask(target_mask: Tensor, a: Tensor, b: Tensor,
                    is_cav: Optional[Tensor] = None) -> Tensor:
    """Loss-time pair HDV mask ``target_mask[a] & target_mask[b] & ~cav``."""
    mask = (target_mask[a].to(torch.bool) & target_mask[b].to(torch.bool))
    if is_cav is not None:
        cav = is_cav.to(torch.bool)
        mask = mask & ~cav[a] & ~cav[b]
    return mask


def source_support_fn(source) -> Callable[[int], dict]:
    """A ``position -> support masks`` accessor for a batching source.

    Prefers a cheap ``support_masks(pos)`` method when the source exposes one
    (``IdeaACorpus``) and otherwise falls back to building the item through
    ``source[pos]``.  A ``HeldOutTrainView`` delegates to its wrapped corpus
    with the view's kept-position mapping, so the positions match
    ``train.dataset`` item order exactly.
    """
    direct = getattr(source, "support_masks", None)
    if callable(direct):
        return direct
    inner = getattr(source, "corpus", None)
    kept = getattr(source, "kept", None)
    if inner is not None and kept is not None:
        inner_fn = getattr(inner, "support_masks", None)
        if callable(inner_fn):
            return lambda pos: inner_fn(int(kept[int(pos)]))

    def _from_item(pos: int) -> dict:
        return item_support_masks(source[int(pos)])

    return _from_item


def build_eligible_pair_pool(source, meta: Sequence[dict], *,
                             support_fn: Optional[Callable[[int], dict]] = None,
                             candidate_pairs: Optional[Sequence] = None,
                             require_outcome: bool = False
                             ) -> list:
    """``[(treatment_pos, reference_pos)]`` pairs with real contrast support.

    ``meta`` is the source's per-position item metadata in dataset order (the
    same list ``train.dataset.contrast_pairs`` consumes).  Only candidate pairs
    whose two supports share an outcome tau -- or, when both sides carry
    verified canonical IDs, at least one common verified HDV vehicle -- are
    kept; slot intersection alone is NOT HDV eligibility for identified
    sources.  An unpaired item is never removed from the epoch (the sampler
    appends pair partners, it does not filter the primary stream).

    ``require_outcome=True`` (the pair-dense option, opt-in) keeps ONLY pairs
    that share at least one realised outcome tau, so the pair-aware sampler
    forces outcome-carrying siblings into every batch instead of spending its
    forced pair on an HDV-only sibling.  This does not fabricate labels: a pair
    whose treatment arm carries no t0 outcome row is simply not forced, and a
    source with no outcome-supported pair yields an empty pool (fail closed
    upstream).  The default ``False`` preserves the historical
    outcome-OR-HDV eligibility exactly.
    """
    from train.dataset import contrast_pairs

    if support_fn is None:
        support_fn = source_support_fn(source)
    if candidate_pairs is None:
        candidate_pairs = contrast_pairs(meta)
    pool: list = []
    for a_pos, b_pos in candidate_pairs:
        shared = shared_support(support_fn(int(a_pos)), support_fn(int(b_pos)))
        if require_outcome:
            keep = bool(shared["outcome"].any())
        else:
            keep = has_support(shared)
        if keep:
            pool.append((int(a_pos), int(b_pos)))
    return pool


__all__ = [
    "item_support_masks", "support_ids", "common_verified_ids",
    "align_hdv_ids", "shared_support", "has_support", "shared_outcome_mask",
    "shared_hdv_mask", "source_support_fn", "build_eligible_pair_pool",
]
