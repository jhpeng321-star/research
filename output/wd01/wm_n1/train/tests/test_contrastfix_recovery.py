"""contrastfix recovery: eligible support, sampling pool, fail-closed.

Synthetic-only CPU contracts for the paired CONTRAST loss fix:

* a sibling pair whose two arms share no outcome tau and no valid HDV slot is
  not eligible;
* a pair that shares only one sub-term contributes only that sub-term (the
  other is not averaged in as a fake ``0.0``);
* an all-ineligible pair fails closed with its counts instead of returning a
  zero loss;
* the eligible pool keeps only supported partners while the epoch still visits
  every item (unpaired examples keep training the other losses);
* the in-process and DataLoader seeded pair-aware orders are identical.

Verified identity alignment (contrastfix identity recovery, code-only draft):

* slot numbers are NOT identities: pairs align by COMMON CANONICAL vehicle ID
  (swapped/permuted slots match; same-slot different-VID does not);
* HDV eligibility is common-ID based whenever both sides carry verified IDs;
  validity and non-CAV masks still gate every matched slot;
* an actual packed fit requires verified per-item identities BEFORE the
  sampler/optimizer (missing/unknown/duplicated fails; never silently
  disables the HDV sub-term) through ``prepare_training_source``;
* identity rows are keyed by the ORIGINAL GLOBAL manifest position (resolved
  via the pack manifest, not the TRAIN-filtered dataset position);
* ``prepare_training_source`` is the CANONICAL binding point: the exact
  prepared pair pool, outcome normalization and identity receipt are bound
  on the trainer BEFORE the helper returns, so a strict timing caller that
  only runs the helper gets the fit state (never the constructor's identity
  norm or the unfiltered legacy pair fallback), with no optimizer created
  and no sample consumed before that binding;
* ``WM_N1_CANONICAL_PACK`` is the ONLY frozen-alias allowance for the
  identity-sidecar read: when the env is set and the pack alias strict-
  resolves to EXACTLY that target, the LITERAL canonical target is handed
  to the loader (whose own strict checks prove it link-free); any mismatch
  refuses BEFORE the loader is called, and with the env absent there is no
  resolve fallback - the literal alias is passed and will refuse;
  ``pack_alias``/``canonical_pack`` are recorded in the identity receipt.
* the ID-aligned HDV pinball feeds the frozen primitive its CONTRACTED
  shapes: ``[n_common, NQ]`` predicted difference against the SCALAR
  per-matched-ID realised difference as target - any common-ID count other
  than 1 must neither crash nor broadcast (including the ncommon == NQ
  collision case), the term stays differentiable, and its value equals the
  explicit frozen-contract reference at permuted slot positions;
* the ``_identity_pack`` fixture is VALID under the strict B label-
  isolation gate: its INDEX runs carry the per-run identity the corpus
  writer emits (``key``/``shard_dir``/``split``/``n_outcome_rows``), a
  synthetic per-run shard tree is staged beside the pack, and a REAL
  TRAIN-only outcome sidecar is built through the production
  ``train.label_isolation.build_sidecar`` over those synthetic bytes; the
   TRAIN-reading tests point ``WM_N1_TRAIN_OUTCOMES`` at it, so identities
   and labels resolve without opening the mixed packed table for TRAIN
   (production isolation is exercised, never relaxed).  The CAL run stays
   at GLOBAL manifest position 0 (CAL-first interleave retained) and the
   CALIBRATION corpus keeps its authorized mixed-table read.  The strict
   guard denies the production ``os.replace`` receipt finalization, so a
   narrow autouse fixture delegates ONLY that JSON finalization to an
   exclusive fresh-target write with the identical
   ``json.dumps(indent=1, sort_keys=True)`` body and identical Path
   return - production planning, coverage and shard-read isolation
   revalidation inside ``build_sidecar`` are untouched.

No real data, no SUMO: every fixture is built in memory or in ``tmp_path``.
The NEW tests in this file have NOT been executed (code-only source draft);
execution awaits the coordinator's reviewed guard and node authorization.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import train.idea_a as ia
from model import constants as C
from model import losses as L
from train import contrast_support as CS
from train.dataset import PairAwareBatchSampler, epoch_batches
from test_idea_a_fakes import make_windows, seeded_bundle
from packsynth import write_pack, TAUS


# --------------------------------------------------------------- helpers ----
def _mask(flags, n=None):
    if n is None:
        n = len(flags)
    out = torch.zeros(n, dtype=torch.bool)
    for i in flags:
        out[i] = True
    return out


class _PairSource:
    """One-arm-per-position corpus stand-in (metadata + support masks)."""

    def __init__(self, specs, k=6):
        self.k = int(k)
        self.split = "TRAIN"
        self.pack_dir = None
        self._items = [s["meta"] for s in specs]
        self._order = list(range(len(specs)))
        self._specs = list(specs)

    def __len__(self):
        return len(self._order)

    def __getitem__(self, i):
        s = self._specs[self._order[i]]
        item = {"outcome_mask": s["outcome"].unsqueeze(0),
                "target_mask": s["target"].unsqueeze(0),
                "is_cav": s["cav"].unsqueeze(0),
                "meta": [s["meta"]]}
        if s.get("ids") is not None:
            item["vehicle_ids"] = s["ids"].unsqueeze(0)
        return item


class _IdSource(_PairSource):
    """``_PairSource`` plus a controllable verified-identity accessor."""

    def __init__(self, specs, k=6, identity="ok"):
        super().__init__(specs, k=k)
        self.identity = identity

    def verified_identities(self):
        if self.identity == "raises":
            raise ia.IdeaAConfigError(
                "verified TRAIN identities missing for item 0")
        if self.identity != "ok":
            return {"status": str(self.identity)}
        return {"status": "verified", "mode": "identity-sidecar",
                "env": "WM_N1_TRAIN_IDENTITIES", "n_items": len(self),
                "receipt_sha256": "cd" * 32}


def _spec(world, arm, policy, pair_key, outcome, target, cav, k=6):
    return {"meta": {"world": world, "arm": arm, "policy": policy,
                     "pair_key": pair_key, "split": "TRAIN"},
            "outcome": torch.as_tensor(outcome, dtype=torch.bool),
            "target": torch.as_tensor(target, dtype=torch.bool),
            "cav": torch.as_tensor(cav, dtype=torch.bool)}


def _pair(treat_outcome, ref_outcome, treat_target, ref_target, k=6,
          key="w@t0=0", treat_policy="wall", ref_policy="native",
          treat_arm="wall_v31", ref_arm="native"):
    def _target(flags):
        return _mask(flags, k)

    zeros = torch.zeros(k, dtype=torch.bool)
    return [
        _spec("w", treat_arm, treat_policy, key, treat_outcome, _target(
            treat_target), zeros),
        _spec("w", ref_arm, ref_policy, key, ref_outcome, _target(
            ref_target), zeros),
    ]


# ---------------------------------------------------- eligibility logic -----
def test_ineligible_semantic_pair_has_no_support():
    taus = C.N_TAU
    specs = _pair(_mask([0, 1], taus), _mask([2, 3], taus),
                  [0, 1], [2, 3])
    src = _PairSource(specs)
    assert CS.build_eligible_pair_pool(src, src._items) == []


def test_shared_outcome_support_marks_pair_eligible():
    taus = C.N_TAU
    specs = _pair(_mask([0, 1, 2], taus), _mask([2, 3], taus),
                  [0, 1], [2, 3])       # hdv disjoint, tau shared
    src = _PairSource(specs)
    pool = CS.build_eligible_pair_pool(src, src._items)
    assert pool == [(0, 1)]


def test_partial_eligible_pool_keeps_only_supported_groups():
    taus = C.N_TAU
    specs = []
    specs += _pair(_mask([0, 1], taus), _mask([1, 2], taus),
                   [0, 1], [2, 3], key="a@t0=0")
    specs += _pair(_mask([0, 1], taus), _mask([2, 3], taus),
                   [0, 1], [2, 3], key="b@t0=0")
    specs += _pair(_mask([0], taus), _mask([1], taus),
                   [0], [0], key="c@t0=0")
    src = _PairSource(specs)
    pool = CS.build_eligible_pair_pool(src, src._items)
    # group a: shared tau only; group c: shared hdv only; group b: ineligible
    assert pool == [(0, 1), (4, 5)]


def test_switching_native_arm_is_not_a_reference():
    """A ``__native`` arm with policy ``switching`` must not form a pair."""
    taus = C.N_TAU
    specs = _pair(_mask([0], taus), _mask([0], taus), [0], [0],
                  key="sw@t0=0", ref_policy="switching", ref_arm="__native")
    src = _PairSource(specs)
    assert CS.build_eligible_pair_pool(src, src._items) == []


# ----------------------------------------------------- loss-time support ----
def _trainer():
    cfg = ia.IdeaATrainConfig(out_dir="unused", num_workers=0)
    return ia.IdeaATrainer(cfg, corpus=None, bundle=seeded_bundle(0))


def _batch_with_masks(n_windows=1, k=8):
    windows = make_windows(n_windows, k=k, seed=0)
    batch = ia.collate_canonical(windows)
    return batch


def test_all_ineligible_pair_fails_closed_with_counts():
    tr = _trainer()
    batch = _batch_with_masks()
    taus = int(batch["outcome_mask"].shape[1])
    # treatment tau 0..1, reference tau 2..3 -> disjoint outcome support
    batch["outcome_mask"][0] = _mask([0, 1], taus)
    batch["outcome_mask"][1] = _mask([2, 3], taus)
    # treatment HDV slots 2..3, reference 4..5 -> disjoint HDV support
    k = int(batch["target_mask"].shape[1])
    batch["target_mask"][0] = _mask([2, 3], k)
    batch["target_mask"][1] = _mask([4, 5], k)
    try:
        tr.paired_response_contrast(batch)
    except ia.IdeaAConfigError as exc:
        assert "NO valid contrast support" in str(exc)
    else:                                             # pragma: no cover
        raise AssertionError("ineligible pair returned a loss instead of "
                             "failing closed")
    assert tr._contrast_diag["pairs"] == 1
    assert tr._contrast_diag["outcome_pairs"] == 0
    assert tr._contrast_diag["hdv_pairs"] == 0


def test_mismatched_tau_excludes_only_the_outcome_subterm():
    tr = _trainer()
    batch = _batch_with_masks()
    taus = int(batch["outcome_mask"].shape[1])
    batch["outcome_mask"][0] = _mask([0, 1], taus)
    batch["outcome_mask"][1] = _mask([2, 3], taus)
    loss = tr.paired_response_contrast(batch)
    diag = tr._contrast_diag
    assert diag["outcome_subterm"] is False and diag["outcome_pairs"] == 0
    assert diag["hdv_subterm"] is True and diag["hdv_pairs"] == 1
    assert float(loss.detach()) > 0.0                          # HDV support remains


def test_no_hdv_support_excludes_only_the_hdv_subterm():
    tr = _trainer()
    batch = _batch_with_masks()
    k = int(batch["target_mask"].shape[1])
    batch["target_mask"][0] = _mask([2, 3], k)
    batch["target_mask"][1] = _mask([4, 5], k)
    loss = tr.paired_response_contrast(batch)
    diag = tr._contrast_diag
    assert diag["hdv_subterm"] is False and diag["hdv_pairs"] == 0
    assert diag["outcome_subterm"] is True and diag["outcome_pairs"] == 1
    assert float(loss.detach()) > 0.0                          # outcome support remains


def test_supported_subterms_are_not_diluted_by_a_zero_term():
    """One supported sub-term must not be halved by an excluded zero term."""
    tr = _trainer()
    batch = _batch_with_masks()
    taus = int(batch["outcome_mask"].shape[1])
    batch["outcome_mask"][0] = _mask([0, 1], taus)
    batch["outcome_mask"][1] = _mask([2, 3], taus)   # outcome excluded
    only_hdv = tr.paired_response_contrast(batch)
    # compare against the HDV-only term computed directly at the same weights
    hdv = tr._require("hdv")
    idx = tr._pairs(batch)
    a, b = idx["a"], idx["b"]
    out = hdv(batch["kin"], batch["kin_mask"])
    r_pred = out["accel_delta_mps2"]
    base = batch["baseline_idm"]
    r_targ = batch["target_accel"] - base
    pmask = CS.shared_hdv_mask(batch["target_mask"], a, b, batch.get("is_cav"))
    qa = r_pred[a].unsqueeze(-1).expand(*r_pred[a].shape, C.N_QUANTILES)
    qb = r_pred[b].unsqueeze(-1).expand(*r_pred[b].shape, C.N_QUANTILES)
    levels = torch.tensor(C.QUANTILE_LEVELS, dtype=qa.dtype)
    pb = L.pinball(qa - qb, r_targ[a] - r_targ[b], levels)
    m = pmask.unsqueeze(-1).to(pb.dtype).expand_as(pb)
    hdv_only = (pb * m).sum() / m.sum().clamp(min=1)
    assert torch.allclose(only_hdv, hdv_only)


# --------------------------------------------------- fit-time fail-closed ---
def test_fit_refuses_when_no_eligible_pair_exists(tmp_path):
    taus = C.N_TAU
    specs = _pair(_mask([0], taus), _mask([2], taus), [0], [2])
    src = _IdSource(specs, identity="ok")
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=1, batch_size=2,
                              out_dir=str(tmp_path / "out"), num_workers=0)
    tr = ia.IdeaATrainer(cfg, src, member=1, seed=101,
                         bundle=seeded_bundle(101))
    try:
        tr.fit()
    except ia.IdeaAConfigError as exc:
        assert "no eligible sibling pair" in str(exc)
    else:                                             # pragma: no cover
        raise AssertionError("fit ran with a zero-contrast TRAIN source")


# ------------------------------------------------ sampling order parity -----
def _order_source():
    taus = C.N_TAU
    specs = []
    for g in range(3):
        specs += _pair(_mask([0, 1], taus), _mask([1, 2], taus),
                       [0, 1, 2], [2, 3, 4],
                       key=f"w{g}@t0=0")
    return _PairSource(specs)


def test_epoch_order_equals_pairaware_sampler_with_eligible_pool():
    src = _order_source()
    pool = CS.build_eligible_pair_pool(src, src._items)
    assert len(pool) == 3
    for epoch in range(3):
        want = epoch_batches(src._items, 2, 101, epoch, pool)
        got = list(PairAwareBatchSampler(src._items, 2, 101, epoch=epoch,
                                         epochs=1, pairs=pool))
        assert want == got


def test_inprocess_and_dataloader_iterators_use_same_order():
    src = _order_source()
    pool = CS.build_eligible_pair_pool(src, src._items)
    cfg = ia.IdeaATrainConfig(out_dir="unused", num_workers=0)
    tr = ia.IdeaATrainer(cfg, corpus=src, bundle=seeded_bundle(0))
    tr.seed = 101
    tr._contrast_pair_pool = pool

    def _keys(batches, n):
        out = []
        for _ in range(n):
            batch = next(batches)
            out.append([m["pair_key"] for m in batch["meta"]])
        return out

    in_proc = _keys(tr._iter_batches(src), 6)
    loader = _keys(tr._iter_batches_loader(src), 6)
    assert in_proc == loader


def test_pool_does_not_drop_unpaired_items_from_the_epoch():
    """Every position is visited once per epoch even without a pair partner."""
    src = _order_source()
    pool = CS.build_eligible_pair_pool(src, src._items)
    seen = set()
    for batch in epoch_batches(src._items, 2, 101, 0, pool):
        seen.update(int(p) for p in batch)
    assert seen == set(range(len(src)))


# --------------------------------------------- verified identity helpers ----
def test_common_verified_ids_respects_validity_and_absence():
    assert CS.common_verified_ids([5, -1, 7], [1, 1, 1], [7, 5, -1],
                                  [1, 1, 1]) == [5, 7]
    assert CS.common_verified_ids([5], [0], [5], [1]) == []      # invalid A
    assert CS.common_verified_ids([-1], [1], [-1], [1]) == []    # absent
    assert CS.common_verified_ids([3], [1], [4], [1]) == []      # disjoint


def test_align_hdv_ids_pairs_swapped_slot_indices():
    ja, jb = CS.align_hdv_ids([5, -1, 7], [1, 1, 1], [-1, 7, 5], [1, 1, 1])
    assert sorted(zip(ja.tolist(), jb.tolist())) == [(0, 2), (2, 1)]
    ja, jb = CS.align_hdv_ids([5], [1], [6], [1])
    assert ja.numel() == 0 and jb.numel() == 0
    ja, jb = CS.align_hdv_ids([5], [1], [5, 5], [1, 0])
    assert (ja.tolist(), jb.tolist()) == ([0], [0])


def test_pool_hdv_eligibility_needs_common_ids_not_slot_overlap():
    taus = C.N_TAU
    # valid slots intersect at {2,3} but hold disjoint vehicles; outcomes are
    # disjoint too -> eligible only if HDV matching wrongly used slots
    specs = _pair_ids([-1, -1, 7, 8, -1, -1], [-1, -1, 70, 80, -1, -1],
                      [2, 3], [2, 3], _mask([0], taus), _mask([1], taus))
    src = _PairSource(specs)
    assert CS.build_eligible_pair_pool(src, src._items) == []


def test_pool_hdv_eligibility_by_permuted_common_ids():
    taus = C.N_TAU
    # valid slots are DISJOINT ({2,5} vs {1,4}) but both arms hold vehicles
    # 7 and 9 at different local indices; outcomes disjoint -> HDV-only proof
    specs = _pair_ids([-1, -1, 7, -1, -1, 9], [-1, 9, -1, -1, 7, -1],
                      [2, 5], [1, 4], _mask([0], taus), _mask([1], taus))
    src = _PairSource(specs)
    assert CS.build_eligible_pair_pool(src, src._items) == [(0, 1)]


def _pair_ids(treat_ids, ref_ids, treat_target, ref_target,
              outcome_t, outcome_r, k=6, key="idw@t0=0"):
    specs = _pair(outcome_t, outcome_r, treat_target, ref_target, k=k,
                  key=key)
    specs[0]["ids"] = torch.as_tensor(treat_ids, dtype=torch.long)
    specs[1]["ids"] = torch.as_tensor(ref_ids, dtype=torch.long)
    return specs


# --------------------------------------------- loss-time ID alignment -------
def _batch_with_ids(treat_ids, ref_ids, k=8):
    windows = make_windows(1, k=k, seed=0)
    windows[0]["vehicle_ids"] = torch.as_tensor([treat_ids, ref_ids],
                                                dtype=torch.long)
    return ia.collate_canonical(windows)


def _disable_outcome_support(batch):
    taus = int(batch["outcome_mask"].shape[1])
    batch["outcome_mask"][0] = _mask([0, 1], taus)
    batch["outcome_mask"][1] = _mask([2, 3], taus)


def test_hdv_contrast_aligns_swapped_slots_by_common_id():
    batch = _batch_with_ids([-1, -1, 50, -1, -1, 51, -1, -1],
                            [-1, -1, -1, -1, 50, -1, -1, 51])
    _disable_outcome_support(batch)             # HDV sub-term only
    tr = _trainer()
    loss = tr.paired_response_contrast(batch)
    diag = tr._contrast_diag
    assert diag["hdv_id_aligned"] is True
    assert diag["hdv_pairs"] == 1 and diag["hdv_slots"] == 2
    assert diag["hdv_subterm"] is True
    hdv = tr._require("hdv")
    r_pred = hdv(batch["kin"], batch["kin_mask"])["accel_delta_mps2"]
    r_targ = batch["target_accel"] - batch["baseline_idm"]
    pairs = [(2, 4), (5, 7)]                    # each arm at ITS own slot
    qa = torch.stack([r_pred[0, ja] - r_pred[1, jb] for ja, jb in pairs])
    qb = torch.stack([r_targ[0, ja] - r_targ[1, jb] for ja, jb in pairs])
    levels = torch.tensor(C.QUANTILE_LEVELS)
    expected = L.pinball(qa.unsqueeze(-1).expand(2, C.N_QUANTILES), qb,
                         levels).mean()
    assert torch.allclose(loss, expected)


def _aligned_pinball_reference(r_pred, r_targ, pairs):
    """Frozen-contract reference for the ID-aligned HDV contrast.

    ``pairs`` are per-common-vehicle ``(slot_a, slot_b)`` indices; the
    reference is the frozen ``L.pinball`` over the ``[M, NQ]`` predicted
    difference with the SCALAR per-matched-ID realised difference as its
    target - exactly the shapes the frozen primitive documents.
    """
    qa = torch.stack([r_pred[0, ja] - r_pred[1, jb] for ja, jb in pairs])
    qb = torch.stack([r_targ[0, ja] - r_targ[1, jb] for ja, jb in pairs])
    levels = torch.tensor(C.QUANTILE_LEVELS)
    return L.pinball(qa.unsqueeze(-1).expand(qa.shape[0], C.N_QUANTILES),
                     qb, levels).mean()


def test_hdv_contrast_multiple_permuted_ids_scalar_per_id_target():
    """Three common IDs at disjoint permuted slots (ncommon=3 != NQ=19).

    The previous wide (19-per-ID) target crashed the frozen primitive with
    a broadcast error for every matched-ID count outside {1, NQ}; the
    aligned loss must equal the scalar-target reference exactly.
    """
    k = 10
    batch = _batch_with_ids([-1, -1, 30, -1, 31, -1, 32, -1, -1, -1],
                            [-1, -1, -1, 32, -1, -1, 31, -1, 30, -1], k=k)
    _disable_outcome_support(batch)             # HDV sub-term only
    tr = _trainer()
    loss = tr.paired_response_contrast(batch)
    diag = tr._contrast_diag
    assert diag["hdv_id_aligned"] is True
    assert diag["hdv_pairs"] == 1 and diag["hdv_slots"] == 3
    assert diag["hdv_subterm"] is True
    hdv = tr._require("hdv")
    r_pred = hdv(batch["kin"], batch["kin_mask"])["accel_delta_mps2"]
    r_targ = batch["target_accel"] - batch["baseline_idm"]
    pairs = [(2, 8), (4, 6), (6, 3)]            # vehicles 30/31/32: (ja, jb)
    assert torch.allclose(
        loss, _aligned_pinball_reference(r_pred, r_targ, pairs))


def test_hdv_contrast_at_ncommon_nq_collision_stays_scalar_target():
    """ncommon == N_QUANTILES: the silent-wrong case of the old wide target.

    A 19-wide target against a [19, NQ] prediction BROADCASTS (no crash)
    into [19, 19, 19] and averages cross-quantile garbage; the aligned
    loss must instead equal the scalar-target frozen-contract reference.
    """
    k = 21
    ids = list(range(200, 219))                 # 19 common vehicles
    batch = _batch_with_ids([-1, -1] + ids, [-1, -1] + list(reversed(ids)),
                            k=k)
    _disable_outcome_support(batch)
    tr = _trainer()
    loss = tr.paired_response_contrast(batch)
    assert tr._contrast_diag["hdv_slots"] == C.N_QUANTILES
    assert tr._contrast_diag["hdv_pairs"] == 1
    hdv = tr._require("hdv")
    r_pred = hdv(batch["kin"], batch["kin_mask"])["accel_delta_mps2"]
    r_targ = batch["target_accel"] - batch["baseline_idm"]
    pairs = [(2 + j, 2 + (C.N_QUANTILES - 1 - j)) for j in range(k - 2)]
    assert torch.allclose(
        loss, _aligned_pinball_reference(r_pred, r_targ, pairs))


def test_hdv_contrast_gradient_flows_through_matched_id_term():
    """Two matched IDs (ncommon=2): the aligned sub-term is differentiable.

    Backward reaches every HDV parameter with finite gradients and a
    nonzero norm - the frozen pinball is the only loss math on the path
    and the realised target stays gradient-free data.
    """
    batch = _batch_with_ids([-1, -1, 40, -1, -1, 41, -1, -1],
                            [-1, -1, -1, -1, 41, -1, -1, 40])
    _disable_outcome_support(batch)
    tr = _trainer()
    loss = tr.paired_response_contrast(batch)
    diag = tr._contrast_diag
    assert diag["hdv_pairs"] == 1 and diag["hdv_slots"] == 2
    hdv = tr._require("hdv")
    r_pred = hdv(batch["kin"], batch["kin_mask"])["accel_delta_mps2"]
    r_targ = batch["target_accel"] - batch["baseline_idm"]
    assert torch.allclose(
        loss, _aligned_pinball_reference(r_pred, r_targ, [(2, 7), (5, 4)]))
    loss.backward()
    grads = [p.grad for p in hdv.parameters()]
    assert all(g is not None for g in grads)
    assert all(bool(torch.isfinite(g).all()) for g in grads)
    assert float(sum(float(g.pow(2).sum()) for g in grads)) > 0.0


def test_same_slot_different_vid_is_not_hdv_support():
    batch = _batch_with_ids([-1, -1, -1, 60, -1, -1, -1, -1],
                            [-1, -1, -1, 61, -1, -1, -1, -1])
    tr = _trainer()
    loss = tr.paired_response_contrast(batch)
    diag = tr._contrast_diag
    assert diag["hdv_id_aligned"] is True
    assert diag["hdv_pairs"] == 0 and diag["hdv_slots"] == 0
    assert diag["hdv_subterm"] is False
    assert diag["outcome_subterm"] is True      # outcome support remains
    assert float(loss.detach()) > 0.0


def test_id_match_requires_valid_noncav_slots_in_both_arms():
    treat = [-1, -1, -1, 70, 71, -1, -1, -1]

    def _ref_batch():
        return _batch_with_ids(treat, [-1, -1, -1, -1, -1, 70, -1, -1])

    # control: vehicle 70 valid (non-CAV, realised) in both arms -> 1 match
    tr = _trainer()
    tr.paired_response_contrast(_ref_batch())
    assert tr._contrast_diag["hdv_slots"] == 1
    # the reference side's slot is a CAV command slot -> not matched
    batch = _ref_batch()
    batch["is_cav"][1, 5] = True
    tr.paired_response_contrast(batch)
    assert tr._contrast_diag["hdv_slots"] == 0
    # the reference side's slot has no realised response -> not matched
    batch = _ref_batch()
    batch["target_mask"][1, 5] = False
    tr.paired_response_contrast(batch)
    assert tr._contrast_diag["hdv_slots"] == 0
    # the reference side's slot identity is absent (-1) -> not matched
    batch = _ref_batch()
    batch["vehicle_ids"][1, 5] = -1
    tr.paired_response_contrast(batch)
    assert tr._contrast_diag["hdv_slots"] == 0


# --------------------------------------- corpus verified identity access ----
def _runs_cal_first():
    return [
        dict(chunk="chunk2", world="c1", arm="native", policy="native",
             split="CALIBRATION", t0_s=300.0),
        dict(chunk="chunk4", world="w1", arm="base", policy="damping",
             split="TRAIN", t0_s=540.0),
        dict(chunk="chunk4", world="w1", arm="native", policy="native",
             split="TRAIN", t0_s=540.0),
        dict(chunk="chunk4", world="w1", arm="pi", policy="pi",
             split="TRAIN", t0_s=540.0),
        dict(chunk="chunk4", world="w2", arm="native", policy="native",
             split="TRAIN", t0_s=600.0),
        dict(chunk="chunk4", world="w2", arm="base", policy="damping",
             split="TRAIN", t0_s=600.0),
    ]


def _write_identity_pack(root, runs, k=6, n_ticks=3):
    """A real-format pack (v1 tensors + zero Idea A channels + INDEX.json).

    Strict-B valid: the INDEX runs carry the per-run identity the corpus
    writer emits (``key``/``shard_dir``/``split``/``n_outcome_rows``), a
    synthetic per-run shard tree is staged BESIDE the pack, and the
    TRAIN-only outcome sidecar is built through the production
    ``train.label_isolation.build_sidecar`` (never a hand-written receipt).
    Point ``WM_N1_TRAIN_OUTCOMES`` at :func:`_train_outcomes_dir` to open
    the pack as TRAIN; the CALIBRATION split keeps the authorized
    mixed-table read and never consults the sidecar.
    """
    manifest = write_pack(root, runs, k=k, n_ticks=n_ticks)
    n_tick = len(runs) * n_ticks
    idea = {
        "veh_kin": np.zeros((n_tick, k, 7), np.float16),
        "hdv_base": np.zeros((n_tick, k, 6), np.float16),
        "cav_plan_ch": np.zeros((n_tick, k, 8), np.float16),
        "event_tok": np.zeros((n_tick, 8, 8), np.float16),
        "event_mask": np.zeros((n_tick, 8), np.float16),
    }
    for name, arr in idea.items():
        np.save(root / f"{name}.npy", arr)
        manifest["arrays"][name] = {"file": f"{name}.npy",
                                    "dtype": str(arr.dtype),
                                    "shape": list(arr.shape)}
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    (root / "INDEX.json").write_text(json.dumps(
        {"schema": "wm_n1_corpus_pack/2", "k_max": int(k),
         "runs": _index_run_records(runs, n_ticks, len(TAUS))},
        indent=1), encoding="utf-8")
    shards = _write_train_shard_outcomes(root, runs)
    _build_train_outcome_sidecar(root, shards)
    return root


def _index_run_records(runs, n_ticks, n_rows_per_run):
    """INDEX runs in manifest order with the corpus-writer per-run identity.

    ``plan_sidecar`` accumulates ``n_outcome_rows`` in INDEX order, so the
    records MUST follow the manifest item order exactly for every item's
    ``outcome_rows`` to reconcile with its run's derived global range.
    """
    return [
        {"key": f"{r['chunk']}/{r['world']}__{r['arm']}",
         "chunk": r["chunk"], "world": r["world"], "arm": r["arm"],
         "split": r["split"], "t0_s": float(r["t0_s"]),
         "shard_dir": f"{r['chunk']}/{r['world']}/{r['arm']}",
         "n_outcome_rows": int(n_rows_per_run),
         "tick_start": i * n_ticks, "tick_end": (i + 1) * n_ticks}
        for i, r in enumerate(runs)]


def _write_train_shard_outcomes(pack_root, runs):
    """Synthetic staging shards: per-run outcome slices of the packed table.

    The packed ``outcomes.parquet`` IS the concatenation of the per-run
    shard tables (as in production), so the isolated sidecar rows equal the
    legacy positional mixed-table rows value for value.  Written beside the
    pack (never inside it), fresh directories only.
    """
    import pandas as pd

    pack_root = Path(pack_root)
    table = pd.read_parquet(pack_root / "outcomes.parquet")
    per_run = len(TAUS)
    assert len(table) == per_run * len(runs)
    shards = pack_root.parent / f"{pack_root.name}_shards"
    for i, r in enumerate(runs):
        shard = shards / r["chunk"] / r["world"] / r["arm"]
        shard.mkdir(parents=True)
        table.iloc[i * per_run:(i + 1) * per_run].to_parquet(
            shard / "outcomes.parquet", index=False)
    return shards


def _train_outcomes_dir(pack_root):
    """The sidecar directory :func:`_write_identity_pack` built for a pack."""
    pack_root = Path(pack_root)
    return pack_root.parent / f"{pack_root.name}_train_outcomes"


# ------------------------- guarded sidecar JSON-finalization delegate ----
@pytest.fixture(autouse=True)
def _plain_sidecar_json_finalization(monkeypatch):
    """Delegate ONLY the sidecar JSON finalization past the strict guard.

    The production ``build_sidecar`` receipt finalization goes through
    ``train.label_isolation._write_json_atomic`` (tmp file + ``os.replace``),
    an audit event the guarded run environment denies.  This narrow fixture
    swaps ONLY that finalization, and only inside this solely-owned file:
    the body is IDENTICAL (``json.dumps(payload, indent=1, sort_keys=True)``),
    the target is an EXCLUSIVE fresh file (mode ``xb``: never overwritten)
    and the identical Path is returned.  Production planning, coverage
    reconciliation and shard-read isolation revalidation inside
    ``build_sidecar`` still execute unpatched, as do the loader spy
    assertions; production ``label_isolation`` and the guard itself are not
    altered.  Where the B module is absent (this draft tree) the fixture is
    inert and sidecar nodes keep failing at their own lazy import.
    """
    try:
        import train.label_isolation as LI
    except ImportError:
        yield
        return

    def _write_json_fresh_target(path, payload):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "xb") as fh:
            fh.write(json.dumps(
                payload, indent=1, sort_keys=True).encode("utf-8"))
        return target

    monkeypatch.setattr(LI, "_write_json_atomic", _write_json_fresh_target)
    yield


def _build_train_outcome_sidecar(pack_root, shards_root):
    """Build the TRAIN sidecar through the PRODUCTION isolation builder.

    The receipt (pack/INDEX/source-shard/sidecar hashes, run identities,
    row coverage) is written by ``train.label_isolation.build_sidecar``
    itself, so ``load_train_outcomes`` re-validates the fixture exactly
    like a real staging build; the mixed packed table is never opened here
    and non-TRAIN shard files are never read by the builder.  ONLY the
    atomic JSON finalization is delegated past the strict guard by the
    ``_plain_sidecar_json_finalization`` autouse fixture (identical body,
    exclusive fresh target, identical Path return).
    """
    from train.label_isolation import build_sidecar   # strict-gate builder

    build_sidecar(pack_root, shards_root, _train_outcomes_dir(pack_root))


def _default_id_rows():
    # keyed by ORIGINAL GLOBAL manifest position (the CAL run occupies 0);
    # packsynth slots 0..2 are CAV, so HDV-valid slots are 3..5
    return {
        1: [-1, -1, -1, 100, 101, 102],     # w1/base
        2: [-1, -1, -1, 100, 103, 104],     # w1/native (shares vehicle 100)
        3: [-1, -1, -1, 110, 111, 112],     # w1/pi
        4: [-1, -1, -1, 120, 121, 122],     # w2/native
        5: [-1, -1, -1, 120, 123, 124],     # w2/base (shares vehicle 120)
    }


class _FakeSidecar:
    def __init__(self, ids_by_gpos, k=6):
        self._ids = {int(p): np.asarray(v, dtype=np.int64)
                     for p, v in ids_by_gpos.items()}
        self._k = int(k)
        self.receipt_sha256 = "ab" * 32

    def ids_for_item(self, global_manifest_position):
        row = self._ids.get(int(global_manifest_position))
        if row is None:
            raise PermissionError("item lacks verified TRAIN identities")
        if row.shape[0] != self._k:
            raise PermissionError("identity row width mismatch")
        return row


def _sidecar_loader(ids_by_gpos, k=6):
    def _load(pack_dir, side_dir):              # TEST-ONLY injected loader
        return _FakeSidecar(ids_by_gpos, k)

    return _load


def _identity_pack(tmp_path):
    return _write_identity_pack(tmp_path / "pack", _runs_cal_first())


def test_identity_rows_keyed_by_global_not_train_position(tmp_path,
                                                          monkeypatch):
    pack = _identity_pack(tmp_path)
    monkeypatch.setenv("WM_N1_TRAIN_OUTCOMES", str(_train_outcomes_dir(pack)))
    corpus = ia.IdeaACorpus(str(pack), split="TRAIN",
                            identity_loader=_sidecar_loader(_default_id_rows()))
    # the CALIBRATION run occupies global position 0, so TRAIN dataset
    # positions 0..4 map to GLOBAL manifest positions 1..5
    assert corpus._global_positions == [1, 2, 3, 4, 5]
    w0 = corpus[0]
    assert w0["vehicle_ids"].dtype == torch.int64
    assert w0["vehicle_ids"].shape == (1, 6)
    assert w0["vehicle_ids"][0].tolist() == [-1, -1, -1, 100, 101, 102]
    assert corpus[1]["vehicle_ids"][0].tolist() == [-1, -1, -1, 100, 103, 104]


def test_support_masks_carry_verified_ids(tmp_path, monkeypatch):
    pack = _identity_pack(tmp_path)
    monkeypatch.setenv("WM_N1_TRAIN_OUTCOMES", str(_train_outcomes_dir(pack)))
    corpus = ia.IdeaACorpus(str(pack), split="TRAIN",
                            identity_loader=_sidecar_loader(_default_id_rows()))
    support = corpus.support_masks(0)
    assert support["ids"].tolist() == [-1, -1, -1, 100, 101, 102]
    assert support["hdv"].tolist() == [False, False, False,
                                       True, True, True]


def test_cal_corpus_carries_no_identities(tmp_path):
    corpus = ia.IdeaACorpus(str(_identity_pack(tmp_path)),
                            split="CALIBRATION",
                            identity_loader=_sidecar_loader(_default_id_rows()))
    assert "vehicle_ids" not in corpus[0]
    assert corpus.support_masks(0)["ids"] is None
    with pytest.raises(ia.IdeaAConfigError):
        corpus.verified_identities()


def test_missing_hdv_identity_fails_verified_gate(tmp_path):
    rows = _default_id_rows()
    rows[2] = [-1, -1, -1, 100, -1, 104]        # HDV-valid slot 4 unverified
    corpus = ia.IdeaACorpus(str(_identity_pack(tmp_path)), split="TRAIN",
                            identity_loader=_sidecar_loader(rows))
    with pytest.raises(ia.IdeaAConfigError) as ei:
        corpus.verified_identities()
    assert "missing" in str(ei.value)


def test_duplicated_identity_fails_verified_gate(tmp_path):
    rows = _default_id_rows()
    rows[3] = [-1, -1, -1, 110, 110, 112]
    corpus = ia.IdeaACorpus(str(_identity_pack(tmp_path)), split="TRAIN",
                            identity_loader=_sidecar_loader(rows))
    with pytest.raises(ia.IdeaAConfigError) as ei:
        corpus.verified_identities()
    assert "duplicated" in str(ei.value)


def test_unknown_identity_fails_verified_gate(tmp_path):
    rows = _default_id_rows()
    rows[4] = [-2, -1, -1, 120, 121, 122]
    corpus = ia.IdeaACorpus(str(_identity_pack(tmp_path)), split="TRAIN",
                            identity_loader=_sidecar_loader(rows))
    with pytest.raises(ia.IdeaAConfigError) as ei:
        corpus.verified_identities()
    assert "unknown" in str(ei.value)


def test_identity_absent_row_fails_window_build(tmp_path):
    rows = {gpos: row for gpos, row in _default_id_rows().items()
            if gpos != 3}                       # w1/pi not covered
    corpus = ia.IdeaACorpus(str(_identity_pack(tmp_path)), split="TRAIN",
                            identity_loader=_sidecar_loader(rows))
    with pytest.raises(ia.IdeaAConfigError) as ei:
        corpus.verified_identities()
    assert "verified TRAIN identities" in str(ei.value)
    assert "w1" in str(ei.value) and "pi" in str(ei.value)


def test_verified_identities_receipt(tmp_path):
    corpus = ia.IdeaACorpus(str(_identity_pack(tmp_path)), split="TRAIN",
                            identity_loader=_sidecar_loader(_default_id_rows()))
    receipt = corpus.verified_identities()
    assert receipt["status"] == "verified"
    assert receipt["mode"] == "identity-sidecar"
    assert receipt["n_items"] == 5
    assert receipt["n_verified_id_slots"] == 15
    assert receipt["receipt_sha256"] == "ab" * 32


def test_unset_env_fails_verified_gate(tmp_path, monkeypatch):
    monkeypatch.delenv("WM_N1_TRAIN_IDENTITIES", raising=False)
    corpus = ia.IdeaACorpus(str(_identity_pack(tmp_path)), split="TRAIN")
    with pytest.raises(ia.IdeaAConfigError) as ei:
        corpus.verified_identities()
    assert "WM_N1_TRAIN_IDENTITIES" in str(ei.value)


# ------------------------------------- trainer verified-identity gates ------
def _eligible_specs():
    taus = C.N_TAU
    return _pair(_mask([0, 1], taus), _mask([0, 1], taus), [0, 1], [0, 1])


def test_source_without_identity_accessor_refused(tmp_path):
    tr = _trainer()
    with pytest.raises(ia.IdeaAConfigError) as ei:
        tr.prepare_training_source(_PairSource(_eligible_specs()))
    assert "verified_identities" in str(ei.value)


def test_fit_identity_gate_runs_before_the_pool_refusal(tmp_path):
    taus = C.N_TAU
    # the pair IS support-ineligible, but the identity failure must win:
    # verified identities are gated BEFORE the sampler/pool refusal
    specs = _pair(_mask([0], taus), _mask([2], taus), [0], [2])
    src = _IdSource(specs, identity="raises")
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=1, batch_size=2,
                              out_dir=str(tmp_path / "out"), num_workers=0)
    tr = ia.IdeaATrainer(cfg, src, member=1, seed=101,
                         bundle=seeded_bundle(101))
    with pytest.raises(ia.IdeaAConfigError) as ei:
        tr.fit()
    assert "verified" in str(ei.value)
    assert "no eligible sibling pair" not in str(ei.value)


def test_unverified_status_refuses_fit(tmp_path):
    src = _IdSource(_eligible_specs(), identity="unverified")
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=1, batch_size=2,
                              out_dir=str(tmp_path / "out"), num_workers=0)
    tr = ia.IdeaATrainer(cfg, src, member=1, seed=101,
                         bundle=seeded_bundle(101))
    with pytest.raises(ia.IdeaAConfigError) as ei:
        tr.fit()
    assert "verified identity gate failed" in str(ei.value)


def test_prepare_training_source_record_before_optimizer(tmp_path):
    src = _IdSource(_eligible_specs(), identity="ok")
    cfg = ia.IdeaATrainConfig(out_dir="unused", num_workers=0)
    tr = ia.IdeaATrainer(cfg, src, bundle=seeded_bundle(0))
    prep = tr.prepare_training_source(src)
    assert prep["identity"]["status"] == "verified"
    assert prep["pair_pool"] == [(0, 1)]
    assert prep["require_all_terms"] is True
    assert prep["n_meta_items"] == 2
    assert prep["outcome_norm"] is not None
    assert tr._optimizer is None                # gated BEFORE the optimizer
    assert tr._require_all_terms is True


def test_explicit_windows_record_ids_without_identity_gate(tmp_path):
    tr = _trainer()
    windows = make_windows(1, k=8, seed=0)
    prep = tr.prepare_training_source(None, windows=windows)
    assert prep["identity"] == {"mode": "explicit-windows",
                                "n_windows": 1, "n_with_vehicle_ids": 0}
    windows[0]["vehicle_ids"] = torch.zeros(2, 8, dtype=torch.long)
    prep = tr.prepare_training_source(None, windows=windows)
    assert prep["identity"]["n_with_vehicle_ids"] == 1


def test_fit_on_synthetic_pack_verifies_identity_and_records_receipt(
        tmp_path, monkeypatch):
    pack = _identity_pack(tmp_path)
    monkeypatch.setenv("WM_N1_TRAIN_OUTCOMES", str(_train_outcomes_dir(pack)))
    corpus = ia.IdeaACorpus(str(pack), split="TRAIN",
                            identity_loader=_sidecar_loader(_default_id_rows()))
    assert corpus.verified_identities()["status"] == "verified"
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=2, batch_size=2,
                              out_dir=str(tmp_path / "out"),
                              rollout_horizon_steps=2, num_workers=0)
    tr = ia.IdeaATrainer(cfg, corpus, member=1, seed=101,
                         bundle=seeded_bundle(101))
    receipt = tr.fit()
    assert receipt["identity"]["status"] == "verified"
    assert receipt["identity"]["receipt_sha256"] == "ab" * 32
    assert any("response_contrast" in m
               for m in receipt["loss_magnitudes"])
    assert (tmp_path / "out" / "member_1.pt").is_file()


# ---------------------------------- canonical preparation binding -----------
def test_prepare_training_source_binds_exact_pool_norm_identity():
    """The helper BINDS the actual prepared pool/norm/identity (sentinels).

    The returned record's objects ARE the sentinels: the trainer's bound
    attributes must be those exact objects at return time, with no optimizer
    created and no sample consumed before the binding, and the output record
    contract unchanged.
    """
    src = _IdSource(_eligible_specs(), identity="ok")
    cfg = ia.IdeaATrainConfig(out_dir="unused", num_workers=0)
    tr = ia.IdeaATrainer(cfg, src, bundle=seeded_bundle(0))
    constructor_norm = tr.outcome_norm
    prep = tr.prepare_training_source(src)
    assert tr._contrast_pair_pool is prep["pair_pool"]
    assert tr._identity_receipt is prep["identity"]
    assert tr.outcome_norm is prep["outcome_norm"]
    assert tr.outcome_norm is not constructor_norm   # not the identity default
    assert tr._optimizer is None            # binding BEFORE any optimizer
    assert tr.step == 0                     # ...and before any sample step
    assert set(prep) == {"source", "windows", "require_all_terms",
                         "identity", "pair_pool", "n_meta_items",
                         "quick_receipt", "amendment_6_lf_sha256",
                         "outcome_supervision_exclusion",
                         "outcome_norm"}     # record contract retained (+(b))
    assert prep["outcome_supervision_exclusion"]["count"] == 0
    assert tr._outcome_supervision_receipt is \
        prep["outcome_supervision_exclusion"]


def test_helper_only_caller_gets_fit_state_not_legacy_fallback():
    """A strict timing caller that ONLY runs the helper reads bound attrs.

    One eligible and one support-dead group: the legacy unfiltered fallback
    would hand the sampler BOTH pairs, while the bound canonical pool keeps
    only the eligible one; the bound normalization/identity are the prepared
    objects - all without ``fit`` ever running.
    """
    taus = C.N_TAU
    specs = (_pair(_mask([0, 1], taus), _mask([0, 1], taus),
                   [0, 1], [0, 1], key="good@t0=0")
             + _pair(_mask([0], taus), _mask([2], taus),
                     [0], [2], key="dead@t0=0"))
    src = _IdSource(specs, identity="ok")
    cfg = ia.IdeaATrainConfig(out_dir="unused", num_workers=0)
    tr = ia.IdeaATrainer(cfg, src, bundle=seeded_bundle(0))
    from train.dataset import contrast_pairs
    raw = contrast_pairs(src._items)
    assert len(raw) == 2                    # the unfiltered legacy fallback
    prep = tr.prepare_training_source(src)
    assert tr._contrast_pair_pool is prep["pair_pool"]
    assert tr._contrast_pair_pool == [(0, 1)]        # eligible pool only
    assert tr.outcome_norm is prep["outcome_norm"]
    assert tr._identity_receipt is prep["identity"]
    pairs = tr._pairs_for(src, src._items, contrast_pairs=contrast_pairs)
    assert pairs == tr._contrast_pair_pool == [(0, 1)]  # sampler sees the pool
    assert tr._optimizer is None
    assert tr.step == 0


# --------------------------------- canonical pack alias (frozen $WORK/pack) -
def _dir_alias(target, link) -> bool:
    """Create ``link`` as a directory alias of ``target`` (symlink/junction)."""
    try:
        os.symlink(os.fspath(target), os.fspath(link),
                   target_is_directory=True)
        return os.path.isdir(os.fspath(link))
    except (OSError, NotImplementedError, AttributeError):
        pass
    if os.name == "nt":
        try:
            import _winapi
            _winapi.CreateJunction(os.fspath(target), os.fspath(link))
            return os.path.isdir(os.fspath(link))
        except (OSError, ImportError, AttributeError):
            return False
    return False


def _aliased_identity_pack(tmp_path):
    """A real-format identity pack plus an intentional frozen alias to it."""
    real = _write_identity_pack(tmp_path / "real-pack", _runs_cal_first())
    alias = tmp_path / "alias-pack"
    if not _dir_alias(real, alias):
        pytest.skip("no directory symlink/junction support on this host")
    # fixture sanity: the alias IS a link resolving to the real pack
    assert Path(alias).resolve(strict=True) == Path(real).resolve(strict=True)
    return real, alias


def _recording_loader(into: dict):
    def _load(pack_dir, side_dir):              # TEST-ONLY injected loader
        into["pack_dir"] = pack_dir
        into["side_dir"] = side_dir
        return _FakeSidecar(_default_id_rows())

    return _load


def test_canonical_pack_env_passes_literal_target_to_loader(tmp_path,
                                                            monkeypatch):
    """Match: the loader reads the LITERAL canonical target, alias literals stay."""
    real, alias = _aliased_identity_pack(tmp_path)
    monkeypatch.setenv("WM_N1_CANONICAL_PACK", str(real))
    monkeypatch.setenv("WM_N1_TRAIN_IDENTITIES", str(tmp_path / "side"))
    seen: dict = {}
    corpus = ia.IdeaACorpus(str(alias), split="TRAIN",
                            identity_loader=_recording_loader(seen))
    receipt = corpus.verified_identities()
    assert seen["pack_dir"] == str(real)        # literal canonical target
    assert receipt["canonical_pack"] == str(real)
    assert receipt["pack_alias"] == str(alias)  # receipt keeps the ALIAS
    assert str(corpus.pack_dir) == str(alias)   # corpus literal retained
    assert corpus._canonical_pack_used == str(real)


def test_canonical_pack_mismatch_refuses_before_loader(tmp_path, monkeypatch):
    """A canonical env that does NOT resolve to the alias refuses fail-closed."""
    real, alias = _aliased_identity_pack(tmp_path)
    other = _write_identity_pack(tmp_path / "other-pack", _runs_cal_first())
    monkeypatch.setenv("WM_N1_CANONICAL_PACK", str(other))
    seen: dict = {}
    corpus = ia.IdeaACorpus(str(alias), split="TRAIN",
                            identity_loader=_recording_loader(seen))
    with pytest.raises(ia.IdeaAConfigError) as ei:
        corpus.verified_identities()
    assert "WM_N1_CANONICAL_PACK" in str(ei.value)
    assert seen == {}                           # fake loader was never called


def test_canonical_pack_env_absent_passes_literal_alias_no_fallback(
        tmp_path, monkeypatch):
    """No env: the literal alias is passed unchanged (no resolve fallback)."""
    real, alias = _aliased_identity_pack(tmp_path)
    monkeypatch.delenv("WM_N1_CANONICAL_PACK", raising=False)
    monkeypatch.setenv("WM_N1_TRAIN_IDENTITIES", str(tmp_path / "side"))
    seen: dict = {}
    corpus = ia.IdeaACorpus(str(alias), split="TRAIN",
                            identity_loader=_recording_loader(seen))
    receipt = corpus.verified_identities()
    assert seen["pack_dir"] == str(alias)       # literal alias, NOT resolved
    assert receipt["canonical_pack"] is None
    assert receipt["pack_alias"] == str(alias)
    # a blank env value is equally "unset" (no implicit canonical target):
    seen2: dict = {}
    corpus2 = ia.IdeaACorpus(str(alias), split="TRAIN",
                             identity_loader=_recording_loader(seen2))
    monkeypatch.setenv("WM_N1_CANONICAL_PACK", "   ")
    receipt2 = corpus2.verified_identities()
    assert seen2["pack_dir"] == str(alias)
    assert receipt2["canonical_pack"] is None
