# ruff: noqa
# VENDORED from output/wd01/wm_n1/dspeed/composite.py (last changed in commit df6a02f; copied at baseline bf7d4cb).
# Only change: absolute imports of sibling modules re-pointed to this vendor package.
# Do not edit: behaviour is frozen evidence; WP-C only mirrors it.
"""A9 step 4b: item-level composite of packed corpora (``--extra-corpus``).

``MemmapCorpus`` owns one pack directory; a single physical pack holding
``pack_v3`` (~50 GB) plus the dspeed pack would not fit on disk.  This module
presents several packs as ONE corpus at the item level:

* ``_items`` / ``_order`` concatenate every pack's split items (tick and
  plan offsets shifted by the preceding packs' full-array lengths), so the
  seeded pair logic (:mod:`train.dataset`) sees trajectory items and their
  ``pack_v3`` native/pi siblings under the same ``pair_key``;
* ``_array`` returns a read-only :class:`ConcatArray` view that dispatches
  slices to the owning pack -- Idea A windows never cross a run boundary, so
  a window slice stays inside one pack;
* base-pack items keep their packed outcomes / sidecar; extra-pack zero-row
  items read their schema/3-style overlay group by full item identity
  (:mod:`dspeed.overlay`).

This is the only disk-feasible way to train on ``pack_v3`` + the extra pack;
the frozen ``MemmapCorpus`` / :mod:`corpus` writers are untouched.
"""
from __future__ import annotations

import bisect
import os

import numpy as np

from . import common as D


def _item_key(raw):
    """Manifest-unique item identity (mirrors IdeaACorpus._item_position_key)."""
    return (str(raw.get("chunk")), str(raw.get("world")), str(raw.get("arm")),
            float(raw.get("t0_s")), int(raw.get("tick")),
            int(raw.get("plan_row")))


class ConcatArray:
    """Read-only concatenation of same-shaped arrays across packs."""

    def __init__(self, arrays, offsets):
        self._parts = list(arrays)
        self._offsets = [int(o) for o in offsets]
        self._ends = [o + len(a) for o, a in zip(self._offsets, self._parts)]
        self.shape = (self._ends[-1],) + tuple(self._parts[0].shape[1:])
        self.dtype = self._parts[0].dtype
        self.ndim = len(self.shape)

    def __len__(self):
        return int(self.shape[0])

    def _loc(self, g: int):
        i = bisect.bisect_right(self._offsets, g) - 1
        if i < 0 or g >= self._ends[i]:
            raise IndexError(g)
        return i, g - self._offsets[i]

    def __getitem__(self, key):
        if isinstance(key, tuple) and key:
            first, rest = key[0], key[1:]
            if isinstance(first, slice):
                start, stop, step = first.indices(self.shape[0])
                if start >= stop:
                    sample = np.asarray(
                        self._parts[0][(slice(0, 0),) + rest])
                    return np.zeros((0,) + sample.shape[1:], self.dtype)
                rows = []
                for g in range(start, stop, step):
                    i, local = self._loc(g)
                    rows.append(np.asarray(self._parts[i][(local,) + rest]))
                return np.stack(rows, axis=0)
            i, local = self._loc(int(first))
            return self._parts[i][(local,) + rest]
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            if step != 1:
                out = np.zeros(((max(0, (stop - start + step - 1) // step),)
                                + self.shape[1:]), self.dtype)
                for j, g in enumerate(range(start, stop, step)):
                    out[j] = self[g]
                return out
            if start >= stop:
                return np.zeros((0,) + self.shape[1:], self.dtype)
            i0, l0 = self._loc(start)
            i1, l1 = self._loc(stop - 1)
            if i0 == i1:
                return np.asarray(self._parts[i0][l0:l1 + 1])
            chunks = [np.asarray(self._parts[i0][l0:])]
            for i in range(i0 + 1, i1):
                chunks.append(np.asarray(self._parts[i]))
            chunks.append(np.asarray(self._parts[i1][:(l1 + 1)]))
            return np.concatenate(chunks, axis=0)
        i, local = self._loc(int(key))
        return self._parts[i][local]


class ConcatCorpus:
    """Multi-pack corpus with a ``MemmapCorpus``-compatible read interface."""

    _multi_pack = True

    def __init__(self, pack_dirs, split: str = "TRAIN", *,
                 pair_sampling: bool = False, k_max=None,
                 extra_overlay_dir=None):
        from wms.legacy.vendor.train.dataset import MemmapCorpus
        self.parts = [MemmapCorpus(str(p), split=split, pair_sampling=False,
                                   k_max=k_max) for p in pack_dirs]
        self.pack_dir = self.parts[0].pack_dir
        self.split = str(split).upper()
        self.pair_sampling = bool(pair_sampling)
        self.k_max = self.parts[0].k_max
        self.format = self.parts[0].format
        for p in self.parts[1:]:
            if p.k_max != self.k_max or p.format != self.format:
                raise D.DspeedError(
                    "extra pack k_max/format differs from the base pack; "
                    "refusing (refeaturize at the base format)")
        # full-array offsets (packed arrays hold every split, not just TRAIN)
        self._tick_offsets, self._plan_offsets = [], []
        to = po = 0
        for part in self.parts:
            self._tick_offsets.append(to)
            self._plan_offsets.append(po)
            to += int(part._array("veh_step").shape[0])
            po += int(part._array("plan").shape[0])
        # concatenated split items with global tile/plan offsets
        self._items = []
        self._manifest_items = []
        for pi, part in enumerate(self.parts):
            # ORIGINAL per-pack manifest position of every split item: the
            # identity sidecar is keyed by the pack's own manifest position,
            # never the (offset-shifted) composite dataset position.
            manifest_pos: dict = {}
            for gpos, man in enumerate(part.manifest.get("items") or []):
                key = _item_key(man)
                manifest_pos[key] = None if key in manifest_pos else int(gpos)
            for local, raw in enumerate(part._items):
                it = dict(raw)
                it["_pack"] = pi
                it["_local"] = local
                gpos = manifest_pos.get(_item_key(raw))
                if gpos is None:
                    raise D.DspeedError(
                        f"composite item {raw.get('world')!r}/"
                        f"{raw.get('arm')!r} does not map to a unique manifest "
                        "position in its pack; identity lookup refused")
                it["_manifest_gpos"] = int(gpos)
                it["tick"] = int(raw["tick"]) + self._tick_offsets[pi]
                it["plan_row"] = (int(raw["plan_row"])
                                  + self._plan_offsets[pi])
                self._items.append(it)
            self._manifest_items.extend(part.manifest.get("items") or [])
        self._order = self._build_order()
        self._pos_of = {gi: pos for pos, gi in enumerate(self._order)}
        # One overlay per extra pack, POSITIONAL with the extra packs (the
        # i-th overlay labels the i-th extra corpus).  A single path is still
        # accepted for the historical one-extra-pack callers.
        n_extra = len(self.parts) - 1
        if extra_overlay_dir is None:
            overlay_dirs = []
        elif isinstance(extra_overlay_dir, (str, os.PathLike)):
            overlay_dirs = [str(extra_overlay_dir)]
        else:
            overlay_dirs = [str(p) for p in extra_overlay_dir
                            if str(p).strip()]
        if overlay_dirs and len(overlay_dirs) != n_extra:
            raise D.DspeedError(
                f"--extra-overlay supplied {len(overlay_dirs)} overlay(s) for "
                f"{n_extra} extra pack(s); one positional overlay per extra "
                "pack is required")
        self._overlays = [None] * n_extra
        self._overlay_excluded = [set() for _ in range(n_extra)]
        self._overlay_receipts = [None] * n_extra
        if overlay_dirs:
            from .overlay import load
            for i, overlay_dir in enumerate(overlay_dirs):
                groups, receipt, excluded = load(overlay_dir)
                self._overlays[i] = groups
                self._overlay_receipts[i] = receipt
                self._overlay_excluded[i] = excluded
        self._mm = {}

    # -- manifest / metadata -------------------------------------------------
    @property
    def manifest(self) -> dict:
        man = dict(self.parts[0].manifest)
        man["items"] = list(self._manifest_items)
        return man

    def _build_order(self):
        from wms.legacy.vendor.train.dataset import _reference_of
        if not self.pair_sampling:
            return list(range(len(self._items)))
        groups = {}
        for i, it in enumerate(self._items):
            groups.setdefault(it["pair_key"], []).append(i)
        order = []
        for pk in sorted(groups):
            group = groups[pk]
            ref = _reference_of(group, self._items)
            order.extend([i for i in group if i != ref])
            if ref is not None:
                order.append(ref)
        return order

    def worlds(self):
        seen = []
        for it in self._items:
            if it["world"] not in seen:
                seen.append(it["world"])
        return sorted(seen)

    def pair_keys(self):
        return sorted({it["pair_key"] for it in self._items})

    @property
    def _runs(self):
        ranges = []
        for pi, part in enumerate(self.parts):
            off = self._tick_offsets[pi]
            idx = part._pack_index()
            recs = (idx or {}).get("runs") or []
            for rec in recs:
                s, e = rec.get("tick_start"), rec.get("tick_end")
                if s is not None and e is not None:
                    ranges.append((off + int(s), off + int(e)))
        return sorted(ranges)

    # -- array access --------------------------------------------------------
    def _array(self, name):
        if name not in self._mm:
            arrs = [p._array(name) for p in self.parts]
            offs = (self._plan_offsets if name == "plan"
                    else self._tick_offsets)
            self._mm[name] = ConcatArray(arrs, offs)
        return self._mm[name]

    # -- outcome access ------------------------------------------------------
    def _overlay_key(self, item):
        return (str(item.get("batch") or item["chunk"]), str(item["chunk"]),
                str(item["world"]), str(item["arm"]), str(item["split"]),
                round(float(item["t0_s"]), 6))

    def _extra_outcomes(self, item):
        from wms.legacy.vendor.train.dataset import map_outcome_targets
        pi = int(item.get("_pack", 0))
        overlay = (self._overlays[pi - 1]
                   if 0 < pi <= len(self._overlays) else None)
        if overlay is None:
            raise D.DspeedError(
                f"extra pack {pi} zero-row item "
                f"{item['world']}__{item['arm']}@{item['t0_s']} has no "
                "--extra-overlay (positional); refusing")
        key = self._overlay_key(item)
        frame = overlay.get(key)
        if frame is None:
            raise D.DspeedError(
                f"dspeed overlay has no group for zero-row item "
                f"{item['world']}__{item['arm']}@{item['t0_s']}")
        return map_outcome_targets(frame, t0_s=item["t0_s"])

    def outcome_target(self, i: int):
        if i < 0:
            i += len(self._order)
        item = self._items[self._order[i]]
        if item.get("outcome_rows"):
            return self.parts[item["_pack"]].outcome_target(
                self.parts[item["_pack"]]._pos_of[item["_local"]])
        if item["_pack"] == 0:
            return self.parts[0].outcome_target(
                self.parts[0]._pos_of[item["_local"]])
        return self._extra_outcomes(item)

    def __len__(self):
        return len(self._order)

    def __getitem__(self, i: int) -> dict:
        from wms.legacy.vendor.train.dataset import label_t0_rows
        import torch
        if i < 0:
            i += len(self._order)
        item = self._items[self._order[i]]
        pi, local = item["_pack"], item["_local"]
        part = self.parts[pi]
        out = part.__getitem__(part._pos_of[local])
        out["tick"] = int(item["tick"])
        out["plan_row"] = int(item["plan_row"])
        out["pair_key"] = item["pair_key"]
        out["policy"] = item.get("policy", item["arm"])
        # carry the composite provenance so the identity resolver can pick the
        # right pack sidecar and the ORIGINAL per-pack manifest position.
        out["_pack"] = pi
        out["_manifest_gpos"] = item["_manifest_gpos"]
        if not item.get("outcome_rows") and pi != 0:
            targets, omask, collision = self._extra_outcomes(item)
            out["outcome_quantiles"] = torch.as_tensor(targets)
            out["outcome_mask"] = torch.as_tensor(omask)
            out["collision"] = torch.as_tensor(collision)
            out["label_t0_rows"] = int(
                label_t0_rows(self._overlays[pi - 1][
                                  self._overlay_key(item)],
                              item["t0_s"]))
            out["label_t0_violations"] = 0
        return out


def pair_stats(items, order, pairs_per_batch, seed, epoch, merge_coverage=True,
               pairs=None):
    """``(n_batches, per-batch trajectories, per-batch pairs)`` summary.

    Runs the frozen pair-dense batching (:func:`train.dataset.
    pair_dense_batches`) over the composite dataset order and reports how
    many trajectory items and complete eligible pairs each batch holds.
    """
    from wms.legacy.vendor.train.dataset import contrast_pairs, pair_dense_batches
    ordered = [items[j] for j in order]
    if pairs is None:
        pairs = contrast_pairs(ordered)
    batches = pair_dense_batches(ordered, pairs_per_batch, seed, epoch,
                                 pairs=pairs, merge_coverage=merge_coverage)
    per_items, per_pairs = [], []
    partners = {}
    for a, b in pairs:
        partners.setdefault(int(a), set()).add(int(b))
        partners.setdefault(int(b), set()).add(int(a))
    for batch in batches:
        traj = sum(1 for p in batch
                   if str(ordered[p].get("policy")).lower() == "trajectory")
        present = set(batch)
        npair = 0
        for p in present:
            for q in partners.get(p, ()):
                if q in present and p < q:
                    npair += 1
        per_items.append(traj)
        per_pairs.append(npair)
    return len(batches), per_items, per_pairs, len(pairs)
