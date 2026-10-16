# ruff: noqa
# VENDORED from output/wd01/wm_n1/train/idea_a.py (last changed in commit aa98fff; copied at baseline bf7d4cb).
# Extracted: IdeaAConfigError (l.83-86), the pack-array constants (l.462-469), remap_canonical_ids
# (l.668-680), _cav_motion_summary (l.647-665) and class IdeaACorpus (l.683-1527) -- the
# read-only window adapter.  Trainer, losses and CLI of idea_a.py are NOT vendored.
# Changes (and only these): (1) absolute sibling imports re-pointed to wms.legacy.vendor.*;
# (1b) module constants _KIN_* / TEACHER_* added from idea_a.py l.474 / l.46;
# (2) CAV_TGT_SPEED_SCALE / CAV_ACCEL_SCALE inlined from features/idea_a.py l.73-74 instead of
# importing that 887-line featurizer module.  The ``plan_horizon_s`` branch of window() still
# imports model.paired_effect, which is not vendored; WM-S never sets plan_horizon_s.
# Do not edit: behaviour is frozen evidence; WP-C only mirrors it.
"""Read-only A11 pack window adapter (vendored IdeaACorpus)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from wms.legacy.vendor.model import constants as C

#: features/idea_a.py l.73-74 (inlined, see header).
CAV_TGT_SPEED_SCALE = 40.0        # tgt_speed_mps / 40
CAV_ACCEL_SCALE = 3.0             # a_cmd_mps2 / 3



class IdeaAConfigError(ValueError):
    """An Idea A training config or request violates the contract."""


_KIN_S, _KIN_L0, _KIN_L1, _KIN_L2, _KIN_V, _KIN_A, _KIN_LEN = range(7)  # idea_a.py l.474

#: teacher column order (idea_a.py l.46)
TEACHER_S, TEACHER_LANE, TEACHER_V, TEACHER_A = 0, 1, 2, 3
_WORLD0_REQUIRED = ("s0", "v0", "lane0", "length0", "present0")

#: the five packed P9 Idea A channels the full canonical batch is built from.
IDEA_A_PACK_ARRAYS = ("veh_kin", "hdv_base", "cav_plan_ch", "event_tok",
                      "event_mask")
#: ``hdv_base`` column of the one-step IDM baseline acceleration (P9 field 0).
HDV_BASE_IDM_IDX = 0

def _cav_motion_summary(pos_seq: Tensor, cav_plan_seq: Tensor,
                        cav_mask: Tensor) -> Tensor:
    """``summary_raw`` ``[B, SUMMARY_RAW_F]`` from the CAV planned channels.

    The P5 correction head consumes the registered 16-feature CAV planned-motion
    summary (``corr.summary.cav_motion_summary``); the pack stores the CAV
    planned speed/accel per tick in ``cav_plan_ch`` (fields 0/1, scaled by the
    frozen :data:`features.idea_a.CAV_TGT_SPEED_SCALE` /
    :data:`features.idea_a.CAV_ACCEL_SCALE`) and the recorded positions in
    ``veh_kin``.  No new field is invented: the same deterministic summary the
    evaluator builds from a CAV rollout is rebuilt from the packed channels.
    """
    from wms.legacy.vendor.corr.summary import cav_motion_summary

    # [T, B, K] (batch axis inserted), planner channels de-scaled to SI.
    speeds = cav_plan_seq[..., 0] * float(CAV_TGT_SPEED_SCALE)
    accel = cav_plan_seq[..., 1] * float(CAV_ACCEL_SCALE)
    return cav_motion_summary(pos_seq, speeds, accel, cav_mask)

def remap_canonical_ids(values, vehicle_map, lookup):
    """Pack-local canonical IDs -> one shared canonical index space.

    ``vehicle_map`` maps a pack-local canonical index to its
    ``(chunk, world, vehicle_id)`` key; ``lookup`` maps that key to a shared
    index.  ``-1`` (absent slot) stays ``-1``.  Pure and deterministic so the
    cross-pack HDV alignment contract is unit-testable without a corpus.
    """
    values = np.asarray(values, dtype=np.int64)
    out = np.empty(values.shape, dtype=np.int64)
    for j, raw in enumerate(values.tolist()):
        out[j] = -1 if raw < 0 else int(lookup[tuple(vehicle_map[int(raw)])])
    return out

class IdeaACorpus:
    """Read-only packed-corpus view yielding the FULL canonical Idea A batch.

    ``train.dataset.MemmapCorpus`` (the frozen v1 shard reader) exposes the v1
    tensors and the packed outcome rows; this view adds the five P9 Idea A
    channels (``veh_kin``/``hdv_base``/``cav_plan_ch``/``event_tok`` and
    ``event_mask``) and the consecutive-tick rollout window at the registered
    horizon, so :meth:`IdeaATrainer.fit` (and the timing/G4/G5 gates) compute
    every registered A3.2 loss term instead of silently dropping them.

    Construction FAILS CLOSED when the pack lacks any Idea A channel; the view
    returns one canonical window per item (leading batch axis 1) that
    :func:`collate_canonical` concatenates exactly like the synthetic windows.

    ``k_max`` (amendment 5) is forwarded to the frozen ``MemmapCorpus``: None
    takes the pack's own K_max, an explicit 128 (v2) / 192 (v3) refuses a
    mismatching pack, so a pack/model K mismatch refuses on the Idea A path too.

    Verified identity access (contrastfix identity recovery): for a TRAIN
    corpus, windows carry verified canonical ``vehicle_ids`` loaded lazily
    through the portable TRAIN identity sidecar
    (``train.identity_sidecar.load_identity_sidecar(pack_dir, sidecar_dir)``,
    sidecar directory from ``WM_N1_TRAIN_IDENTITIES``).  The sidecar loader
    applies strict path checks that reject any symlink/reparse ancestor - but
    a deployed GPU pack location is often an INTENTIONAL frozen alias (e.g.
    ``$WORK/pack`` -> the real upload tree).  The separately
    coordinator-validated REAL canonical target may be supplied through the
    OPTIONAL ``WM_N1_CANONICAL_PACK`` environment: ONLY when it is set, and
    ``pack_dir`` resolves (strict) to EXACTLY that directory, is the literal
    canonical target (never the alias) handed to the identity loader - whose
    own strict checks then prove the target itself is link-free.  A mismatch
    refuses BEFORE the loader reads anything; with the env unset there is NO
    automatic resolve fallback (the literal ``pack_dir`` is passed unchanged,
    so an alias still refuses).  ``pack_dir``, the corpus attributes and the
    identity receipt's ``pack_alias`` keep the ORIGINAL literal for the
    handoff gate; ``canonical_pack`` records the validated target.  Rows are
    indexed by
    the item's ORIGINAL GLOBAL manifest position - resolved here from the pack
    manifest, never from the TRAIN-filtered dataset position - and every row
    is validated against the item's HDV-valid slots: a missing, unknown or
    duplicated identity raises (fail closed, never guessed).  CAL corpora get
    no identities (the sidecar is TRAIN-only; CAL is never read here).  The
    ``identity_loader`` injection is TEST-ONLY (synthetic fixtures).
    """

    def __init__(self, pack_dir, split: str = "TRAIN", *,
                 pair_sampling: bool = False, horizon_steps: int = 20,
                 allow_single_run_fallback: bool = False,
                 k_max: Optional[int] = None,
                 identity_loader: Optional[Callable] = None,
                 extra_pack_dirs: Optional[Sequence] = None,
                 extra_overlay_dir: Optional[str] = None,
                 extra_identity_dirs: Optional[Sequence] = None,
                 extra_identity_loader: Optional[Callable] = None,
                 plan_horizon_s: Optional[float] = None) -> None:
        from wms.legacy.vendor.train.dataset import MemmapCorpus

        pack_dir = Path(pack_dir)
        self._is_multi_pack = bool(extra_pack_dirs)
        self._extra_pack_dirs = [str(p) for p in (extra_pack_dirs or [])]
        # A11 FULL-HORIZON plan summary: when set, each window also carries a
        # fixed-size ``plan_long`` vector (the treatment arm's planned CAV
        # motion over [t0, t0 + H], binned in 10 s bins); None keeps every
        # existing path byte-for-byte unchanged (no plan_long key emitted).
        self._plan_horizon_s = (None if plan_horizon_s is None
                                else float(plan_horizon_s))
        # ``k_max`` is the FORMAT slot width (amendment 5): None takes the
        # pack's own K_max, and an explicit value (128 v2 / 192 v3) makes the
        # frozen MemmapCorpus refuse a mismatching pack before any Idea A
        # channel is read, so a K mismatch refuses on the Idea A path too.
        if extra_pack_dirs:
            # A9: item-level composite of pack_v3 + extra (dspeed/damping)
            # pack; the physical single-pack concatenation would need > the
            # available disk.  The composite keeps every MemmapCorpus
            # contract the rest of this class relies on.
            from wms.legacy.vendor.dspeed.composite import ConcatCorpus
            self._v1 = ConcatCorpus(
                [pack_dir, *extra_pack_dirs], split=split,
                pair_sampling=pair_sampling, k_max=k_max,
                extra_overlay_dir=extra_overlay_dir)
        else:
            self._v1 = MemmapCorpus(pack_dir, split=split,
                                    pair_sampling=pair_sampling, k_max=k_max)
        arrays = set(self._v1.manifest.get("arrays", {}))
        missing = [name for name in IDEA_A_PACK_ARRAYS if name not in arrays]
        if missing:
            raise IdeaAConfigError(
                f"pack {pack_dir} lacks the Idea A channel(s) {missing}; the "
                "registered A3.2 losses (rollout / correction / contrast) "
                "cannot be built - refusing instead of silently dropping them "
                "(re-featurize with the P9 Idea A channels)")
        self.pack_dir = pack_dir
        self.split = self._v1.split
        # resolved format slot width + declared format (None when unclaimed)
        self.k_max = self._v1.k_max
        self.format = self._v1.format
        self.pair_sampling = bool(pair_sampling)
        self._items = self._v1._items
        self._order = self._v1._order
        self.horizon_steps = max(1, int(horizon_steps))
        # TEST-ONLY: allow a pack without INDEX.json to be treated as one run.
        # A real corpus run must leave this False so a missing/unparsable run
        # table fails closed instead of silently crossing run boundaries.
        self.allow_single_run_fallback = bool(allow_single_run_fallback)
        self._mm: dict = {}
        self._runs = self._load_run_ranges()
        # verified identity state (contrastfix identity recovery): the
        # sidecar (or False = unavailable), per-global-position validated ID
        # rows, and the ORIGINAL global manifest position of every dataset
        # position (resolved via the pack manifest, NOT the TRAIN position).
        self._identity_loader = identity_loader
        self._identity = None
        self._id_rows: dict = {}
        # the literal canonical pack target this corpus validated and loaded
        # against (WM_N1_CANONICAL_PACK), or None when the env is unset;
        # recorded in the identity receipt for audit alongside pack_alias
        self._canonical_pack_used: Optional[str] = None
        # A9 composite: one identity sidecar per extra pack (the pack_v3
        # sidecar never covers the extra items), plus the TEST-ONLY injected
        # extra loader.  Each pack's canonical ID index is PACK-LOCAL, so in
        # composite mode every row is re-projected into one shared index space
        # built from the union of the packs' canonical vehicle maps: two packs
        # then agree on the same physical vehicle's canonical ID.
        self._extra_identity_dirs = self._resolve_extra_identity_dirs(
            extra_identity_dirs)
        self._extra_identity_loader = extra_identity_loader
        self._extra_identities: list = [None] * len(self._extra_pack_dirs)
        self._reproject_ids = bool(self._is_multi_pack)
        self._global_lookup: Optional[dict] = None
        self._global_packs: list = []
        self._manifest_pos = self._build_manifest_position_index()
        self._global_positions = self._resolve_global_positions()

    # -- pickling (DataLoader workers) --------------------------------------
    def __getstate__(self) -> dict:
        """Drop open memmaps before pickling to a spawned worker.

        Mirrors :meth:`train.dataset.MemmapCorpus.__getstate__`: the cached
        ``np.memmap`` arrays serialize their whole content, so a spawned
        worker would re-serialize a real pack (multi-GB).  Workers re-open the
        arrays lazily; fork inherits them.  The validated per-item ID-row
        cache is dropped too (re-validated lazily from the immutable
        sidecar); the loaded sidecar itself pickles as compact bytes.
        """
        state = dict(self.__dict__)
        state["_mm"] = {}
        state["_id_rows"] = {}
        state["_global_lookup"] = None      # rebuilt lazily from the maps
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("_mm", {})
        self.__dict__.setdefault("_id_rows", {})
        self.__dict__.setdefault("_global_lookup", None)

    # -- pack access --------------------------------------------------------
    def _array(self, name):
        if name not in self._mm:
            # delegate to the wrapped corpus so a multi-pack composite
            # (A9 --extra-corpus) serves the concatenated view; a plain
            # MemmapCorpus._array loads the identical bytes from pack_dir.
            self._mm[name] = self._v1._array(name)
        return self._mm[name]

    def _load_run_ranges(self) -> list:
        """Per-run ``(tick_start, tick_end)`` from the corpus ``INDEX.json``.

        The window never crosses a run boundary (the concatenated tick arrays
        continue with another run's vehicles).  This FAILS CLOSED: a real pack
        whose ``INDEX.json`` is missing, unparsable, or has any run without
        ``tick_start``/``tick_end`` raises rather than guessing a single range
        and silently windowing across a boundary.  The single-range fallback is
        reachable only through the explicit TEST-ONLY
        ``allow_single_run_fallback`` constructor flag.
        """
        if getattr(self._v1, "_multi_pack", False):
            # A9 composite: the merged per-pack run ranges (global tick
            # offsets) are authoritative, so a window never crosses a run
            # boundary in any part.
            return list(self._v1._runs)
        path = self.pack_dir / "INDEX.json"
        if not path.is_file():
            if self.allow_single_run_fallback:
                n_ticks = int(self._v1.manifest["arrays"]["veh_kin"]["shape"][0])
                return [(0, n_ticks)]
            raise IdeaAConfigError(
                f"pack {self.pack_dir} has no INDEX.json; the run "
                "(tick_start/tick_end) table is required to keep a window "
                "inside one run.  Pass allow_single_run_fallback=True only in "
                "tests (a real corpus run must fail closed)")
        try:
            index = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise IdeaAConfigError(
                f"pack {self.pack_dir} INDEX.json is not parsable: {exc}") \
                from exc
        runs = index.get("runs") if isinstance(index, dict) else None
        records = list(runs.values()) if isinstance(runs, dict) else runs
        if not records:
            raise IdeaAConfigError(
                f"pack {self.pack_dir} INDEX.json carries no runs; the "
                "tick_start/tick_end table is required")
        ranges = []
        for rec in records:
            if not isinstance(rec, dict):
                raise IdeaAConfigError(
                    f"pack {self.pack_dir} INDEX.json run entry is not an "
                    f"object: {rec!r}")
            start, end = rec.get("tick_start"), rec.get("tick_end")
            if start is None or end is None:
                raise IdeaAConfigError(
                    f"pack {self.pack_dir} INDEX.json run {rec.get('world')!r}"
                    f"/{rec.get('arm')!r} lacks tick_start/tick_end; refusing "
                    "to guess the run boundary")
            ranges.append((int(start), int(end)))
        return sorted(ranges)

    def _run_end(self, tick: int) -> int:
        for start, end in self._runs:
            if start <= tick < end:
                return end
        return int(self._array("veh_kin").shape[0])

    # -- verified identity access (contrastfix identity recovery) -----------
    @staticmethod
    def _item_position_key(item: dict) -> tuple:
        """The manifest-unique item identity tuple used for position lookup."""
        return (str(item.get("chunk")), str(item.get("world")),
                str(item.get("arm")), float(item.get("t0_s")),
                int(item.get("tick")), int(item.get("plan_row")))

    def _build_manifest_position_index(self) -> dict:
        """``identity tuple -> ORIGINAL global manifest position`` (or None).

        ``MemmapCorpus._items`` is the SPLIT-FILTERED item list, so a dataset
        position is a TRAIN position, not the sidecar's row key.  The reliable
        mapping is rebuilt here from ``manifest["items"]`` itself; an identity
        tuple that is not unique across the manifest is marked unusable.  In
        composite mode the per-pack positions come from the ``_manifest_gpos``
        the composite stamps on each item (never the offset-shifted dataset
        position), so this base index is only consulted for a single pack.
        """
        index: dict = {}
        for gpos, raw in enumerate(self._v1.manifest.get("items") or []):
            key = self._item_position_key(raw)
            index[key] = None if key in index else gpos
        return index

    def _component_position(self, item: dict) -> tuple:
        """``(pack index, ORIGINAL per-pack manifest position)`` (strict).

        A single pack resolves the position from its own manifest.  A composite
        pack carries ``_pack`` + ``_manifest_gpos`` (set by
        :class:`dspeed.composite.ConcatCorpus`): the composite dataset ``tick``/
        ``plan_row`` are OFFSET-SHIFTED, so the manifest-position lookup MUST
        use the unshifted per-pack value, never the shifted dataset position.
        """
        pack = int(item.get("_pack", 0))
        if self._is_multi_pack or "_manifest_gpos" in item:
            gpos = item.get("_manifest_gpos")
            if gpos is None:
                raise IdeaAConfigError(
                    f"item {item.get('world')!r}/{item.get('arm')!r} does not "
                    "map to a unique manifest position in its own pack; the "
                    "sidecar identity lookup needs the ORIGINAL per-pack "
                    "manifest position, never the composite dataset position")
            return pack, int(gpos)
        gpos = self._manifest_pos.get(self._item_position_key(item))
        if gpos is None:
            raise IdeaAConfigError(
                f"pack {self.pack_dir} item {item.get('world')!r}/"
                f"{item.get('arm')!r} does not map to a unique global "
                "manifest position; the sidecar identity lookup needs "
                "the ORIGINAL manifest position, never the "
                "TRAIN-filtered dataset position")
        return 0, int(gpos)

    def _resolve_global_positions(self) -> list:
        """Dataset position -> ORIGINAL manifest position (strict, per pack).

        ``_global_packs`` is the parallel pack index (all 0 for one pack).
        """
        positions, packs = [], []
        for i in range(len(self._order)):
            pack, gpos = self._component_position(self._items[self._order[i]])
            positions.append(gpos)
            packs.append(pack)
        self._global_packs = packs
        return positions

    def _resolve_extra_identity_dirs(self, explicit) -> list:
        """One identity sidecar directory per extra pack (explicit or env)."""
        if not self._is_multi_pack:
            return []
        dirs = [str(x) for x in (explicit or []) if str(x).strip()]
        if not dirs:
            raw = os.environ.get("WM_N1_EXTRA_TRAIN_IDENTITIES")
            if raw and str(raw).strip():
                dirs = [p for p in str(raw).split(os.pathsep) if p.strip()]
        if dirs and len(dirs) != len(self._extra_pack_dirs):
            raise IdeaAConfigError(
                "the composite corpus has "
                f"{len(self._extra_pack_dirs)} extra pack(s) but "
                f"{len(dirs)} extra identity sidecar dir(s); one verified "
                "sidecar per extra pack is required (--extra-identities / "
                "WM_N1_EXTRA_TRAIN_IDENTITIES, os.pathsep-separated)")
        return dirs

    def _extra_sidecar(self, pack: int):
        """Verified sidecar for extra pack ``pack`` (1-based), or ``None``.

        Loaded lazily through :func:`dspeed.identity.load_extra_identity_sidecar`
        (which dispatches to the physics-sourced dspeed loader or the standard
        shard-sourced ``train.identity_sidecar`` loader by the receipt's
        ``source``) or the TEST-ONLY injected ``extra_identity_loader``.
        """
        if pack <= 0:
            raise IdeaAConfigError("base pack uses the pack_v3 identity sidecar")
        idx = pack - 1
        cached = self._extra_identities[idx]
        if cached is not None:
            return cached if cached is not False else None
        if self.split != "TRAIN":
            self._extra_identities[idx] = False
            return None
        if idx >= len(self._extra_identity_dirs) or \
                not self._extra_identity_dirs[idx]:
            self._extra_identities[idx] = False
            return None
        loader = self._extra_identity_loader
        if loader is None:
            try:
                from wms.legacy.vendor.dspeed.identity import load_extra_identity_sidecar
            except ImportError as exc:
                raise IdeaAConfigError(
                    "an extra identity sidecar is configured but "
                    "dspeed.identity is not importable; deploy the verified "
                    "extra identity module beside this package") from exc
            loader = load_extra_identity_sidecar
        part = self._v1.parts[pack]
        side = loader(str(part.pack_dir), self._extra_identity_dirs[idx])
        self._extra_identities[idx] = side
        return side

    def _global_vehicle_lookup(self) -> dict:
        """One shared canonical index space over every pack's vehicle map.

        Each sidecar's canonical IDs are PACK-LOCAL; re-projecting through the
        union of the ``(chunk, world, vehicle_id)`` maps makes integer equality
        across a trajectory item and its pack_v3 native sibling identify the
        same physical vehicle (fail closed if a map is unavailable).
        """
        if self._global_lookup is not None:
            return self._global_lookup
        maps = []
        base = self._identity_sidecar()
        if base is None or base.vehicle_map is None:
            raise IdeaAConfigError(
                "composite identity re-projection needs the base pack "
                "vehicle_map; refusing to align by pack-local indices")
        maps.append(base.vehicle_map)
        for pack in range(1, len(self._extra_pack_dirs) + 1):
            side = self._extra_sidecar(pack)
            if side is None or side.vehicle_map is None:
                raise IdeaAConfigError(
                    f"composite identity re-projection needs the extra pack "
                    f"{pack} vehicle_map; refusing to align by pack-local "
                    "indices")
            maps.append(side.vehicle_map)
        all_keys = sorted(set().union(*maps))
        self._global_lookup = {key: i for i, key in enumerate(all_keys)}
        return self._global_lookup

    def _reproject_row(self, side, row: Tensor) -> Tensor:
        lookup = self._global_vehicle_lookup()
        return torch.as_tensor(
            remap_canonical_ids(np.asarray(row, dtype=np.int64),
                                side.vehicle_map, lookup))

    def _sidecar_dir(self) -> Optional[str]:
        raw = os.environ.get("WM_N1_TRAIN_IDENTITIES")
        return raw if raw and str(raw).strip() else None

    @staticmethod
    def _canonical_pack_env() -> Optional[str]:
        """The OPTIONAL ``WM_N1_CANONICAL_PACK`` value (None when unset/blank).

        Root sets it ONLY to the separately coordinator-validated REAL
        canonical pack target (after the raw SHA verify); it is never guessed
        here.
        """
        raw = os.environ.get("WM_N1_CANONICAL_PACK")
        return raw if raw and str(raw).strip() else None

    def _canonical_pack_target(self) -> Optional[str]:
        """The validated canonical target for the sidecar read, or ``None``.

        The identity loader's strict path checks reject any symlink/reparse
        ancestor, and a deployed ``$WORK/pack`` location is an INTENTIONAL
        frozen alias.  ONLY when ``WM_N1_CANONICAL_PACK`` is set may the
        loader read the canonical target instead - and only after a strict
        ``Path.resolve(strict=True)`` equality check against ``pack_dir``
        (refused BEFORE any loader read).  The canonical target itself must
        be link-free: the loader's own strict checks enforce that on the
        literal path this corpus hands it.  When the env is absent there is
        NO automatic resolve fallback: ``None`` is returned and the literal
        ``pack_dir`` is passed unchanged (an alias then refuses, as it
        should, until root publishes the validated target).
        """
        canon = self._canonical_pack_env()
        if canon is None:
            return None
        try:
            resolved_alias = Path(self.pack_dir).resolve(strict=True)
            resolved_canon = Path(canon).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise IdeaAConfigError(
                f"WM_N1_CANONICAL_PACK {canon!r} or the pack alias "
                f"{str(self.pack_dir)!r} cannot be resolved (strict): "
                f"{exc}; refusing the identity sidecar read") from exc
        if resolved_alias != resolved_canon:
            raise IdeaAConfigError(
                f"WM_N1_CANONICAL_PACK {canon!r} resolves to "
                f"{str(resolved_canon)!r}, which is NOT the pack alias "
                f"{str(self.pack_dir)!r} ({str(resolved_alias)!r}); refusing "
                "to read the identity sidecar against an unvalidated pack "
                "location (no alias/canonical mismatch may reach the loader)")
        return canon

    def _identity_sidecar(self):
        """The verified TRAIN identity sidecar, or ``None`` when unavailable.

        Loading follows the exact sidecar interface
        (``load_identity_sidecar(pack_dir, sidecar_dir)``); the returned
        object exposes ``ids_for_item(global_manifest_position) -> int64[K]``
        (``-1`` absent) and ``receipt_sha256``.  When
        ``WM_N1_CANONICAL_PACK`` is set and validated (strict-resolve
        equality, checked here BEFORE the loader reads anything) the LITERAL
        canonical target is passed to the loader instead of the frozen
        ``pack_dir`` alias; ``pack_dir`` itself, the receipt's ``pack_alias``
        and every handoff-gate literal keep the original alias.  CAL corpora
        never load one (TRAIN-only provenance; CAL is never read here).  A
        load failure propagates - a broken sidecar refuses the corpus
        instead of degrading to slot-guessing.
        """
        if self._identity is not None:
            return self._identity if self._identity is not False else None
        if self.split != "TRAIN":
            self._identity = False
            return None
        loader = self._identity_loader
        side_dir = self._sidecar_dir()
        if loader is None:
            if not side_dir:
                self._identity = False
                return None
            try:
                from wms.legacy.vendor.train.identity_sidecar import load_identity_sidecar
            except ImportError as exc:
                raise IdeaAConfigError(
                    "WM_N1_TRAIN_IDENTITIES is set but "
                    "train.identity_sidecar is not importable; deploy the "
                    "verified identity module beside this package") from exc
            loader = load_identity_sidecar
        load_pack = str(self.pack_dir)
        canonical = self._canonical_pack_target()
        if canonical is not None:
            # frozen-alias allowance: the loader reads the validated REAL
            # canonical target; the corpus/receipt literals below retain the
            # original alias (the gate refuses any mismatch above)
            load_pack = canonical
        self._canonical_pack_used = canonical
        self._identity = loader(load_pack, side_dir)
        return self._identity

    def _validate_identity_row(self, item: dict, ids) -> Tensor:
        """Fail closed on a missing/unknown/duplicated verified identity.

        Every HDV-valid slot (``target_mask & ~is_cav``) must carry a
        non-negative canonical ID; ``-1`` is allowed only on absent/CAV
        slots; an ID may appear at most once per row.
        """
        ids = np.asarray(ids)
        if ids.dtype != np.int64:
            raise IdeaAConfigError(
                f"item {item.get('world')!r}/{item.get('arm')!r}: verified "
                f"identity row dtype {ids.dtype} is not int64")
        if ids.ndim != 1:
            raise IdeaAConfigError(
                f"item {item.get('world')!r}/{item.get('arm')!r}: verified "
                f"identity row is not 1-D (shape {ids.shape})")
        target = np.asarray(
            self._v1._array("target_mask")[int(item["tick"])], bool)
        veh = self._v1._array("veh_step")[int(item["tick"])]
        cav = np.asarray(veh[:, C.IS_CAV_IDX] > 0.5, bool)
        if ids.shape[0] != int(target.shape[0]):
            raise IdeaAConfigError(
                f"item {item.get('world')!r}/{item.get('arm')!r}: verified "
                f"identity row width {ids.shape[0]} != pack K "
                f"{int(target.shape[0])}")
        if ids.size and int(ids.min()) < -1:
            raise IdeaAConfigError(
                f"item {item.get('world')!r}/{item.get('arm')!r}: unknown "
                "verified identity (< -1) in row")
        present = ids >= 0
        if int(present.sum()) != len(np.unique(ids[present])):
            raise IdeaAConfigError(
                f"item {item.get('world')!r}/{item.get('arm')!r}: "
                "duplicated verified identity in row")
        missing = (target & ~cav) & ~present
        if bool(missing.any()):
            slots = np.flatnonzero(missing).tolist()
            raise IdeaAConfigError(
                f"item {item.get('world')!r}/{item.get('arm')!r}: missing "
                f"verified identity for HDV-valid slot(s) {slots}; refusing "
                "to disable the HDV sub-term or guess the identity")
        # copy: sidecar rows may back onto read-only buffers
        return torch.as_tensor(np.array(ids, dtype=np.int64, copy=True))

    def _ids_for_item(self, item: dict) -> Optional[Tensor]:
        """The item's validated ``int64[K]`` canonical IDs (or ``None``).

        The sidecar is queried with the item's ORIGINAL per-pack manifest
        position (never the TRAIN-filtered / composite-offset dataset
        position).  In composite mode each pack uses its OWN sidecar and rows
        are re-projected into one shared canonical space.  Rows are validated
        once and cached by ``(pack, position)``.
        """
        pack = int(item.get("_pack", 0))
        if pack == 0:
            side = self._identity_sidecar()
        else:
            side = self._extra_sidecar(pack)
            if side is None:
                raise IdeaAConfigError(
                    f"extra corpus item {item.get('world')!r}/"
                    f"{item.get('arm')!r} has no verified identity sidecar; "
                    "set --extra-identities / WM_N1_EXTRA_TRAIN_IDENTITIES to "
                    "one sidecar per extra pack (never guess identities)")
        if side is None:
            return None
        _, gpos = self._component_position(item)
        key = (pack, int(gpos))
        if key not in self._id_rows:
            try:
                row = side.ids_for_item(int(gpos))
            except Exception as exc:
                raise IdeaAConfigError(
                    f"item {item.get('world')!r}/{item.get('arm')!r} "
                    f"(pack {pack} manifest position {int(gpos)}) lacks "
                    f"verified TRAIN identities: {exc}") from exc
            row = self._validate_identity_row(item, row)
            if self._reproject_ids:
                row = self._reproject_row(side, row)
            self._id_rows[key] = row
        return self._id_rows[key]

    def verified_identities(self) -> dict:
        """Verify EVERY TRAIN item's identity; return the gate receipt.

        Raises :class:`IdeaAConfigError` when the sidecar is unavailable
        (``WM_N1_TRAIN_IDENTITIES`` unset) or any item lacks a verified,
        known, non-duplicated ID on every HDV-valid slot.  The receipt is
        recorded in ``TRAIN_RECEIPT.json``; the HDV sub-term is never
        silently disabled for an actual fit.  ``pack_alias`` (the original
        literal pack location) and ``canonical_pack`` (the validated
        ``WM_N1_CANONICAL_PACK`` target, or None) sit alongside the sidecar
        ``receipt_sha256`` for audit.
        """
        side = self._identity_sidecar()
        if side is None:
            raise IdeaAConfigError(
                "an actual fit requires VERIFIED per-item TRAIN vehicle "
                "identities: set WM_N1_TRAIN_IDENTITIES to the identity "
                "sidecar directory built for this pack "
                "(load_identity_sidecar(pack_dir, sidecar_dir)); the corpus "
                "never guesses identities")
        extra_receipts = []
        for pack in range(1, len(self._extra_pack_dirs) + 1):
            extra = self._extra_sidecar(pack)
            if extra is None:
                raise IdeaAConfigError(
                    f"extra corpus pack {self._extra_pack_dirs[pack - 1]!r} "
                    "requires its OWN verified identity sidecar; set "
                    "--extra-identities / WM_N1_EXTRA_TRAIN_IDENTITIES (one "
                    "per extra pack); the corpus never guesses identities")
            extra_sha = getattr(extra, "receipt_sha256", None)
            if not extra_sha:
                raise IdeaAConfigError(
                    "an extra verified identity sidecar exposes no "
                    "receipt_sha256")
            extra_receipts.append(str(extra_sha))
        n_ids = 0
        for i in range(len(self._order)):
            row = self._ids_for_item(self._items[self._order[i]])
            if row is None:
                raise IdeaAConfigError(
                    f"item {self._items[self._order[i]].get('world')!r} "
                    "lacks verified TRAIN identities; the HDV sub-term is "
                    "never silently disabled")
            n_ids += int((row >= 0).sum())
        receipt_sha = getattr(side, "receipt_sha256", None)
        if not receipt_sha:
            raise IdeaAConfigError(
                "the verified identity sidecar exposes no receipt_sha256")
        out = {"status": "verified", "mode": "identity-sidecar",
               "env": "WM_N1_TRAIN_IDENTITIES",
               "pack_alias": str(self.pack_dir),
               "canonical_pack":
                   getattr(self, "_canonical_pack_used", None),
               "n_items": len(self._order),
               "n_verified_id_slots": int(n_ids),
               "receipt_sha256": str(receipt_sha)}
        if self._is_multi_pack:
            out["mode"] = "identity-sidecar+extra"
            out["extra_env"] = "WM_N1_EXTRA_TRAIN_IDENTITIES"
            out["extra_receipt_sha256"] = extra_receipts
            out["extra_identity_dirs"] = list(self._extra_identity_dirs)
        return out

    # -- protocol -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._order)

    def worlds(self) -> list:
        return self._v1.worlds()

    def pair_keys(self) -> list:
        return self._v1.pair_keys()

    def __getitem__(self, i: int) -> dict:
        return self.window(self._v1[i])

    # -- canonical window ---------------------------------------------------
    @staticmethod
    def _lane_of(kin_row: Tensor) -> Tensor:
        """``[..., 7]`` kinematics -> int lane (0/1/2, -1 empty)."""
        one_hot = kin_row[..., _KIN_L0:_KIN_L2 + 1]
        has = one_hot.sum(dim=-1) > 0
        return torch.where(has, one_hot.argmax(dim=-1),
                           torch.full_like(one_hot[..., 0], -1,
                                           dtype=torch.long))

    def window(self, item: dict) -> dict:
        """One packed item -> the full canonical window (batch axis 1)."""
        tick = int(item["tick"])
        veh_kin = np.asarray(self._array("veh_kin")[tick], np.float32)
        hdv_base = np.asarray(self._array("hdv_base")[tick], np.float32)
        event_tok = np.asarray(self._array("event_tok")[tick], np.float32)
        event_mask = np.asarray(self._array("event_mask")[tick], np.float32)
        n_ticks = int(self._array("veh_kin").shape[0])
        end = min(self._run_end(tick), n_ticks)
        horizon = max(0, min(self.horizon_steps, end - tick - 1))

        kin = torch.as_tensor(veh_kin).unsqueeze(0)
        kin_mask = item["veh_mask"].to(torch.bool).unsqueeze(0)
        lane0 = self._lane_of(torch.as_tensor(veh_kin)).unsqueeze(0)
        is_cav = item["veh_step"][:, C.IS_CAV_IDX] > 0.5
        meta = {key: item[key] for key in (
            "world", "arm", "chunk", "split", "t0_s", "pair_key", "policy")}
        # (a) telemetry: carry the v1 item's per-item t0 assertion counts and
        # the (b) exclusion flag into the canonical window's meta, so
        # label_t0_telemetry() sees them (it reads batch["meta"]).
        for key in ("label_t0_rows", "label_t0_violations",
                    "outcome_supervision_excluded"):
            if key in item:
                meta[key] = item[key]
        out = {
            "kin": kin,
            "kin_mask": kin_mask,
            "is_cav": is_cav.unsqueeze(0),
            "hdv_base": torch.as_tensor(hdv_base).unsqueeze(0),
            "baseline_idm": torch.as_tensor(
                hdv_base[:, HDV_BASE_IDM_IDX]).unsqueeze(0),
            "target_accel": item["accel"].unsqueeze(0),
            "target_mask": item["target_mask"].unsqueeze(0),
            "lane_target": item["lane"].unsqueeze(0),
            "counter": item["counter"].unsqueeze(0),
            "arrivals": item["arrivals"].unsqueeze(0),
            "outcome_quantiles": item["outcome_quantiles"].unsqueeze(0),
            "outcome_mask": item["outcome_mask"].unsqueeze(0),
            "collision": item["collision"].unsqueeze(0),
            "event_tok": torch.as_tensor(event_tok).unsqueeze(0),
            "event_mask": torch.as_tensor(event_mask).unsqueeze(0),
            "s0": kin[:, :, _KIN_S],
            "v0": kin[:, :, _KIN_V],
            "a0": kin[:, :, _KIN_A],
            "lane0": lane0,
            "length0": kin[:, :, _KIN_LEN],
            "present0": kin_mask.clone(),
            # top-level world/arm so callers that filter by CAL half can read
            # the item's world without reaching into the meta list
            "world": meta.get("world"),
            "arm": meta.get("arm"),
            "meta": [meta],
        }
        if self.split == "TRAIN":
            ids = self._ids_for_item(item)
            if ids is not None:
                out["vehicle_ids"] = ids.unsqueeze(0)
        if horizon >= 1:
            horizon_max = self.horizon_steps
            k_slots = int(kin.shape[1])
            stop = tick + horizon + 1
            kin_seq = torch.as_tensor(np.asarray(
                self._array("veh_kin")[tick:stop], np.float32))   # [T,K,7]
            plan_seq = torch.as_tensor(np.asarray(
                self._array("cav_plan_ch")[tick:stop], np.float32))  # [T,K,8]
            acc_seq = torch.as_tensor(
                np.asarray(self._array("accel_next")[tick:tick + horizon],
                           np.float32))
            base_seq = torch.as_tensor(np.asarray(
                self._array("hdv_base")[tick:tick + horizon,
                                        ..., HDV_BASE_IDM_IDX], np.float32))
            # ALIVE-SINCE-t0: target_mask[tick+u] is 1 iff the SAME vehicle
            # is still in the slot at tick+u+1 (features/FORMAT.md section 6).
            # A slot reused mid-window must NOT become alive again for the NEW
            # vehicle: the rollout keeps integrating the departed vehicle's
            # state, so a per-step mask would compare the wrong vehicle.  The
            # cumulative product makes alive monotone: once a slot's t0
            # vehicle departs it stays masked, and mid-window entrants are
            # never added.  ``alive_state[t]`` is the t0 vehicle's presence at
            # tick+t (present0 AND the transitions before t); rollout step t
            # predicts tick+t -> tick+t+1 and so uses ``alive_state[t+1]``.
            present0 = kin_mask[0]
            tm = torch.as_tensor(np.asarray(
                self._array("target_mask")[tick:tick + horizon],
                bool)).to(torch.bool)                              # [h,K]
            alive_state = torch.cumprod(
                torch.cat([present0.unsqueeze(0).to(torch.int64),
                           tm.to(torch.int64)], dim=0),
                dim=0).to(torch.bool)                              # [h+1,K]
            mask_seq = alive_state[1:horizon + 1]                  # [h,K]
            # the summary is over the ACTUAL window (never padded); the time
            # tensors are zero-padded to the registered horizon so a batch of
            # runs with different lifetimes collates, with the padded steps
            # masked out of the rollout loss.  The CAV summary uses only the
            # steps each CAV slot is alive: the per-step mask excludes dead
            # steps from the speed/accel statistics, and the trajectory is
            # forward-filled so the displacement feature uses the slot's LAST
            # ALIVE position, never a departed/ reused slot's later state.
            # vectorized forward-fill over ALIVE steps: the last alive row
            # index per (t, slot) is a cummax of the alive positions, and the
            # gathered rows reproduce the per-t loop exactly.
            t_steps = int(alive_state.shape[0])
            ar = torch.arange(t_steps).view(t_steps, 1)
            fill = torch.cummax(
                torch.where(alive_state, ar, torch.zeros_like(ar)),
                dim=0).values                                   # [T,K]
            pos_ff = kin_seq[..., _KIN_S].gather(0, fill)       # [T,K]
            plan_ff = plan_seq.gather(
                0, fill.unsqueeze(-1).expand(-1, -1, plan_seq.shape[-1]))
            cav_alive = (is_cav.unsqueeze(0) & alive_state).unsqueeze(1)
            summary_raw = _cav_motion_summary(
                pos_ff.unsqueeze(1), plan_ff.unsqueeze(1), cav_alive)
            # A11 full-horizon plan: the SAME planned CAV channels, read over
            # [t0, t0+H] (not the 10 s rollout window) and binned.  Only built
            # when the flag is set, so the default batch is unchanged.
            plan_long = None
            plan_horizon_s = getattr(self, "_plan_horizon_s", None)
            if plan_horizon_s is not None:
                from model.paired_effect import long_plan_summary

                n_long_req = max(
                    1, int(round(plan_horizon_s / float(C.DT_S))))
                n_long = max(0, min(n_long_req, end - tick))
                if n_long >= 1:
                    pl = np.asarray(self._array("cav_plan_ch")[
                        tick:tick + n_long], np.float32)
                    sp_l = torch.as_tensor(pl[..., 0]).to(torch.float32) * \
                        float(CAV_TGT_SPEED_SCALE)
                    ac_l = torch.as_tensor(pl[..., 1]).to(torch.float32) * \
                        float(CAV_ACCEL_SCALE)
                    tm_l = torch.as_tensor(np.asarray(
                        self._array("target_mask")[
                            tick:tick + n_long - 1], bool)).to(torch.bool)
                    alive_l = torch.cumprod(torch.cat([
                        kin_mask[0].unsqueeze(0).to(torch.int64),
                        tm_l.to(torch.int64)], dim=0), dim=0).to(torch.bool)
                else:
                    sp_l = torch.zeros(1, kin.shape[1])
                    ac_l = torch.zeros(1, kin.shape[1])
                    alive_l = kin_mask[0].unsqueeze(0).to(torch.bool)
                cav_l = (is_cav.unsqueeze(0) & alive_l).to(
                    torch.float32).unsqueeze(1)
                plan_long = long_plan_summary(
                    sp_l.unsqueeze(1), ac_l.unsqueeze(1), cav_l,
                    horizon_s=float(plan_horizon_s))
            teacher = torch.zeros(horizon_max + 1, 1, k_slots, 4,
                                  dtype=kin_seq.dtype)
            teacher[:horizon + 1, 0, :, TEACHER_S] = kin_seq[..., _KIN_S]
            teacher[:horizon + 1, 0, :, TEACHER_LANE] = \
                self._lane_of(kin_seq).float()
            teacher[:horizon + 1, 0, :, TEACHER_V] = kin_seq[..., _KIN_V]
            # the CAV action of step t is the recorded acceleration that
            # produced tick t+1 (accel_next[tick+t]); HDV slots use the learned
            # residual, so the column is only consumed on CAV slots.
            teacher[0, 0, :, TEACHER_A] = kin_seq[0, :, _KIN_A]
            teacher[1:horizon + 1, 0, :, TEACHER_A] = acc_seq

            def _pad_time(seq):
                pad = torch.zeros(horizon_max - horizon, *seq.shape[1:],
                                  dtype=seq.dtype)
                return torch.cat([seq, pad], dim=0)

            out.update({
                "teacher": teacher,
                "rollout_accel": _pad_time(acc_seq).unsqueeze(1),
                "rollout_baseline_idm": _pad_time(base_seq).unsqueeze(1),
                "rollout_mask": _pad_time(mask_seq.bool()).unsqueeze(1),
                "summary_raw": summary_raw,
            })
            if plan_long is not None:
                out["plan_long"] = plan_long
        return out


    def support_masks(self, i: int) -> dict:
        """Cheap per-item contrast support (``outcome`` taus, ``hdv`` slots).

        Reads only ``target_mask`` + ``veh_step`` for the HDV slots and the
        packed outcome rows for the per-tau availability, so the eligible
        CONTRAST pair pool is precomputed without building a full rollout
        window per item.  When verified TRAIN identities are available the
        result also carries ``ids`` (``int64[K]``, ``-1`` absent); HDV
        eligibility is then common-ID based, not slot intersection.
        """
        if i < 0:
            i += len(self._order)
        if not 0 <= i < len(self._order):
            raise IndexError(f"item position {i} out of range")
        item = self._items[self._order[i]]
        tick = int(item["tick"])
        target = np.asarray(self._v1._array("target_mask")[tick], bool)
        veh = self._v1._array("veh_step")[tick]
        is_cav = np.asarray(veh[:, C.IS_CAV_IDX] > 0.5, bool)
        _, omask, _ = self._v1.outcome_target(i)
        ids = self._ids_for_item(item) if self.split == "TRAIN" else None
        return {"outcome": torch.as_tensor(np.asarray(omask, bool)),
                "hdv": torch.as_tensor(target & ~is_cav),
                "ids": ids}

