"""PackCorpus: narrow adapter around the vendored read-only ``IdeaACorpus``.

``IdeaACorpus`` (vendored unchanged) is a private-attribute-heavy reader of the A11
pack stack.  Everything the data layer needs from it is exposed here as a small public
surface so no other module touches ``corpus._items`` and friends.  Semantics are those of
the legacy reader; this class only renames and groups.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from wms.core.errors import TrainDataRefusal
from wms.data.sources import fixed_sidecar_dir
from wms.legacy.vendor.train.idea_a_corpus import IdeaACorpus

#: Horizon (control ticks of 0.5 s) used to open a corpus when no horizon is given.
DEFAULT_HORIZON_STEPS = 240


class PackCorpus:
    """Read-only TRAIN view of one pack or a composite of several packs."""

    def __init__(self, inner: IdeaACorpus, pack_dirs: Sequence[str | Path]):
        self._inner = inner
        self.pack_dirs = tuple(Path(p) for p in pack_dirs)

    @classmethod
    def open(cls, pack_dir: str | Path, *, extra_pack_dirs: Sequence[str | Path] = (),
             extra_overlay_dirs: Optional[Sequence[str | Path]] = None,
             extra_identity_dirs: Optional[Sequence[str | Path]] = None,
             horizon_steps: int = DEFAULT_HORIZON_STEPS,
             allow_single_run_fallback: bool = False) -> "PackCorpus":
        """Open ``pack_dir`` (plus extra packs as an item-level composite), split TRAIN."""
        try:
            inner = IdeaACorpus(
                str(pack_dir), split="TRAIN", horizon_steps=int(horizon_steps),
                allow_single_run_fallback=allow_single_run_fallback,
                extra_pack_dirs=[str(p) for p in extra_pack_dirs],
                extra_overlay_dir=(None if extra_overlay_dirs is None
                                   else [str(p) for p in extra_overlay_dirs]),
                extra_identity_dirs=([str(p) for p in extra_identity_dirs]
                                     if extra_identity_dirs else None))
        except (ImportError, KeyError, OSError, ValueError, RuntimeError) as exc:
            raise TrainDataRefusal(f"A11 TRAIN window adapter refused: {exc}") from exc
        return cls(inner, [pack_dir, *extra_pack_dirs])

    # -- size / geometry -------------------------------------------------------
    def __len__(self) -> int:
        return len(self._inner)

    @property
    def k_slots(self) -> int:
        return int(self._inner.k_max)

    @property
    def n_runs(self) -> int:
        return len(self._inner._runs)

    @property
    def pack_names(self) -> list[str]:
        """Directory basenames of the opened packs, composite order."""
        return [p.name for p in self.pack_dirs]

    # -- per-window metadata (arrays are never read) ---------------------------
    def meta(self, i: int) -> dict:
        """Raw item metadata of dataset position ``i``."""
        return self._inner._items[self._inner._order[i]]

    def pack_index(self, i: int) -> int:
        return int(self._inner._global_packs[i])

    def manifest_position(self, i: int) -> int:
        """Original per-pack manifest position of dataset position ``i``."""
        return int(self._inner._global_positions[i])

    def item(self, i: int) -> dict:
        """The loader item (metadata + tensors) of dataset position ``i``."""
        return self._inner._v1[i]

    # -- runs ------------------------------------------------------------------
    def run_bounds(self, tick: int) -> Optional[tuple[int, int]]:
        """``(start, end)`` tick range of the run containing ``tick``, else ``None``."""
        for start, end in self._inner._runs:
            if start <= tick < end:
                return start, end
        return None

    def run_start(self, tick: int) -> int:
        """First tick of the run containing ``tick`` (0 when ``tick`` is in no run)."""
        bounds = self.run_bounds(tick)
        return 0 if bounds is None else bounds[0]

    def run_end(self, tick: int) -> int:
        return self._inner._run_end(int(tick))

    def tick_offsets(self) -> Optional[Sequence[int]]:
        """Per-pack global tick offsets of a composite corpus (``None`` for one pack)."""
        return getattr(self._inner._v1, "_tick_offsets", None)

    # -- arrays and windows ----------------------------------------------------
    def array(self, name: str) -> np.ndarray:
        """Memory-mapped pack array (composite-aware)."""
        return self._inner._array(name)

    def window(self, item: dict, *, identities: bool = True) -> dict:
        """Canonical window of ``item``; ``identities=False`` suppresses the identity row.

        A start-shifted item keeps the identity row of its original t0 only by accident
        of slot reuse, so shifted windows are always read without identities.
        """
        if identities:
            return self._inner.window(item)
        original = self._inner._ids_for_item
        self._inner._ids_for_item = lambda _item: None
        try:
            return self._inner.window(item)
        finally:
            del self._inner._ids_for_item
            assert self._inner._ids_for_item == original

    def shifted_item(self, item: dict, tick: int) -> dict:
        """The same run's item re-based to a later ``tick`` (real pre-t0 history exists).

        Only per-tick arrays at ``tick`` are replaced; world/arm/plan/outcome rows are
        those of the original item.  The slot->vehicle identity row of the original t0
        does not apply, so shifted windows carry ``vehicle_ids = -1``; same-vehicle
        continuity inside the window is given by the pack's ``target_mask``.
        """
        out = dict(item)
        out["tick"] = int(tick)
        out["veh_step"] = torch.as_tensor(np.asarray(self.array("veh_step")[tick], np.float32))
        out["veh_mask"] = torch.as_tensor(np.asarray(self.array("veh_mask")[tick]).astype(bool))
        out["accel"] = torch.as_tensor(np.asarray(self.array("accel_next")[tick], np.float32))
        out["target_mask"] = torch.as_tensor(
            np.asarray(self.array("target_mask")[tick]).astype(bool))
        out["lane"] = torch.as_tensor(np.asarray(self.array("lane_next")[tick]).astype(np.int64))
        out["counter"] = torch.as_tensor(np.asarray(self.array("counter")[tick], np.float32))
        out["arrivals"] = torch.as_tensor(np.asarray(self.array("arrivals")[tick], np.float32))
        return out

    def use_identity_dir(self, directory: str) -> None:
        """Override the process-wide ``WM_N1_TRAIN_IDENTITIES`` for this corpus only."""
        self._inner._sidecar_dir = partial(fixed_sidecar_dir, str(directory))
