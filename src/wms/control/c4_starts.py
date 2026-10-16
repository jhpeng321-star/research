"""POLICYV5: Dreamer start windows from converted C4 episode sources (``data.c4_base``).

A WM config with ``data.c4_base`` (V5DATA, e.g. ``configs/c4_wm_v5.toml``) has no packs, sidecars
or window lists; its TRAIN windows come from ``[[data.c4_sources]]``.  The policy only needs each
window's t0 state, its observed pre-t0 history and entry features, so the start windows are built
here directly from the sources' TRAIN-world episodes:

* one split side per world across ALL the config's sources (:func:`wms.data.c4_pairs.world_roles`,
  as the WM assembly), and only the ``train`` side is used (WM validation worlds never start a
  policy rollout);
* ``horizon_steps = 1`` (t0 state only, as the legacy pack path), each source's own
  ``window_stride_s`` / ``start_s``, the WM config's ``junction_fix``, ``k_slots`` of the base;
* NO jam scan and no mix weights: :class:`wms.control.start_states.StartStateSampler` draws
  uniformly over penetrations, then windows (the legacy policy never oversampled jams either),
  so ``wms.data.jam.compute_jam_flags`` (~40 min on the 63k v5 windows) is never called.

Legacy WM configs (``c4_base`` unset) never reach this module.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from torch.utils.data import Dataset


class C4StartWindows(Dataset):
    """Concatenation of per-source C4 TRAIN window datasets with metadata-only lookup."""

    def __init__(self, parts: Sequence[tuple[str, Dataset]]):
        self.parts = [(str(name), dataset) for name, dataset in parts]
        if not self.parts or any(len(d) == 0 for _, d in self.parts):
            raise ValueError("C4 start windows: every start source needs TRAIN windows")
        self.offsets = []
        total = 0
        for _, dataset in self.parts:
            self.offsets.append(total)
            total += len(dataset)
        self.n = total
        self.receipt = {"mode": "c4-start-windows", "split": "TRAIN", "n_records": total,
                        "sources": {name: {"n_windows": len(d),
                                           "source": d.receipt.get("source"),
                                           "n_episodes": d.receipt.get("n_episodes")}
                                    for name, d in self.parts}}

    def __len__(self) -> int:
        return self.n

    def _locate(self, index: int):
        index = int(index)
        if not 0 <= index < self.n:
            raise IndexError(index)
        part = max(i for i, offset in enumerate(self.offsets) if offset <= index)
        name, dataset = self.parts[part]
        return name, dataset, index - self.offsets[part]

    def __getitem__(self, index: int) -> dict:
        name, dataset, local = self._locate(index)
        record = dict(dataset[local])
        record["meta"] = {**record.get("meta", {}), "start_source": name}
        return record

    def start_meta(self, index: int) -> dict:
        """Window metadata WITHOUT decoding the window (StartStateSampler filtering)."""
        name, dataset, local = self._locate(index)
        loader = dataset._loader
        episode, tick = loader.index[local]
        entry = loader.entries[episode]
        return {"source": "c4_episodes", "world": entry["world"], "arm": entry["arm"],
                "episode": entry["episode"], "tick": int(tick), "start_source": name}


def build_c4_start_windows(spec, path: Callable[[str], str], *, history_s, history_stride_s,
                           entries: bool, sources: Optional[Sequence[str]] = None
                           ) -> C4StartWindows:
    """TRAIN-world start windows of ``spec.c4_sources`` (``spec`` = a WM ``DataConfig``).

    ``path`` resolves (and checks) a logical data name.  ``sources`` = the source names to draw
    starts from (policy ``[data] start_sources``); default = every non-paired source with a
    positive ``mix_ratio``.
    """
    from wms.data.c4_episodes import build_c4_datasets, read_source_manifest
    from wms.data.c4_pairs import world_roles

    if spec.c4_base is None:
        raise ValueError("C4 start windows need a WM config with data.c4_base")
    configured = {src.name: src for src in spec.c4_sources}
    roots = {name: path(src.path) for name, src in configured.items()}
    roles, _ = world_roles([read_source_manifest(root) for root in roots.values()])
    k_slots = int(read_source_manifest(roots[spec.c4_base])["k_slots"])
    if sources is None:
        names = [src.name for src in spec.c4_sources if not src.paired and src.mix_ratio > 0.0]
    else:
        names = [str(name) for name in sources]
        unknown = sorted(set(names) - set(configured))
        if unknown or len(set(names)) != len(names):
            raise ValueError(f"[data] start_sources {names}: unknown or repeated names "
                             f"(WM c4_sources: {sorted(configured)})")
    if not names:
        raise ValueError("no C4 start source selected")
    parts = []
    for name in names:
        src = configured[name]
        built = build_c4_datasets(
            roots[name], horizon_steps=1, history_s=history_s,
            history_stride_s=float(history_stride_s), entries_v2=bool(entries),
            stride_s=float(src.window_stride_s), start_s=src.start_s, k_slots=k_slots,
            role_of=roles, junction_fix=getattr(spec, "junction_fix", "off"),
            allow_empty=bool(src.paired))
        if built["train"] is None:
            raise ValueError(f"C4 start source {name!r} has no TRAIN-world window")
        parts.append((name, built["train"]))
    return C4StartWindows(parts)
