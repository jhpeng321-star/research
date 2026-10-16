"""Window lists and window selection (leave-one-controller-out, metadata only).

The C1 protocol trains on TRAIN windows that never contain the held-out controller (PI,
later the wall) on any CAV.  Window lists (``wm_s_lpo_windows/1``) are produced from
metadata only by :mod:`wms.data.window_lists`; this module reads them and selects the
matching corpus positions.  Nothing here reads an array, an outcome or a restricted source.

The list key prefix is the source *batch* (``truearms_chunk2``, ``chunk1_fix27``), which
pack_v3 renames to a shared ``chunk`` field and can therefore collide across batches.
Matching is on the structured identity ``(batch, world, arm, t0_s)`` with ``chunk`` only as
a fallback, so a stale or mis-prefixed list cannot silently select the wrong run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Iterable, Mapping, Optional, Sequence

import numpy as np

from wms.core.errors import TrainDataRefusal
from wms.data.corpus import PackCorpus

WINDOW_SCHEMA = "wm_s_lpo_windows/1"
Key = tuple


# --------------------------------------------------------------- list readers ----
def load_window_manifest(path: str | Path) -> dict:
    """Load and shape-check one window list (metadata only)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = str(data.get("schema", WINDOW_SCHEMA))
    if schema != WINDOW_SCHEMA:
        raise ValueError(f"unexpected window-list schema {schema!r} in {path}")
    windows = data.get("windows")
    if not isinstance(windows, list) or "n_windows" not in data:
        raise ValueError(f"window list {path} has no windows/n_windows")
    if int(data["n_windows"]) != len(windows):
        raise ValueError(
            f"window list {path} n_windows {data['n_windows']} != {len(windows)}")
    return data


def _prefix(record: Mapping) -> str:
    """Batch prefix of a window or corpus item (``chunk`` fallback)."""
    return str(record.get("batch") or record.get("chunk") or "")


def window_key(window: Mapping) -> Key:
    """Structured identity ``(batch, world, arm, t0_s)`` of a listed window."""
    return (_prefix(window), str(window["world"]), str(window["arm"]),
            round(float(window.get("t0_s", 0.0)), 3))


def item_key(item: Mapping) -> Key:
    """Structured identity ``(batch, world, arm, t0_s)`` of a corpus item."""
    return (_prefix(item), str(item.get("world")), str(item.get("arm")),
            round(float(item.get("t0_s", 0.0)), 3))


def window_keys(path: str | Path, *,
                packs: Optional[Iterable[str]] = None) -> dict[str, set[Key]]:
    """``pack -> {structured window key}`` for the requested packs."""
    data = load_window_manifest(path)
    want = None if packs is None else {str(p) for p in packs}
    out: dict[str, set[Key]] = {}
    for window in data["windows"]:
        pack = str(window.get("pack"))
        if want is not None and pack not in want:
            continue
        out.setdefault(pack, set()).add(window_key(window))
    return out


def load_key_list(path: str | Path) -> list[str]:
    """A bare ``keys`` list (e.g. ``pi_excluded_runs.json``), no schema gate."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [str(k) for k in data.get("keys", [])]
    return [str(k) for k in data]


def exclude_runs(path: str | Path) -> set[tuple[str, str]]:
    """``(batch_prefix, run_id)`` pairs from a bare key list.

    A key is ``<batch>/<world>__<arm>``; the run id alone is too coarse because pack_v3
    reuses ``world__arm`` across batches.
    """
    pairs: set[tuple[str, str]] = set()
    for key in load_key_list(path):
        prefix, sep, run = str(key).partition("/")
        pairs.add((prefix, run if sep else ""))
    return pairs


def load_exclusion_keys(paths: Optional[Iterable[str | Path]]) -> set[str]:
    """Normalise exclusion files to a set of pack run keys.

    Accepts the ``pi_excluded_runs.json`` shape (``{"keys": [...]}``), a
    ``{"windows": [{"key": ...}, ...]}`` shape, or a bare list of key strings.
    """
    keys: set[str] = set()
    for path in paths or ():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "keys" in data:
            keys.update(str(k) for k in data["keys"])
        elif isinstance(data, dict) and "windows" in data:
            keys.update(str(w["key"]) for w in data["windows"])
        elif isinstance(data, list):
            keys.update(str(w["key"] if isinstance(w, dict) else w) for w in data)
        else:
            raise ValueError(f"unrecognised exclusion list shape: {path}")
    return keys


def load_window_list(path: str | Path) -> set[tuple[str, str]]:
    """Normalise a validation window list to ``(pack, key)`` pairs.

    The pack name scopes the key because pack_damping re-packs the base runs of pack_v3
    under identical run keys.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    windows = data["windows"] if isinstance(data, dict) and "windows" in data else data
    pairs: set[tuple[str, str]] = set()
    for w in windows:
        if not isinstance(w, dict) or "key" not in w or "pack" not in w:
            raise ValueError(f"window-list entry lacks pack/key: {w!r}")
        pairs.add((str(w["pack"]), str(w["key"])))
    return pairs


# -------------------------------------------------------------- core filtering ----
def _excluded(item: Mapping, exclude: Collection[tuple[str, str]]) -> bool:
    if not exclude:
        return False
    run = f"{item.get('world')}__{item.get('arm')}"
    # Match the source batch only: pack_v3 shares ``chunk`` names across batches, so a
    # chunk match would wrongly drop the same run from a different batch.
    prefix = _prefix(item)
    return any(prefix == batch and run == excluded_run for batch, excluded_run in exclude)


def filter_positions(items: Sequence[Mapping], packs: Sequence[int], names: Sequence[str], *,
                     keep: Optional[Mapping[str, set[Key]]] = None,
                     exclude: Optional[set[tuple[str, str]]] = None):
    """Select dataset positions by structured ``(batch, world, arm, t0_s)``.

    ``items[i]`` is the raw metadata of dataset position ``i`` (no arrays read) and
    ``packs[i]`` its pack index.  A ``keep`` key the corpus never provides is refused
    (fail-closed), so a stale/mismatched list cannot silently train on a held-out controller.
    """
    exclude = set(exclude or ())
    keep_sets = None if keep is None else {str(k): set(v) for k, v in keep.items()}
    selected: list[int] = []
    matched: set[tuple[str, Key]] = set()
    dropped = 0
    for i, item in enumerate(items):
        name = names[int(packs[i])]
        if _excluded(item, exclude):
            dropped += 1
            continue
        key = item_key(item)
        if keep_sets is not None:
            if key not in keep_sets.get(name, set()):
                dropped += 1
                continue
            matched.add((name, key))
        selected.append(i)
    if keep_sets is not None:
        requested = {(name, key) for name, keys in keep_sets.items() for key in keys}
        missing = requested - matched
        if missing:
            preview = sorted(missing)[:5]
            raise ValueError(
                f"window-list keys absent from corpus ({len(missing)} total): {preview}")
    return selected, {"kept": len(selected), "dropped": dropped, "matched": len(matched),
                      "requested": None if keep_sets is None else
                      sum(len(v) for v in keep_sets.values())}


def resolve_window_pack_names(corpus: PackCorpus, names: Sequence[str],
                              keep_raw: Mapping[str, set]) -> tuple[dict[str, set], dict]:
    """Map window-list pack names onto the corpus pack names (FAIL CLOSED).

    Window lists name packs canonically (``pack_v3``), while a deployed pack directory may be
    staged under a different basename (``pack``, ``pack_v3_sparse``).  Matching by basename
    alone silently drops *every* window of the renamed pack: the entries are filtered out
    before the keep-set is compared, so they are neither trained nor reported missing.  Each
    unmatched list name is resolved to the unique corpus pack whose TRAIN items cover its
    window keys (metadata only); the load is refused if it is absent or ambiguous.  Returns
    the resolved keep-set and the ``{window-list pack name: corpus pack name}`` mapping.
    """
    name_index = {str(n): i for i, n in enumerate(names)}
    if len(name_index) != len(names):
        raise TrainDataRefusal("pack names must be unique for window-list matching")
    mapping = {str(q): str(q) for q in keep_raw if str(q) in name_index}
    if set(mapping) == set(str(q) for q in keep_raw):
        return dict(keep_raw), mapping
    unmatched = sorted(str(q) for q in keep_raw if str(q) not in name_index)
    free = [i for i, n in enumerate(names) if n not in keep_raw]
    if len(unmatched) > len(free):
        raise TrainDataRefusal(
            f"window-list pack name(s) {unmatched} do not match any pack directory "
            f"{list(names)}; refusing rather than silently dropping them")
    keys_by_pack: dict[int, set] = {}
    for i in range(len(corpus)):
        keys_by_pack.setdefault(corpus.pack_index(i), set()).add(item_key(corpus.meta(i)))
    used: set[int] = set()
    for q in unmatched:
        best_p, best, ties = None, 0, 0
        for p in free:
            if p in used:
                continue
            n = len(set(keep_raw[q]) & keys_by_pack.get(p, set()))
            if n > best:
                best, best_p, ties = n, p, 1
            elif n == best and n > 0:
                ties += 1
        if best_p is None or best <= 0 or ties > 1:
            raise TrainDataRefusal(
                f"cannot uniquely resolve window-list pack {q!r} to a pack directory "
                f"(best overlap {best}, ties {ties}); refusing")
        mapping[q] = str(names[best_p])
        used.add(best_p)
    resolved: dict[str, set] = {}
    for q, keys in keep_raw.items():
        resolved.setdefault(mapping[str(q)], set()).update(keys)
    return resolved, mapping


def select_lpo_positions(corpus: PackCorpus, names: Sequence[str], *,
                         window_list: Optional[str | Path] = None,
                         exclude_window_list: Optional[str | Path | Sequence[str | Path]] = None):
    """Dataset positions kept by the C1 leave-controller-out window filter.

    Reads only the raw per-item metadata, never an array.  Returns ``(positions, receipt)``.
    """
    items = [corpus.meta(i) for i in range(len(corpus))]
    packs = [corpus.pack_index(i) for i in range(len(corpus))]
    keep = None
    resolution = None
    if window_list is not None:
        keep, mapping = resolve_window_pack_names(
            corpus, names, window_keys(window_list, packs=None))
        resolution = {k: v for k, v in mapping.items() if k != v} or None
    exclude = None
    exclude_paths: Optional[list] = None
    if exclude_window_list is not None:
        exclude_paths = (list(exclude_window_list)
                         if isinstance(exclude_window_list, (list, tuple))
                         else [exclude_window_list])
        exclude = set()
        for path in exclude_paths:
            exclude |= exclude_runs(path)
    selected, receipt = filter_positions(items, packs, names, keep=keep, exclude=exclude)
    receipt.update({"window_list": None if window_list is None else str(window_list),
                    "pack_name_resolution": resolution,
                    "exclude_window_list": (None if exclude_paths is None
                                            else [str(p) for p in exclude_paths])})
    return selected, receipt


# ----------------------------------------------------------------- selection -----
@dataclass(frozen=True)
class WindowSelection:
    """Kept dataset positions of one window list, with per-pack counts."""

    positions: tuple[int, ...]
    receipt: dict = field(compare=False)
    kept_by_pack: dict[str, int] = field(default_factory=dict)


def select_windows(corpus: PackCorpus, *, window_list: Optional[str | Path] = None,
                   exclude_window_list: Optional[str | Path | Sequence[str | Path]] = None,
                   names: Optional[Sequence[str]] = None) -> WindowSelection:
    """Select corpus positions by ``window_list`` minus ``exclude_window_list``."""
    names = list(names) if names else corpus.pack_names
    if len(names) != len(corpus.pack_dirs):
        raise TrainDataRefusal(
            f"pack_name_list has {len(names)} entries for {len(corpus.pack_dirs)} packs")
    positions, receipt = select_lpo_positions(
        corpus, names, window_list=window_list, exclude_window_list=exclude_window_list)
    counts: dict[str, int] = {}
    for i in positions:
        name = names[corpus.pack_index(i)]
        counts[name] = counts.get(name, 0) + 1
    return WindowSelection(tuple(positions), receipt, dict(sorted(counts.items())))


def window_counts(corpus: PackCorpus, *, train_list: str | Path, val_list: str | Path,
                  exclude_window_list: Optional[str | Path | Sequence[str | Path]] = None,
                  ) -> dict[str, tuple[int, int]]:
    """``{pack: (n_train, n_val)}`` of the two window lists (e.g. pack_v3 -> (294, 47))."""
    train = select_windows(corpus, window_list=train_list,
                           exclude_window_list=exclude_window_list)
    val = select_windows(corpus, window_list=val_list,
                         exclude_window_list=exclude_window_list)
    return {name: (train.kept_by_pack.get(name, 0), val.kept_by_pack.get(name, 0))
            for name in sorted(set(train.kept_by_pack) | set(val.kept_by_pack))}


# ----------------------------------------------------- controller exclusion --------
class ControllerExclusion:
    """Read-only per-window controller exclusion, independent of list filenames.

    Used as the ``item_filter`` of :meth:`TrajectoryDataset.from_real_packs`: refuses
    held-out worlds (``heldout``) and, with ``leave_pi_wall_out``, drops windows in which any
    CAV is PI- or wall-owned at any step of the window horizon.
    """

    def __init__(self, *, horizon_steps, heldout=None, leave_pi_wall_out=False):
        self.horizon_steps = int(horizon_steps)
        self.heldout = heldout
        self.leave_pi_wall_out = leave_pi_wall_out

    def __call__(self, corpus: PackCorpus, positions: Sequence[int]):
        kept, dropped = [], 0
        for i in positions:
            meta = corpus.meta(i)
            if self.heldout is not None:
                self.heldout.refuse_worlds([meta["world"]], where="original TRAIN windows")
            if self.leave_pi_wall_out and self._controlled_by_pi_or_wall(corpus, meta):
                dropped += 1
                continue
            kept.append(i)
        return kept, {"leave_pi_wall_out": self.leave_pi_wall_out,
                      "controller_windows_dropped": dropped, "n_kept": len(kept),
                      "held_world_guard": self.heldout is not None}

    def _controlled_by_pi_or_wall(self, corpus: PackCorpus, meta: Mapping) -> bool:
        tick = int(meta["tick"])
        end = min(corpus.run_end(tick), tick + self.horizon_steps + 1)
        for t in range(tick, end):
            row = np.asarray(corpus.array("veh_step")[t])
            cav = np.asarray(corpus.array("veh_mask")[t], bool) & (row[:, 7] > .5)
            if not np.isfinite(row[cav, 8:14]).all():
                raise TrainDataRefusal("nonfinite controller ownership channels")
            if (cav & ((row[:, 10] > .5) | (row[:, 11] > .5) | (row[:, 13] > .5))).any():
                return True
        return False
