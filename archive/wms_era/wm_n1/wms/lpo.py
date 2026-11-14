"""Leave-one-controller-out (C1) window lists and controller-agnostic filtering.

The C1 protocol trains a world model on TRAIN windows that never contain the
held-out controller (PI) on any CAV, then evaluates on held-out PI states.  The
window lists are produced from metadata only by the P5 agent
(``windows_train_noPI.json`` / ``windows_val.json`` / ``pi_eval_states.json``);
this module reads those lists and selects the matching TRAIN corpus positions.
Nothing here reads an array, an outcome or a sealed source.

The P5 key prefix is the source *batch* (``truearms_chunk2``,
``chunk1_fix27``), which pack_v3 renames to a shared ``chunk`` field and can
therefore collide across batches.  Matching is done on the structured identity
``(batch, world, arm, t0_s)`` with ``chunk`` only as a fallback, so a stale or
mis-prefixed list cannot silently select the wrong run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

WINDOW_SCHEMA = "wm_s_lpo_windows/1"
Key = tuple


def load_window_manifest(path: str | Path) -> dict:
    """Load and shape-check one P5 window list (metadata only)."""
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


def _prefix(record: dict) -> str:
    """P5 batch prefix of a window or corpus item (chunk fallback)."""
    return str(record.get("batch") or record.get("chunk") or "")


def window_key(window: dict) -> Key:
    """Structured identity ``(batch, world, arm, t0_s)`` of a P5 window."""
    return (_prefix(window), str(window["world"]), str(window["arm"]),
            round(float(window.get("t0_s", 0.0)), 3))


def item_key(item: dict) -> Key:
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
    """``(batch_prefix, run_id)`` pairs from a bare P5 key list.

    A key is ``<batch>/<world>__<arm>``; the run id alone is too coarse because
    pack_v3 reuses ``world__arm`` across batches.
    """
    pairs: set[tuple[str, str]] = set()
    for key in load_key_list(path):
        prefix, sep, run = str(key).partition("/")
        pairs.add((prefix, run if sep else ""))
    return pairs


def pack_names(paths: Sequence[str | Path]) -> list[str]:
    return [Path(p).name for p in paths]


def _excluded(item: dict, exclude: set[tuple[str, str]]) -> bool:
    if not exclude:
        return False
    run = f"{item.get('world')}__{item.get('arm')}"
    # Match the source batch only: pack_v3 shares ``chunk`` names across batches,
    # so a chunk match would wrongly drop the same run from a different batch.
    prefix = str(item.get("batch") or item.get("chunk") or "")
    return any(prefix == excluded_batch and run == excluded_run
               for excluded_batch, excluded_run in exclude)


def filter_positions(items: Sequence[dict], packs: Sequence[int],
                     names: Sequence[str], *,
                     keep: Optional[dict[str, set[Key]]] = None,
                     exclude: Optional[set[tuple[str, str]]] = None):
    """Select dataset positions by structured ``(batch, world, arm, t0_s)``.

    ``items[i]`` is the raw metadata of dataset position ``i`` (no arrays read)
    and ``packs[i]`` its pack index.  A ``keep`` key the corpus never provides
    is refused (fail-closed), so a stale/mismatched window list cannot silently
    train on the held-out controller.
    """
    exclude = set(exclude or ())
    keep = None if keep is None else {str(k): set(v) for k, v in keep.items()}
    selected: list[int] = []
    matched: set[tuple[str, Key]] = set()
    dropped = 0
    for i, item in enumerate(items):
        name = names[int(packs[i])]
        if _excluded(item, exclude):
            dropped += 1
            continue
        key = item_key(item)
        if keep is not None:
            if key not in keep.get(name, set()):
                dropped += 1
                continue
            matched.add((name, key))
        selected.append(i)
    if keep is not None:
        requested = {(name, key) for name, keys in keep.items() for key in keys}
        missing = requested - matched
        if missing:
            preview = sorted(missing)[:5]
            raise ValueError(
                f"window-list keys absent from corpus ({len(missing)} total): "
                f"{preview}")
    return selected, {"kept": len(selected), "dropped": dropped,
                      "matched": len(matched),
                      "requested": None if keep is None else
                      sum(len(v) for v in keep.values())}
