"""Deterministic subsets of a large state file, selected without loading it whole.

The file is streamed twice: once for ``(world, t0_s, key)`` metadata, once to pull the chosen
records.  Records are always returned sorted by ``(world, t0_s, key)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .errors import EvalError
from .normalize import iter_windows

StateId = tuple[str, float, str]


def _sort_id(record: dict) -> StateId:
    return (str(record.get("world")), float(record.get("t0_s", 0.0)), str(record.get("key")))


def _fetch(path: Path, chosen: set[StateId]) -> list[dict]:
    found = [record for record in iter_windows(path) if _sort_id(record) in chosen]
    return sorted(found, key=_sort_id)


def select_states(path: str | Path, subset: str | None) -> list[dict] | Iterator[dict]:
    """All states of ``path`` (streamed) or the subset named by ``kind:N`` (see ``EvalSetConfig``)."""
    if not subset:
        return iter_windows(path)
    kind, _, count = str(subset).partition(":")
    if kind == "first" and count.isdigit():
        return first_states(path, int(count))
    if kind == "first_per_world" and count.isdigit():
        return first_per_world(path, int(count))
    raise EvalError(f"unknown state subset {subset!r}; use first:N or first_per_world:N")


def first_states(path: str | Path, n: int) -> list[dict]:
    """The first ``n`` states sorted by ``(world, t0_s, key)``."""
    ids = sorted(_sort_id(record) for record in iter_windows(path))
    return _fetch(Path(path), set(ids[:n]))


def first_per_world(path: str | Path, n_worlds: int) -> list[dict]:
    """The earliest state (by ``(t0_s, key)``) of each of the first ``n_worlds`` worlds.

    Worlds are taken in sorted order; this is the 20-state quick diagnostic subset when
    ``n_worlds`` is 20.
    """
    best: dict[str, StateId] = {}
    for record in iter_windows(path):
        sid = _sort_id(record)
        if sid[0] not in best or (sid[1], sid[2]) < (best[sid[0]][1], best[sid[0]][2]):
            best[sid[0]] = sid
    chosen = {best[world] for world in sorted(best)[:n_worlds]}
    return _fetch(Path(path), chosen)
