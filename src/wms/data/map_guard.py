"""Seed / map guards of the WM data pipeline on registered non-original maps (WM-A, 2026-11-06).

The WM TRAIN pipeline may ingest wz3to2 episodes only from N1's registered TRAIN seed range
(``tools/coopmwm/map_scenarios``: TRAIN 989200-989599, HELD-OUT 989600-989699; the merge-map
evaluation ranges 964000-964199 / 962xxx / 977xxx are never used on any map).  Fail closed: an
episode whose seed is outside its map's TRAIN range, or whose map differs from the process road,
is refused before any array is written.  The original merge map keeps its existing guards
(``wms.core.guard`` / ``wms.data.mixdata.HeldOutGuard``) and is not touched here.
"""
from __future__ import annotations

from wms.core.errors import TrainDataRefusal

#: map -> registered seed ranges (inclusive).  Mirrors N1's world registry (c4/nm-abstract
#: 65d5f619 ``map_scenarios``); ``tests/wz`` cross-checks it once that module is importable.
REGISTERED_SEEDS = {
    "wz3to2": {"train": (989200, 989599), "heldout": (989600, 989699)},
}
#: Never on any map (merge-map evaluation / sealed ranges).
FORBIDDEN_SEED_RANGES = ((964000, 964199), (962000, 962999), (977000, 977999))


class MapDataRefusal(TrainDataRefusal):
    """A rollout cannot enter the WM TRAIN pipeline of this map (fail closed)."""


def _inside(seed: int, lo_hi) -> bool:
    return int(lo_hi[0]) <= int(seed) <= int(lo_hi[1])


def check_train_seed(map_name: str, seed: int) -> None:
    """Refuse a seed outside ``map_name``'s registered TRAIN range (non-original maps only)."""
    seed = int(seed)
    if any(_inside(seed, r) for r in FORBIDDEN_SEED_RANGES):
        raise MapDataRefusal(f"seed {seed} is in a forbidden evaluation range")
    ranges = REGISTERED_SEEDS.get(str(map_name))
    if ranges is None:
        raise MapDataRefusal(f"map {map_name!r} has no registered seed ranges")
    if _inside(seed, ranges["heldout"]):
        raise MapDataRefusal(f"{map_name}: seed {seed} is HELD-OUT (refused for WM training)")
    if not _inside(seed, ranges["train"]):
        raise MapDataRefusal(f"{map_name}: seed {seed} is outside the TRAIN range {ranges['train']}")


def check_episode_map(episode_map: str | None, road) -> None:
    """The episode's recorded map must be the process road's map (no cross-map mixing)."""
    name = episode_map or "merge_original"
    if name != road.name:
        raise MapDataRefusal(f"episode map {name!r} != process road {road.name!r} (set "
                             "WMS_MAP_SPEC / COOPMWM_MAP_SPEC to the map's MAP_SPEC.json)")
