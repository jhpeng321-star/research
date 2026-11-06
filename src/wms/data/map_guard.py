"""Seed / map guards of the WM data pipeline on registered non-original maps (WM-A, 2026-11-06).

wz3to2 seed registry (N1 ``tools/coopmwm/map_scenarios`` + MAIN 2026-11-06):
* WM TRAIN/val data: the registered WM-data range 989100-989199 AND N1's TRAIN range 989200-989599,
  MINUS the qualification worlds WM-D reserved inside it (989200-989216, 989350-989353,
  989400-989402, 989580-989599; MAIN message 2026-11-06; the reservation file itself is never
  read here);
* refused: the HELD-OUT range 989600-989699, WM-D's reserved worlds, anything else outside the
  accepted ranges, and the merge-map evaluation / sealed ranges 964000-964199, 962xxx, 977xxx
  (never on any map).
Fail closed: an episode whose seed is not accepted, or whose map differs from the process road,
is refused before any array is written.  The original merge map keeps its existing guards
(``wms.core.guard`` / ``wms.data.mixdata.HeldOutGuard``) and is not touched here.
"""
from __future__ import annotations

from wms.core.errors import TrainDataRefusal

#: map -> seed ranges (inclusive).  ``train`` = accepted for WM training/validation;
#: ``reserved`` = WM-D qualification worlds inside ``train`` (refused); ``heldout`` = refused.
REGISTERED_SEEDS = {
    "wz3to2": {
        "train": ((989100, 989199), (989200, 989599)),
        "reserved": ((989200, 989216), (989350, 989353), (989400, 989402), (989580, 989599)),
        "heldout": ((989600, 989699),),
    },
}
#: Never on any map (merge-map evaluation / sealed ranges).
FORBIDDEN_SEED_RANGES = ((964000, 964199), (962000, 962999), (977000, 977999))


class MapDataRefusal(TrainDataRefusal):
    """A rollout cannot enter the WM TRAIN pipeline of this map (fail closed)."""


def _inside(seed: int, lo_hi) -> bool:
    return int(lo_hi[0]) <= int(seed) <= int(lo_hi[1])


def _any(seed: int, ranges) -> bool:
    return any(_inside(seed, r) for r in ranges)


def check_train_seed(map_name: str, seed: int) -> None:
    """Refuse a seed that is not accepted for ``map_name``'s WM TRAIN/val data."""
    seed = int(seed)
    if _any(seed, FORBIDDEN_SEED_RANGES):
        raise MapDataRefusal(f"seed {seed} is in a forbidden evaluation range")
    ranges = REGISTERED_SEEDS.get(str(map_name))
    if ranges is None:
        raise MapDataRefusal(f"map {map_name!r} has no registered seed ranges")
    if _any(seed, ranges["heldout"]):
        raise MapDataRefusal(f"{map_name}: seed {seed} is HELD-OUT (refused for WM training)")
    if _any(seed, ranges["reserved"]):
        raise MapDataRefusal(f"{map_name}: seed {seed} is a reserved WM qualification world")
    if not _any(seed, ranges["train"]):
        raise MapDataRefusal(f"{map_name}: seed {seed} is outside the WM TRAIN ranges "
                             f"{ranges['train']}")


def train_seed_ok(map_name: str, seed: int) -> bool:
    try:
        check_train_seed(map_name, seed)
    except MapDataRefusal:
        return False
    return True


def check_episode_map(episode_map: str | None, road) -> None:
    """The episode's recorded map must be the process road's map (no cross-map mixing)."""
    name = episode_map or "merge_original"
    if name != road.name:
        raise MapDataRefusal(f"episode map {name!r} != process road {road.name!r} (set "
                             "WMS_MAP_SPEC / COOPMWM_MAP_SPEC to the map's MAP_SPEC.json)")
