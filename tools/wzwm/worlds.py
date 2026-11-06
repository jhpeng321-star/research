"""World bookkeeping for the wz3to2 WM fine-tune (WM-E): seed ranges, split by world, manifests.

Every training/validation window of the fine-tune comes from C4 episode sources
(``C4_EPISODES_MANIFEST.json``, schema ``wms_c4_episode_source/1``) whose episodes carry a
``world`` id and a ``split_role``.  The trainer (``wms.data.c4_pairs.world_roles``) puts a world on
the validation side everywhere if ANY source lists it as validation, so the split is by world by
construction; this module re-derives those roles outcome-blind, refuses every world that may not
be trained on, and writes the full TRAIN/validation world list that goes into the checkpoint
manifest (WM-D qualification hard-fails on any overlap with its reserved worlds).

Ranges (registered 2026-11-06, REGISTRATION.md 14:37 and the MAIN/WM-D relay):
* wz3to2 training-registry worlds 989000-989599 (989000-989199 map-namespaced NEWMAP/PLANGEN2/WMDATA
  range - WM-B's WMDATA tapes are 989100-989199, REGISTRATION 16:10 - plus the TRAIN range
  989200-989599); 989600-989699 is the HELD-OUT evaluation range.
* WM-D qualification reserves 989200-989216, 989350-989353, 989400-989402, 989580-989599:
  never in WM training OR validation.
* Never 964000-964199, 962xxx, 977xxx (merge-map evaluation/sealed ranges).
No outcome is read here: only world ids, roles and episode counts.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Iterable, Mapping, Sequence

TRAIN_RANGE = (989000, 989599)
HELDOUT_RANGE = (989600, 989699)
RESERVED_QUALIFICATION = ((989200, 989216), (989350, 989353), (989400, 989402), (989580, 989599))
FORBIDDEN = ((964000, 964199), (962000, 962999), (977000, 977999))
SCHEMA = "wzwm.worlds/1"
SPLIT_SALT = "wzwm-wz3to2-split-v1"


class WorldError(ValueError):
    """A world that may not enter the fine-tune (fail closed, nothing is trained)."""


def world_seed(world) -> int:
    """The 6-digit world seed inside a world id (``989203``, ``"989203"``, ``"w989203"``,
    ``"wz3to2_989203_c4"``).  Exactly one distinct 6-digit number is required."""
    if isinstance(world, bool):
        raise WorldError(f"world id {world!r} is not a seed")
    if isinstance(world, int):
        return int(world)
    found = set(re.findall(r"(?<!\d)(\d{6})(?!\d)", str(world)))
    if len(found) != 1:
        raise WorldError(f"cannot read one 6-digit world seed from {world!r} (found {sorted(found)})")
    return int(found.pop())


def _inside(seed: int, ranges: Iterable[Sequence[int]]) -> bool:
    return any(lo <= seed <= hi for lo, hi in ranges)


def problems(seed: int) -> list[str]:
    """Every rule a seed violates (empty = admissible for WM training or validation)."""
    out = []
    if _inside(seed, FORBIDDEN):
        out.append("forbidden merge-map evaluation range")
    if HELDOUT_RANGE[0] <= seed <= HELDOUT_RANGE[1]:
        out.append("wz3to2 HELD-OUT evaluation range 989600-989699")
    elif not TRAIN_RANGE[0] <= seed <= TRAIN_RANGE[1]:
        out.append("outside the wz3to2 training range 989000-989599")
    if _inside(seed, RESERVED_QUALIFICATION):
        out.append("reserved for WM-D qualification")
    return out


def check_worlds(worlds: Iterable) -> list[int]:
    """Sorted unique seeds; raises :class:`WorldError` listing every inadmissible world."""
    seeds, bad = set(), {}
    for w in worlds:
        s = world_seed(w)
        seeds.add(s)
        p = problems(s)
        if p:
            bad[str(w)] = p
    if bad:
        raise WorldError("inadmissible worlds for the wz3to2 fine-tune: "
                         + "; ".join(f"{w}: {', '.join(p)}" for w, p in sorted(bad.items())))
    return sorted(seeds)


def split_worlds(worlds: Iterable, *, val_frac: float = 0.2, salt: str = SPLIT_SALT
                 ) -> tuple[list[str], list[str]]:
    """Outcome-blind world split: the ``ceil(frac * n)`` smallest ``sha256(salt:world)`` are
    validation (at least one, never all; none when fewer than two worlds)."""
    unique = sorted({str(w) for w in worlds})
    if not 0.0 <= float(val_frac) < 1.0:
        raise ValueError("val_frac must be in [0, 1)")
    if val_frac <= 0.0 or len(unique) < 2:
        return unique, []
    n_val = min(len(unique) - 1, max(1, math.ceil(float(val_frac) * len(unique))))
    ranked = sorted(unique, key=lambda w: hashlib.sha256(f"{salt}:{w}".encode()).hexdigest())
    val = sorted(ranked[:n_val])
    return sorted(set(unique) - set(val)), val


def roles(docs: Sequence[Mapping]) -> tuple[dict[str, str], list[str]]:
    """``wms.data.c4_pairs.world_roles`` semantics: validation in any source = validation everywhere."""
    seen: dict[str, set[str]] = {}
    for doc in docs:
        for e in doc["episodes"]:
            seen.setdefault(str(e["world"]), set()).add(str(e.get("split_role")))
    out = {w: ("val" if "val" in r else "train") for w, r in seen.items()}
    return out, sorted(w for w, r in seen.items() if len(r) > 1)


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def worlds_manifest(sources: Mapping[str, Mapping], *, map_name: str = "wz3to2") -> dict:
    """Full TRAIN/validation world list over every bound source (fail closed).

    ``sources``: ``{source name: C4 episode-source manifest dict}``.  Raises :class:`WorldError`
    on an inadmissible world, on a source of another map, when the TRAIN or validation side is
    empty, or when a world would sit on both sides.
    """
    if not sources:
        raise WorldError("no data sources bound")
    docs = list(sources.values())
    for name, doc in sources.items():
        if doc.get("split", "TRAIN").upper() != "TRAIN":
            raise WorldError(f"source {name}: split {doc.get('split')!r} is not TRAIN")
        got = doc.get("map")
        if got is not None and got != map_name:
            raise WorldError(f"source {name}: map {got!r} != {map_name!r}")
        if not doc.get("episodes"):
            raise WorldError(f"source {name}: no episodes")
    role_of, overridden = roles(docs)
    check_worlds(role_of)
    train = sorted((w for w, r in role_of.items() if r == "train"), key=world_seed)
    val = sorted((w for w, r in role_of.items() if r == "val"), key=world_seed)
    if not train:
        raise WorldError("no TRAIN worlds")
    if not val:
        raise WorldError("no validation worlds: the sources carry no world-grouped validation split")
    if set(train) & set(val):
        raise WorldError("a world is on both split sides")
    both = {world_seed(w) for w in train} & {world_seed(w) for w in val}
    if both:
        raise WorldError(f"world seeds on both split sides (cells of one seed split apart): {sorted(both)}")
    per_source = {}
    for name, doc in sources.items():
        eps = doc["episodes"]
        per_source[name] = {
            "episodes": len(eps),
            "train_episodes": sum(role_of[str(e["world"])] == "train" for e in eps),
            "val_episodes": sum(role_of[str(e["world"])] == "val" for e in eps),
            "worlds": sorted({str(e["world"]) for e in eps}, key=world_seed),
            "manifest_val_frac": doc.get("val_frac"), "manifest_split_seed": doc.get("split_seed"),
            "map": doc.get("map"), "road_digest": doc.get("road_digest"),
        }
    body = {
        "schema": SCHEMA, "map": map_name,
        "rule": "world-grouped: a world is validation everywhere if any source lists it as "
                "validation (wms.data.c4_pairs.world_roles), no seed on both sides; seeds "
                "989000-989599 only; WM-D "
                "reserved, held-out and forbidden ranges refused",
        "train_worlds": train, "val_worlds": val,
        "train_seeds": [world_seed(w) for w in train], "val_seeds": [world_seed(w) for w in val],
        "n_train_worlds": len(train), "n_val_worlds": len(val),
        "role_overridden_worlds": overridden,
        "reserved_qualification_ranges": [list(r) for r in RESERVED_QUALIFICATION],
        "heldout_range": list(HELDOUT_RANGE), "train_range": list(TRAIN_RANGE),
        "sources": per_source,
    }
    body["sha256"] = canonical_sha(body)
    return body


def overlap(manifest: Mapping, reserved: Iterable[Sequence[int]] = RESERVED_QUALIFICATION) -> list[int]:
    """Seeds of a worlds manifest (train AND val) inside the reserved ranges (WM-D's check)."""
    seeds = list(manifest.get("train_seeds", [])) + list(manifest.get("val_seeds", []))
    return sorted(s for s in seeds if _inside(int(s), reserved))
