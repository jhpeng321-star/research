"""RUN2 E1: build a larger stratified TRAIN evaluation set from the packs.

The C1 materialiser (``wms_eval.materialize``) turns one metadata window into
one self-contained state record.  This module adds the RUN2 path-1 selection
layer: it reads the frozen pack ``INDEX.json`` run lists (TRAIN only), stratifies
them by world / demand / penetration / controller arm, and streams one state at
a time to a JSONL file so neither the set nor a pack is ever held in RAM.

Every record carries:

* ``initial_state`` and ``recorded`` (own controller arm + ``native``/``pi``
  siblings resolved by world/t0, i.e. the PI-native pair "where available"),
* ``world`` / ``world_meta`` (world, demand, penetration pct, t0),
* ``arms`` and ``paired`` (whether both a native and a PI recorded trajectory
  exist for this world/t0),
* ``counter`` when the pack ships the RSU-observable t0 entrance counter, so
  ``--entries`` can be applied identically to every spec.

Packs stay ``mmap_mode='r'``; exactly one state is materialised at a time.
Only ``split == "TRAIN"`` runs are considered, so no CALIBRATION/sealed window
can leak in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .materialize import (
    MaterializeError,
    Pack,
    Identity,
    build_state,
    world_meta,
)

TRAIN_SPLIT = "TRAIN"
DEFAULT_TARGET = 200
DEFAULT_OVERSAMPLE = 2.0
DEFAULT_MAX_PER_WORLD = 3
ARMS = ("native", "pi")


class R2SetError(ValueError):
    """The requested RUN2 evaluation set cannot be built."""


def load_packs(pack_args: Sequence[str], identity_args: Sequence[str]):
    """Parse ``NAME=PATH`` argument pairs into Pack / Identity maps."""
    packs = {name: Pack(name, Path(path)) for name, path in _parse_pairs(pack_args).items()}
    identities = {name: Identity(name, Path(path))
                  for name, path in _parse_pairs(identity_args).items()}
    missing = [name for name in packs if name not in identities]
    if missing:
        raise R2SetError(f"no identity sidecar registered for pack(s): {missing}")
    return packs, identities


def _parse_pairs(entries: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in entries:
        name, sep, path = entry.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise R2SetError(f"expected NAME=PATH, got {entry!r}")
        out[name.strip()] = path.strip()
    return out


# --------------------------------------------------------------------------- #
# candidate windows and stratification
# --------------------------------------------------------------------------- #
def candidate_windows(packs: Mapping[str, Pack]) -> list[dict]:
    """All TRAIN runs of every pack as window descriptors (chunk prefix kept)."""
    windows = []
    for pack_name, pack in packs.items():
        for run in pack.runs.values():
            if str(run.get("split")) != TRAIN_SPLIT:
                continue
            window = dict(run)
            window["pack"] = pack_name
            windows.append(window)
    if not windows:
        raise R2SetError("no TRAIN windows found in the supplied packs")
    return windows


def paired_index(pack: Pack) -> set[tuple[str, float]]:
    """``(world, t0_s)`` keys that carry both a native and a PI run."""
    have: dict[tuple[str, float], set[str]] = {}
    for run in pack.runs.values():
        if str(run.get("split")) != TRAIN_SPLIT:
            continue
        key = (str(run.get("world")), round(float(run.get("t0_s", 0.0)), 6))
        have.setdefault(key, set()).add(str(run.get("arm")))
    return {key for key, arms in have.items() if set(ARMS).issubset(arms)}


def _stratum(window: Mapping) -> tuple:
    meta = world_meta(str(window.get("world")))
    return (str(window.get("pack")), str(window.get("policy")),
            meta["demand"], meta["penetration_pct"])


def _round_robin_capped(strata: Mapping[Any, list], *, seed: int,
                        max_per_world: int, limit: int) -> list[dict]:
    """Deterministic stratified round-robin, capped per world.

    Strata (pack/policy/demand/penetration) are visited in a seeded hash order;
    within a stratum windows are ordered by world then t0 so consecutive picks
    spread over worlds rather than draining one world's t0 grid.
    """
    groups = {k: list(v) for k, v in strata.items()}
    keys = sorted(groups, key=lambda k: hashlib.sha256(
        f"{seed}:{k}".encode("utf-8")).hexdigest())
    for key in keys:
        groups[key].sort(key=lambda w: (str(w.get("world")), float(w.get("t0_s", 0.0)),
                                        str(w.get("key"))))
    index = {k: 0 for k in keys}
    per_world: dict[str, int] = {}
    out: list[dict] = []
    while len(out) < limit:
        progressed = False
        for key in keys:
            if len(out) >= limit:
                break
            bucket = groups[key]
            while index[key] < len(bucket):
                window = bucket[index[key]]
                index[key] += 1
                world = str(window.get("world"))
                if per_world.get(world, 0) >= max_per_world:
                    continue
                per_world[world] = per_world.get(world, 0) + 1
                out.append(window)
                progressed = True
                break
        if not progressed:
            break
    return out


def select_windows(packs: Mapping[str, Pack], *, target: int = DEFAULT_TARGET,
                   seed: int = 0, oversample: float = DEFAULT_OVERSAMPLE,
                   max_per_world: int = DEFAULT_MAX_PER_WORLD,
                   require_paired: bool = True) -> list[dict]:
    """Stratified candidate order (over-sampled; the writer stops at target).

    ``require_paired`` (default) keeps only windows whose world/t0 also has both
    a native and a PI run in the same pack, so every state can be scored under
    both arms and enters the paired PI-native comparison.  Packs whose arms have
    no native/PI counterpart (pack_dspeed arm0-3, pack_damping base/native_base)
    contribute no paired states and are left out of the comparator.
    """
    windows = candidate_windows(packs)
    paired = {name: paired_index(pack) for name, pack in packs.items()}
    strata: dict[tuple, list] = {}
    for window in windows:
        key = _stratum(window)
        is_paired = (str(window.get("world")), round(float(window.get("t0_s", 0.0)), 6)) \
            in paired[str(window.get("pack"))]
        if require_paired and not is_paired:
            continue
        strata.setdefault(key + (is_paired,), []).append(window)
    if not strata:
        raise R2SetError("no paired TRAIN windows found; nothing to build")
    limit = max(int(target), int(round(target * float(oversample))))
    return _round_robin_capped(strata, seed=seed, max_per_world=max_per_world,
                               limit=limit)


# --------------------------------------------------------------------------- #
# streaming build
# --------------------------------------------------------------------------- #
def build_set(packs: Mapping[str, Pack], identities: Mapping[str, Identity],
              windows: Iterable[dict], out_path: Path, *, target: int,
              horizon_s: float, min_vehicles: int, extra_baselines: Sequence[str] = ARMS,
              on_error: str = "skip") -> dict:
    """Materialise up to ``target`` usable states, one at a time, to JSONL."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []
    errors: list[dict] = []
    with out_path.open("w", encoding="utf-8") as fh:
        for window in windows:
            if len(written) >= target:
                break
            try:
                record = build_state(window, packs, identities, horizon_s=horizon_s,
                                     extra_baselines=extra_baselines,
                                     min_vehicles=min_vehicles)
            except MaterializeError as exc:
                if on_error == "raise":
                    raise
                errors.append({"key": window.get("key"), "error": str(exc)})
                continue
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            written.append({
                "key": record.get("key"), "world": record.get("world"),
                "pack": record.get("pack"), "policy": record.get("policy"),
                "arm": record.get("arm"), "t0_s": record.get("t0_s"),
                "arms": record.get("arms"), "paired": record.get("paired"),
            })
    meta = {
        "schema": "wms_eval_r2_set/1",
        "out": str(out_path),
        "n_states": len(written),
        "n_errors": len(errors),
        "target": int(target),
        "horizon_s": float(horizon_s),
        "min_vehicles": int(min_vehicles),
        "extra_baselines": list(extra_baselines),
        "n_worlds": len({row["world"] for row in written}),
        "n_paired": sum(1 for row in written if row.get("paired")),
        "packs": sorted({row["pack"] for row in written}),
        "policies": sorted({str(row["policy"]) for row in written}),
        "states": written,
        "errors": errors,
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wms_eval r2set",
                                description="Build the RUN2 E1 TRAIN evaluation set")
    p.add_argument("--pack", action="append", required=True, help="NAME=PATH (repeatable)")
    p.add_argument("--identities", action="append", required=True,
                   help="PACK_NAME=PATH (repeatable)")
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument("--target", type=int, default=DEFAULT_TARGET)
    p.add_argument("--horizon-s", type=float, default=300.0)
    p.add_argument("--min-vehicles", type=int, default=5)
    p.add_argument("--max-per-world", type=int, default=DEFAULT_MAX_PER_WORLD)
    p.add_argument("--oversample", type=float, default=DEFAULT_OVERSAMPLE,
                   help="candidate over-sampling factor before degenerate drops")
    p.add_argument("--allow-unpaired", action="store_true",
                   help="also keep windows with no native+PI pair (default: paired only)")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        packs, identities = load_packs(args.pack, args.identities)
        windows = select_windows(packs, target=args.target, seed=args.seed,
                                 oversample=args.oversample,
                                 max_per_world=args.max_per_world,
                                 require_paired=not args.allow_unpaired)
        meta = build_set(packs, identities, windows, Path(args.out),
                         target=args.target, horizon_s=args.horizon_s,
                         min_vehicles=args.min_vehicles)
    except (R2SetError, MaterializeError, OSError, ValueError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 2
    print(json.dumps({key: meta[key] for key in
                      ("out", "n_states", "n_errors", "n_worlds", "n_paired",
                       "packs", "policies")}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
