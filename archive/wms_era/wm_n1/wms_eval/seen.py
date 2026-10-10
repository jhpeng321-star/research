"""Build the C1 SEEN held-out state set from ``windows_val.json`` worlds.

C1 compares a model on an *unseen* controller against a *seen but held-out*
reference on the same fleet.  The reference states are drawn from the
leave-PI-out validation windows (``windows_val.json``): worlds reserved whole
from training by the sha256(world) rule, restricted to the ordinary arms
(native / damping / wall) that are *not* the controller under test.

The builder is deterministic and world-balanced so the SEEN set has a
comparable size and condition mix to the 161 primary fleet-PI states.  It
emits a normal state-set JSON the ``c1`` command consumes directly.  Window
descriptors carry no physics rows; the state material is attached with
``--state-index`` (``key -> material``) so pack arrays are read one window at a
time by whatever materialiser produced the index.  Without an index the builder
emits a metadata-only descriptor set for a later materialisation pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .data import DataError, json_default

DEFAULT_ARMS = ("native", "damping", "wall")
DEFAULT_TARGET = 161


class SeenError(ValueError):
    """The SEEN set cannot be built from the supplied windows."""


def load_windows(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        for key in ("windows", "states", "records", "list"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list) or not raw:
        raise DataError("windows file must contain a non-empty JSON list")
    return [dict(item) for item in raw if isinstance(item, dict)]


def _arm_of(window: Mapping) -> str:
    return str(window.get("policy") or window.get("arm") or "").strip().lower()


def select_seen_windows(windows: Iterable[dict], *, arms: Sequence[str] = DEFAULT_ARMS,
                        exclude_worlds: Iterable[str] = (), target_size: int = DEFAULT_TARGET,
                        seed: int = 0) -> list[dict]:
    """Deterministically pick a world-balanced held-out subset.

    Only windows whose policy is one of ``arms`` and whose world is not excluded
    are eligible.  Worlds are visited in a seeded, order-independent order and
    one window at a time is taken per visit (round-robin), so the selection
    spreads across worlds before taking a second window from any world -- the
    same shape as a per-world fleet-PI state sample.
    """
    wanted = {str(a).strip().lower() for a in arms}
    excluded = {str(w).strip() for w in exclude_worlds}
    eligible: dict[str, list[dict]] = {}
    for window in windows:
        if str(window.get("split", "TRAIN")).upper() != "TRAIN":
            continue
        if _arm_of(window) not in wanted:
            continue
        world = str(window.get("world", ""))
        if not world or world in excluded:
            continue
        eligible.setdefault(world, []).append(window)
    if not eligible:
        raise SeenError("no eligible seen windows after arm/split/world filtering")

    def order(value: str) -> str:
        return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()

    for world in eligible:
        eligible[world].sort(key=lambda w: (float(w.get("t0_s", 0.0)), str(w.get("key", ""))))
    worlds = sorted(eligible, key=order)
    index = {world: 0 for world in worlds}
    selected: list[dict] = []
    while len(selected) < target_size:
        progressed = False
        for world in worlds:
            if len(selected) >= target_size:
                break
            position = index[world]
            if position >= len(eligible[world]):
                continue
            selected.append(eligible[world][position])
            index[world] += 1
            progressed = True
        if not progressed:
            break
    return selected


def build(*, windows_path: Path, out_path: Path, arms: Sequence[str] = DEFAULT_ARMS,
          exclude_worlds: Iterable[str] = (), target_size: int = DEFAULT_TARGET, seed: int = 0,
          state_index: Mapping[str, dict] | None = None, metadata_only: bool = False) -> dict:
    windows = load_windows(windows_path)
    selected = select_seen_windows(windows, arms=arms, exclude_worlds=exclude_worlds,
                                   target_size=target_size, seed=seed)
    if metadata_only or state_index is None:
        records = selected
        materialised = False
    else:
        records = []
        missing = []
        for window in selected:
            key = str(window.get("key", ""))
            material = state_index.get(key)
            if material is None:
                missing.append(key)
                continue
            records.append({**material, **{k: v for k, v in window.items() if v is not None}})
        if missing:
            raise SeenError(
                f"{len(missing)} selected windows have no state material, e.g. {missing[:3]}; "
                "supply a complete --state-index or use --metadata-only"
            )
        materialised = True

    digest = hashlib.sha256(Path(windows_path).read_bytes()).hexdigest()
    payload = {
        "schema": "wms_eval_seen/1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_windows": str(windows_path),
        "source_windows_sha256": digest,
        "rule": "TRAIN-only windows, non-PI arms, worlds reserved from training; world-balanced",
        "arms": list(arms),
        "target_size": target_size,
        "materialised": materialised,
        "n_windows": len(records),
        "n_worlds": len({str(r.get("world")) for r in records}),
        "windows": records,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=json_default), encoding="utf-8")
    return payload


def _worlds_from(path: Path) -> list[str]:
    return sorted({str(window.get("world")) for window in load_windows(path) if window.get("world")})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wms_eval seen")
    p.add_argument("--windows", required=True, help="windows_val.json (metadata only)")
    p.add_argument("--out", required=True)
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    p.add_argument("--target-size", type=int, default=DEFAULT_TARGET)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-world", action="append", default=[],
                   help="world to exclude (repeatable)")
    p.add_argument("--exclude-worlds-from", action="append", default=[],
                   help="windows/state file whose worlds must be excluded (e.g. --unseen)")
    p.add_argument("--state-index", help="JSON key->material to emit a materialised state set")
    p.add_argument("--metadata-only", action="store_true",
                   help="emit window descriptors for a later materialisation pass")
    return p


def run(args) -> int:
    exclude = list(args.exclude_world)
    for path in args.exclude_worlds_from:
        exclude.extend(_worlds_from(Path(path)))
    index = None
    if args.state_index:
        with Path(args.state_index).open("r", encoding="utf-8") as fh:
            index = json.load(fh)
        if not isinstance(index, dict):
            raise SeenError("--state-index must be a JSON object keyed by window key")
    payload = build(
        windows_path=Path(args.windows), out_path=Path(args.out),
        arms=[a.strip() for a in args.arms.split(",") if a.strip()],
        exclude_worlds=exclude, target_size=args.target_size, seed=args.seed,
        state_index=index, metadata_only=args.metadata_only,
    )
    print(json.dumps({"out": str(args.out), "n_windows": payload["n_windows"],
                      "n_worlds": payload["n_worlds"],
                      "materialised": payload["materialised"]}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (DataError, SeenError, ValueError, OSError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 2


__all__ = ["DEFAULT_ARMS", "DEFAULT_TARGET", "SeenError", "build", "load_windows",
           "select_seen_windows"]
