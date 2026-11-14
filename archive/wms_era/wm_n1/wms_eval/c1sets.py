"""Clean C1 v2 evaluation sets: U-same-world, U-new-world and SEEN.

The first C1 build shared whole worlds between the *seen* reference and the
*unseen* PI states and let 18 training runs leak into the mixed sample.  This
module rebuilds the three sets under explicit, testable rules (zero simulation;
it only reads the small metadata JSONs and, for the manifest, streams the
materialised state files):

* **U-same-world** -- PI states whose world also has *other, non-PI* arms in
  ``windows_train_noPI.json``.  The world was seen during training under an
  ordinary controller and the PI controller is new here, i.e. the paired
  "same situation, new controller" case.
* **U-new-world** -- PI states whose world has no training window at all (a
  world-grouped validation world).
* **SEEN** -- held-out, PI-free (ordinary) states from the validation worlds,
  excluding every world that appears in the U-same-world unseen set.  The
  validation worlds that also carry the U-new-world PI states are the paired
  "same situation, other controller" reference for that split and are reported
  separately rather than mixed into the strict count.

Every record whose ``run``/``key`` appears in the training- or validation-window
metadata is dropped, and the verifier fails closed on any world overlap with
U-same, any overlap between U-same/U-new, or any training-window ``key``/``run``
leak in any set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

CHUNK = 8 * 1024 * 1024
FIELD = {name: re.compile(rb'"' + name.encode() + rb'"\s*:\s*"([^"\\]*)"')
         for name in ("world", "run", "key")}
# one materialised record carries exactly one initial_state; a bare "world" match
# cannot be used for counting because ``world_meta`` repeats it inside a record
RECORD_MARKER = re.compile(rb'"initial_state"\s*:\s*\{')
UNSEEN_POLICY = "pi"
DEFAULT_TARGET = 200


class C1SetsError(ValueError):
    """A metadata file or the rebuilt split violates a C1 v2 rule."""


# --------------------------------------------------------------------------- #
# metadata loading
# --------------------------------------------------------------------------- #
def load_windows(path: Path) -> list[dict]:
    """Load the window list from a ``windows``/``states``/``list`` JSON object."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for key in ("windows", "states", "records", "list"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list) or not raw:
        raise C1SetsError(f"window file {path} has no non-empty list")
    return [dict(item) for item in raw if isinstance(item, dict)]


def _identity(records: Iterable[Mapping]) -> dict[str, set[str]]:
    index = {"keys": set(), "runs": set(), "worlds": set()}
    for record in records:
        for field, bucket in (("key", "keys"), ("run", "runs"), ("world", "worlds")):
            value = record.get(field)
            if value is not None and str(value):
                index[bucket].add(str(value))
    return index


def _is_leaked(record: Mapping, index: Mapping[str, set[str]]) -> bool:
    key, run = record.get("key"), record.get("run")
    return ((key is not None and str(key) in index["keys"])
            or (run is not None and str(run) in index["runs"]))


def _sort_key(record: Mapping):
    return (str(record.get("world", "")), float(record.get("t0_s", 0.0)),
            str(record.get("key", "")))


# --------------------------------------------------------------------------- #
# unseen split
# --------------------------------------------------------------------------- #
def unseen_split(pi_windows: Sequence[dict], train_windows: Sequence[dict],
                 val_windows: Sequence[dict]) -> dict:
    """Split the PI states into U-same-world and U-new-world under the rules.

    Drops every PI record whose ``run``/``key`` appears in the training or the
    validation window metadata.  U-same-world records live in a world that has
    a training window; U-new-world records live in a world with none.
    """
    train = _identity(train_windows)
    val = _identity(val_windows)
    train_worlds = train["worlds"]
    kept, dropped = [], []
    for record in pi_windows:
        if str(record.get("policy")) != UNSEEN_POLICY:
            continue
        if _is_leaked(record, train) or _is_leaked(record, val):
            dropped.append(record)
        else:
            kept.append(record)
    u_same = sorted((r for r in kept if str(r.get("world")) in train_worlds), key=_sort_key)
    u_new = sorted((r for r in kept if str(r.get("world")) not in train_worlds), key=_sort_key)
    u_same_worlds = {str(r.get("world")) for r in u_same}
    u_new_worlds = {str(r.get("world")) for r in u_new}
    return {
        "u_same": u_same,
        "u_new": u_new,
        "dropped": dropped,
        "u_same_worlds": u_same_worlds,
        "u_new_worlds": u_new_worlds,
        "unseen_worlds": u_same_worlds | u_new_worlds,
    }


# --------------------------------------------------------------------------- #
# seen selection
# --------------------------------------------------------------------------- #
def _world_balanced(windows: Sequence[dict], *, target: int, seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for window in windows:
        groups.setdefault(str(window.get("world", "")), []).append(window)
    for group in groups.values():
        group.sort(key=lambda w: (float(w.get("t0_s", 0.0)), str(w.get("key", ""))))
    worlds = sorted(groups, key=lambda w: hashlib.sha256(
        f"{seed}:{w}".encode("utf-8")).hexdigest())
    index = {world: 0 for world in worlds}
    selected: list[dict] = []
    while len(selected) < target:
        progressed = False
        for world in worlds:
            if len(selected) >= target:
                break
            if index[world] < len(groups[world]):
                selected.append(groups[world][index[world]])
                index[world] += 1
                progressed = True
        if not progressed:
            break
    return selected


def seen_selection(val_windows: Sequence[dict], split: Mapping, train_windows: Sequence[dict],
                   *, target: int = DEFAULT_TARGET, seed: int = 0) -> dict:
    """Held-out PI-free validation states, disjoint from the U-same-world set.

    Strict tier: validation worlds outside *both* unseen subsets.  Paired tier:
    the validation worlds that also carry the U-new-world PI states (their
    ordinary arms are the same-world "other controller" reference).  Every
    candidate must be non-PI and must not be a training window.
    """
    train = _identity(train_windows)
    u_same_worlds = set(split["u_same_worlds"])
    u_new_worlds = set(split["u_new_worlds"])
    eligible = [w for w in val_windows
                if str(w.get("policy")) != UNSEEN_POLICY
                and str(w.get("world")) not in u_same_worlds
                and not _is_leaked(w, train)]
    strict = sorted((w for w in eligible if str(w.get("world")) not in u_new_worlds),
                    key=_sort_key)
    paired = sorted((w for w in eligible if str(w.get("world")) in u_new_worlds),
                    key=_sort_key)
    remaining = max(0, int(target) - len(strict))
    selected = list(strict)
    selected += _world_balanced(paired, target=remaining, seed=seed)
    # deduplicate by (pack, key): the same run key can appear once per pack
    seen, seen_keys = [], set()
    for window in selected:
        token = (str(window.get("pack")), str(window.get("key")))
        if token in seen_keys:
            continue
        seen_keys.add(token)
        seen.append(window)
    return {"seen": seen, "strict": strict, "paired": paired,
            "strict_worlds": {str(w.get("world")) for w in strict},
            "paired_worlds": {str(w.get("world")) for w in paired}}


# --------------------------------------------------------------------------- #
# fail-closed verifier
# --------------------------------------------------------------------------- #
def _verify_indexes(indexes: Mapping[str, Mapping[str, set[str]]],
                    counts: Mapping[str, int], train_windows: Sequence[dict]) -> dict:
    tr = _identity(train_windows)
    us, un, se = indexes["u_same"], indexes["u_new"], indexes["seen"]
    report = {
        "u_same_states": counts["u_same"], "u_new_states": counts["u_new"],
        "seen_states": counts["seen"],
        "u_same_worlds": len(us["worlds"]), "u_new_worlds": len(un["worlds"]),
        "seen_worlds": len(se["worlds"]),
        "train_windows": len(train_windows),
        "u_same_u_new_world_overlap": sorted(us["worlds"] & un["worlds"]),
        "seen_u_same_world_overlap": sorted(se["worlds"] & us["worlds"]),
        "seen_u_new_world_overlap": sorted(se["worlds"] & un["worlds"]),
        "train_key_overlap": {}, "train_run_overlap": {},
    }
    problems: list[str] = []
    if report["u_same_u_new_world_overlap"]:
        problems.append(f"u_same/u_new world overlap={len(report['u_same_u_new_world_overlap'])}")
    if report["seen_u_same_world_overlap"]:
        problems.append(f"seen/u_same world overlap={len(report['seen_u_same_world_overlap'])}")
    for name, index in (("u_same", us), ("u_new", un), ("seen", se)):
        keys = sorted(index["keys"] & tr["keys"])
        runs = sorted(index["runs"] & tr["runs"])
        report["train_key_overlap"][name] = keys
        report["train_run_overlap"][name] = runs
        if keys:
            problems.append(f"{name}/train key overlap={len(keys)}")
        if runs:
            problems.append(f"{name}/train run overlap={len(runs)}")
    report["overlap_ok"] = not problems
    report["refused"] = bool(problems)
    if problems:
        raise C1SetsError("C1 v2 separation refused: " + "; ".join(problems) +
                          f"; counts={json.dumps({k: report[k] for k in ('seen_u_same_world_overlap', 'train_key_overlap', 'train_run_overlap')})}")
    return report


def verify(u_same: Sequence[dict], u_new: Sequence[dict], seen: Sequence[dict],
           train_windows: Sequence[dict]) -> dict:
    """Fail closed on leakage; report the counts even when clean."""
    indexes = {"u_same": _identity(u_same), "u_new": _identity(u_new),
               "seen": _identity(seen)}
    counts = {"u_same": len(u_same), "u_new": len(u_new), "seen": len(seen)}
    return _verify_indexes(indexes, counts, train_windows)


def verify_files(u_same_path: Path, u_new_path: Path, seen_path: Path,
                 train_windows: Sequence[dict]) -> dict:
    """Like :func:`verify`, but streams the large state files (memory rule)."""
    indexes, counts = {}, {}
    for name, path in (("u_same", u_same_path), ("u_new", u_new_path), ("seen", seen_path)):
        info = scan_states(Path(path))
        indexes[name] = {"keys": set(info["keys"]), "runs": set(info["runs"]),
                         "worlds": set(info["worlds"])}
        counts[name] = info["n_states"]
    return _verify_indexes(indexes, counts, train_windows)


# --------------------------------------------------------------------------- #
# streaming manifest
# --------------------------------------------------------------------------- #
def scan_states(path: Path) -> dict:
    """Stream one state-set file: sha256, record count and world/run/key sets."""
    digest = hashlib.sha256()
    values: dict[str, set[str]] = {"world": set(), "run": set(), "key": set()}
    record_positions: set[int] = set()
    total = 0
    carry = b""
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            digest.update(block)
            data = carry + block
            offset = total - len(carry)
            for name, pattern in FIELD.items():
                for match in pattern.finditer(data):
                    values[name].add(match.group(1).decode("utf-8", "replace"))
            for match in RECORD_MARKER.finditer(data):
                record_positions.add(offset + match.start())
            carry = data[-256:]
            total += len(block)
    return {"sha256": digest.hexdigest(), "bytes": total,
            "n_states": len(record_positions),
            "n_worlds": len(values["world"]),
            "worlds": sorted(values["world"]), "runs": sorted(values["run"]),
            "keys": sorted(values["key"])}


def build_manifest(set_paths: Mapping[str, Path], *, source_paths: Mapping[str, Path] | None = None,
                   separation: Mapping | None = None) -> dict:
    sets = {}
    for name, path in set_paths.items():
        info = scan_states(Path(path))
        sets[name] = {k: info[k] for k in
                      ("n_states", "n_worlds", "bytes", "sha256", "worlds")}
        sets[name]["world_list_sha256"] = hashlib.sha256(
            "\n".join(info["worlds"]).encode("utf-8")).hexdigest()
    manifest = {
        "schema": "wms_eval_c1sets/1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sets": sets,
        "source_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for name, path in (source_paths or {}).items()},
        "separation": dict(separation) if separation else None,
    }
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")


def _parse_pairs(entries: Sequence[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for entry in entries:
        name, sep, value = entry.partition("=")
        if not sep or not name.strip() or not value.strip():
            raise C1SetsError(f"expected NAME=PATH, got {entry!r}")
        out[name.strip()] = Path(value.strip())
    return out


def _cmd_verify(args) -> int:
    train = load_windows(Path(args.train_windows))
    report = verify_files(Path(args.u_same), Path(args.u_new), Path(args.seen), train)
    print(json.dumps(report, sort_keys=True))
    if args.out:
        _write_json(Path(args.out), report)
    return 0


def _cmd_manifest(args) -> int:
    set_paths = _parse_pairs(args.set)
    sources = _parse_pairs(args.source)
    separation = None
    if args.verify:
        train = load_windows(Path(args.train_windows)) if args.train_windows else []
        separation = verify_files(Path(args.u_same), Path(args.u_new), Path(args.seen), train)
    manifest = build_manifest(set_paths, source_paths=sources, separation=separation)
    _write_json(Path(args.out), manifest)
    print(json.dumps({"out": args.out, "sets": {k: v["n_states"] for k, v in manifest["sets"].items()}}))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="wms_eval c1sets")
    sub = root.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--u-same", required=True)
    v.add_argument("--u-new", required=True)
    v.add_argument("--seen", required=True)
    v.add_argument("--train-windows", required=True)
    v.add_argument("--out")
    v.set_defaults(func=_cmd_verify)
    m = sub.add_parser("manifest")
    m.add_argument("--set", action="append", required=True, help="NAME=PATH (repeatable)")
    m.add_argument("--source", action="append", default=[], help="NAME=PATH (repeatable)")
    m.add_argument("--out", required=True)
    m.add_argument("--verify", action="store_true")
    m.add_argument("--u-same")
    m.add_argument("--u-new")
    m.add_argument("--seen")
    m.add_argument("--train-windows")
    m.set_defaults(func=_cmd_manifest)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (C1SetsError, OSError, ValueError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 2


__all__ = ["C1SetsError", "build_manifest", "load_windows", "main", "scan_states",
           "seen_selection", "unseen_split", "verify"]
