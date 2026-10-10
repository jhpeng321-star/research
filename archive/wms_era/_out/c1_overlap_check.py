"""Streaming C1 seen/unseen world- and training-window overlap check.

Reads large state-set JSONs in fixed-size chunks (never json.load) and extracts
the per-record top-level ``world`` / ``run`` / ``key`` strings with a regex.
This is a zero-simulation diagnostic; it never runs a model or opens a pack.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CHUNK = 8 * 1024 * 1024
FIELD = {name: re.compile(rb'"' + name.encode() + rb'"\s*:\s*"([^"\\]*)"')
         for name in ("world", "run", "key", "split", "n_states")}


def scan(path: Path) -> dict:
    counts = {"world": 0, "run": 0, "key": 0}
    values: dict[str, set[str]] = {"world": set(), "run": set(), "key": set()}
    carry = b""
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            data = carry + block
            for name, pattern in FIELD.items():
                if name not in counts:
                    continue
                found = pattern.findall(data)
                if found:
                    counts[name] += len(found)
                    values[name].update(m.group(1).decode("utf-8") for m in pattern.finditer(data))
            carry = data[-256:]
    return {"path": str(path), "counts": counts,
            "worlds": sorted(values["world"]), "runs": sorted(values["run"]),
            "keys": sorted(values["key"])}


def load_windows(path: Path) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for key in ("windows", "states", "records", "list"):
            if isinstance(raw.get(key), list):
                return [dict(w) for w in raw[key] if isinstance(w, dict)]
    if isinstance(raw, list):
        return [dict(w) for w in raw if isinstance(w, dict)]
    raise SystemExit(f"no window list in {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-windows")
    ap.add_argument("--unseen", action="append", default=[])
    ap.add_argument("--seen", action="append", default=[])
    args = ap.parse_args()

    seen = [(p, scan(Path(p))) for p in args.seen]
    unseen = [(p, scan(Path(p))) for p in args.unseen]
    for label, group in (("seen", seen), ("unseen", unseen)):
        for path, info in group:
            print(f"[{label}] {path} worlds={info['counts']['world']} "
                  f"runs={info['counts']['run']} keys={info['counts']['key']}")

    summary: dict = {"seen": {}, "unseen": {}, "world_overlap": {},
                     "train_windows": None, "train_overlap": {}}
    for path, info in seen:
        summary["seen"][path] = {"states": info["counts"]["world"], "worlds": info["worlds"]}
    for path, info in unseen:
        summary["unseen"][path] = {"states": info["counts"]["world"], "worlds": info["worlds"]}

    seen_worlds = set().union(*[set(i["worlds"]) for _, i in seen]) if seen else set()
    for path, info in unseen:
        shared = sorted(seen_worlds & set(info["worlds"]))
        summary["world_overlap"][path] = {"count": len(shared), "worlds": shared}
        print(f"[world-overlap] unseen={path} shared={len(shared)} {shared[:5]}")

    if args.train_windows:
        train = load_windows(Path(args.train_windows))
        train_keys = {str(w.get("key")) for w in train if w.get("key")}
        train_runs = {str(w.get("run")) for w in train if w.get("run")}
        summary["train_windows"] = {"path": args.train_windows, "n_windows": len(train),
                                    "n_keys": len(train_keys), "n_runs": len(train_runs)}
        print(f"[train] {args.train_windows} windows={len(train)} "
              f"keys={len(train_keys)} runs={len(train_runs)}")
        for path, info in unseen:
            key_hits = sorted(train_keys & set(info["keys"]))
            run_hits = sorted(train_runs & set(info["runs"]))
            summary["train_overlap"][path] = {"keys": key_hits, "runs": run_hits}
            print(f"[train-overlap] unseen={path} keys={len(key_hits)} "
                  f"runs={len(run_hits)} {sorted(set(key_hits + run_hits))[:5]}")

    print("SUMMARY_JSON " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
