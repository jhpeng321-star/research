"""INC1 zero-shot assertion (registration section 6).

Two independent counts must be zero before the model lock is written:

* every ``event_mask`` row in the training pack is empty (all-false);
* the training source scenario configs contain no ``obstacle`` / ``setStop`` /
  ``stopped_obstacle`` events.

Any nonzero count is a blocker and exits 2.  Pack arrays are opened with
``mmap_mode="r"`` and never copied or modified; the scenario scan only reads
text files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EVENT_MASK_NAME = "event_mask.npy"
MANIFEST_NAME = "manifest.json"
OBSTACLE_TOKENS = ("obstacle", "setStop", "stopped_obstacle")
TEXT_SUFFIXES = (".json", ".yaml", ".yml", ".py", ".xml", ".sumocfg",
                 ".net.xml", ".txt", ".md")


def find_event_mask(pack_dir) -> Path:
    """Locate ``event_mask.npy`` directly or through the pack manifest."""
    pack_dir = Path(pack_dir)
    direct = pack_dir / EVENT_MASK_NAME
    if direct.is_file():
        return direct
    manifest = pack_dir / MANIFEST_NAME
    if manifest.is_file():
        spec = (json.loads(manifest.read_text(encoding="utf-8"))
                .get("arrays", {}).get("event_mask"))
        if spec and (pack_dir / spec["file"]).is_file():
            return pack_dir / spec["file"]
    index = pack_dir / "INDEX.json"
    if index.is_file():
        spec = (json.loads(index.read_text(encoding="utf-8"))
                .get("arrays", {}).get("event_mask"))
        if spec and (pack_dir / spec["file"]).is_file():
            return pack_dir / spec["file"]
    raise FileNotFoundError(f"no event_mask array under {pack_dir}")


def count_event_mask(pack_dir) -> dict:
    """Read-only count of non-empty ``event_mask`` rows (memmap)."""
    path = find_event_mask(pack_dir)
    arr = np.load(path, mmap_mode="r")
    total = int(arr.shape[0]) if arr.ndim else 1
    flat = arr.reshape(total, -1)
    nonempty = 0
    nonzero = 0
    for start in range(0, total, 1 << 16):
        block = np.asarray(flat[start:start + (1 << 16)])
        row_any = (block != 0).any(axis=1)
        nonempty += int(row_any.sum())
        nonzero += int((block != 0).sum())
    return {"file": str(path), "rows": total, "nonempty_rows": nonempty,
            "nonzero_entries": nonzero}


def count_scenario_tokens(source_root, *,
                          tokens=OBSTACLE_TOKENS) -> dict:
    """Case-insensitive token counts over the source scenario text files."""
    root = Path(source_root)
    counts = {t: 0 for t in tokens}
    files: dict = {}
    if not root.exists():
        return {"root": str(root), "counts": counts, "files": files,
                "exists": False}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        hit = {t: text.count(t.lower()) for t in tokens}
        if any(hit.values()):
            files[str(path)] = hit
            for t in tokens:
                counts[t] += hit[t]
    return {"root": str(root), "counts": counts, "files": files,
            "exists": True}


def assert_zero_shot(pack_dir, source_root, *,
                     tokens=OBSTACLE_TOKENS) -> dict:
    """The full zero-shot report; ``blocked`` is true on any nonzero count."""
    events = count_event_mask(pack_dir)
    scenarios = count_scenario_tokens(source_root, tokens=tokens)
    blocked = bool(events["nonempty_rows"] or events["nonzero_entries"]
                   or any(scenarios["counts"].values()))
    return {"schema": "wm_n1_inc1_zeroshot/1",
            "pack": events, "scenarios": scenarios, "blocked": blocked}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="inc1eval.zeroshot",
        description="INC1 zero-shot assertion (event tokens + obstacle events)")
    p.add_argument("--pack", required=True,
                   help="training pack directory (memmap, read-only)")
    p.add_argument("--source-root", required=True,
                   help="root of the training source scenario configs")
    args = p.parse_args(argv)
    report = assert_zero_shot(args.pack, args.source_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["blocked"]:
        print("[inc1eval.zeroshot] BLOCKED: nonzero count -> exit 2", file=sys.stderr)
        return 2
        print("[inc1eval.zeroshot] PASS: zero non-empty event masks and zero " 
          "obstacle/setStop scenario events")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    sys.exit(main())
