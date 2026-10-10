"""Write the small validation window list used by ``configs/smoke_local.toml``.

The local CPU smoke trains two steps per curriculum stage, but the trainer validates on the
WHOLE validation list at the end of every stage (151 windows at up to 120 s is hours on CPU).
This script keeps the first ``--per-pack`` windows of every pack of the real R3 validation list
(metadata only; the lists are TRAIN-scoped and hold no outcomes) and writes them next to the
other window lists as ``runs/active/smoke_lists/windows_val_smoke.json``.

    python scripts/make_smoke_lists.py [--per-pack 2] [--out runs/active/smoke_lists]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="lpwo_v3/windows_val_noPI_noWall.json",
                        help="logical name of the full validation list")
    parser.add_argument("--per-pack", type=int, default=2)
    parser.add_argument("--out", default=str(ROOT / "runs" / "active" / "smoke_lists"))
    args = parser.parse_args(argv)
    from wms.data.registry import require_present

    source = require_present(args.source)
    doc = json.loads(source.read_text(encoding="utf-8"))
    kept, seen = [], {}
    for window in doc["windows"]:
        pack = str(window.get("pack"))
        if seen.get(pack, 0) < args.per_pack:
            seen[pack] = seen.get(pack, 0) + 1
            kept.append(window)
    doc = {**doc, "windows": kept, "n_windows": len(kept),
           "smoke_of": {"source": args.source, "per_pack": args.per_pack}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "windows_val_smoke.json"
    target.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps({"written": str(target), "n_windows": len(kept), "by_pack": seen}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
