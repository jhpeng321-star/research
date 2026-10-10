"""Validate every lc1 jsonl: report files with unparseable lines."""
import json
from pathlib import Path

LC = Path(r"output/wd01/partA_v2g/results/lc1")
for p in sorted(LC.glob("*.jsonl")):
    bad = 0
    first = None
    with open(p, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except Exception:
                bad += 1
                if first is None:
                    first = (i + 1, line[:120])
    print(f"{p.name}: lines={i + 1 if 'i' in dir() else 0} bad={bad}",
          first if first else "")
