"""WD-01-PA2G-CNT1 close-out merge (zero simulation).

Merges the 32 per-world raw row files (results/eval_cnt1/worlds/<w>.jsonl,
written by eval_cnt1_world.py) into results/eval_cnt1/raw_rows.jsonl in
panel order, with the registered close-out assertions:

- every panel world has a row file with exactly EXPECTED_ROWS_PER_WORLD
  rows and an identical per-(model, H) structure;
- the union of the recorded opened-file lists (worlds/<w>.opened.json)
  touches TRAIN panel run files only (no CALIBRATION/HOLDOUT world, no
  non-panel file); worlds salvaged from attempt-5 shards have no
  opened.json -- their opened set is the identical runner code path on
  the same TRAIN panel worlds and is disclosed in the report;
- the merged row count is reported per world.

The eval itself is never re-run; nothing under results/eval_cnt1/worlds/
is modified.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORLDS = HERE / "results" / "eval_cnt1" / "worlds"
PANEL = HERE / "results" / "lc1_a2_worlds32.json"
RAW = HERE / "results" / "eval_cnt1" / "raw_rows.jsonl"
EXPECTED_ROWS = 280
MODELS = ("v2g_cnt", "v2g", "const_speed", "v1_p3o_replica")
HS = (10, 30, 60, 120)


def structure(rows):
    from collections import Counter
    c = Counter((r["model"], int(r["H"])) for r in rows)
    worlds = set(r["world"] for r in rows)
    assert len(worlds) == 1, worlds
    return c, worlds.pop()


def main():
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    assert len(panel) == 32
    total = 0
    ref_struct = None
    opened_union = []
    missing_opened = []
    per_world = {}
    for w in panel:
        rp = WORLDS / f"{w}.jsonl"
        assert rp.exists(), f"missing row file for panel world {w}"
        rows = [json.loads(line) for line in
                rp.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == EXPECTED_ROWS, f"{w}: {len(rows)} rows"
        c, wname = structure(rows)
        assert wname == w, (wname, w)
        assert set(m for m, _H in c) == set(MODELS), c
        if ref_struct is None:
            ref_struct = c
        assert c == ref_struct, f"{w}: per-(model,H) structure differs"
        per_world[w] = len(rows)
        total += len(rows)
        op = WORLDS / f"{w}.opened.json"
        if op.exists():
            opened_union += json.loads(op.read_text(encoding="utf-8"))
        else:
            missing_opened.append(w)
    panel_set = set(panel)
    bad = []
    plan_seen = False
    for p in sorted(set(opened_union)):
        if p.replace("\\\\", "\\").endswith("WORLD_PLAN.json"):
            plan_seen = True
            continue
        m = re.search(r"(w\d+m\d+p\d+t\d+)__native", p)
        if not m or m.group(1) not in panel_set:
            bad.append(p)
    assert plan_seen, "chunk-4 plan not among recorded opened files"
    assert not bad, f"non-panel opened run files: {bad}"
    n_rec = len([w for w in panel if (WORLDS / f'{w}.opened.json').exists()])
    out_lines = []
    for w in panel:
        rp = WORLDS / f"{w}.jsonl"
        out_lines.append(rp.read_text(encoding="utf-8").rstrip("\n"))
    RAW.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"merged {len(panel)} worlds, {total} rows -> {RAW}")
    print(f"opened.json recorded for {n_rec}/32 worlds "
          f"(missing (attempt-5 salvaged): {missing_opened})")
    print("panel-only + plan assertions: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
