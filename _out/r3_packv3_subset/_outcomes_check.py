"""Check subset outcomes.parquet equals the referenced source rows."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SRC = Path(r"D:\rd-staging\wmn1_feat\pack_v3")
SUB = Path(r"D:\rd-staging\pack_v3_r3_subset")

src = pd.read_parquet(SRC / "outcomes.parquet")
sub = pd.read_parquet(SUB / "outcomes.parquet")
man = json.loads((SUB / "manifest.json").read_text(encoding="utf-8"))
index = json.loads((SRC / "INDEX.json").read_text(encoding="utf-8"))
train = json.loads(Path(r"E:\rd-wt\ws-r3-jam\_out\lpwo_v3"
                        r"\windows_train_noPI_noWall.json").read_text(
                            encoding="utf-8"))
val = json.loads(Path(r"E:\rd-wt\ws-r3-jam\_out\lpwo_v3"
                      r"\windows_val_noPI_noWall.json").read_text(
                          encoding="utf-8"))
keys = {w["key"] for w in train["windows"] if w["pack"] == "pack_v3"} | \
    {w["key"] for w in val["windows"] if w["pack"] == "pack_v3"}
orig_item = {}
for it in index["runs"]:
    pass
src_items = {}
for it in json.loads((SRC / "manifest.json").read_text(encoding="utf-8"))["items"]:
    k = f"{it.get('batch') or it['chunk']}/{it['world']}__{it['arm']}"
    src_items.setdefault(k, it)
cursor = 0
assert len(sub) == 22839, len(sub)
for it in man["items"]:
    k = f"{it.get('batch') or it['chunk']}/{it['world']}__{it['arm']}"
    o = src_items[k]["outcome_rows"]
    n = it["outcome_rows"]
    assert n == list(range(cursor, cursor + len(o))), (k, n[:3], cursor)
    a = src.iloc[o].reset_index(drop=True)
    b = sub.iloc[n].reset_index(drop=True)
    assert a.equals(b), f"row mismatch for {k}"
    cursor += len(o)
assert cursor == len(sub)
print(f"OUTCOMES OK: {len(sub)} rows across {len(man['items'])} items "
      "match the source-referenced rows")
