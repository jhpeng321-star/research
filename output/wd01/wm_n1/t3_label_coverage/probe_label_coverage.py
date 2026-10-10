"""T3 (narrowed relaunch) label-coverage probe -- TRAIN ONLY, metadata only.

For every pack_v3 TRAIN (batch, chunk, arm, policy) group this reports:

* item count and how many carry manifest ``outcome_rows``;
* how many are zeroed by the owner decision-(b) exclusion (``base`` /
  ``native_base``);
* the resolved v0.4a label source (``features.outcomes.label_source``);
* whether the corresponding frozen v0.4a label parquet exists on E:;
* whether the per-item shard already holds ``outcomes.parquet``;
* whether the raw run ``physics_steps.jsonl`` named by the shard
  ``meta.json`` still exists, and its byte size.

It reads only TRAIN metadata/manifests; it never opens CAL/HOLDOUT/chunk-5/
J4B/INC1 data, never reads a label parquet, never simulates and writes only
the requested JSON summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PACK = Path("D:/rd-staging/wmn1_feat/pack_v3")
SIDE = Path("D:/rd-staging/wmn1_feat/labels_fix1_20261003T141730Z")
V04A = Path("E:/research-dense/output/wd01_partB_v0data_v04a")
WMN1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WMN1))
from features.outcomes import label_source, labels_path  # noqa: E402

EXCLUDED_ARMS = ("base", "native_base")


def _meta_raw(shard_dir: Path):
    """(raw physics path, bytes) from a shard meta.json, or (None, 0)."""
    mp = shard_dir / "meta.json"
    if not mp.is_file():
        return None, 0
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None, 0
    ip = m.get("input_physics")
    if not ip:
        return None, 0
    p = Path(ip)
    return (str(p), p.stat().st_size) if p.is_file() else (str(p), 0)


def main() -> int:
    man = json.loads((PACK / "manifest.json").read_text(encoding="utf-8"))
    idx = json.loads((PACK / "INDEX.json").read_text(encoding="utf-8"))
    runs = {(r.get("batch") or r["chunk"], r["world"], r["arm"]): r
            for r in idx["runs"]}
    items = [dict(i) for i in man["items"]
             if str(i.get("split", "")).upper() == "TRAIN"]

    groups: dict = {}
    for it in items:
        batch = it.get("batch") or it["chunk"]
        chunk, arm = it["chunk"], it["arm"]
        policy = it.get("policy")
        key = (str(batch), str(chunk), str(arm), str(policy))
        g = groups.setdefault(key, dict(
            n=0, zero_rows=0, has_rows=0, excluded_b=0, v04a_file=0,
            shard_outcomes=0, src=set(), raw_present=0, raw_missing=0,
            raw_unknown=0, raw_bytes=0, sample_world=None))
        g["n"] += 1
        nrows = len([p for p in it.get("outcome_rows", []) if int(p) >= 0])
        if nrows == 0:
            g["zero_rows"] += 1
        else:
            g["has_rows"] += 1
        if str(arm).strip().lower() in EXCLUDED_ARMS:
            g["excluded_b"] += 1
        src = label_source(batch, chunk, arm, policy)
        g["src"].add(src)
        if src and labels_path(chunk, it["world"], src).is_file():
            g["v04a_file"] += 1
        r = runs.get((batch, it["world"], arm))
        sd = Path(str(r["shard_dir"])) if r else None
        if sd and (sd / "outcomes.parquet").is_file():
            g["shard_outcomes"] += 1
        raw, nb = _meta_raw(sd) if sd else (None, 0)
        if raw is None:
            g["raw_unknown"] += 1
        elif nb > 0:
            g["raw_present"] += 1
            g["raw_bytes"] += nb
        else:
            g["raw_missing"] += 1
        if g["sample_world"] is None:
            g["sample_world"] = it["world"]

    out = []
    for key in sorted(groups):
        g = groups[key]
        out.append(dict(batch=key[0], chunk=key[1], arm=key[2], policy=key[3],
                        n_items=g["n"], manifest_has_rows=g["has_rows"],
                        manifest_zero_rows=g["zero_rows"],
                        excluded_owner_b=g["excluded_b"],
                        resolved_sources=sorted(x for x in g["src"]),
                        v04a_label_file_present=g["v04a_file"],
                        shard_outcomes_present=g["shard_outcomes"],
                        raw_run_present=g["raw_present"],
                        raw_run_missing=g["raw_missing"],
                        raw_run_unknown=g["raw_unknown"],
                        raw_physics_bytes=g["raw_bytes"],
                        sample_world=g["sample_world"]))
    summary = dict(
        n_train_items=len(items),
        n_zero_rows=sum(g["zero_rows"] for g in groups.values()),
        n_has_rows=sum(g["has_rows"] for g in groups.values()),
        n_excluded_b=sum(g["excluded_b"] for g in groups.values()),
        groups=out)

    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
