"""FEAT1 shard-key injectivity audit over the REAL refreshed inventory.

Metadata-only (inventory.parquet + pairs.parquet are never run payloads).
The crashed FEAT1 run keyed shards by the plan-level POLICY arm
(``shards/chunk1/<world>/fixed``), so replicated arms ``__fixed1/__fixed2``
and ``__switching1/2/3`` (and the cross-batch WM-72 chunk2/3 vs truearms
native/pi runs) collapsed into the same shard.  The fix keys every shard by
``<batch>/<world>/<run-dir arm suffix>``; this audit proves the mapping is
injective over every readable TRAIN/CALIBRATION run and that the inventory
contrast pairs resolve to distinct shards.
"""
import pandas as pd
import pytest

from corpus import _util as U
from corpus import featurize_all as FA

INV = U.WM_N1_ROOT / "inventory" / "inventory.parquet"
PAIRS = U.WM_N1_ROOT / "inventory" / "pairs.parquet"

pytestmark = pytest.mark.skipif(
    not INV.is_file(), reason="inventory/inventory.parquet not present")


def _readable(df):
    keep = df[df["split"].isin(("TRAIN", "CALIBRATION"))
              & df["present"].astype(bool)
              & df["is_complete"].astype(bool)
              & (~df["is_smoke"].astype(bool))]
    return keep[~keep["world"].astype(str).str.upper().str.startswith(
        "HOLDOUT")]


def _selected(df):
    return df[df["split"].isin(("TRAIN", "CALIBRATION"))]


def _old_shard(r):
    """The crashed keying: chunk label + plan-level POLICY arm."""
    return f"{FA.chunk_label(r['batch'])}/{r['world']}/{r['arm']}"


def _new_shard(r):
    """The fixed keying: batch + exact run-dir arm suffix."""
    return (f"{r['batch']}/{r['world']}/"
            f"{U.arm_suffix(r['run'], r['arm'])}")


def test_shard_key_injective_over_real_inventory():
    df = pd.read_parquet(INV)
    all_tc = _selected(df)
    sel = _readable(df)

    def collisions(rows, key):
        shards = rows.assign(shard=rows.apply(key, axis=1))
        return len(rows) - int(shards["shard"].nunique())

    n_all, n_sel = len(all_tc), len(sel)
    before_all = collisions(all_tc, _old_shard)
    after_all = collisions(all_tc, _new_shard)
    before_sel = collisions(sel, _old_shard)
    after_sel = collisions(sel, _new_shard)
    print("\nshard-key injectivity audit (inventory/inventory.parquet):")
    print(f"  TRAIN+CAL runs in inventory: {n_all} "
          f"(readable present+complete non-smoke: {n_sel})")
    print(f"  shard-path collisions BEFORE fix (policy-keyed): "
          f"{before_all} (readable subset: {before_sel})")
    print(f"  shard-path collisions AFTER fix (batch + arm suffix): "
          f"{after_all} (readable subset: {after_sel})")
    assert n_all >= 2475, (f"expected the refreshed corpus (>= 2475 "
                           f"TRAIN+CAL runs), got {n_all}")
    assert n_sel >= 2463
    assert before_all > 0, "the pre-fix keying must collide on this inventory"
    assert after_all == 0, "the (batch, world, arm-suffix) key must be unique"
    assert after_sel == 0


def test_inventory_pairs_resolve_to_distinct_shards():
    if not PAIRS.is_file():
        pytest.skip("inventory/pairs.parquet not present")
    df = pd.read_parquet(INV)
    sel = _readable(df)
    # key by (batch, run): cross-batch runs can share the run-dir name
    shard_of = {(r["batch"], r["run"]): _new_shard(r)
                for _, r in sel.iterrows()}
    pairs = pd.read_parquet(PAIRS)
    n_total = len(pairs)
    n_missing = 0
    n_same = 0
    for _, p in pairs.iterrows():
        a = shard_of.get((p["native_batch"], p["native_run"]))
        b = shard_of.get((p["other_batch"], p["other_run"]))
        if a is None or b is None:
            n_missing += 1
        elif a == b:
            n_same += 1
    print("\ninventory contrast pairs over the fixed shard key:")
    print(f"  pair rows: {n_total}; runs without a shard: {n_missing}; "
          f"native/other collapsed into one shard: {n_same}")
    assert n_total > 0
    assert n_missing == 0, "every pair run must map to a planned shard"
    assert n_same == 0, "a treatment run must never share its reference shard"
