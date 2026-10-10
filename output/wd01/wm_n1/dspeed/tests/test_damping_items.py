"""Synthetic tests for the damping-snap item list, exclusion marker and the
re-aligned outcome overlay (no real pack / no simulation)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from dspeed import damping as DM


def _manifest(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    items = [
        dict(world="w1", arm="base", chunk="chunk1", batch="chunk1",
             split="TRAIN", t0_s=100.0, outcome_rows=[0, 1]),
        dict(world="w1", arm="native_base", chunk="chunk1", batch="chunk1",
             split="TRAIN", t0_s=100.0, outcome_rows=[2, 3]),
        dict(world="w2", arm="base", chunk="chunk1", batch="chunk1",
             split="TRAIN", t0_s=200.0, outcome_rows=[]),        # zero row
        dict(world="w3", arm="native", chunk="chunk1", batch="chunk1",
             split="TRAIN", t0_s=300.0, outcome_rows=[4]),        # not damping
        dict(world="w4", arm="base", chunk="chunk1", batch="chunk1",
             split="CALIBRATION", t0_s=400.0, outcome_rows=[5]),  # not TRAIN
    ]
    (base / "manifest.json").write_text(
        json.dumps({"schema": "wm_n1_pack/1", "k_max": 192,
                    "items": items}), encoding="utf-8")
    return base, items


def test_load_base_items_filters_and_sorts(tmp_path):
    base, _ = _manifest(tmp_path)
    got = DM.load_base_items(base)
    assert [(it["world"], it["arm"]) for it in got] == [
        ("w1", "base"), ("w1", "native_base")]


def test_arm_policy_maps_to_reference_and_treatment():
    assert DM.ARM_POLICY["base"] == "damping"
    assert DM.ARM_POLICY["native_base"] == "native"


def test_exclusion_marker_bypasses_owner_b():
    from train.dataset import (is_outcome_supervision_excluded,
                               OUTCOME_SUPERVISION_EXCLUDED_ARMS)
    assert OUTCOME_SUPERVISION_EXCLUDED_ARMS == ("base", "native_base")
    assert is_outcome_supervision_excluded({"arm": "base"}) is True
    assert is_outcome_supervision_excluded(
        {"arm": "base", "damping_snap": True}) is False
    assert is_outcome_supervision_excluded(
        {"arm": "native_base", "damping_snap": True}) is False


def test_damping_snap_item_pairs_with_supervised_reference():
    """A damping_snap treatment pairs with the RE-ALIGNED (supervised)
    native_base, never with the owner-(b) excluded pack_v3 origin-0 copy."""
    from train.dataset import _reference_of, contrast_pairs
    items = [
        # pack_v3 origin-0 native_base: owner-(b) excluded (NO marker)
        dict(pair_key="w1@t0=100", arm="native_base", policy="native",
             outcome_rows=[0]),
        # re-aligned damping-snap native_base: supervised -> preferred ref
        dict(pair_key="w1@t0=100", arm="native_base", policy="native",
             damping_snap=True, outcome_rows=[]),
        # re-aligned damping-snap base: the treatment
        dict(pair_key="w1@t0=100", arm="base", policy="damping",
             damping_snap=True, outcome_rows=[]),
    ]
    assert _reference_of([0, 1, 2], items) == 1
    pairs = contrast_pairs(items)
    assert (2, 1) in pairs
    assert all(ref != 0 for _, ref in pairs)


def test_excluded_reference_still_used_when_no_supervised_sibling():
    """Without any supervised reference the historical choice is frozen."""
    from train.dataset import _reference_of, contrast_pairs
    items = [
        dict(pair_key="w1@t0=100", arm="native_base", policy="native",
             outcome_rows=[0]),
        dict(pair_key="w1@t0=100", arm="base", policy="damping",
             outcome_rows=[1]),
    ]
    assert _reference_of([0, 1], items) == 0
    assert contrast_pairs(items) == [(1, 0)]


def _write_shard(shards, arm, t0=100.0):
    sd = shards / "chunk1" / "w1" / arm
    sd.mkdir(parents=True)
    (sd / "meta.json").write_text(json.dumps(
        {"world": "w1", "arm": arm, "chunk": "chunk1", "split": "TRAIN",
         "t_origin": t0, "t0_s": t0, "decision_t": [t0, t0 + 30.0]}),
        encoding="utf-8")
    (sd / "ALIGN.json").write_text(json.dumps(
        {"world": "w1", "arm": arm, "chunk": "chunk1", "t0_s": t0,
         "t_origin": t0, "first_physics_time": t0 + 0.5,
         "snapshot_at_t0": True}), encoding="utf-8")


def _labels(tmp_path, t0=100.0):
    rows = []
    for arm, policy in (("base", "damping_fleet"),
                        ("native_base", "native_fleet")):
        for t in (t0, t0 + 30.0):
            day = t - t0
            for tau in (30.0, 60.0):
                if tau > (t0 + 600.0 - t) + 1e-9:
                    continue
                rows.append(dict(
                    world="w1", source=arm, chunk_id="chunk1", split="TRAIN",
                    t0_s=t0, policy=policy, t=t, tau=tau, tau_available=600.0,
                    total_delay_ref_s=float(day + tau), ttc_lt3_veh_s=0.0,
                    hard_brake_s=0.0, collision_any=0))
    lc = tmp_path / "lc"
    lc.mkdir()
    pd.DataFrame(rows).to_parquet(lc / "base_train_outcomes.parquet",
                                  index=False)
    return lc


def test_overlay_accepts_aligned_items(tmp_path, monkeypatch):
    from dspeed.build_damping_overlay import build
    base, _ = _manifest(tmp_path)
    shards = tmp_path / "shards"
    _write_shard(shards, "base")
    _write_shard(shards, "native_base")
    lc = _labels(tmp_path)
    pack = tmp_path / "packd"
    pack.mkdir()
    (pack / "manifest.json").write_text("{}", encoding="utf-8")
    overlay = tmp_path / "overlay_out"
    monkeypatch.setattr(DM, "BASE_PACK", base)
    monkeypatch.setattr(DM, "SHARDS_ROOT", shards)
    monkeypatch.setattr(DM, "LABELS_COV", lc)
    monkeypatch.setattr(DM, "PACK_ROOT", pack)
    monkeypatch.setattr(DM, "OVERLAY_ROOT", overlay)
    rep = build()
    assert rep["n_accepted_groups"] == 2
    assert rep["n_accepted_rows"] > 0
    assert (overlay / "overlay_outcomes.parquet").is_file()


def test_overlay_fails_closed_on_misalignment(tmp_path, monkeypatch):
    from dspeed.build_damping_overlay import build
    base, _ = _manifest(tmp_path)
    shards = tmp_path / "shards"
    _write_shard(shards, "base")
    _write_shard(shards, "native_base")
    bad = shards / "chunk1" / "w1" / "native_base" / "ALIGN.json"
    bad.write_text(json.dumps(
        {"world": "w1", "arm": "native_base", "chunk": "chunk1",
         "t0_s": 100.0, "t_origin": 100.0, "first_physics_time": 0.5,
         "snapshot_at_t0": True}), encoding="utf-8")
    lc = _labels(tmp_path)
    pack = tmp_path / "packd"
    pack.mkdir()
    (pack / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(DM, "BASE_PACK", base)
    monkeypatch.setattr(DM, "SHARDS_ROOT", shards)
    monkeypatch.setattr(DM, "LABELS_COV", lc)
    monkeypatch.setattr(DM, "PACK_ROOT", pack)
    monkeypatch.setattr(DM, "OVERLAY_ROOT", tmp_path / "overlay_out")
    with pytest.raises(DM.DampingError):
        build()
