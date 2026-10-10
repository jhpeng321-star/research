"""Outcome reuse: key join (world, source, t, tau) from v0.4a label parquet."""
from __future__ import annotations

import pandas as pd
import pytest

from features import outcomes as O


def _labels(root, chunk, world, source, rows):
    p = O.labels_path(chunk, world, source, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(p)


def _item(world, chunk, arm, policy, t0, split="TRAIN"):
    return dict(world=world, chunk=chunk, arm=arm, batch=chunk, policy=policy,
                split=split, t0_s=t0)


def _base_rows():
    return [
        dict(world="w1", source="base", chunk_id="chunk4", split="TRAIN",
             t0_s=300.0, policy="damping_fleet", t=300.0, tau=30.0,
             total_delay_ref_s=1.5, tau_available=100.0),
        dict(world="w1", source="base", chunk_id="chunk4", split="TRAIN",
             t0_s=300.0, policy="damping_fleet", t=300.0, tau=570.0,
             total_delay_ref_s=9.5, tau_available=100.0),
        dict(world="w1", source="base", chunk_id="chunk4", split="TRAIN",
             t0_s=300.0, policy="damping_fleet", t=310.0, tau=30.0,
             total_delay_ref_s=2.5, tau_available=100.0),
    ]


def test_key_join_filters_decision_times_and_taus(tmp_path):
    _labels(tmp_path, "chunk4", "w1", "base", _base_rows())
    df = O.load_outcome_rows("chunk4", "w1", "base", [300.0], [30.0, 570.0],
                             root=tmp_path,
                             item=_item("w1", "chunk4", "base", "damping",
                                        300.0))
    assert len(df) == 2
    assert set(df["tau"]) == {30.0, 570.0}
    assert float(df[df["tau"] == 30.0]["total_delay_ref_s"].iloc[0]) == 1.5


def test_missing_labels_raises(tmp_path):
    with pytest.raises(O.OutcomeSourceError):
        O.load_outcome_rows("chunk4", "nope", "base", [0.0], [30.0],
                            root=tmp_path,
                            item=_item("nope", "chunk4", "base", "damping",
                                       0.0))


# --- WM-N1 labelfix source resolution + hard identity -----------------------

@pytest.mark.parametrize("batch,chunk,arm,policy,want", [
    ("chunk4", "chunk4", "base", "damping", "base"),
    ("chunk4", "chunk4", "native_base", "native", "native_base"),
    ("chunk4", "chunk4", "switching2", "switching", "switching2"),
    ("truearms_chunk2", "chunk2", "native", "native", "native"),
    ("truearms_chunk3", "chunk3", "pi", "pi", "pi"),
    ("chunk2", "chunk2", "native", "switching", "swnative"),
    ("chunk2", "chunk2", "pi", "switching", "swpi"),
    ("chunk1_fix27", "chunk1_fix27", "native", "switching", "swnative"),
    ("chunk1", "chunk1", "pi", "switching", "swpi"),
    # WM-N1 label coverage (2026-10-04): the campaign fixed / switching
    # branch arms and the walls-pilot arms resolve to their own source.
    ("chunk2", "chunk2", "switching1", "switching", "switching1"),
    ("chunk3", "chunk3", "switching3", "switching", "switching3"),
    ("chunk1", "chunk1", "fixed1", "fixed", "fixed1"),
    ("chunk1_fix27", "chunk1_fix27", "fixed2", "fixed", "fixed2"),
    ("walls_pilot_chunk4", "chunk4", "w-lo-early", "wall", "w-lo-early"),
    ("walls_pilot_chunk4", "chunk4", "w-mid-late", "wall", "w-mid-late"),
])
def test_label_source_mapping(batch, chunk, arm, policy, want):
    assert O.label_source(batch, chunk, arm, policy) == want


def test_switching_item_reads_swlabels(tmp_path):
    rows = [dict(world="w1", source="swnative", chunk_id="chunk2",
                 split="TRAIN", t0_s=480.0, policy="switching_mix", t=480.0,
                 tau=30.0, total_delay_ref_s=3.0, tau_available=200.0)]
    _labels(tmp_path, "chunk2", "w1", "swnative", rows)
    df = O.load_outcome_rows(
        "chunk2", "w1", O.label_source("chunk2", "chunk2", "native",
                                       "switching"),
        [480.0], [30.0], root=tmp_path,
        item=_item("w1", "chunk2", "native", "switching", 480.0))
    assert df["source"].tolist() == ["swnative"]
    assert df["policy"].tolist() == ["switching_mix"]


def test_fleet_item_reads_fleet_labels(tmp_path):
    rows = [dict(world="w1", source="native", chunk_id="chunk4",
                 split="TRAIN", t0_s=240.0, policy="native_fleet", t=240.0,
                 tau=60.0, total_delay_ref_s=2.0, tau_available=200.0)]
    _labels(tmp_path, "chunk4", "w1", "native", rows)
    df = O.load_outcome_rows("chunk4", "w1", "native", [240.0], [60.0],
                             root=tmp_path,
                             item=_item("w1", "chunk4", "native", "native",
                                        240.0))
    assert df["policy"].tolist() == ["native_fleet"]


@pytest.mark.parametrize("field,value", [
    ("source", "native"),
    ("policy", "native_fleet"),
    ("world", "other"),
    ("chunk_id", "chunk3"),
    ("t0_s", 481.0),
    ("split", "CALIBRATION"),
])
def test_identity_mismatch_fails_closed(tmp_path, field, value):
    rows = [dict(world="w1", source="swnative", chunk_id="chunk2",
                 split="TRAIN", t0_s=480.0, policy="switching_mix", t=480.0,
                 tau=30.0, total_delay_ref_s=3.0, tau_available=200.0)]
    rows[0][field] = value
    _labels(tmp_path, "chunk2", "w1", "swnative", rows)
    with pytest.raises(O.OutcomeSourceError):
        O.load_outcome_rows("chunk2", "w1", "swnative", [480.0], [30.0],
                            root=tmp_path,
                            item=_item("w1", "chunk2", "native", "switching",
                                       480.0))
