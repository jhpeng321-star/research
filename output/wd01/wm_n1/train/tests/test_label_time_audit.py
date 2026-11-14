"""Generated metadata and projected columns only; no registered labels."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


_SOURCE = Path(__file__).resolve().parents[1] / "label_time_audit.py"
_SPEC = importlib.util.spec_from_file_location("wmn1_label_time_audit", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
A = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(A)


def _fixture(monkeypatch):
    monkeypatch.setattr(A, "N_ITEMS", 2)
    monkeypatch.setattr(A, "N_ROWS", 4)
    monkeypatch.setattr(A, "N_POSITIVE", 1)
    monkeypatch.setattr(A, "N_ZERO", 1)
    select = []
    runs = []
    items = []
    for pos, (world, arm, first, n) in enumerate((
            ("w1", "native", 5, 4), ("w2", "base", 9, 0))):
        key = f"chunk1/{world}__{arm}"
        rel = f"chunk1/{world}/{arm}"
        plan_item = dict(position=pos, key=key, batch="chunk1",
                         chunk="chunk1", world=world, arm=arm,
                         split="TRAIN", t0_s=240.0, tick=pos * 10 + 2,
                         plan_row=pos, tick_start=pos * 10,
                         tick_end=pos * 10 + 10, n_outcome_rows=n,
                         shard_rel=rel, input_sha256="d" * 64)
        select.append(plan_item)
        runs.append(dict(key=key, split="TRAIN", shard_rel=rel,
                         input_sha256="d" * 64, first_global_row=first,
                         n_outcome_rows=n,
                         outcomes_sha256="e" * 64 if n else None))
        items.append(dict(world=world, arm=arm, chunk="chunk1",
                          batch="chunk1", split="TRAIN", t0_s=240.0,
                          tick=pos * 10 + 2, plan_row=pos,
                          policy="native" if arm == "native" else "damping",
                          pair_key=f"{world}@t0=240",
                          first_global_row=first, n_rows=n))
    plan = dict(k_max=192, n_items=2, manifest_sha256="a" * 64,
                index_sha256="b" * 64, selection=select)
    plan_receipt = dict(schema=A.PLAN_SCHEMA,
                        pack="E:/generated/pack", plan=plan,
                        selection_sha256=A.canonical_sha(plan))
    isolation = dict(schema=A.RECEIPT_SCHEMA, split="TRAIN",
                     pack=dict(dir="E:/generated/pack",
                               manifest_sha256="a" * 64,
                               index_sha256="b" * 64),
                     sidecar=dict(file="train_outcomes.parquet",
                                  sha256="c" * 64, parquet_index="_global_row_id",
                                  n_rows=4,
                                  columns={name: "synthetic" for name in A.PROJECTION}),
                     selection=dict(n_runs=2, n_rows=4),
                     runs=runs, items=items)
    spec = dict(inputs={"train_labels": {"sha256": "c" * 64}},
                expected_pack_manifest_sha256="a" * 64,
                expected_pack_index_sha256="b" * 64)
    return plan_receipt, isolation, spec


def _columns():
    return dict(zip(A.PROJECTION, (
        [5, 6, 7, 8], ["w1"] * 4, ["native"] * 4,
        ["chunk1"] * 4, ["TRAIN"] * 4, [240.0] * 4,
        ["native_fleet"] * 4, [240.0, 240.0, 270.0, 300.0],
        [60.0, 120.0, 60.0, 120.0], [60.0, 120.0, 30.0, 120.0])))


def test_last_row_timestamp_and_numeric_duration_are_observational(monkeypatch):
    plan, receipt, spec = _fixture(monkeypatch)
    ordered, by_id = A.validate_receipts(plan, receipt, spec)
    assert len(ordered) == 2 and len(by_id) == 4
    assert ordered[0]["inferred_snapshot_t_s"] == 241.0
    assert ordered[1]["inferred_snapshot_t_s"] == 1.0
    records = A.analyze_projected_rows(_columns(), ordered, by_id)
    cell = next(r for r in records if r["world"] == "w1" and r["tau_s"] == 60.0)
    assert cell["n_rows"] == 2 and cell["n_distinct_t"] == 2
    assert cell["t_in_original_order_s"] == [240.0, 270.0]
    assert cell["selected_global_row_id"] == 7
    assert cell["selected_t_s"] == 270.0
    assert cell["selected_minus_declared_t0_s"] == 30.0
    assert cell["selected_minus_inferred_snapshot_s"] == 29.0
    assert cell["training_mask"] is True
    assert cell["duration_ge_tau"] is False
    assert all(r["selected_t_s"] is None for r in records if r["world"] == "w2")


@pytest.mark.parametrize("column,value", [
    ("split", "CALIBRATION"),
    ("source", "other-arm"),
    ("chunk_id", "chunk5"),
    ("policy", "pi"),
    ("_global_row_id", 99),
])
def test_projected_nontrain_or_foreign_identity_refuses(monkeypatch,
                                                        column, value):
    plan, receipt, spec = _fixture(monkeypatch)
    ordered, by_id = A.validate_receipts(plan, receipt, spec)
    columns = _columns()
    columns[column][0] = value
    with pytest.raises(A.AuditRefusal):
        A.analyze_projected_rows(columns, ordered, by_id)


def test_plan_duplicate_position_refuses_before_row_projection(monkeypatch):
    plan, receipt, spec = _fixture(monkeypatch)
    plan["plan"]["selection"][1]["position"] = 0
    plan["selection_sha256"] = A.canonical_sha(plan["plan"])
    with pytest.raises(A.AuditRefusal, match="positions duplicate"):
        A.validate_receipts(plan, receipt, spec)


@pytest.mark.parametrize("run_index,outcomes_sha", [
    (0, None),
    (1, "e" * 64),
])
def test_positive_and_zero_source_hash_contract_refuses(monkeypatch,
                                                        run_index,
                                                        outcomes_sha):
    plan, receipt, spec = _fixture(monkeypatch)
    receipt["runs"][run_index]["outcomes_sha256"] = outcomes_sha
    with pytest.raises(A.AuditRefusal):
        A.validate_receipts(plan, receipt, spec)


def test_out_of_order_item_timestamps_are_reported_without_reordering(monkeypatch):
    plan, receipt, spec = _fixture(monkeypatch)
    ordered, by_id = A.validate_receipts(plan, receipt, spec)
    columns = _columns()
    columns["t"] = [270.0, 240.0, 300.0, 240.0]
    records = A.analyze_projected_rows(columns, ordered, by_id)
    cell = next(r for r in records if r["world"] == "w1"
                and r["tau_s"] == 60.0)
    assert cell["t_in_original_order_s"] == [270.0, 300.0]
    assert cell["selected_t_s"] == 300.0
    assert cell["item_t_tau_order_inversions"] > 0


def test_duration_uses_producer_tolerance_but_keeps_mapper_mask():
    assert A.availability(60.0 - 5e-10, 60.0)["duration_ge_tau"] is True
    assert A.availability(30.0, 60.0) == dict(
        kind="numeric_duration", raw=30.0, training_mask=True,
        duration_ge_tau=False)
    assert A.availability(None, 60.0)["training_mask"] is True
    assert A.availability(False, 60.0)["training_mask"] is False


def test_only_ten_projected_columns_reach_arrow_read(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "N_ROWS", 4)
    import pyarrow.parquet as pq

    calls = []

    class FakeTable:
        num_rows = 4
        column_names = list(A.PROJECTION)

        def column(self, name):
            return SimpleNamespace(to_pylist=lambda: _columns()[name])

    class FakeParquet:
        metadata = SimpleNamespace(num_rows=4)
        schema_arrow = SimpleNamespace(
            names=list(A.PROJECTION) + ["total_delay_ref_s", "collision_any"])

        def __init__(self, path):
            calls.append(("open", path))

        def read(self, *, columns, use_threads):
            calls.append(("read", columns, use_threads))
            return FakeTable()

    monkeypatch.setattr(pq, "ParquetFile", FakeParquet)
    result = A._read_projection(tmp_path / "generated.parquet")
    assert set(result) == set(A.PROJECTION)
    assert calls[-1] == ("read", list(A.PROJECTION), False)


# --- source-supported policy correspondence (WM-N1-LABEL-TIME-FULL) --------

def test_correspondence_table_is_exact_and_cited():
    assert A.POLICY_CORRESPONDENCE == {
        "damping": "damping_fleet", "native": "native_fleet",
        "pi": "pi_fleet", "fixed": "fixed_mix",
        "switching": "switching_mix"}
    doc = _SOURCE.read_text(encoding="utf-8")
    for receipt, label in A.POLICY_CORRESPONDENCE.items():
        assert f"``{receipt}`` <-> ``{label}``" in doc
    assert A.policy_correspondence("damping", "damping_fleet") == "mapped"


@pytest.mark.parametrize("receipt,label,status", [
    ("wall", "wall_fleet", "unmapped"),
    ("damping", 7, "unmapped"),
    ("switching", "native_fleet", "contradicting"),
    ("native", "native", "contradicting"),
    ("wall", "damping_fleet", "contradicting"),
])
def test_unmapped_or_contradicting_pairs_are_classified(receipt, label,
                                                        status):
    assert A.policy_correspondence(receipt, label) == status


def test_mapped_pair_passes_and_raw_names_are_preserved(monkeypatch):
    plan, receipt, spec = _fixture(monkeypatch)
    ordered, by_id = A.validate_receipts(plan, receipt, spec)
    records = A.analyze_projected_rows(_columns(), ordered, by_id)
    w1 = [r for r in records if r["world"] == "w1"]
    assert all(r["policy"] == "native" for r in w1)
    assert all(r["label_policy"] == ["native_fleet"] for r in w1)
    cell = next(r for r in w1 if r["tau_s"] == 60.0)
    assert cell["has_row_at_declared_t0"] is True
    assert cell["n_rows_duration_lt_tau"] == 1


@pytest.mark.parametrize("label", ["native", "switching_mix", "wall_fleet"])
def test_unmapped_or_contradicting_label_policy_fails_closed(monkeypatch,
                                                             label):
    plan, receipt, spec = _fixture(monkeypatch)
    ordered, by_id = A.validate_receipts(plan, receipt, spec)
    columns = _columns()
    columns["policy"][2] = label
    with pytest.raises(A.AuditRefusal, match="identity differs"):
        A.analyze_projected_rows(columns, ordered, by_id)


def test_timing_summary_counts_misbound_cells(monkeypatch):
    plan, receipt, spec = _fixture(monkeypatch)
    ordered, by_id = A.validate_receipts(plan, receipt, spec)
    records = A.analyze_projected_rows(_columns(), ordered, by_id)
    summary = A.timing_summary(records)
    assert summary["n_zero_label_items"] == 1
    assert summary["n_positive_items"] == 1
    assert summary["n_rows"] == 4
    assert summary["items_by_distinct_decision_times"] == {"3": 1}
    assert summary["rows_by_t0_tau"]["240/60"] == 2
    mis = summary["misbinding"]
    assert mis["n_cells"] == 2 and mis["n_items"] == 1
    assert mis["n_cells_t0_row_present_but_overridden"] == 2
    assert mis["selected_minus_t0_s"] == {"30": 1, "60": 1}
    assert summary["n_rows_duration_lt_tau"] == 1


# --- schema/2 sidecar and resolved-source correspondence (labelfix) --------

def _campaign_fixture(monkeypatch):
    monkeypatch.setattr(A, "N_ITEMS", 1)
    monkeypatch.setattr(A, "N_ROWS", 1)
    monkeypatch.setattr(A, "N_POSITIVE", 1)
    monkeypatch.setattr(A, "N_ZERO", 0)
    key = "chunk2/w1__native"
    rel = "chunk2/w1/native"
    plan_item = dict(position=0, key=key, batch="chunk2", chunk="chunk2",
                     world="w1", arm="native", split="TRAIN", t0_s=300.0,
                     tick=2, plan_row=0, tick_start=0, tick_end=10,
                     n_outcome_rows=1, shard_rel=rel, input_sha256="d" * 64)
    run = dict(key=key, split="TRAIN", shard_rel=rel, input_sha256="d" * 64,
               first_global_row=5, n_outcome_rows=1, outcomes_sha256="e" * 64)
    item = dict(world="w1", arm="native", chunk="chunk2", batch="chunk2",
                split="TRAIN", t0_s=300.0, tick=2, plan_row=0,
                policy="switching", pair_key="w1@t0=300",
                first_global_row=5, n_rows=1)
    plan = dict(k_max=192, n_items=1, manifest_sha256="a" * 64,
                index_sha256="b" * 64, selection=[plan_item])
    plan_receipt = dict(schema=A.PLAN_SCHEMA, pack="E:/generated/pack",
                        plan=plan, selection_sha256=A.canonical_sha(plan))
    isolation = dict(schema=A.RECEIPT_SCHEMA, split="TRAIN",
                     pack=dict(dir="E:/generated/pack",
                               manifest_sha256="a" * 64,
                               index_sha256="b" * 64),
                     sidecar=dict(file="train_outcomes.parquet",
                                  sha256="c" * 64,
                                  parquet_index="_global_row_id", n_rows=1,
                                  columns={name: "synthetic"
                                           for name in A.PROJECTION}),
                     selection=dict(n_runs=1, n_rows=1),
                     runs=[run], items=[item])
    spec = dict(inputs={"train_labels": {"sha256": "c" * 64}},
                expected_pack_manifest_sha256="a" * 64,
                expected_pack_index_sha256="b" * 64)
    columns = dict(zip(A.PROJECTION, (
        [5], ["w1"], ["swnative"], ["chunk2"], ["TRAIN"], [300.0],
        ["switching_mix"], [300.0], [60.0], [100.0])))
    return plan_receipt, isolation, spec, columns


def test_resolved_source_matches_v04a_registry():
    assert A.resolved_source("chunk2", "chunk2", "native", "switching") \
        == "swnative"
    assert A.resolved_source("chunk3", "chunk3", "pi", "switching") == "swpi"
    assert A.resolved_source("truearms_chunk2", "chunk2", "native",
                             "native") == "native"
    assert A.resolved_source("chunk4", "chunk4", "fixed1", "fixed") == "fixed1"
    assert A.resolved_source("chunk1", "chunk1", "native", "native") is None
    assert A.resolved_source("chunk4", "chunk4", "wall", "wall") is None
    assert A.resolved_source("chunk2", "chunk2", "base", "damping") == "base"


def test_campaign_switching_source_correspondence_passes(monkeypatch):
    plan, receipt, spec, columns = _campaign_fixture(monkeypatch)
    ordered, by_id = A.validate_receipts(plan, receipt, spec)
    records = A.analyze_projected_rows(columns, ordered, by_id)
    cell = next(r for r in records if r["tau_s"] == 60.0)
    assert cell["n_rows"] == 1
    assert cell["selected_global_row_id"] == 5
    assert cell["label_policy"] == ["switching_mix"]


def test_campaign_switching_source_contradiction_refuses(monkeypatch):
    plan, receipt, spec, columns = _campaign_fixture(monkeypatch)
    ordered, by_id = A.validate_receipts(plan, receipt, spec)
    columns["source"] = ["native"]
    with pytest.raises(A.AuditRefusal, match="identity differs"):
        A.analyze_projected_rows(columns, ordered, by_id)


def test_v2_receipt_schema_is_accepted(monkeypatch):
    plan, receipt, spec = _fixture(monkeypatch)
    receipt["schema"] = A.RECEIPT_SCHEMA_V2
    ordered, by_id = A.validate_receipts(plan, receipt, spec)
    assert len(ordered) == 2 and len(by_id) == 4


def test_v2_excluded_positive_run_is_zero_label(monkeypatch):
    plan, receipt, spec = _fixture(monkeypatch)
    receipt["schema"] = A.RECEIPT_SCHEMA_V2
    run = receipt["runs"][0]
    run["excluded"] = True
    run["outcomes_sha256"] = None
    run["reason"] = "synthetic exclusion"
    receipt["sidecar"]["n_rows"] = 0
    ordered, by_id = A.validate_receipts(plan, receipt, spec)
    assert by_id == {}
    assert all(row_id not in by_id for row_id in range(5, 9))
    assert ordered[0]["n_rows"] == 4
    empty = {name: [] for name in A.PROJECTION}
    records = A.analyze_projected_rows(empty, ordered, by_id)
    assert records and all(r["n_rows"] == 0 for r in records)


def test_v2_rebound_run_requires_valid_label_source_sha(monkeypatch):
    plan, receipt, spec = _fixture(monkeypatch)
    receipt["schema"] = A.RECEIPT_SCHEMA_V2
    run = receipt["runs"][0]
    run["rebound"] = True
    run["outcomes_sha256"] = None
    run["label_source_sha256"] = "f" * 64
    ordered, by_id = A.validate_receipts(plan, receipt, spec)
    assert len(by_id) == 4
    run["label_source_sha256"] = "0" * 63
    with pytest.raises(A.AuditRefusal):
        A.validate_receipts(plan, receipt, spec)
    del run["label_source_sha256"]
    with pytest.raises(A.AuditRefusal):
        A.validate_receipts(plan, receipt, spec)
