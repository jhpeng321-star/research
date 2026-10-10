"""Generated identifiers only; no pipeline import or registered label read."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SOURCE = Path(__file__).resolve().parents[1] / "label_time_diag.py"
_SPEC = importlib.util.spec_from_file_location("wmn1_label_time_diag", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
D = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(D)


def _fixture():
    rows = [
        (5, "w1", "native", "chunk1", "TRAIN", 240.0, "native", 240., 60., True),
        (6, "w1", "native", "chunk1", "TRAIN", 240.0, "native", 270., 60., True),
        (7, "w2", "base", "chunk2", "TRAIN", 0.0, "damping", 0., 30., None),
    ]
    projected = {name: [row[i] for row in rows]
                 for i, name in enumerate(D.PROJECTION)}
    ordered = [dict(position=0, key="chunk1/w1__native", world="w1",
                    arm="native", chunk="chunk1", policy="native", t0_s=240.0),
               dict(position=1, key="chunk2/w2__base", world="w2",
                    arm="base", chunk="chunk2", policy="damping", t0_s=0.0)]
    by_id = {5: ordered[0], 6: ordered[0], 7: ordered[1]}
    return projected, ordered, by_id


def test_first_mismatch_reports_original_row_and_field_types():
    projected, ordered, by_id = _fixture()
    projected["policy"][1] = 9
    projected["world"][2] = "wrong-later"
    result = D.first_identity_mismatch(projected, ordered, by_id,
                                       expected_rows=3)
    assert result["status"] == "identity_mismatch"
    assert result["row_position"] == 1
    assert result["global_row_id"] == 6
    assert result["item_position"] == 0
    assert result["run_key"] == "chunk1/w1__native"
    assert result["rows_examined"] == 3
    assert result["identity_rows_examined"] == 2
    assert set(result["row"]) == set(D.PROJECTION)
    assert result["differences"] == {"policy": dict(
        expected="native", observed=9,
        expected_type="str", observed_type="int")}


def test_clean_projection_returns_explicit_no_mismatch():
    projected, ordered, by_id = _fixture()
    result = D.first_identity_mismatch(projected, ordered, by_id,
                                       expected_rows=3)
    assert result == dict(status="no_mismatch", rows_examined=3,
                          identity_fields=list(D.IDENTITY_FIELDS))


@pytest.mark.parametrize("bad_ids", ([5, 5, 7], [6, 5, 7], [5, 6, 99]))
def test_duplicate_out_of_order_or_foreign_ids_refuse(bad_ids):
    projected, ordered, by_id = _fixture()
    projected["_global_row_id"] = bad_ids
    with pytest.raises(D.DiagnosticRefusal):
        D.first_identity_mismatch(projected, ordered, by_id, expected_rows=3)


def test_count_and_nonfinite_t0_refuse():
    projected, ordered, by_id = _fixture()
    with pytest.raises(D.DiagnosticRefusal, match="row count"):
        D.first_identity_mismatch(projected, ordered, by_id, expected_rows=4)
    projected["t0_s"][0] = float("nan")
    with pytest.raises(D.DiagnosticRefusal, match="finite numeric"):
        D.first_identity_mismatch(projected, ordered, by_id, expected_rows=3)


def test_later_foreign_id_refuses_even_if_earlier_identity_differs():
    projected, ordered, by_id = _fixture()
    projected["policy"][0] = "wrong"
    projected["_global_row_id"][2] = 99
    with pytest.raises(D.DiagnosticRefusal, match="foreign"):
        D.first_identity_mismatch(projected, ordered, by_id, expected_rows=3)


# --- source-supported policy correspondence (WM-N1-LABEL-TIME-FULL) --------

_ASPEC = importlib.util.spec_from_file_location(
    "wmn1_label_time_audit_for_diag",
    Path(__file__).resolve().parents[1] / "label_time_audit.py")
A = importlib.util.module_from_spec(_ASPEC)
_ASPEC.loader.exec_module(A)


def _fleet_fixture():
    projected, ordered, by_id = _fixture()
    projected["policy"] = ["native_fleet", "native_fleet", "damping_fleet"]
    return projected, ordered, by_id


def test_mapped_policy_pairs_pass_with_correspondence_table():
    projected, ordered, by_id = _fleet_fixture()
    result = D.first_identity_mismatch(projected, ordered, by_id,
                                       expected_rows=3,
                                       policy_match=A.policy_correspondence)
    assert result["status"] == "no_mismatch"
    assert result["policy_comparison"] == "source_correspondence_table"


@pytest.mark.parametrize("row,receipt,label,status", [
    (1, "native", "switching_mix", "contradicting"),
    (1, "native", "native", "contradicting"),
    (2, "wall", "wall_fleet", "unmapped")])
def test_unmapped_or_contradicting_policy_reports_raw_names(row, receipt,
                                                            label, status):
    projected, ordered, by_id = _fleet_fixture()
    by_id[projected["_global_row_id"][row]]["policy"] = receipt
    projected["policy"][row] = label
    result = D.first_identity_mismatch(projected, ordered, by_id,
                                       expected_rows=3,
                                       policy_match=A.policy_correspondence)
    assert result["status"] == "identity_mismatch"
    assert result["global_row_id"] == projected["_global_row_id"][row]
    assert result["row"]["policy"] == label
    assert result["differences"] == {"policy": dict(
        expected=receipt, observed=label, expected_type="str",
        observed_type="str", correspondence=status)}


def test_identity_census_keeps_raw_policy_pairs():
    projected, ordered, by_id = _fleet_fixture()
    projected["policy"][1] = "switching_mix"
    census = D.identity_census(projected, by_id, A.policy_correspondence)
    pairs = {(c["receipt_policy"], c["label_policy"], c["correspondence"]):
             (c["rows"], c["items"]) for c in census["policy_pairs"]}
    assert pairs == {("native", "native_fleet", "mapped"): (1, 1),
                     ("native", "switching_mix", "contradicting"): (1, 1),
                     ("damping", "damping_fleet", "mapped"): (1, 1)}
    assert census["non_policy_field_mismatch_rows"] == {}


# --- resolved-source correspondence (WM-N1 labelfix) -----------------------

def _campaign_fixture():
    projected, ordered, by_id = _fixture()
    item = dict(position=0, key="chunk2/w1__native", world="w1",
                arm="native", chunk="chunk2", policy="switching", t0_s=240.0)
    ordered = [item, ordered[1]]
    by_id = {5: item, 6: item, 7: ordered[1]}
    projected["source"] = ["swnative", "swnative", "base"]
    projected["chunk_id"] = ["chunk2", "chunk2", "chunk2"]
    projected["policy"] = ["switching_mix", "switching_mix", "damping_fleet"]
    return projected, ordered, by_id


def test_resolved_source_correspondence_passes_for_campaign_switching():
    projected, ordered, by_id = _campaign_fixture()
    result = D.first_identity_mismatch(
        projected, ordered, by_id, expected_rows=3,
        policy_match=A.policy_correspondence, source_match=A.resolved_source)
    assert result["status"] == "no_mismatch"
    assert result["source_comparison"] == "resolved_source_table"


def test_contradicting_source_refuses_with_resolved_correspondence():
    projected, ordered, by_id = _campaign_fixture()
    projected["source"][0] = "native"
    result = D.first_identity_mismatch(
        projected, ordered, by_id, expected_rows=3,
        policy_match=A.policy_correspondence, source_match=A.resolved_source)
    assert result["status"] == "identity_mismatch"
    assert result["differences"]["source"] == dict(
        expected="swnative", observed="native",
        expected_type="str", observed_type="str")


def test_source_match_absent_falls_back_to_exact_arm():
    projected, ordered, by_id = _campaign_fixture()
    result = D.first_identity_mismatch(projected, ordered, by_id,
                                       expected_rows=3)
    assert result["status"] == "identity_mismatch"
    assert result["differences"]["source"]["expected"] == "native"


def test_identity_census_uses_resolved_source_when_supplied():
    projected, ordered, by_id = _campaign_fixture()
    census = D.identity_census(projected, by_id, A.policy_correspondence,
                               source_match=A.resolved_source)
    assert census["non_policy_field_mismatch_rows"] == {}
