"""First TRAIN label-identity mismatch from an approved metadata projection.

The caller is responsible for the externally pinned ten-column read and the
approved plan/isolation-receipt item map. This function examines identifiers
only; it does not inspect timestamps beyond validating ``t0_s`` or aggregate
label timing.
"""
from __future__ import annotations

import math

PROJECTION = ("_global_row_id", "world", "source", "chunk_id", "split",
              "t0_s", "policy", "t", "tau", "tau_available")
IDENTITY_FIELDS = ("split", "world", "source", "chunk_id", "policy", "t0_s")


class DiagnosticRefusal(ValueError):
    """The bounded projected-row identity contract is malformed."""


def _require(condition, reason):
    if not condition:
        raise DiagnosticRefusal(reason)


def _typename(value):
    return type(value).__name__


def _expected_source(item, source_match):
    """Expected ``source`` for an item: correspondence or exact arm.

    ``source_match`` is ``label_time_audit.resolved_source``; when supplied the
    expected source is its resolution of the item's ``(batch, chunk, arm,
    policy)`` falling back to the raw ``arm`` for zero-label items.  Without
    it the original exact ``arm`` comparison is kept.
    """
    if source_match is None:
        return item["arm"]
    batch = item.get("batch", item["key"].split("/", 1)[0])
    return source_match(batch, item["chunk"], item["arm"],
                        item["policy"]) or item["arm"]


def first_identity_mismatch(projected, ordered, id_to_item, *,
                            expected_rows=75624, policy_match=None,
                            source_match=None):
    """Return the first identity difference, or explicit ``no_mismatch``.

    ``policy_match`` (optional) is ``label_time_audit.policy_correspondence``:
    the policy field then passes only when it returns ``"mapped"``; any other
    status is a difference reporting both raw names and the status.  Without
    it the original exact-equality comparison is used.  ``source_match``
    (optional) is ``label_time_audit.resolved_source``; the source field then
    compares against the resolved v0.4a source (falling back to the raw arm),
    otherwise it uses exact equality.  All other fields always use exact
    equality.

    Both ``ordered`` and ``id_to_item`` must come from the already verified
    TRAIN-only plan and isolation receipt. Row order is preserved exactly.
    Foreign, duplicate, missing, or out-of-order global IDs refuse analysis.
    """
    _require(type(expected_rows) is int and expected_rows >= 0,
             "invalid expected row count")
    _require(isinstance(projected, dict) and set(projected) == set(PROJECTION),
             "projection must contain exactly ten approved columns")
    _require(all(hasattr(projected[name], "__len__")
                 and len(projected[name]) == expected_rows
                 for name in PROJECTION), "projected row count mismatch")
    _require(isinstance(ordered, list) and isinstance(id_to_item, dict),
             "approved item map missing")
    _require(len(id_to_item) == expected_rows,
             "approved global row coverage count mismatch")
    approved_positions = {item["position"] for item in ordered}
    _require(len(approved_positions) == len(ordered),
             "approved item positions duplicate")
    columns = [projected[name] for name in PROJECTION]
    previous_id = -1
    seen_ids = set()
    for row_id, t0 in zip(projected["_global_row_id"], projected["t0_s"]):
        _require(type(row_id) is int and row_id >= 0,
                 "invalid global row ID")
        _require(row_id > previous_id,
                 "duplicate or out-of-order global row ID")
        previous_id = row_id
        _require(row_id in id_to_item and row_id not in seen_ids,
                 "foreign or duplicate global row ID")
        seen_ids.add(row_id)
        _require(type(t0) in (int, float) and math.isfinite(t0),
                 "label t0_s must be finite numeric")
    _require(seen_ids == set(id_to_item),
             "projected rows do not cover approved global IDs")
    for index, values in enumerate(zip(*columns)):
        row = dict(zip(PROJECTION, values))
        row_id = row["_global_row_id"]
        item = id_to_item[row_id]
        _require(item["position"] in approved_positions,
                 "global row maps to unapproved item")
        expected = dict(split="TRAIN", world=item["world"],
                        source=_expected_source(item, source_match),
                        chunk_id=item["chunk"],
                        policy=item["policy"], t0_s=item["t0_s"])
        differences = {
            field: dict(expected=expected[field], observed=row[field],
                        expected_type=_typename(expected[field]),
                        observed_type=_typename(row[field]))
            for field in IDENTITY_FIELDS
            if row[field] != expected[field]
            and not (field == "policy" and policy_match is not None)
        }
        if policy_match is not None:
            status = policy_match(expected["policy"], row["policy"])
            if status != "mapped":
                differences["policy"] = dict(
                    expected=expected["policy"], observed=row["policy"],
                    expected_type=_typename(expected["policy"]),
                    observed_type=_typename(row["policy"]),
                    correspondence=status)
        if differences:
            return dict(status="identity_mismatch", row_position=index,
                        global_row_id=row_id, item_position=item["position"],
                        run_key=item["key"], row=row,
                        expected_identity=expected, differences=differences,
                        rows_examined=expected_rows,
                        identity_rows_examined=index + 1)
    result = dict(status="no_mismatch", rows_examined=expected_rows,
                  identity_fields=list(IDENTITY_FIELDS))
    if policy_match is not None:
        result["policy_comparison"] = "source_correspondence_table"
    if source_match is not None:
        result["source_comparison"] = "resolved_source_table"
    return result


def identity_census(projected, id_to_item, policy_match, source_match=None):
    """Identity-only census of every raw (receipt, label) policy pair.

    Run only after :func:`first_identity_mismatch` has validated the row-ID
    stream.  Counts rows/items per raw (chunk, arm, receipt policy, label
    policy, correspondence status) and per non-policy field mismatch; no
    timestamp, horizon or availability value is inspected.  ``source_match``
    (optional) is ``label_time_audit.resolved_source``; when supplied the
    source field is compared against the resolved v0.4a source.
    """
    pairs = {}
    other = {}
    for row_id, world, source, chunk, split, t0, policy in zip(
            projected["_global_row_id"], projected["world"],
            projected["source"], projected["chunk_id"], projected["split"],
            projected["t0_s"], projected["policy"]):
        item = id_to_item[row_id]
        key = (item["chunk"], item["key"].split("/", 1)[0], item["arm"],
               item["policy"], policy, policy_match(item["policy"], policy))
        cell = pairs.setdefault(key, dict(rows=0, items=set()))
        cell["rows"] += 1
        cell["items"].add(item["position"])
        for field, got, want in (("split", split, "TRAIN"),
                                 ("world", world, item["world"]),
                                 ("source", source,
                                  _expected_source(item, source_match)),
                                 ("chunk_id", chunk, item["chunk"]),
                                 ("t0_s", t0, item["t0_s"])):
            if got != want:
                other[field] = other.get(field, 0) + 1
    rows = [dict(chunk=k[0], batch=k[1], arm=k[2], receipt_policy=k[3],
                 label_policy=k[4], correspondence=k[5], rows=v["rows"],
                 items=len(v["items"]))
            for k, v in sorted(pairs.items(), key=lambda kv: repr(kv[0]))]
    return dict(policy_pairs=rows, non_policy_field_mismatch_rows=other)
