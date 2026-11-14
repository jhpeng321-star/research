"""Incomplete / missing runs are flagged, never silently counted."""
import json

import inventory_lib as inv
from fixtures import SPEC_A, TRAIN_WORLD, build_fixture_root


def test_failed_receipt_flagged(tmp_path):
    root = build_fixture_root(tmp_path)
    receipt = root / SPEC_A["root_rel"] / f"{TRAIN_WORLD}__fixed1" / "RUN_RECEIPT.json"
    data = json.loads(receipt.read_text(encoding="utf-8"))
    data["exit_status"] = "failed"
    receipt.write_text(json.dumps(data), encoding="utf-8")

    guard = inv.make_guard(root)
    rows, _h, _p = inv.build_rows(root, guard, specs=(SPEC_A,))
    failed = [r for r in rows if r["run"] == f"{TRAIN_WORLD}__fixed1"][0]
    assert failed["is_complete"] is False
    refused = inv.incomplete_runs(rows)
    assert any(r["run"] == f"{TRAIN_WORLD}__fixed1" for r in refused)


def test_missing_run_flagged(tmp_path):
    root = build_fixture_root(tmp_path)
    plan_path = root / SPEC_A["plan_rel"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["runs"].append({"arm": "fixed", "run": f"{TRAIN_WORLD}__fixed9",
                         "world": TRAIN_WORLD, "split": "TRAIN", "duration_s": 600.0,
                         "restores_state": {"at_s": 100.0}})
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    guard = inv.make_guard(root)
    rows, _h, _p = inv.build_rows(root, guard, specs=(SPEC_A,))
    missing = [r for r in rows if r["run"] == f"{TRAIN_WORLD}__fixed9"][0]
    assert missing["present"] is False
    refused = inv.incomplete_runs(rows)
    assert any(r["run"] == f"{TRAIN_WORLD}__fixed9" and r["reason"] == "missing"
               for r in refused)
