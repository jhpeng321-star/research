"""Opened-path audit and micro-sample bounds."""
import inventory_lib as inv
from fixtures import ALL_SPECS, build_fixture_root


def _build(root, guard):
    return inv.build_rows(root, guard, specs=ALL_SPECS)


def test_opened_paths_clean(tmp_path):
    root = build_fixture_root(tmp_path)
    guard = inv.make_guard(root)
    rows, holdout, _planned = _build(root, guard)
    tokens = set(guard.excluded) | {h["world"] for h in holdout}
    guard.assert_clean(tokens)
    assert guard.opened
    assert not any("dispatcher_rows" in p.lower() for p in guard.opened)
    assert not any("insertions" in p.lower() for p in guard.opened)


def test_opened_audit_detects_leak(tmp_path):
    root = build_fixture_root(tmp_path)
    guard = inv.make_guard(root)
    guard.opened.append(str(root / "runs" / "x" / "wLEAKm1p1t1__base" / "RUN_RECEIPT.json"))
    try:
        guard.assert_clean({"wLEAKm1p1t1"})
    except inv.SplitGuardError:
        return
    raise AssertionError("audit failed to detect a sealed token in an opened path")


def test_micro_sample_bounded(tmp_path):
    root = build_fixture_root(tmp_path)
    guard = inv.make_guard(root)
    rows, _h, _p = _build(root, guard)
    before = len(guard.opened)
    micro = inv.micro_sample(rows, root, guard, n_runs=3, max_lines=10)
    assert len(micro["runs"]) <= 3
    assert micro["lines_read"] <= 30
    assert micro["vehicles_per_tick_max"] >= 1
    assert micro["zstd_ratio"] > 0
    assert micro["ticks_per_second"] > 0
    assert all("physics_steps.jsonl" in p for p in guard.opened[before:])
