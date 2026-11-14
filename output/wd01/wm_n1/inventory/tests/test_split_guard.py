"""Split guard: HOLDOUT and excluded worlds are refused before any open."""

import pytest

import inventory_lib as inv
from fixtures import ALL_SPECS, EXCLUDED_WORLD, build_fixture_root


def test_guard_refuses_holdout_before_open(tmp_path):
    root = build_fixture_root(tmp_path)
    guard = inv.make_guard(root)
    with pytest.raises(inv.SplitGuardError):
        guard.assert_world_readable("w1m2600p02t100", inv.HOLDOUT)
    assert guard.opened == []


def test_excluded_world_refused_even_if_plan_says_train(tmp_path):
    root = build_fixture_root(tmp_path)
    guard = inv.make_guard(root)
    with pytest.raises(inv.SplitGuardError):
        guard.assert_world_readable(EXCLUDED_WORLD, inv.TRAIN)
    assert guard.opened == []


def test_holdout_run_counted_by_name_and_never_opened(tmp_path, monkeypatch):
    root = build_fixture_root(tmp_path)
    guard = inv.make_guard(root)
    touched = []
    real_read_run = inv.read_run

    def spy(spec, run, world_meta, g, data_root):
        touched.append(world_meta["world"])
        return real_read_run(spec, run, world_meta, g, data_root)

    monkeypatch.setattr(inv, "read_run", spy)
    rows, holdout, planned = inv.build_rows(root, guard, specs=ALL_SPECS)

    assert EXCLUDED_WORLD not in touched
    assert all(EXCLUDED_WORLD not in p for p in guard.opened)
    assert any(h["world"] == EXCLUDED_WORLD for h in holdout)
    assert all(r["world"] != EXCLUDED_WORLD for r in rows)


def test_open_marker_never_read(tmp_path):
    root = build_fixture_root(tmp_path)
    guard = inv.make_guard(root)
    inv.build_rows(root, guard, specs=ALL_SPECS)
    assert not any("OPENED_MARKER" in p for p in guard.opened)
