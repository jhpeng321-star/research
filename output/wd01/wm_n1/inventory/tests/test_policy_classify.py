"""Policy classification from switch_schedule.json (incl. WM-72 relabel)."""
import inventory_lib as inv
from fixtures import ALL_SPECS, build_fixture_root


def _rows_by(root, batch, arm=None):
    guard = inv.make_guard(root)
    rows, _holdout, _planned = inv.build_rows(root, guard, specs=ALL_SPECS)
    out = [r for r in rows if r["batch"] == batch]
    if arm is not None:
        out = [r for r in out if r["arm"] == arm]
    return out


def test_chunk1_native_pi_are_wm72_switching(tmp_path):
    root = build_fixture_root(tmp_path)
    native = _rows_by(root, "chunk1", "native")[0]
    pi = _rows_by(root, "chunk1", "pi")[0]
    assert native["schedule_arm"] == "switching"
    assert native["policy"] == "switching"
    assert native["wm72_relabel"] is True
    assert pi["policy"] == "switching"
    assert pi["wm72_relabel"] is True


def test_truearms_native_is_true_native(tmp_path):
    root = build_fixture_root(tmp_path)
    native = _rows_by(root, "truearms_chunk2", "native")[0]
    assert native["schedule_arm"] == "native"
    assert native["policy"] == "native"
    assert native["wm72_relabel"] is False


def test_truearms_fleet_pi_is_pi(tmp_path):
    root = build_fixture_root(tmp_path)
    pi = _rows_by(root, "truearms_chunk2", "pi")[0]
    assert pi["schedule_arm"] == "fleet-pi"
    assert pi["policy"] == "pi"
    assert pi["wm72_relabel"] is False


def test_fixed_and_switching_are_percav(tmp_path):
    root = build_fixture_root(tmp_path)
    fixed = _rows_by(root, "chunk1", "fixed")[0]
    switching = _rows_by(root, "chunk1", "switching")[0]
    assert fixed["policy"] == "fixed" and fixed["is_fleet"] is False
    assert switching["policy"] == "switching" and switching["is_fleet"] is False


def test_base_is_fleet_damping(tmp_path):
    root = build_fixture_root(tmp_path)
    base = _rows_by(root, "chunk1", "base")[0]
    assert base["policy"] == "damping"
    assert base["is_fleet"] is True


def test_wall_arm_is_wall_policy(tmp_path):
    root = build_fixture_root(tmp_path)
    walls = _rows_by(root, "walls_pilot_chunk4")
    assert walls and all(w["policy"] == "wall" for w in walls)
    assert all(w["is_fleet"] is False for w in walls)


def test_classify_policy_direct():
    assert inv.classify_policy("native", "switching", None, False) == ("switching", False, True)
    assert inv.classify_policy("pi", "fleet-pi", None, False) == ("pi", True, False)
    assert inv.classify_policy("base", None, "damping", True) == ("damping", True, False)
