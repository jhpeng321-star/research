import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(PKG))

from wms_eval import seen  # noqa: E402


def _window(world, arm, policy, t0, split="TRAIN", key=None):
    return {"world": world, "arm": arm, "policy": policy, "t0_s": t0,
            "split": split, "key": key or f"chunk/{world}__{arm}",
            "run": f"{world}__{arm}", "pack": "pack_v3"}


def _windows():
    rows = []
    for index in range(6):
        world = f"w{index:04d}"
        rows.append(_window(world, "base", "damping", 100.0))
        rows.append(_window(world, "native", "native", 120.0))
        rows.append(_window(world, "wall1", "wall", 140.0))
        rows.append(_window(f"p{index:04d}", "fixed1", "fixed", 100.0))
        rows.append(_window(f"q{index:04d}", "pi", "pi", 100.0))
    return rows


def test_select_seen_windows_filters_and_spreads_over_worlds():
    selected = seen.select_seen_windows(_windows(), target_size=6, seed=3)
    assert len(selected) == 6
    assert len({w["world"] for w in selected}) == 6
    assert {w["policy"] for w in selected} <= {"native", "damping", "wall"}
    assert all(w["split"] == "TRAIN" for w in selected)


def test_select_seen_windows_is_deterministic_and_respects_exclusions():
    first = seen.select_seen_windows(_windows(), target_size=9, seed=1)
    second = seen.select_seen_windows(_windows(), target_size=9, seed=1)
    assert [w["key"] for w in first] == [w["key"] for w in second]
    excluded = seen.select_seen_windows(_windows(), target_size=9, seed=1,
                                        exclude_worlds={"w0000"})
    assert all(w["world"] != "w0000" for w in excluded)


def test_select_seen_windows_returns_secondary_windows_after_full_coverage():
    selected = seen.select_seen_windows(_windows(), target_size=12, seed=0)
    assert len(selected) == 12
    assert len({w["world"] for w in selected}) == 6


def test_build_metadata_only_and_materialised(tmp_path):
    windows_path = tmp_path / "windows_val.json"
    windows_path.write_text(json.dumps({"windows": _windows()}), encoding="utf-8")

    meta = seen.build(windows_path=windows_path, out_path=tmp_path / "meta.json",
                      target_size=6, seed=0, metadata_only=True)
    assert meta["materialised"] is False
    assert meta["n_windows"] == 6

    index = {}
    for window in _windows():
        index[window["key"]] = {
            "initial_state": {"time_s": window["t0_s"],
                              "vehicles": [{"id": "h", "s": 0.0, "lane": 0, "v": 10.0,
                                            "present": True, "is_cav": False}]},
            "recorded": {"native": []},
        }
    materialised = seen.build(windows_path=windows_path, out_path=tmp_path / "states.json",
                              target_size=6, seed=0, state_index=index)
    assert materialised["materialised"] is True
    payload = json.loads((tmp_path / "states.json").read_text(encoding="utf-8"))
    assert len(payload["windows"]) == 6
    assert all("initial_state" in record for record in payload["windows"])


def test_build_fails_closed_on_incomplete_index(tmp_path):
    windows_path = tmp_path / "windows_val.json"
    windows_path.write_text(json.dumps({"windows": _windows()}), encoding="utf-8")
    try:
        seen.build(windows_path=windows_path, out_path=tmp_path / "states.json",
                   target_size=6, seed=0, state_index={})
    except seen.SeenError as exc:
        assert "no state material" in str(exc)
    else:
        raise AssertionError("expected SeenError for an empty index")
