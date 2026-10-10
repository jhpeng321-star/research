import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(PKG))

from wms_eval import c1sets  # noqa: E402


def _state(policy, arm, world, index=0, pack="pack_v3"):
    return {"policy": policy, "arm": arm, "world": world, "pack": pack,
            "chunk": "chunk1", "t0_s": 100.0 + 10.0 * index,
            "key": f"chunk1/{world}__{arm}", "run": f"{world}__{arm}",
            "initial_state": {"time_s": 0.0, "vehicles": []}}


def _pi_set():
    return [_state("pi", "pi", f"p{i}") for i in range(4)]


def test_unseen_split_drops_train_and_val_and_splits_by_world():
    pi = [_state("pi", "pi", f"p{i}") for i in range(4)]
    # p3's PI record is the run named by the validation metadata -> dropped
    pi[3] = {**pi[3], "key": "chunk1/p3__base", "run": "p3__base"}
    train = [_state("damping", "base", "p0"), _state("native", "native", "p1")]
    val = [_state("damping", "base", "p3")]
    # p2 has no train window -> U-new; p0/p1 -> U-same; p3 leaked val -> dropped
    split = c1sets.unseen_split(pi, train, val)
    assert {r["world"] for r in split["u_same"]} == {"p0", "p1"}
    assert {r["world"] for r in split["u_new"]} == {"p2"}
    assert [r["world"] for r in split["dropped"]] == ["p3"]
    assert split["u_same_worlds"] == {"p0", "p1"}
    assert split["unseen_worlds"] == {"p0", "p1", "p2"}


def test_seen_selection_is_disjoint_from_u_same_and_deterministic():
    val = ([_state("damping", "base", "v0"), _state("native", "native", "v0", 1)]
           + [_state("damping", "base", "v1"), _state("native", "native", "v1", 1)]
           + [_state("wall", "w-mid-early", "u0"), _state("native", "native", "u0", 1)])
    split = {"u_same_worlds": {"t1"}, "u_new_worlds": {"u0"}}
    train = [_state("damping", "base", "t1")]
    selection = c1sets.seen_selection(val, split, train, target=10, seed=0)
    worlds = {w["world"] for w in selection["seen"]}
    assert worlds == {"v0", "v1", "u0"}          # u0 is the paired U-new reference
    assert "t1" not in worlds
    assert selection["seen"] == c1sets.seen_selection(
        val, split, train, target=10, seed=0)["seen"]


def test_seen_selection_excludes_training_windows_and_pi():
    val = [_state("damping", "base", "v0"), _state("native", "native", "v1"),
           _state("pi", "pi", "v2")]
    train = [_state("damping", "base", "v0")]      # v0 is a training window -> excluded
    split = {"u_same_worlds": set(), "u_new_worlds": set()}
    seen = c1sets.seen_selection(val, split, train, target=10)["seen"]
    assert [w["world"] for w in seen] == ["v1"]


def test_verify_fails_closed_on_world_overlap_and_train_leak():
    u_same = [_state("pi", "pi", "a")]
    u_new = [_state("pi", "pi", "b")]
    seen_clean = [_state("damping", "base", "c")]
    report = c1sets.verify(u_same, u_new, seen_clean, [])
    assert report["overlap_ok"] is True and report["refused"] is False
    with pytest.raises(c1sets.C1SetsError):
        c1sets.verify(u_same, u_new, [_state("damping", "base", "a")], [])
    train = [_state("damping", "base", "t")]
    leaked = {**u_new[0], "key": "chunk1/t__base", "run": "t__base"}
    with pytest.raises(c1sets.C1SetsError) as excinfo:
        c1sets.verify(u_same, [leaked], seen_clean, train)
    assert "train run overlap=1" in str(excinfo.value)


def test_verify_reports_paired_u_new_overlap_without_refusing():
    u_same = [_state("pi", "pi", "a")]
    u_new = [_state("pi", "pi", "b")]
    seen = [_state("damping", "base", "b")]        # paired same-world reference
    report = c1sets.verify(u_same, u_new, seen, [])
    assert report["seen_u_new_world_overlap"] == ["b"]
    assert report["seen_u_same_world_overlap"] == []


def test_verify_files_streams_and_matches_verify(tmp_path):
    def _write(path, records):
        path.write_text(json.dumps({"n_states": len(records), "windows": records},
                                   separators=(",", ":")), encoding="utf-8")

    us_path, un_path, seen_path = tmp_path / "us.json", tmp_path / "un.json", tmp_path / "se.json"
    _write(us_path, [_state("pi", "pi", "a")])
    _write(un_path, [_state("pi", "pi", "b")])
    _write(seen_path, [_state("damping", "base", "b")])
    report = c1sets.verify_files(us_path, un_path, seen_path, [])
    assert report["overlap_ok"] is True
    assert report["seen_u_new_world_overlap"] == ["b"]
    assert report["seen_u_same_world_overlap"] == []
    assert report["seen_states"] == 1


def test_scan_states_counts_records_and_hashes(tmp_path):
    path = tmp_path / "set.json"
    records = [{"world": "a", "run": "a__pi", "key": "c/a__pi",
                "initial_state": {"time_s": 0.0, "vehicles": []}},
               {"world": "b", "run": "b__pi", "key": "c/b__pi",
                "initial_state": {"time_s": 0.0, "vehicles": []},
                "world_meta": {"world": "b", "world_id": 1, "demand": 2700,
                               "penetration_pct": 2, "t0_s": 1.0}}]
    payload = json.dumps({"schema": "wms_eval_states/1", "n_states": 2, "windows": records},
                         separators=(",", ":"))
    path.write_text(payload, encoding="utf-8")
    info = c1sets.scan_states(path)
    # a nested world_meta "world" must not count as a second state record
    assert info["n_states"] == 2
    assert info["n_worlds"] == 2
    assert info["worlds"] == ["a", "b"]
    assert info["sha256"] == hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_build_manifest_includes_sha256_counts_and_worlds(tmp_path):
    path = tmp_path / "u.json"
    path.write_text('{"windows":[{"world":"a","run":"a__pi","key":"c/a__pi",'
                    '"initial_state":{"time_s":0.0,"vehicles":[]}}]}',
                    encoding="utf-8")
    source = tmp_path / "src.json"
    source.write_text("{}", encoding="utf-8")
    manifest = c1sets.build_manifest({"u": path}, source_paths={"src": source})
    assert manifest["sets"]["u"]["n_states"] == 1
    assert manifest["sets"]["u"]["worlds"] == ["a"]
    assert len(manifest["sets"]["u"]["sha256"]) == 64
    assert manifest["source_sha256"]["src"] == hashlib.sha256(b"{}").hexdigest()
