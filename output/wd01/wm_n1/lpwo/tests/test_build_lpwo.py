"""LPWO guard, schedule, hashing, and ownership regressions on synthetic data."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

WM_N1 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WM_N1))
from lpwo import build_lpwo as B, guard as G, wall_timeline as T  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "WORKSPACE", tmp_path)
    spec = write(tmp_path / "protocols" / "holdout_exclusions.json",
                 {"schema": "holdout_exclusions/1", "excluded": [{"world": "blocked_world"}]})
    tool = tmp_path / "canonical.py"
    tool.write_text('def assert_final_holdout_eligible(worlds, data):\n'
                    '    assert data is not None\n'
                    '    assert not set(worlds) & {r["world"] for r in data["excluded"]}\n')
    guard = G.DataGuard(spec, tool, [tmp_path])
    return tmp_path, guard


def row(name="native", world="synthetic", policy="native"):
    return {"key": "chunk4/" + world + "__" + name, "run": world + "__" + name,
            "world": world, "arm": name, "policy": policy, "pack": "pack_v3",
            "batch": "chunk4", "chunk": "4", "split": "TRAIN", "t0_s": 100,
            "n_ticks": 250, "root_rel": "runs/campaign_stage1_chunk4"}


def schedule(root, r, action="native", malformed=False):
    return write(root / r["root_rel"] / r["run"] / "run" / "switch_schedule.json",
                 {"schedule": {"events": "bad" if malformed else [
                     {"time_s": 120, "vehicle_id": "c0", "action": action}]}})


def classify(setup, r):
    root, guard = setup
    return B.classify_window(r, {r["key"]: r}, root=root, guard=guard)


@pytest.mark.parametrize("policy", ["native", "damping", "fixed", "switching", "trajectory"])
def test_missing_schedule_fails_closed_for_every_label(setup, policy):
    assert classify(setup, row(policy=policy))["reason"] == "unverifiable_schedule_missing"


@pytest.mark.parametrize("action,reason", [("wall-v2", "schedule_wall_action"),
                                         ("pi-external-u", "schedule_pi_action"),
                                         ("unknown", "unverifiable_schedule_malformed")])
def test_hidden_schedule_ownership(setup, action, reason):
    root, _ = setup
    r = row()
    schedule(root, r, action)
    assert classify(setup, r)["reason"] == reason


def test_nested_per_cav_actions_are_scanned(setup):
    root, _ = setup
    r = row()
    p = schedule(root, r)
    write(p, {"schedule": {"events": [{"action": "native", "members": [
        {"controller": "wall"}]}]}})
    assert classify(setup, r)["wall"]


def test_valid_native_requires_hashed_schedule(setup):
    root, _ = setup
    r = row()
    path = schedule(root, r)
    verdict = classify(setup, r)
    assert verdict["wall"] is False
    assert verdict["schedule_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("source", ["CAL", "HOLDOUT", "sealed", "chunk-5", "J4B", "INC1", "blocked_world"])
def test_guard_refuses_before_open(setup, source):
    root, guard = setup
    with pytest.raises(G.GuardError):
        guard.read_json(root / source / "missing.json")


def test_guard_refuses_external_and_missing_split(setup):
    root, guard = setup
    with pytest.raises(G.GuardError, match="outside workspace"):
        guard.check_path(root.parent / "outside.json")
    with pytest.raises(G.GuardError, match="TRAIN"):
        guard.check_record({"world": "synthetic"})


def wall(root, r, *, assembled=103):
    directory = root / r["root_rel"] / r["run"]
    write(directory / "run" / "branch_record.json", {
        "root_id": r["world"], "branch_type": r["arm"], "outcome": "assembled_released",
        "t0_s": 100, "members": ["c0", "c1"], "request_s": 101,
        "assembled_at_s": assembled, "released_at_s": 125})
    (directory / "run" / "wall_events.jsonl").write_text(
        json.dumps({"kind": "wall_request", "timestamp_s": 101}) + "\n")
    return directory


@pytest.mark.parametrize("assembled,engaged", [(103, True), (None, False), (130, False)])
def test_timeline_requires_real_hold_times(setup, assembled, engaged):
    root, guard = setup
    directory = wall(root, row("wall", policy="wall"), assembled=assembled)
    tl = T.read_wall_timeline(directory, guard=guard)
    assert tl["engaged"] is engaged
    assert tl["owned"]
    assert T.phase_at(tl, 0) == "none"


def lpo(root, train, val, excluded=None):
    directory = root / "lpo"
    hashes = {}
    for name, data in (("windows_train_noPI.json", train), ("windows_val.json", val)):
        payload = {"windows": data, "n_windows": len(data), "counts": B._counts(data)}
        path = write(directory / name, payload)
        hashes[name] = B.sha256_file(path)
    path = write(directory / "pi_excluded_runs.json", {"keys": excluded or [], "n_runs": len(excluded or [])})
    hashes[path.name] = B.sha256_file(path)
    write(directory / "BUILD_RECEIPT.json", {"sha256": hashes})
    return directory


def test_build_exact_counts_no_wall_no_pi_and_validation(setup):
    root, guard = setup
    native, w, pi, missing, val = (row(), row("wall", policy="wall"), row("fixed"),
                                  row("damping"), row(world="validation"))
    schedule(root, native)
    schedule(root, pi, "pi")
    schedule(root, val)
    wall(root, w)
    source = lpo(root, [native, w, pi, missing], [val], ["chunk4/other__pi"])
    index = write(root / "INDEX.json", {"runs": [native, w, pi, missing, val]})
    receipt = B.build(source, root / "out", pack_v3_index=index, root=root, guard=guard)
    assert receipt["n_train"] == 1 and receipt["n_val"] == 1
    assert receipt["n_removed_windows"] == 3
    assert receipt["train_by_policy_after"] == {"native": 1}
    assert B.validate(root / "out", source, guard)["ok"]
    removed = json.loads((root / "out" / "wall_removed_windows.json").read_text())
    assert len(removed["windows"]) == 3
    assert removed["windows"][0]["timeline"]["members"] == ["c0", "c1"]
    with pytest.raises(ValueError, match="must be new"):
        B.build(source, root / "out", pack_v3_index=index, root=root, guard=guard)
    (root / "out" / "windows_train_noPI_noWall.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        B.validate(root / "out", source, guard)


def test_stale_source_fails_before_output(setup):
    root, guard = setup
    source = lpo(root, [], [])
    (source / "windows_val.json").write_text("{}")
    with pytest.raises(ValueError, match="stale/modified"):
        B.build(source, root / "out", pack_v3_index=root / "INDEX.json", root=root, guard=guard)
    assert not (root / "out").exists()
