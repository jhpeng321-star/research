"""Synthetic pack fixtures only; no real data, model, training or SUMO."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

WM_N1 = Path(__file__).resolve().parents[2]
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))

from lpwo import materialize_wall as M  # noqa: E402


class FakeGuard:
    def __init__(self):
        self.paths, self.records, self.reads = [], [], []

    def check_path(self, path):
        self.paths.append(Path(path))

    def check_record(self, record):
        self.records.append(dict(record))

    def read_json(self, path):
        self.check_path(path)
        self.reads.append(Path(path))
        return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def timeline(world="synthetic", arm="wall_arm"):
    return {"status": "assembled_released", "owned": True, "engaged": True,
            "world": world, "arm": arm, "members": ["c0", "c1"],
            "t0_s": 100.0, "request_s": 101.0, "assembled_at_s": 103.0,
            "owner_end_s": 120.0, "released_at_s": 120.0}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Test fixtures stay local when invoked with the documented --basetemp.
    monkeypatch.setattr(M, "WORKSPACE", tmp_path)
    root = tmp_path / "pack"
    ids_root = tmp_path / "identity"
    root.mkdir()
    ids_root.mkdir()
    n, k = 250, 5
    kin = np.zeros((n * 2, k, 7), np.float32)
    for start in (0, n):
        for j in range(n):
            kin[start + j, :, 0] = np.arange(k) * 20 - 1000 + j * 2
        kin[start:start + n, :, 1] = 1
        kin[start:start + n, :, 4] = 4
        kin[start:start + n, :, 6] = 5
    step = np.zeros((n * 2, k, 16), np.float32)
    step[:, :2, 7] = 1
    arrays = {"veh_kin": kin, "veh_step": step,
              "veh_mask": np.ones((n * 2, k), np.uint8),
              "target_mask": np.ones((n * 2, k), np.uint8),
              "counter": np.ones((n * 2, 2, 30), np.float32)}
    specs = {}
    for name, data in arrays.items():
        np.save(root / (name + ".npy"), data)
        specs[name] = {"file": name + ".npy"}
    runs, items = [], []
    for arm, start in (("wall_arm", 0), ("native", n)):
        runs.append({"key": "4/synthetic__" + arm, "run": "synthetic__" + arm,
                     "world": "synthetic", "arm": arm, "chunk": "4",
                     "split": "TRAIN", "t0_s": 100.0, "tick_start": start,
                     "tick_end": start + n, "policy": "wall" if start == 0 else "native"})
        items.append({"world": "synthetic", "arm": arm, "chunk": "4",
                      "t0_s": 100.0, "tick": start})
    write_json(root / "INDEX.json", {"runs": runs})
    write_json(root / "manifest.json", {"k_max": k, "arrays": specs, "items": items})
    np.save(ids_root / "identities.npy", np.tile(np.arange(k), (2, 1)))
    np.save(ids_root / "global_positions.npy", np.array([0, 1]))
    write_json(ids_root / "vehicle_map.json", [[0, 0, "c0"], [0, 0, "c1"]]
               + [[0, 0, "h" + str(i)] for i in range(2, k)])
    guard = FakeGuard()
    packs = {"p": M.GuardedPack("p", root, guard)}
    ids = {"p": M.guarded_identity("p", ids_root, guard)}
    window = {**runs[0], "pack": "p", "origin_list": "train", "timeline": timeline()}
    control = {**runs[1], "pack": "p"}
    return {"tmp": tmp_path, "root": root, "ids_root": ids_root, "guard": guard,
            "packs": packs, "ids": ids, "window": window, "control": control,
            "arrays": arrays, "runs": runs}


def state(f, **kwargs):
    return M.materialize_state(f["window"], f["packs"], f["ids"], f["guard"],
                               timeline=timeline(), **kwargs)


def change_array(f, name, change):
    data = f["arrays"][name].copy()
    change(data)
    # Close mmap handles before replacing the synthetic fixture on Windows.
    mm = f["packs"]["p"]._mm.pop(name, None)
    if mm is not None:
        mm._mmap.close()
    np.save(f["root"] / (name + ".npy"), data)


def test_real_history_shift_horizon_counter_timeline_and_native_pair(fixture):
    f = fixture
    r = state(f)
    assert r["t0_s"] == 104.5
    assert r["t0_shift_steps"] == 9
    assert r["horizon_s"] == 120
    assert r["history"]["history_steps"] == 10
    assert r["history"]["kin"][0][0][0] == -1000
    assert r["history"]["kin"][-1][0][0] == -982
    assert r["counter"] == np.ones((2, 30)).tolist()
    assert r["wall_members"] == ["c0", "c1"]
    assert r["wall_timeline"]["request_s"] == 101
    assert r["wall_phase_at_t0"] == "hold"
    assert r["paired"] is True
    assert r["native_pair_status"] == "matched_initial_physical_state"
    assert r["unseen_controller"] is True
    assert r["world_seen_in_training"] is True
    assert len(r["recorded"]["wall"]) == 241
    assert r["recorded"]["wall"][-1]["time_s"] == 224.5
    assert all(isinstance(a, np.memmap) for a in f["packs"]["p"]._mm.values())
    assert isinstance(f["ids"]["p"].ids, np.memmap)


@pytest.mark.parametrize("column", [0, 1, 4, 5, 6])
def test_pair_checks_full_physics_including_acceleration_length(fixture, column):
    change_array(fixture, "veh_kin", lambda a: a.__setitem__((259, 0, column), 999))
    r = state(fixture)
    assert r["paired"] is False
    assert r["native_pair_status"] == "initial_physical_state_mismatch"
    assert r["arms"] == ["wall"]
    assert "native" not in r["recorded"]


def test_counter_difference_prevents_pairing(fixture):
    change_array(fixture, "counter", lambda a: a.__setitem__((259, 0, 0), 2))
    assert state(fixture)["native_pair_status"] == "initial_physical_state_mismatch"


def test_pair_matches_by_identity_not_slot_order(fixture):
    f = fixture
    for name in ("veh_kin", "veh_step", "veh_mask", "target_mask"):
        change_array(f, name, lambda a: a.__setitem__(slice(250, None), a[250:, ::-1].copy()))
    np.asarray(f["ids"]["p"].ids)  # Sidecar is read-only; replace fixture after close.
    f["ids"]["p"].ids._mmap.close()
    np.save(f["ids_root"] / "identities.npy", np.array([np.arange(5), np.arange(5)[::-1]]))
    f["ids"]["p"] = M.guarded_identity("p", f["ids_root"], f["guard"])
    assert state(f)["paired"] is True


@pytest.mark.parametrize("name,change,match", [
    ("veh_mask", lambda a: a.__setitem__((0, 0), 0), "continuous history"),
    ("target_mask", lambda a: a.__setitem__((8, 0), 0), "continuous history"),
    ("counter", lambda a: a.__setitem__((9, 0, 0), float("nan")), "entrance counter"),
    ("counter", lambda a: a.__setitem__((9, 0, 0), -1), "entrance counter"),
    ("veh_kin", lambda a: a.__setitem__((100, 0, 4), float("nan")), "nonfinite future"),
])
def test_incomplete_or_invalid_physical_inputs_fail_closed(fixture, name, change, match):
    change_array(fixture, name, change)
    with pytest.raises(M.MaterializeError, match=match):
        state(fixture)


def test_missing_counter_and_short_horizon(fixture):
    f = fixture
    f["packs"]["p"].arrays.pop("counter")
    with pytest.raises(M.MaterializeError, match="counter"):
        state(f)
    f["packs"]["p"].runs[f["runs"][0]["key"]]["tick_end"] = 249
    with pytest.raises(M.MaterializeError, match="120s"):
        state(f)


def test_missing_native_and_native_horizon_status_explicit(fixture):
    f = fixture
    native = f["packs"]["p"].runs.pop(f["runs"][1]["key"])
    assert state(f)["native_pair_status"] == "missing_native"
    native["tick_end"] = 499
    f["packs"]["p"].runs[native["key"]] = native
    r = state(f)
    assert r["native_pair_status"] == "invalid_native_history_counter_or_horizon"
    assert "native" not in r["recorded"]


@pytest.mark.parametrize("update", [
    {"engaged": False, "status": "not_feasible"},
    {"engaged": False, "status": "aborted_in_assembly"},
    {"assembled_at_s": None}, {"owner_end_s": 102}, {"members": ["h2"]},
    {"world": "wrong"}, {"t0_s": 99},
])
def test_timeline_incomplete_or_unengaged_fails(fixture, update):
    f = fixture
    with pytest.raises(M.MaterializeError):
        M.materialize_state(f["window"], f["packs"], f["ids"], f["guard"],
                            timeline={**timeline(), **update}, source="wall_pilot")


def args_for(f, **extra):
    write_json(f["tmp"] / "removed.json", {"windows": [f["window"]]})
    write_json(f["tmp"] / "train.json", {"windows": [f["control"]]})
    values = dict(wall_removed_windows=str(f["tmp"] / "removed.json"),
                  retained_train=str(f["tmp"] / "train.json"), retained_split=None,
                  pack=["p=" + str(f["root"])], identities=["p=" + str(f["ids_root"])],
                  pilot_index=None, pilot_pack="p", pilot_run_dir=[],
                  out_dir=str(f["tmp"] / "result"), min_vehicles=5)
    values.update(extra)
    return argparse_namespace(values)


def argparse_namespace(values):
    from argparse import Namespace
    return Namespace(**values)


def pilot_args(f, status="assembled_released", **row_update):
    row = {**f["window"], "root_rel": "runs/" + M.PILOT_CAMPAIGN, **row_update}
    run_dir = f["tmp"] / "runs" / M.PILOT_CAMPAIGN / str(row["run"])
    run_dir.mkdir(parents=True)
    index = f["tmp"] / "pilot_index.json"
    write_json(index, {"runs": [row]})
    args = args_for(f, pilot_index=str(index),
                    pilot_run_dir=[row["key"] + "=" + str(run_dir)])
    calls = []
    def reader(path, guard=None):
        calls.append(Path(path))
        assert guard is f["guard"]
        return {**timeline(), "status": status, "engaged": status in M.ENGAGED}
    return args, calls, reader


def test_stream_counts_controls_receipt_and_no_overwrite(fixture):
    f = fixture
    args = args_for(f)
    # One rejected window and one valid, without large states in memory.
    bad = {**f["window"], "arm": "unknown"}
    write_json(f["tmp"] / "removed.json", {"windows": [bad, f["window"]]})
    receipt = M.run(args, guard=f["guard"])
    same = json.loads((f["tmp"] / "result" / "wall_same.json").read_text())
    controls = json.loads((f["tmp"] / "result" / "seen_controls.json").read_text())
    assert same["n_states"] == len(same["windows"]) == 1
    assert receipt["sets"]["wall_same"]["n_requested"] == 2
    assert receipt["sets"]["wall_same"]["n_rejected"] == 1
    assert receipt["status"] == "complete_with_rejections"
    assert controls["n_states"] == 1
    assert controls["windows"][0]["controller_seen_in_training"] is True
    assert controls["windows"][0]["unseen_controller"] is False
    assert json.loads((f["tmp"] / "result" / "BUILD_RECEIPT.json").read_text()) == receipt
    previous = (f["tmp"] / "result" / "wall_same.json").read_bytes()
    with pytest.raises(FileExistsError):
        M.run(args, guard=f["guard"])
    assert (f["tmp"] / "result" / "wall_same.json").read_bytes() == previous


def test_removed_val_origin_and_new_world_are_not_wall_same(fixture):
    f = fixture
    args = args_for(f)
    write_json(f["tmp"] / "removed.json", {"windows": [
        {**f["window"], "origin_list": "val"},
        {**f["window"], "world": "not-in-training"}]})
    receipt = M.run(args, guard=f["guard"])
    assert receipt["sets"]["wall_same"]["n_states"] == 0
    assert receipt["selection_counts"] == {"removed_non_train_origin": 1,
                                            "removed_world_not_retained_in_training": 1}


@pytest.mark.parametrize("status", ["not_feasible", "aborted_in_assembly", "unverifiable"])
def test_pilot_refused_and_unassembled_excluded(fixture, status):
    args, calls, reader = pilot_args(fixture, status)
    receipt = M.run(args, guard=fixture["guard"], timeline_reader=reader)
    assert len(calls) == 1
    assert receipt["sets"]["wall_pilot"]["n_states"] == 0
    assert receipt["selection_counts"]["pilot_excluded:" + status] == 1


def test_explicit_engaged_pilot_and_controls(fixture):
    args, calls, reader = pilot_args(fixture)
    receipt = M.run(args, guard=fixture["guard"], timeline_reader=reader)
    assert len(calls) == 1
    assert receipt["sets"]["wall_pilot"]["n_states"] == 1
    assert receipt["sets"]["seen_controls"]["n_states"] == 1


@pytest.mark.parametrize("update", [
    {"split": "CAL"}, {"split": "HOLDOUT"}, {"split": None},
    {"chunk": "5"}, {"root_rel": "runs/campaign_walls_pilot_chunk5"},
    {"run": "sealed-run"}, {"batch": "J4B"}, {"batch": "INC1"},
    {"root_rel": "runs/other_campaign"},
])
def test_pilot_hard_denial_precedes_any_run_read(fixture, update):
    args, calls, reader = pilot_args(fixture, **update)
    with pytest.raises(M.WallMaterializeError):
        M.run(args, guard=fixture["guard"], timeline_reader=reader)
    assert calls == []
    assert not any(p.name in ("branch_record.json", "wall_events.jsonl")
                   for p in fixture["guard"].paths)


def test_guard_denial_prevents_actual_array_open(fixture, monkeypatch):
    f = fixture
    def deny(path):
        if Path(path).name == "veh_kin.npy":
            raise RuntimeError("denied")
    monkeypatch.setattr(f["guard"], "check_path", deny)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("unguarded array open"))
    with pytest.raises(RuntimeError, match="denied"):
        state(f)


def test_local_paths_deny_outside_and_unsafe_array_spec(fixture):
    f = fixture
    with pytest.raises(M.WallMaterializeError, match="outside workspace"):
        M.local_path(f["tmp"].parent / "outside")
    for path in ("CAL/a.json", "chunk-5/a", "J4B/a", "INC1/a", "sealed/a"):
        with pytest.raises(M.WallMaterializeError, match="forbidden"):
            M.local_path(f["tmp"] / path)
    manifest = json.loads((f["root"] / "manifest.json").read_text())
    manifest["arrays"]["counter"]["file"] = "../outside.npy"
    write_json(f["root"] / "manifest.json", manifest)
    with pytest.raises(M.WallMaterializeError, match="relative to pack"):
        M.GuardedPack("p", f["root"], f["guard"])


def test_cli_requires_all_guard_contract_arguments():
    with pytest.raises(SystemExit):
        M.parser().parse_args([])
    required = {a.dest for a in M.parser()._actions if a.required}
    assert {"guard_spec", "guard_tool", "allowed_root"} <= required


@pytest.mark.parametrize("status", ["aborted_in_assembly", "not_feasible"])
def test_same_world_preserves_verified_nonengaged_status_without_counting_as_pilot(fixture, status):
    f = fixture
    tl = {**timeline(), "status": status, "engaged": False, "assembled_at_s": None}
    if status == "not_feasible":
        tl.update(owned=False, members=None, request_s=None, owner_end_s=None,
                  released_at_s=None)
    r = M.materialize_state(f["window"], f["packs"], f["ids"], f["guard"], timeline=tl)
    assert r["wall_exposure_status"] == status
    assert r["wall_engaged"] is False
    assert r["wall_owned"] is (status != "not_feasible")
    assert r["wall_members"] == ([] if status == "not_feasible" else ["c0", "c1"])


def test_corresponding_controls_only_for_successfully_materialized_targets(fixture):
    f = fixture
    args = args_for(f)
    f["window"]["timeline"]["assembled_at_s"] = None
    write_json(f["tmp"] / "removed.json", {"windows": [f["window"]]})
    receipt = M.run(args, guard=f["guard"])
    assert receipt["sets"]["wall_same"]["n_states"] == 0
    assert receipt["sets"]["wall_same"]["n_rejected"] == 1
    assert receipt["sets"]["seen_controls"]["n_states"] == 0


def test_pilot_new_world_explicitly_marked_and_not_wall_same(fixture):
    f = fixture
    args, calls, reader = pilot_args(f)
    write_json(f["tmp"] / "train.json", {"windows": [{**f["control"], "world": "other"}]})
    receipt = M.run(args, guard=f["guard"], timeline_reader=reader)
    assert calls
    assert receipt["sets"]["wall_same"]["n_states"] == 0
    assert receipt["sets"]["wall_pilot"]["n_states"] == 1
    r = json.loads((f["tmp"] / "result" / "wall_pilot.json").read_text())["windows"][0]
    assert r["world_seen_in_training"] is False
    assert r["unseen_controller"] is True


def test_run_without_guard_injection_constructs_main_agents_guard_contract(fixture, monkeypatch):
    import types
    f = fixture
    args = args_for(f)
    args.guard_spec = str(f["tmp"] / "spec.json")
    args.guard_tool = str(f["tmp"] / "tool.py")
    args.allowed_root = [str(f["tmp"])]
    calls = []
    def factory(spec, tool, allowed_roots):
        calls.append((spec, tool, allowed_roots))
        return f["guard"]
    monkeypatch.setitem(sys.modules, "lpwo.guard", types.SimpleNamespace(DataGuard=factory))
    assert M.run(args)["status"] == "complete"
    assert calls == [(Path(args.guard_spec), Path(args.guard_tool), [f["tmp"]])]


def test_stream_consumes_one_record_at_a_time_and_counts_zero_successes(fixture, monkeypatch):
    import io
    f = fixture
    events = []
    def windows():
        for i in range(3):
            assert len(events) == i, "writer eagerly collected the iterator"
            yield {**f["window"], "key": str(i)}, timeline()
    def build(window, *args, **kwargs):
        events.append(window["key"])
        raise M.MaterializeError("synthetic rejection")
    monkeypatch.setattr(M, "materialize_state", build)
    errors = io.StringIO()
    out = f["tmp"] / "stream.json"
    receipt = M.write_stream(out, "wall_same", windows(), {}, {}, f["guard"], errors,
                             train_worlds={"synthetic"})
    assert events == ["0", "1", "2"]
    assert receipt["n_requested"] == receipt["n_rejected"] == 3
    assert receipt["n_states"] == 0
    assert json.loads(out.read_text())["n_states"] == 0
    assert len(errors.getvalue().splitlines()) == 3


def test_failure_keeps_start_and_stopped_receipts(fixture, monkeypatch):
    f = fixture
    args = args_for(f)
    def fail(*args, **kwargs):
        raise OSError("synthetic pack failure")
    monkeypatch.setattr(M, "GuardedPack", fail)
    with pytest.raises(OSError, match="synthetic pack failure"):
        M.run(args, guard=f["guard"])
    started = json.loads((f["tmp"] / "result" / "MATERIALIZE_STARTED.json").read_text())
    stopped = json.loads((f["tmp"] / "result" / "BUILD_RECEIPT.json").read_text())
    assert started["status"] == "in_progress"
    assert stopped["status"] == "failed"
    assert "synthetic pack failure" in stopped["error"]


def test_denied_manifest_item_is_checked_before_array_access(fixture, monkeypatch):
    f = fixture
    item = f["packs"]["p"]._items_by_run[("4", "synthetic", "wall_arm", 100.0)][0][1]
    item["split"] = "CAL"
    monkeypatch.setattr(f["packs"]["p"], "array", lambda *a: pytest.fail("unsafe array access"))
    with pytest.raises(M.WallMaterializeError, match="forbidden"):
        state(f)


def test_mixed_pilot_requests_all_preflight_before_first_run_read(fixture):
    f = fixture
    args, calls, reader = pilot_args(f)
    index = json.loads(Path(args.pilot_index).read_text())
    denied = {**index["runs"][0], "key": "4/other__wall", "run": "other__wall", "split": "CAL"}
    index["runs"].append(denied)
    write_json(Path(args.pilot_index), index)
    args.pilot_run_dir.append(denied["key"] + "=" + str(
        f["tmp"] / "runs" / M.PILOT_CAMPAIGN / denied["run"]))
    with pytest.raises(M.WallMaterializeError, match="forbidden"):
        M.run(args, guard=f["guard"], timeline_reader=reader)
    assert calls == []


def test_native_unsafe_metadata_never_reaches_native_arrays(fixture):
    f = fixture
    f["packs"]["p"].runs[f["runs"][1]["key"]]["split"] = "HOLDOUT"
    with pytest.raises(M.WallMaterializeError, match="forbidden"):
        state(f)


@pytest.mark.parametrize("removed", [[], "partial"])
def test_same_world_explicit_pilot_supplements_or_completes_removed_metadata(fixture, removed):
    f = fixture
    args, calls, reader = pilot_args(f)
    windows = [] if removed == [] else [{**f["window"], "timeline": {"status": "partial"}}]
    write_json(f["tmp"] / "removed.json", {"windows": windows})
    receipt = M.run(args, guard=f["guard"], timeline_reader=reader)
    assert calls
    assert receipt["sets"]["wall_same"]["n_states"] == 1
    assert receipt["sets"]["wall_same"]["n_rejected"] == 0
    assert receipt["sets"]["wall_pilot"]["n_states"] == 1
    assert receipt["sets"]["seen_controls"]["n_states"] == 1
