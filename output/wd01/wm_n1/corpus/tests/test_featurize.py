"""featurize-all: split-before-open, selection, resume/skip-by-sha, receipt."""
import json

import numpy as np
import pytest

import synth
from corpus import _util as U
from corpus import featurize_all as FA


# --------------------------------------------------------------- sealing ---
def test_sealed_reason_chunk5_and_seed_block():
    assert FA.sealed_reason("chunk5", "w5", {"worlds": [], "seeds": []})
    assert FA.sealed_reason("chunk4", "w8710", {"worlds": [], "seeds": []})
    assert not FA.sealed_reason("chunk4", "w12", {"worlds": [], "seeds": []})
    seal = {"worlds": ["wSEAL"]}
    assert FA.sealed_reason("chunk4", "wSEAL", seal)


def test_select_runs_filters(tmp_path):
    runs = synth.default_runs()
    runs.append(dict(chunk="chunk4", world="w7", arm="native",
                     policy="native", split="TRAIN", t0_s=1.0, n_ticks=1,
                     n_dec=1))
    runs.append(dict(chunk="chunk4", world="HOLDOUTw", arm="native",
                     policy="native", split="TRAIN", t0_s=1.0, n_ticks=1,
                     n_dec=1))
    runs.append(dict(chunk="chunk4", world="wsmoke", arm="native",
                     policy="native", split="TRAIN", t0_s=1.0, n_ticks=1,
                     n_dec=1, run="wsmoke_smoke"))
    inv = synth.make_inventory(tmp_path / "inv.parquet", runs)
    import pandas as pd

    df = pd.read_parquet(inv)
    df.loc[df["run"] == "wsmoke_smoke", "is_smoke"] = True
    df.to_parquet(inv, index=False)

    kept = FA.select_runs(inv)
    words = {r["world"] for r in kept}
    assert "HOLDOUTw" not in words and "wsmoke" not in words
    assert {"w1", "w2", "c1", "w7"} <= words


def test_split_decided_before_run_open(tmp_path):
    data = tmp_path / "data"
    plan = synth.write_world_plan(
        data / "output/campaign_stage1_v3/results_plan_chunk4"
             "/WORLD_PLAN.json",
        [{"world": "w9", "split": "HOLDOUT", "t0_s": 540.0}])
    run = dict(batch="chunk4", chunk="chunk4", world="w9", arm="native",
               policy="native", split="TRAIN", t0_s=540.0,
               root_rel="runs/campaign_stage1_chunk4",
               run="w9__native")
    U.OpenedAudit.reset()
    U.OpenedAudit.start()
    try:
        with pytest.raises(Exception) as ei:
            FA._resolve_run(run, data, tmp_path / "out", force=False)
        assert "HOLDOUT" in str(ei.value)
    finally:
        U.OpenedAudit.stop()
    assert str(plan) in U.OpenedAudit.paths
    assert not any("physics_steps" in p for p in U.OpenedAudit.paths)


# ----------------------------------------------------- resume / skip sha ---
def _shard_dict(tmp_path, *, world="w1", arm="native"):
    k = 4
    inputs = tmp_path / "run" / "run" / "physics_steps.jsonl"
    inputs.parent.mkdir(parents=True, exist_ok=True)
    inputs.write_text('{"time_s": 0.5}\n', encoding="utf-8")
    z = np.zeros
    return dict(
        meta=dict(chunk="chunk4", world=world, arm=arm, split="TRAIN",
                  t0_s=540.0, n_ticks=1, n_slots=k, n_dropped_slots=0,
                  n_dropped_slot_steps=0, input_physics=str(inputs)),
        veh_step=z((1, k, 16), np.float16),
        veh_mask=z((1, k), np.uint8),
        slot_index=z((1, k), np.int32),
        plan=z((1, k, 19, 8), np.float16),
        decision_t=z((1,), np.float64),
        wall=z((1, 4, 11), np.float16),
        wall_mask=z((1, 4), np.uint8),
        counter=z((1, 2, 30), np.float16),
        accel_next=z((1, k), np.float16),
        target_mask=z((1, k), np.uint8),
        lane_next=z((1, k), np.int8),
        arrivals=z((1, 2), np.int16),
        slot_vid=[f"v{i}" for i in range(k)],
        outcomes=None,
    )


def test_writer_skip_by_sha(tmp_path):
    from features import writer as W

    shard = _shard_dict(tmp_path)
    first = W.write_shard(shard, tmp_path / "out")
    assert first["status"] == "written"
    second = W.write_shard(shard, tmp_path / "out")
    assert second["status"] == "skipped"
    assert second["record"]["input_sha256"] == first["record"]["input_sha256"]


# ------------------------------------------------- featurize_all driver ---
def test_featurize_all_bookkeeping(tmp_path, monkeypatch):
    from features import writer as W

    data = tmp_path / "data"
    synth.write_world_plan(
        data / "output/campaign_stage1_v3/results_plan_chunk4"
             "/WORLD_PLAN.json",
        [{"world": "w1", "split": "TRAIN", "t0_s": 540.0},
         {"world": "w2", "split": "TRAIN", "t0_s": 600.0}])
    runs = [
        dict(batch="chunk4", chunk="chunk4", world="w1", arm="native",
             policy="native", split="TRAIN", t0_s=540.0,
             root_rel="runs/campaign_stage1_chunk4", run="w1__native"),
        dict(batch="chunk4", chunk="chunk4", world="w2", arm="native",
             policy="native", split="TRAIN", t0_s=600.0,
             root_rel="runs/campaign_stage1_chunk4", run="w2__native"),
    ]
    inv = synth.make_inventory(tmp_path / "inv.parquet", runs)

    # a real valid shard record per run, produced by the frozen writer
    out = tmp_path / "out"
    out.mkdir()
    records = {}
    for r in runs:
        shard = _shard_dict(tmp_path, world=r["world"])
        records[U.run_key(r["chunk"], r["world"], r["arm"])] = \
            W.write_shard(shard, out / "store")["record"]

    state = {"n": 0}

    def fake_run_one(job):
        key = U.run_key(job["chunk"], job["world"], job["arm"])
        status = "written" if state["n"] == 0 else "skipped"
        state["n"] += 1
        return dict(key=key, status=status, record=records[key])

    monkeypatch.setattr(FA, "_run_one", fake_run_one)
    probe = lambda: {"free_virtual_gb": 99.0, "free_ram_gb": 99.0}  # noqa: E731

    rep1 = FA.featurize_all(inv, data, out, workers=1, probe=probe)
    assert rep1["runs"] == 2 and rep1["written"] == 1 and rep1["skipped"] == 1
    assert (out / FA.ALLOCATION_NAME).is_file()
    assert (out / FA.PLAN_MAP_NAME).is_file()
    assert (out / FA.OPENED_PATHS_NAME).is_file()

    state["n"] = 1
    rep2 = FA.featurize_all(inv, data, out, workers=1, probe=probe)
    assert rep2["skipped"] == 2 and rep2["written"] == 0
    receipt = json.loads((out / FA.ALLOCATION_NAME).read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["n_skipped"] == 2


def test_workers_bounds(tmp_path):
    inv = synth.make_inventory(tmp_path / "inv.parquet",
                               synth.default_runs())
    for bad in (0, 7):
        with pytest.raises(ValueError):
            FA.featurize_all(inv, tmp_path, tmp_path / "out", workers=bad)


# ------------------------------------------- replicated-arm shard keying ---
def _physics_row(t, vehicles, *, departed=(), cav_ids=()):
    return dict(time_s=float(t), live_ids=sorted(vehicles),
                departed_ids=list(departed), arrived_ids=[],
                colliding_ids=[], cav_ids=list(cav_ids),
                boundary_crossings=dict(mainline_entry=0, mainline_exit=0,
                                        ramp_entry=0),
                vehicles=dict(vehicles))


def _veh(pos, v, a=0.0):
    return dict(road_id="1191554636", lane_id="1191554636_0",
                lane_position_m=float(pos), speed_mps=float(v),
                acceleration_mps2=float(a), type_id="normal_car",
                speed_mode=31, lane_changed=False)


def _write_run_dir(root, rows, *, controller="damping"):
    (root / "run").mkdir(parents=True, exist_ok=True)
    with open(root / "run" / "physics_steps.jsonl", "w",
              encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (root / "run" / "manifest.json").write_text(json.dumps(
        dict(controller=controller, scenario=dict(cav_population=0))),
        encoding="utf-8")


def _replicated_arms_inventory(tmp_path, data):
    """One synthetic TRAIN world with the replicated chunk-1 arm shape:
    __base/__native/__fixed1/__fixed2/__switching1..3 (plan arms fixed and
    switching are replicated; only the run-dir suffix discriminates)."""
    synth.write_world_plan(
        data / "output/campaign_stage1_v3/results_plan_chunk4"
             "/WORLD_PLAN.json",
        [{"world": "wrep", "split": "TRAIN", "t0_s": 480.0}])
    base_rows = [_physics_row(t, {"mainline.0": _veh(10.0 + i, 30.0)},
                              departed=["mainline.0"] if i == 0 else ())
                 for i, t in enumerate((0.5, 1.0, 1.5))]
    branch_rows = [_physics_row(t, {"mainline.0": _veh(10.0 + i, 30.0)},
                                cav_ids=["mainline.0"])
                   for i, t in enumerate((480.5, 481.0, 481.5, 482.0))]
    runs = []
    for run_name, arm, policy in (
            ("wrep__base", "base", "damping"),
            ("wrep__native", "native", "native"),
            ("wrep__fixed1", "fixed", "fixed"),
            ("wrep__fixed2", "fixed", "fixed"),
            ("wrep__switching1", "switching", "switching"),
            ("wrep__switching2", "switching", "switching"),
            ("wrep__switching3", "switching", "switching")):
        root = data / "runs/campaign_stage1_chunk4" / run_name
        _write_run_dir(root, base_rows if arm == "base" else branch_rows)
        runs.append(dict(batch="chunk4", chunk="chunk4", world="wrep",
                         arm=arm, policy=policy, split="TRAIN", t0_s=480.0,
                         root_rel="runs/campaign_stage1_chunk4",
                         run=run_name))
    return synth.make_inventory(tmp_path / "inv_repl.parquet", runs)


REPL_ARMS = ("base", "native", "fixed1", "fixed2",
             "switching1", "switching2", "switching3")


def test_replicated_arms_get_distinct_shards_and_pack_items(
        tmp_path, monkeypatch):
    from corpus import pack as PK
    from corpus import verify as VF

    data = tmp_path / "data"
    inv = _replicated_arms_inventory(tmp_path, data)
    out = tmp_path / "out"
    probe = lambda: {"free_virtual_gb": 99.0, "free_ram_gb": 99.0}  # noqa: E731

    rep = FA.featurize_all(inv, data, out, workers=1, probe=probe)
    assert rep["written"] == 7 and rep["skipped"] == 0
    # 7 distinct shard dirs, keyed world/__<run-dir suffix>
    dirs = {out / "chunk4" / "wrep" / a for a in REPL_ARMS}
    assert all(d.is_dir() for d in dirs) and len(dirs) == 7
    # run keys + PLAN_MAP carry the suffix arm and the policy metadata
    pmap = U.read_json(out / FA.PLAN_MAP_NAME)["runs"]
    assert sorted(pmap) == sorted(f"chunk4/wrep__{a}" for a in REPL_ARMS)
    assert pmap["chunk4/wrep__fixed1"]["policy"] == "fixed"
    assert pmap["chunk4/wrep__fixed2"]["policy"] == "fixed"
    assert pmap["chunk4/wrep__switching3"]["policy"] == "switching"
    assert pmap["chunk4/wrep__native"]["policy"] == "native"
    # per-shard metadata: arm = suffix, policy = plan-level label
    meta = U.read_json(out / "chunk4" / "wrep" / "fixed2" / "meta.json")
    assert meta["arm"] == "fixed2" and meta["policy"] == "fixed"
    assert meta["batch"] == "chunk4" and meta["chunk"] == "chunk4"
    done = U.read_json(out / "chunk4" / "wrep" / "fixed1" / "DONE.json")
    assert done["arm"] == "fixed1" and done["policy"] == "fixed"
    idea = U.read_json(out / "chunk4" / "wrep" / "switching1" /
                       "IDEA_A_DONE.json")
    assert idea["arm"] == "switching1" and idea["policy"] == "switching"
    # resume: everything skips by sha
    rep2 = FA.featurize_all(inv, data, out, workers=1, probe=probe)
    assert rep2["written"] == 0 and rep2["skipped"] == 7

    # pack: distinct items per replicate, policy from metadata
    PK.pack_shards(out, tmp_path / "pack", workers=1)
    items = U.read_json(tmp_path / "pack" / PK.MANIFEST_NAME)["items"]
    keys = [(it["world"], it["arm"]) for it in items]
    assert len(set(keys)) == 7 and len(items) == 7
    pol = {it["arm"]: it["policy"] for it in items}
    assert pol["fixed1"] == pol["fixed2"] == "fixed"
    assert (pol["switching1"] == pol["switching2"]
            == pol["switching3"] == "switching")
    assert pol["native"] == "native"
    # one pair group per world/t0; the native item is the sole reference
    groups = {}
    for it in items:
        groups.setdefault(it["pair_key"], []).append(it["arm"])
    assert list(groups) == ["wrep@t0=480"]
    assert set(groups["wrep@t0=480"]) == set(REPL_ARMS)
    assert VF.verify_pack(tmp_path / "pack")["ok"] is True


def test_resolve_run_maps_plan_arm_to_run_dir_suffix(tmp_path):
    run = dict(batch="chunk1", chunk="1", world="w8410m2700p02t480",
               split="TRAIN", root_rel="runs/campaign_stage1_chunk1",
               t0_s=480.0, arm="fixed", policy="fixed", run="w8410__fixed2")
    synth.write_world_plan(
        tmp_path / "output/campaign_stage1_chunk1/results_plan_wm44"
             "/WORLD_PLAN.json",
        [{"world": "w8410m2700p02t480", "split": "TRAIN", "t0_s": 480.0}])
    job = FA._resolve_run(run, tmp_path, tmp_path / "out", force=False)
    assert job["arm"] == "fixed2"          # run-dir suffix, not the plan arm
    assert job["policy"] == "fixed"        # policy label stays metadata
    assert job["batch"] == "chunk1" and job["chunk"] == "chunk1"
    assert job["run_dir"].endswith("w8410__fixed2")
