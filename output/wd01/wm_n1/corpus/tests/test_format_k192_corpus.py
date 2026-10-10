"""A5 K=192 end-to-end: featurize-all(format=v3) -> pack_v3 -> verify.

Synthetic 192-concurrent-vehicle runs only (zero SUMO, nothing outside tmp,
no real data file opened).  Proves the v3 format flows through the corpus
driver, the pack manifest self-describes (k_max 192 / format v3), and
``corpus verify`` enforces the K contract.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import synth
from corpus import _util as U
from corpus import featurize_all as FA
from corpus import pack as PK
from corpus import verify as VF

MAIN_LEFT = "1191554636_1"
MAIN_RIGHT = "1191554636_0"
ACCEL = "11553048#1_0"
N_VEH = 192


def _veh(lane_id, pos, v, a=0.0):
    return dict(road_id=lane_id.rsplit("_", 1)[0], lane_id=lane_id,
                lane_position_m=float(pos), speed_mps=float(v),
                acceleration_mps2=float(a), type_id="normal_car",
                speed_mode=31, lane_changed=False)


def _row(t, vehicles, departed=(), cav_ids=()):
    return dict(time_s=float(t), live_ids=sorted(vehicles),
                departed_ids=list(departed), arrived_ids=[],
                colliding_ids=[], cav_ids=list(cav_ids),
                boundary_crossings=dict(mainline_entry=0, mainline_exit=0,
                                        ramp_entry=0),
                vehicles=dict(vehicles))


def _vehicles():
    lanes = (MAIN_RIGHT, MAIN_LEFT, ACCEL)
    out = {}
    for i in range(N_VEH):
        lane_id = lanes[i % 3]
        pos = 20.0 + 7.0 * i if lane_id != ACCEL else 381.0 + 0.4 * i
        out[f"v{i}"] = _veh(lane_id, pos, 25.0 + (i % 7) * 0.5)
    return out


def _write_run_dir(root, rows):
    (root / "run").mkdir(parents=True, exist_ok=True)
    with open(root / "run" / "physics_steps.jsonl", "w",
              encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (root / "run" / "manifest.json").write_text(json.dumps(
        dict(controller="damping", scenario=dict(cav_population=0))),
        encoding="utf-8")


@pytest.fixture(scope="module")
def v3_pack(tmp_path_factory):
    """featurize-all(format=v3) over one synthetic world -> pack_v3."""
    tmp = tmp_path_factory.mktemp("v3e2e")
    data, out = tmp / "data", tmp / "out"
    vehs = _vehicles()
    base_rows = [_row(t, vehs,
                      departed=sorted(vehs) if t == 0.5 else (),
                      cav_ids=["v0", "v1"])
                 for t in (0.5, 1.0, 1.5)]
    branch_rows = [_row(480.0 + t, vehs, cav_ids=["v0", "v1"])
                   for t in (0.5, 1.0, 1.5, 2.0)]
    _write_run_dir(data / "runs/campaign_stage1_chunk4/w192__base", base_rows)
    _write_run_dir(data / "runs/campaign_stage1_chunk4/w192__native",
                   branch_rows)
    synth.write_world_plan(
        data / "output/campaign_stage1_v3/results_plan_chunk4/WORLD_PLAN.json",
        [{"world": "w192", "split": "TRAIN", "t0_s": 480.0}])
    runs = [
        dict(batch="chunk4", chunk="chunk4", world="w192", arm="base",
             policy="damping", split="TRAIN", t0_s=480.0,
             root_rel="runs/campaign_stage1_chunk4", run="w192__base"),
        dict(batch="chunk4", chunk="chunk4", world="w192", arm="native",
             policy="native", split="TRAIN", t0_s=480.0,
             root_rel="runs/campaign_stage1_chunk4", run="w192__native"),
    ]
    inv = synth.make_inventory(tmp / "inv.parquet", runs)
    probe = lambda: {"free_virtual_gb": 99.0, "free_ram_gb": 99.0}  # noqa: E731

    rep = FA.featurize_all(inv, data, out, workers=1, probe=probe,
                           format="v3")
    assert rep["written"] == 2 and rep["skipped"] == 0
    receipt = U.read_json(out / FA.ALLOCATION_NAME)
    assert receipt["format"] == "v3" and receipt["k_max"] == 192

    pack_dir = tmp / "pack_v3"
    man = PK.pack_shards(out, pack_dir, workers=1)
    return tmp, out, pack_dir, man


def test_v3_pack_manifest_and_verify(v3_pack):
    tmp, out, pack_dir, man = v3_pack
    assert man["k_max"] == 192 and man["format"] == "v3"
    train = U.read_json(pack_dir / PK.MANIFEST_NAME)
    assert train["k_max"] == 192 and train["format"] == "v3"
    a = np.load(pack_dir / "veh_step.npy", mmap_mode="r")
    assert a.shape[1] == 192
    rep = VF.verify_pack(pack_dir)
    assert rep["ok"], rep["errors"]
    assert rep["k_max"] == 192 and rep["format"] == "v3"


def test_v3_pack_items_and_meta_self_describe(v3_pack):
    tmp, out, pack_dir, man = v3_pack
    meta = U.read_json(out / "chunk4" / "w192" / "base" / "meta.json")
    assert meta["format"] == "v3" and meta["k_max"] == 192
    assert meta["schema"] == "wm_n1_features_format_v3/1"
    assert meta["n_dropped_slot_steps"] == 0
    assert man["n_runs"] == 2 and man["totals"]["n_runs"] == 2


def test_verify_refuses_k_contract_tamper(v3_pack):
    tmp, out, pack_dir, man = v3_pack
    # (a) claim a disagreeing format
    idx = U.read_json(pack_dir / PK.INDEX_NAME)
    idx["format"] = "v2"
    U.write_json_atomic(pack_dir / PK.INDEX_NAME, idx)
    rep = VF.verify_pack(pack_dir)
    assert not rep["ok"]
    assert any("format" in e for e in rep["errors"]), rep["errors"]
    # (b) shrink the recorded K below the real slot dim
    idx = U.read_json(pack_dir / PK.INDEX_NAME)
    idx["format"] = "v3"
    idx["k_max"] = 128
    U.write_json_atomic(pack_dir / PK.INDEX_NAME, idx)
    rep = VF.verify_pack(pack_dir)
    assert not rep["ok"]
    assert any("slot dim" in e or "k_max" in e for e in rep["errors"])


def test_pack_refuses_mixed_k_shards(tmp_path):
    """A 192-wide shard cannot silently join a 128-wide shard tree."""
    import synth as S

    data = tmp_path / "shards"
    runs_128 = [dict(chunk="chunk4", world="wa", arm="base",
                     policy="damping", split="TRAIN", t0_s=540.0,
                     n_ticks=2, n_dec=2)]
    runs_192 = [dict(chunk="chunk4", world="wb", arm="base",
                     policy="native", split="TRAIN", t0_s=540.0,
                     n_ticks=2, n_dec=2)]
    S.write_shards(data, runs_128, k=128, seed=0)
    tree192 = tmp_path / "tree192"
    S.write_shards(tree192, runs_192, k=192, seed=1)
    # transplant the 192 run's INDEX entry (its shard_dir stays in tree192;
    # the packer resolves arrays through shard_dir)
    index = U.read_json(data / "INDEX.json")
    ix192 = U.read_json(tree192 / "INDEX.json")
    key = next(iter(ix192["runs"]))
    index["runs"][key] = ix192["runs"][key]
    U.write_json_atomic(data / "INDEX.json", index)
    with pytest.raises(ValueError):
        PK.pack_shards(data, tmp_path / "pack", workers=1)
