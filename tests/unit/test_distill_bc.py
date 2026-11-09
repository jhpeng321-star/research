"""DISTILL student trainer (tools/distill): CPU tests on a synthetic schema fixture (zero SUMO)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.distill import data as D
from tools.distill import train_bc as T

ROOT = Path(__file__).resolve().parents[2]
HAVE_INPUTS = (ROOT / T.DEFAULT_CONFIG["u0_path"]).is_file() and (ROOT / T.DEFAULT_CONFIG["wzl_path"]).is_file()
needs_inputs = pytest.mark.skipif(not HAVE_INPUTS, reason="u0 / S-MAPPOWZL exports not staged in _inputs/")


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    return D.write_fixture(tmp_path_factory.mktemp("distill_fx"), n_episodes=4, steps=24, n_veh=30, seed=3)


@pytest.fixture(scope="module")
def rt():
    return T.setup_runtime(T.DEFAULT_CONFIG)


def test_fixture_reads_and_categories(fixture_dir):
    ds = D.load_dataset(fixture_dir, pre_onset_s=10)
    assert len(ds.episodes) == 4 and ds.n_frames == 4 * 24
    assert ds.frame_off[-1] == len(ds.s)
    plan = int(np.sum(ds.cat == D.CAT_PLAN))
    pre = int(np.sum(ds.cat == D.CAT_PRE))
    assert plan == 4 * 2 * 8                      # 2 members x steps//3 active steps per episode
    assert pre == 4 * 2 * 8                       # onset at step 8: only 8 earlier steps exist (< 10 s)
    assert np.all(ds.cat[~ds.is_cav] == D.CAT_NONE)
    assert set(np.unique(ds.lane_tgt)) <= {D.IGNORE, 0, 1, 2}


def test_receipts_required(fixture_dir, tmp_path):
    import shutil
    shutil.copytree(fixture_dir, tmp_path / "d")
    (tmp_path / "d" / "receipts" / "990001.json").write_text(json.dumps({"status": "interrupted"}))
    files, skipped = D.shard_files(tmp_path / "d")
    assert [s for s, _ in files] == [990000, 990002, 990003] and skipped


def test_split_deterministic():
    seeds = list(range(993100, 993300))
    a = D.split_episodes(seeds, 0.1, 7)
    assert a == D.split_episodes(seeds, 0.1, 7)
    assert len(a[1]) == 20 and not set(a[0]) & set(a[1])


def test_weights_equal_total(fixture_dir):
    ds = D.load_dataset(fixture_dir, pre_onset_s=10)
    cfg = dict(T.DEFAULT_CONFIG, nonplan_keep_ratio=1.0, plan_share=0.5)
    w_p, w_o, keep, c = T.weight_plan(ds, [0, 1, 2], cfg)
    assert abs(w_p * c["train_rows_plan_related"] - w_o * keep * c["train_rows_other"]) < 1e-6 * w_p * c["train_rows_plan_related"]


def test_dense_lane_mask_equals_gate(fixture_dir, rt):
    ds = D.load_dataset(fixture_dir)
    b = T.Batcher(ds, torch.device("cpu"))
    rows, valid = b.rows(np.arange(0, 40, 3))
    state = b.state(rows, valid)
    dense = T.lane_mask(rt, state)
    ref = rt["gate"].feasibility(state, None, rt["rules"])
    assert torch.equal(dense[valid], ref[valid])


@needs_inputs
def test_init_accel_equals_u0_and_actorstep(fixture_dir, rt):
    """Student at init: accel == U0's (speed-only export) on the same input; forward() == ActorStep."""
    from wms.control.mappo.agent import ActorStep
    from tools.nm_mappo.sanity_eval import load_actor
    policy, encoder, _, rec = T.init_networks(T.DEFAULT_CONFIG, np.array([1000, 5, 5]), seed=0)
    policy.eval()
    encoder.eval()
    assert rec["accel_and_encoder"]["accel_bounds"] == [-3.0, 2.6]
    u0_exe, lc = load_actor(ROOT / T.DEFAULT_CONFIG["u0_path"], rt["lanes_cfg"])
    assert lc == "native"
    ds = D.load_dataset(fixture_dir)
    b = T.Batcher(ds, torch.device("cpu"))
    k = rt["record"]["k_slots"]
    for f in (0, 5, 30):
        rows, valid = b.rows([f])
        compact = b.state(rows, valid)
        with torch.no_grad():
            out = T.forward(policy, encoder, rt, compact)
        # same frame as a [1,K] env state (slots from the shard; memory keys from the shard)
        z = np.load(sorted((fixture_dir / "shards").glob("*.npz"))[ds.frame_ep[f]])
        lo, hi = ds.frame_off[f], ds.frame_off[f + 1]
        local = f - int(np.flatnonzero(ds.frame_ep == ds.frame_ep[f])[0])
        off = z["off"]
        slots = torch.as_tensor(z["slot"][off[local]:off[local + 1]].astype(np.int64))
        full = {key: torch.zeros(1, k) for key in ("s_m", "v_mps", "a_mps2", "cmd_accel_mps2", "lane_intent")}
        full["length_m"] = torch.full((1, k), 5.0)
        full["lane"] = torch.full((1, k), -1, dtype=torch.long)
        full["present"] = torch.zeros(1, k, dtype=torch.bool)
        full["is_cav"] = torch.zeros(1, k, dtype=torch.bool)
        for key, src in (("s_m", ds.s), ("v_mps", ds.v), ("a_mps2", ds.a), ("length_m", ds.length),
                         ("cmd_accel_mps2", ds.cmd), ("lane_intent", ds.intent)):
            full[key][0, slots] = torch.as_tensor(src[lo:hi])
        full["lane"][0, slots] = torch.as_tensor(ds.lane[lo:hi].astype(np.int64))
        full["present"][0, slots] = True
        full["is_cav"][0, slots] = torch.as_tensor(ds.is_cav[lo:hi])
        step = ActorStep(policy, encoder, rt["rules"], lane_actions=rt["gate"])
        step.memory.augment = lambda s: s          # the shard already carries the teacher's memory keys
        ref = step(full, deterministic=True)
        cav_slots = slots[torch.as_tensor(ds.is_cav[lo:hi])]
        got = torch.zeros(1, k)
        got[0, slots[out["index"][0][out["cav"][0]]]] = out["accel"][0][out["cav"][0]]
        assert torch.allclose(got[0, cav_slots], ref["accel"][0, cav_slots], atol=1e-5)
        u0_step = ActorStep(u0_exe.policy, u0_exe.group_encoder, rt["rules"])
        u0_step.memory.augment = lambda s: s
        u0_ref = u0_step(full, deterministic=True)
        assert torch.allclose(u0_ref["accel"][0, cav_slots], ref["accel"][0, cav_slots], atol=1e-6)
        assert (ref["lane_choice"][0, cav_slots] == 0).all()     # fresh prior head: stay


@needs_inputs
def test_train_export_check_and_determinism(fixture_dir, tmp_path):
    from tools.distill.check_ckpt import check
    cfg = dict(batch_frames=16)
    r1 = T.train(fixture_dir, tmp_path / "a", epochs=3, seed=5, cfg_overrides=cfg, device="cpu", log=lambda *_: None)
    r2 = T.train(fixture_dir, tmp_path / "b", epochs=3, seed=5, cfg_overrides=cfg, device="cpu", log=lambda *_: None)
    p1 = torch.load(r1["checkpoint"], weights_only=True)
    p2 = torch.load(r2["checkpoint"], weights_only=True)
    for key in ("policy", "group_encoder"):
        assert all(torch.equal(p1[key][n], p2[key][n]) for n in p1[key])
    assert p1["trainer"] == "mappo" and p1["lane_head"] is True and p1["iteration"] == 3
    assert p1["accel_min_mps2"] == -3.0 and p1["accel_max_mps2"] == 2.6
    res = check(r1["checkpoint"])
    assert res["ok"] and res["deterministic"]
    lines = [json.loads(x) for x in (tmp_path / "a" / "metrics.jsonl").read_text().splitlines()]
    assert [x["epoch"] for x in lines] == [0, 1, 2, 3]
    val = lines[-1]["val"]
    assert {"plan", "pre_onset", "other", "plan_related", "non_plan"} <= set(val)
    with pytest.raises(FileExistsError):
        T.train(fixture_dir, tmp_path / "a", epochs=1, seed=5, cfg_overrides=cfg, device="cpu", log=lambda *_: None)


@needs_inputs
def test_plan_rows_fit_improves(fixture_dir, tmp_path):
    r = T.train(fixture_dir, tmp_path / "c", epochs=8, seed=1, cfg_overrides=dict(batch_frames=8, lr=3e-3),
                device="cpu", log=lambda *_: None)
    assert r["final_val"]["plan"]["accel_mae"] < r["init_val"]["plan"]["accel_mae"]


@needs_inputs
def test_wzl_lane_init_loads():
    policy, _, _, rec = T.init_networks(dict(T.DEFAULT_CONFIG, lane_init="wzl"), None, seed=0)
    wzl = torch.load(ROOT / T.DEFAULT_CONFIG["wzl_path"], weights_only=True)["policy"]
    assert torch.equal(policy.lane_head[-1].weight, wzl["lane_head.2.weight"])
    assert rec["lane_head"]["sha256"] == T.DEFAULT_CONFIG["wzl_sha256"]


def test_unsorted_frame_rows_are_resorted(fixture_dir, tmp_path):
    """A shard whose within-frame rows are not in slot order loads identically (labels remapped)."""
    src = sorted((fixture_dir / "shards").glob("*.npz"))[0]
    z = dict(np.load(src, allow_pickle=False))
    off, n = z["off"], int(z["off"][-1])
    perm = np.arange(n)
    rng = np.random.default_rng(0)
    for i in range(len(z["t"])):
        perm[off[i]:off[i + 1]] = rng.permutation(np.arange(off[i], off[i + 1]))
    inverse = np.empty(n, np.int64)
    inverse[perm] = np.arange(n)
    shuffled = dict(z)
    for key in ("slot", "vid", "s_m", "v_mps", "a_mps2", "lane", "junction", "is_cav", "length_m",
                "cmd_accel_mps2", "lane_intent", "dwell_age_s"):
        shuffled[key] = z[key][perm]
    shuffled["cav_row"] = inverse[z["cav_row"]]
    (tmp_path / "shards").mkdir()
    np.savez_compressed(tmp_path / "shards" / src.name, **shuffled)
    a = D.read_shard(src)
    b = D.read_shard(tmp_path / "shards" / src.name)
    for key in ("acc_tgt", "acc_u0", "lane_tgt", "cat"):
        assert np.array_equal(a[key], b[key], equal_nan=True)
    for key in a["veh"]:
        assert np.array_equal(a["veh"][key], b["veh"][key])
