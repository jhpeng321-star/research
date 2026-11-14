import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(PKG))

from wms_eval import materialize as M  # noqa: E402

K = 4
T = 10


def _kin_row(s, lane, v):
    row = np.zeros(7, np.float16)
    row[0] = s
    row[1 + lane] = 1.0
    row[4] = v
    row[6] = 5.0
    return row


def _build_pack(root: Path, counter: bool = False) -> Path:
    pack = root / "pack_v3"
    pack.mkdir(parents=True)
    kin = np.zeros((T, K, 7), np.float16)
    mask = np.zeros((T, K), np.uint8)
    step = np.zeros((T, K, 16), np.float16)
    tmask = np.zeros((T, K), np.uint8)
    for t in range(T):
        kin[t, 0] = _kin_row(10.0 * t, 0, 20.0)
        step[t, 0, 7] = 1.0                     # slot 0 is a CAV
        mask[t, 0] = 1
        if t <= 4:
            kin[t, 1] = _kin_row(500.0 + 15.0 * t, 1, 15.0)
            mask[t, 1] = 1
        if t >= 5:
            kin[t, 3] = _kin_row(800.0 + 10.0 * t, 2, 10.0)
            mask[t, 3] = 1
    for t in range(T - 1):
        tmask[t] = (mask[t] & mask[t + 1]).astype(np.uint8)
    for name, arr in (("veh_kin", kin), ("veh_mask", mask), ("veh_step", step),
                      ("target_mask", tmask)):
        np.save(pack / f"{name}.npy", arr)
    arrays = {
        "veh_kin": {"file": "veh_kin.npy", "shape": [T, K, 7]},
        "veh_mask": {"file": "veh_mask.npy", "shape": [T, K]},
        "veh_step": {"file": "veh_step.npy", "shape": [T, K, 16]},
        "target_mask": {"file": "target_mask.npy", "shape": [T, K]},
    }
    if counter:
        # RSU-observable t0 entrance counter, [N_ticks, 2, 30] like pack_v3
        cnt = np.zeros((T, 2, 30), np.float16)
        cnt[2, 0, 29] = 7.0                     # value AT the snapshot tick
        np.save(pack / "counter.npy", cnt)
        arrays["counter"] = {"file": "counter.npy", "shape": [T, 2, 30]}
    manifest = {
        "schema": "wm_n1_pack/1", "k_max": K,
        "arrays": arrays,
        "items": [{"world": "wTESTm1000p02t100", "arm": "native", "chunk": "chunk1",
                   "t0_s": 1.0, "tick": 2, "pair_key": "wTESTm1000p02t100@t0=1",
                   "policy": "native", "split": "TRAIN"}],
    }
    index = {"schema": "wm_n1_corpus_pack/2", "k_max": K, "runs": [
        {"key": "chunk1/wTESTm1000p02t100__native", "chunk": "chunk1",
         "world": "wTESTm1000p02t100", "arm": "native", "split": "TRAIN",
         "t0_s": 1.0, "n_ticks": T, "tick_start": 0, "tick_end": T,
         "dec_start": 0, "dec_end": 20},
    ]}
    (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (pack / "INDEX.json").write_text(json.dumps(index), encoding="utf-8")

    side = root / "identities"
    side.mkdir()
    np.save(side / "identities.npy", np.asarray([[0, 1, -1, 2]], np.int64))
    np.save(side / "global_positions.npy", np.asarray([0], np.int64))
    (side / "vehicle_map.json").write_text(json.dumps(
        [["chunk1", "wTESTm1000p02t100", "vehA"],
         ["chunk1", "wTESTm1000p02t100", "vehB"],
         ["chunk1", "wTESTm1000p02t100", "vehC"]]), encoding="utf-8")
    return pack, side


def _window():
    return {"key": "chunk1/wTESTm1000p02t100__native", "pack": "pack_v3",
            "run": "wTESTm1000p02t100__native", "world": "wTESTm1000p02t100",
            "arm": "native", "policy": "native", "chunk": "chunk1",
            "t0_s": 1.0, "split": "TRAIN"}


def test_build_state_binds_ids_and_follows_target_mask(tmp_path):
    pack_dir, side_dir = _build_pack(tmp_path)
    pack = M.Pack("pack_v3", pack_dir)
    identity = M.Identity("pack_v3", side_dir)
    record = M.build_state(_window(), {"pack_v3": pack}, {"pack_v3": identity},
                           horizon_s=5.0, min_vehicles=1)
    init = {r["id"] for r in record["initial_state"]["vehicles"]}
    assert init == {"vehA", "vehB"}
    assert record["t0_s"] == 1.0
    frames = record["recorded"]["native"]
    # horizon: 5 s -> 10 steps, capped at run end tick 9 from tick 2 -> 7
    assert len(frames) == 8
    assert [f["time_s"] for f in frames] == pytest.approx([1.0 + 0.5 * i for i in range(8)])
    def present(frame, vid):
        return any(r["id"] == vid and r["present"] for r in frame["vehicles"])
    # vehB departs after tick 4, so it vanishes from frame 3 onward
    assert present(frames[2], "vehB") and not present(frames[3], "vehB")
    # a slot reused at tick 5 never becomes the t0 vehicle vehB again
    assert not present(frames[4], "vehB")
    # vehA is present throughout
    assert all(present(f, "vehA") for f in frames[:4])


def test_build_state_carries_t0_entrance_counter(tmp_path):
    """C1 v2 rebuilds must ship the RSU t0 counter so --entries can be enabled."""
    pack_dir, side_dir = _build_pack(tmp_path, counter=True)
    pack = M.Pack("pack_v3", pack_dir)
    identity = M.Identity("pack_v3", side_dir)
    record = M.build_state(_window(), {"pack_v3": pack}, {"pack_v3": identity},
                           horizon_s=5.0, min_vehicles=1)
    assert record["counter"] is not None
    counter = np.asarray(record["counter"], np.float32)
    assert counter.shape == (2, 30)
    assert counter[0, 29] == pytest.approx(7.0)     # read AT the snapshot tick


def test_build_state_counter_is_none_when_pack_lacks_it(tmp_path):
    pack_dir, side_dir = _build_pack(tmp_path)
    pack = M.Pack("pack_v3", pack_dir)
    identity = M.Identity("pack_v3", side_dir)
    record = M.build_state(_window(), {"pack_v3": pack}, {"pack_v3": identity},
                           horizon_s=5.0, min_vehicles=1)
    assert record["counter"] is None


def test_build_state_rejects_degenerate_snapshot(tmp_path):
    pack_dir, side_dir = _build_pack(tmp_path)
    pack = M.Pack("pack_v3", pack_dir)
    identity = M.Identity("pack_v3", side_dir)
    with pytest.raises(M.MaterializeError):
        M.build_state(_window(), {"pack_v3": pack}, {"pack_v3": identity},
                      horizon_s=5.0, min_vehicles=5)


def test_selection_rules():
    pi = [{"policy": "pi", "arm": "pi", "key": f"k{i}"} for i in range(3)]
    mixed = ([{"policy": "fixed", "arm": "fixed1", "key": f"f{i}"} for i in range(4)] +
             [{"policy": "switching", "arm": "switching1", "key": f"s{i}"} for i in range(4)])
    fleet = M.select_fleet_pi(pi + mixed)
    assert [w["key"] for w in fleet] == ["k0", "k1", "k2"]
    sample = M.select_mixed_sample(pi + mixed, size=5)
    assert len(sample) == 5
    assert {w["policy"] for w in sample} == {"fixed", "switching"}
    # deterministic
    assert sample == M.select_mixed_sample(pi + mixed, size=5)


def test_build_state_baseline_arm_resolution_ignores_chunk_prefix(tmp_path):
    pack_dir, side_dir = _build_pack(tmp_path)
    pack = M.Pack("pack_v3", pack_dir)
    identity = M.Identity("pack_v3", side_dir)
    # metadata key carries a truearms_ batch prefix; resolution is by world/arm
    window = dict(_window(), key="truearms_chunk1/wTESTm1000p02t100__native")
    record = M.build_state(window, {"pack_v3": pack}, {"pack_v3": identity},
                           horizon_s=5.0, min_vehicles=1)
    assert "native" in record["recorded"]


def test_write_stream_emits_valid_json(tmp_path):
    pack_dir, side_dir = _build_pack(tmp_path)
    packs = {"pack_v3": M.Pack("pack_v3", pack_dir)}
    identities = {"pack_v3": M.Identity("pack_v3", side_dir)}
    out = tmp_path / "out.json"
    summary = M._write_stream(out, "seen", [_window()], packs, identities,
                              horizon_s=5.0, on_error="raise", min_vehicles=1)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert summary["n_states"] == 1
    assert payload["windows"][0]["recorded"]["native"][0]["vehicles"][0]["id"] == "vehA"


def test_shift_moves_t0_inside_the_window_and_builds_real_history(tmp_path):
    """Shifted t0 uses the stored post-t0 ticks as the model's pre-t0 history."""
    pack_dir, side_dir = _build_pack(tmp_path)
    pack = M.Pack("pack_v3", pack_dir)
    identity = M.Identity("pack_v3", side_dir)
    record = M.build_state(_window(), {"pack_v3": pack}, {"pack_v3": identity},
                           horizon_s=5.0, min_vehicles=1, shift_steps=2)
    assert record["t0_s"] == 2.0                 # 1.0 + 2 * 0.5 s
    assert record["t0_shift_steps"] == 2 and record["t0_shift_s"] == 1.0
    # snapshot at tick 4: vehA continues, vehB is still present (t <= 4)
    init = {r["id"] for r in record["initial_state"]["vehicles"]}
    assert init == {"vehA", "vehB"}
    frames = record["recorded"]["native"]
    # remaining window from tick 4 is 5 ticks -> 6 frames, rebased at 2.0 s
    assert len(frames) == 6
    assert [f["time_s"] for f in frames] == pytest.approx([2.0 + 0.5 * i for i in range(6)])
    # history now spans the ticks before the new t0 (ticks 0..4 = 5 ticks)
    hist = record["history"]
    assert hist["history_steps"] == 5
    assert hist["kin"][-1][0][0] == pytest.approx(10.0 * 4)   # t0 tick kinematics


def test_shift_reads_the_counter_at_the_new_t0(tmp_path):
    pack_dir, side_dir = _build_pack(tmp_path, counter=True)
    cnt = np.load(pack_dir / "counter.npy")
    cnt[4, 0, 29] = 9.0                          # distinct value at the shifted tick
    np.save(pack_dir / "counter.npy", cnt)
    pack = M.Pack("pack_v3", pack_dir)
    identity = M.Identity("pack_v3", side_dir)
    record = M.build_state(_window(), {"pack_v3": pack}, {"pack_v3": identity},
                           horizon_s=5.0, min_vehicles=1, shift_steps=2)
    counter = np.asarray(record["counter"], np.float32)
    assert counter[0, 29] == pytest.approx(9.0)
