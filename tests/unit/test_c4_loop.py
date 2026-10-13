"""C4 Dreamer data loop: collected-episode converter, window loader, mixing and config wiring.

Fast and synthetic: a hand-built ``c4_sumo_episode/1`` episode (no SUMO, no packs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wms.cli import assemble  # noqa: E402
from wms.config.load import load_train_config  # noqa: E402
from wms.config.train import TrainConfig  # noqa: E402
from wms.core.errors import ConfigError, TrainDataRefusal  # noqa: E402
from wms.data import c4_episodes as C4  # noqa: E402
from wms.data.action_timeline import CONTROLLERS  # noqa: E402
from wms.data.collate import collate  # noqa: E402
from wms.data.dataset import TrajectoryDataset  # noqa: E402
from wms.data.mixdata import MixedTrajectoryDataset, MixSource  # noqa: E402

K = 6
DIRECT = CONTROLLERS.index("direct")
NATIVE = CONTROLLERS.index("native")
DAMPING = CONTROLLERS.index("damping")
WARMUP = 10.0
TOTAL = 40.0


def write_episode(root: Path, *, seed=4107, demand=2700, penetration=0.2, policy="random",
                  designed=False, status="complete") -> Path:
    """Synthetic recorder output.  Slots: 0 HDV lane 1; 1 CAV lane 0; 2 CAV that leaves at
    state 30, slot reused by HDV entrant (vid 4, ramp) from state 34; 3 CAV with a junction
    transit at states 30-31; 4 HDV ramp entrant (vid 5) from state 25; 5 empty."""
    t_n, c_n = int(TOTAL / 0.5) + 1, int(TOTAL)
    time_s = np.arange(t_n) * 0.5
    present = np.zeros((t_n, K), bool)
    vid = np.full((t_n, K), -1, np.int32)
    lane = np.full((t_n, K), -1, np.int8)
    s = np.zeros((t_n, K), np.float32)
    v = np.zeros((t_n, K), np.float32)
    a = np.zeros((t_n, K), np.float32)
    is_cav = np.zeros((t_n, K), bool)
    junction = np.zeros((t_n, K), bool)

    def put(slot, ident, rows, ln, s0, speed, cav=False):
        for i, t in enumerate(rows):
            present[t, slot], vid[t, slot], lane[t, slot] = True, ident, ln
            s[t, slot], v[t, slot] = s0 + speed * 0.5 * i, speed + 0.1 * np.sin(t)
            a[t, slot] = 0.2 * np.cos(t)
            is_cav[t, slot] = cav

    put(0, 0, range(t_n), 1, -1000.0, 20.0)
    put(1, 1, range(t_n), 0, -1200.0, 22.0, cav=True)
    put(2, 2, range(0, 30), 1, -900.0, 18.0, cav=True)
    put(2, 4, range(34, t_n), 2, -370.0, 15.0)
    put(3, 3, range(t_n), 1, -800.0, 19.0, cav=True)
    put(4, 5, range(25, t_n), 2, -375.0, 14.0)
    lane_carry = lane.copy()
    lane[30:32, 3] = -1
    junction[30:32, 3] = True
    controller = np.zeros((c_n, K), np.int8)
    acc = np.zeros((c_n, K), np.float32)
    tgt = np.zeros((c_n, K), np.float32)
    after = np.arange(c_n) >= WARMUP
    for slot in (1, 2, 3):      # slot 2 stays "direct" after its CAV left: must be masked
        controller[after, slot] = DAMPING if (designed and slot == 1) else DIRECT
        acc[after, slot] = -1.0 + 0.1 * slot
        tgt[after, slot] = 20.0
    ep = root / f"s{seed}_m{demand}_p{int(round(penetration * 1000)):04d}"
    ep.mkdir(parents=True)
    meta = dict(schema="c4_sumo_episode/1", seed=seed, demand_vph=demand,
                penetration=penetration, warmup_s=WARMUP, horizon_s=TOTAL - WARMUP,
                controller="direct", k_slots=K)
    np.savez_compressed(
        ep / "episode.npz", time_s=time_s, s_m=s, lane=lane, lane_carry=lane_carry, v_mps=v,
        a_mps2=a, length_true_m=np.where(present, 4.5, 0).astype(np.float32),
        present=present, is_cav=is_cav, junction=junction, vid_index=vid,
        vehicle_ids=np.asarray([f"v{i}" for i in range(6)]),
        cav_vehicle_ids=np.asarray(["v1", "v2", "v3"]),
        control_time_s=np.arange(c_n, dtype=np.float64), controller=controller,
        issued_acceleration_mps2=acc, issued_target_speed_mps=tgt,
        requested_acceleration_mps2=np.where(controller == DIRECT, acc, np.nan).astype(np.float32),
        controller_vocab=np.asarray(CONTROLLERS), meta_json=np.asarray(json.dumps(meta)))
    (ep / "manifest.json").write_text(json.dumps({**meta, "status": status, "policy": policy}))
    (ep / "summary.json").write_text("{}")
    return ep


@pytest.fixture()
def source(tmp_path):
    collect = tmp_path / "collect" / "episodes" / "random"
    write_episode(collect, seed=4107)
    write_episode(collect, seed=4108)
    write_episode(collect, seed=4109, status="running")          # incomplete: skipped
    out = tmp_path / "src"
    doc = C4.convert([tmp_path / "collect"], out, val_frac=0.5, split_seed=0)
    return out, doc, tmp_path


def test_world_ids_follow_the_pack_pattern():
    assert C4.world_id(4107, 2700, 0.2, 300) == "w4107m2700p20t300"
    assert C4.world_id(4120, 2400, 0.05, 300.0) == "w4120m2400p05t300"
    from wms.eval.states import world_meta

    meta = world_meta(C4.world_id(4107, 2600, 0.4, 300))
    assert (meta["demand"], meta["penetration_pct"], meta["t0_s"]) == (2600, 40, 300.0)
    with pytest.raises(TrainDataRefusal):
        C4.world_id(4107, 2700, 0.205, 300)


def test_convert_writes_a_world_split_train_source(source):
    out, doc, _ = source
    assert doc["schema"] == C4.SOURCE_SCHEMA and doc["split"] == "TRAIN"
    assert doc["n_episodes"] == 2 and doc["n_train"] == 1 and doc["n_val"] == 1
    worlds = {e["world"]: e["split_role"] for e in doc["episodes"]}
    assert set(worlds) == {"w4107m2700p20t10", "w4108m2700p20t10"}
    assert sorted(worlds.values()) == ["train", "val"]
    entry = doc["episodes"][0]
    assert entry["arm"] == "random" and entry["controllers_used"] == ["direct", "native"]
    arrays = C4.EpisodeArrays(out / entry["dir"])
    assert arrays.array("veh_kin").shape == (81, K, 7)
    tm = np.asarray(arrays.array("target_mask"), bool)
    assert tm[28, 2] and not tm[29, 2]            # vid 2 leaves after state 29
    assert tm[30, 3]                              # a junction transit keeps the vehicle
    counter = np.asarray(arrays.array("counter"))
    assert counter[0, 0, 0] == 4 and counter[0, 1].sum() == 0
    assert counter[40, 1, 0] == 2 and counter[40, 0, 2] == 4   # ramp entrants; t=0 entries aged 20 s
    with pytest.raises(FileExistsError):
        C4.convert([source[2] / "collect"], out)


def test_designed_controllers_are_refused_unless_allowed(tmp_path):
    write_episode(tmp_path / "c", designed=True, policy="damping")
    with pytest.raises(TrainDataRefusal, match="designed"):
        C4.convert([tmp_path / "c"], tmp_path / "out")
    doc = C4.convert([tmp_path / "c"], tmp_path / "out2", allow_designed=True)
    assert "damping" in doc["episodes"][0]["controllers_used"]


def test_restricted_output_names_are_refused(tmp_path):
    write_episode(tmp_path / "c")
    with pytest.raises(TrainDataRefusal):
        C4.convert([tmp_path / "c"], tmp_path / "HOLDOUT_x")


def test_windows_reproduce_the_pack_window_contract(source):
    out, _, _ = source
    built = C4.build_c4_datasets(out, horizon_steps=20, history_s=60.0, stride_s=10.0, k_slots=K)
    train = built["train"]
    assert len(train) == 3 and len(built["val"]) == 3          # t0 = 10, 20, 30 s
    assert built["train_events"].shape == (3, 2) and (built["train_events"][:, 1] == 0).all()
    rec = train[0]                                             # t0 = 10 s (state 20)
    assert rec["states"].shape == (21, K, 7) and rec["cav_teacher"].shape == (21, K, 4)
    assert rec["driver_hist"].shape == (K, 61, 9) and rec["hist"].shape == (K, 10, 6)
    assert rec["driver_hist_mask"][0, -11:].all() and not rec["driver_hist_mask"][0, :-11].any()  # 1 s stride, t0 = 10 s
    present = rec["present"]
    assert present[:10, 2].all() and not present[10:, 2].any()     # leaver never revives
    assert not present[:, 4].any() and not present[:, 5].any()      # mid-window entrant dropped
    assert present[:, 3].all() and not rec["loss_valid"][10:12, 3].any()   # junction rows
    assert rec["loss_valid"][9, 3] and rec["loss_valid"][12, 3]
    np.testing.assert_array_equal(rec["is_cav"][0], [False, True, True, True, False, False])
    ctrl = rec["action_timeline"]["controller"]
    assert ctrl.shape == (10, K)
    assert (ctrl[:, [1, 3]] == DIRECT).all()
    assert (ctrl[:5, 2] == DIRECT).all() and (ctrl[5:, 2] == NATIVE).all()  # t0 CAV left
    assert (rec["action_timeline"]["acceleration_mps2"][5:, 2] == 0).all()
    assert (ctrl[:, [0, 4, 5]] == NATIVE).all()
    assert rec["meta"]["world"] == "w4107m2700p20t10" or rec["meta"]["world"] == "w4108m2700p20t10"
    entry_rec = C4.build_c4_datasets(out, horizon_steps=20, entries_v2=True)["train"][1]
    assert entry_rec["entry_feat"].ndim == 1 and np.isfinite(entry_rec["entry_feat"]).all()
    with pytest.raises(TrainDataRefusal, match="K="):
        C4.build_c4_datasets(out, horizon_steps=20, k_slots=K + 1)
    with pytest.raises(TrainDataRefusal, match="horizon"):
        C4.build_c4_datasets(out, horizon_steps=200)


def test_windows_match_the_recorder_converter(source):
    """Independent cross-check against traffic_shepherd.c4.recorder.episode_windows."""
    recorder = pytest.importorskip("traffic_shepherd.c4.recorder")
    out, doc, _ = source
    entry = next(e for e in doc["episodes"] if e["split_role"] == "train")
    ref = recorder.episode_windows(entry["source"], window_s=10, stride_s=10, history_s=60)
    ds = C4.build_c4_datasets(out, horizon_steps=20, history_s=60.0)["train"]
    assert len(ds) == ref["states"].shape[0]
    for i in range(len(ds)):
        rec = ds[i]
        alive = rec["present"]
        for ch in (0, 4, 5):
            np.testing.assert_allclose(rec["states"][..., ch][alive],
                                       ref["states"][i][..., ch][alive], atol=1e-6)
        held = alive[0:20:2]
        np.testing.assert_array_equal(rec["action_timeline"]["controller"][held],
                                      ref["controller_timeline"][i][held])


def _plain(n, k=K, horizon=20):
    rng = np.random.default_rng(n)
    states = rng.normal(size=(n, horizon + 1, k, 7)).astype(np.float32)
    timeline = np.zeros((n, horizon // 2, k), np.int64)
    return TrajectoryDataset.from_arrays({"states": states, "controller_timeline": timeline})


def test_add_c4_source_rescales_existing_shares_and_collates(source):
    out, _, _ = source
    c4 = C4.build_c4_datasets(out, horizon_steps=20, stride_s=10.0)
    mixed, val = C4.add_c4_source(_plain(5), _plain(2), c4, mix_ratio=0.3)
    assert [s.name for s in mixed.sources] == ["original", "c4_episodes"]
    assert mixed.shares == pytest.approx([0.7, 0.3])
    assert [s.name for s in val.sources] == ["original_val", "c4_episodes_val"]
    two = MixedTrajectoryDataset([MixSource("original", _plain(4), share=0.7),
                                  MixSource("switching", _plain(3), share=0.3)])
    three, _ = C4.add_c4_source(two, None, c4, mix_ratio=0.3)
    assert three.shares == pytest.approx([0.49, 0.21, 0.3])
    assert len(three) == 4 + 3 + 3
    weights = three.stage_weights(10.0)
    assert sum(weights[-3:]) == pytest.approx(0.3) and sum(weights) == pytest.approx(1.0)
    batch = collate([three[0], three[len(three) - 1]])
    ctrl = batch["action_timeline"]["controller"]
    assert ctrl.shape == (2, 10, K) and (ctrl[1] == DIRECT).any() and not (ctrl[0] == DIRECT).any()
    assert three.receipt["c4_episodes"]["n_train_windows"] == 3
    with pytest.raises(TrainDataRefusal):
        C4.add_c4_source(_plain(2), None, c4, mix_ratio=1.0)


def test_refresh_config_and_knob_validation():
    cfg = load_train_config("c4_wm_refresh")
    base = load_train_config("c4_wm")
    c4 = cfg.data.c4_episodes
    assert c4.enabled and c4.path == "c4_episodes" and c4.mix_ratio == 0.3 and c4.use_val
    assert cfg.model.init_from == "ckpt_c4_wm"
    assert cfg.curriculum.stage_steps == (200, 400, 150)     # cfc1bbc
    assert cfg.curriculum.unroll_s == base.curriculum.unroll_s
    assert cfg.data.switching == base.data.switching and cfg.data.packs == base.data.packs
    assert cfg.model.lane_rule == base.model.lane_rule == "learned_discretionary"
    assert not base.data.c4_episodes.enabled
    names = assemble.declared_data_names(cfg)
    assert names["c4_episodes"] == "c4_episodes" and "c4_episodes" not in \
        assemble.declared_data_names(base)
    for bad in ("data.c4_episodes.mix_ratio=1.0", "data.c4_episodes.window_stride_s=2.5"):
        with pytest.raises(ConfigError):
            load_train_config("c4_wm_refresh", [bad])


def test_assemble_mixes_the_c4_source(source, monkeypatch, tmp_path):
    from wms.data import sidecars

    out, _, _ = source
    monkeypatch.setenv(sidecars.OUTCOMES_ENV, "x")
    monkeypatch.setenv(sidecars.IDENTITIES_ENV, "x")
    calls = []

    def fake_real_packs(paths, **kwargs):
        calls.append(kwargs.get("window_list"))
        return _plain(6 if len(calls) == 1 else 2)

    monkeypatch.setattr(TrajectoryDataset, "from_real_packs", staticmethod(fake_real_packs))
    tree = load_train_config("c4_wm_refresh").to_dict()
    tree["data"].pop("switching")
    tree["data"]["c4_episodes"]["path"] = str(out)
    tree["curriculum"].update(unroll_s=[10.0], stage_steps=[2], truncation_steps_per_stage=[20])
    tree["optim"]["lr_per_stage"] = [1e-4]
    cfg = TrainConfig.from_dict(tree)
    train, val, resolved = assemble.build_datasets(cfg, tmp_path / "run",
                                                   resolver=lambda name: Path(name))
    assert [s.name for s in train.sources] == ["original", "c4_episodes"]
    assert train.shares == pytest.approx([0.7, 0.3]) and len(train) == 6 + 3
    assert [s.name for s in val.sources] == ["original_val", "c4_episodes_val"]
    assert train.receipt["stage_sampling"][0]["sources"]["c4_episodes"]["n_items"] == 3
    assert str(out) in resolved
