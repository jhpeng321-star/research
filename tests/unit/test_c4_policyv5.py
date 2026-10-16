"""POLICYV5: C4 policy launch for the v5 world model.

1. ``configs/c4_policy_v5.toml`` resolves to the intended trainer / policy / counterfactual / data;
2. start windows from a ``data.c4_base`` WM config (``wms.control.c4_starts``): TRAIN worlds only,
   horizon 1, no pack name resolved, no jam scan; legacy configs refuse ``start_sources``;
3. the ``sumo_like`` queue surrogate (``wms.control.srcgrad``): v2 unchanged, agreement with the
   real ``TorchEntryProcessV2`` sumo_like decision, gradient where SUMO blocks;
4. the execution cap passes no gradient to commands above the ceiling (accel-cap rationale);
5. (optional) the start windows of the V5DATA smoke conversions when present.
CPU only; no SUMO, no checkpoint.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wms.control import srcgrad
from wms.control.contract import PHYSICS_DT_S
from wms.model.entries import (ENTRY_S_M, SUMO_POSITION_EPS_M, VTYPE_LENGTH_M, VTYPE_MIN_GAP_M,
                               EntranceHistory, EntryConfig, TorchEntryProcessV2)

REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path("runs/scratch/v5data")


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# ============================================================================ 1. config
def test_policy_v5_config_resolves():
    from wms.control.counterfactual import CounterfactualConfig
    from wms.control.dreamer import DreamerConfig
    from wms.control.train_cli import _read_config, policy_settings

    raw = _read_config(REPO / "configs" / "c4_policy_v5.toml")
    final = _read_config(REPO / "configs" / "c4_policy_final.toml")
    assert raw["data"]["wm_config"] == "configs/c4_wm_v5.toml"
    assert raw["data"]["start_sources"] == ["native_v5", "genrand_plain"]
    assert raw["run"]["output_dir"] == "runs/active/C4-POLICY-V5"
    cfg = DreamerConfig(**raw["trainer"])
    assert cfg.cav_exec_cap == "idm_normal:textbook" and cfg.entry_insert_rule == "sumo_like"
    assert cfg.source_wait_grad and cfg.critic_entry_backlog and cfg.horizon == 30
    assert policy_settings(raw) == {"accel_max": 1.2}
    assert CounterfactualConfig(**raw["counterfactual"]).native_accel_max_mps2 == 1.2
    # everything else is c4_policy_final
    for section in ("trainer", "counterfactual", "lanes", "reward"):
        changed = {k for k in set(raw[section]) | set(final[section])
                   if raw[section].get(k) != final[section].get(k)}
        assert changed <= {"entry_insert_rule", "native_accel_max_mps2"}, (section, changed)


# ============================================================ 2. start windows (fake sources)
class _FakeLoader:
    def __init__(self, entries, index, k):
        self.entries, self.index, self.k = entries, index, k

    def __call__(self, i):
        e, tick = self.index[i]
        states = np.zeros((2, self.k, 7), np.float32)
        states[..., 0] = np.arange(self.k) * 30.0
        states[..., 1] = 1.0
        states[..., 4] = 10.0
        states[..., 6] = 5.0
        present = np.ones((2, self.k), bool)
        is_cav = np.zeros((2, self.k), bool)
        is_cav[:, 0] = True
        return {"states": states, "present": present, "is_cav": is_cav,
                "hist": np.zeros((3, self.k, 4), np.float32), "hist_mask": np.ones((3, self.k), bool),
                "meta": {"source": "c4_episodes", "world": self.entries[e]["world"],
                         "episode": self.entries[e]["episode"], "tick": tick}}


def _fake_sources(monkeypatch, tmp_path):
    import wms.data.c4_episodes as episodes
    import wms.data.c4_pairs  # noqa: F401  (binds the real loaders before they are patched)
    from wms.data.dataset import TrajectoryDataset

    worlds = {"native": ["w1m2700p20t300", "w2m2700p20t300", "w3m2400p40t300", "w4m2400p40t300"],
              "plain": ["w1m2700p20t300", "w3m2400p40t300", "w4m2400p40t300", "w5m1900p10t300"],
              "branch": ["w1m2700p20t120"]}
    val = {"w2m2700p20t300", "w5m1900p10t300"}
    docs, calls = {}, []
    for name, ws in worlds.items():
        root = tmp_path / name
        root.mkdir()
        monkeypatch.setenv(f"WMS_DATA_{name.upper()}_SRC", str(root))
        docs[str(root)] = {"k_slots": 6, "episodes": [
            {"world": w, "split_role": "val" if w in val else "train", "episode": f"{name}_{w}",
             "arm": name, "dir": w} for w in ws]}

    def manifest(root):
        return docs[str(Path(root))]

    def build(root, **kwargs):
        calls.append((Path(root).name, kwargs))
        roles = kwargs["role_of"]
        entries = [e for e in docs[str(Path(root))]["episodes"] if roles[e["world"]] == "train"]
        index = [(i, t) for i in range(len(entries)) for t in (60, 80, 100)]
        loader = _FakeLoader(entries, index, kwargs["k_slots"])
        train = TrajectoryDataset([{} for _ in index], loader=loader,
                                  receipt={"split": "TRAIN", "source": str(root),
                                           "n_episodes": len(entries)})
        return {"train": train, "train_events": None, "val": None, "receipt": {}}

    monkeypatch.setattr(episodes, "read_source_manifest", manifest)
    monkeypatch.setattr(episodes, "build_c4_datasets", build)
    monkeypatch.setenv("C4_ALLOWED_ROOTS", str(tmp_path))
    return calls


def _wm_config(tmp_path, *, jam=2.0):
    text = f"""schema_version = 2
name = "fake-v5"
extends = "{(REPO / 'configs' / 'c4_wm_v5.toml').as_posix()}"
[data]
junction_fix = "ffill"
[data.jam]
oversample = {jam}
[[data.c4_sources]]
name = "native_v5"
path = "native_src"
mix_ratio = 0.65
window_stride_s = 10.0
use_val = true
[[data.c4_sources]]
name = "genrand_plain"
path = "plain_src"
mix_ratio = 0.30
window_stride_s = 10.0
use_val = true
[[data.c4_sources]]
name = "genrand_branch"
path = "branch_src"
mix_ratio = 0.05
window_stride_s = 10.0
use_val = true
paired = true
"""
    path = tmp_path / "wm_fake.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _no_jam_scan(monkeypatch):
    import wms.data.jam as jam

    def boom(*args, **kwargs):
        raise AssertionError("jam scan during policy start-window loading")

    monkeypatch.setattr(jam, "compute_jam_flags", boom)


def test_c4_start_windows_train_worlds_only_without_jam_scan(monkeypatch, tmp_path):
    from wms.control.train_cli import _load_starts

    calls = _fake_sources(monkeypatch, tmp_path)
    _no_jam_scan(monkeypatch)
    model = SimpleNamespace(history_s=60.0, history_stride_s=1.0)
    data = {"wm_config": str(_wm_config(tmp_path)), "val_fraction": 0.25}
    train, val = _load_starts(data, [model], seed=0, entries=True)
    # default sources: the non-paired ones, base first; horizon 1, WM junction_fix, base K
    assert [c[0] for c in calls] == ["native", "plain"]
    for _, kwargs in calls:
        assert kwargs["horizon_steps"] == 1 and kwargs["entries_v2"] is True
        assert kwargs["history_s"] == 60.0 and kwargs["k_slots"] == 6
        assert kwargs["junction_fix"] == "ffill" and kwargs["stride_s"] == 10.0
        assert kwargs["role_of"]["w2m2700p20t300"] == "val"
        assert kwargs["role_of"]["w5m1900p10t300"] == "val"     # val in ANY source -> val
    metas = [train.metadata[i] for g in train.groups.values() for i in g]
    metas += [val.metadata[i] for g in val.groups.values() for i in g]
    assert len(metas) == len(train.dataset) == (3 + 3) * 3
    assert not {m["world"] for m in metas} & {"w2m2700p20t300", "w5m1900p10t300"}
    assert {m["start_source"] for m in metas} == {"native_v5", "genrand_plain"}
    assert sorted(train.penetrations) == [0.2, 0.4]
    batch = train.sample(3)
    assert batch.state["s_m"].shape == (3, 6) and batch.state["is_cav"][:, 0].all()
    assert all(m["start_source"] in ("native_v5", "genrand_plain") for m in batch.metadata)


def test_c4_start_sources_selection_and_refusals(monkeypatch, tmp_path):
    from wms.control.train_cli import _load_starts

    calls = _fake_sources(monkeypatch, tmp_path)
    _no_jam_scan(monkeypatch)
    model = SimpleNamespace(history_s=None, history_stride_s=1.0)
    wm = str(_wm_config(tmp_path))
    train, _ = _load_starts({"wm_config": wm, "start_sources": ["native_v5"]}, [model], seed=0)
    assert [c[0] for c in calls] == ["native"]
    assert {m["start_source"] for m in train.metadata.values()} == {"native_v5"}
    with pytest.raises(ValueError, match="unknown or repeated"):
        _load_starts({"wm_config": wm, "start_sources": ["nope"]}, [model], seed=0)
    # legacy WM config (no c4_base): start_sources is refused before any data access
    legacy = str(REPO / "configs" / "c4_wm.toml")
    with pytest.raises(ValueError, match="needs a WM config with data.c4_base"):
        _load_starts({"wm_config": legacy, "start_sources": ["native_v5"]}, [model], seed=0)


def test_legacy_start_metadata_path_unchanged():
    from wms.control.start_states import _metadata

    class Records:
        records = [{"meta": {"world_id": "w1m2700p20t300"}}]

    assert _metadata(Records(), 0) == {"world_id": "w1m2700p20t300"}


# ============================================================ 3. sumo_like queue surrogate
def _hist(vtype=None):
    h = EntranceHistory.defaults()
    if vtype is not None:
        h.vtype_share = np.asarray(vtype, np.float64)
    names = ("rate", "speed_mean", "gap_mean", "gap_std", "vtype_share")
    return h, {n: torch.as_tensor(getattr(h, n), dtype=torch.float64)[None] for n in names}


def _ramp_state(rear_s, rear_v, k=4):
    s = torch.full((1, k), -2000.0, dtype=torch.float64)
    s[0, 0] = rear_s
    v = torch.zeros(1, k, dtype=torch.float64)
    v[0, 0] = rear_v
    present = torch.zeros(1, k, dtype=torch.bool)
    present[0, 0] = True
    return {"s_m": s, "v_mps": v, "lane": torch.full((1, k), 2), "present": present,
            "a_mps2": torch.zeros(1, k, dtype=torch.float64), "is_cav": torch.zeros(1, k, dtype=torch.bool),
            "length_m": torch.full((1, k), 5.0, dtype=torch.float64)}


def test_v2_surrogate_unchanged_by_the_extra_history_key():
    _, hist = _hist()
    state = _ramp_state(ENTRY_S_M[2] + 18.0, 3.0)
    cfg = EntryConfig()
    with_key = srcgrad.entrance_block_probability(state, state["present"], hist, ENTRY_S_M, cfg)
    no_key = srcgrad.entrance_block_probability(
        state, state["present"], {k: v for k, v in hist.items() if k != "vtype_share"}, ENTRY_S_M, cfg)
    manual = srcgrad.lane_block_probability(state, state["present"], 2, ENTRY_S_M[2],
                                            hist["speed_mean"][:, 1], hist["gap_mean"][:, 1],
                                            hist["gap_std"][:, 1], cfg)
    assert torch.equal(with_key, no_key) and torch.equal(with_key[:, 1], manual)


def _process_admits(rear_s, rear_v, vtype_share):
    h, _ = _hist(vtype_share)
    proc = TorchEntryProcessV2(history=h, dt_s=PHYSICS_DT_S, seed=0, batch=1,
                               cfg=EntryConfig(max_pending=10 ** 9, insert_rule="sumo_like"))
    state = _ramp_state(rear_s, rear_v)
    proc._observe(state)
    proc._started[:] = True
    proc.t_next[:] = math.inf                   # no new arrival; exactly one queued at the ramp
    proc.pending[0, 1] = 1
    return bool(proc.insert(state).any())


@pytest.mark.parametrize("vtype", [0, 1, 2, 3])
def test_sumo_surrogate_matches_the_process_decision(vtype):
    share = np.zeros((2, 4))
    share[:, vtype] = 1.0
    _, hist = _hist(share)
    cfg = EntryConfig(insert_rule="sumo_like")
    threshold = VTYPE_LENGTH_M[vtype] + SUMO_POSITION_EPS_M + VTYPE_MIN_GAP_M[vtype]
    rng = np.random.default_rng(vtype)
    agree = 0
    offsets = rng.uniform(0.0, 40.0, 200)
    for off in offsets:
        rear_v = float(rng.uniform(0.0, 25.0))
        state = _ramp_state(ENTRY_S_M[2] + 5.0 + off, rear_v)     # rear = s - length
        p = float(srcgrad.entrance_block_probability(state, state["present"], hist, ENTRY_S_M,
                                                     cfg)[0, 1])
        admitted = _process_admits(ENTRY_S_M[2] + 5.0 + off, rear_v, share)
        assert admitted == (off >= threshold)                    # the rule: bumper gap >= minGap
        if abs(off - threshold) > 3.0:                           # outside the soft band
            agree += int((p < 0.5) == admitted)
            assert (p < 0.05) if admitted else (p > 0.95)
    assert agree > 100


def test_sumo_surrogate_where_v2_over_blocks():
    """Rear vehicle 16 m past the ramp start at 15 m/s: SUMO (and sumo_like) insert at the safe
    speed; the v2 surrogate (headway x local speed + sampled gap) calls the entrance blocked."""
    _, hist = _hist()
    state = _ramp_state(ENTRY_S_M[2] + 5.0 + 16.0, 15.0)
    v2 = srcgrad.entrance_block_probability(state, state["present"], hist, ENTRY_S_M, EntryConfig())
    sumo = srcgrad.entrance_block_probability(state, state["present"], hist, ENTRY_S_M,
                                              EntryConfig(insert_rule="sumo_like"))
    assert float(v2[0, 1]) > 0.7 and float(sumo[0, 1]) < 1e-3
    assert _process_admits(ENTRY_S_M[2] + 21.0, 15.0, EntranceHistory.defaults().vtype_share)


def test_sumo_surrogate_gradient_and_mainline_two_lanes():
    _, hist = _hist()
    cfg = EntryConfig(insert_rule="sumo_like")
    # rear vehicle near the minGap threshold: moving it forward lowers the blocking probability
    s = torch.tensor(ENTRY_S_M[2] + 5.0 + 5.0 + SUMO_POSITION_EPS_M + 2.5, dtype=torch.float64,
                     requires_grad=True)
    state = _ramp_state(0.0, 0.0)
    state["s_m"] = torch.cat((s[None, None], state["s_m"][:, 1:]), -1)
    p = srcgrad.entrance_block_probability(state, state["present"], hist, ENTRY_S_M, cfg)
    (grad,) = torch.autograd.grad(p[0, 1], s)
    assert 0.3 < float(p[0, 1].detach()) < 0.7 and float(grad) < 0
    # mainline: one blocked lane is not enough (the arrival takes the other lane)
    k = 4
    main = {"s_m": torch.tensor([[ENTRY_S_M[0] + 6.0, -2000.0, -2000.0, -2000.0]], dtype=torch.float64),
            "v_mps": torch.zeros(1, k, dtype=torch.float64), "lane": torch.tensor([[0, 1, 1, 1]]),
            "present": torch.tensor([[True, False, False, False]]),
            "length_m": torch.full((1, k), 5.0, dtype=torch.float64)}
    p_main = srcgrad.entrance_block_probability(main, main["present"], hist, ENTRY_S_M, cfg)
    assert float(p_main[0, 0]) == 0.0
    main["lane"] = torch.tensor([[0, 1, 1, 1]])
    main["present"] = torch.tensor([[True, True, False, False]])
    main["s_m"][0, 1] = ENTRY_S_M[1] + 6.0
    p_both = srcgrad.entrance_block_probability(main, main["present"], hist, ENTRY_S_M, cfg)
    assert float(p_both[0, 0]) > 0.95
    with pytest.raises(ValueError, match="sumo_lane_mode"):
        srcgrad.entrance_block_probability(main, main["present"], hist, ENTRY_S_M,
                                           EntryConfig(insert_rule="sumo_like",
                                                       sumo_lane_mode="best_free"))


def test_sumo_surrogate_in_a_trainer_update(monkeypatch):
    """source_wait_grad + entry_insert_rule sumo_like: finite update, surrogate evaluated."""
    from .test_c4_srcgrad import _trainer, _world_model
    from .test_c4_srcwait_imagination import Starts

    import dataclasses

    calls = []
    original = srcgrad._sumo_entrance_block_probability

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(srcgrad, "_sumo_entrance_block_probability", counted)
    trainer = _trainer(_world_model(), Starts(True), grad=True, horizon=3)
    trainer.config = dataclasses.replace(trainer.config, entry_insert_rule="sumo_like")
    torch.manual_seed(5)
    metrics = trainer.update()
    assert calls and all(math.isfinite(v) for v in metrics.values() if isinstance(v, float))


# ============================================================ 4. execution cap gradient
def test_exec_cap_passes_no_gradient_above_the_ceiling():
    from wms.model.cav import EXEC_CAP_KEY, apply_exec_cap

    command = torch.tensor([[-2.0, 0.5, 1.1, 1.5, 2.6]], requires_grad=True)
    state = {EXEC_CAP_KEY: torch.full((1, 5), 1.2)}
    apply_exec_cap(state, command).sum().backward()
    assert command.grad.tolist() == [[1.0, 1.0, 1.0, 0.0, 0.0]]


# ============================================================ 5. smoke conversions
@pytest.mark.skipif(not (SCRATCH / "conv_native" / "C4_EPISODES_MANIFEST.json").is_file(),
                    reason="V5DATA smoke conversions not present")
def test_start_windows_on_v5_smoke_conversions(monkeypatch):
    from wms.control.train_cli import _load_starts

    monkeypatch.setenv("WMS_DATA_V5_NATIVE", str(SCRATCH / "conv_native"))
    monkeypatch.setenv("WMS_DATA_GENRAND_PLAIN", str(SCRATCH / "conv_generic_random"))
    monkeypatch.setenv("WMS_DATA_GENRAND_BRANCH", str(SCRATCH / "conv_branch"))
    monkeypatch.setenv("C4_ALLOWED_ROOTS", str(SCRATCH))
    _no_jam_scan(monkeypatch)
    model = SimpleNamespace(history_s=60.0, history_stride_s=1.0)
    train, val = _load_starts({"wm_config": "configs/c4_wm_v5.toml",
                               "start_sources": ["native_v5", "genrand_plain"]},
                              [model], seed=0, entries=True)
    metas = list(train.metadata.values()) + list(val.metadata.values())
    assert {m["world"] for m in metas} <= {"w4107m2700p20t300", "w4109m2400p40t300",
                                           "w4110m2400p40t300"}
    batch = train.sample(2)
    assert batch.entry_feat is not None and torch.isfinite(batch.entry_feat).all()
    assert "driver_history" in batch.rollout_kwargs
