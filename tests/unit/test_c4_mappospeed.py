"""Speed-only MAPPO: real actor/PPO/loaders and fake TraCI, zero SUMO."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wms.control.contract import RewardWeights
from wms.control.mappo.agent import ActorStep, build_networks, export_payload
from wms.control.mappo.buffer import flatten_episodes
from wms.control.mappo.config_helpers import build_trainer, merge_override
from wms.control.mappo.eval_bridge import load_policy_fn as load_mappo
from wms.control.mappo.fakeenv import FakeCavEnv
from wms.control.mappo.ppo import PPOConfig, PPOUpdater
from wms.control.mappo.rollout import WorkerPool, collect_episode
from wms.control.policy import load_policy_fn
from wms.control.lanes import LaneRules


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config():
    return {"policy": {"lane_control": "native"}, "env": {"name": "fake"},
            "mappo": {"epochs": 1, "minibatch_frames": 4},
            "rollout": {"iterations": 1, "episodes_per_iteration": 1,
                        "horizon_s": 4., "warmup_s": 0., "keep_every": 1}}


def test_default_and_switch_validation(tmp_path):
    assert build_networks()[0].has_lane_head
    assert not build_trainer(merge_override({}, {"lane_control": "native"}), workers=1,
                             out_dir=tmp_path).policy.has_lane_head
    with pytest.raises(ValueError, match="lane_control"):
        build_networks(lane_control="off")


def test_accel_logprob_entropy_and_parallel_head_parity():
    learned, encoder, critic = build_networks()
    speed, _, _ = build_networks(lane_control="native")
    speed.load_state_dict({k: v for k, v in learned.state_dict().items()
                           if not k.startswith("lane_head.")}, strict=True)
    state = FakeCavEnv().reset(seed=4107, demand_vph=2400, penetration=.4,
                               warmup_s=0, horizon_s=4)
    a = ActorStep(learned, encoder)(state, generator=torch.Generator().manual_seed(9))
    b = ActorStep(speed, encoder)(state, generator=torch.Generator().manual_seed(9))
    # Identical observed lanes/reports; lane branch cannot feed the accel head.
    assert torch.equal(a["accel"], b["accel"])
    assert torch.equal(a["logp_accel"], b["logp_accel"])
    assert b["lanes"] is None and not b["lane_changed"].any()
    assert not b["lane_choice"].any() and not b["logp_lane"].any()

    class NativeEnv(FakeCavEnv):
        def step(self, accel_by_slot=None, *, lanes=None):
            assert lanes is None
            return super().step(accel_by_slot, lanes=lanes)

    ep = collect_episode(NativeEnv(), ActorStep(speed, encoder), weights=RewardWeights(),
                         seed=4107, demand_vph=2400, penetration=.4, warmup_s=0,
                         horizon_s=4, noise_seed=9)
    batch = flatten_episodes([ep])
    updater = PPOUpdater(policy=speed, group_encoder=encoder, critic=critic, config=PPOConfig())
    terms = updater._policy_terms(batch, torch.arange(batch.n_frames))
    assert torch.allclose(terms["new"], terms["old"], atol=2e-4)
    assert torch.allclose(terms["old"], batch.steps["logp_accel"][batch.steps["cav_mask"]])
    assert not terms["new_lane"].any() and not terms["entropy_lane"].any()
    assert terms["entropy_accel"].numel() and torch.isfinite(terms["entropy_accel"]).all()


def test_init_export_trainer_bridge_and_sumo_arm(tmp_path):
    from traffic_shepherd.c4.baselines import run_policy_episode

    policy, encoder, _ = build_networks(accel_max=2.6)
    source = tmp_path / "v5.pt"
    torch.save(export_payload(policy, encoder, LaneRules()), source)
    trainer = build_trainer(config(), workers=1, out_dir=tmp_path / "run")
    init = trainer.init_from_checkpoint(source)
    assert len(init["lane_head_removed_keys"]) == 4
    for k, v in trainer.policy.state_dict().items():
        assert torch.equal(v, policy.state_dict()[k])
    assert trainer.policy.accel_max == 2.6
    assert trainer.run()["iterations_completed"] == 1
    payload = torch.load(tmp_path / "run/policy_latest.pt", weights_only=True)
    assert payload["lane_control"] == "native" and payload["lane_head"] is False
    assert not any(k.startswith("lane_head.") for k in payload["policy"])
    state = FakeCavEnv().reset(seed=4107, demand_vph=2400, penetration=.4,
                               warmup_s=0, horizon_s=4)
    for path, loader in ((tmp_path / "run/policy_latest.pt", load_policy_fn),
                         (tmp_path / "run/policy_latest.pt", load_mappo),
                         (tmp_path / "run/checkpoints/ckpt_00001.pt", load_mappo)):
        exe = loader(path)
        assert exe.native_lanes and not exe.has_lane_head
        assert exe.act_with_lane(state)[1] is None
        calls = []

        class Env:
            def reset(self, **kwargs):
                return state

            def step(self, accel, **kwargs):
                calls.append(kwargs)
                assert kwargs.get("lanes") is None
                return state, {"done": True, "summary": {"ok": True}}

        assert run_policy_episode(Env(), exe, seed=4107, demand_vph=2400, penetration=.4,
                                  warmup_s=0, horizon_s=1) == {"ok": True}
        assert calls == [{}]
    resumed = build_trainer(config(), workers=1, out_dir=tmp_path / "run")
    assert resumed._try_resume()
    bad = build_trainer({**config(), "policy": {}}, workers=1, out_dir=tmp_path / "run")
    with pytest.raises(ValueError, match="resume lane_control"):
        bad._try_resume()


def test_init_does_not_silently_drop_other_keys(tmp_path):
    p, e, _ = build_networks()
    payload = export_payload(p, e, LaneRules())
    payload["policy"]["typo.weight"] = torch.zeros(1)
    source = tmp_path / "bad.pt"
    torch.save(payload, source)
    with pytest.raises(RuntimeError, match="typo.weight"):
        build_trainer(config(), workers=1, out_dir=tmp_path / "run").init_from_checkpoint(source)


def test_spawned_workers_keep_native_lanes():
    p, e, _ = build_networks(lane_control="native")
    spec = {"env_name": "fake", "lane_control": "native", "cleanup_scenarios": False}
    pool = WorkerPool(2, spec)
    job = {"policy_state": p.state_dict(), "encoder_state": e.state_dict(), "seed": 4107,
           "demand_vph": 2400, "penetration": .4, "warmup_s": 0., "horizon_s": 3.}
    try:
        rows = pool.run([job, {**job, "seed": 4108}], deadline_s=120)
        assert all(r["ok"] for r in rows), rows
        assert all(not r["episode"].steps["logp_lane"].any() for r in rows)
        assert all(r["episode"].meta["policy_lane_changes"] == 0 for r in rows)
    finally:
        pool.close()


def test_real_env_tick_never_touches_native_modes_and_restores_previous_mode():
    from traffic_shepherd.c4.env import CavSumoEnv

    env = object.__new__(CavSumoEnv)
    calls = []
    modes = {"cav": 1621}
    clock = [300.]
    vehicle = SimpleNamespace(getIDList=lambda: ["cav"],
                              getLaneChangeMode=lambda v: calls.append(("get", v)) or modes[v],
                              setLaneChangeMode=lambda v, m: (calls.append(("set", v, m)),
                                                             modes.__setitem__(v, m)),
                              changeLane=lambda *a: calls.append(("changeLane", *a)))
    env.api = SimpleNamespace(vehicle=vehicle,
                              simulation=SimpleNamespace(getTime=lambda: clock[0],
                                                         getArrivedIDList=lambda: []),
                              simulationStep=lambda: clock.__setitem__(0, clock[0] + .5))
    env.mode, env.warmup_s, env.direct_accel_max_mps2 = "direct", 300., 3.
    env.cav_ids = {"cav"}
    env._previous_live, env._retired_cavs = set(), set()
    env._table, env._lc_original = {}, {}
    env.sensor = SimpleNamespace(bind=lambda t: None, capture=lambda *a, **kw: {})
    env.slow_sensor, env.fleet, env.pi_session, env.recorder = None, None, None, None
    env._direct_pi = False
    env.ownership = SimpleNamespace(step=lambda *a, **kw: [])
    env._accel_by_vehicle = lambda a: {"cav": float(np.asarray(a)[0, 0])}
    env._after_physics = lambda *a: Counter()
    env.lane_command_counts, env.totals, env.window_totals = Counter(), Counter(), Counter()
    for _ in range(3):
        env._control_tick(np.zeros((1, 1)), lanes=None)
    assert calls == [] and modes == {"cav": 1621}
    # A mode previously owned by a learned-lane caller is restored, not guessed.
    env._lc_original = {"cav": 1621}
    modes["cav"] = 513
    env._control_tick(np.zeros((1, 1)), lanes=None)
    assert calls == [("set", "cav", 1621)] and modes["cav"] == 1621
