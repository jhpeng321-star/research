"""SRCWAIT (MAPPO / SUMO half): entry-queue plumbing in the env, the MAPPO control-step reward
and ``--init-from`` warm starts.  CPU fake env except the SUMO parity test."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest
import torch

from wms.control.contract import RewardWeights
from wms.control.lanes import LaneRules
from wms.control.mappo.agent import ActorStep, build_networks
from wms.control.mappo.fakeenv import FakeCavEnv
from wms.control.mappo.ppo import PPOConfig
from wms.control.mappo.rollout import COMPONENTS, collect_episode, control_step_reward
from wms.control.mappo.trainer import MappoConfig, MappoTrainer, RolloutConfig, sha256_file
from wms.control.reward import team_reward

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "configs" / "c4_mappo.toml"


def _trainer(out: Path, *, env_kwargs=None, seed: int = 7, iterations: int = 1,
             net_seed: int = 11) -> MappoTrainer:
    torch.manual_seed(net_seed)
    policy, encoder, critic = build_networks()
    cfg = MappoConfig(
        ppo=PPOConfig(epochs=2, minibatch_frames=4, target_kl=None),
        rollout=RolloutConfig(seed=seed, iterations=iterations, episodes_per_iteration=2,
                              horizon_s=5.0, warmup_s=0.0, penetrations=(0.05, 0.4),
                              demands_vph=(2400, 2700), seeds="4107-4148",
                              max_sumo_seconds=1e9))
    return MappoTrainer(out_dir=out, config=cfg, policy=policy, group_encoder=encoder,
                        critic=critic,
                        env_spec={"env_name": "fake", "env_kwargs": env_kwargs or {}})


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------ reward
def _transition():
    env = FakeCavEnv(k_slots=8, min_vehicles=8)
    s0 = env.reset(seed=4111, demand_vph=2600, penetration=0.4, warmup_s=0.0, horizon_s=4.0)
    s1, _ = env.step(torch.zeros(1, 8))
    return env, s0, s1


def test_control_step_reward_without_queue_is_the_old_reward():
    env, s0, s1 = _transition()
    w = RewardWeights()
    got = control_step_reward(env, s0, s1, w)                       # no info / no env fields
    assert got["source_wait"] == 0.0 and got["total_wait"] == got["delay"]
    accel = torch.where(s0["present"], s1["a_mps2"], torch.zeros_like(s1["a_mps2"]))
    old = team_reward(s0, s1, accel, w)
    for key in ("reward", "delay", "cost_ttc", "fairness_delay", "throughput"):
        assert abs(got[key] - float(old[key][0])) < 1e-6, key
    assert control_step_reward(env, s0, s1, w, info={"entry_queue_steps": None}) == got


def test_control_step_reward_with_queue_charges_source_wait():
    env, s0, s1 = _transition()
    w = RewardWeights()
    base = control_step_reward(env, s0, s1, w)
    w1, w2 = 1.5, 2.5                                              # vehicle-seconds per physics step
    info = {"source_wait_steps": [w1, w2], "entry_queue_steps": [w1 / 0.5, w2 / 0.5]}
    got = control_step_reward(env, s0, s1, w, info=info)           # single 1 s transition fallback
    assert abs(got["source_wait"] - (w1 + w2)) < 1e-6
    assert abs(got["total_wait"] - (got["delay"] + w1 + w2)) < 1e-6
    assert got["reward"] < base["reward"]
    assert abs(got["delay"] - base["delay"]) < 1e-9
    env.last_entry_queue_steps = [w1 / 0.5, w2 / 0.5]              # env attribute fallback
    assert abs(control_step_reward(env, s0, s1, w, info={})["source_wait"] - (w1 + w2)) < 1e-6


def test_collect_episode_logs_source_wait_components():
    torch.manual_seed(0)
    policy, encoder, _ = build_networks()
    actor = ActorStep(policy, encoder, LaneRules())

    def run(env_kwargs):
        return collect_episode(FakeCavEnv(**env_kwargs), actor, weights=RewardWeights(), seed=4107,
                               demand_vph=2600, penetration=0.4, warmup_s=0.0, horizon_s=6.0,
                               noise_seed=3)
    plain, queued = run({}), run({"entry_queue": 2.0})
    assert "source_wait" in COMPONENTS and "total_wait" in COMPONENTS
    assert float(plain.components["source_wait"].sum()) == 0.0
    # fake queue per control step = 2 + (i % 3); single transition charged 0.5 * (q1 + q2) = q
    expected = sum(2.0 + (i % 3) for i in range(6))
    assert abs(float(queued.components["source_wait"].sum()) - expected) < 1e-4
    assert torch.allclose(queued.components["total_wait"],
                          queued.components["delay"] + queued.components["source_wait"])
    assert float(queued.rewards.sum()) < float(plain.rewards.sum())


def test_trainer_log_rows_show_source_wait(tmp_path):
    _trainer(tmp_path / "q", env_kwargs={"entry_queue": 1.0}).run()
    rows = _lines(tmp_path / "q" / "log.jsonl")
    episodes = [r for r in rows if r["type"] == "episode"]
    it = [r for r in rows if r["type"] == "iter"][0]
    assert episodes and all(
        r["source_wait"] > 0 and abs(r["total_wait"] - r["delay"] - r["source_wait"]) < 1e-3
        for r in episodes)
    assert it["mean_source_wait_per_step"] > 0 and "mean_total_wait_per_step" in it


# ------------------------------------------------------------------ --init-from
def test_init_from_checkpoint_loads_weights_fresh_optimisers_and_alpha(tmp_path):
    src = _trainer(tmp_path / "src")
    src.run()
    ckpt = tmp_path / "src" / "checkpoints" / "ckpt_00001.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state["lagrange_alpha"] = 1.25               # make the carried multiplier unmistakable
    forged = tmp_path / "forged.pt"
    torch.save(state, forged)

    carry = _trainer(tmp_path / "carry", net_seed=99)
    rec = carry.init_from_checkpoint(forged, "carry")
    for name in ("policy", "group_encoder", "critic"):
        a, b = getattr(carry, name).state_dict(), state[name]
        assert all(torch.equal(a[k], b[k]) for k in b), name
    assert carry.updater.alpha == 1.25 and rec["lagrange_alpha_used"] == 1.25
    assert rec["sha256"] == sha256_file(forged) and rec["kind"] == "trainer_checkpoint"
    assert carry.iteration == 0 and not carry.updater.actor_opt.state_dict()["state"]
    assert not carry.updater.critic_opt.state_dict()["state"]
    assert carry.scaler.state_dict() == type(carry.scaler)().state_dict()

    reset = _trainer(tmp_path / "reset", net_seed=99)
    rec = reset.init_from_checkpoint(forged, "reset")
    assert reset.updater.alpha == 0.0 and rec["lagrange"] == "reset"
    assert all(torch.equal(reset.policy.state_dict()[k], state["policy"][k])
               for k in state["policy"])

    carry.run()                                  # trains from iteration 0 and records the source
    rows = _lines(tmp_path / "carry" / "log.jsonl")
    assert rows[0]["type"] == "config" and rows[0]["meta"]["init_from"]["sha256"] == sha256_file(forged)
    assert [r["iteration"] for r in rows if r["type"] == "iter"] == [1]
    summary = json.loads((tmp_path / "carry" / "RUN_SUMMARY.json").read_text("utf-8"))
    assert summary["meta"]["init_from"]["lagrange"] == "carry"
    with pytest.raises(RuntimeError):            # a directory with checkpoints is refused
        carry.init_from_checkpoint(forged, "carry")
    with pytest.raises(ValueError):
        _trainer(tmp_path / "bad").init_from_checkpoint(forged, "keep")


def test_init_from_exported_actor_keeps_critic_fresh(tmp_path):
    src = _trainer(tmp_path / "src")
    src.run()
    dst = _trainer(tmp_path / "dst", net_seed=5)
    fresh_critic = {k: v.clone() for k, v in dst.critic.state_dict().items()}
    rec = dst.init_from_checkpoint(tmp_path / "src" / "policy_latest.pt", "carry")
    assert rec["kind"] == "exported_actor" and rec["critic"] == "fresh"
    assert all(torch.equal(dst.critic.state_dict()[k], fresh_critic[k]) for k in fresh_critic)
    assert all(torch.equal(dst.policy.state_dict()[k], src.policy.state_dict()[k])
               for k in src.policy.state_dict())
    assert dst.updater.alpha == 0.0 and "fresh" in rec["lagrange_note"]


def test_cli_init_from_rules(tmp_path):
    from wms.control.mappo.train import main

    src = tmp_path / "src"
    assert main(["--config", str(CONFIG_PATH), "--out", str(src), "--smoke",
                 "--iterations", "1"]) == 0
    ckpt = src / "checkpoints" / "ckpt_00001.pt"
    assert ckpt.is_file()
    out = tmp_path / "warm"
    assert main(["--config", str(CONFIG_PATH), "--out", str(out), "--smoke", "--iterations", "1",
                 "--init-from", str(ckpt), "--init-lagrange", "reset"]) == 0
    assert (out / "checkpoints" / "ckpt_00001.pt").is_file()
    assert main(["--config", str(CONFIG_PATH), "--out", str(out), "--smoke",
                 "--init-from", str(ckpt)]) == 2                      # --out has checkpoints
    assert main(["--config", str(CONFIG_PATH), "--out", str(tmp_path / "w2"), "--smoke",
                 "--init-from", str(ckpt), "--resume"]) == 2          # excludes --resume
    status = json.loads((out / "status.json").read_text("utf-8"))
    assert status["init_from"]["lagrange"] == "reset"


# ------------------------------------------------------------------ SUMO parity
@pytest.mark.skipif(shutil.which("sumo") is None, reason="SUMO not installed")
def test_sumo_source_wait_steps_match_window_summary(tmp_path):
    from traffic_shepherd.c4.baselines import run_policy_episode
    from traffic_shepherd.c4.env import CavSumoEnv

    env = CavSumoEnv(physics_log=False, ttc=False, scenario_root=tmp_path / "scn",
                     entry_queue_in_state=True)
    ends: list[int] = []                 # entry_queue_now() at the END of each window physics step
    start0: list[int] = []               # ... and at the window start
    original_after, original_reset = env._after_physics, env.reset
    armed = {"on": False}

    def after(*a, **k):
        out = original_after(*a, **k)
        if armed["on"]:
            ends.append(env.entry_queue_now())
        return out

    def reset(**kw):
        state = original_reset(**kw)
        start0.append(env.entry_queue_now())
        armed["on"] = True
        return state
    env._after_physics, env.reset = after, reset
    seen = []

    def on_step(state, info):
        seen.append(info)
        assert state["entry_queue"].shape == (1,)
        assert abs(float(state["entry_queue"][0]) - info["entry_queue_steps"][-1]) < 1e-5
        assert env.last_entry_queue_steps == info["entry_queue_steps"]

    summary = run_policy_episode(env, "native", seed=4107, demand_vph=2700, penetration=0.2,
                                 warmup_s=60, horizon_s=60, on_step=on_step)
    assert len(seen) == 60
    steps = [w for info in seen for w in info["source_wait_steps"]]
    assert len(steps) == 120 == len(ends)
    ledger = summary["window"]["source_wait_vehicle_seconds"]
    assert abs(sum(steps) - ledger) <= 1e-6 * max(1.0, abs(ledger))
    for info in seen:
        assert info["entry_queue_steps"] == [w / 0.5 for w in info["source_wait_steps"]]
    # integer definition: 0.5 * queue count at each step START (= count at the previous step end)
    integer = 0.5 * sum(start0 + ends[:-1])
    # The ledger additionally charges every request that falls INSIDE a step from its request
    # time to the step end; the integer count only sees it from the next step on.
    partial = sum(math.ceil(r.depart_s * 2 - 1e-9) / 2 - r.depart_s
                  for r in env.requests if 60.0 < r.depart_s <= 120.0)
    print(f"ledger={ledger:.3f} sum(info)={sum(steps):.3f} integer-start={integer:.3f} "
          f"in-step partial={partial:.3f} rel.diff={(ledger - integer) / ledger:.3f}")
    assert integer <= ledger + 1e-9
    assert abs(integer + partial - ledger) <= 1e-6 * max(ledger, 1.0), (integer, partial, ledger)
