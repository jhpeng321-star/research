"""CPU unit tests for the fair universal C4 MAPPO baseline (fake env; no SUMO).

The actor, observations, RSU field, lane rules, safety layer and reward are the REAL
``wms.control`` modules; only the environment is faked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from wms.control.contract import RewardWeights
from wms.control.lanes import LaneRules
from wms.control.mappo.agent import ActorStep, build_networks, evaluate_actions
from wms.control.mappo.buffer import (ReturnScaler, compute_gae, discounted_returns, episode_gae,
                                      flatten_episodes)
from wms.control.mappo.config_helpers import build_trainer, config_from_dict, merge_override
from wms.control.mappo.eval_bridge import load_policy_fn as load_mappo
from wms.control.mappo.fakeenv import FakeCavEnv
from wms.control.mappo.ppo import (PPOConfig, PPOUpdater, clipped_surrogate, clipped_value_loss,
                                   pessimistic_cost_surrogate, update_lagrange)
from wms.control.mappo.rollout import (StopRequested, WorkerPool, collect_episode,
                                       control_step_reward, run_job)
from wms.control.mappo.trainer import (Budget, MappoConfig, MappoTrainer, RolloutConfig,
                                       sha256_file)
from wms.control.policy import PolicyExecutor, load_policy_fn
from wms.control.reward import team_reward
from wms.core.toml import loads

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "configs" / "c4_mappo.toml"


def _networks(seed: int = 0, lane_bias: float | None = 2.0):
    torch.manual_seed(seed)
    policy, encoder, critic = build_networks()
    if lane_bias is not None:       # make lane changes likely so the lane path is exercised
        with torch.no_grad():
            policy.lane_head[-1].bias.copy_(torch.tensor([0.0, lane_bias, lane_bias]))
    return policy, encoder, critic


def _episode(seed: int = 4107, horizon: float = 6.0, policy=None, encoder=None):
    if policy is None:
        policy, encoder, _ = _networks()
    actor = ActorStep(policy, encoder, LaneRules())
    return collect_episode(FakeCavEnv(), actor, weights=RewardWeights(), seed=seed,
                           demand_vph=2600, penetration=0.4, warmup_s=0.0, horizon_s=horizon,
                           noise_seed=3)


def _trainer(out: Path, *, seed: int = 7, iterations: int = 2, episodes: int = 2,
             horizon: float = 5.0, env_kwargs=None, seeds: str = "4107-4148",
             penetration_mode: str = "discrete", retries: int = 1) -> MappoTrainer:
    policy, encoder, critic = _networks(11, lane_bias=None)
    cfg = MappoConfig(
        ppo=PPOConfig(epochs=2, minibatch_frames=4, target_kl=None),
        rollout=RolloutConfig(seed=seed, iterations=iterations, episodes_per_iteration=episodes,
                              horizon_s=horizon, warmup_s=0.0, penetrations=(0.05, 0.4),
                              penetration_mode=penetration_mode, demands_vph=(2400, 2700),
                              seeds=seeds, max_sumo_seconds=1e9, retries=retries))
    return MappoTrainer(out_dir=out, config=cfg, policy=policy, group_encoder=encoder,
                        critic=critic, env_spec={"env_name": "fake", "env_kwargs": env_kwargs or {}})


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


# ----------------------------------------------------------------------------- math
def test_gae_matches_hand_computation():
    adv, ret = compute_gae(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.5, 1.0, 0.5]),
                           gamma=0.9, lam=0.5, bootstrap_value=torch.tensor(2.0))
    assert torch.allclose(adv, torch.tensor([2.92325, 3.385, 4.3]), atol=1e-4)
    assert torch.allclose(ret, adv + torch.tensor([0.5, 1.0, 0.5]), atol=1e-4)


def test_gae_termination_drops_bootstrap():
    batch = flatten_episodes([_episode()])
    values = torch.linspace(0.1, 0.5, batch.n_frames)
    finals = torch.tensor([99.0])
    adv_boot, _, _ = episode_gae(batch, values, finals, gamma=0.9, lam=0.5)
    manual, _ = compute_gae(batch.rewards, values, gamma=0.9, lam=0.5, bootstrap_value=0.0)
    batch.terminated[0] = True
    adv_term, _, _ = episode_gae(batch, values, finals, gamma=0.9, lam=0.5)
    assert torch.allclose(adv_term, manual, atol=1e-6)
    assert not torch.allclose(adv_boot, manual, atol=1e-3)


def test_discounted_returns_and_scaler():
    assert torch.allclose(discounted_returns(torch.ones(3), 0.5), torch.tensor([1.75, 1.5, 1.0]))
    scaler = ReturnScaler()
    scaler.update(torch.arange(1, 5, dtype=torch.float32))
    x = torch.tensor([3.0, 4.0])
    assert abs(scaler.mean - 2.5) < 1e-6
    assert torch.allclose(scaler.unscale(scaler.scale(x)), x, atol=1e-5)
    assert torch.allclose(ReturnScaler.from_state(scaler.state_dict()).scale(x), scaler.scale(x))


def test_surrogates_value_clip_and_lagrange():
    a = torch.tensor([1.0])
    surr = [float(clipped_surrogate(torch.tensor([r]), a, 0.2)) for r in (0.8, 1.0, 1.25, 1.4)]
    assert surr == sorted(surr) and abs(surr[2] - surr[3]) < 1e-6
    # cost surrogate is the pessimistic (upper) bound: never below the unclipped value
    for r in (0.5, 0.9, 1.1, 1.6):
        for c in (-1.0, 1.0):
            ratio = torch.tensor([r])
            assert float(pessimistic_cost_surrogate(ratio, torch.tensor([c]), 0.2)) >= r * c - 1e-6
    assert abs(float(clipped_value_loss(torch.tensor([0.5]), torch.tensor([0.0]),
                                        torch.tensor([1.0]), 0.2)) - 0.32) < 1e-6
    assert update_lagrange(1.0, 0.1, 0.02, 0.05, 50.0) > 1.0
    assert update_lagrange(0.01, 0.0, 0.02, 1.0, 50.0) == 0.0
    assert update_lagrange(49.99, 10.0, 0.0, 1.0, 50.0) == 50.0


# ----------------------------------------------------------------------------- fairness
def test_deterministic_actor_step_equals_policy_executor():
    policy, encoder, _ = _networks()
    rules = LaneRules()
    mine = ActorStep(policy, encoder, rules)
    theirs = PolicyExecutor(policy, encoder, lane_rules=rules, deterministic=True)
    env = FakeCavEnv()
    state = env.reset(seed=4110, demand_vph=2600, penetration=0.4, warmup_s=0.0, horizon_s=8.0)
    changed = 0
    for _ in range(8):
        out = mine(state, deterministic=True)
        accel, lanes = theirs.act_with_lane(state)
        cmd = out["command"].numpy()
        assert ((cmd == accel) | (torch.isnan(out["command"]).numpy() & (accel != accel))).all()
        assert (out["lanes"].numpy() == lanes).all()
        changed += int((lanes >= 0).sum())
        state, _ = env.step(out["command"], lanes=out["lanes"])
    assert changed > 0, "lane path not exercised"


def test_evaluate_actions_reproduces_rollout_log_probs():
    policy, encoder, _ = _networks()
    episode = _episode(policy=policy, encoder=encoder)
    batch = flatten_episodes([episode])
    s = batch.steps
    with torch.no_grad():
        ev = evaluate_actions(policy, encoder, batch.states, s["lane_mask"], s["latent"],
                              s["lane_choice"])
    index, cav = ev["index"], ev["cav"]
    assert torch.equal(cav, s["cav_mask"].gather(1, index))
    assert int(cav.sum()) == int(s["cav_mask"].sum())        # compaction keeps every CAV
    for key in ("logp_accel", "logp_lane"):
        old = s[key].gather(1, index)[cav]
        assert torch.allclose(ev[key][cav], old, atol=2e-4), key
    assert float(s["logp_lane"].abs().sum()) > 0              # some feasible lane choices
    assert (ev["entropy_lane"][cav] >= -1e-6).all()
    # the PPO path additionally drops absent slots first: still exact
    updater = PPOUpdater(policy=policy, group_encoder=encoder, critic=build_networks()[2],
                         config=PPOConfig())
    with torch.no_grad():
        terms = updater._policy_terms(batch, torch.arange(batch.n_frames))
        assert terms["new"].numel() == int(s["cav_mask"].sum())
        assert torch.allclose(terms["new"], terms["old"], atol=2e-4)
        values = updater.critic_values(batch.states)
        full = updater.critic(batch.states)
    assert torch.allclose(values, full, atol=1e-4)


def test_control_step_reward_sums_physics_steps_and_carries_exits():
    env = FakeCavEnv(k_slots=8, min_vehicles=8)
    s0 = env.reset(seed=4111, demand_vph=2600, penetration=0.4, warmup_s=0.0, horizon_s=4.0)
    s1, _ = env.step(torch.zeros(1, 8))
    s2, _ = env.step(torch.zeros(1, 8))

    class History:            # CavSumoEnv.history.raw(3) layout; slot 3 leaves at t + 1
        def __len__(self):
            return 3

        def raw(self, n):
            import numpy as np

            def kin(st):
                out = np.zeros((8, 7), np.float32)
                out[:, 0] = st["s_m"][0].numpy()
                for lane in range(3):
                    out[:, 1 + lane] = (st["lane"][0] == lane).numpy()
                out[:, 4], out[:, 5], out[:, 6] = st["v_mps"][0].numpy(), st["a_mps2"][0].numpy(), 5.0
                return out
            vid = np.tile(np.arange(8, dtype=np.int32), (3, 1))
            vid[2, 3] = -1
            mask = np.stack([s0["present"][0].numpy(), s1["present"][0].numpy(),
                             s2["present"][0].numpy()])
            mask[2, 3] = False
            return {"kin": np.stack([kin(s0), kin(s1), kin(s2)]), "mask": mask,
                    "target_mask": (vid[:-1] == vid[1:]) & (vid[:-1] >= 0), "vid_index": vid}

    env.history = History()
    final = dict(s2)
    final["present"] = s2["present"].clone()
    final["present"][0, 3] = False
    w = RewardWeights()
    got = control_step_reward(env, s0, final, w)
    carried = dict(final)
    carried["s_m"] = final["s_m"].clone()
    carried["v_mps"] = final["v_mps"].clone()
    carried["a_mps2"] = final["a_mps2"].clone()
    carried["s_m"][0, 3] = s1["s_m"][0, 3] + 0.5 * s1["v_mps"][0, 3]
    carried["v_mps"][0, 3] = s1["v_mps"][0, 3]
    carried["a_mps2"][0, 3] = s1["a_mps2"][0, 3]
    mid = {**s1, "length_m": s0["length_m"]}
    first = team_reward(s0, mid, torch.where(s0["present"], mid["a_mps2"], 0.0), w)
    second = team_reward(mid, carried, torch.where(mid["present"], carried["a_mps2"], 0.0), w)
    for key in ("reward", "delay", "cost_ttc", "fairness_delay"):
        assert abs(got[key] - float(first[key][0] + second[key][0])) < 1e-4, key


# ----------------------------------------------------------------------------- PPO
def test_ppo_update_moves_joint_logp_toward_advantage():
    policy, encoder, critic = _networks(3)
    batch = flatten_episodes([_episode(policy=policy, encoder=encoder, horizon=8.0)])
    updater = PPOUpdater(policy=policy, group_encoder=encoder, critic=critic,
                         config=PPOConfig(epochs=4, minibatch_frames=4, lr_actor=3e-3, target_kl=None))
    n = batch.n_frames
    adv = torch.linspace(-1.0, 1.0, n)
    diag = updater.update(batch, advantages=adv, returns=torch.zeros(n), cost_returns=torch.zeros(n),
                          scaler=ReturnScaler(), grad_noise=True)
    assert diag["finite"] and diag["pg_grad_minibatches"] >= 2
    assert diag["pg_grad_var_trace"] >= 0 and diag["pg_grad_norm_mean"] > 0
    s = batch.steps
    with torch.no_grad():
        ev = evaluate_actions(policy, encoder, batch.states, s["lane_mask"], s["latent"],
                              s["lane_choice"])
    cav, index = ev["cav"], ev["index"]
    new = (ev["logp_accel"] + ev["logp_lane"])[cav]
    old = (s["logp_accel"] + s["logp_lane"]).gather(1, index)[cav]
    frame = cav.nonzero()[:, 0]
    a = adv[frame] - adv[frame].mean()
    corr = float(((new - old) * a).sum() / ((new - old).norm() * a.norm() + 1e-8))
    assert corr > 0.1, corr


# ----------------------------------------------------------------------------- trainer
def test_two_iteration_run_artifacts_and_deploy_load(tmp_path):
    out = tmp_path / "run"
    trainer = _trainer(out)
    init = {k: v.clone() for k, v in trainer.policy.state_dict().items()}
    result = trainer.run()
    assert result["status"] == "completed" and result["iterations_completed"] == 2
    lines = _lines(out / "log.jsonl")
    iters = [x for x in lines if x["type"] == "iter"]
    assert lines[0]["type"] == "config" and len(iters) == 2
    for row in iters:
        for key in ("sumo_seconds_used", "simulated_s", "env_steps", "mean_episode_return",
                    "pg_grad_var_trace", "pg_grad_snr", "actor_grad_norm", "policy_lane_change_rate",
                    "sim_s_per_wall_s_per_worker", "lagrange_alpha"):
            assert key in row, key
    assert iters[1]["budget"]["used_sumo_seconds"] == 4 * 5.0
    assert iters[1]["budget"]["used_env_steps"] == 20
    final = trainer.policy.state_dict()
    assert any(not torch.allclose(init[k], final[k]) for k in init)
    # deploy path: the exported actor loads with the C4 loader, act_with_lane works
    exe = load_policy_fn(out / "policy_latest.pt")
    assert exe.has_lane_head
    state = FakeCavEnv().reset(seed=4107, demand_vph=2400, penetration=0.4, warmup_s=0, horizon_s=5)
    accel, lanes = exe.act_with_lane(state)
    assert accel.shape == lanes.shape == (1, state["v_mps"].shape[1])
    for key, value in exe.policy.state_dict().items():
        assert torch.equal(value, final[key])
    latest = json.loads((out / "checkpoints" / "LATEST").read_text(encoding="utf-8"))
    exe2 = load_mappo(out / latest["file"])
    assert torch.equal(exe2.policy.lane_head[-1].weight, final["lane_head.2.weight"])


def test_resume_matches_straight_run(tmp_path):
    straight = _trainer(tmp_path / "straight")
    straight.run()
    _trainer(tmp_path / "split").run(1)
    resumed = _trainer(tmp_path / "split")
    assert resumed.run(1)["iterations_completed"] == 2
    for name in ("policy", "group_encoder", "critic"):
        a, b = getattr(straight, name).state_dict(), getattr(resumed, name).state_dict()
        assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a), name
    assert straight.budget.used_sumo_seconds == resumed.budget.used_sumo_seconds
    assert [x["type"] for x in _lines(tmp_path / "split" / "log.jsonl")
            if x["type"] != "episode"] == ["config", "iter", "resume", "iter"]


def test_rolling_checkpoints_pointer_and_status(tmp_path):
    out = tmp_path / "roll"
    trainer = _trainer(out, iterations=5, episodes=1, horizon=3.0)
    trainer.cfg.rollout.keep_last, trainer.cfg.rollout.keep_every = 3, 4
    trainer.run()
    names = sorted(p.name for p in (out / "checkpoints").glob("ckpt_*.pt"))
    assert names == ["ckpt_00003.pt", "ckpt_00004.pt", "ckpt_00005.pt"]
    assert not list((out / "checkpoints").glob("*.tmp"))
    latest = json.loads((out / "checkpoints" / "LATEST").read_text(encoding="utf-8"))
    assert latest["file"] == "checkpoints/ckpt_00005.pt" and latest["iteration"] == 5
    assert latest["sha256"] == sha256_file(out / latest["file"])
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["iteration"] == 5 and status["last_checkpoint"]["sha256"] == latest["sha256"]
    assert status["sumo_seconds_charged"] == 15.0 and "log.jsonl" in status["files"]
    assert (out / "policy_00004.pt").is_file()


def test_resume_skips_truncated_checkpoint_and_orphans_log_tail(tmp_path):
    straight = _trainer(tmp_path / "straight", iterations=3)
    straight.run()
    out = tmp_path / "crash"
    _trainer(out, iterations=3).run()
    # simulate a crash while ckpt 3 was being written: truncated file, stale pointer
    ckpt3 = out / "checkpoints" / "ckpt_00003.pt"
    ckpt3.write_bytes(ckpt3.read_bytes()[:100])
    resumed = _trainer(out, iterations=1)
    result = resumed.run()
    assert result["iterations_completed"] == 3
    rows = _lines(out / "log.jsonl")
    resume = [r for r in rows if r["type"] == "resume"][0]
    assert resume["checkpoint"] == "ckpt_00002.pt"
    assert resume["skipped"][0]["reason"] == "sha256 mismatch"
    assert resume["log"]["orphan_bytes"] > 0 and (out / resume["log"]["orphan_file"]).is_file()
    assert [r["iteration"] for r in rows if r["type"] == "iter"] == [1, 2, 3]
    for name in ("policy", "group_encoder", "critic"):
        a, b = getattr(straight, name).state_dict(), getattr(resumed, name).state_dict()
        assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a), name


def test_stop_request_abandons_iteration_and_resumes_exactly(tmp_path):
    out = tmp_path / "stop"

    class Reclaimed(MappoTrainer):
        def _collect(self):
            if self.iteration == 1:         # spot notice arrives during iteration 2
                self.stop_path.write_text("spot termination notice: test", encoding="ascii")
            return super()._collect()

    first = _trainer(out)
    first.__class__ = Reclaimed
    result = first.run()
    assert result["status"] == "stopped_by_request" and result["iterations_completed"] == 1
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["stopped"] and status["last_checkpoint"]["iteration"] == 1
    assert [r["type"] for r in _lines(out / "log.jsonl")][-1] == "stopped"
    resumed = _trainer(out)                       # new session: the stale stop file is consumed
    assert resumed.run(1)["iterations_completed"] == 2
    assert list(out.glob("STOP_REQUESTED.consumed_*")) and not (out / "STOP_REQUESTED").exists()
    straight = _trainer(tmp_path / "straight")
    straight.run()
    for name in ("policy", "group_encoder", "critic"):
        a, b = getattr(straight, name).state_dict(), getattr(resumed, name).state_dict()
        assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a), name


def test_cli_exit_code_and_resume_flag(tmp_path):
    from wms.control.mappo.train import main

    out = tmp_path / "cli_stop"
    assert main(["--config", str(CONFIG_PATH), "--out", str(out), "--smoke", "--resume"]) == 2
    out.mkdir(parents=True)
    (out / "STOP_REQUESTED").write_text("x", encoding="ascii")   # stale: consumed, run proceeds
    assert main(["--config", str(CONFIG_PATH), "--out", str(out), "--smoke"]) == 0


def test_universal_sampling_and_seed_block(tmp_path):
    trainer = _trainer(tmp_path / "u", penetration_mode="uniform")
    cases = [trainer._sample_case() for _ in range(200)]
    assert all(0.05 <= c["penetration"] <= 0.4 for c in cases)
    assert len({round(c["penetration"], 6) for c in cases}) > 100
    assert {c["demand_vph"] for c in cases} == {2400, 2700}
    assert all(4107 <= c["seed"] <= 4148 for c in cases)
    discrete = _trainer(tmp_path / "d")
    assert {discrete._sample_case()["penetration"] for _ in range(200)} == {0.05, 0.4}


def test_failed_episode_is_retried_with_new_seed(tmp_path):
    out = tmp_path / "fail"
    trainer = _trainer(out, iterations=1, episodes=4, seeds="4107,4108",
                       env_kwargs={"fail_seeds": [4107]})
    trainer.run()
    failures = [x for x in _lines(out / "log.jsonl") if x["type"] == "failure"]
    assert failures, "expected at least one failing seed-4107 episode"
    assert all(f["job"]["seed"] == 4107 and f["retry"]["seed"] == 4108 for f in failures)
    episodes = [x for x in _lines(out / "log.jsonl") if x["type"] == "episode"]
    assert len(episodes) == 4 and all(e["seed"] == 4108 for e in episodes)
    assert trainer.budget.wasted_sim_seconds == 5.0 * len(failures)


def test_budget_blocks_oversized_episodes():
    budget = Budget(max_sumo_seconds=13.0, max_env_steps=None)
    assert budget.can_start(6.0, 0.0, False)
    budget.used_sumo_seconds = 8.0
    assert not budget.can_start(6.0, 0.0, False) and budget.can_start(5.0, 0.0, False)
    charged = Budget(max_sumo_seconds=100.0, max_env_steps=None)
    episode = _episode(horizon=4.0)
    charged.charge(episode, warmup_s=2.0, charge_warmup=False)
    assert charged.used_sumo_seconds == 4.0 and charged.simulated_seconds == 6.0
    charged.charge(episode, warmup_s=2.0, charge_warmup=True)
    assert charged.used_sumo_seconds == 10.0 and charged.used_env_steps == 8


def test_run_job_reports_failure_instead_of_raising():
    policy, encoder, _ = _networks()
    spec = {"env_name": "fake", "env_kwargs": {"fail_seeds": [4120]}}
    job = {"spec": spec, "policy_state": policy.state_dict(), "encoder_state": encoder.state_dict(),
           "seed": 4120, "demand_vph": 2600, "penetration": 0.1, "warmup_s": 0.0, "horizon_s": 3.0}
    row = run_job(job)
    assert not row["ok"] and "fake SUMO failure" in row["error"]
    ok = run_job({**job, "seed": 4121})
    assert ok["ok"] and ok["episode"].n_steps == 3


def test_worker_pool_spawn_survives_worker_death():
    policy, encoder, _ = _networks()
    spec = {"env_name": "fake", "env_kwargs": {"exit_seeds": [4130]},
            "lane_rules": {}, "reward_weights": {}}
    base = {"policy_state": policy.state_dict(), "encoder_state": encoder.state_dict(),
            "demand_vph": 2600, "penetration": 0.2, "warmup_s": 0.0, "horizon_s": 3.0}
    pool = WorkerPool(2, spec)
    try:
        good = pool.run([{**base, "seed": 4131}, {**base, "seed": 4132}], deadline_s=240)
        assert all(r["ok"] for r in good)
        mixed = pool.run([{**base, "seed": 4130}, {**base, "seed": 4133}], deadline_s=240)
        assert not mixed[0]["ok"]
        assert pool.restarts >= 1
        again = pool.run([{**base, "seed": 4134}], deadline_s=240)
        assert again[0]["ok"] and again[0]["episode"].n_steps == 3
        restarts = pool.restarts
        with pytest.raises(StopRequested):
            pool.run([{**base, "seed": 4135, "horizon_s": 50.0}], stop_check=lambda: True)
        assert pool.restarts == restarts + 1 and pool.executor is None
    finally:
        pool.close()


# ----------------------------------------------------------------------------- config / CLI
def test_config_toml_defaults_and_smoke_merge(tmp_path):
    cfg = loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parsed = config_from_dict(cfg)
    assert parsed.ppo.gamma == 0.99 and parsed.ppo.gae_lambda == 0.95
    assert parsed.ppo.cost_limit_per_step == 3.6 and parsed.ppo.lagrange_lr == 0.01
    assert parsed.rollout.penetrations == (0.05, 0.1, 0.2, 0.4)
    assert parsed.rollout.demands_vph == (2400, 2600, 2700)
    assert parsed.rollout.seeds == "4107-4148" and parsed.rollout.max_env_steps is None
    assert parsed.lane_rules == LaneRules()
    policy_cfg = loads((REPO / "configs" / "c4_policy.toml").read_text(encoding="utf-8"))
    lanes = {k: v for k, v in policy_cfg["lanes"].items() if k != "enabled"}
    assert parsed.lane_rules == LaneRules.from_mapping(lanes)
    merged = merge_override(cfg, cfg["smoke"])
    assert config_from_dict(merged).rollout.horizon_s == 6.0
    result = build_trainer(merged, workers=1, out_dir=tmp_path / "cfg").run()
    assert result["status"] == "completed" and result["budget"]["used_sumo_seconds"] <= 64.0


def test_cli_smoke_on_real_config(tmp_path):
    from wms.control.mappo.train import main

    out = tmp_path / "cli"
    assert main(["--config", str(CONFIG_PATH), "--out", str(out), "--smoke",
                 "--max-sumo-seconds", "24"]) == 0
    summary = json.loads((out / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["budget"]["used_sumo_seconds"] <= 24.0
    assert summary["budget"]["max_sumo_seconds"] == 24.0
    assert (out / "policy_latest.pt").is_file()
