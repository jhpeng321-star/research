"""CPU-only RSU checks, with paired fake WMs and a synthetic WM-S benchmark."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from wms.control.contract import RewardWeights
from wms.control.rsu_check import RsuChecker, default_candidates


def state(slots=3):
    s = torch.arange(slots, dtype=torch.float32)[None] * 40 - 300
    return {"s_m": s, "v_mps": torch.full_like(s, 15), "a_mps2": torch.zeros_like(s),
            "lane": torch.arange(slots)[None] % 3, "present": torch.ones_like(s, dtype=torch.bool),
            "is_cav": torch.ones_like(s, dtype=torch.bool), "length_m": torch.full_like(s, 5)}


def factory(accel, created=None):
    def build():
        calls = []
        if created is not None:
            created.append(calls)

        def controller(current):
            assert not torch.is_grad_enabled()
            calls.append(current["s_m"].shape)
            a = torch.full_like(current["s_m"], accel)
            v = (current["v_mps"] + a * 0.5).clamp_min(0)
            return torch.stack((current["s_m"] + (current["v_mps"] + v) * 0.25,
                                current["lane"].float(), v, a), -1)
        return controller
    return build


def reward(current, nxt, accel, weights):
    return {"reward": nxt["v_mps"].mean(-1)}


class FakeWM(torch.nn.Module):
    def __init__(self, gains=None, offset=0):
        super().__init__()
        self.physics = SimpleNamespace(dt_s=0.5)
        self.gains, self.offset, self.calls = gains, offset, []
        self.dropout = torch.nn.Dropout()

    def encode_drivers(self, current, **kwargs):
        assert not torch.is_grad_enabled()
        self.encoded_history = kwargs
        values = torch.ones(*current["s_m"].shape, 1) * 10
        return {"uncertainty": torch.ones_like(values)}, None, {"v0": values[..., 0]}

    def rollout(self, current, *, n_steps, cav_controller, driver_params, **kwargs):
        assert not torch.is_grad_enabled() and not self.training
        assert kwargs["cav_teacher"] is None
        self.calls.append((current, driver_params, kwargs, n_steps))
        motion = cav_controller({**current, "physics_step_index": 0})
        common = driver_params["v0"] + self.offset
        gains = motion[..., 3]
        if self.gains is not None:
            gains = torch.as_tensor(self.gains, dtype=gains.dtype)[:, None].expand_as(gains)
        speeds = common + gains
        def repeat(x):
            return x[:, None].expand(-1, n_steps, -1).clone()

        return {"positions": repeat(current["s_m"]), "speeds": repeat(speeds),
                "lanes": repeat(current["lane"]), "masks": repeat(current["present"]),
                "accel": repeat(motion[..., 3])}


def checker(models=None, candidates=None, **kwargs):
    # sampling-mechanics tests: the pre-2026-10-13 sampled-driver setup (deploy default is now the
    # point estimate, n_driver_samples=1; see test_c4_futures)
    if kwargs.get("n_driver_samples", 4) > 1 and "driver_noise_scale" not in kwargs:
        kwargs.setdefault("n_driver_samples", 4)
        kwargs.update(driver_noise_scale=1.0, embedding_noise_std=kwargs.get("embedding_noise_std", 0.05))
    return RsuChecker(models or [FakeWM()], candidates or {"policy": factory(0), "native": factory(1)},
                      reward, RewardWeights(), **kwargs)


@pytest.mark.parametrize("gain,margin,choice", [(1, 39, "native"), (1, 40, "policy"),
                                                (1, 41, "policy"), (0, 0, "policy"), (-1, 0, "policy")])
def test_paired_margin(gain, margin, choice):
    # point-estimate drivers; the paired spread comes from a two-member WM ensemble
    c = checker(models=[FakeWM(), FakeWM()], candidates={"policy": factory(0), "native": factory(gain)},
                margin=margin, n_driver_samples=1)
    chosen, stats = c.decide(state())
    assert chosen == choice
    assert stats["native"]["paired_mean"] == pytest.approx(40 * gain, abs=1e-4)
    assert stats["native"]["paired_std"] < 1e-4


def test_positive_mean_but_nonpositive_lcb_keeps_policy():
    c = checker(models=[FakeWM(gains=[0, 0, 0, 0, -10, -10, -10, 40])])
    chosen, stats = c.decide(state())
    assert stats["native"]["paired_mean"] > 0
    assert stats["native"]["paired_lcb"] < 0
    assert chosen == "policy"


def test_ensemble_lcb_and_best_eligible_candidate():
    c = checker(models=[FakeWM(offset=100), FakeWM(offset=-100)],
                candidates={"native": factory(1), "policy": factory(0), "pi": factory(2)})
    chosen, stats = c.decide(state())
    assert chosen == "pi"
    assert stats["pi"]["paired_mean"] == pytest.approx(80)
    assert stats["pi"]["std"] > 1000
    assert stats["pi"]["paired_std"] < 1e-3
    assert len(stats["pi"]["paired_diffs"]) == 2


def test_crn_pairing_batch_shapes_history_no_grad_and_eval_restore():
    models = [FakeWM(), FakeWM(offset=20)]
    models[0].dropout.eval()
    candidates = {name: factory(index) for index, name in enumerate(
        ["policy", "policy_group_0", "native", "damping", "pi"])}
    c = checker(models=models, candidates=candidates)
    initial = state(192)
    original = {key: value.clone() for key, value in initial.items()}
    history = torch.ones(1, 192, 6, 6, requires_grad=True)
    c.decide(initial, history, driver_history=history)
    for model in models:
        assert len(model.calls) == 1
        batched, drivers, kwargs, steps = model.calls[0]
        assert batched["s_m"].shape == (20, 192) and steps == 40
        assert kwargs["history"].shape == (20, 192, 6, 6)
        assert kwargs["history_mask"].all() and kwargs["driver_history_mask"].all()
        assert not kwargs["history"].requires_grad
        draws = drivers["v0"].reshape(5, 4, 192)
        assert all(torch.equal(draws[0], draws[k]) for k in range(1, 5))
        assert not torch.equal(draws[0, 0], draws[0, 1])
        assert model.training
    assert not models[0].dropout.training
    assert torch.equal(models[0].calls[0][1]["v0"], models[1].calls[0][1]["v0"])
    assert all(torch.equal(initial[key], value) for key, value in original.items())
    json.dumps(c.decision_log, allow_nan=False)


def test_choice_stability_period_execution_isolation_and_episode_reset():
    created = []
    c = checker(candidates={"policy": factory(0), "native": factory(1, created)}, decision_period_s=5)
    initial = state()
    initial["is_cav"][0, 1] = False
    actions = [c.rsu_policy_fn(initial) for _ in range(11)]
    assert len(c.decision_log) == 3
    assert [record["time_s"] for record in c.decision_log] == [0, 5, 10]
    assert all(torch.equal(a, torch.tensor([[1., 0., 1.]])) for a in actions)
    assert sorted(map(len, created)) == [1, 1, 1, 11]
    with pytest.raises(ValueError, match="strictly increasing"):
        c.rsu_policy_fn(initial, now_s=9)
    c.reset()
    c.rsu_policy_fn(initial, now_s=0)
    assert len(c.decision_log) == 4


def test_switch_controller_at_next_decision_only():
    model = FakeWM()
    c = checker(models=[model], decision_period_s=2)
    assert c.rsu_policy_fn(state(), now_s=0)[0, 0] == 1
    model.gains = [0] * 4 + [-1] * 4
    assert c.rsu_policy_fn(state(), now_s=1)[0, 0] == 1
    assert c.rsu_policy_fn(state(), now_s=2)[0, 0] == 0
    assert [r["choice"] for r in c.decision_log] == ["native", "policy"]


def test_runner_explicit_clock_and_batch_contract():
    ticks = []

    def build():
        base = factory(0)()

        def controller(current):
            if "history" in current:
                ticks.append(current["physics_step_index"])
            return base(current)
        return controller

    c = checker(candidates={"policy": build})
    for timestamp in [0, 1.5, 3, 5]:
        c.rsu_policy_fn(state(), now_s=timestamp)
    assert ticks == [0, 3, 6, 10]
    assert len(c.decision_log) == 2
    with pytest.raises(ValueError, match="single state"):
        c.rsu_policy_fn({**state(), "s_m": torch.ones(2, 3)}, now_s=6)


def test_nonfinite_and_single_sample_evidence_fail_closed():
    c = checker(models=[FakeWM(gains=[0] * 4 + [float("nan")] * 4)])
    chosen, stats = c.decide(state())
    assert chosen == "policy" and stats["native"]["mean"] is None
    json.dumps(c.decision_log, allow_nan=False)
    assert checker(n_driver_samples=1).decide(state())[0] == "policy"


def test_embedding_only_sampling_is_paired_and_encoder_is_eval():
    class EmbeddingWM(FakeWM):
        def encode_drivers(self, current, **kwargs):
            assert not self.training and not torch.is_grad_enabled()
            return None, torch.ones(*current["s_m"].shape, 8), None

        def rollout(self, current, *, driver_embedding, **kwargs):
            self.sampled_embedding = driver_embedding
            return super().rollout(current, driver_params={"v0": driver_embedding[..., 0]}, **kwargs)

    model = EmbeddingWM()
    c = checker(models=[model])
    assert c.decide(state())[0] == "native"
    embedding = model.sampled_embedding.reshape(2, 4, 3, 8)
    assert torch.equal(embedding[0], embedding[1])
    assert not torch.equal(embedding[0, 0], embedding[0, 1])
    assert model.training


def test_failure_restores_model_modes_and_does_not_log_a_decision():
    model = FakeWM()
    c = checker(models=[model], candidates={"policy": lambda: lambda s: torch.zeros(1)})
    with pytest.raises(ValueError, match="CavController"):
        c.decide(state())
    assert model.training and not c.decision_log


def test_identical_candidates_always_keep_policy_and_logs_stream():
    records = []
    c = checker(candidates={"native": factory(0), "policy": factory(0)}, log_fn=records.append)
    for _ in range(3):
        assert c.decide(state())[0] == "policy"
    assert len(records) == 3 and records[0]["oracle_state"]


def test_default_candidates_force_deterministic_and_group_alternatives():
    seen = []

    def actor(obs, signal, *, deterministic=False):
        assert deterministic
        seen.append(signal.clone())
        return signal, None

    def bridge(actor, encoder, safety):
        def run(current):
            signal = encoder(current)
            actor(current, signal, deterministic=False)
            return factory(0)()(current)
        return run

    candidates = default_candidates(None, policy=actor,
                                    group_encoder=lambda s: torch.ones(1, 8), safety=lambda s, a: a,
                                    bridge=bridge,
                                    alternative_group_encoders=[lambda s: torch.full((1, 8), 3.)])
    # direct-output candidates only (no native / damping / pi).
    assert list(candidates) == ["policy", "policy_group_0", "policy_group_1", "policy_group_2",
                                "hold_speed"]
    with torch.no_grad():
        for name in list(candidates)[:4]:
            candidates[name]()(state())
    assert torch.equal(seen[0], torch.ones(1, 8))
    assert torch.equal(seen[1], torch.zeros(1, 8))
    assert not torch.equal(seen[2], seen[0])
    assert torch.equal(seen[3], torch.full((1, 8), 3.))


def test_policy_sample_candidates_share_noise_across_members_and_replay_in_execution():
    """policy_sample_k: same draws for every fresh controller (WM member) and for SUMO replay."""
    from wms.control.rsu_check import NoiseStream

    draws = []

    def policy(obs, signal, *, deterministic=False, noise=None):
        assert not deterministic and noise is not None
        draws.append(noise.accel.clone())
        return noise.accel, noise.accel

    def bridge(actor, encoder, safety):
        def run(current):
            actor(SimpleNamespace(cav_mask=torch.ones(2, 3, dtype=torch.bool),
                                  own=torch.zeros(2, 3, 1)), encoder(current))
            return factory(0)()(current)
        return run

    stream = NoiseStream(7, 0, k_full=10)
    stream.begin(3, slot_ids=torch.tensor([0, 4, 9]))
    candidates = default_candidates(None, policy=policy, group_encoder=lambda s: torch.ones(1, 8),
                                    safety=lambda s, a: a, bridge=bridge, sample_streams=[stream])
    assert "policy_sample_0" in candidates and candidates["hold_speed"]
    with torch.no_grad():
        for _ in range(2):                         # two WM members: fresh controllers
            run = candidates["policy_sample_0"]()
            run(state())
            run(state())
    torch.testing.assert_close(draws[0], draws[2])  # member 1 step 0 == member 2 step 0
    torch.testing.assert_close(draws[1], draws[3])
    assert not torch.equal(draws[0], draws[1])      # steps differ
    torch.testing.assert_close(draws[0][0], draws[0][1])   # same plan for every driver sample
    full = stream.noise(0, (1, 10), slot_ids=torch.arange(10))   # SUMO replay over all slots
    torch.testing.assert_close(full.accel[0, [0, 4, 9]], draws[0][0])


@pytest.mark.parametrize("kwargs", [{"world_models": []}, {"candidates": {"native": factory(0)}},
    {"horizon_s": 9}, {"horizon_s": float("nan")}, {"decision_period_s": 0},
    {"n_driver_samples": 0}, {"n_driver_samples": 1.5}, {"margin": -1}, {"confidence_z": -1}])
def test_configuration_validation(kwargs):
    args = dict(world_models=[FakeWM()], candidates={"policy": factory(0)},
                reward_fn=reward, weights=RewardWeights())
    with pytest.raises(ValueError):
        RsuChecker(**{**args, **kwargs})


def test_real_wms_driver_sampling_and_rollout_smoke():
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel

    torch.manual_seed(1)
    model = WmSModel(ModelSpec(rollout_mode="hybrid"))
    c = checker(models=[model])
    initial = state(6)
    initial["is_cav"][:, 1::2] = False
    chosen, stats = c.decide(initial, torch.ones(1, 6, model.history_len, model.history_features))
    assert chosen in c.candidates
    assert all(math_value is not None for item in stats.values()
               for math_value in (item["mean"], item["std"]))
    assert c.last_timing["steps_per_member"] == 40


def benchmark(repeats=3, warmup=True):
    """Synthetic, randomly initialized hybrid WM-S; no data/checkpoint access."""
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel

    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = WmSModel(ModelSpec(rollout_mode="hybrid")).eval()
    initial = state(192)
    initial["is_cav"][:] = False
    initial["is_cav"][:, ::5] = True
    initial["s_m"] = torch.arange(192)[None].float() * 12 - 1800
    candidates = {name: factory(a) for name, a in
                  [("policy", 0), ("policy_group_0", .2), ("native", 0), ("damping", -.2), ("pi", .4)]}
    c = checker(models=[model], candidates=candidates)
    history = torch.zeros(1, 192, model.history_len, model.history_features)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if warmup:
        c.decide(initial, history)
    timings = []
    for _ in range(repeats):
        c.decide(initial, history)
        timings.append(dict(c.last_timing))
    print(json.dumps({"benchmark": "synthetic hybrid WM-S CPU", "torch": torch.__version__,
                      "threads": torch.get_num_threads(), "warmup": warmup, "measurements": timings,
                      "median_total_ms": sorted(x["total_ms"] for x in timings)[len(timings) // 2],
                      "median_rollout_ms": sorted(x["rollout_ms"] for x in timings)[len(timings) // 2]}))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CPU-only synthetic RSU timing")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    benchmark(args.repeats, not args.no_warmup)
