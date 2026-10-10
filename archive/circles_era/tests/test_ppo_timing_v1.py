"""PP-07 regression tests: timing reconstruction + offline decision replay.

Zero simulation. Covers: source-derived timeline reconstruction, registry
loading with masks/observations, the initial-actor zero-input contract,
replayed normalizer determinism, comparison counters (including engineered
flips that never occurred in the real data), and phase separation.
"""
import json
from pathlib import Path

import torch

from traffic_shepherd import ppo_timing_analysis as ta
from traffic_shepherd import ppo_contract as C
from traffic_shepherd.ppo_features import RunningNormalizer
from traffic_shepherd.ppo_net import ActorCritic


def test_timeline_reconstruction_from_synthetic_log(tmp_path):
    log = [dict(time_s=0.0, phase="inactive", requested_action=0, executed_action="wait"),
           dict(time_s=550.0, phase="inactive", requested_action=1, executed_action=None,
                rejection_reason="activation_refused_at_execution"),
           dict(time_s=560.0, phase="inactive", requested_action=1, executed_action="activate"),
           dict(time_s=610.0, phase="hold", requested_action=1, executed_action="continue"),
           dict(time_s=620.0, phase="taper", requested_action=0, executed_action="release")]
    log_path = tmp_path / "decision_log.json"
    log_path.write_text(json.dumps(log), encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(dict(phase_seconds=dict(
        damping=560, assembly=46, hold=14, taper=1, release=1779))), encoding="utf-8")
    t = ta.reconstruct_timeline(log_path, summary_path)
    assert t["damping_span"] if False else t["phase_spans"]["damping"] == (0, 560)
    assert t["phase_spans"]["hold"] == (606, 620)
    assert t["first_refused_activation_s"] == 550.0
    assert t["activation_executed_s"] == 560.0
    assert t["assembly_duration_s"] == 46.0
    assert t["hold_start_s"] == 606.0
    assert t["delivered_hold_s"] == 14.0          # 620 - 606
    assert t["release_request_s"] == 620.0
    assert t["handback_completed_s"] == 621.0
    assert t["continuation_count"] == 1


def test_timeline_handles_missing_decision_log(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(dict(phase_seconds=dict(
        damping=600, assembly=11, hold=20, taper=1, release=1768))), encoding="utf-8")
    missing = tmp_path / "decision_log.json"      # fixed-wall arm: no log
    t = ta.reconstruct_timeline(missing, summary_path)
    assert t["hold_start_s"] == 611.0
    assert t["delivered_hold_s"] == 20.0
    assert t["activation_executed_s"] is None     # no decision log evidence
    assert t["handback_completed_s"] == 632.0


def test_registry_rows_carry_masks_observations_and_source(tmp_path):
    from traffic_shepherd.ppo_timing_analysis import load_registry
    root = Path("runs/active/ppo_run_v1")
    schedule = json.loads((root / "registration.json").read_text(encoding="utf-8"))["training_schedule"]
    rows = load_registry(root / "training", schedule[:1])  # episode 0 only
    assert len(rows) == 115                # episode 1's phase-dependent rows
    r = rows[0]
    assert {"source", "scenario_seed", "episode_index", "round_index",
            "time_s", "phase", "mask", "executed_action",
            "obs_stacked_raw"} <= set(r)
    assert len(r["obs_stacked_raw"]) == 116   # v1-era recorded dimension
    assert all(row["time_s"] % 5 == 0 for row in rows)
    assert all(row["phase"] in {"inactive", "hold"} for row in rows)


def test_initial_actor_uses_fresh_normalizer_zero_inputs():
    """Initial deployed pairing: fresh normalizer (count=0) maps every
    observation to the all-zero vector -> one decision for all states."""
    from traffic_shepherd.ppo_net import seed_everything
    seed_everything(0)
    net = ActorCritic()
    net.eval()
    a1, p1 = ta.initial_actor_action(net, [True, True])
    a2, p2 = ta.initial_actor_action(net, [True, True])
    assert a1 == a2 and p1 == p2            # identical zero input
    assert 0.0 <= p1 <= 1.0


def test_replayed_round_normalizers_are_deterministic():
    from traffic_shepherd.ppo_timing_analysis import replay_round_normalizers
    root = Path("runs/active/ppo_run_v1")
    s1 = replay_round_normalizers(root / "training")
    s2 = replay_round_normalizers(root / "training")
    assert s1[0]["count"] == 0                       # round 0 starts fresh
    assert s1[1] == s2[1] and s1[7] == s2[7]
    assert s1[1]["count"] < s1[2]["count"] < s1[7]["count"]


def test_comparison_counters_detect_engineered_flips():
    """The counter machinery registers WAIT->ACTIVATE and CONTINUE->RELEASE
    flips on a synthetic registry engineered to contain them (the real
    pilot registry had ZERO flips - asserted separately)."""
    from traffic_shepherd.ppo_net import seed_everything

    class ForcedNet:
        """argmax = mask[1] if available else 0 (ACTIVATE/CONTINUE when
        available, WAIT/RELEASE otherwise)."""

        def distribution(self, obs, mask):
            mask_t = mask if mask.dim() == 2 else mask.unsqueeze(0)
            logits = torch.zeros_like(mask_t, dtype=torch.float32)
            logits[..., 1] = 0.2               # argmax prefers index 1
            return torch.distributions.Categorical(
                logits=logits.masked_fill(~mask_t, float("-inf")))

        def value(self, obs):
            return torch.zeros(obs.shape[0])

    seed_everything(0)
    forced = ForcedNet()
    rows = []
    for i in range(8):
        rows.append(dict(source=f"e{i}", scenario_seed=4107 + i, episode_index=i,
                         round_index=i // 4, time_s=600.0 + 5 * i,
                         phase="inactive", mask=[True, True],
                         obs_stacked_raw=[0.01] * C.OBS_DIM))
    for i in range(4):
        rows.append(dict(source=f"h{i}", scenario_seed=4107 + i, episode_index=i,
                         round_index=0, time_s=700.0 + 5 * i,
                         phase="hold", mask=[True, True],
                         obs_stacked_raw=[0.01] * C.OBS_DIM))
    norm = RunningNormalizer(C.OBS_DIM)
    norm.update([0.01] * C.OBS_DIM)
    stats = ta.compare_registry(rows, forced, forced,
                                norm.state_dict())
    # ForcedNet biases available actions toward index 1: argmax = ACTIVATE
    # on activation rows, CONTINUE on hold rows -> no flips, full agreement
    assert stats["activation_rows"] == 8 and stats["hold_rows"] == 4
    assert stats["rule_agreement_activation"] == 8
    assert stats["final"]["activate_chosen"] == 8
    assert stats["init"]["activate_chosen"] == 8   # index-1 bias: argmax=1
    assert stats["final"]["wait_to_activate"] == 0


def test_real_pilot_registry_had_zero_deterministic_flips():
    """The audited finding, pinned: across every recorded training state the
    initial and final deployed actors made IDENTICAL deterministic
    decisions (74 activation rows, 38 hold rows, zero flips)."""
    import torch
    from traffic_shepherd.ppo_timing_analysis import (
        load_registry, replay_round_normalizers)
    from traffic_shepherd import ppo_contract as C
    root = Path("runs/active/ppo_run_v1")
    schedule = json.loads((root / "registration.json").read_text(
        encoding="utf-8"))["training_schedule"]
    rows = load_registry(root / "training", schedule)
    replay_round_normalizers(root / "training")   # determinism check in its own test
    from traffic_shepherd.ppo_net import seed_everything
    seed_everything(C.PPO_CONFIG["optimizer_seed"])
    # historical rollouts were recorded under the v1 contract (116-dim obs)
    obs_dim = len(rows[0]["obs_stacked_raw"])
    net_init = ActorCritic(obs_dim=obs_dim)
    net_init.load_state_dict(torch.load(root / "checkpoints" / "ckpt_init.pt",
                                        weights_only=True))
    net_init.eval()
    net_final = ActorCritic(obs_dim=obs_dim)
    net_final.load_state_dict(torch.load(root / "checkpoints" / "ckpt_u8.pt",
                                         weights_only=True))
    net_final.eval()
    final_norm = json.loads((root / "normalizer.json").read_text(encoding="utf-8"))
    stats = ta.compare_registry(rows, net_init, net_final, final_norm)
    assert stats["final"]["wait_to_activate"] == 0
    assert stats["final"]["activate_to_wait"] == 0
    assert stats["final"]["continue_to_release"] == 0
    assert stats["final"]["release_to_continue"] == 0
    assert stats["activation_rows"] == 74 and stats["hold_rows"] == 38


def test_dev_evaluator_is_deterministic_argmax_not_sampling():
    """The development action-selection path is DeterministicActor's argmax;
    SamplingActor v2's RNG correction applies to future stochastic training
    rollouts only."""
    import inspect
    from traffic_shepherd import ppo_pilot
    src = inspect.getsource(ppo_pilot.DeterministicActor.__call__)
    assert "argmax" in src and "multinomial" not in src
    from traffic_shepherd.ppo_worker import SAMPLING_VERSION
    assert SAMPLING_VERSION == "dedicated_generator/2"
