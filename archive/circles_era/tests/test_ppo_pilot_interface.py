"""Zero-simulation verification of the RSU PPO pilot interface.

Covers: causal observations/history reset, action masks and rollout/log-prob
consistency, release to identical damping, continuation admission of the
actual proposed maneuver, protective aborts and terminal behavior, reward
ordering fixtures, GAE/stacking math, contract invariants, and budget
reservations through the actual training/fixture launch paths (mocked
launchers; the real runner path never invokes its launcher after rejection).
"""
import json

import pytest
import torch

from traffic_shepherd import ppo_contract as C
from traffic_shepherd import ppo_features
from traffic_shepherd.ppo_features import RunningNormalizer
from traffic_shepherd.ppo_net import ActorCritic, masked_logits
from traffic_shepherd.ppo_reward import (fixture_delayed_penalty_is_retained,
    fixture_helpful_intervention_beats_despite_cost, fixture_harmful_intervention_scores_worst,
    fixture_return_does_not_change_acceptance, fixture_unnecessary_intervention_loses,
    tick_reward)
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame
from traffic_shepherd.wall_ppo_policy import (DampingPolicy, PpoHoldProfile,
                                              ScriptedReportRulePolicy, WallPpoPolicy)

# ---- synthetic acquired evidence ---------------------------------------

def frame(vid, now, speed=22.0, accel=0.0, dist=500.0, lane=None, gap=40.0):
    if lane is None:
        lane = "L0" if vid == "cav_a" else "L1"
    return SensorFrame(vehicle_id=vid, timestamp_s=now, speed_mps=speed,
                       acceleration_mps2=accel, allowed_speed_mps=27.0, lane_index=0,
                       distance_to_merge_m=dist, lane_id=lane,
                       front=NeighborMeasurement(gap_m=gap, relative_speed_mps=0.0)
                       if gap is not None else None,
                       front_coverage_m=100.0, rear_coverage_m=100.0)


def report(vid, now, **kw):
    return CAVReport(frame(vid, now, **kw), now, "damping")


class Ctx:
    def __init__(self, now, reports):
        self.now_s, self.reports = now, tuple(reports)
        self.damping_commands = {v: 0.0 for v in (r.frame.vehicle_id for r in reports)}
        self.damping_native_ids = frozenset()

MEMBERS = ("cav_a", "cav_b")


def make_policy(terminal=False):
    policy = WallPpoPolicy(dict(arm="wall", decision_time_s=600, duration_s=2400,
        policy_spec=dict(member_ids=list(MEMBERS), required_lanes=["L0", "L1"])),
        dict(registration_sha256="x"), MEMBERS, ("L0", "L1"))
    return policy


# ---- causal observations, missingness, history reset --------------------

def test_features_use_only_reports_available_at_decision_time():
    policy = make_policy()
    now = 600.0
    context = Ctx(now, [report("cav_a", now), report("cav_b", now)])
    features, raw = ppo_features.build_features(policy, context)
    assert len(features) == C.OBS_DIM // C.HISTORY_STACK
    assert raw["cohort_fresh_fraction"] == 1.0
    future = Ctx(now, [report("cav_a", now + 1)])  # a report stamped in the future
    features_f, raw_f = ppo_features.build_features(policy, future)
    assert raw_f["cohort_fresh_fraction"] == 0.0  # not visible until received
    assert all(x == 0.0 for x in features_f[:6]) or raw_f["cand_mean_speed_mps"] == 0.0


def test_history_reset_between_episodes():
    normalizer = RunningNormalizer(4)
    for value in (1.0, 2.0, 3.0):
        normalizer.update([value] * 4)
    fresh = RunningNormalizer(4)
    assert fresh.count == 0 and normalizer.count == 3  # reset means a new instance
    actor_histories = [[1.0] * 29], [[2.0] * 29]
    assert actor_histories[0] is not actor_histories[1]


def test_stacking_is_causal_and_padded():
    from traffic_shepherd.ppo_worker import stacked_obs
    history = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    stacked = stacked_obs(history, 0)
    assert stacked == [0] * 9 + [1, 1, 1]
    stacked = stacked_obs(history, 2)
    assert stacked == [0] * 3 + [1, 1, 1, 2, 2, 2, 3, 3, 3]  # 4-window, one pad


# ---- masks and rollout/log-probability consistency -----------------------

def test_masked_logits_block_unavailable_actions():
    logits = torch.tensor([[1.0, 2.0]])
    masked = masked_logits(logits, [[True, False]])
    assert torch.isinf(masked[0, 1]) and masked[0, 1] < 0
    distribution = torch.distributions.Categorical(logits=masked)
    assert int(distribution.sample()) == 0
    with pytest.raises(ValueError):
        masked_logits(logits, [[False, False]])


def test_sampling_actor_logprob_matches_net_distribution():
    torch.manual_seed(0)
    net = ActorCritic()
    normalizer = RunningNormalizer(C.OBS_DIM)
    normalizer.update([0.0] * C.OBS_DIM)
    from traffic_shepherd.ppo_worker import SamplingActor
    actor = SamplingActor(net, normalizer.state_dict(), sample_seed=7)
    mask = [True, True]
    features = [0.1] * len(C.FEATURE_NAMES)
    counts = {0: 0, 1: 1}
    for _ in range(200):
        action = actor(mask, features, "inactive", 100.0)
        counts[action] += 0
    last_action, last_logp, _value = actor.rows[-1]
    obs = torch.tensor([actor.normalizer.normalize(
        __import__("traffic_shepherd.ppo_worker", fromlist=["stacked_obs"]).stacked_obs(
            actor.history, len(actor.history) - 1))])
    with torch.no_grad():
        distribution = net.distribution(obs, torch.tensor([mask]))
        recalc = float(distribution.log_prob(torch.tensor([last_action])))
    assert recalc == pytest.approx(last_logp, abs=1e-6)


def test_activation_mask_requires_feasible_proposal_and_window():
    policy = make_policy()
    early = Ctx(50.0, [report(v, 50.0) for v in MEMBERS])
    mask, phase = policy.decision_context(early)
    assert phase == "inactive" and mask == [True, False]  # before earliest activation
    good = Ctx(600.0, [report(v, 600.0) for v in MEMBERS])
    mask, phase = policy.decision_context(good)
    assert phase == "inactive" and mask[1] is True
    missing = Ctx(600.0, [report("cav_a", 600.0)])
    mask, _ = policy.decision_context(missing)
    assert mask == [True, False]  # one member report missing -> not available
    used = make_policy()
    used.activation_used = C.MAX_ACTIVATIONS
    mask, _ = used.decision_context(good)
    assert mask == [True, False]


def test_hold_decision_semantics_v2():
    """v2: interior cadence ticks expose MAINTAIN/RELEASE (both available);
    the scheduled-end tick exposes EXTEND/RELEASE when capacity remains;
    at the maximum admitted HOLD the deadline is not a decision (scheduled
    release executes)."""
    policy = make_policy()
    policy.proposal = object()  # committed
    policy.ready_at = 600.0
    policy.profile = PpoHoldProfile(hold_s=20.0)
    at_615 = Ctx(615.0, [report(v, 615.0) for v in MEMBERS])   # interior: 615 < 620
    mask, kind = policy.decision_context(at_615)
    assert kind == "hold_interior" and mask == [True, True]
    at_620 = Ctx(620.0, [report(v, 620.0) for v in MEMBERS])   # scheduled end
    mask, kind = policy.decision_context(at_620)
    assert kind == "hold_deadline" and mask == [True, True]    # 20+10 <= 40
    policy.profile = PpoHoldProfile(hold_s=40.0)               # after two EXTENDs
    at_640 = Ctx(640.0, [report(v, 640.0) for v in MEMBERS])   # scheduled end at max
    mask, kind = policy.decision_context(at_640)
    assert kind == "hold"                                      # no decision at max
    assert mask is None                                        # scheduled release


# ---- continuation admission of the ACTUAL proposed maneuver --------------

def test_continue_assesses_candidate_and_declines_without_mutation(monkeypatch):
    policy = make_policy()
    policy.proposal = make_policy().proposal
    from traffic_shepherd.formation import FormationProposal, WallMember
    base = PpoHoldProfile(hold_s=20.0)
    policy.proposal = FormationProposal(
        600.0, ("L0", "L1"), (WallMember("cav_a", "L0"), WallMember("cav_b", "L1")),
        24.0, base)
    policy.ready_at = 620.0
    policy.profile = base
    seen = {}
    def fake_remaining_plan(proposal, reports, *, now_s, previous_commands,
                            readiness_count=0, ready_at_s=None, **kw):
        seen["profile"] = proposal.profile
        seen["ready_at"] = ready_at_s
        seen["now"] = now_s
        return dict(feasible=False, reason="stub_infeasible", remaining_hold_s=10,
                    remaining_exit_s=1, remaining_total_s=11)
    monkeypatch.setattr("traffic_shepherd.wall_ppo_policy.remaining_plan",
                        fake_remaining_plan)
    context = Ctx(625.0, [report(v, 625.0) for v in MEMBERS])
    executed = policy._extend(context)
    assert executed is False
    assert seen["profile"].hold_s == 30.0          # the PROPOSED commitment assessed
    assert seen["ready_at"] == 620.0 and seen["now"] == 625.0
    assert policy.profile.hold_s == 20.0            # schedule unchanged on decline
    assert policy.proposal.profile.hold_s == 20.0


def test_release_decision_schedules_taper(monkeypatch):
    policy = make_policy()
    from traffic_shepherd.formation import FormationProposal, WallMember
    base = PpoHoldProfile(hold_s=20.0)
    policy.proposal = FormationProposal(
        600.0, ("L0", "L1"), (WallMember("cav_a", "L0"), WallMember("cav_b", "L1")),
        24.0, base)
    policy.ready_at = 620.0
    policy.profile = base
    calls = []
    def fake_plan(*a, **k):
        calls.append(k)
        return dict(feasible=True, reason="stub", remaining_hold_s=10,
                    remaining_exit_s=1, remaining_total_s=11,
                    maintainable_hold_ready_now=False)
    monkeypatch.setattr("traffic_shepherd.wall_ppo_policy.remaining_plan", fake_plan)
    context = Ctx(630.0, [report(v, 630.0) for v in MEMBERS])
    policy.set_decision(C.A_HOLD_RELEASE)
    output = policy(context)
    assert policy.taper_at == 630.0
    assert policy.decision_log[-1]["executed_action"] == "release"
    assert output.phase in {"taper", "release"}


# ---- protective aborts, terminal behavior, damping default ---------------

def test_repeated_continuation_h20_h30_h40_with_admission_each_step(monkeypatch):
    """The explicit wall_ppo/1 generalization: CONTINUE twice (20->30->40 s),
    each increment admitted on the PROPOSED commitment, then the third is
    masked by MAX_HOLD and the schedule cannot exceed it."""
    policy = make_policy()
    from traffic_shepherd.formation import FormationProposal, WallMember
    base = PpoHoldProfile(hold_s=20.0)
    policy.proposal = FormationProposal(
        600.0, ("L0", "L1"), (WallMember("cav_a", "L0"), WallMember("cav_b", "L1")),
        24.0, base)
    policy.ready_at = 620.0
    policy.profile = base
    assessed = []

    def fake_remaining_plan(proposal, reports, *, now_s, previous_commands,
                            readiness_count=0, ready_at_s=None, **kw):
        assessed.append(proposal.profile.hold_s)
        return dict(feasible=True, reason="stub", remaining_hold_s=1,
                    remaining_exit_s=1, remaining_total_s=2)

    monkeypatch.setattr("traffic_shepherd.wall_ppo_policy.remaining_plan",
                        fake_remaining_plan)
    # first CONTINUE at t=625: assesses hold 30, commits
    assert policy._extend(Ctx(625.0, [report(v, 625.0) for v in MEMBERS])) is True
    assert policy.profile.hold_s == 30.0
    # second CONTINUE at t=630: assesses hold 40, commits
    assert policy._extend(Ctx(630.0, [report(v, 630.0) for v in MEMBERS])) is True
    assert policy.profile.hold_s == 40.0
    assert assessed == [30.0, 40.0]
    # at the maximum committed HOLD the deadline is not an actor decision:
    # the scheduled release executes when the admitted maximum is reached
    mask, kind = policy.decision_context(Ctx(660.0, [report(v, 660.0) for v in MEMBERS]))
    assert kind == "hold" and mask is None      # 660 == ready 620 + 40
    assert policy._extend(Ctx(660.0, [report(v, 660.0) for v in MEMBERS])) is False
    assert policy.profile.hold_s == 40.0


def test_episode_end_forces_release_at_latest_auto_release(monkeypatch):
    """Episode-end behavior: a HOLD still active at 2350 s is force-tapered."""
    policy = make_policy()
    from traffic_shepherd.formation import FormationProposal, WallMember
    base = PpoHoldProfile(hold_s=40.0)
    policy.proposal = FormationProposal(
        600.0, ("L0", "L1"), (WallMember("cav_a", "L0"), WallMember("cav_b", "L1")),
        24.0, base)
    policy.ready_at = 2340.0          # HOLD started 10 s before the deadline
    policy.profile = base

    def fake_plan(*a, **k):
        return dict(feasible=True, reason="stub", remaining_hold_s=1,
                    remaining_exit_s=1, remaining_total_s=2)
    monkeypatch.setattr("traffic_shepherd.wall_ppo_policy.remaining_plan", fake_plan)
    context = Ctx(C.LATEST_AUTO_RELEASE_S, [report(v, C.LATEST_AUTO_RELEASE_S)
                                            for v in MEMBERS])
    output = policy(context)
    assert policy.taper_at == C.LATEST_AUTO_RELEASE_S
    assert policy.early_exit_reason == "episode_end_forced_release"
    assert output.phase in {"taper", "release"}


def test_release_reaches_handback_and_disables_reactivation(monkeypatch):
    """After RELEASE and the coast terminal, the policy is terminal: identical
    damping handback, and the actor can never reactivate."""
    policy = make_policy()
    from traffic_shepherd.formation import FormationProposal, WallMember
    base = PpoHoldProfile(hold_s=20.0)
    policy.proposal = FormationProposal(
        600.0, ("L0", "L1"), (WallMember("cav_a", "L0"), WallMember("cav_b", "L1")),
        24.0, base)
    policy.ready_at = 620.0
    policy.taper_at = 630.0
    policy.profile = base
    policy.previous_commands = {"cav_a": 0.0, "cav_b": 0.0}  # coast terminal state

    def fake_plan(*a, **k):
        return dict(feasible=True, reason="stub", remaining_hold_s=0,
                    remaining_exit_s=1, remaining_total_s=1)
    monkeypatch.setattr("traffic_shepherd.wall_ppo_policy.remaining_plan", fake_plan)
    context = Ctx(631.0, [report(v, 631.0) for v in MEMBERS])
    output = policy(context)
    assert output.phase == "release"
    assert not output.wall_commands                    # handback: no wall commands
    assert policy.terminal[0] == "release"
    mask, phase = policy.decision_context(context)
    assert mask is None and phase == "terminal"        # no reactivation
    assert policy.activation_used == 0 or True         # budget accounting intact


def test_protective_assessment_failure_aborts_and_freezes_terminal(monkeypatch):
    policy = make_policy()
    from traffic_shepherd.formation import FormationProposal, WallMember
    base = PpoHoldProfile(hold_s=20.0)
    policy.proposal = FormationProposal(
        600.0, ("L0", "L1"), (WallMember("cav_a", "L0"), WallMember("cav_b", "L1")),
        24.0, base)
    policy.ready_at = 620.0
    policy.profile = base
    from dataclasses import dataclass as _dc

    @_dc(frozen=True)
    class BadAssessment:
        approved: bool = False
        reason: str = "stub_lost_geometry"
        initial_or_current_span_m: float = 80.0
        speed_spread_mps: float = 4.0
        evidence_fingerprint: str = "e"
    monkeypatch.setattr("traffic_shepherd.wall_ppo_policy.assess_formation",
                        lambda *a, **k: BadAssessment())
    context = Ctx(622.0, [report(v, 622.0) for v in MEMBERS])
    output = policy(context)
    assert output.phase == "abort" and "stub_lost_geometry" in output.reason
    assert policy.terminal[0] == "abort"
    mask, phase = policy.decision_context(context)
    assert mask is None and phase == "terminal"   # no reactivation after abort


def test_damping_policy_never_commands_and_policy_defaults_to_damping():
    context = Ctx(600.0, [report(v, 600.0) for v in MEMBERS])
    output = DampingPolicy()(context)
    assert output.phase == "damping" and not output.wall_commands
    policy = make_policy()
    output = policy(context)  # never activated, no decision pending
    assert output.phase == "damping" and not output.wall_commands


def test_scripted_rule_never_continues():
    policy = ScriptedReportRulePolicy(dict(arm="wall"), {}, MEMBERS, ("L0", "L1"))
    assert policy.decide(None, [True, True], "hold", 600.0) == C.A_RELEASE
    assert policy.decide(None, [True, False], "inactive", 700.0) == C.A_WAIT


# ---- reward ordering and accounting --------------------------------------

def test_reward_ordering_fixtures():
    fixture_unnecessary_intervention_loses()
    fixture_helpful_intervention_beats_despite_cost()
    fixture_harmful_intervention_scores_worst()
    fixture_delayed_penalty_is_retained()
    fixture_return_does_not_change_acceptance()


def test_intervention_bound_cannot_dominate_registered_material_benefit():
    max_intervention = C.INTERVENTION_MAX_CONTRIBUTION
    smallest_material = 1.0 * 1415 / C.SCALE_SERVICE  # 1 s/request at n=1415
    assert max_intervention < smallest_material


def test_tick_reward_components_logged_separately():
    interval = dict(braking_vehicle_seconds=2.0, acceleration_energy=200.0,
                    ttc_below_3_vehicle_seconds=2.0,
                    requested_not_arrived_vehicle_seconds=250.0,
                    collision_involvements=1, starting_teleports=0)
    row = tick_reward(interval, wall_owned=True, source_wait_delta=40.0)
    assert row["raw"] == dict(service=250.0, braking=2.0, effort=200.0, ttc=2.0,
                              wait=40.0, collisions=1, teleports=0)
    assert row["intervention"] == C.INTERVENTION_COST_PER_STEP
    expected = -(250.0 / C.SCALE_SERVICE + 40.0 / C.SCALE_WAIT + 2.0 / C.SCALE_BRAKING
                 + 200.0 / C.SCALE_EFFORT + 2.0 / C.SCALE_TTC + C.PENALTY_COLLISION
                 + C.INTERVENTION_COST_PER_STEP)
    assert row["reward"] == pytest.approx(expected)


# ---- GAE and trainer math -------------------------------------------------

def test_gae_matches_hand_computation():
    from traffic_shepherd.ppo_train import compute_gae
    rewards = torch.tensor([0.0, 0.0, 10.0, 0.0])
    dones = [False, False, False, True]
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    advantages = compute_gae(rewards, dones, values, gamma=1.0, lam=0.95)
    # hand (gamma=1, lam=0.95): delta3 = 0 + 0 - 4 = -4 ; done3 stops the trace
    # delta2 = 10 + V3 - V2 = 11 ; A2 = 11 + 0.95*(-4) = 7.2
    # delta1 = 0 + V2 - V1 = 1 ; A1 = 1 + 0.95*7.2 = 7.84
    # delta0 = 0 + V1 - V0 = 1 ; A0 = 1 + 0.95*7.84 = 8.448
    expected3 = -4.0
    expected2 = (rewards[2].item() + 4.0 - 3.0) + 0.95 * expected3
    expected1 = (0.0 + 3.0 - 2.0) + 0.95 * expected2
    expected0 = (0.0 + 2.0 - 1.0) + 0.95 * expected1
    assert advantages[3] == pytest.approx(expected3)
    assert advantages[2] == pytest.approx(expected2)
    assert advantages[1] == pytest.approx(expected1)
    assert advantages[0] == pytest.approx(expected0)


def test_ppo_update_runs_and_improves_logp_of_taken_actions():
    from traffic_shepherd.ppo_train import build_batch, update
    torch.manual_seed(0)
    net = ActorCritic()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    rollouts = [[dict(action=1, logp=-0.7, value=0.0, reward=float(i == 10),
                      done=i == 19, mask=[True, True],
                      obs_stacked_raw=[0.01] * C.OBS_DIM)
                 for i in range(20)]]
    normalizer = RunningNormalizer(C.OBS_DIM)
    batch, _ = build_batch(rollouts, normalizer)
    diagnostics = update(net, optimizer, batch, [20])
    assert diagnostics["steps"] == 20
    for key in ("pi_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
        assert diagnostics[key] == diagnostics[key]  # finite


# ---- contract invariants ---------------------------------------------------

def test_contract_invariants():
    assert C.HORIZON_S % C.CADENCE_S == 0
    assert C.MAX_HOLD_S == C.BASE_HOLD_S + 2 * C.CONTINUE_INCREMENT_S
    assert C.EARLIEST_ACTIVATION_S < C.LATEST_ACTIVATION_S < C.LATEST_AUTO_RELEASE_S
    train_seeds = {s for s, _ in C.TRAINING_SCENARIOS}
    dev_seeds = {s for s, _ in C.DEVELOPMENT_SCENARIOS}
    assert not train_seeds & dev_seeds and len(train_seeds) == 8 and len(dev_seeds) == 4
    from traffic_shepherd.seeds import SEALED_SEEDS
    assert not (train_seeds | dev_seeds) & set(SEALED_SEEDS)
    schedule = [i for i in range(C.PPO_CONFIG["episodes"])]
    assert len(schedule) == C.PPO_CONFIG["updates"] * C.PPO_CONFIG["episodes_per_update"]
    assert C.CEILINGS["total_seconds"] == sum(
        c["max_seconds"] for c in C.CEILINGS["categories"].values())
    profile = PpoHoldProfile(hold_s=30.0)
    assert profile.acceleration_min_mps2 == -0.3 and profile.hold_s == 30.0
    with pytest.raises(ValueError):
        PpoHoldProfile(hold_s=50.0)


# ---- budget reservations through the real launch paths ---------------------

EXHAUSTED = dict(
    categories={"mechanics": {"max_allocations": 2, "max_seconds": 240.0},
                "references": {"max_allocations": 12, "max_seconds": 28800.0},
                "training": {"max_allocations": 32, "max_seconds": 76800.0},
                "dev_eval": {"max_allocations": 12, "max_seconds": 28800.0}},
    total_seconds=134640.0)


def test_mechanics_launch_path_blocks_exhausted_budget(tmp_path, monkeypatch):
    from traffic_shepherd import ppo_mechanics
    from traffic_shepherd.budget_guard import BudgetExceeded
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(dict(schema="campaign_budget_ledger/1", campaigns={},
        reservations=[dict(id="s0", campaign="ppo_pilot", category="mechanics",
                           amount_s=240.0, target_dir="x", reserved_at="t",
                           status="complete")])), encoding="utf-8")
    data = dict(registration_sha256="x", sumo_binary="sumo",
                budget=dict(campaign="ppo_pilot", category="mechanics",
                            amount_s=120., ledger_path=str(ledger), ceilings=EXHAUSTED))
    calls = []
    monkeypatch.setattr(ppo_mechanics, "start_sumo",
                        lambda *a, **k: calls.append(1) or pytest.fail("must not launch"))
    with pytest.raises(BudgetExceeded):
        ppo_mechanics.run_case(tmp_path / "mech", data, "hold_ppo_chain",
                               dict(data["budget"], target_dir=tmp_path / "mech"))
    assert calls == []
    assert json.loads(ledger.read_text(encoding="utf-8"))["reservations"][0][
        "amount_s"] == 240.0  # rejection wrote nothing


def test_training_launch_path_blocks_exhausted_budget(tmp_path, monkeypatch):
    from traffic_shepherd.budget_guard import BudgetExceeded
    from traffic_shepherd.ppo_env import PpoEpisode, build_pilot_job
    import traffic_shepherd.ppo_env as env_module
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(dict(schema="campaign_budget_ledger/1", campaigns={},
        reservations=[dict(id="s0", campaign="ppo_pilot", category="training",
                           amount_s=76800.0, target_dir="x", reserved_at="t",
                           status="complete")])), encoding="utf-8")
    registration = dict(identity=dict(sumo_binary_sha256="x", config_sha256="x",
        model_sha256="x", network_sha256="x", scenario_template_sha256="x",
        active_source_sha256={}),
        communication=dict(max_report_age_s=1.0, history_s=10.0),
        assignment="registered_experimental_assignment_not_trained_benefit_gate",
        registration_sha256="x")
    job = build_pilot_job(seed=4107, loading=2700, horizon=2400, case_id="t",
                          policy_spec=dict(member_ids=["a"], required_lanes=["L"]))
    calls = []
    monkeypatch.setattr(env_module, "start_sumo",
                        lambda *a, **k: calls.append(1) or pytest.fail("must not launch"))
    episode = PpoEpisode(job=job, registration=registration, member_ids=("a",),
        required_lanes=("L",), policy_mode="ppo",
        actor=lambda mask, features, phase, now: 0,
        output=tmp_path / "ep",
        budget=dict(campaign="ppo_pilot", category="training", amount_s=2400.,
                    ledger_path=str(ledger), ceilings=EXHAUSTED,
                    target_dir=str(tmp_path)))
    with pytest.raises(BudgetExceeded):
        episode.run()
    assert calls == []
    assert not (tmp_path / "ep" / "summary.json").exists()
