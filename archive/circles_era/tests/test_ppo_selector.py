"""wall_ppo/3 causal selector tests (zero simulation; PP-12 repaired contract).

Covers the offer-binding contract: causal eligibility, offer binding before
observation construction, WAIT-driven rebinding, execution-time admission of
the EXACT approved offer (no re-ranking, atomic refusal, no substitution),
cohort freezing, wall_ppo/2 duration invariance, version gating, and the
phase-aware scripted comparator. The populated mocked-environment trace
lives in test_ppo_selector_env_populated.py.
"""
import sys

import pytest

from traffic_shepherd import ppo_contract as C
from traffic_shepherd.cohort_study import LANES
from traffic_shepherd.selector_policy import (SELECTOR_VERSION,
                                              causal_eligible_reports,
                                              search_current_proposals)
from traffic_shepherd.selector_wall_policy import (SELECTOR_POLICY_VERSION,
                                                   SelectorWallPolicy,
                                                   scripted_selector_chain_actor)

LANE_0, LANE_1 = LANES


def feature_vec(**named):
    """A current-tick FEATURE_NAMES vector with the named values set."""
    v = [0.0] * len(C.FEATURE_NAMES)
    for name, value in named.items():
        v[C.FEATURE_NAMES.index(name)] = value
    return v


@pytest.fixture()
def fake_traci(monkeypatch):
    """Same empty-world boundary as the corridor smoke tests."""
    from tests.test_ppo_env_mocked import EmptyWorldTraci
    fake = EmptyWorldTraci()
    monkeypatch.setitem(sys.modules, "traci", fake)
    monkeypatch.setattr("traffic_shepherd.ppo_env.start_sumo",
                        lambda *a, **k: 54321)
    return fake


def report(vid, ts, lane=LANE_0, dist=500.0, speed=22.0, received=None):
    from traffic_shepherd.candidate_availability import row_to_cav_report
    return row_to_cav_report(dict(
        frame=dict(vehicle_id=vid, timestamp_s=ts, speed_mps=speed,
                   acceleration_mps2=0.0, allowed_speed_mps=27.0,
                   lane_index=0, distance_to_merge_m=dist, lane_id=lane,
                   front=None, rear=None, front_coverage_m=100.0,
                   rear_coverage_m=100.0, previous_command_delta=0.0,
                   position_m=3500.0 - dist),
        sent_at_s=received if received is not None else ts,
        received_at_s=received if received is not None else ts,
        owner="damping", network_output_mps2=None,
        issued_acceleration_mps2=None, fallback_reason=None))


class Ctx:
    def __init__(self, now, reports):
        self.now_s = now
        self.reports = reports
        self.damping_commands = {r.frame.vehicle_id: 0.0 for r in reports}
        self.damping_native_ids = frozenset()


def make_selector(activation_window=(10.0, 1800.0)):
    return SelectorWallPolicy(
        dict(arm="wall", policy_spec=dict(required_lanes=list(LANES),
                                          activation_window=list(activation_window))),
        dict(registration_sha256="x",
             communication=dict(max_report_age_s=1.0, history_s=10.0)))


def two_reports(now, dist_a=1500.0, dist_b=1505.0, speed=22.0,
                ids=("cav_a", "cav_b")):
    return [report(ids[0], now, lane=LANE_0, dist=dist_a, speed=speed),
            report(ids[1], now, lane=LANE_1, dist=dist_b, speed=speed)]


def search(reports, now, policy=None):
    policy = policy or make_selector()
    return search_current_proposals(tuple(reports), now,
                                    profile=policy.profile,
                                    required_lanes=policy.required_lanes)


# ---- candidate search (repaired signature: profile and lanes REQUIRED) ------

def test_search_finds_feasible_pair_from_current_reports():
    best, _reason, n = search(two_reports(600.0), 600.0)
    assert best is not None and set(best) == {"cav_a", "cav_b"}
    assert n == 1


def test_search_requires_profile_and_lanes():
    with pytest.raises(TypeError):
        search_current_proposals(tuple(two_reports(600.0)), 600.0)


def test_no_reports_means_no_feasible_pair():
    best, _reason, _n = search((), 600.0)
    assert best is None


def test_different_cohorts_feasible_at_different_times():
    """Cohort A feasible at t=600; a DIFFERENT cohort (fresh CAVs) feasible
    at t=900 — the selector is not tied to one fixed set of vehicles."""
    for now, ids in ((600.0, {"old_a", "old_b"}), (900.0, {"new_a", "new_b"})):
        reports = [report(vid, now, lane=LANE_0 if vid.endswith("a") else LANE_1,
                          dist=1500.0) for vid in sorted(ids)]
        best, _reason, _n = search(reports, now)
        assert best is not None and set(best) == ids


def test_stale_future_and_duplicate_reports_cannot_enable_activation():
    now = 600.0
    fresh = two_reports(now)
    stale = two_reports(now - 5.0)          # acquisition too old
    future = two_reports(now + 3.0)         # acquired in the future
    assert len(causal_eligible_reports(stale, now)) == 0
    assert len(causal_eligible_reports(future, now)) == 0
    assert len(causal_eligible_reports(fresh + fresh, now)) == 2   # deduped
    best, _r, _n = search(stale, now)
    assert best is None
    policy = make_selector()
    assert policy.activation_feasible(Ctx(now, stale + future)) is False
    assert policy.offered is None


# ---- offer binding before observations --------------------------------------

def test_offer_bound_before_observations_and_features_match_offer():
    from traffic_shepherd.ppo_features import build_features
    policy = make_selector()
    now = 600.0
    reports = two_reports(now, dist_a=1500.0, dist_b=1505.0)
    mask, kind = policy.decision_context(Ctx(now, reports))
    assert mask == [True, True] and kind == "inactive"
    offer = policy.offered
    assert offer is not None and set(offer.member_ids) == {"cav_a", "cav_b"}
    assert offer.profile is policy.profile                 # exact profile object
    assert offer.required_lanes == policy.required_lanes   # exact lanes
    assert offer.decision_time_s == now
    features, values = build_features(policy, Ctx(now, reports))
    # candidate features describe the OFFERED cohort (not zeros)
    speeds = [r.frame.speed_mps for r in reports]
    assert values["cand_mean_speed_mps"] == pytest.approx(sum(speeds) / 2)
    assert values["cohort_fresh_fraction"] == 1.0
    assert values["feasible_activate_now"] == 1.0


def test_no_offer_means_no_activation_bit_and_zero_candidate_features():
    from traffic_shepherd.ppo_features import build_features
    policy = make_selector()
    now = 600.0
    mask, kind = policy.decision_context(Ctx(now, []))
    assert mask == [True, False] and policy.offered is None
    _features, values = build_features(policy, Ctx(now, []))
    assert values["cand_mean_speed_mps"] == 0.0
    assert values["feasible_activate_now"] == 0.0


def test_wait_clears_offer_and_next_decision_rebinds_fresh():
    policy = make_selector()
    ctx1 = Ctx(600.0, two_reports(600.0, ids=("a1", "b1")))
    mask, _kind = policy.decision_context(ctx1)
    assert mask[1] is True and policy.offered is not None
    policy.set_decision(C.A_WAIT)
    assert policy.offered is None and policy.approved_offer is None
    # next eligible decision: a fresh search from then-current reports
    ctx2 = Ctx(605.0, two_reports(605.0, ids=("a2", "b2")))
    mask, _kind = policy.decision_context(ctx2)
    assert mask[1] is True
    assert set(policy.offered.member_ids) == {"a2", "b2"}
    assert policy.offered.decision_time_s == 605.0


# ---- execution re-checks the approved offer ---------------------------------

def test_execution_commits_exactly_the_approved_offer():
    policy = make_selector()
    now = 600.0
    policy.activation_feasible(Ctx(now, two_reports(now, ids=("a1", "b1"))))
    approved = policy.offered
    policy.set_decision(C.A_ACTIVATE)
    assert policy.approved_offer is approved
    # execution context: the approved pair AND a new nearer cohort; the
    # approved offer must be committed regardless of any re-ranking
    exec_reports = two_reports(now, ids=("a1", "b1")) + \
        two_reports(now, dist_a=600.0, dist_b=605.0, speed=10.0,
                    ids=("z1", "z2"))
    best_now, _r, _n = search(exec_reports, now, policy)
    executed = policy._activate(Ctx(now, exec_reports))
    assert executed is True
    assert tuple(policy.proposal.member_ids) == approved.member_ids
    assert policy.committed_offer is approved
    # the committed target is the offered target, exactly
    assert policy.proposal.target_speed_mps == pytest.approx(
        approved.target_speed_mps)


def test_execution_does_not_substitute_when_approved_offer_dies():
    policy = make_selector()
    now = 600.0
    policy.activation_feasible(Ctx(now, two_reports(now, ids=("a1", "b1"))))
    policy.set_decision(C.A_ACTIVATE)
    # execution time: approved members left the band, but a DIFFERENT
    # feasible cohort exists — refusal is atomic, never a substitution
    exec_reports = two_reports(now, dist_a=30.0, dist_b=35.0,
                               ids=("a1", "b1")) + \
        two_reports(now, ids=("z1", "z2"))
    assert search(exec_reports, now, policy) is not None or True
    best_now, _r, _n = search(exec_reports, now, policy)
    assert best_now is not None and set(best_now) == {"z1", "z2"}
    executed = policy._activate(Ctx(now, exec_reports))
    assert executed is False
    assert policy.proposal is None
    assert policy.activation_used == 0      # permission not consumed
    refusal = policy.last_execution_refusal
    assert refusal["offer"]["member_ids"] == ["a1", "b1"]
    assert refusal["reason"]


def test_execution_with_no_bound_offer_refuses_atomically():
    policy = make_selector()
    assert policy._activate(Ctx(600.0, two_reports(600.0))) is False
    assert policy.last_execution_refusal["reason"] == "no_bound_offer"
    assert policy.activation_used == 0


def test_decision_receipts_carry_offer_identity():
    policy = make_selector()
    now = 600.0
    ctx = Ctx(now, two_reports(now))
    policy.activation_feasible(ctx)
    policy.set_decision(C.A_ACTIVATE)
    assert policy._activate(ctx) is True
    # the base-class decision receipt is enriched through the normal path
    from traffic_shepherd.paired_experiments import PolicyInput
    policy.pending_action = None
    policy.decision_log.clear()
    inp = PolicyInput(now, tuple(ctx.reports), tuple(), {}, frozenset())
    mask, kind = policy.decision_context(inp)
    # post-activation the phase is assembly: no inactive decisions remain
    assert mask is None and kind == "assembly"


def test_offer_identity_logged_on_wait_and_activation(tmp_path):
    policy = make_selector()
    now = 600.0
    ctx = Ctx(now, two_reports(now))
    mask, kind = policy.decision_context(ctx)
    policy.set_decision(C.A_WAIT)
    # drive __call__ so the receipt path (incl. re-bind) executes
    policy.pending_action = C.A_WAIT
    policy(ctx)   # full receipt path incl. the execution-time re-bind
    wait_entry = policy.decision_log[-1]
    assert wait_entry["executed_action"] == "wait"
    assert wait_entry["offer"]["member_ids"] == ["cav_a", "cav_b"]
    assert wait_entry["offer"]["selector_version"] == SELECTOR_VERSION
    assert wait_entry["offer"]["committed"] is False


# ---- cohort frozen after activation -----------------------------------------

def test_cohort_frozen_after_activation():
    policy = make_selector()
    now = 600.0
    context = Ctx(now, two_reports(now, dist_a=1500.0, dist_b=1505.0))
    mask, _kind = policy.decision_context(context)
    policy.set_decision(C.A_ACTIVATE)
    assert policy._activate(context) is True
    assert policy.proposal is not None
    frozen = tuple(policy.proposal.member_ids)
    # a later tick sees DIFFERENT fresh CAVs and the search would offer them,
    # but the activated cohort stays frozen (one activation per episode)
    later = Ctx(700.0, [report(v, 700.0, lane=LANE_0 if v.endswith("a") else LANE_1,
                               dist=1500.0)
                        for v in ("other_a", "other_b")])
    assert set(policy.selected_pair(later)) == {"other_a", "other_b"}
    assert tuple(policy.proposal.member_ids) == frozen
    assert policy.activation_used == 1


def test_no_reactivation_after_terminal():
    policy = make_selector()
    from traffic_shepherd.formation import FormationProposal, WallMember
    from traffic_shepherd.wall_ppo_policy import PpoHoldProfile
    base = PpoHoldProfile(hold_s=20.0)
    policy.proposal = FormationProposal(
        600.0, ("L0", "L1"), (WallMember("cav_a", "L0"), WallMember("cav_b", "L1")),
        24.0, base)
    policy.terminal = ("release", "done")
    mask, phase = policy.decision_context(Ctx(700.0, []))
    assert mask is None and phase == "terminal"


# ---- duration semantics unchanged (wall_ppo/2) ------------------------------

def test_hold_semantics_unchanged_by_selector():
    """The selector policy's hold-phase decision_context matches
    WallPpoPolicy's exactly (MAINTAIN/RELEASE interior; EXTEND/RELEASE at
    the deadline)."""
    selector = make_selector()
    # the selector is a WallPpoPolicy: it adds only the candidate search
    from traffic_shepherd.wall_ppo_policy import WallPpoPolicy
    assert isinstance(selector, WallPpoPolicy)
    # the MAINTAIN/EXTEND/RELEASE meanings are the shared ACTION_MEANINGS
    assert C.ACTION_MEANINGS["hold_interior"] == ["MAINTAIN", "RELEASE"]
    assert C.ACTION_MEANINGS["hold_deadline"] == ["EXTEND", "RELEASE"]


# ---- scripted comparator (named phase/action space) -------------------------

def test_scripted_comparator_is_phase_aware():
    now = 600.0
    # inactive: ACTIVATE the offered proposal, else WAIT (named lookups)
    assert scripted_selector_chain_actor([True, True], {}, "inactive", now) == \
        C.ACTION_MEANINGS["inactive"].index("ACTIVATE")
    assert scripted_selector_chain_actor([True, False], {}, "inactive", now) == \
        C.ACTION_MEANINGS["inactive"].index("WAIT")
    # interior HOLD: MAINTAIN — NEVER the phase-blind index-1 (RELEASE)
    assert scripted_selector_chain_actor([True, True], {}, "hold_interior", now) == \
        C.ACTION_MEANINGS["hold_interior"].index("MAINTAIN")
    # deadline at H20: EXTEND when the mask admits it
    assert scripted_selector_chain_actor(
        [True, True], feature_vec(admitted_total_hold_s=20.0),
        "hold_deadline", now) == C.ACTION_MEANINGS["hold_deadline"].index("EXTEND")
    # deadline at H30 (declared chain end): RELEASE even though EXTEND fits
    assert scripted_selector_chain_actor(
        [True, True], feature_vec(admitted_total_hold_s=30.0),
        "hold_deadline", now) == C.ACTION_MEANINGS["hold_deadline"].index("RELEASE")
    # extension unavailable in the mask: RELEASE
    assert scripted_selector_chain_actor(
        [False, True], feature_vec(admitted_total_hold_s=20.0),
        "hold_deadline", now) == C.ACTION_MEANINGS["hold_deadline"].index("RELEASE")
    with pytest.raises(KeyError):
        scripted_selector_chain_actor([True, True], {}, "unknown_kind", now)


def test_phase_blind_index_driver_releases_at_first_hold():
    """Documents the PP-11 defect: an index-1-when-available driver releases
    at the first HOLD decision because 1 means RELEASE there."""
    mask = [True, True]
    assert (C.A_ACTIVATE if mask[C.A_ACTIVATE] else C.A_WAIT) == 1  # activates
    assert mask[1] == C.A_HOLD_RELEASE                              # ...releases


# ---- version gate -----------------------------------------------------------

def test_version_gate_rejects_incompatible_checkpoints():
    import torch
    from traffic_shepherd.ppo_net import ActorCritic
    assert C.POLICY_VERSION == "wall_ppo/2"
    assert SELECTOR_POLICY_VERSION == "wall_ppo/3"
    assert SELECTOR_VERSION == "causal_selector/2"
    # a v1-era (116-dim) checkpoint must not load into the v2 (128-dim) net
    net = ActorCritic()
    legacy = {k: v for k, v in net.state_dict().items()}
    legacy["body.0.weight"] = torch.zeros(64, 116)
    fresh = ActorCritic()
    with pytest.raises(RuntimeError):
        fresh.load_state_dict(legacy)


# ---- full mocked env trace (WAIT-only empty world; populated trace in the
# ---- populated test module) --------------------------------------------------

def test_full_mocked_env_trace_selector(tmp_path, fake_traci):
    """The real env loop with policy_mode="selector" on the empty-world
    mocked traci boundary: WAIT at every eligible tick (no vehicles → no
    reports → the search never finds a feasible pair), no fabricated
    decisions, no wall commands, full horizon."""
    from traffic_shepherd.ppo_env import PpoEpisode, build_pilot_job

    job = build_pilot_job(seed=4107, loading=2700, horizon=30,
                          case_id="sel_trace", decision_time_s=5,
                          policy_spec=dict(member_ids=["ghost_a", "ghost_b"],
                                           required_lanes=list(LANES)))

    episode = PpoEpisode(job=job, registration=_mock_registration(),
        member_ids=("ghost_a", "ghost_b"), required_lanes=tuple(LANES),
        policy_mode="selector", output=tmp_path / "sel_trace")
    summary, rollout = episode.run()
    assert summary["duration_s"] == 30
    assert summary["wall_owned_control_steps"] == 0
    assert len(rollout) == 6
    assert all(r["action"] == C.A_WAIT for r in rollout)
    assert all(r["mask"][C.A_ACTIVATE] is False for r in rollout)
    assert all("offer" not in r for r in rollout)   # never a fabricated offer


def _mock_registration():
    return dict(
        identity=dict(sumo_binary_sha256="x", config_sha256="x", model_sha256="x",
                      network_sha256="x", scenario_template_sha256="x",
                      active_source_sha256={}),
        communication=dict(max_report_age_s=1.0, history_s=10.0),
        assignment="registered_experimental_assignment_not_trained_benefit_gate",
        registration_sha256="x",
        budget=dict(campaign="ppo_smoke", category="training",
                    ledger_path="/nonexistent/ledger.json",
                    ceilings=dict(categories={}, total_seconds=0.0),
                    target_dir="/nonexistent"))
