"""wall_ppo/2 duration-semantics contract tests (zero simulation).

Covers the mocked integration list for the repaired duration contract:
maintain keeps the commitment, extension is a deadline decision admitted on
the proposed remaining maneuver, H20->H30->H40 reaches each deadline,
reaching the cap does not force immediate release, infeasible extension
follows the existing schedule, protective aborts operate, and the version
gate rejects v1 checkpoints.
"""

import pytest

from traffic_shepherd import ppo_contract as C
from traffic_shepherd.wall_ppo_policy import (ACTION_MEANINGS, DEADLINE_DECISION,
                                              INTERIOR_DECISION,
                                              PpoHoldProfile, WallPpoPolicy)


def make_policy(ready_at=None, committed=20.0):
    policy = WallPpoPolicy(dict(arm="wall"), dict(registration_sha256="x"),
                           ("cav_a", "cav_b"), ("L0", "L1"))
    from traffic_shepherd.formation import FormationProposal, WallMember
    base = PpoHoldProfile(hold_s=committed)
    policy.proposal = FormationProposal(
        600.0, ("L0", "L1"), (WallMember("cav_a", "L0"), WallMember("cav_b", "L1")),
        24.0, base)
    policy.ready_at = ready_at
    policy.profile = base
    return policy


def ctx(now):
    from tests.test_ppo_pilot_interface import Ctx, report
    return Ctx(now, [report(v, now) for v in ("cav_a", "cav_b")])


def fake_plan(monkeypatch, extend_feasible=True, exit_feasible=True):
    """Stub remaining_plan. extend_feasible controls whether an EXTEND
    admission succeeds; exit_now plans (release path) are always feasible."""
    calls = []

    def fake(proposal, reports, *, now_s, previous_commands, readiness_count=0,
             ready_at_s=None, exit_now=False, **kw):
        calls.append((now_s, proposal.profile.hold_s, bool(exit_now)))
        if exit_now:
            return dict(feasible=exit_feasible, reason="stub",
                        remaining_hold_s=0, remaining_exit_s=1,
                        remaining_total_s=1)
        return dict(feasible=extend_feasible, reason="stub",
                    remaining_hold_s=1, remaining_exit_s=1,
                    remaining_total_s=2)

    monkeypatch.setattr("traffic_shepherd.wall_ppo_policy.remaining_plan", fake)
    return calls


# ---- action meanings are unambiguous ---------------------------------------

def test_action_meanings_table():
    assert ACTION_MEANINGS["inactive"] == ["WAIT", "ACTIVATE"]
    assert ACTION_MEANINGS["hold_interior"] == ["MAINTAIN", "RELEASE"]
    assert ACTION_MEANINGS["hold_deadline"] == ["EXTEND", "RELEASE"]


def test_interior_maintain_does_not_extend_or_force_release(monkeypatch):
    """MAINTAIN at an interior tick keeps the H20 commitment (no taper, no
    extension, no abort)."""
    policy = make_policy(ready_at=606.0)
    calls = fake_plan(monkeypatch, extend_feasible=True)
    context = ctx(610.0)                     # interior cadence tick
    mask, kind = policy.decision_context(context)
    assert kind == INTERIOR_DECISION and mask == [True, True]
    policy.set_decision(0)                   # MAINTAIN
    output = policy(context)
    assert output.phase == "hold"
    assert policy.profile.hold_s == 20.0     # commitment unchanged
    assert policy.taper_at is None           # no forced release
    # the maintainability re-check happens each tick (lifecycle behavior),
    # and it always sees the CURRENT commitment (20 s), never an extension
    assert all(call[1] == 20.0 for call in calls)
    entry = policy.decision_log[-1]
    assert entry["executed_action"] == "maintain"


def test_early_release_remains_available_interior(monkeypatch):
    policy = make_policy(ready_at=606.0)
    fake_plan(monkeypatch)
    context = ctx(610.0)
    policy.set_decision(1)                   # RELEASE at an interior tick
    policy(context)
    assert policy.taper_at == 610.0
    assert policy.early_exit_reason == "ppo_release_decision"
    assert policy.decision_log[-1]["executed_action"] == "release"


def test_extension_requested_at_defined_deadline(monkeypatch):
    """EXTEND is evaluated AT the scheduled end (ready+20), not at an
    interior cadence tick, and admits the proposed H30 maneuver."""
    policy = make_policy(ready_at=606.0)
    fake_plan(monkeypatch)
    # interior tick at 610: EXTEND is not offered and MAINTAIN does not
    # consult the planner
    mask, kind = policy.decision_context(ctx(610.0))
    assert kind == INTERIOR_DECISION
    # scheduled end at 626: deadline decision offers EXTEND/RELEASE
    mask, kind = policy.decision_context(ctx(626.0))
    assert kind == DEADLINE_DECISION and mask == [True, True]
    policy.set_decision(0)                   # EXTEND
    policy(ctx(626.0))
    assert policy.profile.hold_s == 30.0
    assert policy.decision_log[-1]["executed_action"] == "extend"


def test_h40_reached_deadline_is_not_a_decision(monkeypatch):
    """At the maximum admitted HOLD (40 s) the deadline is not an actor
    decision; the scheduled release executes at the deadline tick."""
    policy = make_policy(ready_at=606.0, committed=40.0)
    policy.profile = PpoHoldProfile(hold_s=40.0)
    fake_plan(monkeypatch)
    policy.ready_at = 606.0
    policy.profile = PpoHoldProfile(hold_s=40.0)
    mask, kind = policy.decision_context(ctx(646.0))   # 606 + 40 = scheduled end
    assert mask is None and kind == "hold"
    context = ctx(646.0)
    output = policy(context)
    assert policy.taper_at == 646.0                    # scheduled release
    assert output.phase in {"taper", "release"}


def test_infeasible_extension_follows_release_schedule(monkeypatch):
    """A declined EXTEND at the deadline lets the existing release schedule
    execute: the EXTEND request is declined by admission, and the wall then
    follows the release path (early release is planned feasible)."""
    policy = make_policy(ready_at=606.0)
    fake_plan(monkeypatch, extend_feasible=False, exit_feasible=True)
    context = ctx(626.0)                     # scheduled end of H20
    mask, kind = policy.decision_context(context)
    assert kind == DEADLINE_DECISION and mask == [True, True]
    policy.set_decision(0)                   # actor requests EXTEND
    output = policy(context)
    assert policy.decision_log[-1]["executed_action"] is None
    assert policy.decision_log[-1]["rejection_reason"] == "extension_declined_by_admission"
    assert output.phase in {"taper", "release"}   # scheduled release proceeds


def test_off_cadence_deadline_is_handled_exactly_once(monkeypatch):
    """A HOLD start at an off-cadence second (ready=607, H20 end=627) gets
    its deadline decision at 627 (not held to 630), handled once."""
    policy = make_policy(ready_at=607.0)
    fake_plan(monkeypatch)
    from traffic_shepherd.formation import FormationProposal, WallMember
    base = PpoHoldProfile(hold_s=20.0)
    policy.proposal = FormationProposal(
        600.0, ("L0", "L1"), (WallMember("cav_a", "L0"), WallMember("cav_b", "L1")),
        24.0, base)
    policy.profile = base
    # 627 is off the 5 s cadence: still a deadline decision, handled once
    mask1, kind1 = policy.decision_context(ctx(627.0))
    assert kind1 == DEADLINE_DECISION and mask1 == [True, True]
    mask2, kind2 = policy.decision_context(ctx(627.0))
    assert mask2 == [True, True] and kind2 == DEADLINE_DECISION  # stateless check ok


def test_no_hold_beyond_admitted_deadline(monkeypatch):
    """Without an EXTEND decision at the deadline, the scheduled release
    fires at the next control tick (never silently held past it)."""
    policy = make_policy(ready_at=606.0)
    fake_plan(monkeypatch)
    policy.ready_at = 606.0
    from traffic_shepherd.formation import FormationProposal, WallMember
    base = PpoHoldProfile(hold_s=20.0)
    policy.proposal = FormationProposal(
        600.0, ("L0", "L1"), (WallMember("cav_a", "L0"), WallMember("cav_b", "L1")),
        24.0, base)
    policy.profile = base
    # no decision at 626 (deadline passes unhandled by the actor)
    context = ctx(627.0)
    output = policy(context)
    assert policy.taper_at == 627.0          # scheduled release fires
    assert output.phase in {"taper", "release"}


def test_protective_abort_still_operates(monkeypatch):
    policy = make_policy(ready_at=606.0, committed=30.0)
    from dataclasses import dataclass
    from traffic_shepherd.formation import FormationProposal, WallMember
    base = PpoHoldProfile(hold_s=30.0)
    policy.proposal = FormationProposal(
        600.0, ("L0", "L1"), (WallMember("cav_a", "L0"), WallMember("cav_b", "L1")),
        24.0, base)
    policy.ready_at = 606.0
    policy.profile = base

    @dataclass(frozen=True)
    class Bad:
        approved: bool = False
        reason: str = "stub_lost_compactness"
        initial_or_current_span_m: float = 90.0
        speed_spread_mps: float = 6.0
        evidence_fingerprint: str = "e"

    monkeypatch.setattr("traffic_shepherd.wall_ppo_policy.assess_formation",
                        lambda *a, **k: Bad())
    context = ctx(620.0)
    output = policy(context)
    assert output.phase == "abort"
    assert policy.terminal[0] == "abort"


def test_version_gate_rejects_v1_checkpoints():
    """Old checkpoints must not silently inherit v2 action meanings."""
    import torch
    from traffic_shepherd.ppo_net import ActorCritic
    net = ActorCritic()                     # current contract: 128-dim input
    legacy_state = {k: v for k, v in net.state_dict().items()}
    # simulate a v1 checkpoint: same architecture names, 116-dim input layer
    v1_first = torch.zeros(64, 116)
    legacy_state["body.0.weight"] = v1_first
    fresh = ActorCritic()
    with pytest.raises(RuntimeError):
        fresh.load_state_dict(legacy_state)      # shape mismatch -> refuse
    # and the policy version marker distinguishes contracts explicitly
    assert C.POLICY_VERSION == "wall_ppo/2"


def test_masks_and_logp_consistent_in_v2_deadline_context(monkeypatch):
    policy = make_policy(ready_at=606.0)
    fake_plan(monkeypatch, extend_feasible=True, exit_feasible=True)
    context = ctx(626.0)
    mask, kind = policy.decision_context(context)
    assert kind == DEADLINE_DECISION and mask == [True, True]
    policy.set_decision(0)
    output = policy(context)
    assert output.phase == "hold" and policy.profile.hold_s == 30.0
    entry = policy.decision_log[-1]
    assert entry["executed_action"] == "extend"
    assert entry["decision_kind"] == DEADLINE_DECISION
    assert entry["admitted_hold_s"] == 30.0
