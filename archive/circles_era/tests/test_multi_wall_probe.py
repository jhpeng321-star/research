from dataclasses import replace
from types import SimpleNamespace

import pytest

from traffic_shepherd.multi_wall_probe import (MultiWallProbePolicy, ProbeExtensionProfile,
                                                select_upstream)
from traffic_shepherd.cohort_study import LANES
from traffic_shepherd.lifecycle_wall import LifecycleProfile
from traffic_shepherd.formation import make_proposal
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import SensorFrame
from traffic_shepherd.paired_experiments import PolicyInput, PolicyOutput


def test_extension_profile_is_frozen_40_seconds():
    p = ProbeExtensionProfile(**dict(ProbeExtensionProfile().__dict__))
    assert p.hold_s == 40.0
    with pytest.raises(ValueError):
        ProbeExtensionProfile(hold_s=30.0)


def test_upstream_selection_excludes_first_and_uses_fresh_reports():
    def report(v, lane, d, now=700.):
        frame = SensorFrame(vehicle_id=v, lane_id=lane, lane_index=0,
                            distance_to_merge_m=d, speed_mps=20., acceleration_mps2=0.,
                            timestamp_s=now, position_m=100., allowed_speed_mps=30.,
                            front_coverage_m=100., rear_coverage_m=100.)
        return CAVReport(frame=frame, sent_at_s=now, owner="damping")
    rs = (report("old0", "1191554636_0", 500.), report("old1", "1191554636_1", 500.),
              report("new0", "1191554636_0", 2000.), report("new1", "1191554636_1", 2100.))
    got = select_upstream(rs, now_s=700., excluded_ids={"old0", "old1"}, minimum_distance_m=500.)
    assert got["selected"]["member_ids"] == ["new0", "new1"]
    stale = select_upstream(rs, now_s=702., excluded_ids=set(), minimum_distance_m=0.)
    assert stale["selected"] is None


def test_policy_has_one_time_second_attempt_guard():
    obj = object.__new__(MultiWallProbePolicy)
    obj.second_attempted = True
    assert obj.second_attempted is True


def test_extension_hook_is_mode_specific(monkeypatch):
    context = PolicyInput(600., (), (), {}, frozenset())
    obj = object.__new__(MultiWallProbePolicy)
    obj.mode = "distinct_upstream_h20"
    obj.second = None
    obj.first_terminal = None
    obj.second_attempted = False
    obj.receipt = {"branch": None, "selection": None,
                   "second_activation_attempted": False}
    obj.first = lambda ctx: PolicyOutput()
    monkeypatch.setattr(MultiWallProbePolicy, "_extension",
                        lambda self, ctx: (_ for _ in ()).throw(AssertionError("wrong mode")))
    assert obj(context).phase == "damping"


def _pair(now=640.):
    out = []
    for v, lane, d in (("a", LANES[0], 1000.), ("b", LANES[1], 1100.)):
        frame = SensorFrame(v, now, 20., 0., 30., 0, d, lane_id=lane,
                            position_m=100., front_coverage_m=100., rear_coverage_m=100.)
        out.append(CAVReport(frame, now, "damping"))
    return tuple(out)


def test_extension_acceptance_mutates_only_after_full_plan(monkeypatch):
    reports = _pair()
    proposal = make_proposal(reports, now_s=640., required_lanes=LANES,
                             member_ids=("a", "b"), profile=LifecycleProfile())
    delegate = SimpleNamespace(ready_at=620., taper_at=None, terminal=None, proposal=proposal,
                               profile=LifecycleProfile(), previous_commands={}, readiness=3)
    obj = object.__new__(MultiWallProbePolicy)
    obj.first = delegate
    obj.receipt = {"branch": None}
    monkeypatch.setattr("traffic_shepherd.multi_wall_probe.remaining_plan",
                        lambda *args, **kwargs: {"feasible": True, "reason": "ok"})
    obj._extension(PolicyInput(640., reports, (), {}, frozenset()))
    assert delegate.profile.hold_s == 40.
    assert obj.receipt["branch"]["selected"] == "extend"


def test_extension_rejection_preserves_original(monkeypatch):
    reports = _pair()
    proposal = make_proposal(reports, now_s=640., required_lanes=LANES,
                             member_ids=("a", "b"), profile=LifecycleProfile())
    original = proposal
    delegate = SimpleNamespace(ready_at=620., taper_at=None, terminal=None, proposal=proposal,
                               profile=LifecycleProfile(), previous_commands={}, readiness=3)
    obj = object.__new__(MultiWallProbePolicy)
    obj.first = delegate
    obj.receipt = {"branch": None}
    monkeypatch.setattr("traffic_shepherd.multi_wall_probe.remaining_plan",
                        lambda *args, **kwargs: {"feasible": False, "reason": "unsafe"})
    obj._extension(PolicyInput(640., reports, (), {}, frozenset()))
    assert delegate.proposal is original and delegate.profile.hold_s == 20.
    assert obj.receipt["branch"]["selected"] == "decline"


def test_receipt_details_are_snapshots():
    obj = object.__new__(MultiWallProbePolicy)
    obj.mode, obj.second, obj.first_terminal, obj.second_attempted = "distinct_upstream_h20", None, None, False
    obj.receipt = {"branch": None, "selection": None, "second_activation_attempted": False}
    obj.first = lambda ctx: PolicyOutput()
    out = obj(PolicyInput(600., (), (), {}, frozenset()))
    obj.receipt["selection"] = {"changed": True}
    assert out.details["multi_wall_probe"]["selection"] is None


def test_selection_rejects_future_and_duplicate_reports():
    reports = list(_pair())
    future = CAVReport(replace(reports[0].frame, timestamp_s=641.), 641., "damping")
    assert select_upstream(reports[:1] + [future], now_s=640., excluded_ids=set(),
                           minimum_distance_m=0.)["selected"] is None
    duplicate = reports + [reports[0]]
    assert select_upstream(duplicate, now_s=640., excluded_ids=set(),
                           minimum_distance_m=0.)["selected"] is None


def test_selection_rejects_missing_first_distance():
    reports = list(_pair())
    missing = CAVReport(replace(reports[0].frame, distance_to_merge_m=None), 640., "damping")
    got = select_upstream([missing, reports[1]], now_s=640., excluded_ids={"a"},
                          minimum_distance_m=0.)
    assert got["selected"] is None


def test_abort_cannot_become_second_activation():
    obj = object.__new__(MultiWallProbePolicy)
    obj.mode, obj.second, obj.first_terminal, obj.second_attempted = "distinct_upstream_h20", None, None, False
    obj.receipt = {"branch": None, "selection": None, "second_activation_attempted": False}
    obj.job = {"policy_spec": {"member_ids": []}}
    obj.first = lambda ctx: PolicyOutput({}, "abort", "safety")
    first = obj(PolicyInput(600., (), (), {}, frozenset()))
    assert first.phase == "abort" and obj.second is None
    assert obj.second_attempted is False


def test_protective_abort_after_engagement_attempts_once(monkeypatch):
    class First:
        def __init__(self): self.calls = 0
        def __call__(self, ctx):
            self.calls += 1
            return PolicyOutput({"a": -.1} if self.calls == 1 else {},
                                "assembly" if self.calls == 1 else "abort", "protective",
                                ("a",))
    obj = object.__new__(MultiWallProbePolicy)
    obj.mode, obj.second, obj.first_terminal, obj.second_attempted = "distinct_upstream_h20", None, None, False
    obj.first_engaged = False
    obj.receipt = {"branch": None, "selection": None, "second_activation_attempted": False}
    obj.job = {"policy_spec": {"member_ids": ["a"]}}
    obj.first = First()
    attempts = []
    monkeypatch.setattr(obj, "_start_second", lambda ctx: (attempts.append(ctx.now_s), setattr(obj, "second_attempted", True)))
    assert obj(PolicyInput(600., (), (), {}, frozenset())).phase == "assembly"
    assert obj(PolicyInput(601., (), (), {}, frozenset())).phase == "abort"
    # The next tick is exactly the one allowed replacement opportunity.
    obj.first_terminal = {"time_s": 601., "phase": "abort", "reason": "protective"}
    obj.first = lambda ctx: PolicyOutput({}, "abort", "protective")
    obj(PolicyInput(602., (), (), {}, frozenset()))
    obj(PolicyInput(603., (), (), {}, frozenset()))
    assert attempts == [602.]
