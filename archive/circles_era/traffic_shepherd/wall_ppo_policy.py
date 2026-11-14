"""Action-driven wall contract for the RSU PPO pilot (wall_ppo/1).

The physics, command law, protective exits and identical-damping handback are
byte-for-byte the lifecycle_v2 machinery (coordination_requests, resolve_command,
coast_terminal, remaining_plan admission). The ONLY change against
finite_lifecycle_wall/2 is who decides the schedule:

- ACTIVATE: the pilot actor may start the existing assembly at a decision tick
  (>= 120 s, <= 1800 s, at most once per episode) when a feasible proposal for
  the frozen per-scenario cohort exists in current report evidence. The mask is
  a cheap acquired-evidence check; the full remaining_plan admission is
  re-checked at execution and a refusal is recorded, never silently retried.
- HOLD: at each decision tick the actor chooses RELEASE (taper at this tick) or
  CONTINUE (+10 s, bounded by MAX_HOLD 40 s). CONTINUE is committed only after
  the hold_extend/2-style admission evaluates the PROPOSED remaining maneuver
  (extended hold + exit); an infeasible continuation is recorded and declined,
  leaving the current commitment unchanged. This is the explicit, verified
  generalization of hold_extend/1's single extension to repeated increments;
  it is NOT silent reuse of that contract.
- Everything else (assembly, protective aborts, coast terminal, handback) is
  autonomous and unchanged.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import isfinite

from .formation import FormationProfile, make_proposal, assess_formation, _local_clearance_problem
from .gentle_wall import GentleProfile, resolve_command
from .lifecycle_wall import (TERMINAL_CONTRACT, VERSION as LIFECYCLE_VERSION,
                             coast_terminal, coordination_requests, remaining_plan)
from .paired_experiments import PolicyOutput
from . import ppo_contract as C

ACTION_MEANINGS = C.ACTION_MEANINGS

POLICY_VERSION = "wall_ppo/2"
REGISTERED_HOLDS = (C.BASE_HOLD_S, C.BASE_HOLD_S + C.CONTINUE_INCREMENT_S, C.MAX_HOLD_S)

# Action meanings per decision context (v2 duration semantics). v1 exposed
# CONTINUE while an admitted HOLD was in progress, so committing to the
# maximum forced RELEASE at the next tick even when most of the admitted
# duration was undelivered. v2 separates maintaining an admitted maneuver
# from extending it:
#   inactive:      0 = WAIT, 1 = ACTIVATE (a currently feasible proposal)
#   hold interior: 0 = MAINTAIN (keep the admitted commitment), 1 = RELEASE
#   hold deadline: 0 = EXTEND (+increment, admitted on the proposed
#                  remaining maneuver), 1 = RELEASE
# At the maximum admitted HOLD the deadline is not an actor decision: the
# scheduled release executes when the admitted maximum is actually reached.
DEADLINE_DECISION = "hold_deadline"
INTERIOR_DECISION = "hold_interior"


@dataclass(frozen=True)
class PpoHoldProfile(GentleProfile):
    """The frozen lifecycle_v2 contract with a registered total HOLD (20/30/40)."""

    version: str = POLICY_VERSION

    def __post_init__(self):
        old = asdict(self)
        old.pop("acceleration_change_limit_mps3")
        old["acceleration_min_mps2"] = -3.
        FormationProfile(**old)
        if (self.version != POLICY_VERSION or self.acceleration_min_mps2 != -.3
                or self.acceleration_change_limit_mps3 != .15
                or self.hold_s not in REGISTERED_HOLDS
                or any(type(v) is bool for v in asdict(self).values())):
            raise ValueError("wall_ppo profile must be the frozen contract with a "
                             "registered total HOLD")


class DampingPolicy:
    """The matched-reference arm: identical CIRCLES damping, no wall commands."""

    def __init__(self, job=None, registration=None):
        self.job, self.registration = job, registration

    def __call__(self, context):
        return PolicyOutput()


class WallPpoPolicy:
    """Registered diagnostic treatment driven by pilot actor decisions."""

    def __init__(self, job, registration, member_ids, required_lanes,
                 activation_window=None):
        if job["arm"] != "wall":
            raise ValueError("wall_ppo is an explicit wall treatment")
        self.job, self.registration = deepcopy(job), registration
        self.member_ids = tuple(member_ids)
        self.required_lanes = tuple(required_lanes)
        # Corridor default is the registered contract window; mechanics
        # fixtures may pass a declared narrower window to exercise the chain
        # inside 120 s. It never widens the contract window.
        self.activation_window = tuple(activation_window or
            (C.EARLIEST_ACTIVATION_S, C.LATEST_ACTIVATION_S))
        self.profile = PpoHoldProfile(hold_s=C.BASE_HOLD_S)
        self.proposal = self.terminal = None
        self.last_time = None
        self.terminal_at = None
        self.previous_commands = {}
        self.readiness = 0
        self.ready_at = self.taper_at = None
        self.early_exit_reason = None
        self.activation_used = 0
        self.decision_log = []
        self.pending_action = None
        self.last_context = None

    # ---- pilot interface -------------------------------------------------

    def phase_now(self):
        if self.proposal is None or self.terminal:
            return "inactive" if self.proposal is None else "terminal"
        if self.taper_at is not None:
            return "taper"
        if self.ready_at is not None:
            return "hold"
        return "assembly"

    def active_member_ids(self):
        """Cohort whose reports the observation features describe. For the
        fixed-cohort wall_ppo/2 contract this is exactly member_ids;
        subclasses with activation-time binding (wall_ppo/3) override it."""
        return tuple(self.member_ids)

    def activation_feasible(self, context):
        """Cheap acquired-evidence mask; the full admission runs at execution."""
        earliest, latest = self.activation_window
        if (self.activation_used >= C.MAX_ACTIVATIONS
                or not (earliest <= context.now_s <= latest)):
            return False
        try:
            proposal = make_proposal(tuple(context.reports), now_s=context.now_s,
                required_lanes=self.required_lanes, member_ids=self.member_ids,
                profile=self.profile)
        except ValueError:
            return False
        assessment = assess_formation(proposal, tuple(context.reports), now_s=context.now_s)
        return bool(assessment.approved)

    def decision_context(self, context):
        """(mask, decision_kind) for the pilot actor at a control tick.

        v2 scheduling: inactive decisions at cadence ticks; interior hold
        decisions at cadence ticks strictly before the scheduled end
        (MAINTAIN/RELEASE); the deadline decision AT the scheduled end
        (EXTEND/RELEASE) when extension capacity remains. Nothing is
        decided outside these events; the scheduled release executes when
        the admitted maximum is reached.
        """
        now = context.now_s
        phase = self.phase_now()
        if phase == "inactive":
            if now % C.CADENCE_S != 0:
                return None, phase
            return [True, self.activation_feasible(context)], phase
        if phase == "hold":
            deadline = self.ready_at + self.profile.hold_s
            at_deadline = now >= deadline - 1e-9
            if at_deadline:
                # deadline decision only if extension capacity remains;
                # otherwise the scheduled release executes (not a choice)
                if self.profile.hold_s + C.CONTINUE_INCREMENT_S <= C.MAX_HOLD_S:
                    return [True, True], DEADLINE_DECISION
                return None, phase
            if now % C.CADENCE_S == 0:
                return [True, True], INTERIOR_DECISION
            return None, phase
        return None, phase

    def set_decision(self, action):
        if self.pending_action is not None:
            raise ValueError("unconsumed pilot decision")
        self.pending_action = int(action)

    # ---- core lifecycle --------------------------------------------------

    def _log_decision(self, now, mask, requested, executed, reason=None,
                      kind=None, admission=None):
        entry = dict(
            time_s=now, phase=self.phase_now(),
            available_actions=None if mask is None else list(mask),
            requested_action=requested, executed_action=executed,
            rejection_reason=reason)
        if kind is not None:
            entry["decision_kind"] = kind
            entry["admitted_hold_s"] = self.profile.hold_s
            entry["ready_at_s"] = self.ready_at
            entry["scheduled_end_s"] = (self.ready_at + self.profile.hold_s
                                        if self.ready_at is not None else None)
            entry["proposal_sha256"] = (self.proposal.fingerprint
                                        if self.proposal is not None else None)
        if admission is not None:
            entry["admission"] = admission
        self.decision_log.append(entry)

    def __call__(self, context):
        now, reports = context.now_s, tuple(context.reports)
        if self.last_time is not None and now != self.last_time + 1:
            raise ValueError("wall_ppo requires exact one-second clock")
        self.last_time = now
        requested = self.pending_action
        self.pending_action = None
        self.last_context = context

        # Agent decision handling (v2: inactive cadence ticks; interior hold
        # cadence ticks = MAINTAIN/RELEASE; the scheduled-end tick =
        # EXTEND/RELEASE; at the maximum the scheduled release executes).
        if requested is not None:
            mask, kind = self.decision_context(context)
            if mask is None:
                self._log_decision(now, None, requested, None, "no_decision_phase",
                                   kind=kind)
                requested = None
            elif kind == "inactive":
                if requested == C.A_ACTIVATE and mask[C.A_ACTIVATE]:
                    executed = self._activate(context)
                    self._log_decision(now, mask, requested,
                                       "activate" if executed else None,
                                       None if executed else "activation_refused_at_execution")
                elif requested == C.A_ACTIVATE:
                    self._log_decision(now, mask, requested, None,
                                       "activate_masked_or_out_of_window")
                else:
                    self._log_decision(now, mask, requested, "wait")
            elif kind == INTERIOR_DECISION:
                # v2 interior: 0 = MAINTAIN, 1 = RELEASE (legacy A_RELEASE=0
                # must NOT be used here — it collides with MAINTAIN)
                if requested == C.A_HOLD_RELEASE:
                    if self.taper_at is None:
                        self.taper_at = now
                        self.early_exit_reason = "ppo_release_decision"
                    self._log_decision(now, mask, requested, "release", kind=kind)
                else:
                    self._log_decision(now, mask, requested, "maintain", kind=kind)
            elif kind == DEADLINE_DECISION:
                if requested == C.A_EXTEND:        # EXTEND (deadline index 0)
                    before = self.profile.hold_s
                    executed = self._extend(context)
                    self._log_decision(now, mask, requested,
                                       "extend" if executed else None,
                                       None if executed else
                                       "extension_declined_by_admission", kind=kind,
                                       admission=dict(
                                           requested_hold_s=before + C.CONTINUE_INCREMENT_S,
                                           admitted_hold_s=self.profile.hold_s))
                else:
                    if self.taper_at is None:
                        self.taper_at = now
                        self.early_exit_reason = "ppo_release_decision"
                    self._log_decision(now, mask, requested, "release", kind=kind)

        if self.proposal is None:
            return PolicyOutput(details=dict(wall_ppo_version=POLICY_VERSION,
                lifecycle_base=LIFECYCLE_VERSION, terminal_contract=TERMINAL_CONTRACT,
                activation_used=self.activation_used,
                decisions=self.decision_log[-1:]))

        proposal, p = self.proposal, self.profile
        assessment = assess_formation(proposal, reports, now_s=now)
        frames = {r.frame.vehicle_id: r.frame for r in reports}
        details = dict(wall_ppo_version=POLICY_VERSION, lifecycle_base=LIFECYCLE_VERSION,
            assessment=asdict(assessment), span_m=assessment.initial_or_current_span_m,
            speed_spread_mps=assessment.speed_spread_mps,
            target_speed_mps=proposal.target_speed_mps,
            readiness_samples_required=p.readiness_samples,
            committed_hold_s=p.hold_s,
            activation_used=self.activation_used,
            decisions=self.decision_log[-1:],
            absolute_wall_min_mps2=p.acceleration_min_mps2,
            command_ramp_mps3=p.acceleration_change_limit_mps3,
            terminal_contract=TERMINAL_CONTRACT,
            experimental_authorization="registered_experimental_assignment_not_trained_benefit_gate")

        def emit(phase, reason, commands=None):
            if phase in {"refused", "abort", "release"} and self.terminal is None:
                self.terminal_at = now
                self.terminal = phase, reason
            values = commands or {}
            details.update(readiness_count=self.readiness,
                assembly_completed_at_s=self.ready_at,
                hold_commitment_started=self.ready_at is not None,
                exit_started_at_s=self.taper_at, early_exit_reason=self.early_exit_reason,
                nominal_maximum_hold_s=C.MAX_HOLD_S,
                actual_hold_elapsed_s=0. if self.ready_at is None else max(0.,
                    (self.taper_at if self.taper_at is not None else
                     self.terminal_at if self.terminal_at is not None else now) - self.ready_at),
                terminal_at_s=self.terminal_at, terminal_contract=TERMINAL_CONTRACT,
                override_released=not bool(values), continued_wall_ownership=bool(values),
                compact=details.get("compact"))
            if values:
                self.previous_commands = dict(values)
            return PolicyOutput(values, phase, reason, proposal.member_ids,
                proposal.fingerprint, assessment.evidence_fingerprint, details)

        if self.terminal:
            return emit(*self.terminal)
        failure_phase = "refused" if now == proposal.start_s else "abort"
        if not assessment.approved:
            return emit(failure_phase, "wall_ppo_safety_or_information:" + assessment.reason)
        members = set(proposal.member_ids)
        if not members <= set(context.damping_commands) | set(context.damping_native_ids) or any(
                not isfinite(context.damping_commands[v]) for v in members
                if v in context.damping_commands):
            return emit(failure_phase, "wall_ppo_missing_current_local_damping_reference")
        rows = [frames[v] for v in proposal.member_ids]
        compact = (assessment.initial_or_current_span_m <= p.formed_span_m
                   and assessment.speed_spread_mps <= p.formed_speed_spread_mps)
        details["compact"] = compact
        if self.ready_at is None and self.taper_at is None:
            self.readiness = self.readiness + 1 if _settled(rows, proposal) else 0
            if (self.readiness < p.readiness_samples
                    and now - proposal.start_s >= p.assembly_deadline_s):
                return emit("abort", "wall_ppo_assembly_deadline_without_measured_readiness")
        if self.ready_at is not None and self.taper_at is None:
            if not compact:
                return emit("abort", "wall_ppo_lost_compact_hold")
            if now >= self.ready_at + p.hold_s:
                self.taper_at = now
                self.early_exit_reason = self.early_exit_reason or "scheduled_hold_end"
        if now >= C.LATEST_AUTO_RELEASE_S and self.taper_at is None and self.ready_at is not None:
            self.taper_at = now
            self.early_exit_reason = "episode_end_forced_release"
        if self.taper_at is not None and coast_terminal(rows, self.previous_commands, p):
            return emit("release", "wall_ppo_coast_terminal_return_to_identical_damping")
        if self.taper_at is not None and now - self.taper_at >= p.taper_s:
            return emit("abort", "wall_ppo_exit_deadline_without_measured_coast_terminal")
        plan = remaining_plan(proposal, reports, now_s=now, previous_commands=self.previous_commands,
            readiness_count=self.readiness, ready_at_s=self.ready_at,
            exit_started_at_s=self.taper_at)
        details["wall_ppo_plan"] = dict(commitment=plan)
        if (self.ready_at is None and self.taper_at is None
                and plan.get("maintainable_hold_ready_now")):
            if self.readiness < p.readiness_samples:
                raise AssertionError("maintainable HOLD cannot bypass measured readiness")
            self.ready_at = now
        if not plan["feasible"]:
            exit_plan = remaining_plan(proposal, reports, now_s=now,
                previous_commands=self.previous_commands, readiness_count=self.readiness,
                ready_at_s=self.ready_at, exit_started_at_s=self.taper_at, exit_now=True)
            details["wall_ppo_plan"]["exit_now"] = exit_plan
            if not exit_plan["feasible"]:
                return emit("abort", "wall_ppo_no_feasible_owned_exit:" + exit_plan["reason"])
            if self.taper_at is None:
                self.taper_at, self.early_exit_reason = now, plan["reason"]
            if coast_terminal(rows, self.previous_commands, p):
                return emit("release", "wall_ppo_early_coast_terminal_return_to_identical_damping")
        phase = "taper" if self.taper_at is not None else "hold" if self.ready_at is not None else "assembly"
        requests = coordination_requests(rows, proposal, exiting=phase == "taper")
        commands, resolutions = {}, {}
        for f in rows:
            acceleration, evidence = resolve_command(f, requests[f.vehicle_id][0], p,
                previous=self.previous_commands.get(f.vehicle_id))
            resolutions[f.vehicle_id] = dict(evidence, desired_speed_mps=requests[f.vehicle_id][1])
            details["command_resolution"] = resolutions
            if acceleration is None:
                return emit(failure_phase, "wall_ppo_measured_response_incompatible")
            problem = _local_clearance_problem(f, p, now, acceleration=acceleration)
            if problem:
                return emit(failure_phase, "wall_ppo_protective_clearance:" + problem[1])
            commands[f.vehicle_id] = acceleration
        return emit(phase, "wall_ppo_finite_remaining_commitment_and_coast_exit", commands)

    # ---- pilot actions ---------------------------------------------------

    def _activate(self, context):
        try:
            proposal = make_proposal(tuple(context.reports), now_s=context.now_s,
                required_lanes=self.required_lanes, member_ids=self.member_ids,
                profile=self.profile)
            plan = remaining_plan(proposal, tuple(context.reports), now_s=context.now_s,
                previous_commands={}, readiness_count=0)
        except ValueError:
            return False
        if not plan["feasible"]:
            return False
        self.proposal = proposal
        self.activation_used += 1
        return True

    def _extend(self, context):
        """v2 EXTEND at the scheduled end: admit the PROPOSED remaining
        maneuver of the extended commitment; commit only if feasible."""
        proposed_total = self.profile.hold_s + C.CONTINUE_INCREMENT_S
        if proposed_total > C.MAX_HOLD_S:
            return False
        candidate = PpoHoldProfile(hold_s=proposed_total)
        candidate_proposal = replace(self.proposal, profile=candidate)
        plan = remaining_plan(candidate_proposal, tuple(context.reports), now_s=context.now_s,
            previous_commands=self.previous_commands, readiness_count=self.readiness,
            ready_at_s=self.ready_at)
        if not plan["feasible"]:
            return False
        self.proposal = candidate_proposal
        self.profile = candidate
        return True


def _settled(rows, proposal):
    p = proposal.profile
    distances = [f.distance_to_merge_m for f in rows]
    speeds = [f.speed_mps for f in rows]
    return (max(distances) - min(distances) <= p.formed_span_m
        and max(speeds) - min(speeds) <= p.formed_speed_spread_mps
        and all(abs(v - proposal.target_speed_mps) <= p.target_speed_tolerance_mps for v in speeds))


class ScriptedReportRulePolicy(WallPpoPolicy):
    """Predeclared simple report-based comparator on the identical contract.

    ACTIVATE at the first decision tick in [600, 1200] where the frozen cohort
    reports fresh with mean speed < 18 m/s and the cheap activation mask passes;
    otherwise WAIT. Always RELEASE at the first HOLD decision tick (20 s HOLD).
    No learning, no tuning: fixed before any development outcome was seen.
    """

    TRIGGER_MEAN_SPEED_MPS = 18.0

    def __init__(self, job, registration, member_ids, required_lanes):
        super().__init__(job, registration, member_ids, required_lanes)

    def decide(self, observation, mask, phase, now):
        if phase == "inactive":
            if (mask is not None and mask[C.A_ACTIVATE]
                    and 600.0 <= now <= 1200.0
                    and self._cohort_mean_speed() is not None
                    and self._cohort_mean_speed() < self.TRIGGER_MEAN_SPEED_MPS):
                return C.A_ACTIVATE
            return C.A_WAIT
        if phase == "hold":
            return C.A_RELEASE
        return None

    def _cohort_mean_speed(self):
        context = self.last_context
        if context is None:
            return None
        fresh = {r.frame.vehicle_id: r for r in context.reports
                 if context.now_s - r.frame.timestamp_s <= 1.0}
        speeds = [fresh[v].frame.speed_mps for v in self.member_ids if v in fresh]
        return sum(speeds) / len(speeds) if len(speeds) == len(self.member_ids) else None
