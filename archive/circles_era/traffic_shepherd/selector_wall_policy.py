"""wall_ppo/3: causal current-report candidate selection for the PPO policy.

Versioned controller (PP-11; repaired in PP-12, selector causal_selector/2).
The ONLY change vs wall_ppo/2 is the inactive-phase WHO-may-activate
decision — a bounded deterministic search over current acquired reports
offers the minimum-maneuver feasible pair; HOLD semantics, wall profile,
command limits and protective exits remain EXACTLY wall_ppo/2.

PP-12 repair contract (supersedes the PP-11 integration):

1. OFFERED PROPOSAL BINDING. At every eligible inactive decision tick the
   policy binds an explicit OfferedProposal BEFORE observations are built:
   candidate member IDs, the exact profile and target, the proposal
   fingerprint, the report-evidence identity and the decision time. The
   feature builder describes the offered cohort while inactive and the
   committed proposal after activation — a feasible candidate is never
   hidden behind zero candidate features.

2. OFFER LIFETIME. An offer is bound at its decision tick and is consumed
   by ACTIVATE or cleared by WAIT (OFFER_TTL_TICKS = 0). The next eligible
   decision performs a fresh search from then-current reports. No future
   reference-selected identities or full-state information ever enter the
   offer: its members come only from causally eligible acquired reports.

3. EXECUTION RE-CHECKS THE APPROVED OFFER. set_decision(ACTIVATE)
   snapshots the bound offer as the approved offer; execution re-assesses
   EXACTLY that offer (same member IDs, profile, lanes and target) against
   execution-time acquired evidence through the unchanged physical checks.
   Candidate ranking is never re-run at execution, so a later/better cohort
   can never silently replace the approved one. The only permitted update
   is refreshing the evidence-derived proposal to the offered target speed,
   which never changes the action the actor selected. A stale, missing or
   infeasible offer is refused ATOMICALLY: no substitution, no consumption
   of the one activation permission, no mutation of the lifecycle.

4. ELIGIBILITY AND CONFIGURATION CONSISTENCY. The activation window,
   decision/search cadence, one-activation-per-episode rule, no-reactivation
   after terminal handback, causal report eligibility and freshness limits
   are enforced exactly as in wall_ppo/2 (window and count here, cadence in
   decision_context, eligibility inside the search). The search, the offer
   and the execution admission all use the SAME required lanes and wall
   profile objects.
"""
from dataclasses import dataclass, replace
import time

from .formation import assess_formation, make_proposal
from .lifecycle_wall import remaining_plan
from .selector_policy import search_current_proposals
from .wall_ppo_policy import WallPpoPolicy
from . import ppo_contract as C

SELECTOR_POLICY_VERSION = "wall_ppo/3"
SELECTOR_VERSION = "causal_selector/2"
OFFER_TTL_TICKS = 0   # an offer lives for exactly its own decision tick


@dataclass(frozen=True)
class OfferedProposal:
    """The exact proposal bound at a decision tick and offered to the actor.

    member_ids/profile/required_lanes/target_speed_mps are the identity the
    actor approves; fingerprint/evidence_fingerprint/decision_time_s bind
    the report evidence it was derived from. refusal records an execution-
    time refusal of this exact offer (atomic, never substituted).
    """
    member_ids: tuple
    required_lanes: tuple
    profile: object
    target_speed_mps: float
    fingerprint: str
    evidence_fingerprint: str
    decision_time_s: float
    remaining_total_s: float | None
    assessed_pairs: int = 0
    refusal: dict | None = None

    def identity_dict(self, *, committed=False):
        return dict(selector_version=SELECTOR_VERSION,
                    member_ids=list(self.member_ids),
                    required_lanes=list(self.required_lanes),
                    target_speed_mps=self.target_speed_mps,
                    proposal_fingerprint=self.fingerprint,
                    evidence_fingerprint=self.evidence_fingerprint,
                    decision_time_s=self.decision_time_s,
                    remaining_total_s=self.remaining_total_s,
                    assessed_pairs=self.assessed_pairs,
                    committed=committed,
                    refusal=self.refusal)


def _current_tick_feature(features, name):
    """Read a named feature from the actor's feature vector. The env actor
    boundary passes the CURRENT-tick FEATURE_NAMES vector; a full stacked
    HISTORY_STACK window (worker path) is also accepted."""
    if features is None or not hasattr(features, "__len__"):
        return None
    n = len(C.FEATURE_NAMES)
    if len(features) == n:
        return features[C.FEATURE_NAMES.index(name)]
    if len(features) >= n * C.HISTORY_STACK:
        return features[n * (C.HISTORY_STACK - 1) + C.FEATURE_NAMES.index(name)]
    return None


def scripted_selector_chain_actor(mask, features, kind, now):
    """Frozen scripted comparator for the selector-enabled environment.

    Declared mechanics chain, in NAMED phase/action space (never raw action
    indices — the index meaning is phase-dependent under wall_ppo/2):

    - inactive:      ACTIVATE the currently offered feasible proposal,
                     otherwise WAIT.
    - hold_interior: MAINTAIN (never an early release).
    - hold_deadline: EXTEND once at the base-HOLD deadline when the mask
                     admits it (H20 -> H30, admitted through the existing
                     remaining_plan check at execution); RELEASE at the
                     extended-HOLD deadline. If an extension is declined by
                     admission the commitment is unchanged and the existing
                     scheduled release executes (protective behavior).
    - at the maximum admitted HOLD there is no deadline decision: the
      scheduled release executes autonomously (wall_ppo/2 behavior).

    This comparator is a SCRIPTED baseline, not a learned actor. The
    selector-enabled environment accepts an external actor through the same
    (mask, features, kind, now) boundary; a supplied actor is never ignored.
    """
    if mask is None:
        return None
    names = C.ACTION_MEANINGS[kind]   # unknown kind -> KeyError: never guess
    if kind == "inactive":
        activate, wait = names.index("ACTIVATE"), names.index("WAIT")
        return activate if mask[activate] else wait
    if kind == "hold_interior":
        return names.index("MAINTAIN")
    if kind == "hold_deadline":
        admitted = _current_tick_feature(features, "admitted_total_hold_s")
        extend = names.index("EXTEND")
        declared_extend_from = C.BASE_HOLD_S
        if (admitted is not None
                and abs(admitted - declared_extend_from) <= 1e-9
                and mask[extend]):
            return extend
        return names.index("RELEASE")
    raise ValueError(f"unknown decision kind: {kind!r}")


class SelectorWallPolicy(WallPpoPolicy):
    """wall_ppo/3: causal candidate search + wall_ppo/2 hold semantics."""

    def __init__(self, job, registration, activation_window=None):
        # member ids are not fixed at construction: the selector binds the
        # cohort at activation time from current acquired reports
        spec = job.get("policy_spec", {})
        window = activation_window or spec.get("activation_window")
        required = tuple(spec.get("required_lanes", ()))
        if not required:
            from .cohort_study import LANES
            required = LANES
        super().__init__(job, registration, tuple(), required,
                         activation_window=window)
        self.max_report_age_s = float(self.registration.get(
            "communication", {}).get("max_report_age_s", 1.0))
        self.offered = None            # bound at the current decision tick
        self.approved_offer = None     # snapshot taken at set_decision(ACTIVATE)
        self.committed_offer = None    # the offer committed at activation
        self.last_bound_offer = None   # last bound offer (for decision receipts)
        self.last_execution_refusal = None
        # search-computation timing (instrumentation only; never a behavior
        # input). Fixture-scale numbers do not establish corridor-scale cost.
        self.search_stats = dict(calls=0, total_s=0.0, max_s=0.0)
        self.last_search_time_s = None
        self.committed_proposal_sha256 = None

    # ---- observation binding --------------------------------------------

    def active_member_ids(self):
        """Cohort described by the observation features: the committed
        proposal after activation, else the approved/bound offer while
        inactive — never zeros while a valid candidate is offered."""
        if self.proposal is not None:
            return tuple(self.proposal.member_ids)
        if self.approved_offer is not None:
            return tuple(self.approved_offer.member_ids)
        if self.offered is not None:
            return tuple(self.offered.member_ids)
        return tuple()

    def offer_metadata(self):
        """Offer identity for rollout/observation metadata (consistency
        checks between features, decisions and the committed admission).
        A WAIT clears the consumable offer but the receipt of what the
        decision saw survives."""
        if self.committed_offer is not None:
            return self.committed_offer.identity_dict(committed=True)
        if self.offered is not None:
            return self.offered.identity_dict()
        if self.last_bound_offer is not None:
            return self.last_bound_offer.identity_dict()
        return None

    # ---- candidate search and offer binding ------------------------------

    def selected_pair(self, context):
        """Minimum-maneuver feasible pair from current acquired reports, or
        None. Shared admission path under THIS policy's profile and lanes —
        the identical objects the execution-time gate uses."""
        best_ids, _reason, _n = self._search(context)
        return best_ids

    def _search(self, context):
        return search_current_proposals(
            tuple(context.reports), context.now_s,
            profile=self.profile, required_lanes=self.required_lanes,
            max_report_age_s=self.max_report_age_s)

    def _bind_offer(self, context):
        """Bind this decision tick's OfferedProposal from the shared-path
        search. A stale bound offer is replaced; nothing feasible leaves
        the policy with no offer (the actor sees only WAIT). The last
        bound offer is kept for the decision receipt (the base __call__
        re-evaluates decision_context before logging, so the receipt must
        survive one rebind)."""
        if self.offered is not None:
            self.last_bound_offer = self.offered
        self.offered = None
        started = time.perf_counter()
        best_ids, _reason, assessed = self._search(context)
        self.last_search_time_s = time.perf_counter() - started
        stats = self.search_stats
        stats["calls"] += 1
        stats["total_s"] += self.last_search_time_s
        stats["max_s"] = max(stats["max_s"], self.last_search_time_s)
        if best_ids is None:
            return None
        reports = tuple(context.reports)
        try:
            proposal = make_proposal(reports, now_s=context.now_s,
                required_lanes=self.required_lanes, member_ids=best_ids,
                profile=self.profile)
            assess = assess_formation(proposal, reports, now_s=context.now_s)
        except ValueError:
            return None
        if not assess.approved:
            return None
        plan = remaining_plan(proposal, reports, now_s=context.now_s,
                              previous_commands={}, readiness_count=0)
        self.offered = OfferedProposal(
            member_ids=tuple(best_ids), required_lanes=self.required_lanes,
            profile=self.profile, target_speed_mps=proposal.target_speed_mps,
            fingerprint=proposal.fingerprint,
            evidence_fingerprint=assess.evidence_fingerprint,
            decision_time_s=context.now_s,
            remaining_total_s=(plan or {}).get("remaining_total_s"),
            assessed_pairs=assessed)
        return self.offered

    # ---- wall_ppo/2 hook overrides (inactive phase only) ------------------

    def activation_feasible(self, context):
        """Bind the offer from current reports, then apply the unchanged
        wall_ppo/2 eligibility gates (window, one activation). The mask bit
        therefore means: a feasible proposal is bound AND activation is
        currently permitted."""
        self._bind_offer(context)
        if self.offered is None:
            return False
        earliest, latest = self.activation_window
        if (self.activation_used >= C.MAX_ACTIVATIONS
                or not (earliest <= context.now_s <= latest)):
            return False
        return True

    def set_decision(self, action):
        self.last_bound_offer = self.offered   # receipt of what this decision saw
        super().set_decision(action)
        action = int(action)
        if action == C.A_ACTIVATE:
            if self.offered is None:
                # no bound offer: execution will refuse atomically
                self.approved_offer = None
            else:
                self.approved_offer = self.offered
        else:
            # WAIT (or any non-activation choice) clears the offer; the
            # next eligible decision searches from then-current reports
            self.offered = None
            self.approved_offer = None

    def _activate(self, context):
        """Execution-time admission of the APPROVED offer (atomic; never a
        re-ranked substitution)."""
        offer = self.approved_offer
        self.last_execution_refusal = None
        if offer is None:
            self.last_execution_refusal = dict(reason="no_bound_offer",
                                               offer=None)
            return False
        reports = tuple(context.reports)
        try:
            proposal = make_proposal(reports, now_s=context.now_s,
                required_lanes=offer.required_lanes,
                member_ids=offer.member_ids, profile=offer.profile)
        except ValueError as exc:
            self.last_execution_refusal = dict(
                reason="proposal_unavailable:" + str(exc),
                offer=offer.identity_dict())
            return False
        if abs(proposal.target_speed_mps - offer.target_speed_mps) > 1e-9:
            # permitted update: refresh the proposal to execution-time
            # evidence while preserving the offered target exactly (this
            # never changes the action the actor selected)
            proposal = replace(proposal, target_speed_mps=offer.target_speed_mps)
        plan = remaining_plan(proposal, reports, now_s=context.now_s,
                              previous_commands={}, readiness_count=0)
        if not plan["feasible"]:
            self.last_execution_refusal = dict(
                reason="admission:" + str(plan.get("reason")),
                offer=offer.identity_dict())
            return False
        self.proposal = proposal
        self.activation_used += 1
        self.committed_offer = offer
        self.committed_proposal_sha256 = proposal.fingerprint
        self.offered = None
        self.approved_offer = None
        self.last_bound_offer = None   # the committed offer supersedes it
        return True

    # ---- decision receipts ------------------------------------------------

    def _log_decision(self, now, mask, requested, executed, reason=None,
                      kind=None, admission=None):
        super()._log_decision(now, mask, requested, executed, reason=reason,
                              kind=kind, admission=admission)
        entry = self.decision_log[-1]
        if executed == "activate":
            offer, committed = self.committed_offer, True
        elif requested == C.A_ACTIVATE:
            offer, committed = self.approved_offer, False
        else:
            offer, committed = self.last_bound_offer, False
        if offer is not None:
            entry["offer"] = offer.identity_dict(committed=committed)
        if executed == "activate":
            # the fingerprint of the proposal ACTUALLY committed at
            # execution — equal to the approved offer's fingerprint when
            # the binding held (same-tick evidence, target preserved)
            entry["committed_proposal_sha256"] = getattr(
                self, "committed_proposal_sha256", None)
        if self.last_search_time_s is not None:
            entry["search_time_s"] = self.last_search_time_s
        if (self.last_execution_refusal is not None
                and requested == C.A_ACTIVATE and executed is None):
            entry["offer_refusal"] = self.last_execution_refusal
