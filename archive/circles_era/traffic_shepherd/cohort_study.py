"""Part C acquired selector and exact assignment/reuse binding primitives.

The separate cohort_campaign entry point enforces physical prerequisites and
budget registration. This module never launches a simulator.
"""
from copy import deepcopy
from dataclasses import replace
from itertools import product
from math import isfinite

from traffic_shepherd.escalation_observations import report_from_payload
from traffic_shepherd.formation import make_proposal
from traffic_shepherd.lifecycle_wall import LifecycleProfile, LifecycleWallPolicy, remaining_plan, VERSION
from traffic_shepherd.paired_experiments import PolicyInput, PolicyOutput, digest

SCHEMA = 'report_cohort_selection_study/1'
SELECTOR = 'finite_feasible_minimum_assembly_then_lane_ordered_ids/1'
LANES = ('1191554636_0','1191554636_1')
MINIMUM_RECRUITMENT_DISTANCE_M = 200.
CURRENT_BAND_MAXIMUM_M = 1900.
PREFIX_KEYS = ('decision_time_s','predecision_state_sha256','predecision_report_evidence_sha256',
               'predecision_report_prefix_sha256','predecision_command_prefix_sha256')


def acquired_evidence(context):
    return dict(now_s=context.now_s,reports=[r.payload() for r in context.reports],
        history=[dict(received_at_s=t,**r.payload()) for t,r in context.history])


def select_cohort(reports, *, now_s, required_lanes=LANES):
    """Complete acquired pair universe; no seed, state census or outcome input."""
    if VERSION != 'finite_lifecycle_wall/2' or tuple(required_lanes) != LANES:
        raise ValueError('Only unchanged V2 and the registered two approach lanes')
    profile = LifecycleProfile()
    reports = tuple(sorted(reports,key=lambda r:r.frame.vehicle_id))
    if (not isfinite(now_s) or len({r.frame.vehicle_id for r in reports}) != len(reports)
            or any(r.sent_at_s > now_s or r.frame.timestamp_s > now_s for r in reports)):
        raise ValueError('Unique already acquired reports required')
    candidates = [[r.frame for r in reports if r.frame.lane_id == lane
        and r.frame.distance_to_merge_m is not None
        and r.frame.distance_to_merge_m >= MINIMUM_RECRUITMENT_DISTANCE_M
        and 0 <= now_s-r.frame.timestamp_s <= profile.max_report_age_s] for lane in required_lanes]
    inventory = []
    for group in product(*candidates):
        proposal = make_proposal(reports,now_s=now_s,required_lanes=tuple(required_lanes),
            member_ids=tuple(f.vehicle_id for f in group),profile=profile)
        span = max(f.distance_to_merge_m for f in group)-min(f.distance_to_merge_m for f in group)
        spread = max(f.speed_mps for f in group)-min(f.speed_mps for f in group)
        settled = (span <= profile.formed_span_m and spread <= profile.formed_speed_spread_mps
            and all(abs(f.speed_mps-proposal.target_speed_mps) <= profile.target_speed_tolerance_mps
                    for f in group))
        plan = remaining_plan(proposal,reports,now_s=now_s,previous_commands={},
                              readiness_count=int(settled))
        inventory.append(dict(member_ids=list(proposal.member_ids),proposal_sha256=proposal.fingerprint,
            feasible=plan['feasible'],reason=plan['reason'],
            predicted_assembly_s=plan.get('remaining_assembly_s'),
            inside_current_band=all(f.distance_to_merge_m <= CURRENT_BAND_MAXIMUM_M for f in group),
            plan=plan))
    feasible = [r for r in inventory if r['feasible']]
    selected = min(feasible,key=lambda r:(r['predicted_assembly_s'],tuple(r['member_ids']))) if feasible else None
    result = dict(selector=SELECTOR,wall_version=VERSION,decision_time_s=now_s,
        complete_acquired_enumeration=True,candidate_count=len(inventory),
        candidates_by_lane={lane:len(group) for lane,group in zip(required_lanes,candidates)},
        selected=selected,inventory=inventory)
    result['selection_sha256'] = digest(result)
    return result


def snapshot_selection(snapshot):
    """Full-state/prefix hashes are binding metadata, never selector features."""
    evidence = snapshot['evidence']
    if digest(evidence) != snapshot['predecision_report_evidence_sha256']:
        raise ValueError('Received evidence hash mismatch')
    if snapshot['decision_time_s'] != 600. or evidence['now_s'] != 600.:
        raise ValueError('One registered 600 s opportunity only')
    reports = tuple(report_from_payload(r) for r in evidence['reports'])
    return select_cohort(reports,now_s=600.)


def bind_selected_job(base_job, snapshot):
    """Derive the concrete registered assignment; do not hard-code seed choices."""
    selection = snapshot_selection(snapshot)
    chosen = selection['selected']
    if chosen is None:
        raise ValueError('No qualifying formation candidate; retain damping')
    job = deepcopy(base_job)
    if (job['seed'] not in (4107,4108) or job['loading'] != 2700 or job['penetration'] != .2
            or job['duration_s'] != 2400 or job['decision_time_s'] != 600 or job['arm'] != 'wall'):
        raise ValueError('Only exact bounded Part B scenarios can supply Part C jobs')
    LifecycleProfile(**job['policy_spec']['profile'])
    job['id'] = f"s{job['seed']}_m2700_p20_t600_cohort_v1_wall"
    job['policy_spec']['member_ids'] = chosen['member_ids']
    job['policy_spec']['cohort_selection'] = dict(selector=SELECTOR,
        evidence_sha256=snapshot['predecision_report_evidence_sha256'],
        selection_sha256=selection['selection_sha256'],
        proposal_sha256=chosen['proposal_sha256'])
    return job,selection


class ReportSelectedWall:
    """At runtime derive the choice again, then use the identical V2 instance.

    The expected assignment/evidence hashes validate replay integrity; they do
    not select a candidate. A mismatch atomically refuses and cannot re-engage.
    """
    def __init__(self,job,registration):
        self.job,self.registration = deepcopy(job),registration
        self.bound = self.job['policy_spec']['cohort_selection']
        if self.bound['selector'] != SELECTOR:
            raise ValueError('Unsupported selector')
        LifecycleProfile(**self.job['policy_spec']['profile'])
        self.delegate,self.selection,self.refusal = None,None,None

    def __call__(self,context):
        if not isinstance(context,PolicyInput):
            raise TypeError('Acquired PolicyInput only')
        if context.now_s < self.job['decision_time_s']:
            return PolicyOutput()
        if self.delegate is None and self.refusal is None:
            if context.now_s != self.job['decision_time_s']:
                self.refusal = 'cohort_no_late_or_replacement_opportunity'
            else:
                self.selection = select_cohort(context.reports,now_s=context.now_s,
                    required_lanes=tuple(self.job['policy_spec']['required_lanes']))
                chosen = self.selection['selected']
                if chosen is None:
                    self.refusal = 'cohort_no_finite_feasible_pair'
                elif (digest(acquired_evidence(context)) != self.bound['evidence_sha256']
                        or self.selection['selection_sha256'] != self.bound['selection_sha256']
                        or chosen['proposal_sha256'] != self.bound['proposal_sha256']
                        or chosen['member_ids'] != self.job['policy_spec']['member_ids']):
                    self.refusal = 'cohort_registered_acquired_selection_mismatch'
                else:
                    self.delegate = LifecycleWallPolicy(self.job,self.registration)
        if self.refusal is not None:
            return PolicyOutput({},'refused',self.refusal,
                tuple(self.job['policy_spec']['member_ids']),None,
                digest(acquired_evidence(context)),dict(cohort_selector=SELECTOR,
                    cohort_selection=self.selection,cohort_selection_verified=False))
        output = self.delegate(context)
        proof = dict(cohort_selector=SELECTOR,cohort_selection_verified=True,
            cohort_selection_sha256=self.selection['selection_sha256'])
        if context.now_s == self.job['decision_time_s']:
            proof['cohort_selection'] = self.selection
        return replace(output,details=dict(output.details,**proof))


def policy_factory(job,registration):
    return ReportSelectedWall(job,registration)


def matched_prefix(left,right):
    if any(left[key] != right[key] for key in PREFIX_KEYS):
        raise ValueError('Exact predecision physical/report/command prefix mismatch')
    return {key:left[key] for key in PREFIX_KEYS}


def validate_reuse_proof(proof, *, selection, source_manifest_sha256, source_job):
    """A snapshot/cohort match cannot stand in for complete same-action replay."""
    expected = dict(source_manifest_sha256=source_manifest_sha256,
        selection_sha256=selection['selection_sha256'],
        source_job_id=source_job['id'],decision_count=2400,
        first_decision_s=0.,last_decision_s=2399.,
        all_phases_and_members_equal=True,all_wall_commands_equal=True,
        all_actual_actuator_commands_and_native_release_flags_equal=True,
        no_simulator_rollout=True,complete=True)
    if any(proof.get(key) != value for key,value in expected.items()):
        raise ValueError('Complete source-bound 2400-decision reuse proof required')
    if selection['selected']['member_ids'] != source_job['policy_spec']['member_ids']:
        raise ValueError('Changed cohort cannot reuse previous physical arm')
    return True


def allocation_contract(jobs,bindings):
    """Exact budget shape; realization requires hash/source checks at activation."""
    if len(jobs) != 2 or {j['seed'] for j in jobs} != {4107,4108}:
        raise ValueError('Two development states, not an expanded panel')
    by_seed = {j['seed']:j for j in jobs}
    if (set(bindings) != {j['id'] for j in jobs}
            or bindings[by_seed[4107]['id']]['mode'] != 'verified_reuse'
            or bindings[by_seed[4108]['id']]['mode'] != 'simulate'
            or any(j['duration_s'] != 2400 for j in jobs)):
        raise ValueError('Exactly one verified reuse and one new full wall episode')
    return dict(maximum_attempts=1,maximum_simulated_seconds=2400,
        new_job_ids=[by_seed[4108]['id']],reused_job_ids=[by_seed[4107]['id']],
        automatic_retries=0,automatic_successors=False,new_mechanics_seconds=0,
        original_manifests_rewritten=False)
