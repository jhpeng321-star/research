"""MAIN nullfix: observable actionable gate for saved LS shortlists, before SUMO."""
import time


def shortlist(record, candidates, bounds):
    from .sumo_verify import actionable_gate
    residuals=record.get('policy_residuals', {})
    sample=dict(context=dict(members=list(residuals)),
                own_action=[[a] for a in residuals.values()])
    gate = actionable_gate(candidates, record['rows'], record['commands'],
        sample, dict(accel_bounds=bounds))
    return gate['actionable'], dict(original_k=len(candidates), available_k=len(gate['actionable']),
        inactionable=gate['inactionable'], members_t0=gate.get('members_t0'),
        gate_source='registered observable actuator feasibility, no SUMO')


def skipped(record, envelope, gate):
    from .exact_grade import _seal
    now = time.time()
    return _seal(dict(schema='coopmwm.observable_feasibility_skip.v1',
        verification_id='inactionable_'+record['state_id'], status='inactionable',
        t0=envelope['time_s'], k=0, available_k=0, candidates_before_gate=gate['original_k'],
        chosen=None, chosen_plan=None, grade_veh_s=None, best_margin_veh_s=None,
        J_policy_veh_s=None, J_candidates_veh_s=None, branches=[],
        actual_sim_s=0, charged_sim_s=0, original_reservation_retained=True,
        reason='no actionable candidate; no snapshot materialization or SUMO branch',
        feasibility=gate, timing=dict(started_unix=now, finished_unix=now, wall_s=0)),
        'receipt_sha256')
