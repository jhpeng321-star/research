"""Independent complete CV08 evidence fixture and semantic corruption checks."""
import copy

import pytest

from output.controller_verification_v8.independent_review import verify


def valid_result():
    """Hand-computed source recurrence; deliberately non-exact plant tracking."""
    data = dict(status='complete', failure=None, actual_progress_s=8., cleanup_errors=[],
        controller_instance_id=11, adapter_instance_id=11, owner_instance_id=22,
        frames=[], observer_frames=[], physical=[], decisions=[], owner_receipts=[],
        adapter_evidence=[], adapter_receipts=[], events=[])
    writes = []
    data['commands'] = writes
    data['timed_writes'] = writes
    for index in range(1, 17):
        t = index * .5
        vehicles = {}
        if t >= 1:
            for v, position, kind, cap in [('leader_hdv', 150., 'leader_type', 15.),
                                          ('cav', 100., 'idm', 30.), ('rear_hdv', 50., 'rear_type', 15.)]:
                vehicles[v] = dict(type_id=kind, speed_mode=31, speed_mps=15., acceleration_mps2=0.,
                    max_speed_mps=cap, length_m=5., position_m=position+15*(t-1))
        data['physical'].append(dict(time_s=t, vehicle_ids=list(vehicles), vehicles=vehicles,
                                     collisions=0, teleports=0))
    previous = None
    for t in range(1, 8):
        observed = dict(vehicle_id='cav', timestamp_s=float(t), speed_mps=15., acceleration_mps2=0.,
            allowed_speed_mps=30., lane_index=0, distance_to_merge_m=None,
            front=dict(gap_m=45., relative_speed_mps=0.), rear=dict(gap_m=45., relative_speed_mps=0.),
            front_coverage_m=100., rear_coverage_m=min(100., 95.+15*(t-1)), previous_command_delta=0.,
            lane_id='road_0', position_m=100.+15*(t-1))
        data['observer_frames'].append(dict(time_s=float(t), frame=observed))
        data['frames'].append(dict(time_s=float(t), provided_to_controller=t != 5,
                                   frame=None if t == 5 else observed))
        active = t in (1, 2, 4, 6, 7)
        command = dict(vehicle_id='cav', timestamp_s=float(t), owner='damping' if active else 'native',
            network_output_mps2=None, wall_request_mps2=None, issued_acceleration_mps2=1.5 if active else None,
            issued_target_speed_mps=16.5 if active else None,
            fallback_reason=None if active else 'selected_native' if t == 3 else 'missing_observation',
            released_to_native=not active, actor_input_si=None)
        data['decisions'].append(command)
        data['owner_receipts'].append(command)
        if active:
            if t in (1, 4, 6):
                previous = 15.
            raw = .5 * 32.29 + .5 * previous
            generation = 1 if t <= 2 else 3 if t == 4 else 5
            data['adapter_evidence'].append(dict(vehicle_id='cav', timestamp_s=float(t), frame=observed,
                u_report=dict(value_mps=31.29, provenance='fixed_registered_setpoint',
                    evidence_id='cv08-road-U31.29', report_timestamp_s=None, receipt_timestamp_s=None, max_age_s=None),
                previous_raw_command_mps=previous, generation=generation, raw_proposal_mps=raw,
                requested_acceleration_mps2=raw-15., issued_acceleration_mps2=1.5,
                issued_target_speed_mps=16.5, owner_record=command))
            data['adapter_receipts'].extend([dict(vehicle_id='cav', timestamp_s=float(t), status=status,
                raw_command_mps=raw, issued_target_speed_mps=16.5, reason=None)
                for status in ('written_owner_receipt', 'committed_raw_proposal')])
            previous = raw
        else:
            data['adapter_receipts'].append(dict(vehicle_id='cav', timestamp_s=float(t),
                status='owner_row_without_pi', raw_command_mps=None, issued_target_speed_mps=None, reason=None))
        writes.append(dict(time_s=float(t), event='control',
            command=['slowDown', 'cav', 16.5, 1.] if active else ['setSpeed', 'cav', -1], succeeded=True))
        data['events'].append(dict(time_s=float(t), event='command_tick', native=t == 3,
            records=1, controller_instance_id=11, adapter_instance_id=11, owner_instance_id=22))
    writes.append(dict(time_s=8., event='cleanup', command=['setSpeed', 'cav', -1], succeeded=True))
    return data


def test_full_evidence_is_accepted_with_tracking_shortfalls():
    result = verify(valid_result())
    assert result['passed'], result


@pytest.mark.parametrize('mutation', ['missing_tick', 'duplicate_tick', 'wrong_reason', 'raw', 'memory',
                                    'generation', 'missing_endpoint', 'wrong_clock', 'hdv_write',
                                    'failed_write', 'release_missing', 'no_leader', 'collision', 'cleanup_error'])
def test_complete_evidence_corruptions_are_rejected(mutation):
    data = copy.deepcopy(valid_result())
    if mutation == 'missing_tick':
        data['decisions'].pop(0)
    elif mutation == 'duplicate_tick':
        data['decisions'].append(copy.deepcopy(data['decisions'][0]))
    elif mutation == 'wrong_reason':
        data['decisions'][4]['fallback_reason'] = 'selected_native'
    elif mutation == 'raw':
        data['adapter_evidence'][0]['raw_proposal_mps'] += 1
    elif mutation == 'memory':
        data['adapter_evidence'][2]['previous_raw_command_mps'] = 27.9675
    elif mutation == 'generation':
        data['adapter_evidence'][2]['generation'] = 1
    elif mutation == 'missing_endpoint':
        data['physical'][-1]['vehicles'].pop('cav')
    elif mutation == 'wrong_clock':
        data['physical'][3]['time_s'] = 1.75
    elif mutation == 'hdv_write':
        data['commands'][0]['command'][1] = 'leader_hdv'
    elif mutation == 'failed_write':
        data['commands'][0]['succeeded'] = False
    elif mutation == 'release_missing':
        data['commands'].pop(2)
    elif mutation == 'no_leader':
        data['observer_frames'][0]['frame']['front'] = None
    elif mutation == 'collision':
        data['physical'][3]['collisions'] = 1
    else:
        data['cleanup_errors'] = ['close failed']
    result = verify(data)
    assert not result['passed'], (mutation, result)
