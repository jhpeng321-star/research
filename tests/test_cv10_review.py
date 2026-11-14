"""Complete synthetic CV10 evidence and independent integrity mutations."""
import copy

import pytest

from output.controller_verification_v10.independent_review import verify


def fixture(case='PI-UNSAT-MIXED'):
    gap, raw = (4.06,1.4173445) if case=='PI-UNSAT-MIXED' else (3.8,.5)
    r=dict(case_id=case,status='complete',failure=None,cleanup_errors=[],actual_progress_s=2.,
           adapter_instance_id=11,controller_instance_id=11,owner_instance_id=22,physical=[])
    for t in (.5,1.,1.5,2.):
        states={}
        if t>=1:
            for vehicle,kind,x,speed,cap in [('leader_hdv','leader_type',105.+gap,.5,.5),
                                            ('cav','idm',100.,1.,30.),('rear_hdv','rear_type',50.,1.,1.)]:
                states[vehicle]=dict(type_id=kind,speed_mode=31,position_m=x+speed*(t-1),
                                     speed_mps=speed,acceleration_mps2=0.,length_m=5.,max_speed_mps=cap)
        r['physical'].append(dict(time_s=t,vehicle_ids=list(states),vehicles=states,collisions=0,teleports=0))
    frame=dict(vehicle_id='cav',timestamp_s=1.,lane_index=0,allowed_speed_mps=30.,speed_mps=1.,
               acceleration_mps2=0.,position_m=100.,front_coverage_m=100.,rear_coverage_m=95.,
               front=dict(gap_m=gap,relative_speed_mps=-.5),rear=dict(gap_m=45.,relative_speed_mps=0.))
    row=dict(vehicle_id='cav',timestamp_s=1.,owner='damping',fallback_reason=None,released_to_native=False,
             network_output_mps2=None,wall_request_mps2=None,actor_input_si=None,
             issued_acceleration_mps2=raw-1.,issued_target_speed_mps=raw)
    r['observer_frames']=[dict(time_s=1.,frame=frame)]
    r['frames']=[dict(time_s=1.,provided_to_controller=True,frame=frame)]
    r['decisions']=[row]
    r['owner_receipts']=[row]
    r['events']=[dict(event='command_tick',time_s=1.,adapter_instance_id=11,controller_instance_id=11,
                      owner_instance_id=22,native=False,records=1)]
    r['adapter_evidence']=[dict(vehicle_id='cav',timestamp_s=1.,frame=frame,generation=1,
        previous_raw_command_mps=1.,raw_proposal_mps=raw,requested_acceleration_mps2=raw-1.,
        issued_acceleration_mps2=raw-1.,issued_target_speed_mps=raw,owner_record=row,
        u_report=dict(value_mps=31.29,provenance='fixed_registered_setpoint',evidence_id='fixture-U',
                      report_timestamp_s=None,receipt_timestamp_s=None,max_age_s=None))]
    r['adapter_receipts']=[dict(vehicle_id='cav',timestamp_s=1.,status=status,reason=None,
                                raw_command_mps=raw,issued_target_speed_mps=raw)
                           for status in ('written_owner_receipt','committed_raw_proposal')]
    r['timed_writes']=[dict(time_s=1.,event='control',command=['slowDown','cav',raw,1.],succeeded=True),
                       dict(time_s=2.,event='cleanup',command=['setSpeed','cav',-1],succeeded=True)]
    return r


@pytest.mark.parametrize('case',['PI-UNSAT-MIXED','PI-BRAKE-ZERO'])
def test_complete_case_accepted_without_perfect_delivery(case):
    verdict=verify(fixture(case),case)
    assert verdict['passed'],verdict
    assert verdict['tracking'][0]['classification'] != 'delivered'
    assert verdict['pi_interval_count']==1


@pytest.mark.parametrize('fault',['clock','insertion','missing_car','raw','memory','generation','U',
                                   'frame','receipt','write','hdv','cleanup','collision','case'])
def test_corrupted_case_rejected(fault):
    r=copy.deepcopy(fixture())
    if fault=='clock':
        r['physical'].pop(2)
    elif fault=='insertion':
        r['physical'][1]['vehicles']['leader_hdv']['position_m']+=.1
    elif fault=='missing_car':
        del r['physical'][-1]['vehicles']['cav']
    elif fault=='raw':
        r['adapter_evidence'][0]['raw_proposal_mps']+=.1
    elif fault=='memory':
        r['adapter_evidence'][0]['previous_raw_command_mps']=2.
    elif fault=='generation':
        r['adapter_evidence'][0]['generation']=3
    elif fault=='U':
        r['adapter_evidence'][0]['u_report']['value_mps']=30.
    elif fault=='frame':
        r['frames'][0]['provided_to_controller']=False
    elif fault=='receipt':
        r['adapter_receipts'].pop()
    elif fault=='write':
        r['timed_writes'][0]['succeeded']=False
    elif fault=='hdv':
        r['timed_writes'][0]['command'][1]='leader_hdv'
    elif fault=='cleanup':
        r['timed_writes'].pop()
    elif fault=='collision':
        r['physical'][-1]['collisions']=1
    elif fault=='case':
        r['case_id']='unknown'
    assert not verify(r)['passed']
