from dataclasses import asdict
import pytest
from traffic_shepherd.escalation_dataset import reconstruct_features
from traffic_shepherd.formation import FormationProfile, make_proposal
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import SensorFrame


def fixture():
    reports=[CAVReport(SensorFrame(vehicle_id=str(i),timestamp_s=10.,speed_mps=20.,
        acceleration_mps2=0.,allowed_speed_mps=30.,lane_index=i,lane_id=f'L{i}',
        distance_to_merge_m=1800.-100*i,front_coverage_m=100.,rear_coverage_m=100.),10.,'damping') for i in range(2)]
    profile=FormationProfile(hold_s=20.)
    p=make_proposal(reports,now_s=10.,required_lanes=('L0','L1'),member_ids=('0','1'),profile=profile)
    job=dict(decision_time_s=10,policy_spec=dict(required_lanes=['L0','L1'],member_ids=['0','1'],profile=asdict(profile)))
    snapshot=dict(decision_time_s=10.,proposal_sha256=p.fingerprint,evidence=dict(
        reports=[r.payload() for r in reports],history=[dict(received_at_s=10.,**r.payload()) for r in reports]))
    return job,snapshot


def test_exact_acquired_packet_reconstructs_bound_proposal_without_outcome_inputs():
    job,snapshot=fixture()
    p,feature,counts=reconstruct_features(job,snapshot)
    assert p.fingerprint==snapshot['proposal_sha256']
    assert feature['proposal_sha256']==p.fingerprint and counts['registered_selection_reproduced']
    assert 'benefit' not in feature['values'] and 'penetration' not in feature['values']


def test_tampered_proposal_and_future_receipt_are_rejected():
    job,snapshot=fixture()
    snapshot['proposal_sha256']='tampered'
    with pytest.raises(ValueError,match='reconstructed exactly'):
        reconstruct_features(job,snapshot)
    job,snapshot=fixture()
    snapshot['evidence']['history'][0]['received_at_s']=11.
    with pytest.raises(ValueError,match='future'):
        reconstruct_features(job,snapshot)


def test_unavailable_registered_member_is_retained_without_fabricated_features():
    job,snapshot=fixture()
    snapshot['evidence']['reports'].pop()
    p,f,counts=reconstruct_features(job,snapshot)
    assert p is f is None and not counts['feature_eligible']
