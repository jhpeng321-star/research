"""Independent SUMO-source anchors; these values predate INT implementation."""
import pytest
import torch

from wms.physics.idm_sumo import SumoIDMParams, idm_iterations, stop_speed, sumo_idm_step
from wms.physics.config import PhysicsConfig
from wms.physics.neighbours import lane_neighbour
from wms.model.factory import build_model
from wms.model.spec import ModelSpec


@pytest.mark.parametrize('v,gap,vl,p,dt,expected', [
    (10,22.5,10,[50/3.6,1,2.6,4.5,2.5],.5,10.505369042766489),
    (0,32.5,50/3.6,[50/3.6,1,2.6,4.5,2.5],.5,1.292304647309232),
    (10,10.5,0,[50/3.6,1,2.6,4.5,2.5],.5,5.5),
    (10,22.5,10,[50/3.6*1.2,1,2.6,4.5,2.5],.5,10.684841865711107),
    (9,12,6,[40/3.6,1.5,1.5,2,2],.5,7.198219766344515),
    (10,17.5,8,[50/3.6,1,2.6,4.5,2.5],1.,9.76782360599685),
])
def test_independent_full_step_vectors(v, gap, vl, p, dt, expected):
    # rb-shared/idmspec/test_vectors.json: T14-17,T19-20; bumper gaps already converted.
    des,tau,a,b,s0 = p
    params = SumoIDMParams(a,b,tau,s0,des)
    tensor = lambda x: torch.tensor([x], dtype=torch.float64)
    actual, _ = sumo_idm_step(tensor(v),tensor(gap),tensor(vl),tensor(des),params,
                              dt_s=dt,iterations=idm_iterations(dt))
    assert actual.item() == pytest.approx(expected, abs=1e-10)


def test_independent_near_stop_reference():
    # T08 omits lane_limit: the reference uses passenger maxSpeed=200km/h.
    params = SumoIDMParams(2.6,4.5,1.,2.5,200/3.6)
    result = stop_speed(torch.tensor([.0005], dtype=torch.float64),
                        torch.tensor([1.], dtype=torch.float64), params,dt_s=.5,iterations=2)
    assert result.item() == pytest.approx(.8301017821871405, abs=1e-10)


def test_free_road_has_no_finite_distance_phantom_leader_and_euler_position():
    physics = PhysicsConfig(cf_model='sumo_idm',merge_model='sumo')
    model = build_model(ModelSpec(rollout_mode='driver_params', lane_model='lc2013'), physics)
    state = dict(s_m=torch.tensor([[100.]]),v_mps=torch.tensor([[20.]]),
        a_mps2=torch.zeros(1,1), lane=torch.zeros(1,1,dtype=torch.long),
        present=torch.ones(1,1,dtype=torch.bool),is_cav=torch.zeros(1,1,dtype=torch.bool),
        length_m=torch.full((1,1),5.))
    from wms.physics.state import state_kin
    gap, _ = lane_neighbour(state_kin(state),state['present'],state['lane'],physics,ahead=True)
    assert gap.item() == 1e6  # IDM.cpp freeSpeed no-leader argument, not 250m fake queue.
    with torch.no_grad():
        out = model.step(state)
    torch.testing.assert_close(out['next_s_m'],state['s_m'] + .5*out['next_v_mps'])
