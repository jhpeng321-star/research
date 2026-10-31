"""CPU gates: restore comparison rejects meaningful errors; objective exit carry."""
import copy
from types import SimpleNamespace

from tools.ensprobe.worker import Measurement, compare


def sample():
    return dict(time=300.,vehicles={'a':dict(lane='road_0',route=['road','exit'],route_index=0,
                                           speed=10.,lane_pos=1.,xy=[1.,2.])},
                pending=['pending'],source_queue=1,lane_queues={'road_0':0})


def test_physical_tolerance_and_fail_closed():
    a=sample();b=copy.deepcopy(a)
    b['vehicles']['a']['speed']+=1e-8
    assert compare(a,b)['pass']
    for key,value in [('lane','road_1'),('route_index',1),('route',['wrong']),('lane_pos',1.1),('speed',10.1)]:
        b=copy.deepcopy(a);b['vehicles']['a'][key]=value
        assert not compare(a,b)['pass']
    for key,value in [('pending',[]),('source_queue',2),('lane_queues',{'road_0':1}),('vehicles',{})]:
        b=copy.deepcopy(a);b[key]=value
        assert not compare(a,b)['pass']


def test_union_objective_carries_exit_and_adds_newborn_and_source():
    m=Measurement()
    env=SimpleNamespace(_table={'new':dict(speed_mps=0.,lane_id='road_0',lane_position_m=1.)},
                        entry_queue_now=lambda:2)
    m.after(env,300.5,{'exit':31.29/2},{})
    assert m.road==.75  # exiting half-speed .25 + newborn stopped .5
    assert m.source==1.
    assert not m.clocks


def test_prefix_bins_copied_without_future_or_mutation():
    p=Measurement();p.sums['upstream'][0]=15.;p.counts['upstream'][0]=3
    p.sums['upstream'][6]=99.
    b=Measurement(p)
    assert b.result()['speed_bins']['upstream'][0]==5.
    assert b.result()['speed_bins']['upstream'][6] is None
    b.counts['upstream'][0]=1
    assert p.counts['upstream'][0]==3
