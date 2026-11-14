"""Independent one-switch contracts with real controllers and sole command owner."""
from types import SimpleNamespace

import pytest

from traffic_shepherd.config import load_config
from traffic_shepherd.damping import CirclesController
from traffic_shepherd.fleet_switch import FleetSwitchSession, SingleSwitchSpec
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.pi_session import PISession
from tests.test_pi_owner_review import frame


def setup(prefix, suffix, boundary=2.):
    config=load_config()
    circles=CirclesController(config)
    pi=PISession(config=config,cav_ids={'a','b'},desired_speed_mps=31.29)
    writes=[]
    fail={'vehicle':None}
    def slow(v,target,duration):
        if v==fail['vehicle']:
            raise RuntimeError('writer failed')
        writes.append(('slowDown',v,target,duration))
    api=SimpleNamespace(vehicle=SimpleNamespace(getSpeedMode=lambda _:31,slowDown=slow,
        setSpeed=lambda v,value:writes.append(('setSpeed',v,value))))
    owner=CommandOwnership(api,{'a','b'},circles)
    router=FleetSwitchSession(SingleSwitchSpec(prefix,suffix,boundary),
        circles_controller=circles,pi_session=pi,cav_ids={'a','b'})
    return router,owner,pi,circles,writes,fail


def step(router,owner,t,ids=('a','b')):
    return router.step(owner,{v:frame(v,t) for v in ids},now_s=t,
                       live_ids=set(ids)|{'hdv'})


def raw(prior):
    return .5*(31.29+13/23)+.5*prior


def test_retained_pi_preserves_raw_recurrence_and_single_owner_call():
    router,owner,pi,_,writes,_=setup('pi-external-u','pi-external-u')
    calls=[]
    original=owner.step
    def counted(*args,**kwargs):
        calls.append(kwargs['now_s'])
        return original(*args,**kwargs)
    owner.step=counted
    previous=10.
    for t in (1.,2.,3.):
        rows=step(router,owner,t)
        previous=raw(previous)
        assert owner.controller is pi.adapter
        assert pi.adapter.pi._state['a'][0]==pytest.approx(previous)
        assert rows[0].issued_target_speed_mps==11.5
    assert calls==[1.,2.,3.]
    assert len(writes)==6 and all(x[1]!='hdv' for x in writes)


def test_circles_to_pi_first_entry_measured_then_raw():
    router,owner,pi,circles,_,_=setup('damping','pi-external-u')
    expected=circles.decide(frame('a',1.),now_s=1.)
    first=step(router,owner,1.)
    assert first[0].network_output_mps2==expected.network_output_mps2
    assert owner.controller is circles and not pi.adapter.pi._state
    step(router,owner,2.)
    assert owner.controller is pi.adapter
    assert pi.adapter.pi._state['a'][0]==pytest.approx(raw(10.))
    step(router,owner,3.)
    assert pi.adapter.pi._state['a'][0]==pytest.approx(raw(raw(10.)))


def test_pi_to_native_releases_once_and_purges_memory():
    router,owner,pi,circles,writes,_=setup('pi-external-u','native')
    step(router,owner,1.)
    rows=step(router,owner,2.)
    assert owner.controller is circles and not pi.adapter.pi._state
    assert all(r.released_to_native for r in rows)
    assert [x for x in writes if x[0]=='setSpeed']==[('setSpeed','a',-1.),('setSpeed','b',-1.)]
    step(router,owner,3.)
    assert len([x for x in writes if x[0]=='setSpeed'])==2


@pytest.mark.parametrize('clock',[float('nan'),float('inf'),1.5,0.,3.])
def test_invalid_clock_before_transition_does_not_purge_pi(clock):
    router,owner,pi,_,writes,_=setup('pi-external-u','native')
    step(router,owner,1.)
    state=dict(pi.adapter.pi._state)
    with pytest.raises((ValueError,RuntimeError)):
        step(router,owner,clock)
    assert pi.adapter.pi._state==state
    assert owner.controller is pi.adapter and len(writes)==2


@pytest.mark.parametrize('prefix,suffix',[('native','damping'),('native','pi-external-u')])
def test_partial_boundary_write_terminal_and_preserves_ack(prefix,suffix):
    router,owner,pi,_,writes,fail=setup(prefix,suffix)
    step(router,owner,1.)
    fail['vehicle']='b'
    with pytest.raises(RuntimeError,match='writer failed'):
        step(router,owner,2.)
    assert [x[1] for x in writes]==['a']
    if suffix=='pi-external-u':
        evidence=pi.pending_evidence()
        proposals=[x for x in evidence if x['kind']=='pi_source_evidence']
        assert len(proposals)==2
        assert proposals[0]['owner_record'] is not None
        assert proposals[1]['owner_record'] is None
    fail['vehicle']=None
    with pytest.raises((ValueError,RuntimeError)):
        step(router,owner,3.)
    assert len(writes)==1


def test_owner_swap_rejected_before_write():
    router,owner,_,circles,writes,_=setup('native','damping')
    step(router,owner,1.)
    foreign=CommandOwnership(owner.api,{'a','b'},circles)
    with pytest.raises((ValueError,RuntimeError)):
        step(router,foreign,2.)
    assert writes==[]


@pytest.mark.parametrize('value',[True,False,float('nan'),float('inf'),-1.,0.,1.5])
def test_invalid_switch_spec(value):
    with pytest.raises((ValueError,TypeError)):
        SingleSwitchSpec('native','damping',value)

