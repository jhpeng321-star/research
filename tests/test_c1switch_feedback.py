import copy
from types import SimpleNamespace as NS

import pytest

from tools.c1switch.feedback import validate, seed_pi, seed_state


def payload(arm='native'):
    rows = []
    for j in range(121):
        now = 240+.5*j
        own = dict(vehicle_id='self',timestamp_s=now,s_m=-200+2*j,lane=2,
                   v_mps=4.,a_mps2=0.,age_s=0,source='cav_self')
        detections = [dict(own,vehicle_id='anonymous',s_m=own['s_m']+15,source='rsu')]
        rows.append(dict(now_s=now,self_reports=[own],detections=detections,
                         coverage=[],counters=None))
    return dict(schema='c1switch_observable/1',history=rows,
                controller_state=dict(incumbent=arm,now_s=300.,pi={},commands={}))


def test_memory_rejects_nonself_and_noncausal_values():
    p = payload('pi')
    p['controller_state']['pi']['hidden'] = [8,299]
    with pytest.raises(ValueError):validate(p)
    p['controller_state']['pi'] = dict(self=[8,298])
    with pytest.raises(ValueError):validate(p)
    p['controller_state']['pi'] = dict(self=[8,299])
    assert validate(p)['pi']['self'][0] == 8
    p['controller_state']['truth'] = 1
    with pytest.raises(ValueError):validate(p)


def test_pi_raw_state_continues_only_incumbent():
    p = payload('pi');p['controller_state']['pi'] = dict(self=[27.3,299])
    tracker = NS(tracks={7:NS(cav_id='self',slot=3,is_cav=True)})
    def timeline():return NS(pi=NS(_state={},_generation={}),pi_active=set())
    t = timeline();seed_pi(t,tracker,p,'pi')
    assert t.pi._state == {'0:3':(27.3,-1.)}
    assert t.pi_active == {'0:3'}
    p['controller_state'] = dict(incumbent='native',now_s=300.,pi={},commands={})
    t = timeline();seed_pi(t,tracker,p,'pi')
    assert not t.pi._state


def test_command_memory_remaps_self_identity_and_resets_on_switch():
    torch = pytest.importorskip('torch')
    p = payload('mappo');p['controller_state']['commands'] = dict(self=[-.7,0])
    tracker = NS(tracks={17:NS(cav_id='self',slot=2,is_cav=True)})
    state = dict(s_m=torch.zeros(1,4))
    seeded = seed_state(copy.deepcopy(state),tracker,p,'mappo')
    assert seeded['cmd_accel_mps2'].tolist() == [[0,0,pytest.approx(-.7),0]]
    assert not seed_state(copy.deepcopy(state),tracker,p,'pi')['cmd_accel_mps2'].any()


def test_real_frozen_models_cpu(tmp_path):
    """Opt-in host gate loads BOTH real checkpoints; SUMO remains intercepted."""
    import os
    if not os.environ.get('C1SWITCH_REAL_CPU'):
        pytest.skip('requires existing frozen host runtime')
    import torch
    from tools.c1switch.adapter import Adapter
    from tools.c1switch.core import medians
    from tools.c1switch.design import sha
    from pathlib import Path
    base = Path(os.environ['C1SWITCH_TEST_BUILD'])
    binding = dict(wm_checkpoint_sha256='b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436',
                   bb_checkpoint_sha256='afaaa32cb15079014d1a9e08f78b26e74fd341298d1aa0292a24009d9f783742',
                   checkpoint_path=str(base/'bb.pt'))
    assert sha(base/'bb.pt') == binding['bb_checkpoint_sha256']
    for name in ('wm','bb'):
        adapter = Adapter(dict(binding,model=name),'cpu')
        for arm in ('native','pi','mappo'):
            p = payload(arm)
            if arm == 'pi':p['controller_state']['pi'] = dict(self=[26.,299.])
            if arm == 'mappo':p['controller_state']['commands'] = dict(self=[-.4,0.])
            samples = adapter.predict(p,steps=4)
            assert len(medians(samples)) == 5
            assert next(adapter.model.parameters()).device == torch.device('cpu')
            adapter.optimized = False
            assert adapter.predict(p,steps=4) == samples
            adapter.optimized = True
