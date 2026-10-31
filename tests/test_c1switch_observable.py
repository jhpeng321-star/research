"""The inference-side association is anonymous and native history is never padded."""
from copy import deepcopy

import pytest

pytest.importorskip('torch', reason='real assimilation CPU gates run in existing cpuQ environment')

from tools.c1switch.core import Belief
from tools.c1switch.observable import tracker_from_payload


def payload():
    belief = Belief()
    for tick in range(121):
        t = 240+.5*tick
        common = dict(timestamp_s=t, lane=1, v_mps=10., a_mps2=0.)
        belief.observe(t, self_reports=[dict(common,vehicle_id='self',s_m=-500+5*tick)],
                       detections=[dict(common,vehicle_id='hidden_truth_id',s_m=-470+5*tick)],
                       coverage=[])
    return belief.payload()


def test_anonymous_association_does_not_depend_on_truth_identity_or_hidden_attributes():
    import torch
    a = payload()
    b = deepcopy(a)
    for tick,row in enumerate(b['history']):
        row['detections'][0].update(vehicle_id=f'unrelated_{tick}',hidden_type='secret',length_m=99.)
    ta, tb = tracker_from_payload(a), tracker_from_payload(b)
    for key,value in ta.state().items():
        assert torch.equal(value,tb.state()[key])
    for key in ('kin','mask','target_mask'):
        assert (ta.history(history_s=60)[key] == tb.history(history_s=60)[key]).all()
    assert ta.state()['present'].sum() == 2
    assert ta.state()['is_cav'].sum() == 1


def test_missing_half_second_history_cannot_be_called_full_history():
    p = payload()
    p['history'] = p['history'][::2]
    with pytest.raises(ValueError, match='half-second'):
        tracker_from_payload(p)


def test_main_capture_projects_both_ticks_and_excludes_unobserved_rows(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from tools.c1switch.main_backend import MainBackend
    from tools.c1switch.sumo import Backend
    # Isolate the sensor override from exact evaluation tape accounting; SUMO is
    # never imported/started. Both .5s snapshots contain a far, unobserved lane0 car.
    def evaluation_only(self,env):
        self.belief.observe(env.now,self_reports=[],detections=[],coverage=[])
    monkeypatch.setattr(Backend,'observe',evaluation_only)
    kin = np.zeros((2,3,7),np.float32)
    kin[:,:,0] = [-2050,-2025,-2200]
    kin[:,:2,2] = 1
    kin[:,2,1] = 1
    kin[:,:,4] = 10
    raw = dict(kin=kin,mask=np.ones((2,3),bool),vid_index=np.tile(np.arange(3),(2,1)))
    env = SimpleNamespace(now=300.,history=SimpleNamespace(raw=lambda n:raw),
                          builder=SimpleNamespace(vehicle_ids=['self','neighbor','secret'],cav_ids={'self'}))
    def capture():
        backend = MainBackend.__new__(MainBackend)
        backend.np=np;backend.belief=Belief()
        backend.observe(env)
        return backend.belief.payload()
    a = capture()
    kin[:,2,0] = -2400
    kin[:,2,4] = 31
    assert capture() == a
    assert [r['now_s'] for r in a['history']] == [299.5,300.]
    assert all([r['vehicle_id'] for r in h['self_reports']] == ['self'] for h in a['history'])
    assert all([r['vehicle_id'] for r in h['detections']] == ['neighbor'] for h in a['history'])
