import json

import pytest
import torch

from tools.rblane.motion_points import MotionPoints
from wms.physics.config import PhysicsConfig


def payload():
    pop = dict(v0=30.,time_headway=1.,accel=1.2,decel=2.,min_gap=2.)
    return dict(schema='rblane.motion_point_table.v1',conditioning_end_s=300,
        fit_worlds=list(range(969400,969432)),native_only=True,population=pop,
        drivers={'969400:11':{**pop,'v0':34.,'accel':1.8}},
        input_sha256={'case':{'veh_kin.npy':'abc'}})


def test_motion_points_use_identity_prior_fallback_and_original_cav_settings(tmp_path):
    path = tmp_path/'points.json'
    path.write_text(json.dumps(payload()))
    physics = PhysicsConfig(sumo_fidelity=True)
    table = MotionPoints(path,physics)
    params = table.prepare(969400,'native',300,{'veh_kin.npy':'abc'},'case')
    got = params.settings(torch.tensor([[11,99,11]]),torch.tensor([[False,False,True]]))
    assert got['v0'][0,0] == 34
    assert got['v0'][0,1] == 30
    assert got['v0'][0,2] == physics.v0_mps
    assert got['accel'][0,0] == 1.8
    assert not any(key.startswith('lc') for key in got)
    with pytest.raises(ValueError,match='native causal'):
        table.prepare(969400,'mappo',300,{'veh_kin.npy':'abc'},'case')
    with pytest.raises(ValueError,match='identity mismatch'):
        table.prepare(969400,'native',300,{'veh_kin.npy':'different'},'case')


def test_motion_points_reject_out_of_fit_or_nonpositive_settings(tmp_path):
    path = tmp_path/'points.json'
    data = payload()
    data['drivers']['969432:11'] = data['drivers'].pop('969400:11')
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='out-of-FIT'):
        MotionPoints(path,PhysicsConfig())
    data = payload()
    data['drivers']['969400:11']['decel'] = -1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='invalid inferred'):
        MotionPoints(path,PhysicsConfig())
