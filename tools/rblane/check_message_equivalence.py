"""Certify new outward message semantics do not rescore teacher-forced choices."""
import hashlib
import json
import subprocess
import sys
import types

import torch

from .operate import RECEIPTS,ROOT
from wms.physics.config import PhysicsConfig
from wms.physics.lc2013 import lane_change,PRECISION

BASELINE='52d2df5c5a2543307e007edc06ca7d464804e8ae'


def main():
    out=RECEIPTS/'MESSAGE_DECISION_EQUIVALENCE.json'
    assert out.parent.is_dir() and not out.exists()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    raw=subprocess.check_output(['git','show',BASELINE+':src/wms/physics/lc2013.py'],cwd=ROOT)
    old=types.ModuleType('wms.physics._message_baseline'); old.__package__='wms.physics'
    old.__file__=str(ROOT/'src/wms/physics/lc2013.py'); sys.modules[old.__name__]=old
    exec(compile(raw,old.__file__,'exec'),old.__dict__)
    torch.manual_seed(1701); p=PhysicsConfig(sumo_fidelity=True)
    before={}; after={}; elements=0; message_changes=0; urgent=0
    for step in range(100):
        k=torch.zeros(8,12,7,dtype=torch.float32)
        k[...,0]=torch.rand(8,12)*2500-2000
        lane=torch.randint(0,3,(8,12)); k[...,1:4]=torch.nn.functional.one_hot(lane,3)
        k[...,4]=torch.rand(8,12)*35; k[...,5]=torch.rand(8,12)*6-3; k[...,6]=5
        mask=torch.rand(8,12)>.1
        params=dict(v0=20+torch.rand(8,12)*15,accel=.5+torch.rand(8,12)*2,
            decel=.5+torch.rand(8,12)*4,time_headway=.5+torch.rand(8,12)*2,
            min_gap=1+torch.rand(8,12)*2.5,lcCooperativeSpeed=torch.rand(8,12))
        left=torch.trunc(torch.rand(8,12,dtype=torch.float64)*60*PRECISION)
        right=torch.trunc(torch.rand(8,12,dtype=torch.float64)*60*PRECISION)
        for memory in (before,after):
            memory.update(lc2013_left=left.clone(),lc2013_right=right.clone(),
                          lc2013_present=mask.clone(),lc2013_lane=lane.clone())
        before=old.lane_change(k,mask,p,memory=before,driver_params=params)
        after=lane_change(k,mask,p,memory=after,driver_params=params)
        for key in before:
            if key=='lc2013_advice_accel': continue
            torch.testing.assert_close(before[key],after[key],rtol=0,atol=0,equal_nan=True)
            elements+=before[key].numel()
        message_changes+=int((before['lc2013_advice_accel']!=after['lc2013_advice_accel']).sum())
        urgent+=int(after['lc2013_speed_gain_urgent'].sum())
    result=dict(status='PASS',baseline_commit=BASELINE,synthetic_vehicle_decisions=9600,
        nonmessage_outputs='BIT_EXACT',compared_tensor_elements=elements,
        changed_outward_message_elements=message_changes,speed_gain_urgent_frames=urgent,
        scope='same observed snapshots; autonomous longitudinal feedback is deliberately changed',
        prior_lc_sha256=hashlib.sha256(raw).hexdigest(),
        current_lc_sha256=hashlib.sha256((ROOT/'src/wms/physics/lc2013.py').read_bytes()).hexdigest())
    out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
