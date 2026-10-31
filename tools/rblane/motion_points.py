"""Native-only sensitivity using already inferred, causal physical point settings."""
import json
import math
import hashlib
import inspect
from pathlib import Path

import torch

from .encoder import EncodedSettings
from wms.model.driver_encoder import PARAMETER_NAMES

PHYSICAL = {'v0','time_headway','accel','decel','min_gap'}


def reader_identity():
    return dict(reader=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                setting_map=hashlib.sha256(inspect.getsource(EncodedSettings.settings).encode()).hexdigest())


class MotionPoints:
    def __init__(self,path,physics):
        self.payload = json.loads(path.read_text())
        p = self.payload
        if (p['schema']!='rblane.motion_point_table.v1' or p['conditioning_end_s']!=300
                or p['fit_worlds']!=list(range(969400,969432)) or p['native_only'] is not True):
            raise ValueError('not the published causal native FIT point table')
        a,b,tau,s0,_ = physics.idm()
        defaults = dict(v0=physics.v0_mps,time_headway=tau,min_gap=s0,accel=a,decel=b,
            politeness=.2,mobil_threshold=physics.mobil_threshold_mps2,
            merge_gap=physics.lane_safe_gap_m,merge_urgency=1.)
        self.defaults = torch.tensor([defaults[k] for k in PARAMETER_NAMES],dtype=torch.float64)
        def vector(row):
            if set(row)!=PHYSICAL or not all(v>0 and math.isfinite(v) for v in row.values()):
                raise ValueError('invalid inferred physical point')
            return torch.tensor([row.get(k,defaults[k]) for k in PARAMETER_NAMES],dtype=torch.float64)
        self.population = vector(p['population'])
        self.drivers = {}
        for identity,row in p['drivers'].items():
            world,vid = map(int,identity.split(':'))
            if not 969400<=world<=969431 or vid<0:
                raise ValueError('out-of-FIT point identity')
            self.drivers.setdefault(world,{})[vid] = vector(row)

    def prepare(self,world,arm,t0,input_hashes,episode):
        if arm!='native' or t0!=300:
            raise ValueError('point table is restricted to its native causal history')
        if self.payload['input_sha256'].get(episode)!=input_hashes:
            raise ValueError('motion fit and diagnostic observation identity mismatch')
        return EncodedSettings(self.drivers.get(world,{}),self.population,self.defaults)
