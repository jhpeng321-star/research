"""Physics-forbidden regression against the frozen FULL-v4 WM executor library.

Runs on GPU hosts with CUDA disabled: cpuQ's older SUMO runtime does not include
the qualified WM execution-cap library. No model rollout or SUMO is launched.
"""
import copy
import os
import unittest
import torch
from tools.coopnative.test_cpu import state


class ActionResponse(unittest.TestCase):
    def test_WM_responds_to_member_action_and_reverts_to_MAPPO(self):
        from tools.wmteach.runtime import option_controller
        from wms.control.mappo.eval_bridge import load_policy_fn
        from .wm import executor_adapter
        from wms.model.cav import WM_LANE_TARGET_KEY
        torch.set_num_threads(1)
        mp=load_policy_fn(os.environ['COOPMWM_MAPPO'],device='cpu')
        r=[dict(vid='member',s_m=-200.,v_mps=20.,lane=0,is_cav=True,length_m=5.)]
        acts=[dict(id='MAPPO',window=10,targets={}),
              dict(id='slow',window=10,targets={'member':dict(speed=0.,tau=2.)})]
        s={k:v.expand(2,*v.shape[1:]).clone() for k,v in state(r).items()}
        s['a_mps2']=torch.zeros_like(s['v_mps'])
        s[WM_LANE_TARGET_KEY]=s['lane'].clone()  # Fixture's native WM lane substep stays.
        one={k:v[:1].clone() for k,v in s.items()}
        with executor_adapter():
            out=option_controller(mp,acts,r)(copy.deepcopy(s))
            baseline=option_controller(mp,[acts[0]],r)(one)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(float(abs(out[0,0,0]-out[1,0,0])),0.)
        self.assertGreater(float(abs(out[0,0,2]-out[1,0,2])),0.)
        self.assertLess(float(out[1,0,3]),float(out[0,0,3]))
        self.assertTrue(torch.equal(out[:1],baseline))
