"""CPU-only execution-count, behaviour-measure and fused-forward regressions."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import gate
from .fused import FusedModel
from .exploration import route,mixture_ratio
from tools.coopnative.test_cpu import rows


class VolumeCPU(unittest.TestCase):
    def test_fused_forward_preserves_private_windows_slots_and_clocks(self):
        class Model:
            training=False
            def rollout(self,state,*,n_steps,cav_controller,k0):
                out=cav_controller(state)
                mask=cav_controller.native_lane_mask(state)
                s=dict(state,s_m=out[...,0],v_mps=out[...,2])
                return dict(state=s,positions=s['s_m'][:,None],masks=mask[:,None])
        proxy=FusedModel(Model(),2)
        def one(i):
            proxy.local.index=i;n=3+i
            state=dict(present=torch.ones(2,n,dtype=torch.bool),s_m=torch.ones(2,n)*(i*10),v_mps=torch.ones(2,n))
            def cb(s):return torch.stack((s['s_m']+(i+1),s['s_m']*0,s['v_mps']+i,s['s_m']*0),-1)
            cb.native_lane_mask=lambda s:s['present']
            for t in range(3):
                v=proxy.rollout(state,n_steps=1,cav_controller=cb,k0=t)
                self.assertEqual(v['positions'].shape,(2,1,n))
                state=v['state']
            self.assertTrue(torch.equal(state['s_m'],torch.full((2,n),float(i*10+3*(i+1)))))
        with ThreadPoolExecutor(2) as p:list(p.map(one,range(2)))

    def test_gate_requires_closed_actual_guidance_and_random_regime_match(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);gate.initialize(p,{'merge-dense':1,'ordinary':2},{'merge-dense':.9,'ordinary':.45})
            with self.assertRaisesRegex(ValueError,'gate unmet'):gate.verify(p,[])
            # A verdict is immutable; use another independent fixture for a passing update.
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);gate.initialize(p,{'merge-dense':1,'ordinary':0},{'merge-dense':.9,'ordinary':.45})
            g=torch.Generator().manual_seed(42)
            sample=dict(context=dict(members=['v'],anchors=[[10.,0.]]),action=[[0.,0.]],
                        behaviour=dict(regime='merge-dense',sample_probability=.9,executed=False))
            prop=dict(accepted=True,option='O1',targets={'v':dict(speed=8.,tau=1.)},expected_saving=1.)
            v=gate.execute(p,sample,prop,1.,g)
            self.assertTrue(v['behaviour']['executed'])
            self.assertEqual(gate.sampling(p),{'merge-dense':0.,'ordinary':0.})
            second=dict(sample,behaviour=dict(sample['behaviour'],executed=False))
            v2=gate.execute(p,second,prop,1.,g)
            self.assertFalse(v2['behaviour']['executed']);self.assertEqual(v2['behaviour']['q'],0.)
            ep=[dict(steps=[dict(sample=v)])]
            self.assertTrue(gate.verify(p,ep)['pass_'])
            with self.assertRaisesRegex(ValueError,'counter differs'):gate.verify(p,[])

    def test_quota_done_own_route_has_exact_density(self):
        s=dict(logp=0.)
        self.assertFalse(route(s,rows(),1.,torch.Generator().manual_seed(1),{'merge-dense':0.,'ordinary':0.}))
        self.assertEqual(s['behaviour']['sample_probability'],0.)
        self.assertEqual(float(mixture_ratio(torch.tensor(0.),s)),1.)

    def test_restarted_checkpoint_namespace_never_reuses_legacy_B(self):
        from .runner import checkpoint_path
        root=Path('/fixture')
        b=dict(volume_amendment=True)
        self.assertEqual(checkpoint_path(root,b,0,'A',4).name,'S0_A_checkpoint_004.pt')
        self.assertEqual(checkpoint_path(root,b,0,'B',0).name,'S0_B1854_checkpoint_000.pt')
        self.assertEqual(checkpoint_path(root,b,0,'R',24).name,'S0_R1854_checkpoint_024.pt')

    def test_adopt_only_registered_unattempted_precharged_job(self):
        from . import runner
        from .policy import Policy
        from .storage import write
        class Pool:
            def __init__(self,**kw):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def map(self,fn,jobs):return [dict(world=j['world']) for j in jobs]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);w=dict(seed=986000);key='adopt_A_986000'
            (root/'CHARGES.jsonl').write_text('{"id":"'+key+'","phase":"dev","sim_s":1200}\n')
            b=dict(runtime='',checkpoint='',workers=1,unstarted_precharged_jobs=[key])
            with patch.object(runner,'admission'),patch.object(runner,'ProcessPoolExecutor',Pool),patch.object(runner,'reserve') as reserve:
                self.assertEqual(len(runner.wave(root,b,Policy(),[w],'A','dev','adopt','dev',True)),1)
                reserve.assert_not_called()
                write(root/'jobs'/key/'CLAIM.json',dict(pid=1))
                with self.assertRaisesRegex(ValueError,'cannot be adopted'):
                    runner.wave(root,b,Policy(),[w],'A','dev','adopt','dev',True)


if __name__=='__main__':unittest.main()
