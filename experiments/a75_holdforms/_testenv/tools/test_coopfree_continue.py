import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tools import coopfree_continue as c
from tools.coopfree.policy import SetPointPolicy


class ContinueCPU(unittest.TestCase):
    def test_budget_is_exact_extension(self):
        old = dict(screen=27360, repeat=13680, train=1650000, dev=172800, discovery=91200)
        self.assertEqual(sum(c.CAPS.values())+44960, 6000000)
        self.assertEqual(sum(c.CAPS[k]-v for k,v in old.items()), 4000000)
        self.assertEqual(c.CAPS['dev']-old['dev'], 4*24*1200)

    def test_random_subset_is_exact_and_outcome_blind(self):
        a = c.random_solos(156, np.random.default_rng(1))
        self.assertEqual(len(a),39)
        self.assertEqual(len(set(a)),39)
        self.assertEqual(a,c.random_solos(156,np.random.default_rng(1)))
        self.assertNotEqual(a,c.random_solos(156,np.random.default_rng(2)))

    def test_hold_key_ignores_noise_but_not_duration_or_world(self):
        w = dict(world=dict(seed=982000),t0=300)
        s = dict(context=dict(members=['x']), action=[[4.,0.]], window=10, rho=.2,sigma=[4.5,.5])
        key = c.hold_key(w,s)
        t = copy.deepcopy(s);t['rho']=.8;t['sigma']=[9,1]
        self.assertEqual(key,c.hold_key(w,t))
        t['window']=20
        self.assertNotEqual(key,c.hold_key(w,t))
        self.assertNotEqual(key,c.hold_key(dict(w,t0=360),s))

    def test_real_next_plan_has_fresh_joint_noise_and_whole_solos(self):
        p,_,sigma,_ = c.load(c.LEGACY,1)
        plan = c.make_plan(c.LEGACY,p,sigma,2,dict(id='test',es=False))
        self.assertEqual(len(plan['windows']),156)
        self.assertGreaterEqual(len({w['world']['seed'] for w in plan['windows']}),12)
        self.assertEqual(len(plan['exact_indices']),39)
        self.assertEqual(len({t['id'] for t in plan['tasks']}),len(plan['tasks']))
        for i in plan['exact_indices']:
            w = plan['windows'][i]
            solos = [t for t in plan['tasks'] if t['id'].startswith(f'cf2_u002_w{i:03}_solo')]
            self.assertEqual(len(solos),len(w['sample']['context']['members']))
            for j,t in enumerate(solos):
                member=w['sample']['context']['members'][j]
                self.assertEqual(t['sample'],c.solo_hold(w['hold'],w['sample'],member))
        other=c.make_plan(c.LEGACY,p,sigma,3,dict(id='test',es=False))
        self.assertNotEqual(plan['windows'][0]['sample'],other['windows'][0]['sample'])

    def test_original_binding_and_no_unfinished_jobs(self):
        c.runner.verify(c.LEGACY)
        for job in (c.LEGACY/'jobs').iterdir():
            if (job/'START.json').exists():
                self.assertTrue((job/'CLOSE.json').exists())

    def test_es_has_no_double_bias_gradient(self):
        p=SetPointPolicy();opt=torch.optim.Adam(p.parameters(),lr=.001)
        context=dict(x=[[0.]*43]*2, rsu=[0.]*12, anchors=[[10.,0.]]*2,members=['a','b'])
        generator=torch.Generator().manual_seed(4)
        rows=[]
        for i in range(156):
            s=p.sample(context,generator)
            s['es_eps']=[1.,-.5];s['es_sigma']=[4.5,.5]
            s['action']=(torch.tensor(s['action'])+torch.tensor(s['es_eps'])*torch.tensor(s['es_sigma'])).tolist()
            rows.append(dict(world=dict(seed=982000+i%24),sample=s,D_G={'600':100.},
                execution=dict(issued_changed=True,motion_changed=True,requests=2,overrides=0)))
        before=copy.deepcopy(p.state_dict())
        out=c.optimize_es(p,opt,rows)
        self.assertTrue(torch.allclose(p.mean.bias.grad,torch.tensor([-200/10000/4.5, .02])))
        self.assertFalse(torch.equal(before['mean.bias'],p.mean.bias))
        self.assertEqual(out['population_es']['additional_sim_s'],0)
        self.assertEqual(rows[0]['sample']['es_eps'],[1.,-.5])

    def test_polite_gate_never_relaxes_load(self):
        with patch.object(c.storage,'admission') as admit,patch.object(c.runner,'admission'):
            # MagicMock attributes are truthy; use a concrete function to exercise wrapper.
            calls=[]
            def original(root,max_load=20):calls.append(max_load)
            c.storage.admission=original
            c.configure()
            c.storage.admission('root',max_load=99)
            c.storage.admission('root',max_load=4)
            self.assertEqual(calls,[12.,4])


if __name__=='__main__':
    unittest.main()
