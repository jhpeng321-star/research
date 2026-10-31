"""Scientific checks on the imported C1CLOSED metrics and panel-size adapter."""
import unittest
import numpy as np
from analyze import load_upstream, UPSTREAM, panel_summarizer

A=load_upstream(UPSTREAM)


class ScientificContracts(unittest.TestCase):
    def test_same_survivors_across_both_draws(self):
        real=dict(ids=[1,2],s=[0.,0.],v=[0.,0.])
        pred=dict(endpoints=[dict(ids=[1,2],s=[1.,1000.],v=[1.,1000.],present=[True,True]),
                             dict(ids=[1,2],s=[3.,9999.],v=[3.,9999.],present=[True,False])])
        result,ids=A.endpoint(pred,real,[1,2])
        self.assertEqual(ids,{1})
        self.assertEqual(result,dict(position=2.,speed=2.,n_hdv=1))

    def test_absent_survivors_are_missing_not_zero(self):
        result,_=A.endpoint(dict(endpoints=[dict(ids=[],s=[],v=[],present=[])]),
                            dict(ids=[1],s=[4],v=[5]),[1])
        self.assertEqual(result,dict(position=None,speed=None,n_hdv=0))

    def test_exact_world_bootstrap_dimension_and_schema(self):
        summarize,receipt=panel_summarizer(A,18)
        rows=[dict(model='SUMO_RNG_floor',plant='recorded',arm='native',world=w,
                   seed=969400+w,horizon_s=10,position=7.,speed=None) for w in range(18)]
        rows=[r for r in summarize(rows) if r['arm']=='native']
        p=next(r for r in rows if r['metric']=='position')
        self.assertEqual((p['estimate'],p['lo'],p['hi'],p['n_worlds']),(7.,7.,7.,18))
        self.assertEqual(p['bootstrap_draws'],2000)
        self.assertEqual(p['bootstrap_seed'],0)
        missing=next(r for r in rows if r['metric']=='speed')
        self.assertEqual((missing['estimate'],missing['n_worlds'],missing['status']),('',0,'no_surviving_ids'))
        self.assertEqual(list(p),['model','plant','arm','horizon_s','metric','estimate','lo','hi',
                                 'n_worlds','world_stat','units','bootstrap_draws','bootstrap_seed','status'])

    def test_pulse_target_exclusion_and_six_world_clusters(self):
        pred=dict(endpoints=[dict(ids=[1,2],s=[100.,3.],v=[100.,.2],present=[True,True])]*2)
        result,ids=A.endpoint(pred,dict(ids=[1,2],s=[0.,0.],v=[0.,0.]),[2])
        self.assertEqual(ids,{2})
        self.assertEqual(result['position'],3.)
        summarize,_=panel_summarizer(A,6)
        rows=[dict(model='SUMO_tiny_perturbation_chaos',plant='recorded',arm='native',
                   world=w,seed=969400+w,horizon_s=10,position=3.) for w in range(6)]
        r=next(r for r in summarize(rows) if r['arm']=='native')
        self.assertEqual((r['estimate'],r['lo'],r['hi'],r['n_worlds']),(3.,3.,3.,6))


if __name__=='__main__':unittest.main()
