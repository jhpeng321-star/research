import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from .warm_history import ObservableHistory, arrays, validate_history


def row(vid='a', speed=10.):
    return dict(vid=vid,s_m=-100.,v_mps=speed,lane=0,is_cav=False,length_m=5.)


class HistoryTests(unittest.TestCase):
    def test_observable_whitelist_and_owned_copy(self):
        h=ObservableHistory();rows=[row()];h.push(0,rows)
        rows[0]['v_mps']=99
        self.assertEqual(h.snapshot(0)[0]['rows'][0]['v_mps'],10)
        for field in ('hidden_setting','future_speed','vType'):
            with self.assertRaises(ValueError):h.push(1,[dict(row(),**{field:1})])

    def test_rolling_clock_and_future_noninterference(self):
        h=ObservableHistory()
        for t in range(121):h.push(t,[row()])
        f=h.snapshot(120)
        self.assertEqual([x['time_s'] for x in f],list(range(60,121)))
        f.append(dict(time_s=121,rows=[row()],entrance_counts=None))
        with self.assertRaises(ValueError):validate_history(f,120)
        with self.assertRaises(ValueError):h.push(120,[row()])

    def test_empty_and_counter_contract(self):
        self.assertIsNone(arrays(dict(time_s=300,rows=[],history=[])))
        h=ObservableHistory()
        with self.assertRaises(ValueError):h.push(300,[row()],[0,1.5])
        h.push(300,[row()],[1,2]);self.assertEqual(h.snapshot(300)[0]['entrance_counts'],[1,2])

    def test_identity_continuity_and_backward_acceleration(self):
        h=ObservableHistory()
        h.push(300,[row('old')]);h.push(301,[row('new'),row('old',12)])
        record=dict(time_s=301,rows=[row('new'),row('old',12)],history=h.snapshot(301))
        a=arrays(record)
        self.assertEqual(a['ids'],['new','old'])
        self.assertFalse(a['mask'][0,0]);self.assertEqual(a['kin'][1,0,5],0.)
        self.assertEqual(a['kin'][1,1,5],2.)
        bad=copy.deepcopy(record);bad['rows'][0]['v_mps']=11
        with self.assertRaises(ValueError):arrays(bad)


class FrozenConditioningTests(unittest.TestCase):
    def test_real_frozen_selector_all_clock_guards_move(self):
        import numpy as np
        import torch
        from .warm_runtime import moving_clock
        path=Path(os.environ['COOPMWM_MOTION_SOURCE'])
        spec=importlib.util.spec_from_file_location('frozen_motion_clock_test',path)
        frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
        for now in (300.,310.,400.,600.,900.):
            history=ObservableHistory()
            for t in range(int(now)-60,int(now)+1):history.push(t,[row('driver')])
            raw=arrays(dict(time_s=now,rows=[row('driver')],history=history.snapshot(now)))
            start=dict(t0=now,seed=986000,state=dict(present=torch.ones((1,1),dtype=torch.bool),
                is_cav=torch.zeros((1,1),dtype=torch.bool)))
            def extract(observed,*,start,stop):
                self.assertEqual((start,stop),(now-60,now))
                # Column0 is the beginning, column9 is the interval duration.
                return np.array([[now-2,0,0,10,10,100,10,-1,0,1]],dtype=float),None,None
            def posterior(*args,**kwargs):
                self.assertEqual(kwargs['t0_s'],now)
                return dict(weights=[1.],conditioning_end_s=now)
            def bind(drivers,*,t0_s):
                self.assertEqual(t0_s,now)
                self.assertEqual(set(drivers),{(986000,0)})
                return SimpleNamespace(provenance={})
            base=SimpleNamespace(population={'support':[[1.]]},with_motion_posteriors=bind)
            with patch.object(frozen,'chain_gaps',lambda observed,unused:observed):
                result,receipt=moving_clock(frozen.condition_population,now)(base,
                    SimpleNamespace(condition_motion=posterior),'frozen-policy',start,raw,
                    kernels=(extract,lambda *args:np.zeros((1,1)),'source-sha'))
            self.assertEqual(result.provenance['conditioning_interval_s'],[now-60,now])
            self.assertEqual(receipt['conditioned_drivers'],1)
            self.assertEqual(receipt['population_fallbacks'],0)

    def test_empty_history_never_invokes_warm_sampler(self):
        from tools.wmteach import marginal
        from .warm_runtime import adapter
        original=marginal.plans
        # The model intentionally has no warm API: empty history must delegate.
        rt=SimpleNamespace(model=object())
        with adapter(rt,dict(time_s=600,rows=[],history=[])):
            self.assertIs(marginal.plans,original)
        self.assertIs(marginal.plans,original)

    def test_posterior_slot_rekey_is_shared_across_future_rows_and_restored(self):
        import torch
        from tools.wmteach import marginal
        from . import warm_runtime
        posterior=dict(weights=[.2,.8],conditioning_end_s=600.)
        captured=[]
        def bind(drivers,*,t0_s):
            captured.append((drivers,t0_s));return 'bound'
        base=SimpleNamespace(with_motion_posteriors=bind)
        rt=SimpleNamespace(model=SimpleNamespace(_wmphys_fitted_state=base))
        fitted=SimpleNamespace(posterior={'drivers':{'986000:1':posterior}})
        def original(runtime,initial,kwargs,**kw):
            self.assertEqual(runtime.model._wmphys_fitted_state.with_motion_posteriors({},t0_s=600.),'bound')
            self.assertTrue(torch.equal(kwargs['driver_history'],torch.ones(1)))
            return [SimpleNamespace(receipt={})]
        prepared=(fitted,{},dict(conditioned_drivers=1))
        with patch.object(warm_runtime,'condition',return_value=prepared),patch.object(
                warm_runtime,'encoded_history',return_value={'driver_history':torch.ones(1)}),patch.object(
                marginal,'plans',original):
            with warm_runtime.adapter(rt,dict(world_id=986000,time_s=600)):
                result=marginal.plans(rt,{}, {},row_keys=[71,71,72,72])
                self.assertTrue(result[0].receipt['causal_motion_conditioning'])
                # Also guarantee cleanup after a failed evaluator.
                with self.assertRaisesRegex(RuntimeError,'probe'):
                    raise RuntimeError('probe')
            self.assertIs(marginal.plans,original)
        self.assertEqual(captured,[({(71,1):posterior,(72,1):posterior},600.)])


if __name__=='__main__':unittest.main()
