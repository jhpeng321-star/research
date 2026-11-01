"""Real released-type CPU regression for LC attachment preserving CAL wrappers."""
import types
import unittest
from unittest.mock import patch


class MarginalIntegration(unittest.TestCase):
    def test_population_contract_preserves_lc_spread_and_discards_saved_ids(self):
        import torch
        from wms.control.fitted_hidden import FittedHiddenState,IDM_ORDER,LC_ORDER
        from wms.control.spread_calibration import SpreadCalibration,CalibratedHidden,CalibratedArrivals
        from wms.control.arrival_fit import LearnedArrivalFit
        from wms.physics.lane_continuation import LaneContinuation
        from tools.wmteach.marginal import plans
        pop=dict(schema='wm_phys.idm_population.v1',parameter_order=IDM_ORDER,
                 distribution='finite_support_mixture',support=[[25.,1.,1.,2.,2.]],weights=[1.])
        post=dict(schema='wm_phys.idm_posteriors.v1',parameter_order=IDM_ORDER,
                  conditioning_end_s=300.,drivers={'969400:99':dict(weights=[1.])})
        fitted=FittedHiddenState(pop,post,provenance=dict(idm_population_sha256='population'))
        calibration=SpreadCalibration(dict(schema='wm_phys.spread_calibration.v1',version='test',
            fit_worlds=list(range(969400,969432)),conditioning_end_s=300,learned_corrections=False,
            idm_population_sha256='population',arrivals_sha256='arrivals',
            parameters=dict(posterior_retention=1.,arrival_variance_multiplier=1.7,entrant_concentration=0.)))
        lc=types.SimpleNamespace(population=dict(schema='wm_phys.lc2013_population.v1',
            parameter_order=LC_ORDER,support=[[1.,1.,1.,1.,1.]],weights=[1.]),provenance={})
        lc.attach=types.MethodType(LaneContinuation.attach,lc)
        arrival=LearnedArrivalFit(dict(schema='wm_phys.observed_arrivals.v1',
            generator_family='piecewise_constant_poisson',conditioning_end_s=300.,observed_phase_end_s=1200.,worlds={}))
        arrival.sha256='arrivals'
        model=types.SimpleNamespace(_wmphys_fitted_state=calibration.hidden(fitted),
            _wmphys_lane_continuation=lc,_wmphys_arrival_fit=calibration.arrivals(arrival))
        initial=dict(present=torch.ones((2,4),dtype=torch.bool))
        observations={'96940000':dict(total=[30,20],exposure_s=307.5,final_backlog=[2,1])}
        with patch('wms.control.hidden_state.plan_hidden_state_futures',
                   return_value=[types.SimpleNamespace(receipt={})]) as forecast:
            result=plans(types.SimpleNamespace(model=model),initial,{},row_keys=[96940000]*2,
                histories=[None]*2,queues=[None]*2,observations=observations,t0_s=307.5,seed=190,n_samples=1)
        kw=forecast.call_args.kwargs
        self.assertIsInstance(kw['fitted_state'],CalibratedHidden)
        self.assertEqual(kw['fitted_state'].posterior['drivers'],{})
        self.assertEqual(kw['fitted_state'].lane_posterior['drivers'],{})
        self.assertIsNotNone(kw['fitted_state'].lane_population)
        self.assertIsInstance(kw['arrival_fit'],CalibratedArrivals)
        self.assertEqual(kw['arrival_fit'].calibration.variance,1.7)
        self.assertEqual(kw['arrival_fit'].observation_receipt(96940000,t0_s=307.5)['total'],[30,20])
        self.assertEqual(fitted.posterior['drivers'],post['drivers'])
        self.assertTrue(result[0].receipt['spread_calibration_preserved'])
        self.assertTrue(torch.equal(initial['vehicle_id'],torch.arange(4).expand(2,4)))


if __name__=='__main__':unittest.main()
