"""Population-only observation adapter preserving released physical LC/CAL fixes.

Only observable rosters/queues/request counters are accepted. No native exporter
identity assumptions, saved-ID posterior lookup or synthetic latent transplant.
"""
import copy


def plans(rt,initial,kwargs,*,row_keys,histories,queues,observations,t0_s,seed,n_samples):
    import torch
    from wms.control.hidden_state import plan_hidden_state_futures
    from wms.control.arrival_fit import LearnedArrivalFit
    fitted=rt.model._wmphys_fitted_state.with_motion_posteriors({},t0_s=t0_s)
    continuation=getattr(rt.model,'_wmphys_lane_continuation',None)
    if continuation is not None:
        calibration=getattr(fitted,'calibration',None)
        fitted=continuation.attach(fitted,{},t0_s=t0_s,
            receipt=dict(lc_conditioning='unobserved_population_only',saved_lc_driver_id_lookup=False))
        if calibration is not None:fitted=calibration.hidden(fitted)
    original=rt.model._wmphys_arrival_fit
    arrival=LearnedArrivalFit({**original.artifact,'conditioning_end_s':t0_s,
                              'worlds':copy.deepcopy(observations)})
    arrival.sha256=original.sha256
    if getattr(original,'calibration',None) is not None:
        arrival=original.calibration.arrivals(arrival)
    ids=torch.arange(initial['present'].shape[1],device=initial['present'].device,dtype=torch.long)
    ids=ids[None].expand_as(initial['present']).clone()
    initial['vehicle_id']=ids
    result=plan_hidden_state_futures(rt.model,initial,kwargs,future_sampling='hidden_state',
        n_samples=n_samples,seed=seed,row_keys=row_keys,fitted_state=fitted,vehicle_ids=ids,
        entrance_histories=histories,queues=queues,t0_s=t0_s,arrival_fit=arrival)
    for plan in result:
        plan.receipt.update(observation_contract='roster_and_observable_source_population_marginal',
            saved_ID_lookup=False,causal_motion_conditioning=False,lc_wish_initialization='cold',
            fitted_lc_population=continuation is not None,
            spread_calibration_preserved=getattr(fitted,'calibration',None) is not None)
    return result


def bind(rt,initial,kwargs,plan):
    import os
    from tools.rbint.recipe import InstrumentedEntries
    process=plan.kwargs.pop('entries');entries=InstrumentedEntries(process,rt.frozen.Entries)
    kwargs.update(plan.kwargs,entries=entries)
    if getattr(rt.model,'_wmphys_match_cav_execution_settings',False):
        from wms.control.execution_settings import bind_execution_settings
        plan.receipt.update(bind_execution_settings(kwargs,entries,initial,os.environ['C4_EXECUTION_DRIVER']))
    return kwargs,entries,plan.receipt
