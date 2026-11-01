"""One-factor development-only adapter to the frozen five-arm WM inference harness.

No fitting or future observations. Interventions are diagnostics, not deployment
recipes. The registered start extractor is filtered BEFORE any world job opens.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
DEV = tuple(range(969420, 969432))
FACTORS = ('baseline', 'no_mainline_lc', 'no_merge_advice', 'depart_last',
           'depart_first', 'population', 'raw_posterior', 'point_v0', 'point_tau', 'fixed_rate',
           'lane_limits', 'cav_settings', 'native_uncommanded')


def filtered_starts(code, worlds):
    """Inject an exact world allowlist as the FIRST operation in the world loop."""
    tree = ast.parse(code)
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Tuple)
             and [getattr(x, 'id', None) for x in n.target.elts] == ['i', 'world']]
    if len(loops) != 1 or not set(worlds) <= set(DEV):
        raise ValueError('unexpected extraction contract or nondevelopment world')
    loops[0].body.insert(0, ast.parse(f"if world['seed'] not in {tuple(worlds)!r}: continue").body[0])
    return ast.fix_missing_locations(tree)


def install_factor(factor):
    import torch
    from wms.control.arrival_fit import LearnedArrivalFit
    from wms.control.fitted_hidden import FittedHiddenState
    from wms.model.sumo_insertion import SumoInsertion
    from wms.physics import junction_merge, lc2013
    if factor == 'no_mainline_lc':
        original = lc2013.lane_change
        def lanes(kin, present, *args, **kwargs):
            result = original(kin, present, *args, **kwargs)
            result['next_lane'] = kin[..., 1:4].argmax(-1).clone()
            result['lc2013_position_shift'] = torch.zeros_like(kin[..., 0])
            return result
        lc2013.lane_change = lanes
        from wms.model import lc2013_adapter
        lc2013_adapter.lane_change = lanes
    elif factor == 'no_merge_advice':
        original = junction_merge.merge_speed_advice
        def advice(kin, present, accel, physics, driver_params=None, **kwargs):
            _, geometry = original(kin, present, accel, physics, driver_params, **kwargs)
            p = junction_merge.parameters(kin, physics, driver_params)
            stop = junction_merge.idm_next_speed(kin[..., 4],
                (physics.merge_zone_s1_m-kin[..., 0]).clamp_min(0),
                torch.zeros_like(accel), p, physics.dt_s, stop=True)
            wanted = kin[..., 4]+physics.dt_s*accel
            wanted = torch.where(geometry['ramp'], torch.minimum(wanted, stop), wanted)
            return torch.where(present, (wanted-kin[..., 4])/physics.dt_s,
                               torch.zeros_like(accel)), geometry
        junction_merge.merge_speed_advice = advice
    elif factor in ('depart_last', 'depart_first'):
        original = SumoInsertion.__init__
        def init(self, physics, mix=None, **kwargs):
            kwargs['depart_speed' if factor == 'depart_last' else 'depart_lane'] = (
                'last' if factor == 'depart_last' else 'first')
            original(self, physics, mix, **kwargs)
        SumoInsertion.__init__ = init
    elif factor == 'fixed_rate':
        original = LearnedArrivalFit.draw
        def draw(self, *args, **kwargs):
            kwargs['sample_rate'] = False
            return original(self, *args, **kwargs)
        LearnedArrivalFit.draw = draw
    elif factor == 'raw_posterior':
        from wms.physics.motion_fit import MotionFit
        original = MotionFit.condition_motion
        def condition(self, *args, **kwargs):
            kwargs['policy_path'] = None  # exact zero-prior-count endpoint
            return original(self, *args, **kwargs)
        MotionFit.condition_motion = condition
    elif factor in ('point_v0', 'point_tau'):
        # Collapse exactly one physical coordinate, preserving joint particle keys,
        # all other support coordinates and weights. Diagnostic, not a refit.
        coordinate = 0 if factor == 'point_v0' else 1
        original = FittedHiddenState.draw
        def draw(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if not kwargs.get('lane', False):
                name = 'v0' if coordinate == 0 else 'time_headway'
                result[name] = float((self.support[:, coordinate]*self.weights).sum()/self.weights.sum())
            return result
        FittedHiddenState.draw = draw
        # Entrants use as_population support directly, not draw().
        pop = FittedHiddenState.as_population
        def as_population(self, *args, **kwargs):
            result = pop(self, *args, **kwargs)
            from wms.model.driver_encoder import PARAMETER_NAMES
            index = PARAMETER_NAMES.index('v0' if coordinate == 0 else 'time_headway')
            for pool in result.hdv:
                pool[:, index] = float((self.support[:, coordinate]*self.weights).sum()/self.weights.sum())
            return result
        FittedHiddenState.as_population = as_population


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--factor', required=True, choices=FACTORS)
    parser.add_argument('--dev-worlds', nargs='+', type=int, default=list(DEV))
    parser.add_argument('--futures', nargs='+', type=int, default=[0])
    options, rest = parser.parse_known_args()
    if (not options.dev_worlds or len(set(options.dev_worlds)) != len(options.dev_worlds)
            or not set(options.dev_worlds) <= set(DEV)):
        raise ValueError('ONLY development969420..969431 permitted')
    if not options.futures or not set(options.futures) <= set(range(8)):
        raise ValueError('future indices0..7 only')
    sys.argv = [sys.argv[0], *rest]
    os.environ['WMPHYS_VARIANT'] = 'FULL'
    from tools.rbeval import run_wm
    from tools.rbint import recipe
    original_reference = run_wm.reference
    def reference(key, namespace):
        if key != 'starts':
            return original_reference(key, namespace)
        ref = run_wm.load(run_wm.HERE/'references.json')[key]
        assert hashlib.sha256(ref['code'].encode()).hexdigest() == ref['code_sha256']
        exec(compile(filtered_starts(ref['code'], options.dev_worlds),
                     '<development-only registered starts>', 'exec'), namespace)
        return ref['code_sha256']
    run_wm.reference = reference
    install_factor(options.factor)
    def load(*args):
        models, receipt = recipe.load_candidate(*args)
        receipt['rb2break_factor'] = options.factor
        receipt['rb2break_world_allowlist'] = options.dev_worlds
        if options.factor == 'lane_limits':
            from dataclasses import replace
            models[0].physics = replace(models[0].physics, sumo_cf_lane_speed_limits=True)
            receipt['sumo_cf_lane_speed_limits'] = True
        if options.factor == 'native_uncommanded':
            models[0]._wmphys_native_model = 'sumo_uncommanded'
            models[0]._wmphys_match_cav_execution_settings = True
            receipt['native_model'] = 'sumo_uncommanded'
            receipt['match_cav_execution_settings'] = True
        if options.factor == 'population':
            # Keep causal arrivals; disable only own-driver posterior information.
            models[0]._wmphys_condition_motion = False
            fitted = models[0]._wmphys_fitted_state
            fitted.posterior = {**copy.deepcopy(fitted.posterior), 'drivers': {}}
            fitted._cache.clear()
        return models, receipt
    sources = [Path(__file__), ROOT/'tools/rbint/recipe.py', ROOT/'tools/rbint/rebuild.py',
        ROOT/'tools/rbint/departure.py', ROOT/'tools/rbint/observations.py',
        ROOT/'tools/rbint/motion_posterior.py', ROOT/'tools/rbfit/idm_fit.py',
        ROOT/'tools/rbint/endpoints.py', ROOT/'configs/native_cav_head_v5.json',
        ROOT/'artifacts/rbfit/shrinkage_v1/idm_population.json',
        ROOT/'artifacts/rbfit/shrinkage_v1/idm_posteriors.json.gz',
        ROOT/'artifacts/rbfit/shrinkage_v1/idm_shrinkage.json',
        ROOT/'docs/paper_results/rbunc/observed_arrivals.json']
    def future(**kwargs):
        planned, entries, receipt = recipe.candidate_future(**kwargs)
        if options.factor == 'cav_settings':
            from wms.control.execution_settings import bind_execution_settings
            receipt.update(bind_execution_settings(planned, entries, kwargs['initial'],
                ROOT/'configs/execparity_cav_idm_v1.json'))
        return planned, entries, receipt
    run_wm.main(model_loader=load, future_builder=future,
                identity_sources=sources, worlds=options.dev_worlds,
                future_indices=options.futures)


if __name__ == '__main__':
    main()
