"""Strict checkpoint -> encoder-preserving WM-PHYS, with frozen latent sampling."""
from dataclasses import replace
from pathlib import Path
import os
import tomllib
import hashlib
import json

import torch

from tools.rbint.rebuild import rebuild_with_motion_encoder


def native_controller(model=None):
    """Native-CAV law selected by the recipe; default is the registered head."""
    if getattr(model, '_wmphys_native_model', 'registered_head') == 'sumo_physics':
        from wms.model.sumo_native import native_physics_controller
        return native_physics_controller(dt_s=model.physics.dt_s)
    from wms.control.counterfactual import native_fn
    from wms.control.continuous import project_native
    from wms.model.cav import WM_MERGE_KEY, WM_LANE_TARGET_KEY
    native = native_fn('head', a_max=1.2)

    def callback(state):
        mask = state['present'].bool() & state['is_cav'].bool()
        state[WM_MERGE_KEY] = mask & (state['lane'] == 2)
        lane = torch.where(state['lane'] != 2,
                           state.get(WM_LANE_TARGET_KEY, state['lane']), state['lane'])
        speed = (state['v_mps'] + .5 * project_native(state, native(state))).clamp_min(0.)
        return torch.stack((state['s_m'] + .5 * speed, lane, speed,
                            (speed - state['v_mps']) / .5), -1)

    callback.native_lane_mask = lambda s: s['present'].bool() & s['is_cav'].bool()
    return callback


def load_candidate(checkpoints, configs, device):
    from wms.control.policy14_cli import load_members
    recipe = tomllib.loads(Path(configs[0]).read_text(encoding='utf-8'))
    variant = os.environ['WMPHYS_VARIANT']
    if variant not in ('A', 'B', 'C', 'FULL'):
        raise ValueError('unknown WM-PHYS variant')
    originals, source = load_members(checkpoints, [recipe['source']['config']], device)
    old = originals[0]
    options = dict(recipe['physics'])
    if options.get('emergency_decel_mps2') == 'off':
        options['emergency_decel_mps2'] = None
    physics = replace(old.physics, **options)
    spec = replace(old.spec, **recipe['model'])
    candidate, receipt = rebuild_with_motion_encoder(old, spec, physics)
    candidate.requires_grad_(False)
    candidate._wmphys_variant = variant
    candidate._wmphys_departure_phase = recipe.get('rollout', {}).get('departure_phase', 'before_movement')
    if candidate._wmphys_departure_phase not in ('before_movement', 'after_movement_and_lanes'):
        raise ValueError('unsupported departure phase')
    receipt['departure_phase'] = candidate._wmphys_departure_phase
    if variant == 'FULL':
        from wms.control.fitted_hidden import FittedHiddenState
        from wms.control.arrival_fit import LearnedArrivalFit
        root = Path(__file__).resolve().parents[2]
        options_full = recipe['variants']['FULL']
        candidate._wmphys_match_cav_execution_settings = options_full.get('match_cav_execution_settings', False)
        if not isinstance(candidate._wmphys_match_cav_execution_settings, bool):
            raise ValueError('match_cav_execution_settings must be boolean')
        if candidate._wmphys_match_cav_execution_settings:
            receipt['match_cav_execution_settings'] = True
        if options_full['fitted_lc']:
            raise ValueError('fitted LC has not passed its baseline')
        candidate._wmphys_native_model = options_full.get('native_model', 'registered_head')
        if candidate._wmphys_native_model not in ('registered_head', 'sumo_physics'):
            raise ValueError('unsupported native-CAV law')
        receipt['native_model'] = candidate._wmphys_native_model
        candidate._wmphys_fitted_state = FittedHiddenState.from_directory(
            root/options_full['parameter_directory'])
        candidate._wmphys_arrival_fit = LearnedArrivalFit.from_file(root/options_full['arrival_json'])
        candidate._wmphys_condition_motion = options_full.get('condition_motion', False)
        if candidate._wmphys_condition_motion:
            if (physics.cf_model != 'sumo_idm' or physics.sumo_cf_stepping_s != .25
                    or physics.sumo_cf_emergency_decel_mps2 != 9.
                    or physics.sumo_cf_vtype_max_mps is not None):
                raise ValueError('motion-profile predictor must match the frozen SUMO CF recipe')
            from wms.physics.motion_fit import MotionFit
            candidate._wmphys_motion_fit = MotionFit(root/options_full['parameter_directory'])
            candidate._wmphys_shrinkage_policy = root/options_full['parameter_directory']/'idm_shrinkage.json'
            candidate._wmphys_conditioned_cache = {}
            receipt['saved_driver_id_lookup'] = False
            receipt['runtime_motion_conditioning'] = True
            receipt['shrinkage_policy_sha256'] = hashlib.sha256(
                candidate._wmphys_shrinkage_policy.read_bytes()).hexdigest()
        receipt['fitted_hidden'] = candidate._wmphys_fitted_state.provenance
        receipt['learned_arrivals_sha256'] = candidate._wmphys_arrival_fit.sha256
    if variant == 'C':
        artifact = Path(__file__).resolve().parents[2] / recipe['variants']['C']['parameter_json']
        data = json.loads(artifact.read_text(encoding='utf-8'))
        assert data['schema'] == 'wm_phys.idm_pooled.v1'
        assert data['fit_interval_s'] == [240, 300]
        assert data['fit_worlds'] == list(range(969400, 969432))
        candidate._wmphys_fit = data['parameters']
        receipt['fitted_physics'] = dict(path=str(artifact),
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(), parameters=data['parameters'],
            uncertainty=data['uncertainty_note'])
    receipt.update(source=source, variant=variant, recipe=recipe,
                   settings_B='inherited physics population defaults, not encoder centre',
                   sampling=('fitted_hidden_state_and_learned_arrivals' if variant == 'FULL' else
                             'hidden_state' if variant == 'A' else 'poisson_arrivals_fixed_population'))
    return [candidate], receipt


def fixed_parameters(model, state):
    from wms.physics.state import state_kin
    kin = state_kin(state)
    values = model._default_driver_parameters(kin, *kin.shape[:2]).clone()
    if model._wmphys_variant == 'C':
        from wms.model.driver_encoder import PARAMETER_NAMES
        hdv = ~state['is_cav'].bool()
        for name, value in model._wmphys_fit.items():
            i = PARAMETER_NAMES.index(name)
            values[..., i] = torch.where(hdv, float(value), values[..., i])
    return values


def fitted_state_at_start(model, seed):
    fitted = model._wmphys_fitted_state
    if not getattr(model, '_wmphys_condition_motion', False):
        return fitted, {}
    from tools.rbint.observations import load_motion_history
    from tools.rbint.motion_posterior import condition_population
    if int(seed) not in model._wmphys_conditioned_cache:
        start, raw = load_motion_history(seed)
        model._wmphys_conditioned_cache[int(seed)] = condition_population(
            fitted, model._wmphys_motion_fit, model._wmphys_shrinkage_policy, start, raw)
    return model._wmphys_conditioned_cache[int(seed)]


class InstrumentedEntries:
    """Delegate ordinal/queue interfaces while preserving the frozen cost observer.

    The Poisson source has _next rather than _headway. Observe its already-drawn
    request times to integrate fractional-step waiting exactly, with no extra draw.
    """
    def __init__(self, process, observer_type):
        self.process = process
        self.observer = observer_type(process)
        original = process._next

        def next_wait(b, entrance):
            self.observer.due_area[b] += max(0., process.now[b] - process.t_next[b, entrance])
            return original(b, entrance)

        process._next = next_wait
        self.initial_backlog = process.pending.copy()

    def __getattr__(self, name):
        if hasattr(self.observer, name):
            return getattr(self.observer, name)
        return getattr(self.process, name)

    def insert(self, state):
        return self.observer.insert(state)

    def audit_population(self, initial, rollout):
        import numpy as np
        masks = rollout['masks'].bool()
        births = torch.stack(self.new_masks, 1).bool()
        before = torch.stack([s['present'] for s in self.states], 1).bool()
        previous = torch.cat((initial['present'][:, None].bool(), masks[:, :-1]), 1)
        if not torch.equal(before, previous | births) or (previous & births).any():
            raise AssertionError('admission/current cohort mismatch')
        exits = before & ~masks
        if (masks & ~before).any():
            raise AssertionError('unrecorded vehicle creation')
        boundary = initial['exit_s_m'][:, None]
        if (exits & ~(rollout['positions'] > boundary)).any():
            raise AssertionError('vehicle vanished before road exit')
        for name in ('positions', 'speeds', 'accel', 'lanes'):
            if not torch.isfinite(rollout[name]).all():
                raise AssertionError('nonfinite '+name)
        if rollout['raw_residual_mps2'].count_nonzero():
            raise AssertionError('learned correction emitted')
        lanes = torch.stack([s['lane'] for s in self.states], 1)
        admitted = np.stack([(((lanes == 2) == bool(e)) & births).sum((1, 2)).cpu().numpy()
                             for e in (0, 1)], -1)
        if not np.array_equal(self.process.due_total + self.initial_backlog,
                              self.process.pending + admitted):
            raise AssertionError('request/admission/source ledger not conserved')
        return dict(finite=True, conserved=True, due=self.process.due_total.tolist(),
            initial_backlog=self.initial_backlog.tolist(), admitted=admitted.tolist(),
            final_pending=self.process.pending.tolist(), exits=exits.sum((1, 2)).tolist())


def candidate_future(*, model, batch, initial, future, seed, frozen, **unused):
    from wms.control.hidden_state import Population, plan_hidden_state_futures
    from wms.model.driver_encoder import PARAMETER_NAMES
    from wms.physics.state import state_kin
    rows = initial['present'].shape[0]
    kwargs = frozen.expand(batch.rollout_kwargs, rows)
    settings = model._wmphys_variant
    population = None
    defaults = None
    extra = {}
    observation_receipt = {}
    if settings == 'FULL':
        from tools.rbint.observations import fitted_inputs
        ids, arrivals, observation_receipt = fitted_inputs(initial, seed,
            batch.metadata[0]['initial_state_sha256'], model._wmphys_arrival_fit)
        fitted, motion_receipt = fitted_state_at_start(model, seed)
        observation_receipt.update(motion_receipt)
        extra = dict(fitted_state=fitted, vehicle_ids=ids,
                     t0_s=300., arrival_fit=arrivals)
    if settings in ('B', 'C'):
        defaults = fixed_parameters(model, initial)
        cav_defaults = model._default_driver_parameters(state_kin(initial), *defaults.shape[:2])
        hdv_defaults = cav_defaults[:, :1].clone()
        if settings == 'C':
            for name, value in model._wmphys_fit.items():
                hdv_defaults[..., PARAMETER_NAMES.index(name)] = float(value)
        population = Population(
            [hdv_defaults[b].clone() for b in range(rows)],
            [cav_defaults[b, :1].clone() for b in range(rows)],
            [defaults.new_ones(1) for _ in range(rows)])
    # All arms share a world key. Rebuild streams per task; never reuse mutable entries.
    spec = batch.entry_spec
    plans = plan_hidden_state_futures(
        model, initial, kwargs, future_sampling='hidden_state', n_samples=8,
        seed=190, row_keys=[int(seed)] * rows, population=population,
        entrance_histories=[spec['histories'][0]] * rows,
        queues=[spec['queues'][0]] * rows,
        sample_present=settings in ('A', 'FULL'), **extra)
    plan = plans[future]
    if defaults is not None:
        plan.kwargs['driver_params'] = {n: defaults[..., i] for i, n in enumerate(PARAMETER_NAMES)}
        plan.receipt.update(posterior_source=('fixed_inherited_physics_population'
            if settings == 'B' else 'motion_fitted_pooled_point'), variant=settings)
    else:
        plan.receipt['variant'] = settings
    plan.receipt.update(observation_receipt)
    plan.receipt['departure_phase'] = getattr(model, '_wmphys_departure_phase', 'before_movement')
    process = plan.kwargs.pop('entries')
    entries = InstrumentedEntries(process, frozen.Entries)
    kwargs.update(plan.kwargs, entries=entries)
    if getattr(model, '_wmphys_match_cav_execution_settings', False):
        from wms.control.execution_settings import bind_execution_settings
        path = os.environ.get('C4_EXECUTION_DRIVER')
        if not path:
            raise ValueError('matched CAV settings require the registered execution-driver artifact')
        plan.receipt.update(bind_execution_settings(kwargs, entries, initial, path))
    return kwargs, entries, plan.receipt
