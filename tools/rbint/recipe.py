"""Strict checkpoint -> encoder-preserving WM-PHYS, with frozen latent sampling."""
from dataclasses import replace
from pathlib import Path
import os
import tomllib

import torch

from tools.rbint.rebuild import rebuild_with_motion_encoder


def native_controller():
    """Same native CAV command/plant used by the registered five-arm evaluator."""
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
    if variant not in ('A', 'B'):
        raise ValueError('fitted C requires its motion-fit artifact before construction')
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
    receipt.update(source=source, variant=variant, recipe=recipe,
                   settings_B='inherited physics population defaults, not encoder centre',
                   sampling='hidden_state' if variant == 'A' else 'poisson_arrivals_fixed_population')
    return [candidate], receipt


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
    if settings == 'B':
        kin = state_kin(initial)
        defaults = model._default_driver_parameters(kin, *kin.shape[:2])
        population = Population(
            [defaults[b, :1].clone() for b in range(rows)],
            [defaults[b, :1].clone() for b in range(rows)],
            [defaults.new_ones(1) for _ in range(rows)])
    # All arms share a world key. Rebuild streams per task; never reuse mutable entries.
    spec = batch.entry_spec
    plans = plan_hidden_state_futures(
        model, initial, kwargs, future_sampling='hidden_state', n_samples=8,
        seed=190, row_keys=[int(seed)] * rows, population=population,
        entrance_histories=[spec['histories'][0]] * rows,
        queues=[spec['queues'][0]] * rows,
        sample_present=settings == 'A')
    plan = plans[future]
    if defaults is not None:
        plan.kwargs['driver_params'] = {n: defaults[..., i] for i, n in enumerate(PARAMETER_NAMES)}
        plan.receipt.update(posterior_source='fixed_inherited_physics_population', variant='B')
    else:
        plan.receipt['variant'] = 'A'
    process = plan.kwargs.pop('entries')
    entries = InstrumentedEntries(process, frozen.Entries)
    kwargs.update(plan.kwargs, entries=entries)
    return kwargs, entries, plan.receipt
