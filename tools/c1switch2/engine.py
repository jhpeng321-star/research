"""Frozen v7 WM2/BB execution extended to six rows, including real WALL feedback.

MAIN00:29 permits common road-only scoring. Entries/mix are OFF; evaluation is
full-team. This module never invents a source cost or calls it predicted team delay.
"""
import copy
import importlib.util
import os
from pathlib import Path

from tools.c1switch import adapter as v7
from tools.c1switch.feedback import seed_state

from .design import CONTROLLERS, DESIGN, HORIZON, K, PROTOCOL
from .observable import anonymous_payload, future_seed
from .wall import Wall


def legacy(payload):
    result = copy.deepcopy(payload)
    result['controller_state'].pop('wall')
    if result['controller_state']['incumbent'] == 'wall':
        result['controller_state']['incumbent'] = 'native'
    return result


def circles_callback(adapter, payload, tracker, state):
    # Private namespace, as in backend.frozen_overlay: reuse v7 code without
    # mutating any loaded five-way module or a frozen runtime source on disk.
    spec = importlib.util.spec_from_file_location('tools.c1switch._wall_fallback', Path(v7.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CONTROLLERS = ('circles',)
    return module.Adapter.callback(adapter, payload, tracker, state)


class Engine(v7.Adapter):
    @staticmethod
    def execution_dependencies():
        from wms.model.cav import WM_LANE_TARGET_KEY, WM_MERGE_KEY
        from wms.control.continuous import project_native
        return WM_LANE_TARGET_KEY, WM_MERGE_KEY, project_native

    def __init__(self, binding, device='cuda'):
        from tools.c1switch.design import sha
        if binding['model'] not in ('wm', 'bb') or binding['objective_design_sha256'] != sha(DESIGN):
            raise ValueError('road-only engine must bind the current registered design')
        super().__init__(binding, device)
        self.attestation.update(protocol=PROTOCOL, library=list(CONTROLLERS), history_s=60,
            horizon_s=HORIZON, K=K, objective='on_road_fleet_delay',
            evaluation_objective='full_team_delay_fleet_plus_source',
            objective_design_sha256=binding['objective_design_sha256'],
            observable_source_admissions=False, controller_feedback=True,
            wall_start_stop=True, anonymous_seed=True, entries=False,
            continuation='self-only wall schema/1; missing member abort; fresh candidate memory')

    def callback6(self, payload, tracker, state):
        import torch
        from tools.c1switch.library import make_wall_rollout
        from tools.c1switch.merge_fast import gap6
        from wms.model.cav import apply_exec_cap
        WM_LANE_TARGET_KEY, WM_MERGE_KEY, project_native = self.execution_dependencies()
        old = legacy(payload)
        peers = super().callback(old, tracker, state)
        fallback = circles_callback(self, old, tracker, state)
        native = self.frozen.native_fn('head', a_max=1.2)

        def execute(row, request, target, mask, base):
            remaining = 1. if wall.step_index % 2 else .5
            request = torch.minimum(request, (target-row['v_mps'])/remaining)
            accel = torch.minimum(project_native(row, native(row)), request)
            accel = apply_exec_cap(row, accel, mask)
            motion = self.frozen.parity_motion(row, row['lane'], accel)
            return torch.where(mask[..., None], motion, base)

        wall = make_wall_rollout(self.reference, fallback=fallback, execute=execute,
                                start_s=payload['history'][-1]['now_s'])
        memory = payload['controller_state']['wall']
        if payload['controller_state']['incumbent'] == 'wall' and memory is not None:
            slots = {tr.cav_id: tr.slot for tr in tracker.tracks.values() if tr.is_cav}
            absent = torch.where(~state['present'][0].bool())[0].tolist()
            wall.controllers = [Wall.restore(memory, slots=slots, absent_slots=absent,
                                             k=state['s_m'].shape[1])]

        def select_rows(state, start, stop):
            return {k: v[start:stop] if torch.is_tensor(v) and v.ndim and v.shape[0] == 6 else v
                    for k, v in state.items()}

        def callback(current):
            current[WM_LANE_TARGET_KEY] = current['lane']
            first = peers(select_rows(current, 0, 5))
            row = select_rows(current, 5, 6)
            # The wall candidate and its CIRCLES fallback use the SAME native lane plant.
            row[WM_MERGE_KEY] = row['present'].bool() & row['is_cav'].bool() & (row['lane'] == 2)
            last = wall(row)
            cav = row['present'].bool() & row['is_cav'].bool()
            last[..., 1] = torch.where(cav, gap6(row, cav), last[..., 1])
            current[WM_MERGE_KEY] = torch.zeros_like(current['present'], dtype=torch.bool)
            return torch.cat((first, last))
        return callback

    def imagine_all(self, raw, *, K=8, horizon_s=60, steps=None):
        import torch
        from tools.c1switch.observable import rollout_inputs
        if K != 8 or horizon_s != 60:
            raise ValueError('registered K8/60s required')
        payload = anonymous_payload(raw, Wall.validate_memory)
        old = legacy(payload)
        n = 120 if steps is None else steps
        if not 1 <= n <= 120:
            raise ValueError('invalid validation rollout length')
        os.environ['WMS_SPEED_PIVOT_DIAG_OFF'] = '1'
        state, kwargs, tracker = rollout_inputs(old, self.model)
        present = torch.where(state['present'][0])[0]
        vacant = torch.where(~state['present'][0])[0][:2]
        keep = torch.cat((present, vacant))
        if not len(keep):
            raise ValueError('observable rollout has no slots')
        remap = {int(slot): i for i, slot in enumerate(keep)}
        for tr in tracker.tracks.values():
            tr.slot = remap[tr.slot]
        state = v7.move({key: value[:, keep] for key, value in state.items()}, self.device)
        kwargs = v7.move({key: value[:, keep] for key, value in kwargs.items()}, self.device)
        state['exit_s_m'] = torch.full_like(state['s_m'], self.frozen.road_end_exit_bound(self.reference.physics))
        rows = [seed_state(copy.deepcopy(state), tracker, old, c if c != 'wall' else 'native')
                for c in CONTROLLERS]
        initial = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
        seed = future_seed(payload)
        torch.manual_seed(seed)
        plans = (self.frozen.SampledFutures(self.frozen.FuturesConfig.from_trainer(self.cfg), seed)
                 .plan([self.model], state, kwargs, n)[0]) if self.binding['model'] == 'wm' else [{}]*8
        samples = {c: [] for c in CONTROLLERS}
        deterministic_bb = self.binding['model'] == 'bb'
        if deterministic_bb and (type(self.model).__name__ != 'BlackBoxTrajModel' or
                                bool(getattr(self.model, '_samples_lanes', False)) or self.model.training):
            raise ValueError('BB deterministic memoization eligibility failed')
        for k in range(1 if deterministic_bb else 8):
            st = copy.deepcopy(initial)
            self.frozen.pair_merge_streams(st, torch.zeros(6, dtype=torch.long, device=self.device), 1, seed+k)
            kw = self.frozen.expand({**kwargs, **plans[k]}, 6)
            lanes = self.frozen.lane_uniform_draws([self.model], 1, n, state['s_m'].shape[1],
                                                  seed+1000003*k, device=self.device)
            if lanes is not None:
                kw['lane_uniform'] = self.frozen.expand(lanes, 6)
            costs = torch.zeros(6, device=self.device)

            class RoadRecorder:
                def insert(self, current):
                    costs.add_(.5*(torch.clamp(1-current['v_mps']/31.29, min=0)*current['present']).sum(-1))
                    return torch.zeros_like(current['present'], dtype=torch.bool)

            kw['entries'] = RoadRecorder()
            with torch.inference_mode():
                roll = self.model.rollout(st, n_steps=n, cav_controller=self.callback6(payload, tracker, st), **kw)
            if not torch.isfinite(roll['speeds']).all() or not torch.isfinite(costs).all():
                raise ValueError('nonfinite model rollout')
            for i, c in enumerate(CONTROLLERS):
                samples[c].append(float(costs[i]))
        return {c: values*8 for c, values in samples.items()} if deterministic_bb else samples


def create(binding, device='cuda'):
    from .inference import Predictor
    return Predictor(Engine(binding, device), Wall())
