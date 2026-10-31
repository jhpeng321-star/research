"""Plant-independent wall rollout adapter for WM2 and WM-PHYS.

The fallback callback is the consumer's existing frozen CIRCLES callback. An
optional execution adapter keeps the consumer's native plant/cap/lane/safety
machinery intact. No checkpoint, privileged data or SUMO process is needed.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from wms.sim_bridge import make_switchable_wall


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


class WallRolloutController:
    """``cav_controller(state) -> [B,K,4]`` with a 1 s control clock.

    Construct per independent future; deepcopy preserves a continuation. ``s_m``
    is merge-relative. ``vid_index`` [B,K] is optional for fixed-roster rollouts,
    required when slots may be recycled. ``execute(state, request, target, mask,
    base_motion)`` may apply the frozen consumer's execution path; without it the
    TimelineControllers speedMode31 native-minimum execution convention is used.
    """

    def __init__(self, model, *, fallback, execute=None, start_s=0.):
        if float(model.physics.dt_s) != 0.5:
            raise ValueError("wall rollout requires the frozen 0.5 s physics clock")
        if not np.isfinite(start_s) or start_s != round(start_s):
            raise ValueError("start_s must be on the one-second decision grid")
        self.model, self.fallback, self.execute = model, fallback, execute
        self.start_s = float(start_s)
        self.step_index = 0
        self.controllers = None
        self.held = self.target = self.mask = self.identities = None
        self.running = True

    def stop(self, *, now_s):
        for controller in self.controllers or ():
            controller.stop(now_s=now_s)
        self.running = False
        self.held = self.target = self.mask = self.identities = None

    def _decide(self, state, now):
        shape = state['s_m'].shape
        if self.controllers is None:
            self.controllers = [make_switchable_wall() for _ in range(shape[0])]
        if len(self.controllers) != shape[0] or (self.held is not None and self.held.shape != shape):
            raise ValueError("wall batch/slot layout changed; create a fresh rollout controller")
        keys = ('s_m', 'v_mps', 'lane', 'present', 'is_cav')
        arrays = {key: _numpy(state[key]) for key in keys}
        ids = _numpy(state.get('vid_index', torch.arange(shape[1], device=state['s_m'].device)
                               .expand(shape)))
        junction = _numpy(state.get('junction', torch.zeros_like(state['present'])))
        self.held = torch.zeros_like(state['s_m'])
        self.target = torch.zeros_like(state['s_m'])
        self.mask = torch.zeros_like(state['present'], dtype=torch.bool)
        self.identities = torch.as_tensor(ids.copy(), device=state['s_m'].device)
        for b, controller in enumerate(self.controllers):
            frame = SimpleNamespace(s=arrays['s_m'][b], v=arrays['v_mps'][b],
                lane=arrays['lane'][b], lane_carry=arrays['lane'][b], junction=junction[b],
                present=arrays['present'][b], is_cav=arrays['is_cav'][b], vid_index=ids[b])
            registry = {int(i): str(int(i)) for i in ids[b]}
            builder = SimpleNamespace(vehicle_ids=registry)
            live = {registry[int(ids[b, k])] for k in range(shape[1])
                    if arrays['present'][b, k] and arrays['is_cav'][b, k]}
            commands = controller.commands(frame, builder, now_s=now, live_cavs=live)
            for k in range(shape[1]):
                vid = registry[int(ids[b, k])]
                if arrays['present'][b, k] and arrays['is_cav'][b, k] and vid in commands:
                    self.held[b, k] = commands[vid]
                    self.target[b, k] = state['v_mps'][b, k] + commands[vid]
                    self.mask[b, k] = True

    @torch.no_grad()
    def __call__(self, state):
        now = self.start_s + self.step_index * 0.5
        base = self.fallback(state)
        if self.running and self.step_index % 2 == 0:
            self._decide(state, now)
        self.step_index += 1
        if not self.running:
            return base
        mask = self.mask & state['present'].bool() & state['is_cav'].bool()
        if 'vid_index' in state:
            mask &= state['vid_index'] == self.identities
        if self.execute is not None:
            return self.execute(state, self.held, self.target, mask, base)
        # Same native-safe minimum and held absolute-target bound as TimelineControllers.
        request = torch.minimum(self.held, (self.target - state['v_mps']) / 0.5)
        executed = torch.minimum(self.model.physics_accel(state), request)
        speed = (state['v_mps'] + executed * 0.5).clamp_min(0)
        position = state['s_m'] + 0.25 * (state['v_mps'] + speed)
        wall = torch.stack((position, base[..., 1], speed, executed), dim=-1)
        return torch.where(mask[..., None], wall, base)
