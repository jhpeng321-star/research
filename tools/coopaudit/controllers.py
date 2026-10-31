"""Frozen controller wiring, COOPMEAS2 identity memory and contextual wall masking."""
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np


def observations(env):
    f = env.frame
    return [dict(vid=env.builder.vehicle_ids[int(f.vid_index[s])], s=float(f.s[s]),
                 lane=int(f.lane[s]), junction=bool(f.junction[s]), is_cav=bool(f.is_cav[s]),
                 v=float(f.v[s]) if hasattr(f, 'v') else float(env.history.raw(1)['kin'][-1,s,4]))
            for s in np.flatnonzero(f.present)]


def wall_roster(rows, group):
    """Closest cross-mainline-lane pair; no speed, outcome or type criterion."""
    eligible = [r for r in rows if r['vid'] in group and r['lane'] in (0,1)]
    pairs = [(a,b) for i,a in enumerate(eligible) for b in eligible[i+1:]
             if a['lane'] != b['lane'] and abs(a['s']-b['s']) <= 75.]
    if not pairs:
        return []
    best = min(pairs, key=lambda ab: (max(abs(r['s']) for r in ab),
                                    abs(ab[0]['s']-ab[1]['s']), sorted(r['vid'] for r in ab)))
    return sorted(r['vid'] for r in best)


class WallControl:
    def __init__(self, roster, speed=12.):
        from traffic_shepherd.wall_group import WallGroupSession
        self.session = WallGroupSession('audit-wall', members=frozenset(roster), wall_speed_mps=speed) if len(roster)>=2 else None
        self.issued_ticks = 0

    def commands(self, rows, now, enabled):
        if self.session is None or not enabled:
            return {}
        frames = {r['vid']: SimpleNamespace(speed_mps=r['v'], distance_to_merge_m=-r['s'],
                                           lane_index=r['lane']) for r in rows}
        # Full frozen context is evaluated even in ONLY:i. Disable command, not membership.
        commands = self.session.tick(frames, now, control_dt_s=1.)
        out = {v:a for v,a in commands.items() if v in enabled}
        self.issued_ticks += bool(out)
        return out

    @property
    def events(self):
        return self.session.events if self.session else []


class BoundController:
    """COOPMEAS2 runner lines118-147, native-lane-only call retained explicitly."""
    def __init__(self, exe, name):
        from traffic_shepherd.c4 import branching as B
        self.exe, self.controller, self.previous = exe, B.BranchController(exe,name=name), None
        self.policy = lambda state: exe(state)  # exact speed-only MAPPO

    def command(self, env, members):
        import torch
        from traffic_shepherd.c4 import branching as B
        if members == frozenset():
            return None, {}
        ids = np.asarray(env.frame.vid_index)
        keep = np.zeros(len(ids),bool) if self.previous is None else (ids==self.previous)&(ids>=0)
        self.previous = ids.copy()
        memory, dwell = self.exe.memory, self.exe.dwell
        for obj in (memory,dwell):
            if obj.present is not None:
                obj.present = obj.present & torch.as_tensor(keep,device=obj.present.device)[None]
        accel,kw = self.controller.command(env,env.state(),members,self.policy)
        mask = torch.as_tensor(B.member_mask(env,members))[None]
        if memory.accel is not None:
            mask = mask.to(memory.accel.device)
            memory.accel = torch.where(mask,memory.accel,torch.zeros_like(memory.accel))
            memory.intent = torch.where(mask,memory.intent,torch.zeros_like(memory.intent))
            memory.present = memory.present & mask
        if kw.get('lanes') is not None:
            raise ValueError('controller attempted lane control')
        return accel,kw


@contextmanager
def ctrl0_scope():
    """Exact zero.py forced_base_commands; NaN denotes disabled, never forced-native."""
    from tools.ctrlarm.zero import forced_base_commands
    from wms.control import nativeinit
    original = nativeinit.sumo_residual_commands
    def masked(api, residuals, lo, hi, **kwargs):
        return forced_base_commands(api, {v:r for v,r in residuals.items() if np.isfinite(r)},
                                    lo, hi, dt=kwargs.get('dt', .5))
    nativeinit.sumo_residual_commands = masked
    try:
        yield
    finally:
        nativeinit.sumo_residual_commands = original


class Controller:
    def __init__(self, name, inputs, roster):
        self.name, self.wall, self.bound, self.zero = name, None, None, None
        if name == 'wall':
            self.wall = WallControl(roster)
        elif name in ('mappo','ctrl0'):
            from wms.control.policy import load_policy_fn
            exe = load_policy_fn(inputs[name], device='cpu')
            exe.reset()
            if name == 'mappo':
                self.bound = BoundController(exe, name)
            else:
                from tools.ctrlarm.zero import ZeroResidual
                self.zero = ZeroResidual(exe)

    def command(self, env, members):
        from traffic_shepherd.c4 import branching as B
        if not members:
            return None, {}, {}
        if self.name in ('pi','circles'):
            return None, dict(fallback='pi' if self.name=='pi' else 'damping', fallback_ids=members), {}
        if self.bound:
            a,kw = self.bound.command(env,members)
            return a,kw,{}
        if self.zero:
            from wms.control.nativeinit import NativeResidualCommand
            command = self.zero(env.state())
            residual = command.residual.copy()
            residual[0,~B.member_mask(env,members)] = np.nan
            return NativeResidualCommand(residual,command.accel_min,command.accel_max),{},{}
        commands = self.wall.commands(observations(env),env.now,members)
        array = np.full((1,env.k),np.nan)
        for i in np.flatnonzero(env.frame.present & env.frame.is_cav):
            v = env.builder.vehicle_ids[int(env.frame.vid_index[i])]
            if v in commands:
                array[0,i] = commands[v]
        return array,{},commands
