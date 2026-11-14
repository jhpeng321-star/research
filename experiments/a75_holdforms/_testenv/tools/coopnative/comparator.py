"""Evaluation-only frozen comparator paths. Never constructed for train/learned/native."""
from contextlib import contextmanager
import sys
import numpy as np
import torch
from .protocol import checked, sha

MAPPO_SHA = 'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'
CTRL0_SHA = '64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c'
ZERO_SHA = '24095a1092eafb4c894804aa8142478a802eaea046142ec777e5f24f49f73596'


class Comparator:
    def __init__(self, arm, binding):
        if arm not in ('mappo','ctrl0'):
            raise ValueError('comparator-only constructor')
        self.arm, self.previous = arm, None
        checkpoint = binding[arm]
        if sha(checkpoint) != (MAPPO_SHA if arm=='mappo' else CTRL0_SHA):
            raise ValueError('comparator checkpoint mismatch')
        from wms.control.policy import load_policy_fn
        self.exe = load_policy_fn(checkpoint,device='cpu')
        self.exe.reset()
        if arm == 'ctrl0':
            zero = checked(binding['zero'])
            if sha(zero) != ZERO_SHA:
                raise ValueError('CTRL0 wrapper mismatch')
            sys.path.insert(0,str(zero.parent))

    @contextmanager
    def scope(self):
        if self.arm != 'ctrl0':
            yield
            return
        from zero import forced_base_commands
        from wms.control import nativeinit
        original = nativeinit.sumo_residual_commands
        nativeinit.sumo_residual_commands = forced_base_commands
        try:
            yield
        finally:
            nativeinit.sumo_residual_commands = original

    def step(self,env):
        ids = np.asarray(env.frame.vid_index)
        keep = np.zeros(len(ids),bool) if self.previous is None else (ids == self.previous) & (ids >= 0)
        self.previous = ids.copy()
        for obj in (self.exe.memory,self.exe.dwell):
            if obj.present is not None:
                obj.present &= torch.as_tensor(keep,device=obj.present.device)[None]
        if self.arm == 'ctrl0':
            from zero import ZeroResidual
            cmd = ZeroResidual(self.exe)(env.state())
        else:
            cmd,lanes = self.exe(env.state(),with_lanes=True)
            if lanes is not None:
                raise ValueError('native lanes required')
        with self.scope():
            _,info = env.step(cmd,lanes=None,fallback='native')
        return info,[]
