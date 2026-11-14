"""COOPFREE tracking + original safety; sole actuator remains env.step/CommandOwnership."""
import torch
from tools.coopfree.policy import requests
from .policy import targets


class Controller:
    def __init__(self):
        self.targets, self.until = {}, None

    def hold(self, sample, now):
        self.targets = targets(sample)
        self.until = now+sample['window']

    def command(self, env, rows, now):
        from wms.control import safety
        desired = requests(self.targets,rows,zone=None) if self.until is not None and now < self.until else {}
        ids = env.slot_vehicle_ids()
        raw = torch.zeros((1,len(ids)),dtype=torch.float32)
        active = torch.zeros_like(raw,dtype=torch.bool)
        for i, vid in enumerate(ids):
            if vid in desired:
                raw[0,i] = min(1.5,max(-3.,desired[vid]['requested_raw']))
                active[0,i] = True
        # Preserve the exact original filter. Inactive slots are released AFTER safety.
        safe = safety.safe_accel(env.state(),raw,accel_max=1.5)
        command = torch.where(active,safe,torch.full_like(safe,float('nan')))
        audit = [dict(vid=vid,**desired[vid],issued_safe=float(safe[0,i]),
                      clamped=bool(raw[0,i] != safe[0,i])) for i,vid in enumerate(ids) if vid in desired]
        return command, audit

    def step(self, env, rows, now):
        command, audit = self.command(env,rows,now)
        _, info = env.step(command,lanes=None,fallback='native')
        return info, audit
