"""Sustained set-points injected at the original executor's pre-safety boundary."""
from contextlib import contextmanager
from .policy import encode, decode, requests


class Controller:
    def __init__(self, policy=None, generator=None, sample=None, start=None, members=None,
                 deterministic=False, capture=None):
        self.policy, self.generator = policy, generator
        self.sample, self.start, self.members = sample, start, members
        self.deterministic, self.capture = deterministic, capture
        self.targets = decode(sample) if sample is not None else {}
        self.until = start+sample['window'] if sample is not None else None
        self.fixed = sample is not None
        self.events = []

    def apply(self, rows, commands, now):
        context = encode(rows,commands)
        if self.capture is not None:
            self.capture(context)
        if self.fixed:
            active = self.start <= now < self.until
        else:
            if self.until is None or now >= self.until:
                self.targets = {}
                if context['members'] and self.policy is not None:
                    self.sample = self.policy.sample(context,self.generator,deterministic=self.deterministic)
                    self.targets = decode(self.sample)
                    self.until = now+self.sample['window']
                    self.events.append(dict(time_s=now,sample=self.sample))
                else:
                    self.until = None
            active = bool(self.targets)
        selected = self.targets
        if self.members is not None:
            selected = {v:t for v,t in selected.items() if v in self.members}
        return requests(selected,rows) if active else {}


@contextmanager
def inject(executor, env, rows, controller, now, audit):
    # Mirrors the tested LEVERPROBE pre_safety boundary, not SUMO direct commands.
    from wms.control import safety
    original = safety.safe_accel
    ids = env.slot_vehicle_ids()
    calls = []
    def apply(state, accel, **kwargs):
        commands = {vid:float(accel[0,i]) for i,vid in enumerate(ids)
                    if vid in {r['vid'] for r in rows if r['is_cav']}}
        desired = controller.apply(rows,commands,now)
        raw = accel.clone()
        for i,vid in enumerate(ids):
            if vid in desired:
                raw[0,i] = min(executor.accel_max,max(executor.accel_min,desired[vid]['requested_raw']))
        safe = original(state,raw,**kwargs)
        baseline = original(state,accel,**kwargs) if desired else safe
        # Receipt-only same-state counterfactual at the existing safety boundary.
        # It does not command vehicles or advance physics. Absent metadata keeps
        # the original command path and number of safety calls unchanged.
        move_rows = {v:r for v,r in desired.items() if 'without_library_raw' in r}
        without_move = None
        if move_rows:
            counterfactual = raw.clone()
            for i,vid in enumerate(ids):
                if vid in move_rows:
                    counterfactual[0,i] = min(executor.accel_max,max(executor.accel_min,
                        move_rows[vid]['without_library_raw']))
            without_move = original(state,counterfactual,**kwargs)
        for i,vid in enumerate(ids):
            if vid in desired:
                audit.append(dict(vid=vid,**desired[vid],mappo_raw=float(accel[0,i]),
                    mappo_safe=float(baseline[0,i]),issued_raw=float(raw[0,i]),
                    issued_safe=float(safe[0,i]),clamped=bool(raw[0,i] != safe[0,i])))
                if vid in move_rows:
                    audit[-1].update(without_library_issued_safe=float(without_move[0,i]),
                        library_issued_delta_mps2=float(safe[0,i]-without_move[0,i]))
        calls.append(True)
        return safe
    safety.safe_accel = apply
    try:
        yield
        if len(calls) != 1:
            raise ValueError('expected one native-lane safety boundary')
    finally:
        safety.safe_accel = original
