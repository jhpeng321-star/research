"""Tensor mirror of the plant's post-safety CommandOwnership/slowDown path.

This does not replace SUMO's speedMode27 ceiling: the inherited callback still
applies ``inferred_cap`` after this stage on each 0.5 s physics step. Constants
are the frozen direct interface (1 s control, -3/3 acceleration), not fitted
behavioural cutoffs. Numerical activation requires a source-bound qualification.
"""


def ownership_target(speed, safe_command, allowed_speed, *, lo=-3., hi=3., dt=1.):
    """Same float64 scalar arithmetic/order as damping.direct_target.

    Keep the issued acceleration AND target speed in float64 until the physics
    request is formed. The original plant's scalar sensor frame is float64.
    """
    import torch
    speed=speed.to(torch.float64)
    command=safe_command.to(torch.float64)
    allowed=allowed_speed.to(device=speed.device,dtype=torch.float64)
    lower=torch.maximum(torch.full_like(speed,float(lo)),-speed/float(dt))
    upper=torch.minimum(torch.full_like(speed,float(hi)),(allowed-speed)/float(dt))
    feasible=torch.isfinite(command) & (lower<=upper)
    issued=torch.minimum(torch.maximum(command,lower),upper)
    target=(speed+float(dt)*issued).clamp_min(0.)
    return issued,target,feasible


class SlowDown:
    """One current-speed -> target-speed timeline per one-second command.

    The odd half-step follows the same timeline, rather than reapplying the
    safety filter to a new state or pretending slowDown is setAcceleration.
    Newly entered CAVs on an odd tick have no command until the next decision;
    the caller supplies their native/SUMO ceiling as fallback.
    """
    def __init__(self):
        self.initial=self.target=self.commanded=None

    def apply(self,state,safe_command,tick,allowed_speed,native):
        import torch
        speed=state['v_mps']
        active=state['present'].bool() & state['is_cav'].bool() & (state['lane']>=0)
        if tick%2==0:
            _,self.target,feasible=ownership_target(speed,safe_command,allowed_speed)
            self.initial=speed.to(torch.float64)
            self.commanded=active & feasible
        if self.initial is None or self.initial.shape!=speed.shape:
            raise ValueError('slowDown requires a decision before a physics continuation')
        elapsed=.5*(tick%2+1)
        timeline=self.initial+(self.target-self.initial)*elapsed
        request=((timeline-speed.to(torch.float64))/.5).to(speed.dtype)
        return torch.where(active & self.commanded,request,native)
