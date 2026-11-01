"""Source-physical support constraints for motion-only IDM inference (opt-in).

SUMO maxNextSpeed/finalizeSpeed never accelerates an uncontrolled IDM driver by
more than its own accel*dt. Representation uncertainty is bounded by adjacent
float32 values, matching the deployed physical settings/state representation.
No behavioral threshold, acceleration residual, fitting or RNG is introduced.
"""
import torch


def constrain_acceleration(weights, support, speed_before, speed_after, dt_s):
    if (weights.ndim != 1 or support.shape != (len(weights), 5)
            or speed_before.ndim != 1 or speed_after.shape != speed_before.shape
            or dt_s.shape != speed_before.shape
            or any(not torch.isfinite(value).all() for value in
                   (weights, support, speed_before, speed_after, dt_s))
            or (weights < 0).any() or weights.sum() <= 0 or (dt_s <= 0).any()
            or (speed_before < 0).any() or (speed_after < 0).any()):
        raise ValueError('invalid physical acceleration-support inputs')
    if not len(speed_before):
        return weights, dict(acceleration_envelope=False, acceleration_evidence=0,
                             discarded_weight=0.)
    before = speed_before.to(torch.float32)
    after = speed_after.to(torch.float32)
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError('observed speed exceeds deployed float32 representation')
    lower = ((torch.nextafter(after, torch.full_like(after, -torch.inf)).double()
              - torch.nextafter(before, torch.full_like(before, torch.inf)).double())
             / dt_s.double()).max()
    accel = support[:, 2].to(torch.float32)
    upper = torch.nextafter(accel, torch.full_like(accel, torch.inf)).double()
    keep = upper >= lower
    result = torch.where(keep, weights, 0.)
    if result.sum() <= 0:
        raise ValueError('no positive-weight physical profile satisfies observed acceleration envelope')
    discarded = float(weights[~keep].sum()/weights.sum())
    return result/result.sum(), dict(acceleration_envelope=True,
        acceleration_evidence=len(speed_before), acceleration_lower_mps2=float(lower),
        discarded_weight=discarded, remaining_profiles=int((result > 0).sum()),
        representation_bound='outward adjacent float32 values')
