"""Frozen motion-only driver inference and explicit IDM-to-SUMO mapping."""
from pathlib import Path
import hashlib

SPEED_LIMIT = 31.29
MEDIANS = dict(accel=1.2, decel=2.0, tau=1.2, minGap=2.0,
               speedFactor=1.0, speedDev=0.0, delta=4.0,
               emergencyDecel=9.0, lcStrategic=1.0, lcCooperative=.65,
               lcSpeedGain=1.0, lcAssertive=1.0, impatience=.3)


def sumo_attributes(values=None):
    """Do not reinterpret MOBIL/merge parameters as LC2013 parameters."""
    out = dict(MEDIANS)
    if values is not None:
        out.update(speedFactor=float(values['v0']) / SPEED_LIMIT,
                   tau=float(values['time_headway']), minGap=float(values['min_gap']),
                   accel=float(values['accel']), decel=float(values['decel']))
    return out


def infer(start, raw, checkpoint):
    import torch
    from traffic_shepherd.c4.wms_glue import driver_history_tensors
    from wms.model.checkpoint import load_checkpoint
    loaded = load_checkpoint(checkpoint, device='cpu')
    model = loaded.model if hasattr(loaded, 'model') else loaded
    model.eval()
    duration = getattr(model, 'history_s', None)
    if duration is None and getattr(model, 'long_history', False):
        raise ValueError('long-history checkpoint without duration')
    # The native long-history encoder uses the authentic last 60 seconds. A
    # legacy checkpoint keeps its trained rolling-window contract, not padding.
    h, mask = driver_history_tensors(raw, history_s=duration,
        history_stride_s=getattr(model, 'history_stride_s', 1.0),
        history_len=getattr(model, 'history_len', 10))
    with torch.inference_mode():
        encoded, _, params = model.encode_drivers(start['state'],
            driver_history=h, driver_history_mask=mask)
    if params is None:
        raise ValueError('checkpoint does not expose inferred physical parameters')
    slots = raw['vid_index'][-1]
    result = {}
    for k, active in enumerate(start['state']['present'][0].tolist()):
        if active:
            vid = start['vehicle_ids'][int(slots[k])]
            values = {name: float(v[0, k]) for name, v in params.items()}
            result[vid] = sumo_attributes(values)
    return result, dict(checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                        history_s=duration, history_samples=int(h.shape[-2]),
                        per_vehicle_observed_samples=mask.sum(-1)[0].tolist(),
                        mapping='v0/31.29->speedFactor; headway->tau; gap->minGap; accel/decel identity; LC population medians')
