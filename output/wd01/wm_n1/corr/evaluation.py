"""Inference wiring for the trained P5 composition; no learned layers created.

The trajectory contract includes t0 and uses original-vehicle survival, as
the trainer's canonical batch does. This module never reads corpus files.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch

from .head import CorrectionHead
from .summary import CavMotionSummary, cav_motion_summary


def channel(obj, name):
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def module_data(module, value, *, boolean=False):
    """Move data only to the module's floating parameter dtype/device."""
    ref = next((p for p in module.parameters() if p.is_floating_point()), None)
    if ref is None:
        raise ValueError("inference module has no floating parameters")
    return torch.as_tensor(value, device=ref.device).to(
        dtype=torch.bool if boolean else ref.dtype)


def trained_correction(bundle):
    """Require the existing raw16 -> trained projection -> trained head path."""
    corr = channel(bundle, "corr")
    if not (isinstance(getattr(corr, "proj", None), CavMotionSummary)
            and isinstance(getattr(corr, "head", None), CorrectionHead)
            and callable(corr)):
        raise ValueError(
            "correction requires the trained raw16 projection/head composition "
            "(RealCorrBundle); missing or unsupported P5 wiring")
    return corr


@torch.no_grad()
def trajectory_summary(pos, spd, acc, masks, *, t0_kin, t0_mask,
                       t0_is_cav, t0_slot_id, slot_ids):
    """Raw16 for one sample: [T+1,B,K] states and [T,B,K] acceleration.

    Identity must be carried explicitly. A slot that disappears or changes
    identity never becomes alive again, even if its original ID reappears.
    Departed states are forward-filled before the frozen summary is called.
    """
    required = dict(t0_kin=t0_kin, t0_mask=t0_mask, t0_is_cav=t0_is_cav,
                    t0_slot_id=t0_slot_id, slot_ids=slot_ids, masks=masks,
                    accel=acc)
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"correction requires explicit channels: {missing}")
    pos = torch.as_tensor(pos)
    device = pos.device
    spd, acc, kin = (torch.as_tensor(x, device=device)
                     for x in (spd, acc, t0_kin))
    mask, initial, cav = (torch.as_tensor(x, device=device, dtype=torch.bool)
                          for x in (masks, t0_mask, t0_is_cav))
    ids, ids0 = (torch.as_tensor(x, device=device) for x in (slot_ids, t0_slot_id))
    if pos.ndim != 3 or pos.shape[0] < 1 or spd.shape != pos.shape:
        raise ValueError("positions/speeds must be equal [T+1,B,K] tensors")
    if acc.shape != pos[1:].shape or mask.shape != pos.shape or ids.shape != pos.shape:
        raise ValueError("accel must be [T,B,K]; masks/slot_ids must be [T+1,B,K]")
    shape = pos.shape[1:]
    if (kin.shape != (*shape, 7) or initial.shape != shape
            or cav.shape != shape or ids0.shape != shape):
        raise ValueError("t0 channels must match [B,K] trajectory slots (kin: 7 fields)")
    if ids.is_floating_point() or ids0.is_floating_point():
        raise ValueError("slot identities must be integers")
    for b in range(shape[0]):
        occupied_ids = ids0[b, initial[b]]
        if bool((occupied_ids < 0).any()) or occupied_ids.unique().numel() != occupied_ids.numel():
            raise ValueError("occupied t0 slots require unique nonnegative vehicle identities")
    if not torch.equal(mask[0], initial) or not torch.equal(ids[0], ids0):
        raise ValueError("trajectory t0 mask/slot identity differs from original t0")
    if not (torch.equal(pos[0].to(kin.dtype), kin[..., 0])
            and torch.equal(spd[0].to(kin.dtype), kin[..., 4])):
        raise ValueError("trajectory must include the original t0 position/speed")
    alive = torch.cumprod((mask & (ids == ids0.unsqueeze(0))
                          & initial.unsqueeze(0)).to(torch.int64), dim=0).bool()
    time = torch.arange(pos.shape[0], device=device).view(-1, 1, 1)
    fill = torch.cummax(torch.where(alive, time, torch.zeros_like(time)), dim=0).values
    acc_states = torch.cat([kin[..., 5].unsqueeze(0).to(acc.dtype), acc], dim=0)
    return cav_motion_summary(pos.gather(0, fill), spd.gather(0, fill),
                              acc_states.gather(0, fill),
                              cav.unsqueeze(0) & alive)


@torch.no_grad()
def state_token(bundle, reference, *, kin=None, mask=None, counter=None,
                d_model=None):
    """A6 token on encoder placement, then transferred to correction input."""
    enc = channel(bundle, "state_enc")
    cfg = channel(bundle, "config") or channel(bundle, "cfg")
    flag = channel(cfg, "state_global_token")
    # An explicit OFF config wins even if a stale encoder is attached, just
    # like the trainer. Legacy bundles without a flag retain encoder-presence
    # inference; real checkpoints carry the explicit configuration.
    enabled = bool(flag) if flag is not None else enc is not None
    width = int(d_model if d_model is not None else reference.shape[-1])
    if not enabled:
        return reference.new_zeros((reference.shape[0], width))
    if enc is None:
        raise ValueError("state_global_token is ON but no state encoder is present")
    if kin is None or mask is None or counter is None:
        raise ValueError("state_global_token is ON: explicit t0 kin/mask/counter required")
    kin = module_data(enc, kin)
    mask = module_data(enc, mask, boolean=True)
    counter = module_data(enc, counter)
    expected_counter = (reference.shape[0], int(getattr(enc, "counter_ch", 2)),
                        int(getattr(enc, "counter_bins", 30)))
    if tuple(counter.shape) != expected_counter:
        raise ValueError(f"t0 counter must be {expected_counter}, got {tuple(counter.shape)}")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("state_global_token is ON but a world has no occupied t0 slots")
    token = enc(kin, mask, counter)
    if token.shape != (reference.shape[0], width):
        raise ValueError(f"state encoder emits {tuple(token.shape)}, expected "
                         f"{(reference.shape[0], width)}")
    return token.to(device=reference.device, dtype=reference.dtype)
