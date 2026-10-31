"""Rebuild an inference architecture while preserving the loaded motion encoder.

The original checkpoint is loaded strictly by the normal loader first. This
adapter permits removal of unused correction heads, but never silently replaces
an encoder weight or changes the encoder's input/parameter semantics.
"""
from __future__ import annotations

import torch

from wms.model.factory import build_model


ENCODER_FIELDS = ("history_features", "history_len", "history_hidden", "history_s",
                  "history_stride_s", "driver_param_mode")


def rebuild_with_motion_encoder(loaded_model, spec, physics):
    if loaded_model.black_box or spec.black_box:
        raise ValueError("WM-PHYS requires a structured motion encoder")
    if spec.rollout_mode == "residual":
        raise ValueError("rebuilt settings must enter the physical driver parameters")
    for name in ENCODER_FIELDS:
        if getattr(loaded_model.spec, name) != getattr(spec, name):
            raise ValueError(f"rebuild changes motion encoder semantics: {name}")
    old = loaded_model.state_dict()
    encoder_keys = {name for name in old if name.startswith("driver_encoder.")}
    if not encoder_keys:
        raise ValueError("loaded checkpoint has no motion encoder")
    # Initialization of inactive heads must not perturb the caller's future RNG.
    with torch.random.fork_rng(devices=[]):
        candidate = build_model(spec, physics)
    new = candidate.state_dict()
    expected_encoder = {name for name in new if name.startswith("driver_encoder.")}
    if expected_encoder != encoder_keys:
        raise ValueError("rebuild changes motion encoder key set")
    incompatible = [name for name in encoder_keys if new[name].shape != old[name].shape]
    if incompatible:
        raise ValueError(f"rebuild changes motion encoder tensor shapes: {incompatible}")
    matched = {name: value for name, value in old.items()
               if name in new and new[name].shape == value.shape}
    candidate.load_state_dict(matched, strict=False)
    for name in encoder_keys:
        if not torch.equal(candidate.state_dict()[name].cpu(), old[name].cpu()):
            raise AssertionError(f"motion encoder copy was not exact: {name}")
    # The source model is untouched; callers keep WM2 and WM-PHYS side by side.
    device = next(loaded_model.parameters()).device
    candidate.to(device).eval()
    receipt = dict(encoder_exact=True, encoder_tensors=len(encoder_keys),
                   copied_tensors=len(matched),
                   unused_source_tensors=sorted(set(old) - set(matched)),
                   initialized_inactive_tensors=sorted(set(new) - set(matched)))
    return candidate, receipt
