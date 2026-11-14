"""Deployment-honesty: the model takes ONLY frozen FORMAT.md tensors."""
from __future__ import annotations

import inspect

import pytest

from model.net import WmN1Net
from synth_model import make_batch, small_cfg

ALLOWED = {"self", "veh_step", "veh_mask", "plan", "wall", "wall_mask",
           "counter"}
FORBIDDEN = {"seed", "t0", "t0_s", "type_id", "vtype", "speed_mode",
             "pre_fix", "world", "penetration", "driver_type", "time_s"}


def test_forward_signature_only_accepts_format_tensors():
    params = set(inspect.signature(WmN1Net.forward).parameters)
    assert params <= ALLOWED, params - ALLOWED
    assert not (params & FORBIDDEN)


def test_forward_rejects_forbidden_kwargs():
    net = WmN1Net(small_cfg()).eval()
    b = make_batch(B=1, K=6)
    with pytest.raises(TypeError):
        net(**b, t0=300.0)
    with pytest.raises(TypeError):
        net(**b, seed=3)
    with pytest.raises(TypeError):
        net(**b, type_id=1)


def test_forward_consumes_all_six_tensors():
    net = WmN1Net(small_cfg()).eval()
    b = make_batch(B=1, K=6)
    assert set(b) == ALLOWED - {"self"}
    out = net(**b)
    assert "outcome_quantiles" in out
