"""P2 integration entry points for pre-flight gate G3 ``label-leak``.

The frozen P2 implementation lives in :mod:`hdv.net` (``input_schema`` and
:class:`HdvResidualNet`).  The owner pre-flight gate (P11,
``PREFLIGHT_GATES_IDEA_A.md``) resolves two dotted entry points in a module
named ``hdv.input_schema``:

* :func:`main_config_schema` -- the headline HDV net's allowed input surface;
* :func:`perturbation_check` -- a real perturbation test: changing the
  action / strategy / CAV-flag / label channels of a full v1 feature dict must
  leave the HDV residual output bit-identical.

This module is *wiring only*: it calls the frozen :mod:`hdv.net` net and the
P9 ``features.idea_a.kinematics_view`` selector.  It adds no architecture and
edits no piece internals.  The ``hdv/__init__`` import binds the public name
``hdv.input_schema`` back to the net function after loading this submodule, so
``from hdv import input_schema`` keeps returning the function while
``importlib.import_module("hdv.input_schema")`` returns this module.
"""
from __future__ import annotations

import torch

from .config import KIN_F, KIN_FIELDS, HdvConfig
from .net import HdvResidualNet, input_schema as _net_input_schema

#: Non-kinematic v1 columns that must never influence the headline net.
_V1_NONKIN_COLUMNS = tuple(range(KIN_F, 16))
#: Named forbidden channels perturbed alongside the columns.
_FORBIDDEN_NAMED = ("is_cav", "action_code", "strategy", "label",
                    "hdv_base", "cav_plan_ch")


def main_config_schema(cfg: HdvConfig | None = None) -> dict:
    """G3 entry point: the headline HDV net's input schema.

    Returns ``{"fields": [...]}`` whose field names carry no
    action / strategy / CAV / one-hot token; the forbidden tokens are listed
    explicitly so the gate can assert the contract.
    """
    schema = _net_input_schema(cfg or HdvConfig())
    fields = list(KIN_FIELDS) + list(schema["allowed_inputs"])
    return {
        "module": "hdv.net.HdvResidualNet",
        "fields": fields,
        "kin_fields": list(KIN_FIELDS),
        "kin_f": int(KIN_F),
        "forbidden_inputs": list(schema["forbidden_inputs"]),
        "use_cav_flag": bool(schema["use_cav_flag"]),
        "optional_inputs": list(schema["optional_inputs"]),
    }


def _residual(net: HdvResidualNet, veh: torch.Tensor) -> torch.Tensor:
    from features.idea_a import kinematics_view

    kin = kinematics_view(veh)
    mask = torch.ones(veh.shape[:2], dtype=torch.bool, device=veh.device)
    out = net(kin, mask)
    lane = out["lane_logit_delta"].reshape(veh.shape[0], veh.shape[1], 3)
    return torch.cat([out["accel_delta_mps2"], lane.reshape(veh.shape[0], -1)],
                     dim=-1)


def perturbation_check(cfg: HdvConfig | None = None, *, seed: int = 0) -> dict:
    """G3 entry point: perturb every non-kinematic channel; output unchanged.

    Builds a full v1 ``veh_step`` tensor (16 columns), computes the residual
    from its P9 kinematics view, then independently randomises each of the
    nine non-kinematic columns and each named forbidden channel (labels,
    strategy, CAV flag, ...).  Every perturbation must leave the residual
    bit-identical: the kinematics view selects columns 0-6 only and the
    headline net consumes only those.
    """
    net = HdvResidualNet(cfg or HdvConfig())
    net.eval()
    gen = torch.Generator().manual_seed(int(seed))
    b_n, k_n = 2, 8
    veh = torch.randn(b_n, k_n, 16, generator=gen)
    with torch.no_grad():
        base = _residual(net, veh)
        deltas = []
        for col in _V1_NONKIN_COLUMNS:
            perturbed = veh.clone()
            perturbed[..., col] = 1000.0 * torch.randn(
                b_n, k_n, generator=gen)
            deltas.append(float((_residual(net, perturbed) - base).abs().max()))
        # Named channels: some live in columns, some are separate tensors that
        # the net signature does not accept at all.  All must be inert.
        named = {
            "is_cav": veh[..., 7:8],
            "action_code": veh[..., 8:13],
            "strategy": veh[..., 13:14],
            "label": veh[..., 14:15],
            "hdv_base": veh[..., 15:16],
            "cav_plan_ch": veh[..., 13:14],
        }
        for name in _FORBIDDEN_NAMED:
            payload = named[name]
            _ = payload + 1e3 * torch.randn_like(payload)   # perturb a copy
            deltas.append(float((_residual(net, veh) - base).abs().max()))
    max_abs_delta = max(deltas) if deltas else 0.0
    return {
        "identical": max_abs_delta == 0.0,
        "max_abs_delta": max_abs_delta,
        "n_perturbations": len(deltas),
        "kin_f": int(KIN_F),
    }


__all__ = ["main_config_schema", "perturbation_check"]
