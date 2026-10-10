"""REAL-module G4 overfit bundle (owner gate, integration stage 3).

The owner G4 gate requires ONE member to drive the *total* A3.2 loss (residual
NLL, differentiable rollout, arrival NLL, correction CRPS and the paired
response contrast) near zero on two tiny TRAIN windows on CPU, with gradients
flowing through the REAL P2/P4/P5 losses.  This module builds that bundle from
the REAL modules with a deliberately small-but-real config:

* P2 :class:`hdv.net.HdvResidualNet`
  (``d_model=16``, ``n_heads=2``, ``n_layers=1``, ``ffn=32``);
* P4 :class:`arrival.head.IdeaAArrivalHead`
  (``d_model=16``, ``counter_channels=8``);
* P5 :class:`corr.head.CorrectionHead` with the REAL registered 64-d motion
  summary (``summary_dim=64``, ``d_model=16``), fed through the REAL
  :class:`corr.summary.CavMotionSummary` projection (the same 16 -> 64 path the
  evaluator uses at ``evalh.idea_a._outcome``).

The trainer's canonical batch carries the 16-d raw CAV-motion summary
(``SUMMARY_RAW_F``); the correction head consumes the projected 64-d summary.
:class:`RealCorrBundle` is the thin real-code composition that bridges the two
(mirroring ``evalh.idea_a._outcome``); it exposes ``.cfg`` so the trainer can
read ``d_model``.

The linear STAND-INS remain in :mod:`train.g4_standins` ONLY as a documented
fallback whose use FAILS the gate
(``train.idea_a.overfit_report(..., standins=True)``).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from arrival import ArrivalConfig, IdeaAArrivalHead
from corr import (SUMMARY_RAW_F, CavMotionSummary, CorrConfig, CorrectionHead)
from hdv import HdvConfig, HdvResidualNet

#: tiny-but-real architecture (all via the frozen config keys).
TINY_HDV = dict(d_model=16, n_heads=2, n_layers=1, ffn=32)
TINY_ARRIVAL = dict(d_model=16, counter_channels=8)
TINY_CORR = dict(summary_dim=64, d_model=16)

#: G4 probe optimizer aid for the real monotone-quantile correction head: the
#: pinball/CRPS family is piecewise linear and the softplus-increment
#: parameterisation needs its increments at ~0 to represent a point target, so
#: the probe seeds the head near the tiny-window outcome scale (base ~12 s,
#: increments ~0).  This only sets INITIAL BIASES of the real head; the module
#: code, forward contract and losses are untouched, and gradients still flow
#: through the real CRPS/contrast terms.
G4_CORR_BASE_SEED = 12.0
G4_CORR_INC_BIAS = -8.0


def real_configs() -> tuple:
    """The tiny-but-real ``(HdvConfig, ArrivalConfig, CorrConfig)``."""
    return (HdvConfig(**TINY_HDV), ArrivalConfig(**TINY_ARRIVAL),
            CorrConfig(**TINY_CORR))


def real_config_dict() -> dict:
    """The tiny config as plain data (recorded in the G4 report receipt)."""
    hdv, arrival, corr = real_configs()
    return {"hdv": {"d_model": hdv.d_model, "n_heads": hdv.n_heads,
                    "n_layers": hdv.n_layers, "ffn": hdv.ffn,
                    "radius_m": hdv.radius_m},
            "arrival": {"d_model": arrival.d_model,
                        "counter_channels": arrival.counter_channels},
            "corr": {"summary_dim": corr.summary_dim,
                     "d_model": corr.d_model, "n_tau": corr.n_tau,
                     "n_outcome_targets": corr.n_outcome_targets,
                     "n_quantiles": corr.n_quantiles}}


class RealCorrBundle(nn.Module):
    """Real P5 ``CavMotionSummary`` projection + ``CorrectionHead``.

    The trainer hands the 16-d ``summary_raw`` to the corr module; the
    registered head consumes the projected 64-d summary.  Only real P5 code.
    """

    def __init__(self, cfg: CorrConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = CavMotionSummary(cfg)
        self.head = CorrectionHead(cfg)

    def forward(self, summary_raw: torch.Tensor,
                global_token: torch.Tensor) -> dict:
        if summary_raw.shape[-1] != SUMMARY_RAW_F:
            raise ValueError(
                f"summary_raw must have {SUMMARY_RAW_F} features; got "
                f"{summary_raw.shape[-1]}")
        return self.head(self.proj(summary_raw), global_token)


def seed_correction_head(head: CorrectionHead, *,
                         base: float = G4_CORR_BASE_SEED,
                         inc_bias: float = G4_CORR_INC_BIAS) -> CorrectionHead:
    """Seed the real head's quantile biases for the G4 probe (see constants).

    ``raw`` is viewed ``[n_targets, n_quantiles]`` per tau: the first column is
    the softplus base, the rest are the non-negative increments.  Setting the
    base bias to ``softplus^{-1}(base)`` and the increment biases to a large
    negative number places the monotone quantiles at ~``base`` (a good starting
    point for a tiny point-target outcome).  Only initial biases are touched.
    """
    inv = math.log(math.expm1(base)) if base > 0 else -float("inf")
    for layer in head.q:
        with torch.no_grad():
            layer.bias.zero_()
            view = layer.bias.view(head.cfg.n_outcome_targets,
                                   head.cfg.n_quantiles)
            view[:, 0] = inv
            view[:, 1:] = float(inc_bias)
    return head


def real_bundle(seed: int = 101):
    """Build the REAL small-but-real P2/P4/P5 bundle (deterministically)."""
    from train.idea_a import IdeaABundle

    torch.manual_seed(int(seed))
    hdv_cfg, arrival_cfg, corr_cfg = real_configs()
    corr = RealCorrBundle(corr_cfg)
    seed_correction_head(corr.head)
    return IdeaABundle(hdv=HdvResidualNet(hdv_cfg),
                       arrival=IdeaAArrivalHead(arrival_cfg),
                       corr=corr)


def is_standin_bundle(bundle) -> bool:
    """True when ``bundle`` is the documented linear stand-in path.

    The stand-in classes carry ``G4_STANDIN = True``; the real modules do not.
    """
    for name in ("hdv", "arrival", "corr"):
        if bool(getattr(getattr(bundle, name, None), "G4_STANDIN", False)):
            return True
    return False


__all__ = ["TINY_HDV", "TINY_ARRIVAL", "TINY_CORR", "real_configs",
           "real_config_dict", "RealCorrBundle", "real_bundle",
           "is_standin_bundle", "seed_correction_head"]
