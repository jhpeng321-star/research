"""WM-N1 deep ensemble (design section 3, uncertainty).

A 5-member wrapper with independent seeds and a mixture predictive.  No
calibration is fitted here: per-tau conformal calibration is CAL-only and
lives in the (separately registered) calibration step.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import constants as C
from .net import WmN1Config, build_member, parameter_count

DEFAULT_SEEDS = (101, 202, 303, 404, 505)


class WmN1Ensemble(nn.Module):
    """``N_MEMBERS`` independently seeded WM-N1 members (design section 3)."""

    def __init__(self, cfg: WmN1Config | None = None,
                 seeds=DEFAULT_SEEDS):
        super().__init__()
        self.cfg = cfg or WmN1Config()
        self.seeds = tuple(int(s) for s in seeds)
        if len(self.seeds) != C.N_MEMBERS:
            raise ValueError(f"ensemble expects {C.N_MEMBERS} seeds, got "
                             f"{len(self.seeds)}")
        self.members = nn.ModuleList(
            [build_member(seed=s, cfg=self.cfg) for s in self.seeds])

    def forward(self, **inputs):
        return [m(**inputs) for m in self.members]

    # -- mixture predictive -------------------------------------------------
    @staticmethod
    def mixture_quantiles(member_outputs):
        """Average member quantiles (equal-weight mixture summary).

        The mixture CDF is the mean of the member CDFs; the reported summary
        is the mean of the member quantile vectors, a monotone-preserving
        proxy.  Calibration is deliberately NOT applied here.
        """
        stack = torch.stack([o["outcome_quantiles"] for o in member_outputs],
                            dim=0)
        return stack.mean(dim=0)

    @staticmethod
    def mixture_collision_logit(member_outputs):
        probs = torch.stack(
            [torch.sigmoid(o["collision_logits"]) for o in member_outputs],
            dim=0).mean(dim=0)
        return torch.logit(probs.clamp(1e-6, 1 - 1e-6))

    def mixture_predictive(self, member_outputs):
        return dict(outcome_quantiles=self.mixture_quantiles(member_outputs),
                    collision_logits=self.mixture_collision_logit(
                        member_outputs))

    def parameter_count(self) -> int:
        return parameter_count(self)


def build_ensemble(cfg: WmN1Config | None = None, seeds=DEFAULT_SEEDS):
    return WmN1Ensemble(cfg=cfg, seeds=seeds)
