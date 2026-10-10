"""5-member ensemble wrapper (design section 3)."""
from __future__ import annotations

import torch

from model import constants as C
from model.ensemble import DEFAULT_SEEDS, WmN1Ensemble
from synth_model import make_batch, small_cfg


def test_ensemble_has_five_independent_members():
    ens = WmN1Ensemble(cfg=small_cfg())
    assert len(ens.members) == C.N_MEMBERS == 5
    assert len(set(DEFAULT_SEEDS)) == 5
    p0 = ens.members[0].veh_embed[0].weight.detach()
    p1 = ens.members[1].veh_embed[0].weight.detach()
    assert not torch.allclose(p0, p1)


def test_mixture_predictive_shapes_and_finite():
    ens = WmN1Ensemble(cfg=small_cfg()).eval()
    b = make_batch(B=2, K=8)
    with torch.no_grad():
        outs = ens(**b)
        mix = ens.mixture_predictive(outs)
    B = 2
    assert len(outs) == 5
    assert mix["outcome_quantiles"].shape == (
        B, C.N_TAU, C.N_OUTCOME_TARGETS, C.N_QUANTILES)
    assert mix["collision_logits"].shape == (B, C.N_TAU)
    assert torch.isfinite(mix["outcome_quantiles"]).all()
