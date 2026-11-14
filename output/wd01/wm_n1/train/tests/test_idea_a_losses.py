"""P7 Idea A loss terms: finite values, gradient flow, contrast sign.

Each A3.2 term is computed on synthetic paired windows with the P2/P4/P5
fakes; every term must be finite and backprop into its module's parameters,
and the paired-branch contrast must follow the treatment-minus-reference
sign convention.
"""
from __future__ import annotations

import pytest
import torch

import train.idea_a as ia
from test_idea_a_fakes import make_windows, seeded_bundle


def _trainer(contrast_gain: float = 0.0, **cfg) -> ia.IdeaATrainer:
    kwargs = dict(out_dir="unused", epochs=0, steps_per_epoch=0,
                  ensemble_k=5)
    kwargs.update(cfg)
    return ia.IdeaATrainer(ia.IdeaATrainConfig(**kwargs), corpus=None,
                           bundle=seeded_bundle(0, contrast_gain))


def test_each_loss_term_finite_and_gradients_flow():
    tr = _trainer()
    batch = ia.collate_canonical(make_windows(2))
    terms = tr.loss_terms(batch)
    assert set(terms) == {"residual_nll", "rollout", "arrival",
                          "correction_crps", "response_contrast"}
    owners = {"residual_nll": "hdv", "rollout": "hdv", "arrival": "arrival",
              "correction_crps": "corr", "response_contrast": "hdv"}
    for name, value in terms.items():
        assert torch.isfinite(value).all(), name
        tr.zero_grad_modules()
        value.backward(retain_graph=False)
        module = getattr(tr.bundle, owners[name])
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"{name} produced no gradients"
        norms = [float(g.norm()) for g in grads]
        assert max(norms) > 0.0, f"{name} gradient is zero"
        assert all(torch.isfinite(g).all() for g in grads), name


def test_total_loss_weighted_sum_and_requires_modules():
    tr = _trainer()
    batch = ia.collate_canonical(make_windows(1))
    terms = tr.loss_terms(batch)
    w = tr.config.loss_weights()
    expected = sum(w[name] * value for name, value in terms.items())
    total = tr.total_loss(batch)
    assert torch.isfinite(total).all()
    assert torch.allclose(total, expected, atol=1e-6)
    weights = {"rollout": 0.0, "arrival": 2.0}
    scaled = tr.total_loss(batch, weights=weights)
    expected2 = sum((weights.get(name, w[name])) * value
                    for name, value in terms.items())
    assert torch.allclose(scaled, expected2, atol=1e-6)
    bare = ia.IdeaATrainer(ia.IdeaATrainConfig(out_dir="u", epochs=0,
                                               steps_per_epoch=0),
                           corpus=None)
    with pytest.raises(ia.IdeaAConfigError):
        bare.loss_terms(batch)
    with pytest.raises(ia.IdeaAConfigError):
        bare.residual_nll(batch)


def test_rollout_loss_requires_teacher_and_is_nonnegative():
    tr = _trainer()
    windows = make_windows(1)
    no_teacher = {k: v for k, v in windows[0].items()
                  if k not in ("teacher", "rollout_accel")}
    with pytest.raises(ia.IdeaAConfigError):
        tr.rollout_loss(no_teacher)
    with torch.no_grad():
        value = tr.rollout_loss(ia.collate_canonical(windows))
    assert torch.isfinite(value).all()
    assert float(value) >= 0.0


def test_contrast_sign_convention_treatment_minus_reference():
    # aligned: predicted contrast (gain>0) agrees with the realised
    # treatment-minus-reference outcome difference -> strictly lower loss
    # than the opposite sign.
    batch = ia.collate_canonical(make_windows(1))
    aligned = float(_trainer(contrast_gain=+0.5)
                    .paired_response_contrast(batch).detach())
    opposite = float(_trainer(contrast_gain=-0.5)
                     .paired_response_contrast(batch).detach())
    assert aligned < opposite
    # zero predicted contrast sits between the two (pinball of dq=0)
    zero = float(_trainer(contrast_gain=0.0)
                 .paired_response_contrast(batch).detach())
    assert aligned < zero < opposite


def test_response_contrast_uses_treatment_minus_reference_targets():
    tr = _trainer(contrast_gain=0.0)
    batch = ia.collate_canonical(make_windows(1))
    # equalise the realised outcomes across arms so the corr-head outcome
    # contrast vanishes and the response term is isolated
    batch["outcome_quantiles"] = batch["outcome_quantiles"][0:1] \
        .repeat(2, 1, 1)
    from train.dataset import contrast_batch_indices

    idx = contrast_batch_indices(batch["meta"])
    a, b = int(idx["a"][0]), int(idx["b"][0])
    pair_mask = (~batch["is_cav"][a]) & (~batch["is_cav"][b]) \
        & batch["target_mask"][a]
    dy = batch["target_accel"][a] - batch["target_accel"][b]
    dy = dy[pair_mask]
    assert float(dy.mean()) > 0.0            # treatment slows HDVs more
    loss = tr.paired_response_contrast(batch)
    # with zero predicted contrast the pinball of dq=0 is |dy| * mean(tau);
    # the outcome sub-term is 0 here, so the average is a quarter of |dy|
    expected = float(dy.abs().mean()) * 0.25
    assert abs(float(loss.detach()) - expected) < 5e-2


def test_contrast_outcome_weight_default_off_and_validated():
    assert ia.IdeaATrainConfig().contrast_outcome_weight is None
    with pytest.raises(ia.IdeaAConfigError):
        ia.IdeaATrainConfig(contrast_outcome_weight=-1.0)
    with pytest.raises(ia.IdeaAConfigError):
        ia.IdeaATrainConfig(contrast_outcome_weight=float("nan"))
    args = ia.build_parser().parse_args(
        ["--member", "1", "--corpus", "P", "--out", "O",
         "--contrast-outcome-weight", "1.5",
         "--require-outcome-contrast"])
    assert args.contrast_outcome_weight == 1.5
    assert args.require_outcome_contrast is True
    assert ia.build_parser().parse_args(
        ["--member", "1", "--corpus", "P", "--out", "O"]
    ).require_outcome_contrast is False


def test_combine_contrast_subterms_own_weight_removes_the_halving():
    outcome, hdv = torch.tensor(2.0), torch.tensor(1.0)
    default = _trainer(contrast_outcome_weight=None)
    # default = historical equal average (the outcome sub-term is halved)
    assert float(default._combine_contrast_subterms(outcome, hdv)) == \
        pytest.approx(1.5)
    # a lone sub-term is returned unchanged in every mode
    assert float(default._combine_contrast_subterms(outcome, None)) == 2.0
    assert float(default._combine_contrast_subterms(None, hdv)) == 1.0
    # w=1.0 gives the outcome sub-term its own weight (not halved by the HDV
    # average); w=2.0 doubles it; the HDV sub-term keeps weight 1.0
    w1 = _trainer(contrast_outcome_weight=1.0)
    assert float(w1._combine_contrast_subterms(outcome, hdv)) == \
        pytest.approx(3.0)
    w2 = _trainer(contrast_outcome_weight=2.0)
    assert float(w2._combine_contrast_subterms(outcome, hdv)) == \
        pytest.approx(5.0)
    assert float(w2._combine_contrast_subterms(outcome, None)) == 2.0


def test_loss_terms_empty_batch_refused():
    tr = _trainer()
    with pytest.raises(ia.IdeaAConfigError):
        tr.loss_terms({"meta": []})
    with pytest.raises(ia.IdeaAConfigError):
        tr.paired_response_contrast({"meta": [{"arm": "native",
                                               "pair_key": "p"}]})
