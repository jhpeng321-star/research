"""Synthetic checks for observational paired-contrast residual telemetry."""

import pytest
import torch

from model import constants as C
from train.idea_a import IdeaABundle, IdeaAConfigError, IdeaATrainConfig, IdeaATrainer
from test_idea_a_fakes import make_windows, seeded_bundle


class _EchoHdv(torch.nn.Module):
    def forward(self, kin, kin_mask):
        return {"accel_delta_mps2": kin[..., 5]}


def _hdv_case():
    batch = make_windows(1, k=8)[0]
    batch["vehicle_ids"] = torch.arange(8).expand(2, 8).clone()
    batch["outcome_mask"].zero_()
    batch["baseline_idm"].zero_()
    batch["target_accel"] = batch["kin"][..., 5].clone()
    trainer = IdeaATrainer(IdeaATrainConfig(), corpus=None,
                           bundle=IdeaABundle(hdv=_EchoHdv()))
    return trainer, batch


def test_hdv_zero_proves_nonzero_predicted_contrast_matches_target():
    trainer, batch = _hdv_case()
    assert bool((batch["kin"][0, 2:, 5] - batch["kin"][1, 2:, 5] != 0).any())
    assert float(trainer.paired_response_contrast(batch)) == 0.0
    diag = trainer._contrast_diag
    assert diag["hdv_subterm"] and not diag["outcome_subterm"]
    assert diag["residual_checks"] == {
        "hdv": {"count": 6, "nonzero_count": 0,
                "max_abs": 0.0, "finite": True}}


def test_hdv_nonzero_residual_reports_only_changed_matched_id():
    trainer, batch = _hdv_case()
    batch["target_accel"][0, 2] += 1.0
    assert float(trainer.paired_response_contrast(batch)) > 0.0
    check = trainer._contrast_diag["residual_checks"]["hdv"]
    assert check["count"] == 6 and check["nonzero_count"] == 1
    assert check["max_abs"] > 0.0 and check["finite"]
    assert "outcome" not in trainer._contrast_diag["residual_checks"]


def test_outcome_residual_counts_only_shared_tau_targets_and_quantiles():
    batch = make_windows(1, k=8)[0]
    batch["outcome_quantiles"].zero_()
    batch["outcome_quantiles"][0] = 2.0
    batch["outcome_mask"].zero_()
    batch["outcome_mask"][:, 0] = True
    bundle = seeded_bundle(0, contrast_gain=1.0)
    bundle.hdv = None
    trainer = IdeaATrainer(IdeaATrainConfig(), corpus=None, bundle=bundle)
    assert float(trainer.paired_response_contrast(batch)) == 0.0
    check = trainer._contrast_diag["residual_checks"]["outcome"]
    assert check == {"count": C.N_OUTCOME_TARGETS * C.N_QUANTILES,
                     "nonzero_count": 0, "max_abs": 0.0, "finite": True}
    assert "hdv" not in trainer._contrast_diag["residual_checks"]
    batch["outcome_quantiles"][0, 0, 0] += 1.0
    assert float(trainer.paired_response_contrast(batch)) > 0.0
    check = trainer._contrast_diag["residual_checks"]["outcome"]
    assert check["count"] == C.N_OUTCOME_TARGETS * C.N_QUANTILES
    assert check["nonzero_count"] == C.N_QUANTILES
    assert check["max_abs"] == 1.0 and check["finite"]


def test_no_support_has_no_residual_proof():
    trainer, batch = _hdv_case()
    batch["target_mask"].zero_()
    with pytest.raises(IdeaAConfigError, match="NO valid contrast support"):
        trainer.paired_response_contrast(batch)
    assert trainer._contrast_diag["residual_checks"] == {}
