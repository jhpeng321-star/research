"""Fail-closed factory boundary for the flat, trained Idea A checkpoint.

The current M1 checkpoint is *not* declared evaluation-ready here.  Its
source-level training target and correction-summary time contracts require
independent proof before this factory may deserialize or run it.  A caller
must present an externally SHA-pinned compatibility receipt tied to the
released CAL source and the exact checkpoint member bytes.
"""
from __future__ import annotations

import math

from .real_source import RealSourceRefusal, _read_pinned_json

CONTRACT_SCHEMA = "wm_n1_train_eval_compatibility/1"
LABEL_BINDING = "unique-t0-decision-per-tau"
ACTION_BINDING = "packed-per-cav-plan-parity-verified"
IDENTITY_BINDING = "canonical-t0-slot-identity-verified"
TRUTH_BINDING = "same-t0-roster-and-mask-parity-verified"


class RealCheckpointRefusal(RealSourceRefusal):
    """The trained checkpoint lacks an independently pinned eval contract."""


def verify_compatibility_contract(contract: dict, source) -> int:
    """Validate a reviewed receipt; return its matched summary horizon.

    This requires a positive, identical summary duration for training and
    inference.  It does not require retraining at 120 s: a future evaluator
    could compute the original 10 s correction summary while separately
    measuring dynamics at 120 s.  This receipt alone cannot enable scoring;
    the real action-schedule and identity adapters remain unimplemented.
    """
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise RealCheckpointRefusal("unknown train/eval compatibility schema")
    spec = source.spec
    exact = {
        "allocation_id": spec["allocation_id"],
        "data_release_id": spec["data_release_id"],
        "cal_half": source.half,
        "source_head": spec["source_head"],
        "pack_manifest_sha256": spec["manifest_sha256"],
        "pack_index_sha256": spec["index_sha256"],
        "checkpoint_member_sha256": spec["checkpoint_member_sha256"],
        "label_binding": LABEL_BINDING,
        "action_plan_binding": ACTION_BINDING,
        "slot_identity_binding": IDENTITY_BINDING,
        "rollout_truth_binding": TRUTH_BINDING,
    }
    mismatches = [key for key, value in exact.items()
                  if contract.get(key) != value]
    if mismatches:
        raise RealCheckpointRefusal(
            f"compatibility receipt missing or conflicts with {mismatches}")
    train_h = contract.get("training_summary_horizon_s")
    infer_h = contract.get("inference_summary_horizon_s")
    try:
        train_h, infer_h = float(train_h), float(infer_h)
    except (TypeError, ValueError) as exc:
        raise RealCheckpointRefusal("summary horizons are missing") from exc
    if not (math.isfinite(train_h) and math.isfinite(infer_h)
            and 0 < train_h == infer_h
            and train_h / 0.5 == round(train_h / 0.5)):
        raise RealCheckpointRefusal(
            "correction summary durations must be matching positive 0.5 s "
            "multiples; default 10 s training and 120 s inference mismatch")
    if contract.get("review_status") != "independently_verified":
        raise RealCheckpointRefusal("compatibility proof is not independently verified")
    return int(round(infer_h / 0.5))


def load_released_idea_a(source, *, checkpoint_dir, contract_path,
                         expected_contract_sha256, batch_builder=None):
    """Refuse the current unfinished production adapter before checkpoint I/O.

    ``IdeaABundlePredictor`` currently invokes ``rollout2`` with no per-CAV
    actions, defaulting all CAVs to native.  Supplying a batch builder or a
    nominally valid receipt cannot make that trajectory policy-conditioned.
    The future implementation must bind the actual packed schedule, canonical
    slot identities and training-time correction summary explicitly.
    """
    try:
        contract = _read_pinned_json(contract_path, expected_contract_sha256,
                                     "train/eval compatibility receipt")
    except RealSourceRefusal as exc:
        raise RealCheckpointRefusal(str(exc)) from exc
    verify_compatibility_contract(contract, source)
    raise RealCheckpointRefusal(
        "real checkpoint scoring adapter is not implemented: the current "
        "rollout ignores packed per-CAV plans and lacks proved roster/summary "
        "parity; no checkpoint bytes were read")
