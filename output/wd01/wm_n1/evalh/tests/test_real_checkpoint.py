"""Generated compatibility receipts only; no checkpoint or data is opened."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from evalh import real_checkpoint as R


def _source():
    return SimpleNamespace(
        half="cal_tune",
        spec=dict(allocation_id="generated-allocation",
                  data_release_id="generated-release",
                  source_head="generated-source-head",
                  manifest_sha256="a" * 64,
                  index_sha256="b" * 64,
                  checkpoint_member_sha256="c" * 64))


def _contract(source, *, train_h=10.0, infer_h=10.0):
    return dict(
        schema=R.CONTRACT_SCHEMA,
        allocation_id=source.spec["allocation_id"],
        data_release_id=source.spec["data_release_id"],
        cal_half=source.half,
        source_head=source.spec["source_head"],
        pack_manifest_sha256=source.spec["manifest_sha256"],
        pack_index_sha256=source.spec["index_sha256"],
        checkpoint_member_sha256=source.spec["checkpoint_member_sha256"],
        label_binding=R.LABEL_BINDING,
        action_plan_binding=R.ACTION_BINDING,
        slot_identity_binding=R.IDENTITY_BINDING,
        rollout_truth_binding=R.TRUTH_BINDING,
        training_summary_horizon_s=train_h,
        inference_summary_horizon_s=infer_h,
        review_status="independently_verified")


def _write_contract(tmp_path, contract):
    path = tmp_path / "generated-contract.json"
    raw = json.dumps(contract).encode("utf-8")
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


class _PoisonCheckpointPath:
    def __fspath__(self):
        raise AssertionError("checkpoint path was touched")


def test_matching_ten_second_receipt_still_refuses_before_checkpoint_io(tmp_path):
    source = _source()
    contract = _contract(source, train_h=10.0, infer_h=10.0)
    path, sha = _write_contract(tmp_path, contract)
    assert R.verify_compatibility_contract(contract, source) == 20
    with pytest.raises(R.RealCheckpointRefusal, match="not implemented"):
        R.load_released_idea_a(
            source, checkpoint_dir=_PoisonCheckpointPath(),
            contract_path=path, expected_contract_sha256=sha,
            batch_builder=lambda rows: rows)


def test_ten_second_train_and_120_second_inference_refuse():
    source = _source()
    with pytest.raises(R.RealCheckpointRefusal, match="durations must be matching"):
        R.verify_compatibility_contract(
            _contract(source, train_h=10.0, infer_h=120.0), source)


@pytest.mark.parametrize("field,bad", [
    ("cal_half", "cal_cal"),
    ("source_head", "other-source"),
    ("checkpoint_member_sha256", "d" * 64),
    ("pack_manifest_sha256", "e" * 64),
    ("pack_index_sha256", "f" * 64),
])
def test_contract_refuses_wrong_half_or_identity(field, bad):
    source = _source()
    contract = _contract(source)
    contract[field] = bad
    with pytest.raises(R.RealCheckpointRefusal, match="conflicts with"):
        R.verify_compatibility_contract(contract, source)


def test_contract_file_pin_refuses_before_checkpoint_io(tmp_path):
    source = _source()
    path, _sha = _write_contract(tmp_path, _contract(source))
    with pytest.raises(R.RealCheckpointRefusal, match="differs from its pin"):
        R.load_released_idea_a(
            source, checkpoint_dir=_PoisonCheckpointPath(),
            contract_path=path, expected_contract_sha256="0" * 64,
            batch_builder=lambda rows: rows)
