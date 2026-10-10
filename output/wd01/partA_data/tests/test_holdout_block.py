"""HOLDOUT BLOCK tests: holdout roots require a previously recorded
registration hash; synthetic registration only (no real holdout data is
loaded in tests)."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

P2C = Path(__file__).resolve().parents[2] / "partA_data"
sys.path.insert(0, str(P2C))

import holdout_block as HB  # noqa: E402
import loader as LD  # noqa: E402

HOLDOUT_ROOT = "seed_7012m2700"      # a real holdout root NAME (split only)
TRAIN_ROOT = "cv17"


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _temp_registry(tmp_path, reg_sha, name="reg.json"):
    reg_file = tmp_path / "REGISTRATION_TEST.md"
    reg_file.write_bytes(b"synthetic registration body")
    assert _sha(reg_file.read_bytes()) == reg_sha
    rec = tmp_path / "record.json"
    rec.write_text(json.dumps({"sha256": reg_sha, "file": str(reg_file)}),
                   encoding="utf-8")
    return tmp_path


def test_holdout_without_hash_raises():
    with pytest.raises(HB.HoldoutAccessError):
        HB.assert_holdout_allowed(HOLDOUT_ROOT)


def test_holdout_with_wrong_hash_raises(tmp_path):
    reg_dir = _temp_registry(tmp_path, _sha(b"synthetic registration body"))
    with pytest.raises(HB.HoldoutAccessError):
        HB.assert_holdout_allowed(HOLDOUT_ROOT,
                                  registration_sha256=_sha(b"wrong body"),
                                  records_dirs_list=[reg_dir])


def test_holdout_with_correct_synthetic_registration_passes(tmp_path):
    reg_sha = _sha(b"synthetic registration body")
    reg_dir = _temp_registry(tmp_path, reg_sha)
    HB.assert_holdout_allowed(HOLDOUT_ROOT, registration_sha256=reg_sha,
                              records_dirs_list=[reg_dir])
    # stale registry file (content changed after recording) must fail
    (reg_dir / "REGISTRATION_TEST.md").write_bytes(b"tampered")
    with pytest.raises(HB.HoldoutAccessError):
        HB.assert_holdout_allowed(HOLDOUT_ROOT,
                                  registration_sha256=reg_sha,
                                  records_dirs_list=[reg_dir])


def test_train_val_roots_unaffected():
    HB.assert_holdout_allowed(TRAIN_ROOT)
    HB.assert_holdout_allowed("cv20_7005m2000")     # validation root


def test_loader_load_root_blocked_for_holdout():
    with pytest.raises(HB.HoldoutAccessError):
        LD.load_root(HOLDOUT_ROOT)
