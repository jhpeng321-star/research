"""Seal tests: CAL read-once, chunk-5 TEST gate, HOLDOUT refusal."""
from __future__ import annotations

import pytest

from evalh import seals as S


def test_lf_normalisation_and_crlf_hash():
    assert S.lf_normalize(b"a\r\nb") == b"a\nb"
    assert S.lf_normalize(b"a\rb") == b"a\nb"
    assert S.lf_sha256(S.REGISTRATION_WM_N1) == S.LF_SHA_WM_N1
    assert S.lf_sha256(S.REGISTRATION_CHUNK5) == S.LF_SHA_CHUNK5
    # raw bytes are CRLF in this checkout, so the raw hash differs
    import hashlib
    raw = hashlib.sha256(S.REGISTRATION_WM_N1.read_bytes()).hexdigest()
    assert raw != S.LF_SHA_WM_N1


def test_unknown_split_refused(tmp_path):
    with pytest.raises(S.SealError):
        S.guard_split("holdout", marker_dir=tmp_path)


def test_missing_final_test_refused(tmp_path):
    with pytest.raises(S.SealError):
        S.guard_split("final_test", marker_dir=tmp_path)


def test_wrong_chunk5_sha_refused(tmp_path):
    with pytest.raises(S.SealError):
        S.guard_split("final_test", marker_dir=tmp_path, final_test=True,
                      chunk5_sha="deadbeef", registration_sha=S.LF_SHA_WM_N1)


def test_wrong_wmn1_sha_refused(tmp_path):
    with pytest.raises(S.SealError):
        S.guard_split("final_test", marker_dir=tmp_path, final_test=True,
                      chunk5_sha=S.LF_SHA_CHUNK5,
                      registration_sha="deadbeef")


def test_correct_hashes_accepted(tmp_path):
    r = S.guard_split("final_test", marker_dir=tmp_path, final_test=True,
                      chunk5_sha=S.LF_SHA_CHUNK5, registration_sha=S.LF_SHA_WM_N1)
    assert r["allowed"] is True and r["split"] == "final_test"
    assert (tmp_path / S.TEST_MARKER).is_file()


def test_crlf_copy_accepted_via_lf_normalisation(tmp_path):
    crlf_wmn1 = tmp_path / "REGISTRATION_WM_N1.md"
    crlf_c5 = tmp_path / "REGISTRATION_CHUNK5_TEST.md"
    for src, dst in ((S.REGISTRATION_WM_N1, crlf_wmn1),
                     (S.REGISTRATION_CHUNK5, crlf_c5)):
        text = src.read_text(encoding="utf-8").replace("\r\n", "\n")
        dst.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    assert S.lf_sha256(crlf_wmn1) == S.LF_SHA_WM_N1
    assert S.lf_sha256(crlf_c5) == S.LF_SHA_CHUNK5
    r = S.guard_split("final_test", marker_dir=tmp_path / "m",
                      final_test=True, wmn1_path=crlf_wmn1,
                      chunk5_path=crlf_c5)
    assert r["allowed"] is True


def test_cal_is_read_once(tmp_path):
    first = S.guard_split("cal", marker_dir=tmp_path)
    assert first["sealed"] is True
    assert (tmp_path / S.CAL_MARKER).is_file()
    with pytest.raises(S.SealError):
        S.guard_split("cal", marker_dir=tmp_path)


def test_test_is_read_once(tmp_path):
    S.guard_split("final_test", marker_dir=tmp_path, final_test=True,
                  chunk5_sha=S.LF_SHA_CHUNK5, registration_sha=S.LF_SHA_WM_N1)
    with pytest.raises(S.SealError):
        S.guard_split("final_test", marker_dir=tmp_path, final_test=True,
                      chunk5_sha=S.LF_SHA_CHUNK5,
                      registration_sha=S.LF_SHA_WM_N1)


def test_dry_run_does_not_consume(tmp_path):
    S.guard_split("cal", marker_dir=tmp_path, dry_run=True)
    assert not (tmp_path / S.CAL_MARKER).exists()
    S.guard_split("cal", marker_dir=tmp_path)
    assert (tmp_path / S.CAL_MARKER).is_file()


def test_train_oof_free():
    r = S.guard_split("train_oof", marker_dir="ignored-not-created")
    assert r["allowed"] is True and r["sealed"] is False


def test_literal_holdout_refused():
    with pytest.raises(S.HoldoutRefusal):
        S.assert_holdout(["HOLDOUT"])


def test_excluded_root_refused():
    data = {"schema": "holdout_exclusions/1",
            "excluded": [{"world": "wExposed"}], "kept_disclosed": []}
    with pytest.raises(S.HoldoutRefusal):
        S.assert_holdout(["wExposed__native"], data=data)


def test_clean_worlds_pass():
    data = {"schema": "holdout_exclusions/1",
            "excluded": [{"world": "wExposed"}], "kept_disclosed": []}
    S.assert_holdout(["wFresh001", "wFresh002"], data=data)
