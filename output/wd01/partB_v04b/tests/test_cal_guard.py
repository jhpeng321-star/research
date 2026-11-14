"""The ONE-CAL-read guard: OWNER_ACK gate + read-once marker + the
TRAIN-only hard guard (registration s.7)."""
from __future__ import annotations

import json

import pytest

from partB_v04b import cal_guard as CG
from partB_v04b import reg


@pytest.fixture()
def sha():
    return "e00cd98e" + "a" * 56


def _ack(path, sha, ack_text="go"):
    path.write_text(json.dumps(dict(
        ack=ack_text, registration_lf_sha256=sha, owner="owner",
        utc="2026-10-01T00:00:00Z")), encoding="utf-8")
    return path


def test_owner_ack_refuses_when_missing(tmp_path, sha):
    with pytest.raises(CG.CalReadNotAllowed, match="no OWNER_ACK"):
        CG.require_owner_ack(tmp_path / "ack.json", reg_sha=sha)


def test_owner_ack_refuses_a_stale_or_foreign_sha(tmp_path, sha):
    p = _ack(tmp_path / "ack.json", "deadbeef" + "b" * 56)
    with pytest.raises(CG.CalReadNotAllowed, match="stale or foreign"):
        CG.require_owner_ack(p, reg_sha=sha)


def test_owner_ack_refuses_an_empty_statement(tmp_path, sha):
    p = _ack(tmp_path / "ack.json", sha, ack_text="  ")
    with pytest.raises(CG.CalReadNotAllowed, match="no ack statement"):
        CG.require_owner_ack(p, reg_sha=sha)


def test_owner_ack_refuses_broken_json(tmp_path, sha):
    p = tmp_path / "ack.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(CG.CalReadNotAllowed, match="not valid JSON"):
        CG.require_owner_ack(p, reg_sha=sha)


def test_owner_ack_accepts_the_bound_ack(tmp_path, sha):
    p = _ack(tmp_path / "ack.json", sha)
    got = CG.require_owner_ack(p, reg_sha=sha)
    assert got["ack"] == "go"


def test_read_once_marker_refuses_the_second_read(tmp_path, sha):
    cr = CG.CalReadOnce(tmp_path, reg_sha=sha)
    cr.begin()
    with pytest.raises(CG.CalReadNotAllowed, match="already started"):
        cr.begin()
    assert not cr.spent()
    cr.finish()
    assert cr.spent()
    with pytest.raises(CG.CalReadNotAllowed, match="second read"):
        cr.begin()


def test_read_once_marker_refuses_finish_before_begin(tmp_path, sha):
    with pytest.raises(CG.CalReadNotAllowed, match="begin"):
        CG.CalReadOnce(tmp_path, reg_sha=sha).finish()


def test_read_once_marker_binds_the_registration(tmp_path, sha):
    other = "12345678" + "c" * 56
    CG.CalReadOnce(tmp_path, reg_sha=other).begin()
    with pytest.raises(CG.MarkerMismatch, match="DIFFERENT"):
        CG.CalReadOnce(tmp_path, reg_sha=sha).begin()
    # and the finish of the SAME registration is unaffected
    CG.CalReadOnce(tmp_path, reg_sha=other).finish()


def test_marker_survives_as_written(tmp_path, sha):
    cr = CG.CalReadOnce(tmp_path, reg_sha=sha)
    m = cr.begin()
    raw = json.loads((tmp_path / reg.CAL_MARKER_NAME).read_text(
        encoding="utf-8"))
    assert raw["registration_lf_sha256"] == sha
    assert raw["finished_utc"] is None
    assert m["started_utc"]


def test_context_manager_finishes_and_propagates_crashes(tmp_path, sha):
    with CG.CalReadOnce(tmp_path / "a", reg_sha=sha):
        pass
    assert CG.CalReadOnce(tmp_path / "a", reg_sha=sha).spent()
    with pytest.raises(RuntimeError):
        with CG.CalReadOnce(tmp_path / "b", reg_sha="00" + "d" * 62):
            raise RuntimeError("crash after the CAL read")
    # the crash left the marker open: reported as is (s.7)
    m = json.loads((tmp_path / "b" / reg.CAL_MARKER_NAME).read_text(
        encoding="utf-8"))
    assert m["finished_utc"] is None


def test_assert_train_only_refuses_non_train():
    CG.assert_train_only(["TRAIN", "TRAIN"], "ok")
    with pytest.raises(CG.CalReadNotAllowed, match="CALIBRATION"):
        CG.assert_train_only(["TRAIN", "CALIBRATION"], "leak")


def test_default_ack_path_is_the_results_dir():
    assert CG.owner_ack_path() == \
        reg.RESULTS_DIR / reg.OWNER_ACK_NAME
