"""Approval-gate tests: frozen LF-shas, CRLF acceptance, wrong/missing refusals."""
from __future__ import annotations

import hashlib
import json

import pytest

from bundle import approval as A
from evalh.seals import LF_SHA_WM_N1

# Exact verbatim content of REGISTRATION_WM_N1_AMENDMENT_1.md (commit 79e686e).
AMENDMENT_TEXT = (
    "# REGISTRATION_WM_N1 \u2014 AMENDMENT 1: training ORDER (member-1-first) "
    "and dedicated GPU box\n"
    "\n"
    "**Status:** owner decision relayed by main discussion, 2026-09-30.\n"
    "\n"
    "- Amends: `REGISTRATION_WM_N1.md`, LF-sha256\n"
    "  `666233aa1d1e466750d6d944b9102c84a2f32850ebeb4a97c9f3c9b56dd93246`. That\n"
    "  file is NOT edited.\n"
    "- Changes the training ORDER and the compute location only. The \u00a74 "
    "targets,\n"
    "  the chunk-5 TEST (`REGISTRATION_CHUNK5_TEST.md`, LF-sha256\n"
    "  `f8c7d760baedeb359c501cd90a3e2a599fb0ddfef390ea555cf304c38214b31e`), the\n"
    "  three variants (N/T/C), the losses, the architecture and the CAL-only\n"
    "  calibration are UNCHANGED.\n"
    "\n"
    "## A1. Member-1-first\n"
    "\n"
    "1. After the 1-h timing gate and the owner's approval file, train ensemble\n"
    "   member 1 ALONE to completion, giving an early working model.\n"
    "2. Then train members 2\u20135, with the same config and independent seeds\n"
    "   recorded before training.\n"
    "\n"
    "## A2. Use of the early (1-member) model\n"
    "\n"
    "- Only internal checks on TRAIN (OOF/in-sample) and CAL are allowed, for\n"
    "  example loss curves, calibration diagnostics and a planner integration\n"
    "  smoke.\n"
    "- It supports NO selection or superiority claim.\n"
    "- CAL use for the early model is diagnostic only. It must not change the\n"
    "  architecture, the hyperparameters or the variant rule. The registered\n"
    "  CAL-only calibration and stacking are fitted once, on the full 5-member\n"
    "  ensemble.\n"
    "- Chunk 5 stays SEALED until the single final test. That test uses the full\n"
    "  5-member ensemble and the frozen PLANNER_VARIANT.json. The early model is\n"
    "  never run on chunk 5.\n"
    "\n"
    "## A3. Compute\n"
    "\n"
    "- A dedicated second GPU box runs WM-N1 only, independent of the recon GPU\n"
    "  box. It uses the same env spec and the same guard and sync.\n"
    "- Its queue is: verify upload \u2192 memmap verify \u2192 1-h timing gate STOP "
    "\u2192 owner\n"
    "  approval file \u2192 member 1 \u2192 members 2\u20135.\n"
    "- The design \u00a77 \"shared box\" text is superseded for location only.\n"
)


def _lf_sha(data: bytes) -> str:
    return hashlib.sha256(
        data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _crlf(text: str) -> bytes:
    return text.replace("\n", "\r\n").encode("utf-8")


def test_frozen_constants_and_live_registration():
    assert A.LF_SHA_AMENDMENT_1 == (
        "decb96562ebfb549b56406d980be9c78bf4a377bd50c5eadeea08aac16e16c57")
    assert LF_SHA_WM_N1 == (
        "666233aa1d1e466750d6d944b9102c84a2f32850ebeb4a97c9f3c9b56dd93246")
    # the transcribed amendment fixture really is the frozen record
    assert _lf_sha(AMENDMENT_TEXT.encode("utf-8")) == A.LF_SHA_AMENDMENT_1


def _records(tmp_path, monkeypatch):
    """LF and CRLF copies of both frozen records; point the module at them."""
    reg_lf = A.REGISTRATION_PATH
    assert reg_lf.is_file(), reg_lf
    reg = tmp_path / "REGISTRATION_WM_N1.md"
    reg_bytes = reg_lf.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    reg.write_bytes(reg_bytes.replace(b"\n", b"\r\n"))          # CRLF copy
    amd = tmp_path / "REGISTRATION_WM_N1_AMENDMENT_1.md"
    amd.write_bytes(_crlf(AMENDMENT_TEXT))                      # CRLF copy
    monkeypatch.setattr(A, "REGISTRATION_PATH", reg)
    monkeypatch.setattr(A, "AMENDMENT_PATH", amd)
    # Amendment 4 + the CAL split (A4.2), copied as CRLF (LF-sha preserved).
    a4 = tmp_path / A.AMENDMENT_4_PATH.name
    a4_bytes = A.AMENDMENT_4_PATH.read_bytes().replace(
        b"\r\n", b"\n").replace(b"\r", b"\n")
    a4.write_bytes(a4_bytes.replace(b"\n", b"\r\n"))
    split = tmp_path / A.CAL_SPLIT_PATH.name
    split_bytes = A.CAL_SPLIT_PATH.read_bytes().replace(
        b"\r\n", b"\n").replace(b"\r", b"\n")
    split.write_bytes(split_bytes.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(A, "AMENDMENT_4_PATH", a4)
    monkeypatch.setattr(A, "CAL_SPLIT_PATH", split)
    # Amendment 6 (state global token, A6), copied as CRLF (LF-sha preserved).
    a6 = tmp_path / A.AMENDMENT_6_PATH.name
    a6_bytes = A.AMENDMENT_6_PATH.read_bytes().replace(
        b"\r\n", b"\n").replace(b"\r", b"\n")
    a6.write_bytes(a6_bytes.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(A, "AMENDMENT_6_PATH", a6)
    return reg, amd


def _approval(tmp_path, gate_bytes=b"{\"g\": 1}\n", **overrides):
    gate = tmp_path / "TIMING_GATE.json"
    gate.write_bytes(gate_bytes)
    body = {
        "schema": "wm_n1_approval/1",
        "approved": True,
        "owner": "owner",
        "approved_utc": "2026-09-30T00:00:00Z",
        "stage": "member-1",
        "timing_gate_sha256": hashlib.sha256(gate_bytes).hexdigest(),
        "registration_lf_sha256": LF_SHA_WM_N1,
        "amendment_lf_sha256": A.LF_SHA_AMENDMENT_1,
        "amendment_4_lf_sha256": A.LF_SHA_AMENDMENT_4,
        "cal_split_lf_sha256": A.LF_SHA_CAL_SPLIT_A4,
        "amendment_6_lf_sha256": A.LF_SHA_AMENDMENT_6,
    }
    body.update(overrides)
    path = tmp_path / "APPROVAL.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path, gate


def test_require_approval_accepts_crlf_copies(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, _ = _approval(tmp_path)
    got = A.require_approval(path, stage="member-1")
    assert got["owner"] == "owner"
    assert got["stage_requested"] == "member-1"


def test_approval_present_true(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, _ = _approval(tmp_path)
    assert A.approval_present(path) is True


def test_require_approval_refuses_missing_file(tmp_path):
    assert A.approval_present(tmp_path / "nope.json") is False
    with pytest.raises(A.ApprovalMissing):
        A.require_approval(tmp_path / "nope.json", stage="member-1")


def test_require_approval_refuses_not_approved(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, _ = _approval(tmp_path, approved=False)
    assert A.approval_present(path) is False
    with pytest.raises(A.ApprovalMissing):
        A.require_approval(path, stage="member-1")


def test_require_approval_refuses_wrong_registration_sha(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, _ = _approval(tmp_path, registration_lf_sha256="0" * 64)
    with pytest.raises(A.ApprovalMissing):
        A.require_approval(path, stage="member-1")


def test_require_approval_refuses_wrong_amendment_sha(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, _ = _approval(tmp_path, amendment_lf_sha256="0" * 64)
    with pytest.raises(A.ApprovalMissing):
        A.require_approval(path, stage="member-1")


def test_require_approval_refuses_wrong_timing_gate_sha(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, _ = _approval(tmp_path, timing_gate_sha256="0" * 64)
    with pytest.raises(A.ApprovalMissing):
        A.require_approval(path, stage="member-1")


def test_require_approval_refuses_missing_timing_gate(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, gate = _approval(tmp_path)
    gate.unlink()
    with pytest.raises(A.ApprovalMissing):
        A.require_approval(path, stage="member-1")


def test_require_approval_refuses_tampered_amendment_file(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, _ = _approval(tmp_path)
    # declared field is still the frozen constant, but the record was edited:
    # the gate must re-hash and refuse.
    A.AMENDMENT_PATH.write_bytes(
        (AMENDMENT_TEXT + "tampered\n").encode("utf-8"))
    with pytest.raises(A.ApprovalMissing):
        A.require_approval(path, stage="member-1")


def test_require_approval_refuses_missing_binding_fields(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, _ = _approval(tmp_path)
    body = json.loads(path.read_text(encoding="utf-8"))
    del body["timing_gate_sha256"]
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(A.ApprovalMissing):
        A.require_approval(path, stage="member-1")


# --------------------------------------------------- APPROVAL_A_M2_5 gate -----
def _write_default_winner(path):
    from bundle import winning_config as WC

    rec = WC.build_winning_config(
        selected="default",
        knobs={"learning_rate": 1.0, "epochs": 1.0,
               "rollout_loss_weight": 1.0, "contrast_loss_weight": 1.0},
        rounds=[{"round": 0, "configs": ["default"],
                 "cal_tune_scores": {"default": 0.5}}],
        cal_tune_scores={"default": 0.5})
    WC.write_winning_config(path, rec)
    return path


def _m1_only_approval(tmp_path, *, m1_bytes=b"member-1-ckpt", **overrides):
    m1 = tmp_path / "member_1.pt"
    m1.write_bytes(m1_bytes)
    win = _write_default_winner(tmp_path / A.WINNING_CONFIG_FILENAME)
    body = {
        "schema": A.APPROVAL_A_M2_5_SCHEMA,
        "approved": True,
        "owner": "owner",
        "approved_utc": "2026-10-01T00:00:00Z",
        "registration_lf_sha256": LF_SHA_WM_N1,
        "amendment_lf_sha256": A.LF_SHA_AMENDMENT_1,
        "amendment_4_lf_sha256": A.LF_SHA_AMENDMENT_4,
        "cal_split_lf_sha256": A.LF_SHA_CAL_SPLIT_A4,
        "amendment_6_lf_sha256": A.LF_SHA_AMENDMENT_6,
        "m1_checkpoint_sha256": hashlib.sha256(m1_bytes).hexdigest(),
        "winning_config_sha256": hashlib.sha256(
            win.read_bytes()).hexdigest(),
    }
    body.update(overrides)
    path = tmp_path / A.APPROVAL_A_M2_5_FILENAME
    path.write_text(json.dumps(body), encoding="utf-8")
    return path, m1, win


def test_require_a_m2_5_accepts_binding(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, m1, win = _m1_only_approval(tmp_path)
    got = A.require_a_m2_5(path, m1_checkpoint_path=m1,
                           winning_config_path=win)
    assert got["owner"] == "owner"
    assert got["m1_checkpoint_resolved"] == str(m1)
    assert got["winning_config_resolved"] == str(win)


def test_require_a_m2_5_refuses_missing_file(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    _, m1, win = _m1_only_approval(tmp_path)
    with pytest.raises(A.ApprovalMissing):
        A.require_a_m2_5(tmp_path / "nope.json", m1_checkpoint_path=m1,
                         winning_config_path=win)


def test_require_a_m2_5_refuses_wrong_m1_sha(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, m1, win = _m1_only_approval(tmp_path, m1_checkpoint_sha256="0" * 64)
    with pytest.raises(A.ApprovalMissing) as ei:
        A.require_a_m2_5(path, m1_checkpoint_path=m1, winning_config_path=win)
    assert "m1_checkpoint_sha256" in str(ei.value)


def test_require_a_m2_5_refuses_wrong_winning_sha(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, m1, win = _m1_only_approval(tmp_path,
                                      winning_config_sha256="0" * 64)
    with pytest.raises(A.ApprovalMissing) as ei:
        A.require_a_m2_5(path, m1_checkpoint_path=m1, winning_config_path=win)
    assert "winning_config_sha256" in str(ei.value)


def test_require_a_m2_5_refuses_tampered_checkpoint(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, m1, win = _m1_only_approval(tmp_path)
    m1.write_bytes(b"different-bytes")
    with pytest.raises(A.ApprovalMissing):
        A.require_a_m2_5(path, m1_checkpoint_path=m1, winning_config_path=win)


def test_require_a_m2_5_refuses_missing_checkpoint(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, m1, win = _m1_only_approval(tmp_path)
    with pytest.raises(A.ApprovalMissing):
        A.require_a_m2_5(path, m1_checkpoint_path=tmp_path / "missing.pt",
                         winning_config_path=win)


# ------------------------------------- Amendment 4 + CAL split binding -------
def test_frozen_a4_and_cal_split_constants():
    assert A.LF_SHA_AMENDMENT_4 == (
        "d392b30030f2b04ddb5db25c681cf92bca80fe1fc4e77d234303f7bbb810822f")
    assert A.LF_SHA_CAL_SPLIT_A4 == (
        "bc4e32f2b6efcf0e0165cb175a0b3ffaec0aaa6f419076463f9242ad90820d46")
    assert A.AMENDMENT_4_PATH.is_file()
    assert A.CAL_SPLIT_PATH.is_file()


def test_require_approval_refuses_wrong_amendment_4_sha(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, _ = _approval(tmp_path, amendment_4_lf_sha256="0" * 64)
    with pytest.raises(A.ApprovalMissing):
        A.require_approval(path, stage="member-1")


def test_require_approval_refuses_wrong_cal_split_sha(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, _ = _approval(tmp_path, cal_split_lf_sha256="0" * 64)
    with pytest.raises(A.ApprovalMissing):
        A.require_approval(path, stage="member-1")


def test_require_approval_refuses_tampered_amendment_4(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, _ = _approval(tmp_path)
    A.AMENDMENT_4_PATH.write_bytes(
        A.AMENDMENT_4_PATH.read_bytes() + b"tampered\n")
    with pytest.raises(A.ApprovalMissing):
        A.require_approval(path, stage="member-1")


def test_require_approval_refuses_tampered_cal_split(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, _ = _approval(tmp_path)
    A.CAL_SPLIT_PATH.write_bytes(A.CAL_SPLIT_PATH.read_bytes() + b"\n")
    with pytest.raises(A.ApprovalMissing):
        A.require_approval(path, stage="member-1")


def test_require_a_m2_5_refuses_wrong_amendment_4_sha(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, m1, win = _m1_only_approval(tmp_path, amendment_4_lf_sha256="0" * 64)
    with pytest.raises(A.ApprovalMissing):
        A.require_a_m2_5(path, m1_checkpoint_path=m1,
                         winning_config_path=win)


def test_require_a_m2_5_refuses_wrong_cal_split_sha(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, m1, win = _m1_only_approval(tmp_path, cal_split_lf_sha256="0" * 64)
    with pytest.raises(A.ApprovalMissing):
        A.require_a_m2_5(path, m1_checkpoint_path=m1,
                         winning_config_path=win)


def test_require_a_m2_5_refuses_invalid_winning_config(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, m1, win = _m1_only_approval(tmp_path)
    win.write_bytes(b'{"not": "a winning config"}\n')
    body = json.loads(path.read_text(encoding="utf-8"))
    body["winning_config_sha256"] = hashlib.sha256(
        win.read_bytes()).hexdigest()
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(A.ApprovalMissing):
        A.require_a_m2_5(path, m1_checkpoint_path=m1,
                         winning_config_path=win)


# ------------------------------------- Amendment 6 (state global token, A6) --
def test_frozen_a6_constant_and_file():
    assert A.LF_SHA_AMENDMENT_6 == (
        "28dbc184b0178d5f74f681e0701771725fc8abc80e534e1087f0a9e49835c948")
    assert A.AMENDMENT_6_PATH.is_file()


def test_require_approval_refuses_wrong_amendment_6_sha(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, _ = _approval(tmp_path, amendment_6_lf_sha256="0" * 64)
    with pytest.raises(A.ApprovalMissing) as ei:
        A.require_approval(path, stage="member-1")
    assert "amendment_6_lf_sha256" in str(ei.value)


def test_require_approval_refuses_missing_amendment_6_field(tmp_path,
                                                            monkeypatch):
    _records(tmp_path, monkeypatch)
    path, _ = _approval(tmp_path)
    body = json.loads(path.read_text(encoding="utf-8"))
    del body["amendment_6_lf_sha256"]
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(A.ApprovalMissing):
        A.require_approval(path, stage="member-1")


def test_require_approval_refuses_tampered_amendment_6(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, _ = _approval(tmp_path)
    A.AMENDMENT_6_PATH.write_bytes(
        A.AMENDMENT_6_PATH.read_bytes() + b"tampered\n")
    with pytest.raises(A.ApprovalMissing):
        A.require_approval(path, stage="member-1")


def test_require_approval_refuses_missing_amendment_6_file(tmp_path,
                                                           monkeypatch):
    _records(tmp_path, monkeypatch)
    path, _ = _approval(tmp_path)
    A.AMENDMENT_6_PATH.unlink()
    with pytest.raises(A.ApprovalMissing):
        A.require_approval(path, stage="member-1")


def test_require_a_m2_5_refuses_wrong_amendment_6_sha(tmp_path, monkeypatch):
    _records(tmp_path, monkeypatch)
    path, m1, win = _m1_only_approval(tmp_path, amendment_6_lf_sha256="0" * 64)
    with pytest.raises(A.ApprovalMissing):
        A.require_a_m2_5(path, m1_checkpoint_path=m1,
                         winning_config_path=win)
