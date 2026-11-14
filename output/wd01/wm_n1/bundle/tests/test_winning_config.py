"""WINNING_CONFIG.json schema + writer, and its use by APPROVAL_A_M2_5."""
from __future__ import annotations

import hashlib

import pytest

from bundle import approval as A
from bundle import winning_config as WC

KNOBS = {"learning_rate": 1.0, "epochs": 1.0,
         "rollout_loss_weight": 1.0, "contrast_loss_weight": 1.0}


def _rounds():
    return [
        {"round": 0, "configs": ["default", "lr2"],
         "cal_tune_scores": {"default": 0.50, "lr2": 0.55}},
        {"round": 1, "configs": ["default", "lr2"], "cal_tune_scores": {}},
    ]


def test_build_write_verify_roundtrip(tmp_path):
    rec = WC.build_winning_config(selected="default", knobs=KNOBS,
                                  rounds=_rounds(),
                                  cal_tune_scores={"default": 0.5})
    raw = WC.write_winning_config(tmp_path / WC.WINNING_CONFIG_FILENAME, rec)
    got = WC.verify_winning_config(tmp_path / WC.WINNING_CONFIG_FILENAME)
    assert got["selected"] == "default"
    assert got["amendment_4_lf_sha256"] == WC.cal_split.LF_SHA_AMENDMENT_4
    assert got["cal_split_lf_sha256"] == WC.cal_split.LF_SHA_CAL_SPLIT_A4
    assert raw == hashlib.sha256(
        (tmp_path / WC.WINNING_CONFIG_FILENAME).read_bytes()).hexdigest()


@pytest.mark.parametrize("bad", [
    {"learning_rate": 3.0, "epochs": 1.0, "rollout_loss_weight": 1.0,
     "contrast_loss_weight": 1.0},
    {"learning_rate": 1.0, "epochs": 2.0, "rollout_loss_weight": 1.0,
     "contrast_loss_weight": 1.0},
    {"learning_rate": 1.0, "epochs": 1.0, "rollout_loss_weight": 1.0},
])
def test_knob_grid_enforced(bad):
    with pytest.raises(WC.WinningConfigError):
        WC.build_winning_config(selected="default", knobs=bad, rounds=_rounds())


def test_selected_must_be_logged():
    with pytest.raises(WC.WinningConfigError):
        WC.build_winning_config(selected="nope", knobs=KNOBS,
                                rounds=_rounds())


def test_round_cap_enforced():
    rounds = [{"round": i, "configs": ["default"], "cal_tune_scores": {}}
              for i in range(WC.MAX_ROUNDS + 1)]
    with pytest.raises(WC.WinningConfigError):
        WC.build_winning_config(selected="default", knobs=KNOBS, rounds=rounds)


def test_verify_detects_tamper(tmp_path):
    rec = WC.build_winning_config(selected="default", knobs=KNOBS,
                                  rounds=_rounds())
    p = tmp_path / WC.WINNING_CONFIG_FILENAME
    WC.write_winning_config(p, rec)
    p.write_text(p.read_text(encoding="utf-8").replace("0.5", "9.9"),
                 encoding="utf-8")
    with pytest.raises(WC.WinningConfigError):
        WC.verify_winning_config(p)


def test_write_approval_a_m2_5_is_accepted_by_gate(tmp_path):
    rec = WC.build_winning_config(selected="default", knobs=KNOBS,
                                  rounds=_rounds(),
                                  cal_tune_scores={"default": 0.5})
    win = tmp_path / WC.WINNING_CONFIG_FILENAME
    WC.write_winning_config(win, rec)
    m1 = tmp_path / "member_1.pt"
    m1.write_bytes(b"member-1-bytes")
    appr_path = tmp_path / A.APPROVAL_A_M2_5_FILENAME
    WC.write_approval_a_m2_5(appr_path, winning_config_path=win,
                             m1_checkpoint_path=m1, owner="owner")
    got = A.require_a_m2_5(appr_path, m1_checkpoint_path=m1,
                           winning_config_path=win)
    assert got["stage_requested"] == "members-2-5"
    assert got["winning_config"]["selected"] == "default"


def test_write_approval_a_m2_5_binds_amendment_6(tmp_path):
    """The A_M2_5 writer emits the frozen A6 (state global token) binding."""
    rec = WC.build_winning_config(selected="default", knobs=KNOBS,
                                  rounds=_rounds(),
                                  cal_tune_scores={"default": 0.5})
    win = tmp_path / WC.WINNING_CONFIG_FILENAME
    WC.write_winning_config(win, rec)
    m1 = tmp_path / "member_1.pt"
    m1.write_bytes(b"member-1-bytes")
    appr_path = tmp_path / A.APPROVAL_A_M2_5_FILENAME
    body = WC.write_approval_a_m2_5(appr_path, winning_config_path=win,
                                    m1_checkpoint_path=m1, owner="owner")
    assert body["amendment_6_lf_sha256"] == A.LF_SHA_AMENDMENT_6


def test_require_a_m2_5_refuses_tampered_amendment_6(tmp_path, monkeypatch):
    rec = WC.build_winning_config(selected="default", knobs=KNOBS,
                                  rounds=_rounds(),
                                  cal_tune_scores={"default": 0.5})
    win = tmp_path / WC.WINNING_CONFIG_FILENAME
    WC.write_winning_config(win, rec)
    m1 = tmp_path / "member_1.pt"
    m1.write_bytes(b"member-1-bytes")
    appr_path = tmp_path / A.APPROVAL_A_M2_5_FILENAME
    WC.write_approval_a_m2_5(appr_path, winning_config_path=win,
                             m1_checkpoint_path=m1, owner="owner")
    a6 = tmp_path / "a6.md"
    a6.write_bytes(A.AMENDMENT_6_PATH.read_bytes() + b"tampered\n")
    monkeypatch.setattr(A, "AMENDMENT_6_PATH", a6)
    with pytest.raises(A.ApprovalMissing):
        A.require_a_m2_5(appr_path, m1_checkpoint_path=m1,
                         winning_config_path=win)
