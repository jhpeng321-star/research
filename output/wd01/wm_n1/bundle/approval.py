"""Owner approval-file gate.

The training STOPS after the timing gate until the owner's APPROVAL file is
present (REGISTRATION_WM_N1_AMENDMENT_1.md A1/A3).  This module is the only
place that reads that file, so a queue stage cannot start training without it.

The approval binds six frozen identities (LF-normalised, CRLF-safe):

* the ``TIMING_GATE.json`` sha256 written by the bundle timing wrapper (raw
  bytes, as written -- JSON is not line-ending normalised);
* ``REGISTRATION_WM_N1.md`` LF-sha256 ``666233aa...93246``;
* ``REGISTRATION_WM_N1_AMENDMENT_1.md`` LF-sha256 ``decb9656...16c57``;
* ``REGISTRATION_WM_N1_AMENDMENT_4_M1_TUNING.md`` LF-sha256
  ``d392b300...0822f`` (CAL-only selection; A4.2);
* ``CAL_SPLIT_A4.json`` LF-sha256 ``bc4e32f2...20d46``;
* ``REGISTRATION_WM_N1_AMENDMENT_6_STATE_TOKEN.md`` LF-sha256
  ``28dbc184...c948`` (state-only global token; A6).

The LF-sha256 convention (``\\r\\n`` -> ``\\n``) is imported from
:mod:`evalh.seals` so the queue and the eval harness can never disagree on a
frozen registration hash.

APPROVAL.json format::

    {
      "schema": "wm_n1_approval/1",
      "approved": true,
      "owner": "<owner name>",
      "approved_utc": "2026-09-30T00:00:00Z",
      "stage": "member-1",
      "timing_gate_sha256": "<64 hex>",
      "registration_lf_sha256": "666233aa...93246",
      "amendment_lf_sha256": "decb9656...16c57"
    }
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evalh import cal_split
from evalh.seals import LF_SHA_AMENDMENT_6, LF_SHA_WM_N1, lf_sha256

APPROVAL_SCHEMA = "wm_n1_approval/1"

#: The separate owner file that unlocks Idea A members 2-5 ("m1-first, then
#: STOP"): members 2-5 train only after member 1 has completed AND this file
#: binds the completed member-1 checkpoint plus the selected winning config.
APPROVAL_A_M2_5_SCHEMA = "wm_n1_approval_a_m2_5/1"
APPROVAL_A_M2_5_FILENAME = "APPROVAL_A_M2_5.json"
WINNING_CONFIG_FILENAME = "WINNING_CONFIG.json"

# Frozen LF-normalised sha256 of the training-order amendment (owner decision,
# 2026-09-30; commit 79e686e).  CRLF-normalised so an LF and a CRLF checkout
# hash identically.
LF_SHA_AMENDMENT_1 = ("decb96562ebfb549b56406d980be9c78bf4a377bd50c5eadeea"
                      "08aac16e16c57")

# Frozen Amendment 4 (CAL-only selection) LF-sha and the materialised CAL
# split LF-sha (see evalh.cal_split; A4_FREEZE.json).  The member-1 approval
# and the A_M2_5 gate both bind these, so no training, improvement-loop
# selection or members-2-5 step can start against an unbound split.
LF_SHA_AMENDMENT_4 = cal_split.LF_SHA_AMENDMENT_4
LF_SHA_CAL_SPLIT_A4 = cal_split.LF_SHA_CAL_SPLIT_A4

# Sibling locations of the frozen records (monkeypatchable for tests).
WM_N1_DIR = Path(__file__).resolve().parents[1]
REGISTRATION_PATH = WM_N1_DIR / "REGISTRATION_WM_N1.md"
AMENDMENT_PATH = WM_N1_DIR / "REGISTRATION_WM_N1_AMENDMENT_1.md"
AMENDMENT_4_PATH = WM_N1_DIR / cal_split.AMENDMENT_4_FILENAME
CAL_SPLIT_PATH = WM_N1_DIR / cal_split.CAL_SPLIT_FILENAME
AMENDMENT_6_PATH = WM_N1_DIR / "REGISTRATION_WM_N1_AMENDMENT_6_STATE_TOKEN.md"

TIMING_GATE_FILENAME = "TIMING_GATE.json"

_REQUIRED_FIELDS = ("timing_gate_sha256", "registration_lf_sha256",
                    "amendment_lf_sha256", "amendment_4_lf_sha256",
                    "cal_split_lf_sha256", "amendment_6_lf_sha256")


class ApprovalMissing(RuntimeError):
    """The owner approval file is absent or does not authorise the stage."""


def _load(path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise ApprovalMissing(f"owner approval file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                # noqa: BLE001
        raise ApprovalMissing(f"owner approval file is not JSON: {p}: {e}") \
            from e
    if not isinstance(data, dict):
        raise ApprovalMissing(f"owner approval file is not an object: {p}")
    return data


def _approved(data: dict) -> bool:
    return bool(data.get("approved"))


def approval_present(path) -> bool:
    """True iff ``path`` exists and records an owner approval.

    Structural check only (exists, parseable, ``approved`` truthy); the full
    frozen-hash binding is enforced by :func:`require_approval`.
    """
    try:
        return _approved(_load(path))
    except ApprovalMissing:
        return False


def _sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _actual(path) -> str:
    """LF-sha256 of an existing record, or raise ApprovalMissing."""
    p = Path(path)
    if not p.is_file():
        raise ApprovalMissing(
            f"required frozen record not found: {p} (place the frozen file "
            "next to this bundle or point APPROVAL timing_gate_path at it)")
    return lf_sha256(p)


def _timing_gate_path(approval_path, data: dict) -> Path:
    rel = data.get("timing_gate_path")
    base = Path(approval_path).resolve().parent
    if rel:
        p = Path(rel)
        return p if p.is_absolute() else (base / p)
    return base / TIMING_GATE_FILENAME


def _bind_amendment_4(data: dict) -> None:
    """Bind the frozen Amendment 4 LF-sha and the CAL split LF-sha.

    The declared fields must equal the frozen constants AND (when the records
    are present) the re-hashed files, so a CRLF checkout hashes identically and
    a stale/truncated split is refused rather than trusted.
    """
    if data["amendment_4_lf_sha256"] != LF_SHA_AMENDMENT_4:
        raise ApprovalMissing(
            "approval amendment_4_lf_sha256 does not match the frozen "
            f"{cal_split.AMENDMENT_4_FILENAME} LF-sha256 "
            f"{LF_SHA_AMENDMENT_4}")
    if data["cal_split_lf_sha256"] != LF_SHA_CAL_SPLIT_A4:
        raise ApprovalMissing(
            "approval cal_split_lf_sha256 does not match the frozen "
            f"{cal_split.CAL_SPLIT_FILENAME} LF-sha256 {LF_SHA_CAL_SPLIT_A4}")
    got_a4 = _actual(AMENDMENT_4_PATH)
    if got_a4 != LF_SHA_AMENDMENT_4:
        raise ApprovalMissing(
            f"{cal_split.AMENDMENT_4_FILENAME} at {AMENDMENT_4_PATH} "
            f"LF-sha256 {got_a4} != {LF_SHA_AMENDMENT_4}")
    got_split = _actual(CAL_SPLIT_PATH)
    if got_split != LF_SHA_CAL_SPLIT_A4:
        raise ApprovalMissing(
            f"{cal_split.CAL_SPLIT_FILENAME} at {CAL_SPLIT_PATH} LF-sha256 "
            f"{got_split} != {LF_SHA_CAL_SPLIT_A4}")


def _bind_amendment_6(data: dict) -> None:
    """Bind the frozen Amendment 6 (state global token) LF-sha.

    A6 is what lets ``state_global_token`` be ON; the declared field must equal
    the frozen constant AND the record, re-hashed CRLF-safe, must match, so an
    approval cannot authorise training against a missing/tampered freeze.
    """
    if data["amendment_6_lf_sha256"] != LF_SHA_AMENDMENT_6:
        raise ApprovalMissing(
            "approval amendment_6_lf_sha256 does not match the frozen "
            f"REGISTRATION_WM_N1_AMENDMENT_6_STATE_TOKEN.md LF-sha256 "
            f"{LF_SHA_AMENDMENT_6}")
    got = _actual(AMENDMENT_6_PATH)
    if got != LF_SHA_AMENDMENT_6:
        raise ApprovalMissing(
            f"REGISTRATION_WM_N1_AMENDMENT_6_STATE_TOKEN.md at "
            f"{AMENDMENT_6_PATH} LF-sha256 {got} != {LF_SHA_AMENDMENT_6}")


def require_approval(path, *, stage: str) -> dict:
    """Return the parsed approval or raise :class:`ApprovalMissing`.

    Refuses unless the file records an owner approval AND binds the actual
    timing-gate sha and both frozen registration LF-shas.  The frozen records
    are re-hashed from their configured paths (CRLF-safe), so a wrong or
    truncated copy is refused rather than trusted from the declared fields.
    """
    data = _load(path)
    if not _approved(data):
        raise ApprovalMissing(f"approval file {path} is not an owner approval")

    schema = data.get("schema")
    if schema is not None and schema != APPROVAL_SCHEMA:
        raise ApprovalMissing(
            f"approval schema {schema!r} != {APPROVAL_SCHEMA!r}")

    missing = [k for k in _REQUIRED_FIELDS if not data.get(k)]
    if missing:
        raise ApprovalMissing(
            f"approval {path} is missing binding fields: {missing}")

    # 1) registration + amendment: declared fields must equal the frozen
    #    constants AND (when the records are present) the re-hashed files.
    if data["registration_lf_sha256"] != LF_SHA_WM_N1:
        raise ApprovalMissing(
            "approval registration_lf_sha256 does not match the frozen "
            f"REGISTRATION_WM_N1.md LF-sha256 {LF_SHA_WM_N1}")
    if data["amendment_lf_sha256"] != LF_SHA_AMENDMENT_1:
        raise ApprovalMissing(
            "approval amendment_lf_sha256 does not match the frozen "
            f"REGISTRATION_WM_N1_AMENDMENT_1.md LF-sha256 "
            f"{LF_SHA_AMENDMENT_1}")
    got_reg = _actual(REGISTRATION_PATH)
    if got_reg != LF_SHA_WM_N1:
        raise ApprovalMissing(
            f"REGISTRATION_WM_N1.md at {REGISTRATION_PATH} LF-sha256 "
            f"{got_reg} != {LF_SHA_WM_N1}")
    got_amd = _actual(AMENDMENT_PATH)
    if got_amd != LF_SHA_AMENDMENT_1:
        raise ApprovalMissing(
            f"REGISTRATION_WM_N1_AMENDMENT_1.md at {AMENDMENT_PATH} "
            f"LF-sha256 {got_amd} != {LF_SHA_AMENDMENT_1}")

    # 1b) Amendment 4 (CAL-only selection) + the frozen CAL split.
    _bind_amendment_4(data)

    # 1c) Amendment 6 (state-only global token): every approval binds its
    #     frozen LF-sha, so training cannot start with the flag ON unbound.
    _bind_amendment_6(data)

    # 2) timing gate: declared field must equal the actual written file.
    gate_path = _timing_gate_path(path, data)
    if not Path(gate_path).is_file():
        raise ApprovalMissing(f"timing gate file not found: {gate_path}")
    got_gate = _sha256_file(gate_path)
    if data["timing_gate_sha256"] != got_gate:
        raise ApprovalMissing(
            f"APPROVAL.json timing_gate_sha256 {data['timing_gate_sha256']} "
            f"!= {gate_path} sha256 {got_gate}")

    return dict(data, approval_path=str(Path(path)), timing_gate_resolved=str(
        gate_path), stage_requested=stage)


_A_M2_5_REQUIRED = ("registration_lf_sha256", "amendment_lf_sha256",
                    "amendment_4_lf_sha256", "cal_split_lf_sha256",
                    "amendment_6_lf_sha256", "m1_checkpoint_sha256",
                    "winning_config_sha256")


def require_a_m2_5(path, *, m1_checkpoint_path, winning_config_path) -> dict:
    """Owner gate unlocking Idea A members 2-5 after member 1 completes.

    A SEPARATE owner file (schema ``wm_n1_approval_a_m2_5/1``) is required on
    top of the member-1 approval.  It binds the same registration and
    amendment-1 LF-shas as :func:`require_approval`, the frozen Amendment 4
    LF-sha and the CAL split LF-sha, PLUS the raw-bytes sha256 of the completed
    ``members/member_1.pt`` checkpoint and of the selected winner config
    (``WINNING_CONFIG.json``, itself schema-checked and sha-checked).  Members
    2-5 cannot start until the declared shas equal the actual bytes, so a stale
    or tampered winner is refused rather than trusted.
    """
    data = _load(path)
    if not _approved(data):
        raise ApprovalMissing(f"approval file {path} is not an owner approval")

    schema = data.get("schema")
    if schema is not None and schema != APPROVAL_A_M2_5_SCHEMA:
        raise ApprovalMissing(
            f"approval schema {schema!r} != {APPROVAL_A_M2_5_SCHEMA!r}")

    missing = [k for k in _A_M2_5_REQUIRED if not data.get(k)]
    if missing:
        raise ApprovalMissing(
            f"approval {path} is missing binding fields: {missing}")

    if data["registration_lf_sha256"] != LF_SHA_WM_N1:
        raise ApprovalMissing(
            "approval registration_lf_sha256 does not match the frozen "
            f"REGISTRATION_WM_N1.md LF-sha256 {LF_SHA_WM_N1}")
    if data["amendment_lf_sha256"] != LF_SHA_AMENDMENT_1:
        raise ApprovalMissing(
            "approval amendment_lf_sha256 does not match the frozen "
            f"REGISTRATION_WM_N1_AMENDMENT_1.md LF-sha256 "
            f"{LF_SHA_AMENDMENT_1}")
    got_reg = _actual(REGISTRATION_PATH)
    if got_reg != LF_SHA_WM_N1:
        raise ApprovalMissing(
            f"REGISTRATION_WM_N1.md at {REGISTRATION_PATH} LF-sha256 "
            f"{got_reg} != {LF_SHA_WM_N1}")
    got_amd = _actual(AMENDMENT_PATH)
    if got_amd != LF_SHA_AMENDMENT_1:
        raise ApprovalMissing(
            f"REGISTRATION_WM_N1_AMENDMENT_1.md at {AMENDMENT_PATH} "
            f"LF-sha256 {got_amd} != {LF_SHA_AMENDMENT_1}")

    # Amendment 4 (CAL-only selection) + frozen CAL split (CRLF-safe).
    _bind_amendment_4(data)

    # Amendment 6 (state-only global token): members 2-5 also bind the freeze.
    _bind_amendment_6(data)

    m1 = Path(m1_checkpoint_path)
    if not m1.is_file():
        raise ApprovalMissing(
            f"member-1 checkpoint not found: {m1} (members 2-5 need a "
            "completed member 1 before the A_M2_5 gate can match it)")
    got_m1 = _sha256_file(m1)
    if data["m1_checkpoint_sha256"] != got_m1:
        raise ApprovalMissing(
            f"APPROVAL_A_M2_5 m1_checkpoint_sha256 "
            f"{data['m1_checkpoint_sha256']} != {m1} sha256 {got_m1}")

    win = Path(winning_config_path)
    if not win.is_file():
        raise ApprovalMissing(
            f"winning config not found: {win} (place the selected "
            f"{WINNING_CONFIG_FILENAME} next to the member-1 run)")
    got_win = _sha256_file(win)
    if data["winning_config_sha256"] != got_win:
        raise ApprovalMissing(
            f"APPROVAL_A_M2_5 winning_config_sha256 "
            f"{data['winning_config_sha256']} != {win} sha256 {got_win}")
    # The bound winner must be a well-formed WINNING_CONFIG.json whose own
    # frozen-shas agree with the A_M2_5 bindings.
    from . import winning_config as wc

    try:
        rec = wc.verify_winning_config(win)
    except wc.WinningConfigError as exc:
        raise ApprovalMissing(
            f"APPROVAL_A_M2_5 winning config {win} is invalid: {exc}") from exc
    if rec.get("amendment_4_lf_sha256") != LF_SHA_AMENDMENT_4:
        raise ApprovalMissing(
            "winning config amendment_4_lf_sha256 does not match the frozen "
            f"Amendment 4 LF-sha256 {LF_SHA_AMENDMENT_4}")
    if rec.get("cal_split_lf_sha256") != LF_SHA_CAL_SPLIT_A4:
        raise ApprovalMissing(
            "winning config cal_split_lf_sha256 does not match the frozen "
            f"CAL split LF-sha256 {LF_SHA_CAL_SPLIT_A4}")

    return dict(data, approval_path=str(Path(path)),
                m1_checkpoint_resolved=str(m1),
                winning_config_resolved=str(win),
                winning_config=rec, stage_requested="members-2-5")
