"""Fail-closed checkpoint binding for the WM-N1 CAL-tune scorer (E4).

:func:`load_scoring_checkpoint` binds the trained Idea A member-1 checkpoint
(``members/member_1/member_1.pt``) for scoring against the CAL-tune half:

1. The raw file bytes are SHA-256 hashed BEFORE any ``torch.load``; a
   mismatch raises :class:`CheckpointRefusal` with NOTHING deserialized.
2. The payload must declare amendment 6 (A6) ``state_global_token`` ON; an
   OFF/absent flag is refused, and a recorded ``amendment_6_lf_sha256``
   binding must equal the frozen A6 LF-sha when present.
3. Contract 3: ``model.constants.CORRECTION_WINDOW_STEPS`` must be 20 and the
   checkpoint's recorded training ``rollout_horizon_steps`` (when recorded in
   its config) must equal 20.  The correction head consumes exactly that
   trained 20-step window; any LONGER dynamics rollout this scorer reports is
   labelled :data:`DYNAMICS_ROLLOUT_LABEL` (``"extrapolation"``).
4. The bundle itself is rebuilt ONLY through ``wm_api.checkpoint``'s member
   builder (``_build_member``), which reuses
   ``train.idea_a.build_idea_a_bundle``; no architecture is duplicated here.

The returned :class:`ScoringCheckpoint` carries ``bundle`` usable by
``evalh.idea_a.IdeaAVariantPredictor(bundle=...)`` and by
``wm_api.engine.outcome_head(bundle, ...)``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

#: The pinned member-1 checkpoint bytes (shared context, 2026-10-03).
EXPECTED_MEMBER_SHA256 = (
    "94984cc09a2d9850f1731a7cd197255401ab80ad593a869ff9aeba680d1de92a")

#: Label for any reported dynamics rollout longer than the trained
#: correction window (contract 3): it is extrapolation, not in-window scoring.
DYNAMICS_ROLLOUT_LABEL = "extrapolation"

_TRAINED_WINDOW_STEPS = 20


class CheckpointRefusal(RuntimeError):
    """The member checkpoint was refused before or during binding."""


@dataclass(frozen=True)
class ScoringCheckpoint:
    """One bound scoring member: its bundle and the validated identity."""

    bundle: object             # train.idea_a.IdeaABundle (float64, eval mode)
    member: int
    seed: int
    sha256: str                # actual raw-bytes sha256 of the member file
    path: str
    config: dict               # the checkpoint's recorded training config
    flags: dict                # ensemble_k / members / partial / final_test...
    amendment_6_lf_sha256: object  # recorded binding, or None when absent
    dynamics_rollout_label: str = DYNAMICS_ROLLOUT_LABEL


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _expected(expected_sha256) -> str:
    pin = str(expected_sha256)
    if (len(pin) != 64 or pin != pin.lower()
            or any(c not in "0123456789abcdef" for c in pin)):
        raise CheckpointRefusal(
            "expected_sha256 must be an exact lowercase SHA-256")
    return pin


def load_scoring_checkpoint(member_path, *,
                            expected_sha256=EXPECTED_MEMBER_SHA256
                            ) -> ScoringCheckpoint:
    """Bind one member checkpoint for scoring; refuse everything else.

    Hashes the raw bytes before any deserialization, refuses on a pin
    mismatch with nothing loaded, requires A6 ``state_global_token`` ON and
    the recorded amendment-6 binding (if present), asserts the trained
    correction window is 20 steps (contract 3), and rebuilds the bundle via
    ``wm_api.checkpoint``'s member builder.
    """
    pin = _expected(expected_sha256)
    path = Path(member_path)
    if not path.is_file():
        raise CheckpointRefusal(f"member checkpoint missing: {path}")
    digest = _sha256_file(path)
    if digest != pin:
        raise CheckpointRefusal(
            f"member checkpoint sha256 {digest} != expected {pin}; refused "
            "before any deserialization")
    from model import constants as _MC

    if int(_MC.CORRECTION_WINDOW_STEPS) != _TRAINED_WINDOW_STEPS:
        raise CheckpointRefusal(
            f"model.constants.CORRECTION_WINDOW_STEPS is "
            f"{_MC.CORRECTION_WINDOW_STEPS}, not {_TRAINED_WINDOW_STEPS}; "
            "contract 3 drifted")
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    from wm_api.checkpoint import CHECKPOINT_FORMAT, _build_member

    if not isinstance(payload, dict) or payload.get("format") != \
            CHECKPOINT_FORMAT:
        got = payload.get("format") if isinstance(payload, dict) else \
            type(payload).__name__
        raise CheckpointRefusal(f"member payload format {got!r} is not "
                                f"{CHECKPOINT_FORMAT!r}")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise CheckpointRefusal("member payload has no 'config' mapping")
    top_flag = payload.get("state_global_token")
    cfg_flag = config.get("state_global_token")
    if top_flag is not None and cfg_flag is not None \
            and bool(top_flag) != bool(cfg_flag):
        raise CheckpointRefusal(
            "checkpoint records conflicting state_global_token values "
            f"(payload {top_flag!r} vs config {cfg_flag!r})")
    flag = top_flag if top_flag is not None else cfg_flag
    if flag is not True:
        raise CheckpointRefusal(
            f"A6 state_global_token must be ON for the scoring checkpoint; "
            f"got {flag!r}")
    recorded_a6 = payload.get("amendment_6_lf_sha256")
    if recorded_a6 is not None:
        from evalh.seals import LF_SHA_AMENDMENT_6

        if str(recorded_a6) != LF_SHA_AMENDMENT_6:
            raise CheckpointRefusal(
                f"recorded amendment-6 binding {recorded_a6!r} != frozen "
                f"{LF_SHA_AMENDMENT_6!r}")
    steps = config.get("rollout_horizon_steps")
    if steps is not None and int(steps) != int(_MC.CORRECTION_WINDOW_STEPS):
        raise CheckpointRefusal(
            f"checkpoint recorded training rollout_horizon_steps="
            f"{steps!r}, expected {_MC.CORRECTION_WINDOW_STEPS}; the "
            f"trained correction window is contract 3 and longer dynamics "
            f"rollouts are reported separately, labelled "
            f"{DYNAMICS_ROLLOUT_LABEL!r}")
    try:
        member = _build_member(payload, path)
    except CheckpointRefusal:
        raise
    except Exception as exc:                             # noqa: BLE001
        raise CheckpointRefusal(
            f"member builder refused {path.name}: {type(exc).__name__}: "
            f"{exc}") from exc
    return ScoringCheckpoint(
        bundle=member.bundle, member=int(member.member),
        seed=int(member.seed), sha256=digest, path=str(path),
        config=dict(config),
        flags={key: payload.get(key) for key in
               ("ensemble_k", "members", "partial", "final_test_eligible")},
        amendment_6_lf_sha256=(None if recorded_a6 is None
                               else str(recorded_a6)))
