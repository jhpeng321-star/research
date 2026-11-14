"""Idea A member-checkpoint directory loading with sha256 checks.

Directory layout (api_version 1.0)::

    ckpt/
      member_1.pt .. member_k.pt    train.IdeaATrainer payloads
                                    (format "wm_n1_idea_a_member/1")
      member_k.json                 sidecars (optional, never required)
      ood_model.json                optional OOD density model
                                    {"mean": [...], "std": [...]}
      SHA256SUMS                    optional; "<sha256>  <name>" lines

Loading rebuilds each member's composed Idea A bundle through
``train.idea_a.build_idea_a_bundle`` from the payload's own config, loads
the flat namespaced ``state_dict`` (``hdv.*`` / ``arrival.*`` / ``corr.*``,
the trainer's contract) and casts the learned modules to float64 — the
``rollout2`` engine's internal math dtype.  ``torch.load`` always sets
``weights_only=True``.

Refusals: a missing directory or member file, an unknown payload format, a
state-dict mismatch, and any SHA256SUMS mismatch or listed-file absence.
Computed member hashes are always returned (``LoadedCheckpoint.shas``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from .errors import CheckpointError, CheckpointNotFoundError, \
    CheckpointShaMismatch

#: The trainer's frozen member-checkpoint format tag.
CHECKPOINT_FORMAT = "wm_n1_idea_a_member/1"

#: Namespaced module prefixes inside the flat state_dict (train.idea_a).
#: ``state.`` is the M1FIX2 state-only global-token encoder; ``paired.`` is the
#: optional climatology-anchored paired-effect head (plan-aware or not).  Each
#: is loaded when the rebuilt bundle has the matching module.
MODULE_PREFIXES = ("hdv.", "arrival.", "corr.", "state.", "paired.")


def sha256_of(path: Path) -> str:
    """sha256 hex digest of a file's bytes (opaque preservation only)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class MemberModules:
    """One loaded ensemble member (its composed Idea A bundle)."""

    member: int                # 1-based member number (trainer convention)
    seed: int
    bundle: object             # train.idea_a.IdeaABundle (float64 modules)
    file: str
    sha256: str


@dataclass
class LoadedCheckpoint:
    """A fully loaded member-checkpoint directory."""

    directory: str
    members: list              # MemberModules, sorted by member number
    shas: dict                 # {file name: sha256}
    config: dict
    flags: dict                # ensemble_k / members / partial / ...
    ood_model: Optional[dict] = None
    sha_verified: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def n_members(self) -> int:
        return len(self.members)


def _load_member_payload(path: Path) -> dict:
    if not path.is_file():
        raise CheckpointNotFoundError(f"member checkpoint missing: {path}")
    try:
        payload = torch.load(str(path), map_location="cpu",
                             weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"cannot load {path}: {exc}") from exc
    if not isinstance(payload, dict) or \
            payload.get("format") != CHECKPOINT_FORMAT:
        got = payload.get("format") if isinstance(payload, dict) else \
            type(payload).__name__
        raise CheckpointError(
            f"{path}: format {got!r} != {CHECKPOINT_FORMAT!r}")
    return payload


#: Namespaced prefix -> bundle attribute (``state.`` is ``state_enc``,
#: ``paired.`` is ``paired_effect``).
_PREFIX_ATTR = {"hdv.": "hdv", "arrival.": "arrival", "corr.": "corr",
                "state.": "state_enc", "paired.": "paired_effect"}


def _bundle_module(bundle, prefix: str):
    return getattr(bundle, _PREFIX_ATTR.get(prefix, prefix[:-1]), None)


def _build_member(payload: dict, path: Path) -> MemberModules:
    from train.idea_a import IdeaATrainConfig, build_idea_a_bundle

    config = payload.get("config")
    if not isinstance(config, dict):
        raise CheckpointError(f"{path}: payload has no 'config' mapping")
    try:
        cfg = IdeaATrainConfig.from_dict(config)
    except Exception as exc:
        raise CheckpointError(
            f"{path}: config does not rebuild an IdeaATrainConfig: "
            f"{exc}") from exc
    bundle = build_idea_a_bundle(cfg)
    flat = payload.get("state_dict")
    if not isinstance(flat, dict) or not flat:
        raise CheckpointError(f"{path}: payload has no 'state_dict'")
    for name in MODULE_PREFIXES:
        module = _bundle_module(bundle, name)
        if not isinstance(module, torch.nn.Module):
            continue
        if name.startswith("arrival"):
            continue              # the P4 CounterEncoder runs float32
        module.to(torch.float64)
    missing, stray = [], []
    for prefix in MODULE_PREFIXES:
        module = _bundle_module(bundle, prefix)
        own = {k[len(prefix):]: v for k, v in flat.items()
               if k.startswith(prefix)}
        if not isinstance(module, torch.nn.Module):
            if own:
                raise CheckpointError(
                    f"{path}: checkpoint has {prefix} weights but the "
                    "configured bundle has no matching module")
            continue
        if not own and module.state_dict():
            missing.append(prefix[:-1])
            continue
        try:
            module.load_state_dict(own)
        except Exception as exc:
            raise CheckpointError(
                f"{path}: state_dict mismatch for {prefix[:-1]}: "
                f"{exc}") from exc
    stray = [k for k in flat
             if not any(k.startswith(p) for p in MODULE_PREFIXES)]
    if missing or stray:
        raise CheckpointError(
            f"{path}: checkpoint mismatch; missing_modules={missing} "
            f"unknown_keys={stray[:5]}")
    for name in MODULE_PREFIXES:
        module = _bundle_module(bundle, name)
        if isinstance(module, torch.nn.Module):
            module.eval()
    norm_payload = payload.get("outcome_norm")
    if isinstance(norm_payload, dict):
        try:
            from model.outcome_norm import OutcomeNorm

            bundle.outcome_norm = OutcomeNorm.from_dict(norm_payload)
        except Exception as exc:                            # noqa: BLE001
            raise CheckpointError(
                f"{path}: cannot rebuild OUTCOME_NORM: {exc}") from exc
    # The external input is raw16; cfg.summary_dim is the trained internal
    # projection width (normally 64), not the external input width.
    from corr.evaluation import trained_correction

    try:
        trained_correction(bundle)
    except ValueError as exc:
        raise CheckpointError(
            f"{path}: unsupported correction composition: {exc}") from exc
    return MemberModules(member=int(payload.get("member", 0)),
                         seed=int(payload.get("seed", 0)),
                         bundle=bundle, file=path.name,
                         sha256=sha256_of(path))


def _read_sha_sums(ckpt_dir: Path) -> dict:
    sums = {}
    for line in (ckpt_dir / "SHA256SUMS").read_text(encoding="utf-8")\
            .splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise CheckpointError(
                f"SHA256SUMS: malformed line {line!r}")
        sums[parts[1].strip().lstrip("*")] = parts[0].strip().lower()
    return sums


def load_checkpoint(checkpoint, *, verify_sha: bool = True,
                    ood_model: bool = True) -> LoadedCheckpoint:
    """Load a member-checkpoint directory (refusing every broken case)."""
    ckpt_dir = Path(checkpoint)
    if not ckpt_dir.is_dir():
        raise CheckpointNotFoundError(
            f"checkpoint directory not found: {ckpt_dir}")
    member_files = sorted(
        ckpt_dir.glob("member_*.pt"),
        key=lambda p: (int(p.stem.split("_")[1]), p.name))
    if not member_files:
        raise CheckpointNotFoundError(
            f"checkpoint directory {ckpt_dir} holds no member_*.pt "
            "payloads")
    sums_path = ckpt_dir / "SHA256SUMS"
    sums = _read_sha_sums(ckpt_dir) if (verify_sha and sums_path.is_file()) \
        else None
    members = []
    for path in member_files:
        if sums is not None:
            listed = sums.get(path.name)
            if listed is None:
                raise CheckpointError(
                    f"{path.name} is not listed in SHA256SUMS")
            actual = sha256_of(path)
            if actual != listed:
                raise CheckpointShaMismatch(
                    f"{path.name}: sha256 {actual} != recorded {listed}")
        members.append(_build_member(_load_member_payload(path), path))
    if sums is not None:
        for name, listed in sorted(sums.items()):
            path = ckpt_dir / name
            if not path.is_file():
                raise CheckpointNotFoundError(
                    f"SHA256SUMS lists missing file {name!r}")
            actual = sha256_of(path)
            if actual != listed:
                raise CheckpointShaMismatch(
                    f"{name}: sha256 {actual} != recorded {listed}")
    numbers = [m.member for m in members]
    if numbers != sorted(set(numbers)):
        raise CheckpointError(
            f"member numbers must be unique and sorted: {numbers}")
    if any(m.member <= 0 for m in members):
        raise CheckpointError(f"member numbers must be 1-based: {numbers}")
    flags = {}
    if members:
        flags = {
            "ensemble_k": len(members),
            "members": numbers,
            "partial": len(members) < 5,
            "final_test_eligible": len(members) == 5,
        }
    ood = None
    ood_path = ckpt_dir / "ood_model.json"
    if ood_model and ood_path.is_file():
        try:
            ood = json.loads(ood_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CheckpointError(
                f"cannot parse {ood_path}: {exc}") from exc
        if not isinstance(ood, dict) or "mean" not in ood or "std" not in ood:
            raise CheckpointError(
                f"{ood_path}: expected a density model with 'mean'/'std'")
    config = members[0].bundle.config.to_dict() if members else {}
    return LoadedCheckpoint(
        directory=str(ckpt_dir), members=members,
        shas={m.file: m.sha256 for m in members}, config=config,
        flags=flags, ood_model=ood, sha_verified=sums is not None,
        meta=dict(n_member_files=len(member_files),
                  sha_sums_present=bool(sums_path.is_file())))
