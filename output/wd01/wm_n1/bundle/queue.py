"""GPU-box queue: the ordered, sequential WM-N1 run.

Stages (amendment A3), each recorded as a receipt and never repeated silently:

1. ``verify-upload``  -- re-check the uploaded pack against its manifest sha
   and REFUSE any chunk-5 / HOLDOUT shard (sealed).
2. ``convert+verify`` -- ``corpus.pack.convert`` then ``corpus.verify`` on the
   GPU box; when the upload is ALREADY a packed corpus (GPU boxes with
   ``upload``/``pack`` symlinked to the extracted ``pack_v3``) the real
   ``$WORK/pack`` is verified in place and never re-packed/overwritten.
3. ``timing-gate``    -- the REAL-member timing wrapper (:mod:`bundle.timing`);
   a hard STOP before any training (INTERFACES Mismatch 2).
4. ``approval``       -- ``bundle.approval.require_approval`` (owner file).
5. ``member-1``       -- ``train`` member 1 alone to completion (A1).
6. ``members-2-5``    -- same config, independent recorded seeds (A1); refuses
   unless member 1 already completed AND the separate owner file
   ``APPROVAL_A_M2_5.json`` binds the member-1 checkpoint sha and the winning
   config sha.  The M1-FIRST mode stops after stage 5 (``stop_after`` =
   ``"member-1"``).
7. ``idea-a-ensemble``  -- collects the member checkpoints and applies the
   final-test gate (``train.idea_a.require_full_ensemble``): a partial
   ensemble (k < 5) is recorded and refused here.
8. ``idea-a-held-out``  -- A3.4 held-out dry-run: builds the (i) J4->J4B,
   (ii) leave-PI-out and (iii) leave-highest-J4-wall-speed-out train/eval
   splits from the packed TRAIN corpus with the excluded arms ABSENT
   (``train.idea_a.held_out_dry_run``); CPU only, no training.

Every stage appends to ``<work_dir>/logs/<stage>.log`` and writes an
idempotent, resumable ``<work_dir>/state.json``: re-running skips completed
stages and never silently repeats a stage.  A failure raises and stops.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

from corpus import pack as corpus_pack
from corpus import verify as corpus_verify
from train import cli as train_cli
from train import config as train_config

from . import approval as approval_mod
from . import timing as timing_mod

STAGE_ORDER = ("verify-upload", "convert+verify", "timing-gate", "approval",
               "member-1", "members-2-5", "idea-a-ensemble",
               "idea-a-held-out")

STATE_FILENAME = "state.json"
LOG_DIRNAME = "logs"
UPLOAD_DIRNAME = "upload"
PACK_DIRNAME = "pack"
MEMBERS_DIRNAME = "members"
TIMING_GATE_FILENAME = approval_mod.TIMING_GATE_FILENAME
A_M2_5_APPROVAL_FILENAME = approval_mod.APPROVAL_A_M2_5_FILENAME
WINNING_CONFIG_FILENAME = approval_mod.WINNING_CONFIG_FILENAME

# Tests / CPU smoke runs flip this (or pass --tiny to the CLI).
TINY = False

_SEALED_TOKENS = ("holdout", "chunk-5", "chunk5", "chunk_5")


class BundleError(RuntimeError):
    """A bundle stage failed."""


class BundleOrderError(BundleError):
    """A stage was requested before its predecessors completed."""


class BundleStageError(BundleError):
    """A stage's own work failed (bad pack, missing checkpoint...)."""


class SealedShardRefusal(PermissionError):
    """A chunk-5 / HOLDOUT shard reached the upload manifest."""


# --------------------------------------------------------------------- util ---
def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _state_path(work_dir) -> Path:
    return Path(work_dir) / STATE_FILENAME


def _load_state(work_dir) -> dict:
    path = _state_path(work_dir)
    if not path.is_file():
        return {"schema": "wm_n1_bundle_state/1", "completed": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:                                # noqa: BLE001
        raise BundleError(f"corrupt bundle state {path}: {e}") from e
    data.setdefault("schema", "wm_n1_bundle_state/1")
    data.setdefault("completed", {})
    return data


def _save_state(work_dir, state) -> Path:
    path = _state_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_utc"] = _utc()
    body = json.dumps(state, indent=1).encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(body)
    os.replace(tmp, path)
    return path


def _log(work_dir, stage: str, message: str) -> None:
    log_dir = Path(work_dir) / LOG_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)
    line = f"{_utc()} [{stage}] {message}\n"
    with open(log_dir / f"{stage}.log", "a", encoding="utf-8") as fh:
        fh.write(line)
    print(f"[bundle:{stage}] {message}")


def _sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- manifest ----
def _load_manifest(pack_manifest, work_dir):
    """Return ``(manifest_dict, base_dir)`` from a path or dict."""
    if isinstance(pack_manifest, dict):
        base = Path(work_dir) if work_dir else Path.cwd()
        root = pack_manifest.get("root")
        return pack_manifest, (Path(root) if root else base)
    path = Path(pack_manifest)
    if not path.is_file():
        raise BundleStageError(f"upload manifest not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:                                # noqa: BLE001
        raise BundleStageError(f"upload manifest is not JSON: {path}: {e}") \
            from e
    root = data.get("root") if isinstance(data, dict) else None
    return data, (Path(root) if root else path.parent)


def _manifest_entries(manifest, base):
    """Yield ``(name, Path, expected_sha_or_None)`` for every manifest file."""
    entries = []
    shas = manifest.get("sha256") or {}
    files = manifest.get("files") or []
    for item in files:
        if isinstance(item, str):
            name, sha = item, shas.get(item)
        elif isinstance(item, dict):
            name = item.get("path") or item.get("name")
            sha = item.get("sha256") or shas.get(name)
        else:
            raise BundleStageError(f"bad manifest file entry: {item!r}")
        if name is None:
            raise BundleStageError(f"manifest entry without a path: {item!r}")
        p = Path(name)
        entries.append((str(name), p if p.is_absolute() else base / p, sha))
    return entries


def _is_sealed(name: str) -> bool:
    low = str(name).lower().replace("\\", "/")
    return any(tok in low for tok in _SEALED_TOKENS)


def _is_prepacked_corpus_dir(path) -> bool:
    """True iff ``path`` is an already-extracted packed corpus.

    A GPU box can arrive with ``upload``/``pack`` already symlinked to the
    extracted ``pack_v3`` tree.  Detecting the corpus ``INDEX.json`` schema
    (metadata only) lets ``convert+verify`` verify it in place instead of
    re-packing and overwriting the (possibly frozen) symlink target.
    """
    path = Path(path)
    index_p = path / "INDEX.json"
    if not (index_p.is_file() and (path / "manifest.json").is_file()):
        return False
    try:
        data = json.loads(index_p.read_text(encoding="utf-8"))
    except Exception:                                     # noqa: BLE001
        return False
    return isinstance(data, dict) and data.get("schema") == corpus_pack.PACK_SCHEMA


def _has_prepacked_markers(path) -> bool:
    """True iff ``path`` carries any pack-metadata marker file.

    Presence of ``INDEX.json`` / ``manifest.json`` / ``SHA256SUMS`` means the
    upload is (or claims to be) a packed corpus.  If such an upload is NOT a
    valid prepacked corpus the queue must fail closed instead of falling through
    to ``corpus.pack.convert`` (which would write through the pack target).
    """
    path = Path(path)
    return any((path / name).is_file()
               for name in ("INDEX.json", "manifest.json", "SHA256SUMS"))


def _is_link_or_reparse(path) -> bool:
    """True for a symlink or a Windows reparse point (e.g. a junction)."""
    p = Path(path)
    try:
        if p.is_symlink():
            return True
    except OSError:
        return True
    if os.name == "nt":
        try:
            st = os.lstat(p)
        except FileNotFoundError:
            return False
        except OSError:
            return True
        attrs = getattr(st, "st_file_attributes", 0)
        return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return False


def _pack_target_is_fresh(path) -> bool:
    """True iff ``path`` is a NEW/empty pack target safe to convert into.

    The legacy raw-shard flow may only write a pack that does not exist yet (or
    is an empty real directory).  An existing non-empty dir, a file, a symlink
    or a reparse point is refused so a frozen/prior target is never overwritten.
    """
    p = Path(path)
    if _is_link_or_reparse(p):
        return False
    if not p.exists():
        return True
    if not p.is_dir():
        return False
    return not any(p.iterdir())


def _refuse_sealed(manifest, entries) -> None:
    sealed = [name for name, _, _ in entries if _is_sealed(name)]
    declared = [str(x) for x in (manifest.get("refused") or [])]
    if sealed or declared:
        raise SealedShardRefusal(
            "upload manifest contains sealed shard(s); chunk-5 TEST and "
            f"HOLDOUT are never uploaded: {sealed + declared}")


# ------------------------------------------------------------------- stages ---
def _stage_verify_upload(pack_manifest, work_dir, device) -> dict:
    manifest, base = _load_manifest(pack_manifest, work_dir)
    if not isinstance(manifest, dict):
        raise BundleStageError("upload manifest is not an object")
    entries = _manifest_entries(manifest, base)
    _refuse_sealed(manifest, entries)
    missing, mismatch, verified = [], [], 0
    for name, path, sha in entries:
        if not path.is_file():
            missing.append(name)
            continue
        if sha is None:
            verified += 1
            continue
        got = _sha256_file(path)
        if got != sha:
            mismatch.append({"path": name, "expected": sha, "got": got})
        else:
            verified += 1
    if missing or mismatch:
        raise BundleStageError(
            f"upload verify failed: missing={missing} mismatch={mismatch}")
    return dict(stage="verify-upload", ok=True, n_files=len(entries),
                verified=verified, upload_dir=str(base),
                manifest_sha256=hashlib.sha256(
                    json.dumps(manifest, sort_keys=True).encode()).hexdigest())


def _stage_convert_verify(pack_manifest, work_dir, device) -> dict:
    """Convert (legacy) or verify in place (prepacked) the real corpus.

    GPU boxes can arrive with ``upload``/``pack`` already symlinked to the
    extracted ``pack_v3`` tree.  When the upload is already a packed corpus we
    NEVER call ``corpus.pack.convert`` (which would re-pack and overwrite the
    frozen symlink target): the real ``$WORK/pack`` is verified in place with
    the opaque-byte ``SHA256SUMS`` + ``INDEX.json``/``manifest.json`` K_max and
    format metadata checks.  Outcomes are never interpreted - only their
    presence is checked.

    An upload that carries prepacked markers but whose metadata is unreadable or
    not a valid pack FAILS CLOSED (never falls through to conversion).  The
    legacy raw-shard flow only converts into a NEW/empty, non-linked
    ``$WORK/pack``; an existing/non-empty/reparse/symlink target is refused so
    frozen or prior data is never written through.
    """
    work_dir = Path(work_dir)
    upload_dir = work_dir / UPLOAD_DIRNAME
    pack_dir = work_dir / PACK_DIRNAME
    if _is_prepacked_corpus_dir(upload_dir):
        if not _is_prepacked_corpus_dir(pack_dir):
            raise BundleStageError(
                f"prepacked upload at {upload_dir} but no extracted pack at "
                f"{pack_dir}; point $WORK/pack at the prepacked tree "
                "(symlink) rather than re-packing frozen data")
        verdict = corpus_verify.verify_pack(str(pack_dir))
        if not isinstance(verdict, dict) or not verdict.get("ok"):
            raise BundleStageError(f"pack verify failed: {verdict!r}")
        return dict(stage="convert+verify", ok=True, mode="prepacked-verified",
                    pack_dir=str(pack_dir), pack_source="prepacked",
                    upload_dir=str(upload_dir), convert=None, verify=verdict,
                    note=("prepacked corpus verified in place; no convert/"
                          "re-pack, frozen pack bytes untouched"))
    # The upload claims to be a packed corpus (markers present) but its INDEX/
    # manifest metadata is unreadable or not a valid pack: FAIL CLOSED rather
    # than fall through to conversion (which could write through the target).
    if _has_prepacked_markers(upload_dir):
        raise BundleStageError(
            f"upload {upload_dir} carries prepacked pack markers but its "
            "INDEX.json/manifest.json metadata is unreadable or is not a valid "
            f"{corpus_pack.PACK_SCHEMA} pack; refusing to fall through to "
            "conversion (malformed prepacked data fails closed)")
    # Legacy raw-shard flow: only ever convert into a NEW/empty, non-linked pack
    # target so an existing/non-empty/reparse/symlink target is never written.
    if not _pack_target_is_fresh(pack_dir):
        raise BundleStageError(
            f"refusing to convert into an existing/non-empty/linked pack "
            f"target {pack_dir}; the raw-shard flow needs a new (or empty, "
            "non-symlink) $WORK/pack so frozen/prior data is never overwritten")
    converted = corpus_pack.convert(str(upload_dir), str(pack_dir),
                                    workers=4)
    verdict = corpus_verify.verify_pack(str(pack_dir))
    if not isinstance(verdict, dict) or not verdict.get("ok"):
        raise BundleStageError(f"pack verify failed: {verdict!r}")
    return dict(stage="convert+verify", ok=True, pack_dir=str(pack_dir),
                pack_source="converted",
                convert=converted if isinstance(converted, dict) else None,
                verify=verdict)


def _stage_timing_gate(pack_manifest, work_dir, device) -> dict:
    """Registered gate on the REAL converted+verified pack.

    The queue has already run ``convert+verify`` into ``<work_dir>/pack``; that
    pack dir (and the requested device, CUDA allowed) is handed to
    :func:`bundle.timing.run_timing_gate`, so the gate measures the real Idea A
    corpus path (IdeaACorpus + pair-aware sampler + all five losses) rather
    than a synthetic pack.  The synthetic path is kept for ``--tiny`` (CPU
    tests / smoke).
    """
    work_dir = Path(work_dir)
    cfg = timing_mod.make_config(tiny=TINY)
    if not TINY:
        # real gate: host-scaled DataLoader workers (recorded in the gate)
        cfg.num_workers = None
    pack_dir = work_dir / PACK_DIRNAME
    kwargs = dict(device=device, tiny=TINY)
    if not TINY and (pack_dir / "manifest.json").is_file():
        kwargs["pack_dir"] = str(pack_dir)
    gate = timing_mod.run_timing_gate(cfg, **kwargs)
    path = work_dir / TIMING_GATE_FILENAME
    sha = timing_mod.write_timing_gate(path, gate)
    return dict(stage="timing-gate", ok=True, path=str(path), sha256=sha,
                passed=bool(gate.get("passed")),
                pack_source=gate.get("pack_source"),
                measured_model=gate.get("measured_model"),
                k_max=gate.get("k_max"),
                active_loss_terms=gate.get("active_loss_terms"),
                total_hours=gate.get("extrapolation", {}).get("total_hours"),
                next="STOP until the owner APPROVAL.json binds this sha256")


def _stage_approval(pack_manifest, work_dir, approval_path) -> dict:
    if approval_path is None:
        raise approval_mod.ApprovalMissing(
            "no approval path supplied (--approval is required)")
    appr = approval_mod.require_approval(approval_path, stage="member-1")
    return dict(stage="approval", ok=True, approval_path=str(approval_path),
                owner=appr.get("owner"),
                timing_gate_sha256=appr.get("timing_gate_sha256"))


def _members_dir(work_dir) -> Path:
    return Path(work_dir) / MEMBERS_DIRNAME


def _pack_dir(work_dir) -> Path:
    return Path(work_dir) / PACK_DIRNAME


def _train_member(member: int, seed: int, work_dir, device) -> dict:
    out_dir = _members_dir(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_dir = _pack_dir(work_dir)
    train_cli.record_seed(str(out_dir), member, seed)
    argv = ["--member", str(member), "--seed", str(seed),
            "--corpus", str(pack_dir), "--out", str(out_dir),
            "--device", str(device)]
    rc = train_cli.main(argv)
    if rc not in (0, None):
        raise BundleStageError(f"train member {member} exited {rc}")
    ckpt = out_dir / f"member_{member}.pt"
    if not ckpt.is_file():
        raise BundleStageError(
            f"train member {member} produced no checkpoint {ckpt}")
    return dict(member=member, seed=seed, checkpoint=str(ckpt))


def _stage_member_1(pack_manifest, work_dir, approval_path, device) -> dict:
    approval_mod.require_approval(approval_path, stage="member-1")
    receipts = [_train_member(1, int(train_config.DEFAULT_SEEDS[0]), work_dir,
                              device)]
    return dict(stage="member-1", ok=True, members=receipts)


def _stage_members_2_5(pack_manifest, work_dir, approval_path, device, *,
                       a_m2_5_approval=None, winning_config=None) -> dict:
    """Train Idea A members 2-5, gated by a SEPARATE owner file.

    The member-1 approval is unchanged; on top of it, this stage requires
    ``APPROVAL_A_M2_5.json`` binding the same registration/amendment-1 shas,
    the completed ``members/member_1.pt`` sha and the winning-config sha.
    """
    approval_mod.require_approval(approval_path, stage="members-2-5")
    first = _members_dir(work_dir) / "member_1.pt"
    if not first.is_file():
        raise BundleOrderError(
            "members 2-5 refused: member 1 has not completed "
            f"({first} is absent; amendment A1)")
    m2_5 = Path(a_m2_5_approval) if a_m2_5_approval is not None else (
        Path(work_dir) / A_M2_5_APPROVAL_FILENAME)
    win = Path(winning_config) if winning_config is not None else (
        Path(work_dir) / WINNING_CONFIG_FILENAME)
    approval_mod.require_a_m2_5(m2_5, m1_checkpoint_path=first,
                                winning_config_path=win)
    receipts = []
    for member in (2, 3, 4, 5):
        seed = int(train_config.DEFAULT_SEEDS[member - 1])
        receipts.append(_train_member(member, seed, work_dir, device))
    return dict(stage="members-2-5", ok=True, members=receipts,
                a_m2_5_approval=str(m2_5))


def _stage_idea_a_held_out(pack_manifest, work_dir, approval_path, device
                           ) -> dict:
    """A3.4 held-out dry-run over the packed TRAIN corpus (CPU, no training)."""
    from train.dataset import MemmapCorpus
    from train.idea_a import held_out_dry_run

    pack_dir = _pack_dir(work_dir)
    corpus = MemmapCorpus(pack_dir, split="TRAIN")
    report = held_out_dry_run(corpus)
    return dict(stage="idea-a-held-out", ok=True, pack_dir=str(pack_dir),
                tests=report)


def _stage_idea_a_ensemble(pack_manifest, work_dir, approval_path, device
                           ) -> dict:
    """Collect the member checkpoints; final-test gate refuses k < 5."""
    from train.idea_a import IdeaAConfigError, require_full_ensemble

    members_dir = _members_dir(work_dir)
    found = [m for m in (1, 2, 3, 4, 5)
             if (members_dir / f"member_{m}.pt").is_file()]
    k = len(found)
    if k == 0:
        raise BundleStageError(
            f"no member checkpoints under {members_dir}")
    try:
        members = require_full_ensemble(k)
    except IdeaAConfigError as exc:
        raise BundleStageError(str(exc)) from exc
    return dict(stage="idea-a-ensemble", ok=True, ensemble_k=k,
                members=list(members), found=found,
                final_test_eligible=(k == 5),
                note=("partial ensemble recorded; the final test refuses "
                      "k < 5" if k < 5 else "full ensemble"))


_STAGE_FUNCS = {
    "verify-upload": lambda m, w, a, d, x: _stage_verify_upload(m, w, d),
    "convert+verify": lambda m, w, a, d, x: _stage_convert_verify(m, w, d),
    "timing-gate": lambda m, w, a, d, x: _stage_timing_gate(m, w, d),
    "approval": lambda m, w, a, d, x: _stage_approval(m, w, a),
    "member-1": lambda m, w, a, d, x: _stage_member_1(m, w, a, d),
    "members-2-5": lambda m, w, a, d, x: _stage_members_2_5(m, w, a, d, **x),
    "idea-a-ensemble": lambda m, w, a, d, x: _stage_idea_a_ensemble(m, w, a, d),
    "idea-a-held-out": lambda m, w, a, d, x: _stage_idea_a_held_out(m, w, a, d),
}


def run_stage(stage: str, *, pack_manifest=None, work_dir=None,
              approval_path=None, device: str = "cuda",
              a_m2_5_approval=None, winning_config=None) -> dict:
    """Execute one named stage; raise on an out-of-order call.

    Idempotent/resumable: a stage already recorded complete in the state file
    is returned as-is and never re-run.
    """
    if stage not in STAGE_ORDER:
        raise BundleError(
            f"unknown stage {stage!r}; expected one of {STAGE_ORDER}")
    if work_dir is None:
        raise BundleError("work_dir is required")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state(work_dir)
    completed = state.get("completed", {})
    if stage in completed:
        _log(work_dir, stage, "already complete; skipping (idempotent)")
        return dict(completed[stage], skipped=True)

    idx = STAGE_ORDER.index(stage)
    missing = [s for s in STAGE_ORDER[:idx] if s not in completed]
    if missing:
        raise BundleOrderError(
            f"stage {stage!r} requested before {missing} completed")

    _log(work_dir, stage, f"start device={device}")
    extra = {"a_m2_5_approval": a_m2_5_approval,
             "winning_config": winning_config}
    try:
        receipt = _STAGE_FUNCS[stage](pack_manifest, work_dir, approval_path,
                                      device, extra)
    except Exception as e:                                # noqa: BLE001
        _log(work_dir, stage, f"FAILED: {type(e).__name__}: {e}")
        raise
    receipt["completed_utc"] = _utc()
    completed[stage] = receipt
    state["completed"] = completed
    state["last_stage"] = stage
    _save_state(work_dir, state)
    _log(work_dir, stage, "done")
    return receipt


def bundle_queue(pack_manifest, work_dir, approval_path, *, device: str = "cuda",
                 stop_after: str | None = None, a_m2_5_approval=None,
                 winning_config=None) -> dict:
    """Run the stages in order and return a per-stage receipt dict.

    ``stop_after=None`` means the hard default ``"timing-gate"``: the queue
    STOPS before the approval check, so no member can train until the owner
    approval exists.  Pass an explicit later stage (e.g. ``"member-1"`` for the
    M1-FIRST mode, or ``"members-2-5"``) to continue.  Never trains members 2-5
    before member 1; members 2-5 additionally need the separate
    ``APPROVAL_A_M2_5.json`` gate (``a_m2_5_approval`` + ``winning_config``).
    """
    stop = stop_after or "timing-gate"
    if stop not in STAGE_ORDER:
        raise BundleError(f"stop_after {stop!r} not in {STAGE_ORDER}")
    if work_dir is None:
        raise BundleError("work_dir is required")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    receipts = {}
    for stage in STAGE_ORDER[:STAGE_ORDER.index(stop) + 1]:
        receipts[stage] = run_stage(stage, pack_manifest=pack_manifest,
                                    work_dir=work_dir,
                                    approval_path=approval_path, device=device,
                                    a_m2_5_approval=a_m2_5_approval,
                                    winning_config=winning_config)
    state = _load_state(work_dir)
    state["stop_after"] = stop
    _save_state(work_dir, state)
    return dict(work_dir=str(work_dir), stop_after=stop,
                stopped_after=(stop if stop in receipts else None),
                stages=receipts,
                stage_order=list(STAGE_ORDER))
