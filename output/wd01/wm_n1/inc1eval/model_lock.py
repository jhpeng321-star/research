"""INC1 model lock (registration section 6).

Before any incident simulation runs, ``INC1_MODEL_LOCK.json`` must record,
and the runner must check, the SHA-256 of every loaded model member, the pack
manifest SHA, the code commit and the Part A v2g parameter SHA.

The lock is data, not a capability: :func:`verify_lock` re-hashes exactly the
files the caller says it loaded and refuses on any mismatch, so an eval cannot
silently run against a swapped checkpoint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional

LOCK_NAME = "INC1_MODEL_LOCK.json"
DONE_NAME = "INC1_EVAL_DONE.json"

REQUIRED_KEYS = ("members", "pack_sha", "code_commit", "v2g_params_sha")


class ModelLockError(RuntimeError):
    """A missing, stale or mismatched INC1 model lock."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(Path(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_lock(*, members: Mapping[str, object], pack_manifest: object,
               code_commit: str, v2g_params: object,
               extra: Optional[Mapping] = None) -> dict:
    """Build the lock payload from paths or pre-computed values.

    ``members`` maps a member name (``"member_1.pt"`` ...) to a path or an
    already-computed sha.  ``pack_manifest`` / ``v2g_params`` are paths or
    shas.  ``code_commit`` is the full git commit.
    """
    def _sha(value) -> str:
        s = str(value)
        if len(s) == 64 and all(c in "0123456789abcdef" for c in s.lower()):
            return s.lower()
        return sha256_file(value)

    payload = dict(
        schema="wm_n1_inc1_model_lock/1",
        members={str(k): _sha(v) for k, v in dict(members).items()},
        pack_sha=_sha(pack_manifest),
        code_commit=str(code_commit),
        v2g_params_sha=_sha(v2g_params),
    )
    if extra:
        payload["extra"] = dict(extra)
    return payload


def write_lock(out_dir, payload: Mapping) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / LOCK_NAME
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True)
                    + "\n", encoding="utf-8")
    return path


def load_lock(out_dir) -> dict:
    path = Path(out_dir) / LOCK_NAME
    if not path.is_file():
        raise ModelLockError(
            f"missing {LOCK_NAME} in {out_dir}: the INC1 runner refuses to "
            "start before the model lock is written (registration section 6)")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_lock(lock: Mapping, *, members: Mapping[str, object],
                pack_manifest: object, v2g_params: object,
                code_commit: Optional[str] = None) -> list:
    """Return the list of mismatch messages (empty == verified)."""
    errors: list = []
    for key in REQUIRED_KEYS:
        if key not in lock:
            errors.append(f"lock missing required key {key!r}")
    for name, ref in dict(members).items():
        got = sha256_file(ref)
        want = (lock.get("members") or {}).get(str(name))
        if want is None:
            errors.append(f"member {name!r} not present in lock")
        elif want.lower() != got:
            errors.append(
                f"member {name!r} sha mismatch: lock {want} loaded {got}")
    got_pack = sha256_file(pack_manifest)
    if str(lock.get("pack_sha", "")).lower() != got_pack:
        errors.append(
            f"pack sha mismatch: lock {lock.get('pack_sha')} loaded {got_pack}")
    got_v2g = sha256_file(v2g_params)
    if str(lock.get("v2g_params_sha", "")).lower() != got_v2g:
        errors.append(
            f"v2g params sha mismatch: lock {lock.get('v2g_params_sha')} "
            f"loaded {got_v2g}")
    if code_commit is not None and str(lock.get(
            "code_commit", "")) != str(code_commit):
        errors.append(
            f"code commit mismatch: lock {lock.get('code_commit')} "
            f"loaded {code_commit}")
    return errors


def require_lock(out_dir, *, members: Mapping[str, object],
                 pack_manifest: object, v2g_params: object,
                 code_commit: Optional[str] = None) -> dict:
    """Load and verify the lock; raise :class:`ModelLockError` on mismatch."""
    lock = load_lock(out_dir)
    errors = verify_lock(lock, members=members, pack_manifest=pack_manifest,
                         v2g_params=v2g_params, code_commit=code_commit)
    if errors:
        raise ModelLockError("; ".join(errors))
    return lock
