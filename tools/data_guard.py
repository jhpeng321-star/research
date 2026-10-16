"""Workspace guard for TRAIN-only data access (port of the legacy ``lpwo/guard.py``).

Refuses any path or record identity that carries a restricted token (sealed, CAL,
HOLDOUT, chunk-5, J4B, INC1) or whose world is listed in the explicit exclusion spec.
``DataGuard`` is the object the window-list builders (:mod:`wms.data.window_lists`) call
for every file they open; :func:`assert_unrestricted` is the stateless check used by tests
and the data registry.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
DENIED = re.compile(r"(?:^|[^a-z0-9])cal(?:ibration)?(?:$|[^a-z0-9])|"
                    r"holdout|sealed|chunk[-_ ]?5|j4b|inc1", re.I)
SPEC_SCHEMA = "holdout_exclusions/1"


class GuardError(ValueError):
    """A source is forbidden, unverifiable, or outside the explicit workspace."""


def assert_unrestricted(value) -> None:
    """Raise :class:`GuardError` if ``value`` (path or identity text) names restricted data."""
    text = str(value)
    if DENIED.search(text):
        raise GuardError(f"forbidden source identity: {text}")


def workspace_path(value, workspace: Path = WORKSPACE) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(workspace):
        raise GuardError(f"path outside workspace: {path}")
    return path


class DataGuard:
    """Path/record guard bound to an exclusion spec and explicit allowed roots."""

    def __init__(self, spec, allowed_roots, *, workspace: Path = WORKSPACE):
        self.workspace = Path(workspace).resolve()
        # The spec is the sole permitted restricted-named input (read, never traversed).
        self.spec = workspace_path(spec, self.workspace)
        self.allowed_roots = tuple(workspace_path(p, self.workspace) for p in allowed_roots)
        if not self.allowed_roots:
            raise GuardError("explicit allowed roots are required")
        self.data = json.loads(self.spec.read_text(encoding="utf-8"))
        if self.data.get("schema") != SPEC_SCHEMA:
            raise GuardError(f"expected canonical {SPEC_SCHEMA} metadata")
        self.excluded = {str(r["world"]) for r in self.data["excluded"]}
        self.paths: set[str] = set()
        self.worlds: set[str] = set()
        self.refusals: list[str] = []
        self.identities = {"spec_sha256": hashlib.sha256(self.spec.read_bytes()).hexdigest()}

    def _deny(self, value) -> None:
        text = str(value)
        if DENIED.search(text) or any(w and w in text for w in self.excluded):
            self.refusals.append(text)
            raise GuardError(f"forbidden source identity: {text}")

    def check_path(self, value) -> bool:
        self._deny(value)
        path = workspace_path(value, self.workspace)
        self._deny(path)
        if not any(path.is_relative_to(root) for root in self.allowed_roots):
            raise GuardError(f"path not in allowed roots: {path}")
        self.paths.add(str(path))
        return True

    def check_record(self, record) -> bool:
        if str(record.get("split", "")).upper() != "TRAIN":
            raise GuardError("source must explicitly declare TRAIN")
        if str(record.get("chunk", "")) == "5":
            raise GuardError("chunk 5 is forbidden")
        for field in ("world", "root_id", "key", "run", "batch", "chunk", "root_rel",
                      "run_dir", "pack", "source_arm"):
            if record.get(field) is not None:
                self._deny(record[field])
        world = record.get("world") or record.get("root_id")
        if not world:
            raise GuardError("source needs a world identity")
        if str(world) in self.excluded:
            raise GuardError(f"world is excluded by the spec: {world}")
        self.worlds.add(str(world))
        return True

    def read_json(self, path):
        self.check_path(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def audit(self) -> dict:
        return {"schema": "lpwo_guard_audit/1", **self.identities,
                "workspace_only": True, "excluded_world_count": len(self.excluded),
                "checked_paths": sorted(self.paths), "checked_worlds": sorted(self.worlds),
                "refusals": self.refusals, "n_refusals": len(self.refusals)}
