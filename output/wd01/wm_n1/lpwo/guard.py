"""Workspace-only wrapper around the explicitly staged canonical holdout guard."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[4]
DENIED = re.compile(r"(?:^|[^a-z0-9])cal(?:ibration)?(?:$|[^a-z0-9])|"
                    r"holdout|sealed|chunk[-_ ]?5|j4b|inc1", re.I)


class GuardError(ValueError):
    """A source is forbidden, unverifiable, or outside the explicit workspace."""


def workspace_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise GuardError(f"path outside workspace: {path}")
    return path


class DataGuard:
    def __init__(self, spec, tool, allowed_roots):
        # These two governance files are the sole permitted holdout-named inputs.
        self.spec = workspace_path(spec)
        self.tool = workspace_path(tool)
        self.allowed_roots = tuple(workspace_path(p) for p in allowed_roots)
        if not self.allowed_roots:
            raise GuardError("explicit allowed roots are required")
        self.data = json.loads(self.spec.read_text(encoding="utf-8"))
        if self.data.get("schema") != "holdout_exclusions/1":
            raise GuardError("expected canonical holdout_exclusions/1 metadata")
        self.excluded = {str(r["world"]) for r in self.data["excluded"]}
        module_spec = importlib.util.spec_from_file_location("lpwo_canonical_guard", self.tool)
        if module_spec is None or module_spec.loader is None:
            raise GuardError("canonical guard cannot be imported")
        self.external = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(self.external)
        if not callable(getattr(self.external, "assert_final_holdout_eligible", None)):
            raise GuardError("canonical guard lacks assert_final_holdout_eligible(worlds, data)")
        self.paths, self.worlds, self.refusals = set(), set(), []
        self.identities = {"spec_sha256": hashlib.sha256(self.spec.read_bytes()).hexdigest(),
                           "tool_sha256": hashlib.sha256(self.tool.read_bytes()).hexdigest()}

    def _deny(self, value):
        text = str(value)
        if DENIED.search(text) or any(w and w in text for w in self.excluded):
            self.refusals.append(text)
            raise GuardError(f"forbidden source identity: {text}")

    def check_path(self, value):
        self._deny(value)
        path = workspace_path(value)
        self._deny(path)
        if not any(path.is_relative_to(root) for root in self.allowed_roots):
            raise GuardError(f"path not in allowed roots: {path}")
        self.paths.add(str(path))
        return True

    def check_record(self, record):
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
        self.external.assert_final_holdout_eligible([str(world)], self.data)
        self.worlds.add(str(world))
        return True

    def read_json(self, path):
        self.check_path(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def audit(self):
        return {"schema": "lpwo_guard_audit/1", **self.identities,
                "workspace_only": True, "canonical_guard_applied": True,
                "excluded_world_count": len(self.excluded),
                "checked_paths": sorted(self.paths), "checked_worlds": sorted(self.worlds),
                "refusals": self.refusals, "n_refusals": len(self.refusals)}
