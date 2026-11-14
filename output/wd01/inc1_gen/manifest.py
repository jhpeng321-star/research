"""``INC1_GENERATORS.json`` writer (addendum 2, A2.6).

The INC1 launcher refuses unless the shared R5/R6 generator module's sha
matches the sha recorded here, and the manifest is recorded BEFORE the model
lock.  It binds the LF-normalised sha256 (the repository ``lf_sha256``
convention) of every TEXT package module file plus that of
``REGISTRATION_WM_N1_INC1_ADDENDUM_2.md``.  Binary modules keep their raw
sha256.  Text is normalised so a ``core.autocrlf=true`` checkout verifies
against the LF shas recorded on a Linux tree.

The default write is write-once (an existing manifest is never overwritten);
pass ``overwrite=True`` only while authoring the package.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_SCHEMA = "wm_n1_inc1_generators/1"

PACKAGE_ROOT = Path(__file__).resolve().parent
WD01_ROOT = PACKAGE_ROOT.parent
ADDENDUM_PATH = (WD01_ROOT / "wm_n1"
                 / "REGISTRATION_WM_N1_INC1_ADDENDUM_2.md")
DEFAULT_MANIFEST_PATH = PACKAGE_ROOT / "INC1_GENERATORS.json"
#: the gated generator module the launcher checks specifically (A2.6)
GENERATOR_MODULE_REL = "inc1_gen/generators.py"


def raw_sha256(path: str | Path) -> str:
    """sha256 over the file's raw bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lf_sha256(path: str | Path) -> str:
    """sha256 over the text with CRLF normalised to LF (registration rule)."""
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


#: text suffixes hashed with the LF convention; everything else is raw.
TEXT_SUFFIXES = frozenset({
    ".py", ".pyi", ".md", ".txt", ".json", ".jsonl", ".diff", ".patch",
    ".xml", ".cfg", ".ini", ".csv", ".toml", ".yaml", ".yml", ".rst",
})


def file_sha256(path: str | Path) -> str:
    """The gate sha of one file: LF-normalised for text, raw for binaries."""
    return (lf_sha256(path) if Path(path).suffix.lower() in TEXT_SUFFIXES
            else raw_sha256(path))


def module_files(root: Path | None = None) -> dict[str, Path]:
    """Every package ``*.py`` file (recursive), as relative posix paths."""
    root = Path(root) if root is not None else PACKAGE_ROOT
    return {
        path.relative_to(root.parent).as_posix(): path
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def build_manifest(*, package_root: Path | None = None,
                   addendum_path: str | Path | None = None,
                   now_utc: str | None = None) -> dict:
    """The manifest dict: module file sha256s + the addendum LF sha."""
    root = Path(package_root) if package_root is not None else PACKAGE_ROOT
    addendum = Path(addendum_path) if addendum_path is not None \
        else ADDENDUM_PATH
    if not addendum.is_file():
        raise FileNotFoundError(f"registration addendum missing: {addendum}")
    files = module_files(root)
    if not files:
        raise FileNotFoundError(f"no package modules under {root}")
    shas = {rel: file_sha256(path) for rel, path in files.items()}
    generator_rel = (Path(root).name + "/"
                     + GENERATOR_MODULE_REL.split("/", 1)[1])
    if generator_rel not in shas:
        raise FileNotFoundError(f"generator module missing: {generator_rel}")
    return {
        "schema": MANIFEST_SCHEMA,
        "recorded_utc": now_utc or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "registration_addendum": Path(addendum).name,
        "registration_addendum_lf_sha256": lf_sha256(addendum),
        "generator_module": generator_rel,
        "generator_module_sha256": shas[generator_rel],
        "files": shas,
    }


def write_manifest(path: str | Path | None = None, *, overwrite: bool = False,
                   **build_kwargs) -> Path:
    """Write ``INC1_GENERATORS.json`` (write-once by default)."""
    out = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite the recorded manifest: {out}")
    manifest = build_manifest(**build_kwargs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    return out


def verify_manifest(manifest_path: str | Path | None = None, *,
                    package_root: Path | None = None,
                    addendum_path: str | Path | None = None) -> dict:
    """Recompute the shas and compare (the A2.6 launcher-side gate).

    Returns ``{"ok": bool, "detail": str}``; a mismatch is a hard refusal for
    the launcher, never auto-repaired here.
    """
    out = Path(manifest_path) if manifest_path is not None \
        else DEFAULT_MANIFEST_PATH
    recorded = json.loads(out.read_text(encoding="utf-8"))
    fresh = build_manifest(package_root=package_root,
                           addendum_path=addendum_path,
                           now_utc=recorded.get("recorded_utc"))
    for key in ("registration_addendum_lf_sha256", "generator_module_sha256"):
        if recorded.get(key) != fresh.get(key):
            return {"ok": False,
                    "detail": f"{key} mismatch: recorded={recorded.get(key)} "
                              f"fresh={fresh.get(key)}"}
    recorded_files = recorded.get("files", {})
    if set(recorded_files) != set(fresh["files"]):
        return {"ok": False,
                "detail": "recorded module set differs from the package"}
    for rel, sha in fresh["files"].items():
        if recorded_files.get(rel) != sha:
            return {"ok": False,
                    "detail": f"module sha mismatch: {rel}"}
    return {"ok": True, "detail": "all recorded shas match"}
