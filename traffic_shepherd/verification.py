"""Read-only adoption checks. No simulation, training or acceptance-data access."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

import numpy as np

from .config import CONFIG_PATH, ROOT
from .damping import CirclesController

HISTORICAL_IMPORTS = frozenset({"env", "idm", "policies", "config", "counterfactual",
                              "experiments", "tools", "archive", "ray", "torch"})


def deleted_roots(root=ROOT):
    """Locations recorded as permanently deleted (MIG-04)."""
    path = Path(root) / "protocols" / "deleted_evidence.json"
    if not path.is_file():
        return ()
    record = json.loads(path.read_text(encoding="utf-8"))
    return tuple(r["path"] for r in record.get("fully_deleted_roots", []))


def _recorded_deleted(rel, roots):
    rel = str(rel).replace("\\", "/")
    return any(rel == r or rel.startswith(r + "/") for r in roots)


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def active_import_errors(root=ROOT):
    errors = []
    for path in sorted((Path(root) / "traffic_shepherd").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            elif isinstance(node, ast.Call) and node.args:
                function = node.func
                if ((isinstance(function, ast.Name) and function.id == "__import__") or
                    (isinstance(function, ast.Attribute) and function.attr == "import_module")):
                    if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        names = [node.args[0].value]
            for name in names:
                if name.split(".")[0] in HISTORICAL_IMPORTS:
                    errors.append(f"{path.relative_to(root)}:{node.lineno}: {name}")
    return errors


def documentation_errors(root=ROOT):
    root = Path(root)
    paths = [root / "README.md"]
    agents = root / "AGENTS.md"
    if agents.is_file():
        paths.append(agents)
    paths.extend(sorted((root / "docs").rglob("*.md")))
    roots = deleted_roots(root)
    errors = []
    for path in paths:
        if not path.is_file():
            errors.append(f"missing document: {path}")
            continue
        content = path.read_text(encoding="utf-8-sig")
        # Inline local links, including <space-containing targets>. Fenced commands
        # and external URLs are not link targets; archived prose is not rewritten.
        content = re.sub(r"```.*?```", "", content, flags=re.S)
        for target in re.findall(r"\]\((<[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\)", content):
            target = unquote(target.strip("<>"))
            if target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0]
            if target and not (path.parent / target).exists():
                resolved = (path.parent / target).resolve()
                if resolved.is_relative_to(root.resolve()) and _recorded_deleted(
                        resolved.relative_to(root.resolve()).as_posix(), roots):
                    continue
                errors.append(f"{path.relative_to(root)}: broken link {target}")
    return errors


def replay_saved_observations(*, full_trajectories=False):
    controller = CirclesController()
    provenance = json.loads((CONFIG_PATH.parent / "provenance.json").read_text())
    if hash_file(CONFIG_PATH.parent / "LICENSE") != provenance["license_sha256"]:
        raise ValueError("license identity mismatch")
    fixtures = json.loads((CONFIG_PATH.parent / "replay_samples.json").read_text())
    for row in fixtures["samples"]:
        if controller.infer(row["raw_si"]) != row["network_output_mps2"]:
            raise ValueError("saved singleton inference changed")
    result = {"fixture_rows": len(fixtures["samples"]), "fixture_exact": True,
              "model_sha256": controller.config.model_sha256, "license_hash_matches": True}
    if full_trajectories:
        # Only the two already exposed direct-transfer trajectories; never glob
        # historical results or open acceptance outcomes.
        sources = {row["source_file"]: row["source_sha256"] for row in fixtures["samples"]}
        count, maximum = 0, 0.
        roots = deleted_roots()
        deleted = [n for n in sources
                   if _recorded_deleted(f"archive/pre_circles/results/{n}", roots)
                   and not (ROOT / "archive/pre_circles/results" / n).is_file()]
        if deleted:
            result.update(trajectory_sources_deleted=deleted,
                          trajectory_note="saved trajectories recorded as deleted (MIG-04); "
                                          "replay not possible; singleton fixtures still checked")
            return result
        for name, digest in sources.items():
            path = ROOT / "archive/pre_circles/results" / name
            if hash_file(path) != digest:
                raise ValueError(f"saved trajectory hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as data:
                for observation, action in zip(data["observations"], data["actions"], strict=True):
                    delta = abs(controller.infer(observation[1:4]) - float(action[0]))
                    if not np.isfinite(delta) or delta > 2e-6:
                        raise ValueError(f"trajectory output mismatch: {path}, row {count}")
                    count += 1
                    maximum = max(maximum, delta)
        result.update(trajectory_rows=count, trajectory_max_abs_error_mps2=maximum,
                      trajectory_sources=list(sources))
    return result


def verify_archive(*, all_hashes=False, root=ROOT):
    root = Path(root)
    manifest_path = root / "archive/pre_circles/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors, checked = [], 0
    if manifest.get("status") != "relocated_with_explicit_frozen_reader_aliases":
        errors.append("archive relocation is not complete")
    rows = manifest.get("files", [])
    if not rows or len(rows) != manifest.get("file_count"):
        errors.append("archive inventory is empty or disagrees with declared file count")
    if len({r["original_path"] for r in rows}) != len(rows) or len({r["current_path"] for r in rows}) != len(rows):
        errors.append("duplicate original or archived artifact mapping")
    roots = deleted_roots(root)
    deleted = 0
    for row in rows:
        path = root / row["current_path"]
        if not path.is_file() and _recorded_deleted(row["current_path"], roots):
            deleted += 1
        elif not path.is_file() or path.stat().st_size != row["bytes"]:
            errors.append(row["original_path"] + ": missing or size mismatch")
        elif all_hashes:
            checked += 1
            if hash_file(path) != row["sha256"]:
                errors.append(row["original_path"] + ": hash mismatch")
    return {"manifest": str(manifest_path), "manifest_status": manifest["status"],
            "files": len(rows), "hashes_checked": checked, "recorded_deleted": deleted,
            "errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-trajectories", action="store_true")
    parser.add_argument("--archive", action="store_true", help="check every archived path/size")
    parser.add_argument("--archive-hashes", action="store_true", help="also hash all archived bytes")
    args = parser.parse_args(argv)
    report = {"schema": "circles_adoption_verification/1",
              "inference": replay_saved_observations(full_trajectories=args.saved_trajectories),
              "import_errors": active_import_errors(), "documentation_errors": documentation_errors()}
    if args.archive or args.archive_hashes:
        report["archive"] = verify_archive(all_hashes=args.archive_hashes)
    report["passed"] = not (report["import_errors"] or report["documentation_errors"] or
                            report.get("archive", {}).get("errors"))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
