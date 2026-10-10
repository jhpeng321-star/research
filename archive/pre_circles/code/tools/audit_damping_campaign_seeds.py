"""Audit candidate partitions without opening sealed traffic outcomes."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


SEALED = re.compile(r"final|sealed|acceptance|holdout|heldout|prospective|validation|locked", re.I)
METADATA = {
    "manifest.json",
    "run_manifest.json",
    "config.json",
    "run_config.json",
    "args.json",
    "arguments.json",
    "params.json",
    "protocol.json",
}
SOURCE_EXT = {".py", ".ps1", ".md", ".toml", ".yaml", ".yml", ".sh", ".sumocfg"}


def audit(roots, blocks):
    seeds = set().union(*map(set, blocks.values()))
    pattern = re.compile(r"(?<!\d)(?:" + "|".join(map(str, sorted(seeds))) + r")(?!\d)")
    counts, hits, errors = {}, [], []
    inventory = hashlib.sha256()
    for root in map(Path, roots):
        if not root.exists():
            errors.append({"root": str(root), "error": "missing audit root"})
            continue
        command = ["rg", "--files", "--hidden", "--no-ignore"]
        for excluded in (".git", "__pycache__", "node_modules", ".venv", ".pytest_cache"):
            command += ["-g", f"!**/{excluded}", "-g", f"!**/{excluded}/**"]
        result = subprocess.run(
            command + [str(root)], capture_output=True, encoding="utf-8", errors="replace"
        )
        if result.returncode not in (0, 1):
            errors.append({"root": str(root), "error": result.stderr})
        files = [Path(line) for line in result.stdout.splitlines()]
        scanned = skipped = 0
        for path in files:
            relative = path.relative_to(root).as_posix()
            inventory.update((str(root) + "/" + relative + "\n").encode())
            if pattern.search(relative):
                hits.append({"path": str(path), "kind": "filename_reference"})
            artifact = any(part in {"runs", "results", "outputs", "output"} for part in path.parts)
            if artifact and SEALED.search(relative):
                skipped += 1
                continue
            eligible = (
                not artifact and path.suffix.lower() in SOURCE_EXT
            ) or path.name.lower() in METADATA
            if not eligible:
                continue
            try:
                if path.stat().st_size > 5_000_000:
                    errors.append(
                        {"path": str(path), "error": "eligible metadata/source exceeds size cap"}
                    )
                    continue
                content = path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeError) as exc:
                errors.append({"path": str(path), "error": str(exc)})
                continue
            scanned += 1
            found = sorted({int(match.group()) for match in pattern.finditer(content)})
            if found:
                hits.append(
                    {"path": str(path), "kind": "source_or_metadata_literal", "seeds": found}
                )
            if path.suffix.lower() != ".py":
                continue
            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue  # Literal scanning still applies to historical Python.
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.For)):
                    continue
                if not any(
                    (isinstance(part, ast.Name) and "seed" in part.id.lower())
                    or (
                        isinstance(part, ast.Constant)
                        and isinstance(part.value, str)
                        and "seed" in part.value.lower()
                    )
                    for part in ast.walk(node)
                ):
                    continue
                for call in ast.walk(node):
                    if (
                        not isinstance(call, ast.Call)
                        or not isinstance(call.func, ast.Name)
                        or call.func.id != "range"
                    ):
                        continue
                    try:
                        args = [ast.literal_eval(arg) for arg in call.args]
                        if not 1 <= len(args) <= 3 or not all(isinstance(v, int) for v in args):
                            continue
                        declared = range(*args)
                    except (ValueError, TypeError):
                        continue
                    overlap = sorted(seed for seed in seeds if seed in declared)
                    if overlap:
                        hits.append(
                            {
                                "path": str(path),
                                "kind": "static_seed_range",
                                "line": call.lineno,
                                "seeds": overlap,
                            }
                        )
        counts[str(root)] = dict(
            inventoried_files=len(files),
            scanned_source_or_metadata_files=scanned,
            sealed_artifact_paths_not_opened=skipped,
        )
    return dict(
        schema="native_damping_campaign_seed_audit/1",
        blocks=blocks,
        counts=counts,
        inventory_sha256=inventory.hexdigest(),
        hits=hits,
        errors=errors,
        passed=not hits and not errors,
        scope="filename inventory, source literals/static seed ranges, nonsealed metadata; no outcome payload scan",
        limitation="Absence of recorded references cannot prove absence of undocumented execution.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--training-start", type=int, required=True)
    parser.add_argument("--development-start", type=int, required=True)
    parser.add_argument("--acceptance-start", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    blocks = dict(
        training=list(range(args.training_start, args.training_start + 16)),
        development=list(range(args.development_start, args.development_start + 8)),
        acceptance_reserved=list(range(args.acceptance_start, args.acceptance_start + 32)),
    )
    if sum(map(len, blocks.values())) != len(set().union(*map(set, blocks.values()))):
        raise ValueError("candidate partitions overlap")
    if args.output.exists():
        raise FileExistsError("seed audits are immutable")
    result = audit(args.root, blocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
