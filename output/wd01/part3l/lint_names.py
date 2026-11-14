"""Tiny AST linter for the part 3l scripts.  Catches:
  1. bare module-level names that are loaded but never defined/imported;
  2. attribute loads on IMPORTED MODULE ALIASES that do not exist on the
     target module (e.g. L.MS_BREQ_HIST when the lib defines
     MS_BREQ_HIST) -- the transcription-typo class this part's scripts
     are prone to.
Names assigned anywhere in the file (incl. locals) count as defined for
(1); only undeclared UPPERCASE globals and unknown imported-module
attributes are reported.  Imports the target modules (safe: the part 3l
modules import cleanly).  Pure stdlib."""
from __future__ import annotations

import ast
import builtins
import importlib
import sys
from pathlib import Path


def check(path, extra_defs=()):
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    # make the script's own package and the part3 siblings importable so
    # imported-module attributes can be resolved
    here = Path(path).resolve().parent
    for sib in (here, here.parent / "part3k", here.parent / "part3",
                here.parent / "partA_data", here.parent / "part3d"):
        sp = str(sib)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    aliases = {}                     # local alias -> module name
    defined = set(extra_defs)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        defined.add(n.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                nm = (a.asname or a.name).split(".")[0]
                defined.add(nm)
                if isinstance(node, ast.Import) and a.name == nm:
                    aliases[nm] = nm
                elif isinstance(node, ast.ImportFrom) and a.asname:
                    try:
                        aliases[a.asname] = getattr(
                            importlib.import_module(node.module), a.name)
                    except Exception:
                        pass
                elif isinstance(node, ast.Import):
                    aliases[nm] = a.name        # import a.b as c
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            nm = node.id
            if nm not in defined and nm not in vars(builtins) \
                    and nm.isupper() | nm.replace("_", "").isupper():
                issues.append(f"undefined name {nm}")
        elif isinstance(node, ast.Attribute) and \
                isinstance(node.value, ast.Name) and \
                isinstance(node.value.ctx, ast.Load):
            base = aliases.get(node.value.id)
            if base is not None:
                try:
                    mod = (importlib.import_module(base)
                           if isinstance(base, str) else base)
                except Exception as exc:
                    issues.append(f"{node.value.id} unimportable: {exc}")
                    continue
                if not hasattr(mod, node.attr):
                    issues.append(f"{node.value.id}.{node.attr} missing")
    return issues


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        unk = check(p)
        if unk:
            bad += 1
            print(f"{p}: ISSUES: {sorted(set(unk))}")
        else:
            print(f"{p}: ok")
    sys.exit(1 if bad else 0)
