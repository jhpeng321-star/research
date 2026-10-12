"""Enforce the acyclic layer rule of ARCHITECTURE_V4 section 2.1 on every wms module."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "wms"

# layer -> layers it may import (itself is always allowed)
ALLOWED = {
    "core": set(),
    "config": {"core"},
    "physics": {"core"},
    "model": {"core", "config", "physics"},
    "data": {"core", "config"},
    "train": {"core", "config", "physics", "model", "data", "sim_bridge"},
    "eval": {"core", "config", "physics", "model", "data", "sim_bridge"},
    "sim_bridge": {"core", "config"},
    "legacy": {"core", "config", "physics", "model", "data", "sim_bridge"},
    "cli": {"core", "config", "physics", "model", "data", "train", "eval", "sim_bridge", "legacy"},
    # C4 learned control (2026-10-12): policy/trainer/RSU check on top of the world model.
    "control": {"core", "config", "physics", "model", "data", "train", "eval", "sim_bridge"},
    "root": {"core", "config", "cli"},  # wms/__init__.py and wms/__main__.py
}
# Only these layers may import `legacy` (the eval/cli model factories).
LEGACY_IMPORTERS = {"legacy", "cli", "eval"}
# Sanctioned single edges that break the layer rule on purpose (module path -> {layer: reason}).
# Anything not listed here is a violation. Keep this table short; every entry is lazy/narrow.
EXCEPTIONS = {
    "data/corpus.py": {"legacy": "adapter around the vendored IdeaACorpus pack reader (ARCHITECTURE_V4 1.4, decision 7)"},
    "data/pack_history.py": {"model": "entries-v2 feature builder is the one sanctioned data->model edge (lazy import)"},
    "model/factory.py": {"legacy": "lazy construction of the black-box baselines kept only to score old checkpoints"},
}


def layer_of(parts: tuple[str, ...]) -> str:
    """Layer of a module path relative to ``src/wms`` (``("config", "load")`` -> ``config``)."""
    head = parts[0]
    if head.endswith(".py"):
        head = head[:-3]
    return head if head in ALLOWED and head != "root" else "root"


def imported_layers(path: Path) -> set[str]:
    rel = path.relative_to(SRC).with_suffix("")
    package = rel.parts[:-1] if rel.name != "__init__" else rel.parts[:-1]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            targets = [a.name.split(".") for a in node.names if a.name.split(".")[0] == "wms"]
            found.update(layer_of(tuple(t[1:]) or ("__init__",)) for t in targets)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: resolve against this module's package
                base = list(package)[: len(package) - (node.level - 1)] if node.level > 1 else list(package)
                target = base + (node.module.split(".") if node.module else [])
                if not target and node.names:  # `from .. import x`
                    target = [node.names[0].name]
            elif node.module and node.module.split(".")[0] == "wms":
                target = node.module.split(".")[1:] or [node.names[0].name]
            else:
                continue
            found.add(layer_of(tuple(target) or ("__init__",)))
    return found


MODULES = sorted(SRC.rglob("*.py"))


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(SRC)))
def test_module_respects_the_layer_rule(path):
    own = layer_of(path.relative_to(SRC).parts)
    allowed = ALLOWED[own] | {own} | set(EXCEPTIONS.get(path.relative_to(SRC).as_posix(), {}))
    bad = imported_layers(path) - allowed
    assert not bad, f"{path.relative_to(SRC)} (layer {own}) imports forbidden layer(s) {sorted(bad)}"


def test_only_cli_and_eval_import_legacy():
    offenders = [str(p.relative_to(SRC)) for p in MODULES
                 if layer_of(p.relative_to(SRC).parts) not in LEGACY_IMPORTERS
                 and "legacy" in imported_layers(p)
                 and "legacy" not in EXCEPTIONS.get(p.relative_to(SRC).as_posix(), {})]
    assert not offenders


def test_exceptions_are_still_needed():
    """A stale exception hides nothing but must not outlive the edge it documents."""
    for rel, layers in EXCEPTIONS.items():
        found = imported_layers(SRC / rel)
        assert set(layers) <= found, f"{rel}: exception for {sorted(set(layers) - found)} is stale"


# The one sanctioned sim -> wms edge: C4 SUMO runner glue (lazy, optional adapters only:
# wall executor, trainer history featurisers, contract constants); the C4 environment,
# conversion, recorder and collector themselves never import wms.
SIM_TO_WMS_EXCEPTIONS = {"c4/wms_glue.py"}


def test_eval_never_imports_train_and_sim_layer_never_imports_wms():
    eval_modules = [p for p in MODULES if layer_of(p.relative_to(SRC).parts) == "eval"]
    assert all("train" not in imported_layers(p) for p in eval_modules)
    shepherd = SRC.parents[1] / "traffic_shepherd"
    if shepherd.is_dir():
        for path in shepherd.rglob("*.py"):
            if path.relative_to(shepherd).as_posix() in SIM_TO_WMS_EXCEPTIONS:
                continue
            text = path.read_text(encoding="utf-8")
            assert "import wms" not in text and "from wms" not in text, path


def test_layer_detector_flags_a_violation(tmp_path, monkeypatch):
    (tmp_path / "core").mkdir()
    bad = tmp_path / "core" / "x.py"
    bad.write_text("from ..config import load\nimport wms.train.trainer\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "SRC", tmp_path)
    assert imported_layers(bad) == {"config", "train"}
