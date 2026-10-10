"""Shared fakes for the pre-flight gate tests."""
from __future__ import annotations

from pathlib import Path

from preflight.imports import PieceMissing
from preflight.settings import Settings


class FakeResolver:
    """Resolve only the dotted names present in ``mapping``."""

    def __init__(self, mapping=None):
        self.mapping = dict(mapping or {})

    def __call__(self, dotted):
        if dotted not in self.mapping:
            raise PieceMissing(f"piece missing: {dotted}")
        return self.mapping[dotted]


class FakeFn:
    """Callable returning a fixed payload (or raising)."""

    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.payload


def settings(tmp_path, *, wm_n1_dir=None) -> Settings:
    return Settings(wm_n1_dir=Path(wm_n1_dir or tmp_path),
                    reports_dir=Path(tmp_path) / "reports", device="cpu")


def clean_torch_tree(tmp_path) -> Path:
    """A source tree with no forbidden torch APIs (for G5's scan)."""
    root = Path(tmp_path) / "clean_pkg"
    root.mkdir(parents=True, exist_ok=True)
    (root / "mod.py").write_text(
        "import torch\nx = torch.clamp(torch.zeros(1), 0, 1)\n",
        encoding="utf-8")
    return root


def dirty_torch_tree(tmp_path) -> Path:
    """A source tree containing a forbidden torch API."""
    root = Path(tmp_path) / "dirty_pkg"
    root.mkdir(parents=True, exist_ok=True)
    (root / "bad.py").write_text(
        "import torch\nm = torch.compile(lambda x: x)\n", encoding="utf-8")
    return root
