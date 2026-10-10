"""Torch >= 2.7, < 2.13 compatibility: static API scan + version-gated op test.

The GPU boxes run ``torch>=2.7,<2.13`` from the CUDA 12.8 (cu128) wheel index
for Blackwell-class sm_120 GPUs; this CPU box carries 2.12.1+cpu.  G5 therefore
runs:

1. a static scan of the WM-N1 source for deprecated / unsupported torch APIs
   (deprecated ``torch.cuda.amp`` -> ``torch.amp``, removed ``torch._six``,
   ``torch.compile``, ``scaled_dot_product_attention``, ``flex_attention``,
   ``torch.export``); and
2. an op-level test that RUNS when ``torch.__version__`` is inside the
   supported ``>=2.7,<2.13`` window and is explicitly SKIPPED (documented)
   otherwise.

To run the gated test for real, use the pinned GPU-box environment::

    python -m pip install --index-url https://download.pytorch.org/whl/cu128 \\
        "torch>=2.7,<2.13"
    python -m preflight e2e-smoke

Then verify the box end to end with ``python -m bundle.envcheck``.
"""
from __future__ import annotations

from pathlib import Path

#: Human-readable pin the GPU boxes install (also recorded in reports).
EXPECTED_TORCH = ">=2.7,<2.13"
#: Inclusive minimum / exclusive maximum supported (major, minor).
TORCH_MIN = (2, 7)
TORCH_MAX = (2, 13)
#: APIs that are absent, removed or deprecated across torch 2.7-2.12.
FORBIDDEN_APIS = ("torch.compile", "torch.cuda.amp", "torch._six",
                  "scaled_dot_product_attention", "flex_attention",
                  "torch.export")
_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", "reports", "tests"}


def _major_minor(version: str):
    """Parse ``"2.12.1+cpu"`` -> ``(2, 12)``; return ``None`` if unparseable."""
    head = str(version).split("+", 1)[0].split("-", 1)[0]
    parts = head.split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return None


def supports(version: str) -> bool:
    """True when ``version`` lies in the supported ``[2.7, 2.13)`` window."""
    mm = _major_minor(version)
    return mm is not None and TORCH_MIN <= mm < TORCH_MAX


def scan_sources(root) -> dict:
    """Scan ``*.py`` under ``root`` for forbidden/deprecated torch APIs."""
    root = Path(root)
    hits = []
    n_files = 0
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if set(rel.parts) & _SKIP_DIRS:
            continue
        if path.name == "torch_compat.py":
            continue
        n_files += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for token in FORBIDDEN_APIS:
            if token in text:
                hits.append({"file": rel.as_posix(), "api": token})
    return {"ok": not hits, "n_files": n_files, "forbidden_hits": hits,
            "forbidden_apis": list(FORBIDDEN_APIS)}


def version_gate(expected: str = EXPECTED_TORCH) -> dict:
    """Run a small allowed-op test on 2.7-2.12; skip (documented) otherwise."""
    import torch
    version = str(torch.__version__)
    if not supports(version):
        lo, hi = TORCH_MIN, TORCH_MAX
        return {"status": "skipped", "ok": True, "torch_version": version,
                "expected": expected,
                "reason": (f"torch {version} is outside the supported "
                           f">={lo[0]}.{lo[1]},<{hi[0]}.{hi[1]} cu128 window; "
                           "the version-gated op test runs in that env")}
    try:
        x = torch.randn(4, 3)
        layer = torch.nn.Linear(3, 3)
        y = torch.clamp(layer(x), min=-1.0, max=1.0)
        fin = bool(torch.isfinite(y).all().item())
        if not fin:
            raise AssertionError("non-finite output")
        return {"status": "ran", "ok": True, "torch_version": version,
                "expected": expected, "reason": None}
    except Exception as e:                                # noqa: BLE001
        return {"status": "ran", "ok": False, "torch_version": version,
                "expected": expected,
                "reason": f"{type(e).__name__}: {e}"}


def scan_and_check(root, expected: str = EXPECTED_TORCH) -> dict:
    scan = scan_sources(root)
    gate = version_gate(expected)
    return {"ok": bool(scan["ok"] and gate["ok"]), "scan": scan,
            "version_gate": gate, "torch_version": gate["torch_version"],
            "expected": expected}
