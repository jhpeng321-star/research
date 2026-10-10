"""Workspace root resolution for the planner (PL-13).

Every former hard-coded ``D:`` drive literal resolves through these helpers:
research data lives under the research-dense tree and sibling worktrees under
the rd-wt worktree root. Both roots come from environment variables with E:
defaults, read AT CALL TIME so tests (``monkeypatch.setenv``) and other
machines can relocate the trees without code changes:

- ``RESEARCH_DENSE_ROOT`` (default ``E:/research-dense``) -> research data,
  runs, output and assets.
- ``RD_WT_ROOT`` (default ``E:/rd-wt``) -> sibling worktrees (wd01/wd01b/
  wd01c predict kit trees, wm34 saved-state tree, ...).
"""

import os
from pathlib import Path

__all__ = ["rd_wt_root", "research_dense_root"]

RESEARCH_DENSE_ROOT_ENV = "RESEARCH_DENSE_ROOT"
RD_WT_ROOT_ENV = "RD_WT_ROOT"

DEFAULT_RESEARCH_DENSE_ROOT = "E:/research-dense"
DEFAULT_RD_WT_ROOT = "E:/rd-wt"


def research_dense_root() -> Path:
    """The research-dense data root (env ``RESEARCH_DENSE_ROOT``)."""
    return Path(os.environ.get(RESEARCH_DENSE_ROOT_ENV) or DEFAULT_RESEARCH_DENSE_ROOT)


def rd_wt_root() -> Path:
    """The sibling-worktrees root (env ``RD_WT_ROOT``)."""
    return Path(os.environ.get(RD_WT_ROOT_ENV) or DEFAULT_RD_WT_ROOT)


def export_research_env() -> None:
    """Publish the resolved research-dense root to the well-known env vars
    the read-only SIBLING trees key off internally: ``WD01_SRC`` (wd01's p2C
    ``pdl`` loader, D: default) and ``RESEARCH_DENSE_ROOT`` (research-dense's
    ``hybrid_state_v1/repo`` module, D: default). Without this export that
    sibling code would touch the doomed D: junctions even though every
    planner-side path resolves here. Owner-set values always win
    (``setdefault``); call before the first sibling import."""
    root = str(research_dense_root())
    os.environ.setdefault("WD01_SRC", root)
    os.environ.setdefault(RESEARCH_DENSE_ROOT_ENV, root)
