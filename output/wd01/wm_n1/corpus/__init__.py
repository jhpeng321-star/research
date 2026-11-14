"""WM-N1 corpus build package.

Piece ``CORPUS``.  Turns the inventory into per-run shards, packs them into a
trainable memmap, verifies the pack, and produces an upload manifest that
refuses HOLDOUT / excluded / chunk-5 runs.  Also ships the real
:class:`corpus.source.PackedEvalSource` (INTERFACES Mismatch 5).
"""
from __future__ import annotations

__all__ = ["featurize_all", "pack", "verify", "manifest", "source", "memory",
           "cli"]

from . import featurize_all, pack, verify, manifest, source, memory, cli
