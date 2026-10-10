"""WM-N1 training package (piece ``TRAIN``).

Consumes packed memmap shards from :mod:`corpus` (frozen tensors of
``features/FORMAT.md``) and writes member checkpoints consumed by
:mod:`adapter`.  Public API: :class:`config.TrainConfig`,
:class:`dataset.MemmapCorpus`, :class:`trainer.Trainer` and :mod:`cli`.
"""
from __future__ import annotations

__all__ = ["config", "dataset", "trainer", "cli"]

from . import config, dataset, trainer, cli
