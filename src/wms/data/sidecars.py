"""TRAIN sidecars: verified slot->vehicle identity bundles and the process environment hooks.

Outcome and label-overlay sidecars are consumed (and hash-verified) by the vendored pack
reader through ``WM_N1_TRAIN_OUTCOMES``; outcome labels are offline-only and never enter a
training window.  :func:`configure_train_sidecars` sets the environment hooks.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from wms.core.errors import TrainDataRefusal
from wms.data.sources import validate_train_source

IDENTITY_SCHEMA = "wm_n1_train_identity_sidecar/1"
OUTCOMES_ENV = "WM_N1_TRAIN_OUTCOMES"
IDENTITIES_ENV = "WM_N1_TRAIN_IDENTITIES"


def read_identity_rows(sidecar: str | Path, *, corpus_k: int) -> dict[int, np.ndarray]:
    """Validate a TRAIN identity bundle and return ``{manifest position: int64[K] ids}``.

    Verifies the receipt (split, schema), the sha256 of every file, dtypes/shapes, position
    uniqueness and the id range; the bundle path is not followed through junctions.
    """
    root = validate_train_source(sidecar)
    receipt_path = root / "TRAIN_IDENTITY_RECEIPT.json"
    if not receipt_path.is_file():
        raise TrainDataRefusal(f"identity receipt missing: {root}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("split") != "TRAIN" or receipt.get("schema") != IDENTITY_SCHEMA:
        raise TrainDataRefusal(f"identity sidecar is not TRAIN: {root}")
    files = receipt.get("files")
    if not isinstance(files, dict):
        raise TrainDataRefusal(f"identity sidecar has no file hashes: {root}")
    for name in ("identities.npy", "global_positions.npy", "vehicle_map.json"):
        path = root / name
        if not path.is_file():
            raise TrainDataRefusal(f"identity sidecar file missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != files.get(name):
            raise TrainDataRefusal(f"identity sidecar hash mismatch: {path}")
    ids = np.load(root / "identities.npy", mmap_mode="r", allow_pickle=False)
    positions = np.load(root / "global_positions.npy", mmap_mode="r", allow_pickle=False)
    if ids.dtype != np.int64 or positions.dtype != np.int64 or ids.ndim != 2 \
            or ids.shape[1] != int(corpus_k) or positions.shape != (ids.shape[0],):
        raise TrainDataRefusal(f"identity sidecar shape/dtype mismatch: {root}")
    if len(set(positions.tolist())) != len(positions):
        raise TrainDataRefusal(f"identity positions are duplicated: {root}")
    count = len(json.loads((root / "vehicle_map.json").read_text(encoding="utf-8")))
    if np.any(ids < -1) or np.any(ids >= count):
        raise TrainDataRefusal(f"identity IDs out of range: {root}")
    return {int(position): np.asarray(row, dtype=np.int64)
            for position, row in zip(positions.tolist(), ids)}


def configure_train_sidecars(*, outcomes: Optional[str | Path] = None,
                             identities: Optional[str | Path] = None) -> None:
    """Point the vendored pack reader at the TRAIN outcome / identity sidecars.

    The reader resolves both variables lazily, when a window is built, so they must stay
    set for the whole life of every dataset (this is the legacy ``--outcomes`` behaviour).
    ``None`` leaves a variable untouched.
    """
    if outcomes is not None:
        os.environ[OUTCOMES_ENV] = str(outcomes)
    if identities is not None:
        os.environ[IDENTITIES_ENV] = str(identities)
