"""TRAIN-source validation shared by every data reader (fail closed)."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from wms.core.errors import TrainDataRefusal

#: Path fragments that mark restricted (non-TRAIN) material; matched case-insensitively.
FORBIDDEN_TOKENS = ("CAL_SPLIT_A4.json", "HOLDOUT", "CALIBRATION", "chunk-5",
                    "chunk5", "J4B", "INC1", "sealed")


def validate_train_source(path: str | Path, manifest: Optional[Mapping] = None) -> Path:
    """Validate source naming and manifest split claims before reading arrays."""
    p = Path(path)
    text = str(p).replace("\\", "/")
    for token in FORBIDDEN_TOKENS:
        if token.lower() in text.lower():
            raise TrainDataRefusal(f"forbidden TRAIN source token: {token}")
    if manifest is not None:
        split = str(manifest.get("split", "TRAIN")).upper()
        if split not in ("TRAIN",):
            raise TrainDataRefusal(f"source manifest split is not TRAIN: {split}")
        for item in manifest.get("items", ()):  # metadata only, before arrays
            item_split = str(item.get("split", "TRAIN")).upper()
            if item_split != "TRAIN":
                raise TrainDataRefusal(f"item split is not TRAIN: {item_split}")
    return p


def fixed_sidecar_dir(value: str) -> str:
    """Picklable replacement for ``IdeaACorpus._sidecar_dir`` (per-corpus identity dir)."""
    return value
