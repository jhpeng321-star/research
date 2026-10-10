"""Shared, crash-safe artifact writing for maintained experiment drivers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON beside its destination and atomically replace the target."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)
