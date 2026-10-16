"""Append-only JSON-lines run logs (train, validation, loss shares)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class JsonlLog:
    """One ``*.jsonl`` file: truncated on :meth:`reset`, one sorted-key record per append."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def reset(self) -> None:
        self.path.write_text("", encoding="utf-8")

    def append(self, record: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Pretty, key-sorted JSON with a trailing newline (config / receipt files)."""
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
