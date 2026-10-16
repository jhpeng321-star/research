"""Small, dependency-free file helpers (hashing, JSON, JSONL, run keys)."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Union

PathLike = Union[str, "os.PathLike[str]"]
_CHUNK = 1024 * 1024


def sha256_file(path: PathLike) -> str:
    """Hex SHA-256 of a file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: PathLike) -> Any:
    """Parse a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: PathLike, obj: Any, *, sort_keys: bool = True) -> None:
    """Write JSON through a temp file in the same directory, then rename over ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(obj, stream, indent=2, sort_keys=sort_keys)
            stream.write("\n")
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def iter_jsonl(path: PathLike) -> Iterator[dict]:
    """Yield one dict per non-blank line of a JSONL file."""
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def run_key(meta: Mapping[str, Any]) -> str:
    """Pack run key ``<batch>/<world>__<arm>`` used by the window lists.

    ``batch`` is the campaign directory; ``chunk`` is not unique across campaigns
    and is only a fallback for packs that name the campaign ``chunk``.
    """
    batch = meta.get("batch") or meta.get("chunk")
    return f"{batch}/{meta['world']}__{meta['arm']}"
