"""Run focused pytest with a fail-closed research-file access boundary."""
from pathlib import Path
import os
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT)
RUNTIME = Path(sys.prefix).resolve()
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")


def boundary(event, args):
    if event != "open" or isinstance(args[0], int):
        return
    if os.fsdecode(args[0]).lower().rstrip("\\/") in (os.devnull.lower(), "\\\\.\\nul"):
        return
    path = Path(os.fsdecode(args[0]))
    resolved = path.resolve()
    if resolved.is_relative_to(RUNTIME):
        return  # Python/pytest/PyTorch installed runtime, not research data.
    text = (str(path) + "/" + str(resolved)).lower()
    if (not resolved.is_relative_to(ROOT) or any(word in text for word in FORBIDDEN)
            or ("truth" in text and resolved.suffix.lower() in (".npy", ".npz", ".csv", ".parquet"))
            or any(964000 <= int(m[0]) <= 964199
                   for m in re.finditer(r"(?<!\d)964\d{3}(?!\d)", text))):
        raise PermissionError(f"R9 verification access boundary: {path}")


if __name__ == "__main__":
    import pytest

    sys.addaudithook(boundary)
    raise SystemExit(pytest.main(sys.argv[1:]))
