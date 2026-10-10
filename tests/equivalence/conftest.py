"""Path setup for the equivalence suite: new package (src), legacy tree (repo root)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
