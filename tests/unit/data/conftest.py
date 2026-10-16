"""Path setup for the data unit tests: new package (src) and synthetic-pack fixtures."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for entry in (ROOT / "src", ROOT / "tests" / "fixtures"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
