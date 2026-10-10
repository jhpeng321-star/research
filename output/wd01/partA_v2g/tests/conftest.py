import sys
from pathlib import Path

WD01 = Path(__file__).resolve().parents[2]
if str(WD01) not in sys.path:
    sys.path.insert(0, str(WD01))
