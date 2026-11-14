import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WM_N1 = HERE.parents[1]
PARTB = WM_N1.parent / "partB_v0data"
for p in (str(WM_N1), str(PARTB)):
    if p not in sys.path:
        sys.path.insert(0, p)
