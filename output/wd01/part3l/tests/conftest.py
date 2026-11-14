import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent),                      # part3l
          str(HERE.parent.parent / "part3k"),    # decompose_p3k
          str(HERE.parent.parent / "part3d")):   # models (untrained nets)
    if p not in sys.path:
        sys.path.insert(0, p)
