import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent),                      # part3n
          str(HERE.parent.parent / "part3k"),    # decompose_p3k
          str(HERE.parent.parent / "part3l"),    # p3l_lib + aggregate_p3l
          str(HERE.parent.parent / "part3m"),    # p3m_lib + aggregate_p3m
          str(HERE.parent.parent / "part3")):    # engine + p3lib
    if p not in sys.path:
        sys.path.insert(0, p)
