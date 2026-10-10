import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent),                      # part3o
          str(HERE.parent.parent / "part3n"),    # p3n_lib
          str(HERE.parent.parent / "part3k"),    # decompose_p3k
          str(HERE.parent.parent / "part3l"),    # p3l_lib + aggregate
          str(HERE.parent.parent / "part3m"),    # p3m_lib
          str(HERE.parent.parent / "part3")):    # engine + p3lib
    if p not in sys.path:
        sys.path.insert(0, p)
