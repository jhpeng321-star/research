"""Env probe for S-BACKLOG CPU hosts."""
import sys
print(sys.version.split()[0], sys.executable)
try:
    import numpy
    print("numpy", numpy.__version__)
except Exception as e:
    print("numpy MISSING", e)
try:
    import torch
    print("torch", torch.__version__)
except Exception as e:
    print("torch MISSING", type(e).__name__)
try:
    import scipy
    print("scipy", scipy.__version__)
except Exception as e:
    print("scipy MISSING", type(e).__name__)
from pathlib import Path
for d in (Path("/home/ubuntu/backlog"),):
    print(d, "exists" if d.is_dir() else "missing")
import os
print("cpus", os.cpu_count())
