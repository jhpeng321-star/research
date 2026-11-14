import sys
from pathlib import Path

# make `import predict` work when pytest runs from any cwd:
# tests -> predict -> wd01  (package root = output/wd01)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
