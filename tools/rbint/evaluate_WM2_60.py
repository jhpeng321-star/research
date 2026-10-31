"""Short endpoint evidence from the ORIGINAL WM2 runtime; no600s panel repeat.

run_wm keeps its original runtime/config defaults and600s padding/random plans;
only120 halfsteps are executed. Do not import INT's rebuilt model before runtime
selection: that would accidentally replace WM2's frozen source modules.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'tools/rbeval')]
from tools.rbeval.run_wm import main

if __name__ == '__main__':
    if '--model-name' not in sys.argv:
        sys.argv += ['--model-name', 'wm2_hdv60']
    sys.argv += ['--endpoint-only']
    main(identity_sources=[Path(__file__)])
