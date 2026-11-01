"""FULL inference restricted to the twelve exposed development worlds."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'tools/rbeval')]

from tools.rbeval import run_wm
from tools.rbint.evaluate import main


if __name__ == '__main__':
    # The original evaluator defaults to reserved-first. Change its explicitly
    # bound world list before argument parsing, target/start reads or inference.
    run_wm.SEEDS = tuple(range(969420, 969432))
    main('FULL')
