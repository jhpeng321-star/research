"""Exact WM-PHYS evaluator entrypoint; delegates the registered panel and scorer."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'tools/rbeval')]

from tools.rbint.recipe import load_candidate, candidate_future
from tools.rbeval import run_wm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--variant', required=True, choices=('A', 'B'))
    args, rest = parser.parse_known_args()
    os.environ['WMPHYS_VARIANT'] = args.variant
    sys.argv = [sys.argv[0], *rest]
    run_wm.main(model_loader=load_candidate, future_builder=candidate_future,
                identity_sources=[ROOT/'tools/rbint/rebuild.py', Path(__file__)])
