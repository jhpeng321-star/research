"""Exact WM-PHYS evaluator entrypoint; delegates the registered panel and scorer."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'tools/rbeval')]

from tools.rbint.recipe import load_candidate, candidate_future
from tools.rbeval import run_wm


def main(variant):
    os.environ['WMPHYS_VARIANT'] = variant
    sources = [ROOT/'tools/rbint/rebuild.py', Path(__file__), Path(sys.argv[0]).resolve(),
               ROOT/'configs/native_cav_head_v5.json', ROOT/'tools/rbint/endpoints.py',
               ROOT/'tools/rbint/observations.py', ROOT/'tools/rbint/departure.py']
    import tomllib
    parser_config = argparse.ArgumentParser(add_help=False)
    parser_config.add_argument('--wm-config', required=True)
    config_args, _ = parser_config.parse_known_args()
    physics = tomllib.loads(Path(config_args.wm_config).read_text())['physics']
    if physics.get('physics_params', 'off') != 'off':
        sources.append(ROOT/physics['physics_params_file'])
    if variant == 'C':
        sources.append(ROOT/'artifacts/rbfit/pooled_full_v5/idm_pooled.json')
    if variant == 'FULL':
        sources += [ROOT/'tools/rbint/motion_posterior.py', ROOT/'tools/rbfit/idm_fit.py',
                    ROOT/'artifacts/rbfit/shrinkage_v1/idm_population.json',
                    ROOT/'artifacts/rbfit/shrinkage_v1/idm_posteriors.json.gz',
                    ROOT/'artifacts/rbfit/shrinkage_v1/idm_shrinkage.json',
                    ROOT/'docs/paper_results/rbunc/observed_arrivals.json']
    run_wm.main(model_loader=load_candidate, future_builder=candidate_future,
                identity_sources=sources)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--variant', required=True, choices=('A', 'B', 'C', 'FULL'))
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    main(args.variant)
