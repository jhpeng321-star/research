"""One development-only future for each of two fixed worlds, all five arms/H600."""
from contextlib import ExitStack
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from tools.rbint.audit import forbid_learned_corrections
from tools.rbint.recipe import load_candidate, candidate_future
from tools.rbeval import run_wm
from tools.rbeval.core import digest, load, save_new


def main():
    os.environ['WMPHYS_VARIANT'] = 'FULL'
    args = run_wm.arguments()
    out = Path(args.out)
    recipe = Path(args.wm_config).resolve()
    if recipe not in tuple(ROOT/'configs'/n for n in ('wm_phys_v5.toml', 'wm_phys_v5_lp.toml')):
        raise ValueError('development canary uses only the two explicit integrated recipes')
    # Bind active helper code/artifacts, not the worker's large descriptive data.
    sources = [Path(__file__)]
    for directory in ('tools/rbint', 'tools/rblane', 'tools/rbfit'):
        sources += sorted((ROOT/directory).glob('*.py'))
    sources += [ROOT/p for p in (
        'artifacts/rbfit/shrinkage_v1/idm_population.json',
        'artifacts/rbfit/shrinkage_v1/idm_posteriors.json.gz',
        'artifacts/rbfit/shrinkage_v1/idm_shrinkage.json',
        'artifacts/rbfit/lane_raw_c5602102891e/lc2013_population.json',
        'docs/paper_results/rbunc/observed_arrivals.json',
        'docs/paper_results/rb2cal/v1/fit/calibration.json',
        'artifacts/rb2lphys/learned_v1.json')]
    with ExitStack() as stack:
        guards = []

        def loader(*a, **k):
            models, receipt = load_candidate(*a, **k)
            for model in models:
                guards.append(stack.enter_context(forbid_learned_corrections(model)))
            return models, receipt

        run_wm.main(model_loader=loader, future_builder=candidate_future,
                    identity_sources=sources, worlds=(969420, 969421), future_indices=(0,))
    tasks = [out/f'w{s}_k0.json' for s in (969420, 969421)]
    for path in tasks:
        task = load(path)
        assert len(task['episodes']) == 5
        assert task['future_sampling']['population_audit']['conserved']
        assert task['future_sampling']['population_audit']['finite']
        assert all('endpoint60' in e for e in task['episodes'])
    save_new(out/'DEV_CANARY_PASS.json', dict(passed=True, development_only=True,
        worlds=[969420, 969421], K_completed=1, arms=5, horizon_s=600,
        learned_corrections_guard=True, recipe_sha256=digest(recipe),
        task_sha256={p.name: digest(p) for p in tasks}, encoder_calls=[len(g) for g in guards],
        final_panel_tasks=False))
    print('INTEGRATED DEVELOPMENT CANARY PASS', flush=True)


if __name__ == '__main__':
    main()
