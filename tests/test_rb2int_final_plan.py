from collections import Counter

from tools.rb2int.audit import ROOT, load
from tools.rb2int.final_plan import build_plan, MODELS


def test_final_world_plan_reuses_reserved_baselines_and_keeps_paired_v5_hardware():
    reg = load(ROOT/'docs/rb2int/EVALUATION_REGISTRATION.json')
    plan = build_plan(reg)
    assert len({j['id'] for j in plan['jobs']}) == 216
    counts = Counter(j['model'] for j in plan['jobs'])
    assert counts == dict(WM2=50, **{'FULL-v4':50,'FULL-v5':58,'FULL-v5-LP':58})
    mapping = {(j['model'],j['seed']):j for j in plan['jobs']}
    for seed in reg['final_comparison_worlds']:
        assert mapping['FULL-v5',seed]['host'] == mapping['FULL-v5-LP',seed]['host']
        for model in MODELS:
            if seed in reg['reserved8'] and model in ('WM2','FULL-v4'):
                assert (model,seed) not in mapping
            elif model in ('WM2','FULL-v4'):
                assert mapping[model,seed]['host'] == ('gpuR' if model == 'WM2' else 'gpuA')
    assert all(j['futures'] == list(range(8)) for j in plan['jobs'])


def test_primary_small_cells_are_prioritized_without_changing_membership():
    reg = load(ROOT/'docs/rb2int/EVALUATION_REGISTRATION.json')
    jobs = build_plan(reg)['jobs']
    rank = [(not j['primary'],j['stratum_worlds'],j['seed']) for j in jobs]
    assert rank == sorted(rank)
    assert jobs[0]['stratum_worlds'] == 2
    assert jobs[-1]['primary'] is False
