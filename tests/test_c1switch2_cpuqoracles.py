"""Only previously unstarted original full-prefix alternatives may continue."""
import copy

import pytest

from tools.c1switch2 import cpuqoracles
from tools.c1switch2.design import CONTROLLERS, measurement_indices


def plan():
    return dict(predecessor=cpuqoracles.OLD, historical_sim_s=5460,
        new_sim_s=14640, new_gpu_s=0, branches=[dict(seed=s, controller=c,
        sim_s=360+20*measurement_indices(s)[0]) for s in cpuqoracles.SEEDS for c in CONTROLLERS])


def test_exact_cap_and_all_original_six_way_sets():
    value = plan()
    cpuqoracles.validate(value)
    assert sum(b['sim_s'] for b in value['branches']) == 14640
    compile(cpuqoracles.SUPERVISOR_CPUQ, '<original-cpuq-oracles>', 'exec')
    assert '<=14640' in cpuqoracles.SUPERVISOR_CPUQ
    assert 'branches=24,planned_sim_s=14640' in cpuqoracles.SUPERVISOR_CPUQ


@pytest.mark.parametrize('mutation', ('subset', 'new_world', 'cap', 'prefix'))
def test_rejects_replacement_world_subset_or_budget_expansion(mutation):
    value = copy.deepcopy(plan())
    if mutation == 'subset':
        value['branches'].pop()
    elif mutation == 'new_world':
        value['branches'][0]['seed'] = 969141
    elif mutation == 'cap':
        value['new_sim_s'] += 1
    else:
        value['branches'][0]['sim_s'] += 20
    with pytest.raises(ValueError):
        cpuqoracles.validate(value)
