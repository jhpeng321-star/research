"""Original completed WM picks admit only their previously unstarted oracle jobs."""
import copy

import pytest

from tools.c1switch2 import wmoracles
from tools.c1switch2.design import CONTROLLERS


def plan():
    return dict(predecessor=wmoracles.OLD, host='gpuR', new_sim_s=6000, new_gpu_s=0,
        historical_sim_s=2700, not_before_unix=wmoracles.NOT_BEFORE,
        branches=[dict(seed=s, controller=c, id=f'wm__s{s}__{c}', sim_s=cost)
                  for s, cost in ((969139, 380), (969143, 620)) for c in CONTROLLERS])


def test_all_original_unstarted_futures_fit_residual_host_allocation():
    value = plan()
    wmoracles.validate_plan(value)
    assert sum(b['sim_s'] for b in value['branches']) == 6000
    assert 2700+6000 <= 16320
    compile(wmoracles.SUPERVISOR, '<original-wm-oracles>', 'exec')


@pytest.mark.parametrize('mutation', ['failed_world', 'missing_controller', 'cost', 'gpu', 'history'])
def test_no_failed_world_replay_partial_library_or_budget_change(mutation):
    value = copy.deepcopy(plan())
    if mutation == 'failed_world':
        value['branches'][0]['seed'] = 969147
    elif mutation == 'missing_controller':
        value['branches'].pop()
    elif mutation == 'cost':
        value['branches'][0]['sim_s'] += 20
    elif mutation == 'gpu':
        value['new_gpu_s'] = 1
    else:
        value['historical_sim_s'] = 0
    with pytest.raises(ValueError):
        wmoracles.validate_plan(value)
