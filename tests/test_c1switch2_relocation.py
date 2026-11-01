"""The unstarted BB arm may move hosts once, without widening its science budget."""
import copy
from datetime import datetime
import json

import pytest

from tools.c1switch.storage import canonical
from tools.c1switch2 import relocation
from tools.c1switch2 import bbrelocate, modelops
from tools.c1switch2.design import assigned, local_budget, worlds
from tools.c1switch2.supervisor import episode_jobs


def binding():
    return dict(relocation=relocation.plan(), enabled_arms=['bb'], dispatch_phase='model',
                cpu_host='gpuR', sim_s=dict(bb=79800))


def test_exact_unused_arm_keeps_all_sixteen_worlds_and_original_global_cap():
    value = binding()
    jobs = episode_jobs(value, 'gpuR')
    assert len(jobs) == 16 and len({name for name, _ in jobs}) == 16
    assert {int(args[-1]) for _, args in jobs} == {w['seed'] for w in worlds()}
    assert episode_jobs(value, 'cpuQ') == []
    limits = relocation.execution_budget(value)
    assert limits['bb'] == sum(local_budget(host)['bb'] for host in ('cpuQ', 'gpuR')) == 79800
    assert sum(limits.values()) == 79800


@pytest.mark.parametrize('mutation', ['subset', 'wm', 'prior', 'cap', 'host'])
def test_relocation_cannot_retry_wm_skip_worlds_or_expand_budget(mutation):
    value = binding()
    if mutation == 'subset':
        value['relocation']['seeds'].pop()
    elif mutation == 'wm':
        value['enabled_arms'] = ['wm']
    elif mutation == 'prior':
        value['relocation']['prior_sim_s'] = 1
    elif mutation == 'cap':
        value['relocation']['sim_s'] += 1
    else:
        value['cpu_host'] = 'cpuQ'
    with pytest.raises(ValueError, match='exact unused BB'):
        relocation.validate(value)


def test_binding_requires_both_main_authority_and_exact_prospective_registration():
    value = binding()
    line = '- S-C1SWITCH2 BB RELOCATION REGISTERED ' + canonical(value['relocation'])
    for text in ('07:07 MAIN DECISION', line):
        with pytest.raises(ValueError, match='authority'):
            relocation.validate(value, text)
    assert relocation.validate(value, '07:07 MAIN DECISION\n' + line)
    wrong = copy.deepcopy(value)
    wrong['relocation']['gpu_s'] += 1
    with pytest.raises(ValueError):
        relocation.validate(wrong, '07:07 MAIN DECISION\n' + line)


def test_original_partitions_and_original_budgets_are_unchanged():
    original = dict(cpu_host='gpuR')
    assert relocation.execution_budget(original) == local_budget('gpuR')
    assert all(relocation.execution_host(original, w['seed']) == assigned(w['seed']) for w in worlds())


def test_relocated_relay_does_not_release_after_original_four_worlds(monkeypatch):
    calls = []
    config = dict(relocation=relocation.plan(), cpus=dict(gpuR='/root/own_cpu'),
                  gpus=dict(gpuA='/root/own_gpu', gpuR='/root/own_gpu'), engine=dict(model='bb'))

    def ssh(host, code, timeout):
        calls.append(code)
        if 'expected=' in code:
            assert 'expected=16' in code
            return dict(stop=None, episodes=4, expected=16, complete=False)
        assert 'if False and not path.exists()' in code
        return dict(stop=None, finished=None)

    monkeypatch.setattr(modelops, 'ssh', ssh)
    monkeypatch.setattr(modelops, 'transfer', lambda *args: 0)
    result = modelops.relay_cycle(config)
    assert result['episodes_done'] is False and len(calls) == 3


def test_priority_time_gate_precedes_every_gpu_allocation(tmp_path, monkeypatch):
    move = relocation.plan()
    bound, registration = tmp_path/'bound.json', tmp_path/'registration.md'
    bound.write_text(json.dumps(dict(relocation=move)))
    registration.write_text('07:07 MAIN DECISION\n- S-C1SWITCH2 BB RELOCATION REGISTERED ' + canonical(move))
    monkeypatch.setattr(bbrelocate, 'BOUND_PATH', bound)
    monkeypatch.setattr(bbrelocate, 'REGISTRATION', registration)
    monkeypatch.setattr(bbrelocate, 'approval', lambda text: True)
    monkeypatch.setattr(bbrelocate, 'cpu_authority', lambda text: True)
    monkeypatch.setattr(bbrelocate.time, 'time', lambda: datetime.fromisoformat(move['not_before']).timestamp()-1)
    monkeypatch.setattr(bbrelocate, 'resources', lambda config: pytest.fail('no allocation during priority window'))
    with pytest.raises(ValueError, match='not before11:30'):
        bbrelocate.start()


def test_registry_conflict_is_rejected_before_gpu_resources_or_claim(tmp_path, monkeypatch):
    move = relocation.plan()
    bound, registration = tmp_path/'bound.json', tmp_path/'registration.md'
    bound.write_text(json.dumps(dict(relocation=move)))
    registration.write_text('07:07 MAIN DECISION\n- S-C1SWITCH2 BB RELOCATION REGISTERED '
                            + canonical(move) + '\n- other task reserves seed969136')
    monkeypatch.setattr(bbrelocate, 'BOUND_PATH', bound)
    monkeypatch.setattr(bbrelocate, 'REGISTRATION', registration)
    monkeypatch.setattr(bbrelocate, 'approval', lambda text: True)
    monkeypatch.setattr(bbrelocate, 'cpu_authority', lambda text: True)
    monkeypatch.setattr(bbrelocate, 'resources', lambda config: pytest.fail('allocated before audit'))
    with pytest.raises(ValueError, match='registry conflicts'):
        bbrelocate.start()
