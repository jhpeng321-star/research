import copy
import json
import math

import pytest

from tools.rbunc.interim_watch import inspect_world, summarize


def make_panel(tmp_path):
    shard = tmp_path/'shard0'
    shard.mkdir()
    arms = ['native', 'ctrl0', 'mappo', 'pi', 'circles']
    for k in range(8):
        clocks = dict(team_delay_vs=float(20+k), road_delay_vs=float(10+k), source_wait_vs=10.)
        task = dict(schema='rbeval/1', seed=969420, future=k, arms=arms,
            identity_sha256='frozen-candidate',
            future_sampling=dict(variant='FULL', future_sampling='hidden_state', future=k,
                physics_noise=False, embedding_noise=False,
                fitted_artifacts=dict(idm_population_sha256='population', idm_posterior_sha256='posterior'),
                learned_arrival_sha256='rates', demand_rate_draws=[[[.5, .1]]]*5,
                population_audit=dict(finite=True, conserved=True)),
            episodes=[dict(seed=969420, future=k, arm=arm, initial_sha256='start',
                           clocks={'120':clocks}) for arm in arms])
        (shard/f'w969420_k{k}.json').write_text(json.dumps(task))
    return dict(initial_sha256='start', real=dict(team_delay_vs=23., road_delay_vs=13., source_wait_vs=10.))


def mutate(tmp_path, fn, k=7):
    path = tmp_path/f'shard0/w969420_k{k}.json'
    task = json.loads(path.read_text())
    fn(task)
    path.write_text(json.dumps(task))


def test_complete_native_fit_spread_crps_and_range_are_independently_recomputed(tmp_path):
    target = make_panel(tmp_path)
    record = inspect_world(tmp_path, 969420, target)
    result = summarize([record])
    team = result['metrics']['team_delay_vs']
    oracle = sum(abs(x-23.) for x in range(20, 28))/8 - .5*sum(
        abs(a-b) for a in range(20, 28) for b in range(20, 28))/64
    assert team['empirical_crps']['mean'] == pytest.approx(oracle)
    assert team['bias']['mean'] == .5
    assert team['within_K_sd']['mean'] == pytest.approx(math.sqrt(6.))
    assert team['range_covered']['mean'] == 1.
    assert team['bias']['world_bootstrap_ci95'] is None
    assert result['range_exchangeability_reference'] == 7/9
    assert not result['physical_sampling_scale_changed']
    assert record['source_receipts'][0]['learned_arrival_observations'] is None


def test_missing_k8_never_creates_an_ensemble_score(tmp_path):
    target = make_panel(tmp_path)
    # Rename within the test fixture, preserving content while making k7 absent.
    (tmp_path/'shard0/w969420_k7.json').rename(tmp_path/'shard0/incomplete.json')
    assert inspect_world(tmp_path, 969420, target) is None
    assert summarize([]) is None


@pytest.mark.parametrize('corruption,message', [
    ('identity', 'mixed candidate'), ('rates', 'share demand-rate'),
    ('start', 'target/start'), ('conservation', 'source audit'),
    ('future', 'task identity'), ('ledger', 'source/road'), ('noise', 'source audit')])
def test_bad_source_pairing_identity_or_ledger_fails_closed(tmp_path, corruption, message):
    target = make_panel(tmp_path)
    def corrupt(task):
        if corruption == 'identity': task['identity_sha256'] = 'other'
        if corruption == 'rates': task['future_sampling']['demand_rate_draws'] = [[[.6, .1]]]+[[[.5, .1]]]*4
        if corruption == 'start': task['episodes'][0]['initial_sha256'] = 'other'
        if corruption == 'conservation': task['future_sampling']['population_audit']['conserved'] = False
        if corruption == 'future': task['future'] = 2
        if corruption == 'ledger': task['episodes'][0]['clocks']['120']['team_delay_vs'] = 999.
        if corruption == 'noise': task['future_sampling']['physics_noise'] = True
    mutate(tmp_path, corrupt)
    with pytest.raises(ValueError, match=message):
        inspect_world(tmp_path, 969420, target)


def test_only_explicit_fit_task_paths_are_eligible_and_duplicates_rejected(tmp_path):
    target = make_panel(tmp_path)
    with pytest.raises(ValueError, match='explicit FIT'):
        inspect_world(tmp_path, 969432, target)
    other = tmp_path/'shard1'
    other.mkdir()
    (other/'w969420_k0.json').write_bytes((tmp_path/'shard0/w969420_k0.json').read_bytes())
    with pytest.raises(ValueError, match='duplicate'):
        inspect_world(tmp_path, 969420, target)


def test_world_clustering_and_mixed_candidate_rejection(tmp_path):
    target = make_panel(tmp_path)
    first = inspect_world(tmp_path, 969420, target)
    second = copy.deepcopy(first)
    second['world'] = 969421
    second['target']['real']['team_delay_vs'] += 10.
    team = summarize([first, second])['metrics']['team_delay_vs']
    assert team['bias']['world_bootstrap_ci95'] == [-9.5, .5]
    assert team['real_across_subset_world_sd'] == pytest.approx(math.sqrt(50.))
    second['identity_sha256'] = 'other-source'
    with pytest.raises(ValueError, match='mixed candidate'):
        summarize([first, second])
