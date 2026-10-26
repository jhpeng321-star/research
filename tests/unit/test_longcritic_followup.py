"""Share selection and single-CAV long-arm accounting without any simulator."""
import copy
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wms.control.real_share import development_world, select_largest, split_provider, calibrate_share, CANDIDATES

spec = importlib.util.spec_from_file_location('longcritic_fixtures', Path(__file__).with_name('test_longcritic.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
helper, loop, fast = fixtures.helper, fixtures.loop, fixtures.fast


def test_combined_group_single_reserve_and_failed_attempts(tmp_path, monkeypatch):
    cfg = helper('test_fastloop').fixture(tmp_path)
    cfg['collect'].update(branch_horizon=300, long_group={'enabled': True}, long_single_hold={'enabled': True})
    assert loop.long_group_sim_s(cfg) == 1800
    state = loop.load_state(cfg)
    reg = loop.read_json(cfg['registration'])
    reg['budget'].update(sim_s_per_cycle=1e6, sim_s_total=1e6)
    Path(cfg['registration']).write_text(json.dumps(reg))
    state['registration_sha256'] = loop.sha256(cfg['registration'])
    loop.save_state(cfg, state)
    root = Path(cfg['loop_root'])
    monkeypatch.setattr(loop, 'disk_guard', lambda *a: None)
    monkeypatch.setattr(fast, '_world', lambda *a: (_ for _ in ()).throw(ConnectionError('actual zero')))
    for _ in range(4):
        with pytest.raises(RuntimeError):
            fast.collect(loop, cfg, state, {'attempt': 1}, root)
    with pytest.raises(loop.Refused, match='8,000'):
        fast.collect(loop, cfg, state, {'attempt': 5}, root)
    rows = [json.loads(s) for s in (root/'seed_ledger.jsonl').read_text().splitlines()]
    assert sum(r.get('long_credit_sim_s_charged', 0) for r in rows) == 7200
    assert sum(r.get('long_single_sim_s_charged', 0) for r in rows) == 3600
    assert sum(r.get('long_group_sim_s_charged', 0) for r in rows) == 3600
    cfg['collect']['worlds_per_cycle'] = 13
    assert loop.long_group_sim_s(cfg) == 7800
    cfg['collect']['worlds_per_cycle'] = 14
    with pytest.raises(loop.Refused, match='8,000'):
        loop.long_group_sim_s(cfg)


def test_single_hold_stable_upstream_id_selection_and_full_ledger(monkeypatch):
    from traffic_shepherd.c4.sumocredit import PairExtras
    extra = PairExtras(long_single_horizon=300, long_single_time=360)
    frame = SimpleNamespace(present=np.ones(4, bool), is_cav=np.ones(4, bool),
                            lane=np.array([0, 1, 2, 0]), s=np.array([-10., -10., -20., 5.]),
                            vid_index=np.arange(4))
    env = SimpleNamespace(frame=frame, k=4, builder=SimpleNamespace(vehicle_ids=['z', 'a', 'ramp', 'past']))
    monkeypatch.setattr(extra, '_load', lambda *a: None)
    calls = []
    def run(env, point, horizon, step):
        calls.append(horizon)
        acceleration, lanes = step(env, None)
        assert acceleration[0, 1] == 0
        assert np.isnan(acceleration[0, [0, 2, 3]]).all()
        assert lanes == {}
        return {'J': 80., 'road': 50., 'source': 30., 'sim_s': 300, 'collisions': 0, 'teleports': 0}
    monkeypatch.setattr(extra, '_run_arm', run)
    native = {'J': 100., 'road': 70., 'source': 30., 'collisions': 0, 'teleports': 0}
    out = extra.run_point(env, {'t_s': 360}, {('B', 300): native}, policy_factory=None,
                          seed=968150, world='synthetic', branch_h=300)
    assert calls == [300] and out['sim_s'] == 300
    assert out['single_long']['selected_vehicle'] == 'a'
    assert out['single_long']['D_hold'] == 20
    assert out['single_long']['source_wait'] == {'B': 30., 'SINGLE_HOLD': 30.}
    assert extra.sim_s_per_point(300) == 300
    frame.s[:] = 1.
    out = extra.run_point(env, {'t_s': 360}, {('B', 300): native}, policy_factory=None,
                          seed=968150, world='synthetic', branch_h=300)
    assert out['sim_s'] == 0 and len(calls) == 1


def test_real_share_world_split_and_largest_nonregressing_rule():
    assert [s for s in range(968129, 968142) if development_world(s)] == [968135, 968137, 968138]
    seeds = np.repeat([968135, 968137, 968138], 2)
    errors = np.array([[10.] * 6, [9.] * 6, [8.] * 6, [12.] * 6, [11.] * 6])
    selected, receipt = select_largest(errors, seeds)
    assert selected == .7 and receipt['candidates']['0.9']['eligible'] is False
    errors[4] = 7.
    assert select_largest(errors, seeds)[0] == 1.
    assert select_largest(errors, np.ones(6))[0] == .3


def test_share_selection_freezes_actor_wm_rng_and_development_labels(tmp_path):
    D = helper('test_c4_dreamer')
    trainer = D._trainer(gamma=1.)
    n = 26
    path = tmp_path/'real.npz'
    path.write_bytes(b'synthetic provenance')
    source = SimpleNamespace(path=path, seed=965140, _states=D._state(n),
        _returns=torch.full((n,), -20.), provenance={'source_kind': 'current_policy',
            'policy_checkpoint': 'pinned', 'seeds': np.repeat(np.arange(968129, 968142), 2)})
    fit, dev = split_provider(source)
    assert not set(fit.indices) & set(dev.indices)
    assert not set(fit.provenance['seeds']) & set(dev.provenance['seeds'])
    native = copy.deepcopy(trainer.critic).requires_grad_(False)
    trainer.longcritic = SimpleNamespace(native_critic=native,
        targets=lambda trainer, batch: ((batch.state, torch.zeros(2), torch.ones(2)), {}))
    actor, wm = copy.deepcopy(trainer.policy.state_dict()), copy.deepcopy(trainer.world_models[0].state_dict())
    sampler, rng = copy.deepcopy(trainer.start_states.state_dict()), torch.get_rng_state()
    receipt = calibrate_share(trainer, fit, dev, policy_sha256='pinned')
    assert receipt['selected_share'] in CANDIDATES
    assert trainer.config.real_return_mix == receipt['selected_share']
    assert torch.equal(rng, torch.get_rng_state())
    D._assert_equal(actor, trainer.policy.state_dict())
    D._assert_equal(wm, trainer.world_models[0].state_dict())
    assert trainer.start_states.state_dict() == sampler
    assert all(p.grad is None for p in trainer.policy.parameters())
    receipt = calibrate_share(trainer, fit, dev, policy_sha256='different')
    assert receipt['selected_share'] == .3 and 'refused' in receipt


def test_enabled_share_mixes_real_and_difference_without_imagined_levels():
    R = helper('test_policy14_critic')
    provider = R.Provider()
    trainer = R._trainer(provider=provider, real_return_mix=.3, real_return_scale=1.)
    trainer._return_scale = lambda returns: 1.
    expected = ((trainer.critic(provider.batch.states) - provider.batch.returns) ** 2).mean()
    trainer.longcritic = SimpleNamespace(loss=lambda trainer, starts, scale:
        (trainer.critic.bias.square() + 2., {'longcritic_test': True}))
    metrics = trainer.update()
    assert metrics['critic_imagined_loss'] is None
    assert metrics['critic_loss'] == pytest.approx(.3 * float(expected.detach()) + .7 * 2.)
    assert metrics['real_return_mix'] == .3


def test_policy_command_requires_and_threads_paired_native_returns(tmp_path):
    cfg = helper('test_fastloop').fixture(tmp_path)
    state = loop.load_state(cfg)
    cfg['policy']['longcritic'] = True
    with pytest.raises(loop.Refused, match='paired native'):
        loop.policy_cmd(cfg, state, tmp_path)
    state['native_real_returns'] = '/owned/native300.npz'
    argv, _ = loop.policy_cmd(cfg, state, tmp_path)
    assert argv[argv.index('--longcritic-native-returns') + 1] == '/owned/native300.npz'
