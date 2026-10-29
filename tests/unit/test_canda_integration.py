"""S-INTEGRATE candidate-A launch-runtime integration tests (CPU only).

Bindings verified here are the ASTRA4 morning-GO items MAIN listed at 06:00:
flag-default parity, executor/mirror shared-law binding, sole writer, monitors
ON in the candidate-A config, stale-secant direct credit OFF, --new-phase
resume refusing silent overrides, and nine-row N4 exactness (CPU where
possible, plus the registered update-181 reference receipt).
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.unit.test_longmem import trainer as make_trainer, assert_tree_close

ROOT = Path(__file__).resolve().parents[2]
CANDA_CONFIG = ROOT / 'configs/c4_policy14_candA.toml'


def read_canda_config():
    import sys
    sys.path[:0] = [str(ROOT / 'src')]
    from wms.control.train_cli import _read_config
    return _read_config(CANDA_CONFIG)


def test_integration_switches_default_off():
    """Every merged default-OFF flag is OFF without explicit opt-in."""
    from wms.control.dreamer import DreamerConfig
    from wms.control.gated300 import LongCreditConfig
    from wms.control import execution
    from wms.control.parallel_futures import parallel_forward, worker_count
    cfg = DreamerConfig()
    assert cfg.exploit_monitor is False and cfg.exploit_disable_direct_credit is False
    assert cfg.exploit_ensemble_pessimism is False and cfg.exploit_stop_on_conflict is False
    long = LongCreditConfig()
    assert long.enabled is False and long.mode == 'cct1' and long.score_training_horizon is False
    assert execution.mode() == 0 and execution.enabled() is False
    assert parallel_forward() is False and worker_count() == 1


def test_flag_default_update_parity():
    """Explicit-OFF integration switches leave a full update bitwise identical."""
    from wms.model import WmSModel, ModelSpec
    base = make_trainer(1, 2, True, horizon=2, batch=1, counterfactual=True, real_world=True,
                        future_samples=8, pessimistic_credit=True, uncertainty_weight=0.,
                        exploit_monitor=False, exploit_disable_direct_credit=False,
                        exploit_ensemble_pessimism=False)
    base.world_models = [WmSModel(ModelSpec(rollout_mode='driver_params',
                                            history_hidden=8, k_hidden=8)).eval().requires_grad_(False)]
    explicit = copy.deepcopy(base)
    explicit.config = copy.deepcopy(explicit.config)
    torch.manual_seed(3141)
    a = base.update()
    rng = torch.get_rng_state()
    torch.manual_seed(3141)
    b = explicit.update()
    assert torch.equal(rng, torch.get_rng_state())
    for key in ('actor_loss', 'critic_loss', 'returns', 'actor_grad_norm', 'memory_spill'):
        assert_tree_close(a[key], b[key], atol=0, rtol=0)
    for x, y in zip(base.actor_parameters, explicit.actor_parameters):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_candA_config_resolves_with_monitors_on():
    """configs/c4_policy14_candA.toml resolves to the registered candidate-A flags."""
    from wms.control.dreamer import DreamerConfig
    raw = read_canda_config()
    cfg = DreamerConfig(**raw['trainer'])
    # monitors ON, unqualified pessimism OFF, anchor OFF (measured fallback)
    assert cfg.exploit_monitor is True and cfg.exploit_disable_direct_credit is True
    assert cfg.exploit_ensemble_pessimism is False and cfg.exploit_stop_on_conflict is False
    assert cfg.native_anchor is False and cfg.checkpoint_every == 1
    # unchanged registered training base
    assert cfg.pessimistic_credit is True and cfg.future_samples == 8
    assert cfg.entrant_control is True and cfg.deterministic_crn is True
    assert cfg.real_credit_individual_floor is True
    assert cfg.imagination_microbatch == 1 and cfg.imagination_checkpoint_segment == 2  # c31d6d70 memory-only recipe (POLSPEED2)
    assert cfg.imagination_offload_futures and cfg.imagination_offload_saved_tensors
    assert 'hold_move' not in raw['trainer']  # HOLD shelved; not in the chain
    # nine-row weights, 60% focus, dense 90s horizon, gates off
    cf = raw['counterfactual']
    assert cf['curriculum_updates'] == [0, 1]
    assert cf['w_individual'] == [8.9, 8.9] and cf['w_group'] == [9.9, 9.9]
    assert cf['w_coop'] == [8.9, 8.9] and cf['w_team'] == [10.3, 10.3]
    assert cf['w_all'] == [1.0, 1.0]
    assert raw['data']['focus_cell_fraction'] == 0.6
    assert raw['policy14']['fd_gate'] is False and raw['policy14']['free_road_gate'] is False
    horizon = raw['horizon']
    assert horizon['horizons_s'] == [10, 90] and horizon['entry_pad_max'] == [32, 288]
    assert horizon['curriculum_updates'] == [0, 1]
    # S-LAUNCHA 18:25 fallback (23:25 rule): phase 1 dense 90 s; the mix-qualified
    # CCT-2 stays bound as the registered second-phase authority
    long = raw['long_credit']
    assert long['enabled'] is False and long['mode'] == 'gain_only'
    assert long['cct2_file'] == 'authority/CCT2_{wm_sha256}_mix.json'
    assert long['score_training_horizon'] is False
    # S-INTEGRATE2: imagination-mirror leader rule ON (C4_SUMO_LEADER_RULE=1 launch env)
    assert raw['execution']['leader_rule'] is True
    # the six-row preset is NOT in the candidate-A chain
    chain = {str(p) for p in [CANDA_CONFIG]}
    text = CANDA_CONFIG.read_text(encoding='utf-8')
    assert 'sixrow' not in text


def test_executor_and_mirror_share_one_law(monkeypatch):
    """Imagination's executed law IS sumo_idm: one shared function, not a copy."""
    from wms.control import alwayscmd, sumo_idm
    calls = []
    real = sumo_idm.executed_accel

    def counting(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(sumo_idm, 'executed_accel', counting)
    from wms.control import native_cav
    holder = {'gap': torch.tensor([[15., 40.]]), 'dv': torch.tensor([[3., 0.]]),
              'has': torch.tensor([[True, False]])}
    monkeypatch.setattr(native_cav, 'leader_gap_dv', lambda state: holder.values())
    state = {'v_mps': torch.tensor([[12., 5.]], dtype=torch.float64),
             's_m': torch.tensor([[400., 20.]]),
             'lane': torch.tensor([[1, 2]]), 'present': torch.ones(1, 2, dtype=torch.bool),
             'length_m': torch.full((1, 2), 5.)}
    out = alwayscmd.executed(state, torch.zeros((1, 2), dtype=torch.float64))
    assert len(calls) == 1 and out.shape == (1, 2)
    # and the composed base comes from the same module
    base_calls = []
    real_base = sumo_idm.base_accel
    monkeypatch.setattr(sumo_idm, 'base_accel',
                        lambda *a, **k: base_calls.append(1) or real_base(*a, **k))
    alwayscmd.inferred_base(state)
    assert base_calls


def test_mode2_sole_writer_commands_every_cav_once(monkeypatch):
    """Mode-2: only setSpeed(-1)+setAcceleration per CAV; zero-residual included."""
    from tests.unit.test_execparity_alwayscmd import domain
    from traffic_shepherd.ownership import CommandOwnership
    monkeypatch.setenv('C4_EXECUTION_PARITY', '2')
    api = SimpleNamespace(vehicle=domain())
    controller = SimpleNamespace(config=SimpleNamespace(control_dt_s=1., sensor_range_m=100.))
    own = CommandOwnership(api, {'accepted'}, controller, cav_speed_mode=27)
    frame = SimpleNamespace(vehicle_id='accepted', timestamp_s=300., speed_mps=8.,
                            allowed_speed_mps=28., front_coverage_m=100., rear_coverage_m=100.,
                            front=None, rear=None)
    rows = own.step({'accepted': frame}, now_s=300., live_ids={'accepted'},
                    wall_commands={'accepted': .3}, defer_wall=True)
    assert rows[0].owner == 'wall' and not api.vehicle.calls
    issued = own.execute_residual({'accepted': 0.0}, -3., 1.2,
                                  now_s=300., live_ids={'accepted'})
    kinds = [c[0] for c in api.vehicle.calls]
    # sole longitudinal writer: exactly one setSpeed(-1) + one setAcceleration,
    # no slowDown/setMaxSpeed competing path, zero-residual CAV commanded too.
    assert kinds == ['setSpeed', 'setAcceleration']
    assert set(issued) == {'accepted'} and issued['accepted'] != 0.0


def test_stale_secant_direct_credit_disabled_and_flagged():
    """exploit_disable_direct_credit skips the stale-secant real term, flags it."""
    base = make_trainer(counterfactual=True, exploit_disable_direct_credit=False)
    disabled = make_trainer(counterfactual=True, exploit_disable_direct_credit=True)

    class Credit:
        def __init__(self, trainer):
            self.scale = trainer.actor_parameters[0].sum() * 0.

        def loss(self, trainer, scale):
            direct = 0.05 * trainer.actor_parameters[0].square().sum() + self.scale
            return direct, {'direct_real_credit': True}

    torch.manual_seed(61)
    base.looplong_real_credit = Credit(base)
    with_credit = base.update()
    torch.manual_seed(61)
    disabled.looplong_real_credit = Credit(disabled)
    off = disabled.update()
    assert with_credit['direct_real_credit'] is True
    assert with_credit['direct_real_credit_grad_norm'] > 0.
    assert off['exploit_direct_credit_disabled'] is True
    assert off['direct_real_credit_grad_norm'] == 0.
    assert off['actor_loss'] != with_credit['actor_loss']
    torch.manual_seed(61)
    none_trainer = make_trainer(counterfactual=True)
    torch.manual_seed(61)
    none = none_trainer.update()
    assert off['actor_loss'] == pytest.approx(none['actor_loss'], abs=1e-9)


def test_new_phase_resume_refuses_silent_overrides(tmp_path, monkeypatch):
    """--new-phase refuses a differing TOML before mutating weights; checkpoints each update."""
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    old = make_trainer(checkpoint_every=1)
    path = tmp_path / 'c1.pt'
    old.save_checkpoint(path)
    tampered = make_trainer(checkpoint_every=1)
    from dataclasses import replace as _replace
    tampered.reward_weights = _replace(old.reward_weights,
                                       delay=old.reward_weights.delay + 1.0)
    with pytest.raises(ValueError):
        tampered.load_checkpoint(path, new_phase=True)
    assert tampered.update_count == 0  # refused before loading/mutating
    fresh = make_trainer(checkpoint_every=1)
    fresh.output_dir = tmp_path
    fresh.load_checkpoint(path, new_phase=True)
    fresh.update()
    assert (tmp_path / 'policy_000001.pt').is_file()


NINE_ROW_CF = dict(enabled=True, max_starts=1, n_individual=3, n_group=1, n_flip=3,
                   group_min_size=3, group_max_size=3, curriculum_updates=[0],
                   w_individual=[8.9], w_coop=[8.9], w_group=[9.9], w_team=[10.3], w_all=[1.])


@pytest.mark.parametrize('forward', [False, True])
def test_nine_row_n4_parallel_exact_on_cpu(tmp_path, monkeypatch, forward):
    """The candidate-A path (nine rows, N4 workers) is bitwise the serial update."""
    pytest.importorskip('cloudpickle')
    from wms.control.counterfactual import CounterfactualConfig, build_variants
    from wms.model import WmSModel, ModelSpec
    serial = make_trainer(1, 2, True, horizon=2, batch=1, counterfactual=True, real_world=True,
                          future_samples=8, pessimistic_credit=True, uncertainty_weight=0.)
    serial.world_models = [WmSModel(ModelSpec(rollout_mode='driver_params',
                                              history_hidden=8, k_hidden=8)).eval().requires_grad_(False)]
    serial.counterfactual = CounterfactualConfig(**NINE_ROW_CF)
    variants = build_variants(serial.start_states.batch.state, serial.counterfactual)
    assert variants.n_rows == 9  # base + 3 individual + group + 3 flip + all
    parallel = copy.deepcopy(serial)
    parallel.output_dir = tmp_path
    torch.manual_seed(1603)
    expected = serial.update()
    rng = torch.get_rng_state()
    monkeypatch.setenv('WMS_SPEED_FUTURE_WORKERS', '4')
    monkeypatch.setenv('WMS_SPEED_PARALLEL_FORWARD', '1' if forward else '0')
    torch.manual_seed(1603)
    actual = parallel.update()
    assert torch.equal(rng, torch.get_rng_state())
    for key in ('actor_loss', 'returns', 'actor_grad_norm', 'critic_loss', 'memory_spill'):
        assert_tree_close(expected[key], actual[key], atol=0, rtol=0)
    for x, y in zip(serial.actor_parameters, parallel.actor_parameters):
        torch.testing.assert_close(x.grad, y.grad, atol=0, rtol=0)
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    assert_tree_close(serial.actor_optimizer.state_dict(), parallel.actor_optimizer.state_dict(),
                      atol=0, rtol=0)


def test_nine_row_reference_receipt_matches_registered_update181():
    """The committed nine-row N4 receipt: 158 bitwise leaves at 350.756 s/update."""
    report = json.loads((ROOT / 'tools/polspeed2/evidence/REPORT.json').read_text(encoding='utf-8'))
    nine_n4 = report['nine_N4']
    assert nine_n4['parity']['bitwise_identical'] is True
    assert nine_n4['parity']['compared_leaves'] == 158
    assert nine_n4['parity']['n_diff'] == 0
    assert nine_n4['FGA_parity']['bitwise_identical'] is True
    assert abs(nine_n4['s_per_update'] - 350.756) < 0.01
    # the six-row preset is NOT bitwise (why candidate A keeps nine rows)
    for name in ('six_N2', 'six_N4', 'six_N8'):
        entry = report[name].get('parity') or report[name].get('six_serial_state_parity')
        assert entry is None or entry.get('bitwise_identical') is not True, name


def _write_cct2(path, wm, hg=300, mix=False):
    horizons = {}
    for h, passes in ((120, hg >= 120), (180, hg >= 180), (300, hg >= 300)):
        horizons[str(h)] = {'sigma': {120: 41.22, 180: 42.11, 300: 43.91}[h],
                            'threshold': 2 * {120: 41.22, 180: 42.11, 300: 43.91}[h],
                            'n_gain': 26 if passes else 0,
                            'gain_sign_agreement': 1.0 if passes else 0.5,
                            'verdict': 'PASS' if passes else 'FAIL'}
    doc = {'schema': 'gated300_cct2/1', 'wm_sha256': wm,
           'panels': ['pairs300', 'validation'], 'policy_authorized': True,
           'H_G': hg, 'horizons': horizons, 'entrant_control': True, 'deterministic_crn': True}
    if mix:
        doc['learned_mix_qualified'] = True
    path.write_text(json.dumps(doc), encoding='utf-8')
    return doc


def _write_driver(path):
    doc = {'schema': 'execparity_learned_idm/1', 'fit_controller': 'direct',
           'evaluation_data_used': False, 'fit_worlds': ['a'], 'score_worlds': ['b'],
           'parameters': {'accel': 1.1, 'decel': 2., 'time_headway': 1.2,
                          'min_gap': 2., 'v0': 28.}}
    path.write_text(json.dumps(doc), encoding='utf-8')


def _write_parity_receipt(path, driver):
    import hashlib
    doc = {'schema': 'execparity_always_commanded_acceleration/1',
           'backend_contract': 'always_commanded_ctrl0/1', 'measured_real_sumo': True,
           'zero_ctrl0_identity': True, 'passed': True, 'worlds': ['977001'],
           'driver_sha256': hashlib.sha256(driver.read_bytes()).hexdigest(),
           'native_arm_head_error': {'rmse': 0.1603},
           'c1_residual_execution_error': {'rmse': 0.1}, 'error': {'ours': {'rmse': 0.1}}}
    path.write_text(json.dumps(doc), encoding='utf-8')


def _authority(tmp_path, monkeypatch):
    import sys
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools/candA')]
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    monkeypatch.setenv('C4_EXECUTION_PARITY', '2')
    import importlib
    authority = importlib.import_module('authority')
    return importlib.reload(authority)


def test_authority_binds_and_refuses_on_mismatch(tmp_path, monkeypatch):
    authority = _authority(tmp_path, monkeypatch)
    wm = tmp_path / 'wm.pt'
    wm.write_bytes(b'wm-bytes')
    wmc = tmp_path / 'wm_config.json'
    wmc.write_bytes(b'{"wm": 1}')
    cct2 = tmp_path / 'CCT2.json'
    wm_sha = __import__('hashlib').sha256(wm.read_bytes()).hexdigest()
    _write_cct2(cct2, wm_sha, hg=300)
    driver = tmp_path / 'driver.json'
    _write_driver(driver)
    monkeypatch.setenv('C4_EXECUTION_DRIVER', str(driver))
    receipt = tmp_path / 'parity.json'
    _write_parity_receipt(receipt, driver)
    out = tmp_path / 'authority.json'
    args = SimpleNamespace(root=str(ROOT), wm_ckpt=str(wm), wm_config=str(wmc),
                           cct2=str(cct2), out=str(out), cct1=None, actor=None,
                           actor_checkpoint=None, learned_mix=False, mix_module=None,
                           parity_receipt=str(receipt), leader_rule_status=None,
                           leader_rule_commit=None, allow_dirty=True, dense_phase1=True)
    doc = authority.build(args)
    assert doc['actor_c1_sha256'] == '64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c'
    # declared credit mode matches the bound config bytes (phase 1 dense 90 s;
    # the CCT-2 file stays bound as the registered phase-2 authority)
    assert doc['long_credit']['mode'] == 'dense_90s_phase1'
    assert doc['long_credit']['H_G'] == 300 and doc['long_credit']['phase2']['H_long'] == 300
    assert doc['new_phase'] is True
    assert doc['monitors']['exploit_monitor'] is True
    # S-INTEGRATE2: the execparity3 slot is BOUND (source hashes + receipt), not OPEN
    assert doc['leader_rule']['status'] == 'bound-with-receipt'
    assert doc['leader_rule']['commit'] == 'a6a1894d'
    for name in authority.LEADER_RULE_FILES:
        assert doc['leader_rule']['files'][name] == doc['runtime_manifest'][name]
    v_args = SimpleNamespace(root=str(ROOT), authority=str(out))
    assert authority.verify(v_args)['verified'] is True
    # any mismatch refuses: tamper the WM checkpoint
    wm.write_bytes(b'wm-bytes-tampered')
    with pytest.raises(ValueError, match='WM checkpoint mismatch'):
        authority.verify(v_args)
    wm.write_bytes(b'wm-bytes')
    # tamper an execution-law source hash in the manifest
    stored = json.loads(out.read_text(encoding='utf-8'))
    real_law = stored['runtime_manifest']['src/wms/control/sumo_idm.py']
    stored['runtime_manifest']['src/wms/control/sumo_idm.py'] = '0' * 64
    out.write_text(json.dumps(stored), encoding='utf-8')
    with pytest.raises(ValueError, match='runtime manifest mismatch'):
        authority.verify(v_args)
    stored['runtime_manifest']['src/wms/control/sumo_idm.py'] = real_law
    # tamper a leader-rule source hash: the bound slot must refuse
    real_hash = stored['leader_rule']['files']['src/wms/physics/lane_chain.py']
    stored['leader_rule']['files']['src/wms/physics/lane_chain.py'] = '0' * 64
    out.write_text(json.dumps(stored), encoding='utf-8')
    with pytest.raises(ValueError, match='leader rule source mismatch'):
        authority.verify(v_args)
    stored['leader_rule']['files']['src/wms/physics/lane_chain.py'] = real_hash
    out.write_text(json.dumps(stored), encoding='utf-8')
    # learned mix without a mix-qualified CCT-2 refuses (ERRSRC mix on WM2 only)
    mix_cct2 = tmp_path / 'CCT2_mix.json'
    _write_cct2(mix_cct2, wm_sha, hg=300, mix=False)
    mix_args = SimpleNamespace(root=str(ROOT), wm_ckpt=str(wm), wm_config=str(wmc),
                               cct2=str(mix_cct2), out=str(tmp_path / 'mix_authority.json'),
                               cct1=None, actor=None, actor_checkpoint=None,
                               learned_mix=True, mix_module='src/wms/control/entrant.py',
                               parity_receipt=None, leader_rule_status=None,
                               leader_rule_commit=None, allow_dirty=True, dense_phase1=True)
    with pytest.raises(ValueError, match='mix-qualified'):
        authority.build(mix_args)
    # gain_only declaration against the dense phase-1 config refuses (fail-closed
    # both ways), and a tampered authority mode fails verify against the config
    dense_refuses = SimpleNamespace(**{**vars(mix_args), 'learned_mix': False,
                                       'dense_phase1': False})
    with pytest.raises(ValueError, match='dense-phase1'):
        authority.build(dense_refuses)
    stored = json.loads(out.read_text(encoding='utf-8'))
    stored['long_credit']['mode'] = 'gain_only'
    out.write_text(json.dumps(stored), encoding='utf-8')
    with pytest.raises(ValueError, match='long credit'):
        authority.verify(v_args)
