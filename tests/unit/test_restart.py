"""Integration contracts at the native/resume/longcritic merge boundaries; no SUMO."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/c4')]
import loop14 as loop  # noqa: E402
from wms.control.real_share import RealShareConfig, calibrate_share  # noqa: E402


def test_native_veto_precedes_adoption(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('restart_fixture', ROOT/'tests/unit/test_loop14.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    cfg = loop.read_json(fixture._config(tmp_path, fixture._registration(tmp_path)))
    cfg['policy'].update(native_init=True, device='cpu')
    cfg['hosts'].update(cpuP=cfg['hosts']['cpu'], cpuQ=cfg['hosts']['cpu'])
    cfg['fastloop'] = {'two_box': True, 'boxes': ['cpuP', 'cpuQ'], 'workers_per_box': 1}
    loop.init(cfg, SimpleNamespace(adopt_policy=None))
    state = loop.load_state(cfg)
    out = Path(cfg['loop_root'])/'policy/c00'
    def train(*args, **kwargs):
        out.mkdir(parents=True)
        (out/'policy_latest.pt').write_bytes(b'rejected policy without native receipt')
        loop.write_atomic(out/'RUN_SUMMARY.json', {'status': 'completed', 'updates': []})
        return 0
    monkeypatch.setattr(loop, 'run_cmd', train)
    with pytest.raises(FileNotFoundError):
        loop.run_policy(cfg, state, {'attempt': 1}, Path(cfg['loop_root']))
    assert state['policy'] is None
    assert state['sim_s_charged'] == 0


def test_pinned_owner_mix_is_not_actor_adoption(tmp_path, monkeypatch):
    from wms.control.dreamer import DreamerConfig
    source = tmp_path/'returns.npz'
    source.write_bytes(b'pinned authoritative real returns')
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    evidence = {'selected_share': 1.0, 'source_sha256': sha, 'policy_sha256': 'rejected',
                'fit_worlds': [1], 'development_worlds': [2],
                'candidates': {'1.0': {'eligible': True, 'delta_upper95': -1.0}}}
    path = tmp_path/'evidence.json'
    path.write_text(json.dumps(evidence))
    import numpy as np
    import wms.control.real_share as share
    monkeypatch.setattr(share, 'tensor_digest', lambda module: 'fresh critic')
    import wms.control.start_states as starts
    monkeypatch.setattr(starts, 'checked_path', lambda p: Path(p))
    trainer = SimpleNamespace(config=DreamerConfig(), critic=object(), world_model_ids=['round16'])
    provider = SimpleNamespace(path=source, provenance={'source_kind': 'current_policy', 'policy_checkpoint': 'rejected'})
    fit = SimpleNamespace(provider=provider, provenance={'seeds': np.array([1])})
    dev = SimpleNamespace(provenance={'seeds': np.array([2])})
    receipt = calibrate_share(trainer, fit, dev, policy_sha256=None,
                              config=RealShareConfig(enabled=True, bootstrap_receipt=str(path)))
    assert receipt['selected_share'] == trainer.config.real_return_mix == 1.0
    assert receipt['offpolicy_bootstrap'] and not receipt['actor_adopted'] and not receipt['critic_adopted']
    source.write_bytes(b'different cohort')
    with pytest.raises(ValueError, match='stale/mismatched'):
        calibrate_share(trainer, fit, dev, policy_sha256=None,
                        config=RealShareConfig(enabled=True, bootstrap_receipt=str(path)))


def test_residual_persistent_exploration_keeps_exact_zero_init():
    import torch
    from wms.control.policy import CavPolicy, PolicyNoise
    from wms.control.obs import build_observations
    from wms.control.contract import GROUP_SIGNAL_DIM
    spec = importlib.util.spec_from_file_location('native_fixture', ROOT/'tests/unit/test_nativeinit.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    # Use the same real observation builder and physical synthetic state as the
    # native identity integration suite, with two distinct supplied noise draws.
    state = fixture.scene()
    obs = build_observations(state)
    p = CavPolicy(lane_head=False, residual_on_native=True)
    p.residual_persistent_exploration = True
    signal = torch.zeros(*obs.cav_mask.shape, GROUP_SIGNAL_DIM)
    a = PolicyNoise(torch.ones_like(signal[..., 0]), None)
    b = PolicyNoise(-torch.ones_like(signal[..., 0]), None)
    assert torch.equal(p(obs, signal, noise=a)[0], p(obs, signal, noise=b)[0])
    assert not p(obs, signal, noise=a)[0].count_nonzero()
    with torch.no_grad():
        p.residual_head.bias.fill_(.1)
    assert not torch.equal(p(obs, signal, noise=a)[0], p(obs, signal, noise=b)[0])
    assert torch.equal(p(obs, signal, noise=a, deterministic=True)[0],
                       p(obs, signal, noise=b, deterministic=True)[0])
