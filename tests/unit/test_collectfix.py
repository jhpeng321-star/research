"""Missing binary assets and charged single-box -> two-box retry accounting regressions."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from traffic_shepherd.c4.collector_readiness import check_controller_assets, check_runtime_dependencies


def test_missing_and_corrupt_model_refuse_before_onnx(tmp_path):
    config = SimpleNamespace(model_path=tmp_path / 'model.onnx', model_sha256='0' * 64)
    def forbidden(_):
        pytest.fail('must validate bytes before ONNX session')
    with pytest.raises(FileNotFoundError):
        check_controller_assets(config, forbidden)
    config.model_path.write_bytes(b'bad')
    with pytest.raises(ValueError, match='checksum'):
        check_controller_assets(config, forbidden)


def test_readiness_constructs_session_and_propagates_missing_dependency(tmp_path):
    path = tmp_path / 'model.onnx'
    path.write_bytes(b'frozen')
    config = SimpleNamespace(model_path=path, model_sha256=hashlib.sha256(b'frozen').hexdigest())
    seen = []
    assert check_controller_assets(config, seen.append)['onnx_session'] == 'ready'
    assert seen == [config]
    def missing(_):
        raise ImportError('onnxruntime')
    with pytest.raises(ImportError):
        check_controller_assets(config, missing)


def test_missing_sumo_binary_refuses_without_simulator_start(monkeypatch):
    import traffic_shepherd.c4.collector_readiness as readiness
    monkeypatch.setattr(readiness.shutil, 'which', lambda _: None)
    monkeypatch.setattr(readiness.subprocess, 'run', lambda *a, **k: pytest.fail('no simulator invocation'))
    with pytest.raises(RuntimeError, match='SUMO binary'):
        check_runtime_dependencies()


def test_missing_eager_reset_import_refuses_before_environment_creation(monkeypatch):
    import traffic_shepherd.c4.collector_readiness as readiness
    monkeypatch.setattr(readiness.shutil, 'which', lambda _: '/sumo')
    monkeypatch.setattr(readiness.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='SUMO version'))
    def missing(name):
        raise ModuleNotFoundError('output.controller_verification_v7')
    monkeypatch.setattr(readiness.importlib, 'import_module', missing)
    with pytest.raises(ModuleNotFoundError, match='controller_verification_v7'):
        check_runtime_dependencies()


def test_two_box_adoption_retains_legacy_charge_and_never_reuses_old_seeds(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('collectfix_fast_tests', Path(__file__).with_name('test_fastloop.py'))
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    loop, fast = tests.loop, tests.fast
    cfg = tests.fixture(tmp_path)
    root = Path(cfg['loop_root'])
    state = loop.load_state(cfg)
    old = [968000, 968001, 968004]
    loop.append(root / 'seed_ledger.jsonl', {'cycle': 0, 'attempt': 1, 'seeds': old,
                                         'sim_s_charged': 3 * 721, 'host': 'cpu'})
    # Simulate a crash after append: the ledger must restore the old charge and seed exclusion.
    monkeypatch.setattr(loop, 'fast_module', lambda: fast)
    def world(loop_, cfg_, snapshot, ph, root_, item, box, attempt):
        assert item['seed'] not in old
        out = root / f"artifact_{item['seed']}"
        tests.artifact(out)
        return {'artifact': str(out), 'case': item, 'collect': {}}
    monkeypatch.setattr(fast, '_world', world)
    receipt = fast.collect(loop, cfg, state, {'attempt': 2}, root)
    assert receipt['sim_s_charged'] == 6 * 721
    assert state['sim_s_charged'] == 6 * 721
    assert not set(receipt['seeds']) & set(old)
