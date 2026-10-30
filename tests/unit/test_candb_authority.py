"""Source/input mutation refuses a staged launch; no GPU, SUMO or training."""
import json

import pytest

from tools.candB import authority


def test_build_verify_and_input_source_mutation(tmp_path, monkeypatch):
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    monkeypatch.setattr(authority, 'clean_tree', lambda root: True)
    monkeypatch.setattr(authority, 'source_commit', lambda root: 'commit')
    monkeypatch.setattr(authority, 'inspect', lambda inputs: {})
    source = {'module': 'original'}
    monkeypatch.setattr(authority, 'manifest', lambda: dict(source))
    data = tmp_path / 'wm.pt'
    data.write_bytes(b'first')
    path = tmp_path / 'authority.json'
    authority.build({'wm': str(data)}, path)
    assert authority.verify(path)['fresh_zero_start'] is True
    source['module'] = 'changed'
    with pytest.raises(ValueError, match='manifest'):
        authority.verify(path)
    source['module'] = 'original'
    data.write_bytes(b'changed')
    with pytest.raises(ValueError, match='input hash'):
        authority.verify(path)
    assert json.loads(path.read_text())['training_launched'] is False


def test_rank_gate_required_before_cli_loads_any_models(tmp_path, monkeypatch, capsys):
    from wms.control.policy14_cli import main
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    result = main(['--config', 'configs/c4_policy14_candB.toml', '--wm-ckpt', 'absent',
        '--wm-config', 'absent', '--real-returns', 'absent', '--output-dir', str(tmp_path/'out')])
    assert result != 0
    assert 'requires --rankcheck' in capsys.readouterr().out
    assert not (tmp_path/'out').exists()
