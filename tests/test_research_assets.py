"""Migration must work without opening the retired shared directory."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from traffic_shepherd.config import ROOT, load_config
from traffic_shepherd.research_assets import (
    legacy_relative, resolve_research_path, sumo_input_overrides,
)
from traffic_shepherd.scenarios import scenario_identity
from traffic_shepherd.simulation import start_sumo


def fixture_assets(root):
    asset = root / 'assets/research/highway_merge_shepherd_long/osm.net.xml.gz'
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b'unchanged network bytes')
    manifest = dict(files=[dict(path=asset.relative_to(root).as_posix(),
                               bytes=asset.stat().st_size,
                               sha256=hashlib.sha256(asset.read_bytes()).hexdigest())])
    (root / 'assets/research/migration_manifest.json').write_text(json.dumps(manifest))
    return asset


def test_old_path_maps_to_local_bytes_and_rejects_missing_or_tampered(tmp_path):
    asset = fixture_assets(tmp_path)
    old = r'D:\research\highway_merge_shepherd_long\osm.net.xml.gz'
    assert resolve_research_path(old, root=tmp_path) == asset
    with pytest.raises(FileNotFoundError):
        resolve_research_path('D:/research/highway_merge_shepherd_long/unknown.xml', root=tmp_path)
    asset.write_bytes(b'tampered network bytes')
    with pytest.raises(ValueError, match='differs'):
        resolve_research_path(old, root=tmp_path)


def test_exact_root_boundary_and_traversal():
    assert legacy_relative('D:/research-dense/file.xml') is None
    assert legacy_relative('D:/research_other/file.xml') is None
    assert legacy_relative('D:/Research/map.net.xml').as_posix() == 'map.net.xml'
    with pytest.raises(ValueError, match='traversal'):
        legacy_relative('D:/research/../private.xml')


def test_legacy_sumo_override_preserves_config_and_resolves_mixed_paths(tmp_path):
    asset = fixture_assets(tmp_path)
    cfg = tmp_path / 'saved.sumocfg'
    original = ('<configuration><input><net-file value="D:/research/'
                'highway_merge_shepherd_long/osm.net.xml.gz"/>'
                '<route-files value="requests.rou.xml"/></input></configuration>')
    cfg.write_text(original)
    assert sumo_input_overrides(cfg, root=tmp_path) == ['--net-file', str(asset)]
    assert cfg.read_text() == original
    cfg.write_text('<configuration><input><route-files value="requests.rou.xml,'
                   'D:/research/highway_merge_shepherd_long/osm.net.xml.gz"/></input></configuration>')
    assert sumo_input_overrides(cfg, root=tmp_path) == [
        '--route-files', f'{tmp_path / "requests.rou.xml"},{asset}']


def deny_retired_reads(monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        if legacy_relative(path) is not None:
            raise AssertionError(f'Unexpected retired-directory access: {path}')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)


def test_active_config_is_local_and_independent_of_cwd(tmp_path, monkeypatch):
    deny_retired_reads(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.network_path.is_relative_to(ROOT / 'assets/research')
    assert cfg.network_path.is_file()
    assert cfg.network_path.read_bytes()


def test_saved_identity_uses_local_network_without_changing_frozen_scene(tmp_path, monkeypatch):
    scene = tmp_path / 'scenario.sumocfg'
    frozen = b'<configuration><input><net-file value="D:/research/highway_merge_shepherd_long/osm.net.xml.gz"/></input></configuration>'
    scene.write_bytes(frozen)
    (tmp_path / 'requests.rou.xml').write_bytes(b'fixed requests')
    cfg = load_config()
    expected = hashlib.sha256(b'[]' + frozen + b'fixed requests'
                              + hashlib.sha256(cfg.network_path.read_bytes()).digest()).hexdigest()
    deny_retired_reads(monkeypatch)
    assert scenario_identity(scene, []) == expected
    assert scene.read_bytes() == frozen


def test_actual_launch_boundary_records_local_override_without_launching_sumo(tmp_path, monkeypatch):
    scene = tmp_path / 'scenario.sumocfg'
    scene.write_text('<configuration><input><net-file value="D:/research/'
                     'highway_merge_shepherd_long/osm.net.xml.gz"/></input></configuration>')
    frozen = scene.read_bytes()
    commands = []
    api = SimpleNamespace(isLoaded=lambda: False,
                          start=lambda command, port: commands.append(command))
    record = []
    deny_retired_reads(monkeypatch)
    start_sumo('mock-sumo-never-executed', scene, seed=1, api=api,
               lock_path=tmp_path / 'mock_launch.lock', command_record=record)
    assert len(commands) == 1 and commands == record
    command = commands[0]
    assert command[command.index('--net-file') + 1] == str(load_config().network_path)
    assert scene.read_bytes() == frozen
