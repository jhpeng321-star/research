import os
from pathlib import Path
import pytest
from .nullfix_geometry import install


def artifacts():
    root = Path(__file__).resolve().parents[2]
    return (Path(os.environ.get('COOPMWM_NULLFIX_GEOMETRY', root/'traffic_shepherd/c4/geometry.py')),
            Path(os.environ.get('COOPMWM_NULLFIX_NETWORK', root/'assets/research/highway_merge_shepherd_long/osm.net.xml.gz')))


def test_bound_network_is_complete_without_changing_data_roots(monkeypatch):
    from traffic_shepherd.c4 import geometry
    source, net = artifacts()
    before = dict(os.environ)
    with monkeypatch.context() as m:
        for key in ('lane_allowed_speeds_mps', 'NET_RELPATH', '_LANE_SPEEDS_CACHE'):
            m.setattr(geometry, key, getattr(geometry, key))
        install(source, net)
        table = geometry.lane_allowed_speeds_mps()
        assert set(table) == {v[0] for v in geometry.LANE_TABLE.values()}
        assert all(v > 0 for v in table.values())
        m.setenv('RESEARCH_DENSE_ROOT', 'deliberately-unavailable')
        assert geometry.lane_allowed_speeds_mps() == table
    assert dict(os.environ) == before


def test_missing_bound_artifact_fails_before_activation(tmp_path):
    source, _ = artifacts()
    with pytest.raises(AssertionError):
        install(source, tmp_path/'missing.net.xml.gz')
