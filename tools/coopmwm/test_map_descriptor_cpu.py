"""N1 map descriptor loading (CPU, no SUMO)."""
import json
import shutil
from pathlib import Path

import pytest

from tools.coopmwm import map_descriptor as md

REPO = Path(__file__).resolve().parents[2]
MAP_DIR = REPO/'assets/research/newmap_candidates/wz3to2'


@pytest.fixture(autouse=True)
def _unbound(monkeypatch):
    monkeypatch.delenv(md.ENV, raising=False)
    md._reset_for_tests()
    yield
    md._reset_for_tests()


def test_wz3to2_descriptor_projection_and_rsu():
    d = md.load(MAP_DIR)
    assert d == md.load(MAP_DIR/'MAP_SPEC.json') and d.digest == md.load(MAP_DIR).digest
    assert d.name == 'wz3to2' and d.kind == 'lane_drop' and not d.is_original and d.k_slots == 512
    assert d.lane_table == {('up', 2): (0, 0.0), ('up', 1): (1, 0.0), ('up', 0): (2, 0.0),
                            ('wz', 1): (0, 2704.0), ('wz', 0): (1, 2704.0)}
    assert d.here_offsets == (2620.56,)*3 and d.frame is md.FRAME
    b = d.bottleneck
    assert (b['drop_x_m'], b['drop_s_m'], b['closing_lane'], b['adjacent_lane'], b['through_lanes']) == (2696.0, 75.44, 2, 1, [0, 1])
    assert d.lane_and_s('up_0', 2696.)[0] == 2 and abs(d.lane_and_s('up_0', 2696.)[1]-75.44) < 1e-9
    assert d.sumo_index('up', 2) == 0 and d.sumo_index('wz', 2) is None and d.sumo_index('wz', 0) == 1
    assert d.lane_end_distance == {':n1_0_0': -8.0, ':n1_0_1': -8.0, 'up_0': 0.0, 'up_1': 0.0, 'up_2': 0.0,
                                   'wz_0': -1304.0, 'wz_1': -1304.0}
    assert d.boundary_edges() == {'mainline_entry': 'up', 'mainline_exit': 'wz', 'ramp_entry': '__no_ramp_entry__'}
    assert d.rsu.source == 'cav_reports' and d.rsu.block_zone_s_m == ((-100., 175.44), (-100., 175.44), (-100., 75.44))
    assert (d.rsu.field_mainline_s_m, d.rsu.field_ramp_s_m, d.rsu.field_cell_m) == ((-800., 225.44), (-378.66, 75.44), 50.)
    assert (d.rsu.report_range_m, d.rsu.report_lane_reach, d.rsu.receipt_age_s) == (100., 1, 0.)
    rec = json.loads(json.dumps(d.record()))
    assert rec['name'] == 'wz3to2' and rec['network_sha256'] == md.REGISTERED['wz3to2']['network_sha256']


def test_registered_digests_fail_closed(tmp_path):
    copy = tmp_path/'wz3to2'
    shutil.copytree(MAP_DIR, copy)
    spec = json.loads((copy/'MAP_SPEC.json').read_text())
    spec['bottleneck']['x_m'] = 2600.0
    (copy/'MAP_SPEC.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='MAP_SPEC differs'):
        md.load(copy)
    shutil.copy(MAP_DIR/'MAP_SPEC.json', copy/'MAP_SPEC.json')
    net = (copy/'wz3to2.net.xml').read_text().replace('length="1296.00"', 'length="1290.00"', 1)
    (copy/'wz3to2.net.xml').write_text(net)
    with pytest.raises(ValueError, match='network differs'):
        md.load(copy)
    other = json.loads((MAP_DIR/'MAP_SPEC.json').read_text())
    other['name'] = 'wz2to1'
    (copy/'MAP_SPEC.json').write_text(json.dumps(other))
    with pytest.raises(ValueError, match='not registered'):
        md.load(copy)


def test_line_endings_do_not_change_the_descriptor(tmp_path):
    copy = tmp_path/'wz3to2'
    copy.mkdir()
    for p in MAP_DIR.iterdir():
        (copy/p.name).write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))
    assert md.load(copy).digest != '' and md.load(copy).network_sha256 == md.load(MAP_DIR).network_sha256
    assert md.load(copy).spec_sha256 == md.load(MAP_DIR).spec_sha256


def test_process_selection_and_one_map_per_process(monkeypatch):
    assert md.current() is md.ORIGINAL and md.ORIGINAL.is_original
    md._reset_for_tests()
    monkeypatch.setenv(md.ENV, str(MAP_DIR/'MAP_SPEC.json'))
    d = md.current()
    assert d.name == 'wz3to2' and md.bind(md.load(MAP_DIR)) is not None
    with pytest.raises(ValueError, match='one map per process'):
        md.bind(md.ORIGINAL)
