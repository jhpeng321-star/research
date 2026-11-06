"""N9 probe lane-family converter: exact conversion + TraCI command-sequence parity (CPU only)."""
import pytest

from . import lc_convert_probe as cp, moves, nm_map
from .test_nm_families_cpu import probe_rows


@pytest.fixture
def wz():
    previous = nm_map.bind(nm_map.stub)
    yield nm_map.lane_drop('wz3to2')
    nm_map.bind(previous)


def lane_rows():
    return [(name, row) for name, row in probe_rows() if cp.family_of(row) is not None]


def test_all_probe_lane_rows_convert_with_exact_command_parity(wz):
    rows = lane_rows()
    assert len(rows) == 36
    counts = {}
    for name, row in rows:
        program = cp.convert(row, wz)                          # zero refusals
        assert cp.to_traci(program, wz) == cp.member_log(row)  # exact TraCI command-sequence parity
        assert set(program['dither_excluded']) <= {d['id'] for d in row.get('dither') or []}
        for vid, params in program['member_params'].items():
            if 'release_s' in params and params['release_s'] == wz.drop_s_m:
                assert program['lanes'][vid] in (0, 1, 2)
        with pytest.raises(ValueError):
            moves.from_plan(row)                               # never a legacy projection
        counts[program['family']] = counts.get(program['family'], 0)+1
    assert counts == dict(lane_balance=21, early_merge=6, zipper=3, zipper_order=3, discharge=3)


def test_probe_frame_mapping_through_the_n1_descriptor(wz):
    lb = next(row for _, row in lane_rows() if cp.family_of(row) == 'lane_balance')
    program = cp.convert(lb, wz)
    vid = program['members'][0]
    # SUMO lane 1 -> WM 1 (adjacent); target SUMO 2 -> WM 0 (far); giveup X_B-300 -> s 75.44-300.
    assert program['lanes'][vid] == 1 and program['member_params'][vid]['target_lane'] == 0
    assert program['member_params'][vid]['giveup_s'] == pytest.approx(75.44-300.)
    assert all(c[1] != 'changeLane' or c[2] == 0 for c in program['commands'][vid])


def test_non_lane_and_tampered_rows_are_refused(wz):
    others = [row for _, row in probe_rows() if cp.family_of(row) is None]
    for row in others:
        with pytest.raises(ValueError, match='not a probe lane-family row'):
            cp.convert(row, wz)
    row = dict(lane_rows()[0][1])
    row['member_lanes'] = [9]*len(row['members'])
    with pytest.raises(ValueError):
        cp.convert(row, wz)
