"""N9 probe lane-family converter -> LCEXEC free_program: exact binding of all 36 lane rows (CPU)."""
import copy
from pathlib import Path

import pytest

from . import lc_convert_probe as cp, moves
from .lc_program import FreeProgram, LC_MODE_TRACI_ONLY, check_registered
from .test_nm_families_cpu import probe_rows

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def frame():
    from . import map_descriptor as md
    from .lc_program import frame_from_descriptor
    return frame_from_descriptor(md.load(REPO/'assets/research/newmap_candidates/wz3to2'))


def lane_rows():
    return [(name, row) for name, row in probe_rows() if cp.family_of(row) is not None]


def keyed():
    return {(cp.family_of(r), r['seed'], name): r for name, r in lane_rows()}


def test_all_36_lane_rows_bind_exactly_with_zero_refusals(frame):
    rows = lane_rows()
    counts = {}
    for name, row in rows:
        program = cp.probe_free_program(row, frame, file=name)
        assert isinstance(program, FreeProgram) and check_registered(program) == []
        assert list(program.members) == row['members'] and program.join_el == (0,)*len(row['members'])
        assert program.duration_s == row['duration_s'] and program.source['verification_id'] == row['verification_id']
        assert [p.role for p in program.programs] == row['member_roles']
        assert FreeProgram.from_dict(program.to_dict()) == program            # record round trip
        assert cp.probe_log(cp.wm_log(row, frame), frame) == cp.member_log(row)   # exact frame inverse
        with pytest.raises(ValueError):
            moves.from_plan(row)                                               # never a legacy box
        counts[program.source['probe_family']] = counts.get(program.source['probe_family'], 0)+1
    assert len(rows) == 36
    assert counts == dict(lane_balance=21, early_merge=6, zipper=3, zipper_order=3, discharge=3)


def test_role_mapping_in_the_wm_frame(frame):
    rows = keyed()
    lb = cp.probe_free_program(rows[('lane_balance', 1, 'COOPKINDS_MOVES_B')], frame)
    p = lb.programs[0]
    # SUMO target 2 -> WM 0; giveup X_B-300 -> s 75.44-300; persistent request rule; 300 s hold.
    assert (p.lane.target, p.lane.rule, p.hold_s) == (0, 'right_of_target', 300)
    assert p.lane.giveup_s == pytest.approx(75.44-300.) and p.release_s == pytest.approx(75.44)
    em = cp.probe_free_program(rows[('early_merge', 1, 'COOPKINDS_MOVES_B')], frame)
    merge, room = em.programs[0], em.programs[-1]
    assert (merge.lane.target, merge.lane.rule, merge.lane.source_lane) == (1, 'from_lane', 2)
    assert merge.lane.from_s == pytest.approx(75.44-1200.)
    assert room.tau == 2.0 and room.zone == pytest.approx((75.44-1500., 75.44-700.))
    zp = cp.probe_free_program(rows[('zipper', 1, 'COOPKINDS_MOVES_B')], frame)
    m, y = zp.programs
    assert m.lc_mode == LC_MODE_TRACI_ONLY and m.lane.from_s == pytest.approx(75.44-100.)
    assert m.fallback_s == pytest.approx(75.44-15.)
    assert y.lon.kind == 'track' and y.lon.off == -25. and y.partner == zp.members[0]
    ds = cp.probe_free_program(rows[('discharge', 1, 'COOPKINDS_MOVES_B')], frame)
    assert {q.tau for q in ds.programs} == {.6} and ds.programs[0].release_s == pytest.approx(75.44+800.)


def test_non_lane_and_tampered_rows_are_refused(frame):
    for _, row in probe_rows():
        if cp.family_of(row) is None:
            with pytest.raises(ValueError, match='not a probe lane-family row'):
                cp.probe_free_program(row, frame)
    rows = keyed()
    lb = copy.deepcopy(rows[('lane_balance', 1, 'COOPKINDS_MOVES_B')])
    lb['commands'][lb['members'][0]][0][2] = 1                 # changeLane to a lane it never targets
    with pytest.raises(ValueError, match='changeLane'):
        cp.probe_free_program(lb, frame)
    ds = copy.deepcopy(rows[('discharge', 1, 'COOPKINDS_MOVES_B')])
    ds['commands'][ds['members'][0]][0][2] = 0.9               # tau differs from the member spec
    with pytest.raises(ValueError, match='tau'):
        cp.probe_free_program(ds, frame)
    pg = copy.deepcopy(rows[('lane_balance', 1, 'PLANGEN2_MOVES_B')])
    pg['commands'].append([0, 'main.999999', 'changeLane', 2])  # a non-member, non-dither command
    with pytest.raises(ValueError, match='non-member'):
        cp.probe_free_program(pg, frame)
    zp = copy.deepcopy(rows[('zipper', 1, 'COOPKINDS_MOVES_B')])
    zp['commands'][zp['members'][1]][5][2] = 3.0               # issued accel outside [-dmax, amax]
    with pytest.raises(ValueError, match='accel'):
        cp.probe_free_program(zp, frame)
