import numpy as np
import pytest

from tools.rblane.snapshot import postmotion_snapshot, proposal_position
from wms.physics.config import PhysicsConfig


def row(s, lane):
    k = np.zeros((1, 7), np.float32)
    k[:, 0], k[:, 1 + lane], k[:, 4], k[:, 6] = s, 1, 10, 5
    return k


@pytest.mark.parametrize('source,destination,recorded_s', [(0, 1, -98.55), (1, 0, -101.45)])
def test_recorded_lane_origin_jump_is_undone_before_decision(source, destination, recorded_s):
    p = PhysicsConfig()
    moved = postmotion_snapshot(row(-100, source), row(recorded_s, destination), p)
    assert moved[0, 0] == pytest.approx(-100, abs=1e-5)
    predicted = proposal_position(moved[0, 0], np.int64(source), np.int64(destination), p)
    assert predicted == pytest.approx(recorded_s, abs=1e-5)
    assert moved[0, 1 + source] == 1


@pytest.mark.parametrize('position', [20., 100.])
def test_merge_and_downstream_lane_changes_have_no_frame_jump(position):
    moved = postmotion_snapshot(row(position - 5, 0), row(position, 1), PhysicsConfig())
    assert moved[0, 0] == position


def test_hypothetical_proposal_position_uses_its_own_destination_origin():
    p = PhysicsConfig()
    moved = postmotion_snapshot(row(-100, 1), row(-95, 1), p)
    assert moved[0, 0] == -95
    assert proposal_position(moved[0, 0], np.int64(1), np.int64(0), p) == pytest.approx(-96.45)
